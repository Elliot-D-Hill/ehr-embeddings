from pytorch_lightning import LightningModule
from torch import LongTensor, Tensor, bmm
from torch.nn import Module, Linear, Embedding, Sequential, BCEWithLogitsLoss, ReLU
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torchmetrics.functional import auroc

from ehrembeddings.config import Config
from ehrembeddings.utils import get_best_checkpoint


class SkipGram(Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.target_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.context_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

    def forward(self, target: LongTensor, context: LongTensor) -> Tensor:
        target_embeddings = self.target_embeddings(target)
        context_embeddings = self.context_embeddings(context)
        dot_product = bmm(
            target_embeddings, context_embeddings.transpose(1, 2)
        ).squeeze(1)
        return dot_product


class PretrainModel(LightningModule):
    def __init__(
        self,
        model: SkipGram,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: Module,
        monitor: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.monitor = monitor

    def forward(self, target: LongTensor, context: LongTensor) -> Tensor:
        return self.model(target, context)

    def step(self, batch, step_type: str):
        context, target, labels = batch
        outputs = self(target, context)
        loss = self.criterion(outputs, labels)
        if step_type != "train":
            predicted_positive_index = outputs.argmax(dim=1, keepdim=True)
            auccuracy = (predicted_positive_index == 0).float().mean()
            self.log(f"{step_type}_accuracy", auccuracy, prog_bar=True)
        self.log(f"{step_type}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="val")

    def test_step(self, batch, batch_idx: int):
        self.step(batch, step_type="test")

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": self.monitor,
        }


# TODO make_optimizer and make_scheduler should select the method and initialize it
def make_pretrain_model(
    config: Config, vocab_size: int, pretrain: bool
) -> PretrainModel:
    model = SkipGram(
        vocab_size=vocab_size, embedding_dim=config.model.pretrain.embedding_dim
    )
    optimizer = SGD(
        model.parameters(),
        **config.optimizer.pretrain.dict(),
        **config.optimizer.shared.dict(),
    )
    scheduler = ReduceLROnPlateau(optimizer, **config.scheduler.dict())
    criterion = BCEWithLogitsLoss()
    if pretrain:
        return PretrainModel(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            monitor=config.monitor,
        )
    best_checkpoint = get_best_checkpoint(
        ckpt_folder=config.filepaths.pretrain_checkpoints, mode=config.mode
    )
    return PretrainModel.load_from_checkpoint(
        checkpoint_path=best_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        monitor=config.monitor,
    )


class EmbeddingModel(Module):
    def __init__(
        self,
        embeddings: Tensor,
        hidden_dim: int,
        output_dim: int,
        freeze: bool,
        random_embeddings: bool,
    ) -> None:
        super().__init__()
        if random_embeddings:
            self.embeddings = Embedding(
                num_embeddings=embeddings.shape[0],
                embedding_dim=embeddings.shape[1],
                freeze=freeze,
            )
        else:
            self.embeddings = Embedding.from_pretrained(
                embeddings, freeze=freeze, padding_idx=0
            )
        self.parallel_mlp = Sequential(
            Linear(embeddings.shape[1], hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
        )

    def forward(self, events: LongTensor):
        out: Tensor = self.parallel_mlp(self.embeddings(events))
        return self.mlp(out.mean(dim=1))


class FinetuneModel(LightningModule):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: Module,
        monitor: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.monitor = monitor

    def forward(self, events: LongTensor):
        return self.model(events)

    def step(self, batch, step_type: str):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.log(
            f"{step_type}_loss",
            loss.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        if step_type != "train":
            auc = auroc(outputs, labels.long(), task="binary")
            self.log(
                f"{step_type}_auc",
                auc.item(),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
        return loss

    def training_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="val")

    def test_step(self, batch, batch_idx: int):
        self.step(batch, step_type="test")

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": self.monitor,
        }


# TODO make_optimizer and make_scheduler should select the method and initialize it
def make_finetune_model(embeddings, config: Config, finetune: bool):
    model = EmbeddingModel(embeddings=embeddings, **config.model.finetune.dict())
    optimizer = SGD(
        params=model.parameters(),
        **config.optimizer.finetune.dict(),
        **config.optimizer.shared.dict(),
    )
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode=config.mode, **config.scheduler.dict()
    )
    criterion = BCEWithLogitsLoss()
    if finetune:
        return FinetuneModel(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            monitor=config.monitor,
        )
    best_checkpoint = get_best_checkpoint(
        ckpt_folder=config.filepaths.finetune_checkpoints, mode=config.mode
    )
    return FinetuneModel.load_from_checkpoint(
        checkpoint_path=best_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        monitor=config.monitor,
    )
