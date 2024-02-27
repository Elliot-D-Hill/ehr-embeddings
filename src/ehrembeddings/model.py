from pytorch_lightning import LightningModule
from torch import LongTensor, Tensor, bmm
from torch.nn import Linear, Embedding, Sequential, BCEWithLogitsLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss


class EHR2Vec(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.target_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.context_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = BCEWithLogitsLoss()

    def forward(self, target: LongTensor, context: LongTensor) -> Tensor:
        target_embeddings = self.target_embeddings(target)
        context_embeddings = self.context_embeddings(context)
        dot_product = bmm(
            target_embeddings, context_embeddings.transpose(1, 2)
        ).squeeze(1)
        return dot_product

    def step(self, batch, step_type: str):
        context, target, labels = batch
        outputs = self(target, context)
        loss = self.criterion(outputs, labels)
        predicted_positive_index = outputs.argmax(dim=1, keepdim=True)
        auccuracy = (predicted_positive_index == 0).mean()
        self.log(f"{step_type}_accuracy", auccuracy, prog_bar=True, logger=True)
        self.log(f"{step_type}_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="val")

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            nesterov=True,
            momentum=0.99,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class EmbeddingModel(LightningModule):
    def __init__(
        self,
        embeddings: Tensor,
        output_dim: int,
        lr: float,
        weight_decay: float,
        freeze: bool,
    ) -> None:
        super().__init__()
        self.embeddings = Embedding.from_pretrained(
            embeddings, freeze=freeze, padding_idx=0
        )
        self.linear = Linear(embeddings.shape[1], output_dim)
        self.model = Sequential(self.embeddings, self.linear)
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = BCEWithLogitsLoss()

    def forward(self, events: LongTensor):
        return self.model(events)

    def step(self, batch, step_type: str):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        predicted_labels = outputs.argmax(dim=1)
        auccuracy = (predicted_labels == labels).mean()
        self.log(f"{step_type}_accuracy", auccuracy, prog_bar=True, logger=True)
        self.log(f"{step_type}_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_idx: int):
        return self.step(batch, step_type="val")

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            nesterov=True,
            momentum=0.99,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
