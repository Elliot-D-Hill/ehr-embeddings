from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor
from torch.nn import Linear, Embedding, Sequential, Dropout
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss


class EHR2Vec(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout_rate: float,
        metric_embedding_dim: int,
        learning_rate: float,
        weight_decay: float,
        temperature: float,
        initial_restart_iter: int,
        warm_restart_factor: int,
        step_frequency: int,
    ) -> None:
        super().__init__()
        self.embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.linear = Linear(
            in_features=embedding_dim, out_features=metric_embedding_dim
        )
        self.dropout = Dropout(p=dropout_rate)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = Sequential(self.embeddings, self.dropout, self.linear)
        self.criterion = SelfSupervisedLoss(NTXentLoss(temperature=temperature))
        self.initial_restart_iter = initial_restart_iter
        self.warm_restart_factor = warm_restart_factor
        self.step_frequency = step_frequency

    def forward(
        self,
        inputs: LongTensor,
        outputs: LongTensor,
    ) -> tuple[FloatTensor, FloatTensor]:
        input_embeddings = self.model(inputs)
        output_embeddings = self.model(outputs)
        return input_embeddings, output_embeddings

    def training_step(self, batch: tuple[LongTensor, LongTensor]):
        inputs, outputs = batch
        input_embeddings, output_embeddings = self(inputs, outputs)
        loss = self.criterion(input_embeddings, output_embeddings)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        learning_rate_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.initial_restart_iter,
            T_mult=self.warm_restart_factor,
        )
        scheduler_config = {
            "scheduler": learning_rate_scheduler,
            # "monitor": "train_loss",
            "interval": "step",
            "frequency": self.step_frequency,
            "name": "scheduler",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
