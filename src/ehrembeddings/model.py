from pytorch_lightning import LightningModule
from torch import LongTensor, Tensor, exp
from torch.distributions import Normal
from torch.nn import Linear, Embedding, Sequential, Dropout, ReLU
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss


class EHR2Vec(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        metric_embedding_dim: int,
        sigma: float,
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
        self.linear1 = Linear(
            in_features=embedding_dim,
            out_features=hidden_dim,
        )
        self.relu = ReLU()
        self.dropout = Dropout(p=dropout_rate)
        self.linear2 = Linear(
            in_features=hidden_dim,
            out_features=metric_embedding_dim,
        )
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = Sequential(self.linear1, self.relu, self.dropout, self.linear2)
        self.criterion = SelfSupervisedLoss(NTXentLoss(temperature=temperature))
        self.initial_restart_iter = initial_restart_iter
        self.warm_restart_factor = warm_restart_factor
        self.step_frequency = step_frequency

    def forward(
        self,
        events: LongTensor,  # (batch_size, max_batch_sequence_length)
        event_times: Tensor,  # (batch_size, max_batch_sequence_length, 1)
        targets: LongTensor,  # (batch_size,)
        target_times: Tensor,  # (batch_size, 1, 1)
    ) -> tuple[Tensor, Tensor]:
        normal_distribution = Normal(loc=target_times, scale=self.sigma)
        relative_likelihood = exp(normal_distribution.log_prob(event_times))
        probability = relative_likelihood / relative_likelihood.sum(dim=1, keepdim=True)
        event_embeddings = self.embeddings(events)
        context_embeddings = sum(event_embeddings * probability, dim=1)
        target_embeddings = self.embeddings(targets)
        return self.model(context_embeddings), self.model(target_embeddings)

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
