import toml
from box import Box
from polars import read_csv
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import tensor

from ehrembeddings.model import ICD2Vec
from ehrembeddings.dataset import EmbeddingsDataModule, make_data, make_vocabulary


def main():
    config = Box(toml.load("config.toml"))
    seed_everything(config.random_seed)
    df = read_csv(config.paths.data)
    vocabulary = make_vocabulary(corpus=df["icd_code"])
    df = df.join(vocabulary, on="icd_code")
    data = make_data(df=df)
    contexts = tensor(data["context"].to_numpy()).long()
    targets = tensor(data["target"].to_numpy()).long()
    vocab_size = vocabulary.shape[0]
    model = ICD2Vec(
        vocab_size=vocab_size,
        **config.model,
        **config.loss,
        **config.optimizer,
        **config.lr_scheduler,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoints,
        filename="{epoch}_{step}_{train_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=config.logging.checkpoint_every_n_steps,
    )
    logger = TensorBoardLogger(
        config.paths.logs,
        name="embeddings",
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip,
        max_epochs=config.training.max_epochs,
        auto_lr_find=True,
        accelerator="cpu",  # FIXME should be "auto" eventually
        devices="auto",
        log_every_n_steps=config.logging.log_every_n_steps,
        track_grad_norm=2,
        fast_dev_run=config.fast_dev_run,
    )
    data_module = EmbeddingsDataModule(
        contexts=contexts, targets=targets, batch_size=config.training.batch_size
    )
    trainer.tune(model, datamodule=data_module)
    trainer.fit(
        model,
        datamodule=data_module,
        # ckpt_path="",
    )


if __name__ == "__main__":
    main()
