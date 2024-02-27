import toml
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from ehrembeddings.config import Config
from ehrembeddings.model import EHR2Vec
from ehrembeddings.dataset import EmbeddingsDataModule
from ehrembeddings.preprocess import get_data


def main():
    config = Config(**toml.load("config.toml"))
    seed_everything(config.random_seed)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train, val, test, inverse_frequency = get_data(config=config, tokenizer=tokenizer)
    model = EHR2Vec(
        vocab_size=tokenizer.vocab_size,
        **config.model.dict(),
        **config.optimizer.dict(),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.filepaths.checkpoints,
        filename="{epoch}_{step}_{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
    )
    logger = TensorBoardLogger(
        config.filepaths.logs,
        name="embeddings",
    )
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)
    callbacks = [
        checkpoint_callback,
        lr_logger,
        RichProgressBar(),
        EarlyStopping(monitor="val_loss", mode="min", patience=2),
    ]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip,
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=config.logging.log_every_n_steps,
        fast_dev_run=config.fast_dev_run,
        check_val_every_n_epoch=1,
    )
    data_module = EmbeddingsDataModule(
        train=train,
        val=val,
        test=test,
        tokenizer=tokenizer,
        inverse_frequency=inverse_frequency,
        sigma=config.training.sigma,
        n_negatives=config.training.n_negatives,
        batch_size=config.training.batch_size,
    )
    trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
