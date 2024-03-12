from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
)

# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from ehrembeddings.config import Config


def make_trainer(config: Config, ckpt_folder):
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_folder,
        filename="{epoch}_{step}_{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor=config.monitor,
        mode=config.mode,
        every_n_epochs=config.logging.check_val_every_n_epoch,
    )
    # logger = TensorBoardLogger(
    #     config.filepaths.logs,
    #     name="embeddings",
    # )
    # lr_logger = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint_callback,
        # lr_logger,
        RichProgressBar(),
        EarlyStopping(
            monitor=config.monitor,
            mode=config.mode,
            patience=config.training.early_stopping_patience,
        ),
    ]
    return Trainer(
        logger=False,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip,
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=config.logging.log_every_n_steps,
        fast_dev_run=config.fast_dev_run,
        check_val_every_n_epoch=config.logging.check_val_every_n_epoch,
        profiler="simple" if config.profile else None,
        val_check_interval=config.logging.val_check_interval,
    )
