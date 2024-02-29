from pydantic import BaseModel
from pathlib import Path


class Filepaths(BaseModel):
    data: Path
    train: Path
    val: Path
    test: Path
    pretrain_checkpoints: Path
    finetune_checkpoints: Path
    logs: Path


class PretrainModel(BaseModel):
    embedding_dim: int


class FinetuneModel(BaseModel):
    hidden_dim: int
    output_dim: int
    freeze: bool
    random_embeddings: bool


class Model(BaseModel):
    pretrain: PretrainModel
    finetune: FinetuneModel


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float


class OptimizerShared(BaseModel):
    momentum: float
    nesterov: bool


class Optimizer(BaseModel):
    pretrain: OptimizerConfig
    finetune: OptimizerConfig
    shared: OptimizerShared


class Scheduler(BaseModel):
    patience: int
    factor: float


class Training(BaseModel):
    batch_size: int
    max_epochs: int
    gradient_clip: float
    sigma: float
    n_negatives: int
    early_stopping_patience: int


class Logging(BaseModel):
    check_val_every_n_epoch: int
    val_check_interval: int
    log_every_n_steps: int


class Config(BaseModel):
    random_seed: int
    fast_dev_run: bool
    regenerate: bool
    pretrain: bool
    finetune: bool
    evaluate: bool
    profile: bool
    train_size: float
    monitor: str
    mode: str
    # tune: bool
    # refit: bool
    # plot: bool
    # n_trials: int
    # verbose: bool
    # n_trials: int
    # method: str
    filepaths: Filepaths
    training: Training
    logging: Logging
    optimizer: Optimizer
    scheduler: Scheduler
    model: Model
