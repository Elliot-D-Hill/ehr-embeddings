from pydantic import BaseModel
from pathlib import Path


class Filepaths(BaseModel):
    data: Path
    checkpoints: Path
    logs: Path
    train: Path
    val: Path
    test: Path
    inverse_frequency: Path


class Model(BaseModel):
    embedding_dim: int


class Optimizer(BaseModel):
    lr: float
    weight_decay: float


class Training(BaseModel):
    batch_size: int
    max_epochs: int
    gradient_clip: float
    sigma: float
    n_negatives: int


class Logging(BaseModel):
    checkpoint_every_n_steps: int
    log_every_n_steps: int


class Config(BaseModel):
    fast_dev_run: bool
    random_seed: int
    train_size: float
    regenerate: bool
    # regenerate: bool
    # tune: bool
    # refit: bool
    # evaluate: bool
    # plot: bool
    # n_trials: int
    # verbose: bool
    # n_trials: int
    # method: str
    filepaths: Filepaths
    training: Training
    logging: Logging
    optimizer: Optimizer
    model: Model
