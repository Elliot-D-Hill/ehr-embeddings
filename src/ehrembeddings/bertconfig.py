from pydantic import BaseModel
from pathlib import Path


class Model(BaseModel):
    vocab_size: int
    sequence_length: int
    hidden_dim: int
    num_heads: int
    num_layers: int


class Optimizer(BaseModel):
    lr: float
    momentum: float
    weight_decay: float


class Filepaths:
    dataset: Path


class Config(BaseModel):
    random_seed: int
    fast_dev_run: bool
    pretrain: bool
    model: Model
    optimizer: Optimizer
