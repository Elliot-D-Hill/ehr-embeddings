from functools import partial
from pytorch_lightning import LightningDataModule
from torch import arange, concat, multinomial, randint, tensor, zeros, exp, max, min
from torch.backends.mps import is_available
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import cpu_count
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Normal
import polars as pl

from ehrembeddings.config import Config


class SkipGramDataset(Dataset):
    def __init__(
        self,
        dfs: list[pl.DataFrame],
        window_size: int,
        vocab_size: int,
        n_negatives: int,
    ) -> None:
        self.dfs = dfs
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, index):
        df: pl.DataFrame = self.dfs[index]
        inv_freq = tensor(df["inv_freq"].to_numpy())
        target_index = multinomial(input=inv_freq, num_samples=1, replacement=True)
        inputs = df["inputs"]
        anchor = inputs[target_index]
        window_start = max(target_index - self.window_size, 0)
        window_stop = min(target_index + self.window_size + 1, df.shape[0])
        weights = zeros(df.shape[0])
        weights[window_start:window_stop] = 1
        weights[target_index] = 0
        positive_index = multinomial(weights, num_samples=1, replacement=True)
        positive = inputs[positive_index]
        negatives = randint(low=0, high=self.vocab_size, size=(self.n_negatives,))
        return anchor, positive, negatives


class SoftSkipGramDataset(Dataset):
    def __init__(
        self,
        dfs: list[pl.DataFrame],
        vocab_size: int,
        sigma: float,
        n_negatives: int,
    ) -> None:
        self.dfs = dfs
        self.vocab_size = vocab_size
        self.sigma = sigma
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, index):
        df: pl.DataFrame = self.dfs[index]
        inputs = tensor(df["inputs"]).long()
        input_times = tensor(df["time"].to_numpy())
        inv_freq = tensor(df["inv_freq"].to_numpy())
        anchor_index = multinomial(input=inv_freq, num_samples=1, replacement=True)
        anchor = inputs[anchor_index]
        anchor_time = input_times[anchor_index]
        normal_distribution = Normal(loc=anchor_time, scale=self.sigma)
        log_probs = normal_distribution.log_prob(input_times)
        probs = exp(log_probs)
        positive_index = multinomial(input=probs, num_samples=1, replacement=True)
        positive = inputs[positive_index]
        negatives = randint(low=0, high=self.vocab_size, size=(self.n_negatives,))
        return anchor, positive, negatives


class KernelDataset(Dataset):
    def __init__(
        self,
        dfs: list[pl.DataFrame],
        vocab_size: int,
        sigma: float,
    ) -> None:
        self.dfs = dfs
        self.vocab_size = vocab_size
        self.sigma = sigma

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, index):
        df: pl.DataFrame = self.dfs[index]
        inputs = tensor(df["inputs"]).long()
        input_times = tensor(df["time"].to_numpy())
        inv_freq = tensor(df["inv_freq"].to_numpy())
        anchor_index = multinomial(input=inv_freq, num_samples=1, replacement=True)
        anchor = inputs[anchor_index]
        anchor_time = input_times[anchor_index]
        normal_distribution = Normal(loc=anchor_time, scale=self.sigma)
        log_probs = normal_distribution.log_prob(input_times)
        return anchor, inputs, log_probs


def make_dataset(df: pl.DataFrame, vocab_size: int, config: Config):
    match config.method:
        case "w2v":
            dataset_class = partial(
                SkipGramDataset,
                window_size=config.training.window_size,
                n_negatives=config.training.n_negatives,
            )
        case "soft":
            dataset_class = partial(
                SoftSkipGramDataset,
                sigma=config.training.sigma,
                n_negatives=config.training.n_negatives,
            )
        case "kernel":
            dataset_class = partial(KernelDataset, sigma=config.training.sigma)
        case _:
            raise ValueError(f"Unknown method: {config.method}")
    dfs = df.partition_by("user", maintain_order=True)
    dataset = dataset_class(dfs=dfs, vocab_size=vocab_size)
    return dataset


class PretrainDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        train: Dataset,
        val: Dataset,
        test: Dataset,
    ) -> None:
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size
        self.num_workers = cpu_count() - 1
        self.multiprocessing_context = "fork" if is_available() else None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )


def make_pretrain_data_module(
    train,
    val,
    test,
    vocab_size,
    config: Config,
):
    train = make_dataset(df=train, vocab_size=vocab_size, config=config)
    val = make_dataset(df=val, vocab_size=vocab_size, config=config)
    test = make_dataset(df=test, vocab_size=vocab_size, config=config)
    return PretrainDataModule(
        train=train, val=val, test=test, batch_size=config.training.batch_size
    )


class FinetuneDataset(Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        self.dataset: list[pl.DataFrame] = df.partition_by(
            ["user", "ids"], maintain_order=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        df: pl.DataFrame = self.dataset[index]
        inputs = tensor(df["input"]).long()
        label = tensor(df["label"][0])
        return inputs, label


def collate(batch):
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = tensor(labels).unsqueeze(1).float()
    return inputs, labels


class FinetuneDataModule(LightningDataModule):
    def __init__(self, train, val, test, batch_size) -> None:
        super().__init__()
        self.train_dataset = FinetuneDataset(df=train)
        self.val_dataset = FinetuneDataset(df=val)
        self.test_dataset = FinetuneDataset(df=test)
        self.batch_size = batch_size
        self.num_workers = cpu_count() - 1
        self.multiprocessing_context = "fork" if is_available() else None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=True,
        )
