from pytorch_lightning import LightningDataModule
from torch import concat, multinomial, randint, softmax, tensor
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import cpu_count
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Normal
import polars as pl


class PretrainDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        vocab_size: int,
        sigma: float,
        n_negatives: int,
    ) -> None:
        self.dataset = df.partition_by("user", maintain_order=True)
        self.vocab_size = vocab_size
        self.sigma = sigma
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        df: pl.DataFrame = self.dataset[index]
        inputs = tensor(df["inputs"]).long()
        input_times = tensor(df["time"].to_numpy())
        inv_freq = tensor(df["inv_freq"].to_numpy())
        target_index = multinomial(input=inv_freq, num_samples=1, replacement=True)
        target = inputs[target_index]
        target_time = input_times[target_index]
        normal_distribution = Normal(loc=target_time, scale=self.sigma)
        log_prob = normal_distribution.log_prob(input_times)
        probs = softmax(log_prob, dim=0)
        positive_index = multinomial(input=probs, num_samples=1, replacement=True)
        positive = inputs[positive_index]
        negatives = randint(low=0, high=self.vocab_size, size=(self.n_negatives,))
        context = concat([positive, negatives]).long()
        labels = tensor([1] + ([0] * self.n_negatives)).float()
        return context, target, labels


class PretrainDataModule(LightningDataModule):
    def __init__(
        self,
        train,
        val,
        test,
        vocab_size,
        sigma,
        n_negatives,
        batch_size,
    ) -> None:
        super().__init__()
        self.train_dataset = PretrainDataset(
            df=train,
            vocab_size=vocab_size,
            sigma=sigma,
            n_negatives=n_negatives,
        )
        self.val_dataset = PretrainDataset(
            df=val,
            vocab_size=vocab_size,
            sigma=sigma,
            n_negatives=n_negatives,
        )
        self.test_dataset = PretrainDataset(
            df=test,
            vocab_size=vocab_size,
            sigma=sigma,
            n_negatives=n_negatives,
        )
        self.batch_size = batch_size
        self.num_workers = cpu_count() - 1

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            multiprocessing_context="fork",
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            multiprocessing_context="fork",
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            multiprocessing_context="fork",
            persistent_workers=True,
        )


class FinetuneDataset(Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        self.dataset = df.partition_by(["user", "ids"], maintain_order=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        df: pl.DataFrame = self.dataset[index]
        inputs = tensor(df["inputs"]).long()
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

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            multiprocessing_context="fork",
            # persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            multiprocessing_context="fork",
            # persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            multiprocessing_context="fork",
            # persistent_workers=True,
        )
