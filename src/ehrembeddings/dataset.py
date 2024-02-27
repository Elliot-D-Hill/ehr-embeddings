from pytorch_lightning import LightningDataModule
from torch import concat, multinomial, randint, softmax, tensor
from torch.utils.data import DataLoader, Dataset
import polars as pl
from torch.distributions import Normal
from transformers import AutoTokenizer


class EHRDataset(Dataset):
    def __init__(
        self,
        dfs: pl.DataFrame,
        tokenizer: AutoTokenizer,
        inverse_frequency: dict,
        sigma: float,
        n_negatives: int,
    ) -> None:
        self.dataset = dfs
        self.tokenizer = tokenizer
        self.inverse_frequency = inverse_frequency
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
        negatives = randint(
            low=0, high=self.tokenizer.vocab_size, size=(self.n_negatives,)
        )
        context = concat([positive, negatives]).long()
        labels = tensor([1] + ([0] * self.n_negatives)).float()
        return context, target, labels


class EmbeddingsDataModule(LightningDataModule):
    def __init__(
        self,
        train,
        val,
        test,
        tokenizer,
        inverse_frequency,
        sigma,
        n_negatives,
        batch_size,
    ) -> None:
        super().__init__()
        self.train_dataset = EHRDataset(
            dfs=train,
            tokenizer=tokenizer,
            inverse_frequency=inverse_frequency,
            sigma=sigma,
            n_negatives=n_negatives,
        )
        self.val_dataset = EHRDataset(
            dfs=val,
            tokenizer=tokenizer,
            inverse_frequency=inverse_frequency,
            sigma=sigma,
            n_negatives=n_negatives,
        )
        self.test_dataset = EHRDataset(
            dfs=test,
            tokenizer=tokenizer,
            inverse_frequency=inverse_frequency,
            sigma=sigma,
            n_negatives=n_negatives,
        )
        self.batch_size = batch_size
        self.num_workers = 0  # cpu_count()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            # persistent_workers=True,
            # multiprocessing_context="fork",
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            # persistent_workers=True,
            # multiprocessing_context="fork",
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            # persistent_workers=True,
            # multiprocessing_context="fork",
        )
