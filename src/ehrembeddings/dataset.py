from multiprocessing import cpu_count
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
import polars as pl


def make_vocabulary(corpus: pl.Series) -> pl.DataFrame:
    return corpus.unique().sort().to_frame().with_row_count("code_index")


def make_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.concat_str(["icd_code", "icd_version"], separator="_"))
        .groupby(["subjectkey", "encounter"])
        .agg(pl.col("context"))
        .with_columns(pl.col("context").arr.to_struct(fields=["context", "target"]))
        .unnest("context")
    )


class EmbeddingsDataModule(LightningDataModule):
    def __init__(self, contexts, targets, batch_size) -> None:
        super().__init__()
        self.train_dataset = TensorDataset(contexts, targets)
        self.batch_size = batch_size
        self.num_workers = cpu_count()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
