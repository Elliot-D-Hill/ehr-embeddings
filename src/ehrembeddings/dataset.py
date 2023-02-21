from multiprocessing import cpu_count
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
from polars import DataFrame, Series, col, Int64, concat_str


def make_vocabulary(corpus: Series) -> dict:
    return corpus.unique().sort().to_frame().with_row_count("code_index")


def make_data(df: DataFrame) -> DataFrame:
    return (
        df.with_columns(concat_str(["icd_code", "icd_version"], sep="_"))
        .with_columns(col("code_index").cast(Int64, strict=False))
        .groupby(by=["subject_id", "hadm_id"])
        .agg(col("code_index").alias("target"))
        .with_columns(col("target").alias("context"))
        .select(["context", "target"])
        .explode("target")
        .explode("context")
        .filter(col("context") != col("target"))
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
