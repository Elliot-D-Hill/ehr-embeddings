from polars import DataFrame, Series, col, Int64
from torch.utils.data import default_collate
from torch import tensor


def collate_fn(batch, device):
    return tuple(batch_.to(device) for batch_ in default_collate(batch))


def make_vocabulary(corpus: Series) -> dict:
    return corpus.unique().to_frame().with_row_count("code_index")


def make_data(df: DataFrame) -> DataFrame:
    data = (
        df.with_columns(col("code_index").cast(Int64, strict=False))
        .groupby(by=["subject_id", "hadm_id"])
        .agg(col("code_index").alias("target"))
        .with_columns(col("target").alias("context"))
        .select(["context", "target"])
        .explode("target")
        .explode("context")
        .filter(col("context") != col("target"))
    )
    contexts = tensor(data["context"].to_numpy()).long()
    targets = tensor(data["target"].to_numpy()).long()
    return contexts, targets
