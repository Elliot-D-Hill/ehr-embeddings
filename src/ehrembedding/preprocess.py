from polars import DataFrame, Series, col, Int64
from torch import float64, tensor


def make_vocabulary(corpus: Series) -> dict:
    return corpus.unique().to_frame().with_row_count("code_index")


def make_data(df: DataFrame) -> DataFrame:
    data = (
        df.with_columns(col("code_index").cast(Int64, strict=False))
        .groupby(by=["subject_id", "hadm_id"])
        .agg([col("code_index").alias("context")])
        .with_columns(col("context").alias("target"))
        .explode("target")
        .explode("context")
        .filter(col("context") != col("target"))
        .groupby(by=["subject_id", "hadm_id", "target"])
        .agg(col("context"))
        .select(["context", "target"])
    )
    contexts = data["context"].to_list()
    targets = tensor(data["target"].to_numpy(), dtype=float64)
    return contexts, targets
