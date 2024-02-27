import polars as pl
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ehrembeddings.config import Config

SECONDS_PER_DAY = 24 * 60 * 60


def make_data(config: Config, tokenizer: AutoTokenizer):
    column_names = ["label", "ids", "date", "flag", "user", "text"]
    df = (
        pl.read_csv(
            config.filepaths.data,
            has_header=False,
            n_rows=100 if config.fast_dev_run else None,
        )
        .rename({f"column_{i+1}": name for i, name in enumerate(column_names)})
        .with_columns(
            pl.col("text").str.strip("@(\w+)"),
            pl.col("label").replace({4: 1}),  # 0 = negative, 4 = positive
            pl.col("date")
            .str.replace("PDT ", "")
            .str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %Y"),
        )
        .with_columns(
            pl.col("date")
            .sub(pl.col("date").min())
            .dt.seconds()
            .truediv(SECONDS_PER_DAY)
            .alias("time"),
        )
        .sort("user", "time")
        .select(["user", "time", "text", "label"])
    )
    text = df["text"].to_list()
    inputs = tokenizer(text, padding=False)
    df = (
        df.select(pl.exclude("text"))
        .with_columns(inputs=pl.Series(inputs["input_ids"]))
        .explode("inputs")
    )
    value_counts = df["inputs"].value_counts(parallel=True)
    inverse_frequency = value_counts.with_columns(
        (1 / value_counts["count"]).alias("inv_freq")
    ).select(["inputs", "inv_freq"])
    df = df.join(other=inverse_frequency, on="inputs")
    return df


def get_data(config: Config, tokenizer: AutoTokenizer):
    if config.regenerate:
        df = make_data(config=config, tokenizer=tokenizer)
        dfs = df.partition_by("user")
        train, val_test = train_test_split(
            dfs, train_size=config.train_size, random_state=config.random_seed
        )
        val, test = train_test_split(
            val_test, train_size=0.5, random_state=config.random_seed
        )
        train = pl.concat(train)
        val = pl.concat(val)
        test = pl.concat(test)
        train.write_parquet(config.filepaths.train)
        val.write_parquet(config.filepaths.val)
        test.write_parquet(config.filepaths.test)
    else:
        train = pl.read_parquet(config.filepaths.train)
        val = pl.read_parquet(config.filepaths.val)
        test = pl.read_parquet(config.filepaths.test)
    train = train.partition_by("user", maintain_order=True)
    val = val.partition_by("user", maintain_order=True)
    test = test.partition_by("user", maintain_order=True)
    return train, val, test
