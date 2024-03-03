from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
)
from torch.utils.data import Dataset
import evaluate
from ehrembeddings.bertconfig import Config
from toml import load
import polars as pl
from torch.nn import BCEWithLogitsLoss


class FineTuneDataset(Dataset):
    def __init__(self, sentences, attention_mask, labels) -> None:
        super().__init__()
        self.sentences = sentences
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return {
            "label": self.labels[index],
            "attention_mask": self.attention_mask[index],
            "input_ids": self.sentences[index],
        }


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"])
        labels = inputs["labels"].float()
        loss = self.criterion(outputs.logits.view(-1), labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    roc_auc_metric = evaluate.load("roc_auc")
    roc_auc = roc_auc_metric.compute(references=labels, prediction_scores=logits)
    return roc_auc


def finetune():
    config_data = load("bert_config.toml")
    config = Config(**config_data)
    tokenizer = RobertaTokenizerFast.from_pretrained("data/tokenizer")
    column_names = ["label", "ids", "date", "flag", "user", "text"]
    df = (
        (
            pl.read_csv(
                "data/datasets/sentiment_analysis.csv", has_header=False
            ).rename({f"column_{i+1}": name for i, name in enumerate(column_names)})
        )
        .with_columns(pl.col("label").replace({4: 1}))
        .sample(10_000)
    )
    train, test = train_test_split(df, test_size=0.2)
    # train = pl.read_parquet("data/datasets/train.csv")
    # test = pl.read_parquet("data/datasets/test.csv")
    train_text = train["text"].to_list()
    test_text = test["text"].to_list()
    train_labels = train["label"].to_list()
    test_labels = test["label"].to_list()
    train_tokens = tokenizer(
        train_text,
        max_length=config.model.sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    test_tokens = tokenizer(
        test_text,
        max_length=config.model.sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    train_dataset = FineTuneDataset(
        sentences=train_tokens.input_ids,
        attention_mask=train_tokens.attention_mask,
        labels=train_labels,
    )
    test_dataset = FineTuneDataset(
        sentences=test_tokens.input_ids,
        attention_mask=test_tokens.attention_mask,
        labels=test_labels,
    )
    classifier = RobertaForSequenceClassification.from_pretrained(
        "data/model", num_labels=1
    )
    training_args = TrainingArguments(
        output_dir="data/finetuned_bert",
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        use_mps_device=True,
        learning_rate=5e-4,
        evaluation_strategy="steps",
        eval_steps=1000,
    )
    trainer = CustomTrainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    print(results)


if __name__ == "__main__":
    finetune()
