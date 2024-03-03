from toml import load
from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    # RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from ehrembeddings.custombert import CustomRobertaForMaskedLM
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset
import polars as pl
from sklearn.model_selection import train_test_split
from ehrembeddings.bertconfig import Config


class MLMDataset(Dataset):
    def __init__(self, sentences) -> None:
        super().__init__()
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return {"input_ids": self.sentences[index]}


# class CustomRobertaEmbeddings(RobertaEmbeddings):
#     def __init__(self, config, *args, **kwargs):
#         super().__init__(config, *args, **kwargs)
#         # Initialize any additional parameters for your custom embeddings here.
#         self.time2vec = Time2Vec()

#     def forward(self, *args, **kwargs):

#         outputs = super().forward(*args, **kwargs)
#         time2vec_outputs = self.time2vec(outputs)
#         outputs = outputs + time2vec_output
#         return outputs


def train_bert():
    config_data = load("bert_config.toml")
    config = Config(**config_data)
    column_names = ["label", "ids", "date", "flag", "user", "text"]
    df = (
        pl.read_csv("data/datasets/sentiment_analysis.csv", has_header=False)
        .rename({f"column_{i+1}": name for i, name in enumerate(column_names)})
        .sample(10_000)
    )
    text = df["text"].to_list()
    train, val = train_test_split(text, test_size=0.2)
    if config.pretrain:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            train,
            vocab_size=config.model.vocab_size,
            min_frequency=0,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )
        tokenizer.truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_model("data/tokenizer")
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained("data/tokenizer")
    train_tokens = tokenizer(
        train,
        max_length=config.model.sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    val_tokens = tokenizer(
        val,
        max_length=config.model.sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    X_train = MLMDataset(sentences=train_tokens.input_ids)
    X_val = MLMDataset(sentences=val_tokens.input_ids)
    roberta_config = RobertaConfig(
        tokenizer.vocab_size,
        max_position_embeddings=config.model.sequence_length + 2,
        hidden_size=config.model.hidden_dim,
        num_attention_heads=config.model.num_heads,
        num_hidden_layers=config.model.num_layers,
        type_vocab_size=1,
    )
    model = CustomRobertaForMaskedLM(config=roberta_config)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir="data/model",
        learning_rate=5e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        use_mps_device=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=X_train,
        eval_dataset=X_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    model.save_pretrained("data/model")
    eval_results = trainer.evaluate()
    print(f"Perplexity: {eval_results['eval_loss']:.2f}")


if __name__ == "__main__":
    train_bert()
