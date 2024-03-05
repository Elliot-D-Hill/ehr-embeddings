import toml
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
import faiss

from ehrembeddings.config import Config
from ehrembeddings.model import (
    make_finetune_model,
    make_pretrain_model,
)
from ehrembeddings.dataset import (
    FinetuneDataModule,
    PretrainDataModule,
)
from ehrembeddings.preprocess import get_data
from ehrembeddings.trainer import make_trainer


def main():
    config = Config(**toml.load("config.toml"))
    seed_everything(config.random_seed)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    train, val, test = get_data(config=config, tokenizer=tokenizer)
    pretrain_data_module = PretrainDataModule(
        train=train,
        val=val,
        test=test,
        vocab_size=tokenizer.vocab_size,
        sigma=config.training.sigma,
        n_negatives=config.training.n_negatives,
        batch_size=config.training.batch_size,
    )
    pretrainer = make_trainer(
        config=config, ckpt_folder=config.filepaths.pretrain_checkpoints
    )
    pretrain_model = make_pretrain_model(
        config=config, vocab_size=tokenizer.vocab_size, pretrain=config.pretrain
    )
    if config.pretrain:
        pretrainer.fit(pretrain_model, datamodule=pretrain_data_module)
    if config.evaluate:
        pretrainer.test(model=pretrain_model, datamodule=pretrain_data_module)
    trainer = make_trainer(
        config=config, ckpt_folder=config.filepaths.finetune_checkpoints
    )
    finetune_data_module = FinetuneDataModule(
        train=train, val=val, test=test, batch_size=config.training.batch_size
    )
    finetune_model = make_finetune_model(
        embeddings=pretrain_model.model.target_embeddings.weight,
        config=config,
        finetune=config.finetune,
    )
    if config.finetune:
        trainer.fit(model=finetune_model, datamodule=finetune_data_module)
    if config.evaluate:
        trainer.test(model=finetune_model, datamodule=finetune_data_module)

    k = 10
    embeddings = pretrain_model.model.target_embeddings.weight.detach().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    print(index.is_trained)
    index.add(embeddings)
    text = ["happy sad angry hungry dog violence game research statistics"]
    indices = tokenizer(text=text)["input_ids"]
    distances, neighbors = index.search(embeddings[indices], k)
    for i, (distance, k_neighbors) in enumerate(zip(distances, neighbors)):
        if i not in {0, k}:
            print(tokenizer.convert_ids_to_tokens(k_neighbors))
            print(distances)
            print()


if __name__ == "__main__":
    main()
