from functools import partial
from box import Box
from polars import read_csv
import toml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ehrembedding.dataset import EHRDataset, collate_fn, create_unigram_dist
from ehrembedding.model import CBOW, NegativeSamplingLoss
from ehrembedding.preprocess import make_data, make_vocabulary


def main():
    config = Box(toml.load("config.toml"))
    df = read_csv(config.data_path)
    corpus = df["icd_code"]
    vocabulary = make_vocabulary(corpus=corpus)
    df = df.join(vocabulary, on="icd_code")
    contexts, targets = make_data(df=df)
    dataset = EHRDataset(contexts=contexts, targets=targets)
    collate_batch = partial(collate_fn, context_size=config.context_size)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch
    )
    vocab_size = vocabulary.shape[0]
    model = CBOW(vocab_size=vocab_size, embedding_dim=config.embedding_dim)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Is mps built:", torch.backends.mps.is_built())
    print("has mps:", torch.has_mps)
    print("Is mps available:", torch.backends.mps.is_available())
    print("Device:", device)
    model.to(device)
    unigram_dist = create_unigram_dist(corpus=corpus)
    criterion = NegativeSamplingLoss(
        cbow=model,
        n_negative_samples=config.n_negative_samples,
        unigram_dist=unigram_dist,
        batch_size=config.batch_size,
        context_size=config.context_size,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_epochs = 10
    for epoch in range(num_epochs):
        for context, target in dataloader:
            optimizer.zero_grad()
            outputs = model(context)
            loss = criterion(outputs)
            loss.backward()
            optimizer.step()
            print(loss.item())


# Save the trained model
# torch.save(model.state_dict(), 'word2vec_model.pt')

if __name__ == "__main__":
    main()
