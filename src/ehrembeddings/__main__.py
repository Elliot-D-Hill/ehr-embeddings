from functools import partial
import toml
from box import Box
from polars import read_csv
from torch import arange, cat, backends, device as torch_device
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from pytorch_metric_learning.losses import NTXentLoss
from ehrembeddings.model import CBOW
from ehrembeddings.preprocess import collate_fn, make_data, make_vocabulary


def main():
    config = Box(toml.load("config.toml"))
    df = read_csv(config.data_path)
    vocabulary = make_vocabulary(corpus=df["icd_code"])
    df = df.join(vocabulary, on="icd_code")
    contexts, targets = make_data(df=df)
    dataset = TensorDataset(contexts, targets)
    device_type = "mps" if backends.mps.is_available() else "cpu"
    device = torch_device(device_type)
    collate_batch = partial(collate_fn, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    vocab_size = vocabulary.shape[0]
    model = CBOW(vocab_size=vocab_size, embedding_dim=config.embedding_dim)
    model.to(device)
    criterion = NTXentLoss(temperature=config.temperature)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (inputs, outputs) in enumerate(dataloader):
            optimizer.zero_grad()
            input_embeddings, output_embeddings = model(inputs, outputs)
            embeddings = cat((input_embeddings, output_embeddings))
            # for NTXent loss, labels that are equal are positive pairs
            # and labels that are not equal on negative pairs
            labels = arange(input_embeddings.shape[0]).repeat(2)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(loss.item())


# Save the trained model
# torch.save(model.state_dict(), 'word2vec_model.pt')

if __name__ == "__main__":
    main()
