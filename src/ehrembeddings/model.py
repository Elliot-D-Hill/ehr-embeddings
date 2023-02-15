import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)
        self.model = nn.Sequential(self.embeddings, self.linear)

    def forward(self, inputs, outputs):
        input_embeddings = self.model(inputs)
        output_embeddings = self.model(outputs)
        return input_embeddings, output_embeddings
