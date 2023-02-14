import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).mean(dim=1)
        outputs = self.linear(embeddings)
        return outputs


class NegativeSamplingLoss(nn.Module):
    def __init__(
        self, cbow, n_negative_samples, unigram_dist, batch_size, context_size, device
    ):
        super().__init__()
        self.cbow = cbow
        self.n_negative_samples = n_negative_samples
        self.unigram_dist = unigram_dist
        self.batch_size = batch_size
        self.context_size = context_size
        self.device = device

    def forward(self, positive_outputs):
        # TODO exclude positive codes from the negative codes; could use negative indexing
        negative_samples = torch.multinomial(
            self.unigram_dist,
            self.n_negative_samples * self.batch_size,
            replacement=True,
        ).view(self.batch_size, self.n_negative_samples)
        negative_outputs = self.cbow(negative_samples)
        outputs = torch.cat((positive_outputs, negative_outputs)).to(self.device)
        positive_targets = torch.ones(positive_outputs.shape[0]).float()
        negative_targets = torch.zeros(negative_samples.shape[0]).float()
        targets = (
            torch.cat((positive_targets, negative_targets)).unsqueeze(1).to(self.device)
        )
        # FIXME this is missing the dot product between outputs and embeddings
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="mean")
        return loss
