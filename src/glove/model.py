"""GloVe."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter, defaultdict


class GloVe(nn.Module):
    """
    Global Vector Embedding.
    """

    def __init__(
        self, embed_size, vocab_size, min_occurance=1, x_max=100, alpha=0.75
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.x_max = x_max
        self.alpha = alpha
        self.min_occurrance = min_occurance
        self.vocab_size = vocab_size

        self.focal_embedding = nn.Embedding(vocab_size, embed_size)
        self.context_embedding = nn.Embedding(vocab_size, embed_size)
        self.focal_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

        for param in self.parameters():
            nn.init.xavier_normal_(param)

    def forward(self, focal_input, context_input, cooccurance_count):
        """Forward pass to calculate loss"""
        # get embedding
        focal_embed = self.focal_embedding(focal_input)
        context_embed = self.context_embedding(context_input)
        focal_bias = self.focal_biases(focal_input)
        context_bias = self.context_biases(context_input)
        # compute loss
        weight_factor = torch.pow(cooccurance_count / self.x_max, self.alpha)
        weight_factor[weight_factor > 1] = 1

        embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        log_coocurrances = torch.log(cooccurance_count)

        loss = (
            weight_factor
            * (embedding_products + focal_bias + context_bias - log_coocurrances) ** 2
        )

        return loss


def context_windows(record: list, left_size: int, right_size: int):
    """
    Create context windows from a record/sentence.
    """

    def _window(start_idx, end_idx):
        # inclusive interval
        last_index = len(record)
        window = record[max(0, start_idx) : min(last_index, end_idx + 1)]
        return window

    for i, target in enumerate(record):
        start_idx = i - left_size
        end_idx = i + right_size
        left_context = _window(start_idx, i - 1)
        rigth_context = _window(i + 1, end_idx)
        yield left_context, target, rigth_context


def compute_co_occur_matrix(
    corpus: Dataset, left_size: int, right_size: int, transform
):
    """
    Compute context-window-based co-occurance matrix of corpus.
    """
    token_counter = Counter()
    co_occur_counts = defaultdict(float)
    for record in tqdm(corpus, desc="compute co-occurance matrix"):
        if transform is not None:
            record = transform(record)
        token_counter.update(record)
        for left_context, target_token, right_context in context_windows(
            record, left_size, right_size
        ):
            for i, context_token in enumerate(left_context[::-1]):
                # i is the positional distance between target_token and context_token
                co_occur_counts[(target_token, context_token)] += 1 / (i + 1)
            for i, context_token in enumerate(right_context):
                co_occur_counts[(target_token, context_token)] += 1 / (i + 1)
    return co_occur_counts
