import polars as pl
import torch
from torch.utils.data import Dataset


def create_unigram_dist(corpus):
    word_counts = pl.Series(corpus).value_counts()["counts"]
    return torch.tensor(word_counts / word_counts.sum())


# TODO this function will need to be updated once EmbeddingBag is used
def collate_fn(batch, context_size):
    contexts, targets = zip(*batch)
    context_windows = []
    for context in contexts:
        indices = torch.randint(len(context), (context_size,))
        context_window = torch.tensor(context, dtype=torch.long)[indices]
        context_windows.append(context_window)
    return torch.stack(context_windows), targets


class EHRDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets

    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]
