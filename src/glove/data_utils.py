from torch.utils.data import Dataset, ConcatDataset
from torchtext.vocab import build_vocab_from_iterator
from collections import defaultdict


class Corpus(Dataset):

    def __init__(self, data) -> None:
        self.data = data

    def __getitem__(self, index) -> list:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def build_vocab(corpus_list: list[Corpus]):
    total_corpus = ConcatDataset(corpus_list)
    vocab = build_vocab_from_iterator(
        total_corpus, specials=["<PAD>", "<UNK>"], special_first=True
    )
    vocab.set_default_index(vocab["<UNK>"])
    return vocab


def context_windows(record: list, left_size: int, right_size: int):
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


class GloVeDataset(Dataset):

    def __init__(self, cooccurrance: defaultdict) -> None:
        super().__init__()
        self.cooccurrance: list = list(cooccurrance.items())

    def __getitem__(self, index) -> tuple:
        return self.cooccurrance[index]

    def __len__(self) -> int:
        return len(self.cooccurrance)
