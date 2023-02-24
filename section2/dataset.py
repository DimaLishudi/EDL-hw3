from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset, Sampler
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np

MAX_LENGTH = 640


class BrainDataset(Dataset):
    def __init__(self, data_path: str="data/wikitext-103-raw/", max_length: int = MAX_LENGTH):
        tokenizer = get_tokenizer("basic_english")
        raw_texts = []
        with open(data_path + "wiki.valid.raw") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    raw_texts.append(tokenizer(line))
        with open(data_path + "wiki.test.raw") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    raw_texts.append(tokenizer(line))

        vocab = build_vocab_from_iterator(raw_texts, specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        data=torch.zeros((len(raw_texts), max_length), dtype=int)
        for i, line in enumerate(raw_texts):
            idxs = vocab(line)[:MAX_LENGTH]
            data[i,:len(idxs)] = torch.tensor(idxs, dtype=int)

        self.tokenizer = tokenizer
        self.vocab=vocab
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str="data/wikitext-103-raw/", max_length: int = MAX_LENGTH):
        tokenizer = get_tokenizer("basic_english")
        raw_texts = []
        with open(data_path + "wiki.valid.raw") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    raw_texts.append(tokenizer(line))
        with open(data_path + "wiki.test.raw") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    raw_texts.append(tokenizer(line))

        vocab = build_vocab_from_iterator(raw_texts, specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        data = []
        for line in raw_texts:
            idxs = vocab(line)[:max_length]
            data.append(torch.tensor(idxs, dtype=int))

        self.tokenizer = tokenizer
        self.vocab=vocab
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str="data/wikitext-103-raw/", max_length: int = MAX_LENGTH, n_bins: int = 1):
        tokenizer = get_tokenizer("basic_english")
        raw_texts = []
        with open(data_path + "wiki.valid.raw") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    raw_texts.append(tokenizer(line))
        with open(data_path + "wiki.test.raw") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    raw_texts.append(tokenizer(line))

        vocab = build_vocab_from_iterator(raw_texts, specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        data = []
        for line in raw_texts:
            idxs = vocab(line)[:max_length]
            data.append(torch.tensor(idxs, dtype=int))

        data = sorted(data, key=lambda x: len(x))

        self.tokenizer = tokenizer
        self.vocab=vocab
        self.data = data
        self.n_bins = n_bins

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class UltraDuperBigBrainSampler(Sampler):
    def __init__(self, batch_size, n_bins, size):
        self.batch_size = batch_size
        self.n_bins = n_bins
        self.max_idx = size
        self.bin_length = size // n_bins
        self.remainder = size % n_bins
    
    def __iter__(self):
        while True:
            bin_idx = np.random.randint(self.n_bins)
            bin_start = bin_idx * self.bin_length
            # первые remainder бинов будут больше остальных на 1
            bin_start += min(bin_idx, self.remainder)
            bin_end = bin_start + self.bin_length
            bin_end += (bin_idx < self.remainder)
            replacement = self.batch_size > (bin_end - bin_start)
            yield np.random.choice(np.arange(bin_start, bin_end), self.batch_size, replacement)

    def __len__(self):
        return float("inf")



# Не использовал max_length, т.к. проще сразу в трейне до него всё обрезать.
def collate_fn(
    batch: List[Tuple[str, torch.Tensor]], max_length: Optional[int] = MAX_LENGTH
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """

    text_list = torch.nn.utils.rnn.pad_sequence(batch, padding_value=0, batch_first=True)
    return text_list