# я удалил ненужное, и добавил GPT2
import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class miniGPT2(nn.Module):
    def __init__(self, vocab_size, d_model: int=8, max_len: int=5000, nhead: int=8, pad_idx: int=0):
        super().__init__()
        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.emb = nn.Embedding(vocab_size, d_model, pad_idx)
        self.decoder = nn.TransformerEncoderLayer(d_model, nhead, d_model)
        self.pad_idx = pad_idx

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pad_mask = (x == self.pad_id)
        att_mask = generate_square_subsequent_mask(x.shape[0])
        x = self.pe(self.emb(x))
        return self.decoder(x, att_mask, pad_mask)
