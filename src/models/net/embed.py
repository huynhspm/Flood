import torch
import math
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, frequency: float = 10000.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(frequency) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeEmbedding(nn.Module):
    def __init__(self, 
                d_model: int, 
                dropout: float = 0.1, 
                max_len: int = 5000, 
                frequency: float = 10000.0,
                amplitude: float = 1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(frequency) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = amplitude * torch.sin(position * div_term)
        pe[:, 1::2] = amplitude * torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, time: Tensor) -> Tensor:
        # time: batch, seq_len
        # Output: batch, seq_len, d_model
        return self.pe[time]