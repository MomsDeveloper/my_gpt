import torch
import torch.nn as nn
from gelu import GELU

class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(emb_size, 4 * emb_size)
        self.act = GELU()
        self.c_proj = nn.Linear(4 * emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x