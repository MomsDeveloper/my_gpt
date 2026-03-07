import torch
import torch.nn as nn
from gelu import GELU

class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            GELU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.tensor):
        return self.ffn(x)