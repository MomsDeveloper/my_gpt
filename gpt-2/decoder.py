import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForward

class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.mlp = FeedForward(emb_size, dropout)
        self.ln_1 = nn.LayerNorm(emb_size)
        self.ln_2 = nn.LayerNorm(emb_size)
    
    def forward(self, x: torch.tensor, use_cache: bool = True, cache: tuple = None):
        prev_x = x
        x = self.ln_1(x)
        x, cache_new = self.attn(x, use_cache, cache)
        x = x + prev_x
        prev_x = x
        x = self.ln_2(x)
        x = self.mlp(x) + prev_x
        return x, cache_new



