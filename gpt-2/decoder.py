import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import FeedForward

class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.mh_attention = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.ffn = FeedForward(emb_size, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
    
    def forward(self, x: torch.tensor):
        prev_x = x
        x = self.norm1(x)
        x = self.mh_attention(x) + prev_x
        prev_x = x
        x = self.norm2(x)
        x = self.ffn(x) + prev_x
        return x



