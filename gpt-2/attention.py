import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = head_size
        self.emb_size = emb_size

        self.c_attn = nn.Linear(emb_size, 3 * emb_size)
        self.c_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

        # to move the mask to the same device as the input
        self.register_buffer(
            'mask', 
            torch.tril(torch.ones((max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))
        )

    def forward(self, x: torch.tensor, use_cache: bool = True, cache: tuple = None):
        B, T, C = x.size()

        # (B, T, 3 * emb_size)
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(self.emb_size, dim=-1)

        Q = Q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)                                                                                       
        K = K.view(B, T, self.num_heads, self.head_size).transpose(1, 2)                                                                                       
        V = V.view(B, T, self.num_heads, self.head_size).transpose(1, 2)   

        if cache:
            K = torch.cat([cache[0], K], dim=2)
            V = torch.cat([cache[1], V], dim=2)
            att = Q @ K.transpose(-2, -1) / self.head_size ** 0.5
        else:
            att = Q @ K.transpose(-2, -1) / self.head_size ** 0.5
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)

        if use_cache:
            new_cache = (K, V)
            return y, new_cache
        else:
            return y, None
