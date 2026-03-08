import torch
import torch.nn as nn

class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.head_size = head_size
        self.wk = nn.Linear(emb_size, head_size)
        self.wq = nn.Linear(emb_size, head_size)
        self.wv = nn.Linear(emb_size, head_size)
        
        # to move the mask to the same device as the input
        self.register_buffer('mask', torch.tril(torch.ones((max_seq_len, max_seq_len))))

    def forward(self, x: torch.tensor, use_cache: bool = True, cache: tuple = None):
        seq_len = x.size(1)

        K = self.wk(x)
        Q = self.wq(x)
        V = self.wv(x)

        if cache:
            K = torch.cat([cache[0], K], dim=1)
            V = torch.cat([cache[1], V], dim=1)
            attention_matrix = Q @ K.transpose(-2, -1) / self.head_size ** 0.5
        else:
            attention_matrix = Q @ K.transpose(-2, -1) / self.head_size ** 0.5
            attention_matrix = attention_matrix.masked_fill(self.mask[:seq_len, :seq_len] == 0, float('-inf'))
        attention_matrix = torch.softmax(attention_matrix, dim=-1)
        result = attention_matrix @ V

        if use_cache:
            new_cache = (K, V)
            return result, new_cache
        else:
            return result, None

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.head_attentions = nn.ModuleList(
            [HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)]
        )
        self.out = nn.Linear(head_size * num_heads, emb_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.tensor, use_cache: bool = True, cache: list = None):
        x_out = []
        cache_out = []
        if cache:
            for head in range(len(self.head_attentions)):
                head_x, head_cache = self.head_attentions[head](x, use_cache, cache[head])
                x_out.append(head_x)
                cache_out.append(head_cache)
        else:
            for head in range(len(self.head_attentions)):
                head_x, head_cache = self.head_attentions[head](x, use_cache, None)
                x_out.append(head_x)
                cache_out.append(head_cache)

        x = torch.cat(x_out, dim=-1)
        x = self.out(x)
        x = self.dropout(x)

        if use_cache:
            return x, cache_out
        return x, None

# num_heads = 10
# batch_size = 10
# seq_len = 20
# emb_size = 4
# head_size = 4
# max_seq_len = 25

# ha = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len)
# x = torch.rand((batch_size, seq_len, emb_size))
# ha(x)