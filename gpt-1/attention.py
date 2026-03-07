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

    def forward(self, x: torch.tensor):
        seq_len = x.size(1)

        K = self.wk(x)
        Q = self.wq(x)
        V = self.wv(x)

        attention_matrix = Q @ K.transpose(-2, -1) / self.head_size ** 0.5
        attention_matrix = attention_matrix.masked_fill(self.mask[:seq_len, :seq_len] == 0, float('-inf'))
        attention_matrix = torch.softmax(attention_matrix, dim=-1)

        return attention_matrix @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.head_attentions = nn.ModuleList(
            [HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)]
        )
        self.out = nn.Linear(head_size * num_heads, emb_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.tensor):
        head_outputs = [head(x) for head in self.head_attentions]
        x = torch.cat(head_outputs, dim=-1)
        x = self.out(x)
        x = self.dropout(x)
        return x

# num_heads = 10
# batch_size = 10
# seq_len = 20
# emb_size = 4
# head_size = 4
# max_seq_len = 25

# ha = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len)
# x = torch.rand((batch_size, seq_len, emb_size))
# ha(x)