import torch
import torch.nn as nn
from embedings import TokenEmbeddings, PositionalEmbeddings
from decoder import Decoder

class GPT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, emb_size: int, num_heads: int,  head_size: int, num_layers: int, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = TokenEmbeddings(vocab_size, emb_size)
        self.pos_emb = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.decoders = nn.ModuleList(
            [Decoder(num_heads, emb_size, head_size, max_seq_len, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(emb_size, vocab_size)
    
    def forward(self, x: torch.tensor):
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(x.size(1))
        embedding = token_embeddings + position_embeddings
        x = self.dropout(embedding)
        for decoder in self.decoders:
            x = decoder(x)
        x = self.linear(x)
        return x

    def generate(self, x: torch.tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            x_croppped = x[:, -self.max_seq_len:]
            logits = self.forward(x_croppped)
            last_vector = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(last_vector, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
        return x

# Example parameters
vocab_size = 100
max_seq_len = 16
emb_size = 32
num_heads = 4
head_size = 8
num_layers = 2
dropout = 0.1

model = GPT(vocab_size, max_seq_len, emb_size, num_heads, head_size, num_layers, dropout)

# Test input: batch_size=2, seq_len=10
x = torch.randint(0, vocab_size, (2, 10))  # random token IDs
print("Input shape:", x.shape)        # (2, 10)

output = model(x)
print("Output shape:", output.shape)  # should be (2, 10, vocab_size)

model.generate(x, 10)