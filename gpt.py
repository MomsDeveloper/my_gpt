import torch
import torch.nn as nn
from embedings import TokenEmbeddings, PositionalEmbeddings
from decoder import Decoder

class GPT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, emb_size: int, num_heads: int,  head_size: int, num_layers: int, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
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
