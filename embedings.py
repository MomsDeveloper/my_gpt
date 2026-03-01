import torch.nn as nn
import torch

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.emb_matrix = nn.Embedding(self.vocab_size, self.emb_size)
    
    def forward(self, x: torch.Tensor):
        return self.emb_matrix(x)

model = TokenEmbeddings(700, 4)
x = torch.tensor([[113, 456, 76, 345],
        [345, 678, 454, 546]])
