import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import Adam
from embedings import TokenEmbeddings, PositionalEmbeddings
from decoder import Decoder
from tqdm import tqdm

class GPT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, emb_size: int, num_heads: int,  head_size: int, num_layers: int, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.loss_lst = []
        self.loss_lst_val = []

        self.token_emb = TokenEmbeddings(vocab_size, emb_size)
        self.pos_emb = PositionalEmbeddings(max_seq_len, emb_size)
        self.drop = nn.Dropout(dropout)
        self.decoders = nn.ModuleList(
            [Decoder(num_heads, emb_size, head_size, max_seq_len, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(emb_size, vocab_size)
    
    def fit(self, train_loader: data.DataLoader, valid_loader: data.DataLoader, num_epoch: int, learning_rate: float):
        self.to(self.device)
        optimizer = Adam(self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()

        description = f'Train Loss: {self.loss_lst[-1]:.4f}, Val Loss: {self.loss_lst_val[-1]:.4f}' if self.loss_lst and self.loss_lst_val else 'Training...'
        for _ in tqdm(range(num_epoch), desc=description, unit='epoch'):
            self.train()
 
            loss_mean = 0 
            lm_count = 0
            for inputs, targets in tqdm(train_loader, desc='Training', unit='batch', leave=False):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # logits shape: (batch_size, seq_len, vocab_size)
                # result shape: (batch_size * seq_len, vocab_size)
                logits = self(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.flatten()
                loss = loss_func(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean                   

            self.eval()
            
            Q_val = 0
            count_val = 0

            for x_val, y_val in valid_loader:
                with torch.no_grad():
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    logits = self(x_val)
                    logits = logits.view(-1, logits.size(-1))
                    y_val = y_val.flatten()
                    loss = loss_func(logits, y_val)
                    Q_val += loss.item()
                    count_val += 1

            Q_val /= count_val

            self.loss_lst.append(loss_mean)
            self.loss_lst_val.append(Q_val)


    def forward(self, x: torch.tensor):
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(x.size(1))
        embedding = token_embeddings + position_embeddings
        x = self.drop(embedding)
        for decoder in self.decoders:
            x = decoder(x)
        x = self.linear(x)
        return x

    def generate(self, x: torch.tensor, max_new_tokens: int, do_sample: bool, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        for _ in range(max_new_tokens):
            x_croppped = x[:, -self.max_seq_len:]
            logits = self.forward(x_croppped)
            last_logit = logits[:, -1, :] / temperature
            if top_k and do_sample:
                top_values, top_indicies = torch.topk(last_logit, top_k, dim=-1)
                filtered = torch.full_like(last_logit, float('-inf'))
                filtered.scatter_(-1, top_indicies, top_values)
                last_logit = filtered

            if top_p and do_sample:
                sorted_logits, sorted_indicies = last_logit.sort(-1, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)

                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs >= top_p
                mask[:, 0] = False

                sorted_logits[mask] = float('-inf')

                filtered = sorted_logits.scatter(-1, sorted_indicies, sorted_logits)
                last_logit = filtered

            last_vector = torch.softmax(last_logit, dim=-1)

            if not do_sample:
                next_token = torch.argmax(last_vector, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(last_vector, num_samples=1)
            
            x = torch.cat([x, next_token], dim=1)
        return x

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)
    
    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

# Example parameters
# vocab_size = 100
# max_seq_len = 16
# emb_size = 32
# num_heads = 4
# head_size = 8
# num_layers = 2
# dropout = 0.1

# model = GPT(vocab_size, max_seq_len, emb_size, num_heads, head_size, num_layers, dropout)

# # Test input: batch_size=2, seq_len=10
# x = torch.randint(0, vocab_size, (2, 10))  # random token IDs
# print("Input shape:", x.shape)        # (2, 10)

# output = model(x)
# print("Output shape:", output.shape)  # should be (2, 10, vocab_size)

# model.generate(x, 10, True, 1, None, 0.1)