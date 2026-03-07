
import torch
from torch.utils import data

class GetData(data.Dataset):
    def __init__(self, data: list, seq_len: int, device: str):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device
    
    def __len__(self):
        N = len(self.data)
        return N - self.seq_len - 1
    
    def __getitem__(self, index: int):
        x = torch.tensor(self.data[index:index + self.seq_len], device=self.device)
        y = torch.tensor(self.data[index + 1: index + self.seq_len + 1], device=self.device)
        return x, y

