import glob
import os

import torch
from bpe import BPE
from torch.utils.data import DataLoader
from dataset import GetData
from gpt import GPT
import matplotlib.pyplot as plt

vocab_size = 2000
max_seq_len = 512
emb_size = 512
num_heads = 8
head_size = emb_size // num_heads
num_layers = 12
dropout = 0.2
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
seq_len = 64
learning_rate = 0.00001
num_epoch = 48

# Get all text from the files in the texts folder
all_text = []
for file_path in glob.glob('./texts/*.*'):
    file = open(file_path, 'r', encoding='utf8')
    all_text.append(file.read())

all_text = '\n\n\n'.join(all_text)
print(sorted(set(all_text)))

# Fit BPE tokenizer and encode the text
if not os.path.exists('./data/bpe.dill'):
    bpe = BPE(vocab_size)
    bpe.fit(all_text)
    bpe.save('./data/bpe.dill')
else: 
    bpe = BPE.load('./data/bpe.dill')

encoded_text = bpe.encode(all_text)
print(encoded_text)

# Split the data into train and validation sets
n = int(0.9*len(encoded_text)) # 90% train
train_token_ids = encoded_text[:n]
valid_token_ids = encoded_text[n:]
train_dataset = GetData(data=train_token_ids, seq_len=seq_len, device=device)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

valid_dataset = GetData(data=valid_token_ids, seq_len=seq_len, device=device)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Initialize the model and start training
model = GPT(vocab_size, max_seq_len, emb_size, num_heads, head_size, num_layers, dropout, device)

model.fit(train_loader=train_loader, valid_loader=valid_loader, num_epoch=num_epoch, learning_rate=learning_rate)
    
model.save('./data/gpt.dill')

# plot of the training and validation loss
plt.plot(model.loss_lst, label='train loss')
plt.plot(model.loss_lst_val, label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig('./data/loss_plot.png')