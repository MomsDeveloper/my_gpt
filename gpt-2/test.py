from gpt import GPT2                                                                                                                                           
import torch                                                                                                       
m = GPT2(vocab_size=50257, max_seq_len=1024, emb_size=768,                                                                                                     
        num_heads=12, head_size=64, num_layers=12, dropout=0.0)                                                                                                
m.eval()                                                                                                                                                       
x = torch.randint(0, 50257, (1, 10))                                                                                                                           
with torch.no_grad():                                                                                                                                          
    out, _ = m(x, use_cache=False)                                                                                 
print(out.shape)  # (1, 10, 50257)   