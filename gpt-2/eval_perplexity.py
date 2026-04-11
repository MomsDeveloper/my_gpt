import torch
from datasets import load_dataset                                                                                                                       
from transformers import GPT2Tokenizer, GPT2LMHeadModel                                                                             
from gpt import GPT2                                                                                                                                    
from load_hf_weights import load_hf_weights                                                                        
from tqdm import tqdm                   

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'                                                 
                                        
tok = GPT2Tokenizer.from_pretrained("gpt2")                                                                                                             
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")                                                                                        
text = "\n\n".join(ds["text"])
ids = tok(text, return_tensors="pt").input_ids[0]                                                                                                       
print(f"corpus: {ids.size(0)} tokens")                                                                             
                                                                                                                                                        
model = GPT2(vocab_size=50257, max_seq_len=1024, emb_size=768,                                                                                          
            num_heads=12, head_size=64, num_layers=12, dropout=0.0)                                                                                    
load_hf_weights(model)                                                                                                                                  
model.to(device).eval()                                                                                                                                 
                                                                                                                    
max_len = 1024                                                                                                                                          
stride = 512                                                                                                       
                                                                                                                                                        
nll_sum = 0.0                                                                                                      
n_tokens = 0
prev_end = 0                                                                                                                                            
                                        
for begin in tqdm(range(0, ids.size(0), stride), desc='eval'):                                                                                          
    end = min(begin + max_len, ids.size(0))                                                                                                             
    trg_len = end - prev_end  # сколько новых токенов считаем в loss
                                                                                                                                                        
    chunk = ids[begin:end].unsqueeze(0).to(device)                                                                 
    target = chunk.clone()                                                                                                                              
    target[:, :-trg_len] = -100  # не учитываем перекрывающиеся токены                                                                                  
                                            
    with torch.no_grad():                                                                                                                               
        logits, _ = model(chunk, use_cache=False)                                                                  
        shift_logits = logits[:, :-1, :].contiguous()                                                                                                   
        shift_labels = target[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(                                                                                                       
            shift_logits.view(-1, shift_logits.size(-1)),                                                          
            shift_labels.view(-1),                                                                                                                      
            ignore_index=-100,                                                                                                                          
            reduction='sum',                                                                                                                            
        )                                                                                                                                               
    nll_sum += loss.item()                                                                                                                              
    n_tokens += (target[:, 1:] != -100).sum().item()
    prev_end = end                                                                                                                                      
                                                                                                                    
    if end == ids.size(0):                  
        break                                                                                                                                           

ppl = torch.exp(torch.tensor(nll_sum / n_tokens)).item()                                                                                                

hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()                                                                                          
                                                                                                                                                        
nll_sum_hf = 0.0                                                                                                                                        
n_tokens_hf = 0                                                                                                                                         
prev_end = 0                                                                                                                                            
                                            
for begin in tqdm(range(0, ids.size(0), stride), desc='hf eval'):                                                                                       
    end = min(begin + max_len, ids.size(0))                                                                        
    trg_len = end - prev_end                                                                                                                            
                                        
    chunk = ids[begin:end].unsqueeze(0).to(device)                                                                                                      
    target = chunk.clone()                                                                                                                              
    target[:, :-trg_len] = -100
                                                                                                                                                        
    with torch.no_grad():                                                                                          
        logits = hf(chunk).logits
        shift_logits = logits[:, :-1, :].contiguous()                                                                                                   
        shift_labels = target[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(                                                                                                       
            shift_logits.view(-1, shift_logits.size(-1)),                                                          
            shift_labels.view(-1),                                                                                                                      
            ignore_index=-100,          
            reduction='sum',                                                                                                                            
        )                                                                                                                                               
    nll_sum_hf += loss.item()
    n_tokens_hf += (target[:, 1:] != -100).sum().item()                                                                                                 
    prev_end = end                                                                                                 
                                        
    if end == ids.size(0):
        break                                                                                                                                           

ppl_hf = torch.exp(torch.tensor(nll_sum_hf / n_tokens_hf)).item()                                                                                       
print(f"\nMy implementation: PPL = {ppl:.6f}")                                                                     
print(f"HF GPT2LMHeadModel: PPL = {ppl_hf:.6f}")
print(f"Difference: {abs(ppl - ppl_hf):.4f}")     