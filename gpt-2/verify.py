import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gpt import GPT2
from load_hf_weights import load_hf_weights

tok = GPT2Tokenizer.from_pretrained("gpt2")
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()

my = GPT2(vocab_size=50257, max_seq_len=1024, emb_size=768,
        num_heads=12, head_size=64, num_layers=12, dropout=0.0)
load_hf_weights(my)
my.eval()

ids = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids
with torch.no_grad():
    hf_logits = hf(ids).logits
    my_logits, _ = my(ids, use_cache=False)

# 1. Абсолютная разница (в float32)
diff_abs = (my_logits - hf_logits).abs().max().item()

# 2. Относительная разница
diff_rel = ((my_logits - hf_logits).abs() / hf_logits.abs().clamp(min=1e-6)).max().item()

# 3. Top-1 prediction match (главный практический тест)
my_top1 = my_logits.argmax(-1)
hf_top1 = hf_logits.argmax(-1)
top1_match = (my_top1 == hf_top1).float().mean().item()

# 4. Top-5 prediction match
my_top5 = my_logits.topk(5, dim=-1).indices
hf_top5 = hf_logits.topk(5, dim=-1).indices
top5_match = (my_top5 == hf_top5).all(dim=-1).float().mean().item()

# 5. Cosine similarity softmax-распределений
my_probs = torch.softmax(my_logits, dim=-1)
hf_probs = torch.softmax(hf_logits, dim=-1)
cos = torch.nn.functional.cosine_similarity(my_probs.flatten(0, 1), hf_probs.flatten(0, 1), dim=-1).mean().item()

print(f"max abs diff:    {diff_abs:.2e}")
print(f"max rel diff:    {diff_rel:.2e}")
print(f"top-1 match:     {top1_match*100:.2f}%")
print(f"top-5 match:     {top5_match*100:.2f}%")
print(f"prob cosine sim: {cos:.6f}")

assert top1_match == 1.0, "top-1 predictions differ from HF"
assert cos > 0.9999, f"probability distributions differ: cos={cos}"
print("\nimplementation matches HF GPT-2")