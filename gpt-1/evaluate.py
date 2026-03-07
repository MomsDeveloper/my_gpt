from gpt import GPT
from bpe import BPE
import torch

def evaluate(model_path, bpe_path, prompt, device):
    model = GPT.load(model_path, device)
    bpe = BPE.load(bpe_path)

    model.eval()
    with torch.no_grad():
        token_ids = bpe.encode(prompt)
        x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_text = model.generate(x, max_new_tokens=150, do_sample=True, temperature=0.2, top_k=None, top_p=0.3)
        generated_text = generated_text.squeeze(0).tolist()
        text = bpe.decode(generated_text)
        return text
            
prompt = 'Привет, дядя'
model_path = './data/gpt.dill'
bpe_path = './data/bpe.dill'

print(evaluate(model_path, bpe_path, prompt, 'cpu'))