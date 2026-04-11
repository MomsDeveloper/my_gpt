import torch
from transformers import GPT2LMHeadModel
from gpt import GPT2


def load_hf_weights(my_model: GPT2, hf_name: str = "gpt2") -> GPT2:
    hf = GPT2LMHeadModel.from_pretrained(hf_name)
    hf_sd = hf.state_dict()
    my_sd = my_model.state_dict()

    transpose_keys = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    mapping = {
        "wte.emb_matrix.weight": "transformer.wte.weight",
        "wpe.emb_matrix.weight": "transformer.wpe.weight",
        "ln_f.weight": "transformer.ln_f.weight",
        "ln_f.bias": "transformer.ln_f.bias",
    }
    for i in range(my_model.num_layers):
        for suffix in [
            "ln_1.weight", "ln_1.bias",
            "ln_2.weight", "ln_2.bias",
            "attn.c_attn.weight", "attn.c_attn.bias",
            "attn.c_proj.weight", "attn.c_proj.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]:
            mapping[f"h.{i}.{suffix}"] = f"transformer.h.{i}.{suffix}"

    with torch.no_grad():
        for my_key, hf_key in mapping.items():
            assert my_key in my_sd, f"missing in my model: {my_key}"
            assert hf_key in hf_sd, f"missing in HF: {hf_key}"

            w = hf_sd[hf_key]
            if any(tk in my_key for tk in transpose_keys):
                w = w.t()
            assert my_sd[my_key].shape == w.shape, f"{my_key}: {my_sd[my_key].shape} vs {w.shape}"
            my_sd[my_key].copy_(w)

    my_model.load_state_dict(my_sd)
    return my_model

