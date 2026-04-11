# my_gpt

A from-scratch PyTorch implementation of GPT-1 and GPT-2 decoder-only Transformers. Every component (Multi-Head Attention, BPE tokenizer, KV-cache, GELU, sampling) is implemented manually rather than using `nn.Transformer`.

The `gpt-2/` implementation is faithful to the GPT-2 124M architecture: its correctness is validated by loading OpenAI's pretrained weights into it and reproducing the reference logits / perplexity of HuggingFace's `GPT2LMHeadModel`.

## Repository layout

```
my_gpt/
├── gpt-1/                # GPT-1 style (post-norm, ReLU, no KV-cache)
│   ├── attention.py      # Per-head MultiHeadAttention (naive)
│   ├── decoder.py        # Transformer decoder block
│   ├── ffn.py            # Position-wise feed-forward
│   ├── embedings.py      # Token + positional embeddings
│   ├── bpe.py            # Byte-Pair Encoding tokenizer
│   ├── dataset.py        # Sliding-window dataset
│   ├── gpt.py            # GPT model + train loop + generation
│   ├── pre_training.py   # Training on Russian corpus
│   └── evaluate.py       # Generation / evaluation
└── gpt-2/                # GPT-2 124M faithful reimplementation
    ├── attention.py      # Fused-QKV MultiHeadAttention + KV-cache
    ├── decoder.py        # Pre-norm decoder block
    ├── ffn.py            # c_fc / c_proj with manual GELU
    ├── gelu.py           # GELU tanh approximation (manual)
    ├── embedings.py      # Token + learned positional embeddings
    ├── bpe.py            # BPE tokenizer (from scratch)
    ├── dataset.py        # Sliding-window dataset
    ├── gpt.py            # GPT2 model with weight-tied lm_head
    ├── load_hf_weights.py  # Load OpenAI GPT-2 weights into this model
    ├── verify.py         # Bit-level equivalence check against HuggingFace
    ├── eval_perplexity.py  # WikiText-2 perplexity, compared to HF reference
    ├── pre_training.py   # Training entry point
    └── evaluate.py       # Generation / evaluation
```

## Architecture (`gpt-2/`)

Matches GPT-2 124M exactly:

| Hyperparameter      | Value  |
|---------------------|--------|
| Layers              | 12     |
| Attention heads     | 12     |
| Embedding dim       | 768    |
| Head dim            | 64     |
| FFN inner dim       | 3072   |
| Max sequence length | 1024   |
| Vocab size          | 50257  |
| Parameters          | 124M   |

### Components implemented from scratch

- **BPE tokenizer** (`bpe.py`) — character-level init, iterative greedy pair merging, greedy longest-token encoding. Used in `gpt-1/`; for `gpt-2/` HF validation, the pipeline switches to the matching `tiktoken`-style GPT-2 BPE so that vocab aligns with OpenAI weights.
- **Token + learned positional embeddings** (`embedings.py`) — `PositionalEmbeddings.forward(seq_len, start_pos)` supports the KV-cache decoding path via `start_pos` offset.
- **Multi-Head Attention** (`attention.py`) — fused QKV in a single `c_attn` linear, split into `(B, num_heads, T, head_dim)` via `view`+`transpose`, scaled dot-product attention with a registered causal mask, output projected by `c_proj`. Layout matches HuggingFace's `GPT2Attention` so weights map one-to-one.
- **GELU** (`gelu.py`) — tanh approximation `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` computed manually, bit-exact to HF's `NewGELUActivation`.
- **Pre-norm decoder block** (`decoder.py`) — `x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))`.
- **Weight tying** (`gpt.py`) — `lm_head.weight` shares storage with the token embedding (`self.linear.weight = self.token_emb.emb_matrix.weight`), matching GPT-2 and halving output-layer parameters.
- **Sampling** (`gpt.py::generate`) — greedy, temperature, top-k and top-p (nucleus) sampling.

### KV-cache

`gpt-2/attention.py` implements a per-layer KV-cache. During generation:

1. **Prefill** — the first call processes the full prompt and produces `cache: List[Tuple[K, V]]` of length `num_layers`, with `K`/`V` of shape `(B, num_heads, prompt_len, head_dim)`.
2. **Decode** — subsequent calls feed only the last token. `K`/`V` for the new token are concatenated to the cached ones along the sequence dimension. The attention matrix shrinks from `(L × L)` to `(1 × L)` per step. Position embeddings are offset by `start_pos = cache[-1][0].size(2)`.

This turns the per-token decode cost from `O(L²)` to `O(L)`.

## Validation against OpenAI GPT-2

`gpt-2/load_hf_weights.py` loads the official OpenAI GPT-2 124M weights from `transformers.GPT2LMHeadModel` into this implementation. HuggingFace stores the linear layers as `Conv1D` (transposed `nn.Linear`), so the loader transposes `c_attn`, `c_proj`, `c_fc`, and `mlp.c_proj` when copying.

### `verify.py` — bit-level equivalence

Runs the same input through both models and compares their logits:

| Metric                          | Value      |
|---------------------------------|------------|
| `max abs diff`                  | `1.22e-4`  |
| `max rel diff`                  | `1.08e-6`  |
| Top-1 prediction match          | **100.00%** |
| Top-5 prediction match          | **100.00%** |
| Softmax cosine similarity       | `1.000000` |

The absolute difference is pure float32 accumulation noise through 12 blocks; the relative difference (`~1e-6`) is at the level of float32 machine epsilon. Top-1 and top-5 match 100% — for every token position, this implementation and HuggingFace select the same next token.

### `eval_perplexity.py` — WikiText-2 test

Sliding-window perplexity with `max_len=1024`, `stride=512`. The script runs the same eval loop on both this implementation and HuggingFace's `GPT2LMHeadModel` with identical weights.

| Model                        | PPL (WikiText-2 test) |
|------------------------------|-----------------------|
| This implementation          | **25.17**             |
| HuggingFace `GPT2LMHeadModel` | **25.17**             |

Identical pipeline, identical weights → identical perplexity.

## `gpt-1/`

A smaller, post-norm variant (GPT-1 style with ReLU, no KV-cache) trained from scratch on a small Russian literary corpus (~913 files, ~4 MB, Pushkin etc.) with a 2000-token BPE vocabulary learned from the same text. Configuration: 12L / 8H / 512d / seq_len 64, Adam lr `1e-5`, batch 128, 48 epochs on MPS.

```bash
cd gpt-1
python pre_training.py
```

This trains the model and saves `data/gpt.dill` + `data/loss_plot.png`. Artifacts are not committed; `.dill` files are tracked via Git LFS.

## Reproducing the `gpt-2/` validation

```bash
cd gpt-2
python verify.py          # logit-level equivalence with HF
python eval_perplexity.py # WikiText-2 PPL, both models
```

Both scripts download OpenAI GPT-2 weights via `transformers` on first run.
