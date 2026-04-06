# my_gpt

A from-scratch PyTorch implementation of the GPT-1 and GPT-2 decoder-only Transformer architectures, trained on a small Russian-language corpus. Every component (Multi-Head Attention, BPE tokenizer, KV-cache, sampling) is implemented manually rather than using `nn.Transformer`.

## Repository layout

```
my_gpt/
├── gpt-1/                # GPT-1 style (post-norm, ReLU, no KV-cache)
│   ├── attention.py      # HeadAttention + MultiHeadAttention
│   ├── decoder.py        # Transformer decoder block
│   ├── ffn.py            # Position-wise feed-forward
│   ├── embedings.py      # Token + positional embeddings
│   ├── bpe.py            # Byte-Pair Encoding tokenizer
│   ├── dataset.py        # Sliding-window dataset
│   ├── gpt.py            # GPT model + train loop + generation
│   ├── pre_training.py   # Training entry point
│   └── evaluate.py       # Generation / evaluation
└── gpt-2/                # GPT-2 style (pre-norm, GELU, KV-cache)
    └── ... (same layout, plus gelu.py)
```

## Architecture

Both models share the same hyperparameters in this repo:

| Hyperparameter      | Value |
|---------------------|-------|
| Layers              | 12    |
| Attention heads     | 8     |
| Embedding dim       | 512   |
| Head dim            | 64    |
| FFN inner dim       | 2048  |
| Max sequence length | 512   |
| Vocab size (BPE)    | 2000  |
| Dropout             | 0.2   |

### Components implemented from scratch

- **BPE tokenizer** (`bpe.py`) — character-level init, iterative greedy pair merging until target vocab is reached. Greedy longest-token encoding.
- **Token + learned positional embeddings** (`embedings.py`).
- **Multi-Head Attention** (`attention.py`) — scaled dot-product attention with a registered causal mask buffer; each head has its own `W_q / W_k / W_v` projections, outputs are concatenated and projected by `W_o`.
- **Position-wise FFN** (`ffn.py`) — `Linear → activation → Linear`.
- **Decoder block** (`decoder.py`) — residual connections + LayerNorm.
- **GELU** (`gpt-2/gelu.py`) — implemented manually for the GPT-2 variant.
- **Sampling** (`gpt.py::generate`) — supports greedy, temperature, top-k and top-p (nucleus) sampling.

### GPT-1 vs GPT-2 differences in this repo

| Aspect              | GPT-1 (`gpt-1/`)            | GPT-2 (`gpt-2/`)               |
|---------------------|-----------------------------|--------------------------------|
| LayerNorm placement | Post-norm (after sublayer)  | Pre-norm (before sublayer)     |
| Activation          | ReLU                        | GELU                           |
| Final LayerNorm     | —                           | Yes (before output projection) |
| KV-cache            | —                           | Yes                            |
| `forward` returns   | logits                      | `(logits, cache)`              |

### KV-cache (GPT-2)

`gpt-2/attention.py` implements a per-head KV-cache. During generation:

1. **Prefill** — first call processes the full prompt and produces `cache: List[List[Tuple[K, V]]]` of shape `[num_layers][num_heads]`, each `K`/`V` of shape `(B, prompt_len, head_dim)`.
2. **Decode** — subsequent calls feed only the last token. Each head concatenates the new `K`/`V` to the cached ones, so the attention matrix shrinks from `(L × L)` to `(1 × L)` per step. Position embeddings are offset by `start_pos = cache[-1][0][0].size(1)`.

## Training

| Setting         | Value             |
|-----------------|-------------------|
| Optimizer       | Adam              |
| Learning rate   | 1e-5              |
| Batch size      | 128               |
| Sequence length | 64                |
| Epochs          | 48                |
| Train/val split | 90 / 10           |
| Loss            | Cross-entropy     |
| Device          | MPS (Apple GPU)   |

**Data.** ~913 Russian text files (~4 MB total) of classical literature, concatenated with `\n\n\n` separators, BPE-encoded to a 2000-token vocabulary, then split 90/10. Sliding windows of length 64 are produced by `dataset.py`.

The `texts/` corpus and trained `data/*.dill` artifacts are not committed (`.dill` files were previously tracked via Git LFS).

### Running training

```bash
cd gpt-2
python pre_training.py
```

This will:
1. Read `./texts/*`
2. Fit (or load) a BPE tokenizer at `./data/bpe.dill`
3. Train for 48 epochs and save the model to `./data/gpt.dill`
4. Save the loss curve to `./data/loss_plot.png`

### Generation

```bash
cd gpt-2
python evaluate.py
```
