# llm.c — Generic124M fork

Fork of [karpathy/llm.c](https://github.com/karpathy/llm.c) focused on training a strong **124M-parameter** GPT-style model (**Generic124M**) with modern architecture upgrades and a high-quality open education data mix.

Upstream llm.c trains GPT-2/GPT-3-style models in pure C/CUDA with no PyTorch dependency. This fork keeps that core and layers on:

| Feature | Status |
|---------|--------|
| **RoPE** (replaces absolute position embeddings) | Done |
| **FP16** training with dynamic loss scaling | Done |
| Education-heavy open data mix | In progress |
| **SwiGLU** FFN (replaces GeLU) | Planned |
| Chinchilla-optimal ~2.5B-token run | Planned |

**Goal:** a fully open 124M model that is competitive with (and where possible stronger than) baselines in the 100–150M class, trained only on permissively licensed data.

---

## Quick start (Colab / single GPU)

### 1. Smoke test on Tiny Shakespeare

Verifies compile + RoPE + FP16 path in ~1 minute on an A100:

```bash
git clone https://github.com/michaelp91-dev/llm.c.git
cd llm.c
bash scripts/short_test.sh
```

### 2. Longer Shakespeare check (optional)

```bash
bash scripts/shakespeare_1.5k.sh
```

### 3. Manual training example

```bash
make train_gpt2cu PRECISION=FP16

./train_gpt2cu \
  -e d12 \
  -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
  -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
  -o models/my_run \
  -b 8 -t 1024 -d 65536 \
  -l 1e-4 -u 50 -x 150 \
  -v 25 -s 50
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `-e d12` | 12-layer GPT-2 124M |
| `-b` | micro-batch size |
| `-t` | sequence length (≤ 1024) |
| `-d` | total batch size in tokens |
| `-l` | peak learning rate |
| `-u` | warmup steps |
| `-x` | max steps |
| `-o` | output / log directory |
| `-n` | checkpoint every N steps |
| `-y 1` | resume from latest checkpoint in `-o` |

For RoPE + FP16, prefer a slightly lower LR and longer warmup than the absolute-PE defaults (e.g. `1e-4` / 50 steps) until the run is stable.

---

## Dataset: Generic124M mix

All sources are **open-license** (no ClimbMix / NC data):

| Source | Role | Approx mix |
|--------|------|------------|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT` | Educational web | ~55% |
| [Cosmopedia-v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Synthetic textbooks | ~25% |
| [Python-Edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Educational Python | ~15% |
| OpenHermes (optional, late stage) | Instruction-style | ~5% |

Prep script (same style as other `dev/data` loaders):

```bash
# FineWeb-Edu, keep only score >= 4, up to 2B tokens → Drive
python dev/data/generic124m_mix.py --source fineweb-edu --output-dir /content/drive/MyDrive/llm_c_data/generic124m --min-score 4 --max-tokens 2000000000

# Cosmopedia slice
python dev/data/generic124m_mix.py --source cosmopedia --output-dir /content/drive/MyDrive/llm_c_data/generic124m --max-tokens 800000000
```

- Streams from Hugging Face (works well with Colab + mounted Drive).
- First shard of each source is `*_val_*.bin`; the rest are train.
- Point `train_gpt2cu` at the resulting `.bin` shards with `-i` / `-j`.

Also available: the upstream `dev/data/fineweb.py`, `tinyshakespeare.py`, `tinystories.py`, etc.

---

## Architecture notes

### RoPE
- Absolute `wpe` is unused in the forward/backward path; token embeddings only.
- Cos/sin tables are precomputed once on CPU and kept on device (`llmc/rope.cuh`).
- Applied inside the attention kernels after the Q/K projection.

### FP16
- Compile with `make train_gpt2cu PRECISION=FP16`.
- Dynamic loss scaling starts at `2^16` and halves on overflow / grows after sustained success.
- Master weights in FP32 for the AdamW update (default).

### Still upstream-compatible
- Same CLI, dataloader `.bin` format, and checkpoint layout as karpathy/llm.c.
- CPU reference (`train_gpt2.c`) and PyTorch reference (`train_gpt2.py`) remain for debugging.

---

## Repo layout

```
train_gpt2.cu          # main CUDA training loop (RoPE + FP16)
llmc/                  # kernels + utilities (rope.cuh, attention.cuh, …)
dev/data/              # dataset prep scripts
  generic124m_mix.py   # FineWeb-Edu / Cosmopedia / Python-Edu → .bin
  tinyshakespeare.py
  fineweb.py
  …
scripts/
  short_test.sh        # 100-step d12 smoke test
  shakespeare_1.5k.sh  # longer Shakespeare run + checkpoints
```

---

## Roadmap

1. **Stabilize RoPE + FP16** on real data (done on Tiny Shakespeare).
2. **Build the education mix** on Drive (`generic124m_mix.py`).
3. **Chinchilla-scale run** (~2.5B tokens) on A100-class GPUs.
4. **SwiGLU** FFN swap (next architecture change).
5. Eval vs GPT-2 124M and small open models (HellaSwag, ARC, etc.).

---

## Upstream

This is a derivative of [karpathy/llm.c](https://github.com/karpathy/llm.c). See that repo for the original design notes, multi-node scripts, and the large list of language ports.

## License

MIT (same as upstream).
