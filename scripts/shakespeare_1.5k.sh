#!/bin/bash
# 1500-step d12 (124M) training on Tiny Shakespeare with periodic checkpoints.
# Intended for A100 (or any higher-VRAM GPU). Much faster than the 100-step smoke test.
#
# Usage (from llm.c root):
#   bash scripts/shakespeare_1.5k.sh

set -e

OUT_DIR="models/Generic124M_shakespeare_1.5k"

# ---------------------------------------------------------------------------
# 1. Prepare Tiny Shakespeare (fast, ~300k tokens)
# ---------------------------------------------------------------------------
echo ">>> Preparing Tiny Shakespeare dataset..."
python dev/data/tinyshakespeare.py --model_desc=gpt-2

# ---------------------------------------------------------------------------
# 2. Compile (FP16, no cuDNN – works on Colab A100 / T4 / etc.)
# ---------------------------------------------------------------------------
echo ">>> Compiling train_gpt2cu (PRECISION=FP16)..."
make train_gpt2cu PRECISION=FP16

# ---------------------------------------------------------------------------
# 3. 1500-step training run with checkpoints
# ---------------------------------------------------------------------------
echo ">>> Starting 1500-step d12 training → $OUT_DIR"
mkdir -p "$OUT_DIR"

./train_gpt2cu \
    -i "dev/data/tinyshakespeare/tiny_shakespeare_train.bin" \
    -j "dev/data/tinyshakespeare/tiny_shakespeare_val.bin" \
    -o "$OUT_DIR" \
    -e "d12" \
    -b 8 \
    -t 1024 \
    -d 65536 \
    -l 0.0003 \
    -u 50 \
    -x 1500 \
    -v 50 \
    -s 100 \
    -g 128 \
    -c 0.1 \
    -n 100 \
    -nk 5 \
    -z 1 \
    -r 0 \
    -y 0

echo ">>> 1500-step run finished. Check $OUT_DIR for logs and checkpoints."
