#!/bin/bash
# Short test using the built-in Tiny Shakespeare dataset + 100-step d12 (124M) training.
# Intended for quick verification inside a Colab runtime.
#
# Usage (from llm.c root):
#   bash scripts/short_test.sh

set -e

OUT_DIR="models/Generic124M_test_100steps"

# ---------------------------------------------------------------------------
# 1. Prepare Tiny Shakespeare (fast, ~300k tokens)
# ---------------------------------------------------------------------------
echo ">>> Preparing Tiny Shakespeare dataset..."
python dev/data/tinyshakespeare.py --model_desc=gpt-2

# ---------------------------------------------------------------------------
# 2. Compile (Colab-friendly: FP16, no cuDNN)
# ---------------------------------------------------------------------------
echo ">>> Compiling train_gpt2cu (PRECISION=FP16)..."
make train_gpt2cu PRECISION=FP16

# ---------------------------------------------------------------------------
# 3. 100-step training run
# ---------------------------------------------------------------------------
echo ">>> Starting 100-step d12 training → $OUT_DIR"
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
    -u 20 \
    -x 100 \
    -v 20 \
    -s 50 \
    -g 128 \
    -c 0.1 \
    -z 1 \
    -r 0 \
    -y 1

echo ">>> Short test finished. Check $OUT_DIR for logs and checkpoints."
