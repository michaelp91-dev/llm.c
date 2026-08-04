#!/bin/bash
# Short test: create a tiny mixed dataset (~2.5M tokens) + 100-step d12 (124M) training run
# Intended for quick verification inside a Colab runtime.
# Usage (from llm.c root):
#   bash scripts/short_test.sh
# or:
#   chmod +x scripts/short_test.sh && ./scripts/short_test.sh

set -e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR="models/Generic124M_test_100steps"
TRAIN_BIN="Generic124M_test_train_0.bin"
VAL_BIN="Generic124M_test_val.bin"
TARGET_TOKENS=2500000
VAL_TOKENS=32768

# ---------------------------------------------------------------------------
# 1. Create tiny dataset if it does not already exist
# ---------------------------------------------------------------------------
if [[ -f "$TRAIN_BIN" && -f "$VAL_BIN" ]]; then
    echo ">>> Tiny dataset already present ($TRAIN_BIN, $VAL_BIN) – skipping creation"
else
    echo ">>> Creating tiny mixed dataset (~${TARGET_TOKENS} tokens)..."
    # Note: we use os._exit(0) at the end to avoid a known HF datasets
    # streaming cleanup crash (PyGILState_Release) that happens after
    # the files have already been written successfully.
    python3 - << 'PY'
import struct, os, numpy as np
from datasets import load_dataset, interleave_datasets
import tiktoken
from tqdm.auto import tqdm
from datetime import datetime

tokenizer = tiktoken.get_encoding("gpt2")

TARGET_TOKENS = 2_500_000
VAL_TOKENS    = 32_768

print("Loading datasets (streaming)...")
ds_edu     = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_cosmo   = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
ds_python  = load_dataset("ajibawa-2023/Python-Code-Large", split="train", streaming=True)
ds_instruct= load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

mixed = interleave_datasets(
    [ds_edu, ds_cosmo, ds_python, ds_instruct],
    probabilities=[0.55, 0.25, 0.15, 0.05],
    seed=42
)

print("Tokenizing...")
tokens = []
pbar = tqdm(total=TARGET_TOKENS, unit="tok", unit_scale=True, desc="Tokenizing",
            mininterval=1.0, dynamic_ncols=True)

for example in mixed:
    text = (
        example.get("text")
        or example.get("content")
        or example.get("code")
        or ""
    )
    if not text and "conversations" in example:
        conv = example["conversations"]
        if isinstance(conv, list):
            text = "\n".join(f"{m.get('from','')}: {m.get('value','')}" for m in conv)

    if not text or len(text) < 50:
        continue

    ids = tokenizer.encode(text, disallowed_special=())
    tokens.extend(ids)
    pbar.update(len(ids))
    if len(tokens) >= TARGET_TOKENS:
        break

pbar.close()
tokens = np.array(tokens[:TARGET_TOKENS], dtype=np.uint16)
print(f"Collected {len(tokens):,} tokens")

val_tokens   = tokens[:VAL_TOKENS]
train_tokens = tokens[VAL_TOKENS:]

def write_bin(path, arr):
    with open(path, "wb") as f:
        header = [0] * 256
        header[0] = 20240520
        header[1] = 1
        header[2] = len(arr)
        f.write(struct.pack("<256I", *header))
        f.write(arr.tobytes())

write_bin("Generic124M_test_val.bin", val_tokens)
write_bin("Generic124M_test_train_0.bin", train_tokens)

with open("Generic124M_test_dataset_info.txt", "w") as f:
    f.write(f"Created: {datetime.now().isoformat()}\n")
    f.write(f"Total tokens: {len(tokens):,}\n")
    f.write(f"Val tokens:   {len(val_tokens):,}\n")
    f.write(f"Train tokens: {len(train_tokens):,}\n")
    f.write("Mix: FineWeb-Edu 55% | Cosmopedia-v2 25% | Python-Code-Large 15% | OpenHermes-2.5 5%\n")

print("Wrote Generic124M_test_val.bin and Generic124M_test_train_0.bin")

# Force a clean exit so HF datasets streaming threads don't crash the process
os._exit(0)
PY
fi

# ---------------------------------------------------------------------------
# 2. Compile (idempotent)
# ---------------------------------------------------------------------------
echo ">>> Compiling train_gpt2cu (USE_CUDNN=1)..."
make train_gpt2cu USE_CUDNN=1

# ---------------------------------------------------------------------------
# 3. 100-step training run
# ---------------------------------------------------------------------------
echo ">>> Starting 100-step d12 training → $OUT_DIR"
mkdir -p "$OUT_DIR"

./train_gpt2cu \
    -i "Generic124M_test_train_*.bin" \
    -j "Generic124M_test_val.bin" \
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
