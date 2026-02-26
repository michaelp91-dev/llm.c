"""
FineWeb / FineWeb-Edu - crash-proof version for Colab
Only downloads exactly the tokens you need (95/5 split)
"""

import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from transformers import AutoTokenizer

from data_common import write_datafile

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, default="classic", choices=["classic", "edu"])
parser.add_argument("-v", "--version", type=str, default="10B", choices=["10B", "100B"])
parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"])
parser.add_argument("--max-tokens", type=int, default=None)
args = parser.parse_args()

local_dir, remote_name = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}[(args.type, args.version)]

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

if args.max_tokens is None:
    args.max_tokens = 1_000_000_000_000

val_target = int(args.max_tokens * 0.05)
train_target = args.max_tokens - val_target

print(f"Target split: {train_target:,} train + {val_target:,} val = {args.max_tokens:,} total")

fw = load_dataset(
    "HuggingFaceFW/fineweb" if args.type == "classic" else "HuggingFaceFW/fineweb-edu",
    name=remote_name, 
    split="train", 
    streaming=True
)

def tokenize(doc):
    if args.model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        tokens = [enc._special_tokens['<|endoftext|>']] + enc.encode_ordinary(doc["text"])
        return np.array(tokens, dtype=np.uint16)
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        tokens = [tokenizer.encode('')[0]] + tokenizer.encode(doc["text"], add_special_tokens=False, verbose=False, split_special_tokens=True)
        return np.array(tokens, dtype=np.uint32)

val_tokens = []
train_tokens = []
total_val = 0
total_train = 0

print("Tokenizing (streaming - only downloading what we need)...")
for example in fw:
    tokens = tokenize(example)

    if total_val < val_target:
        needed = val_target - total_val
        val_tokens.extend(tokens[:needed])
        total_val += len(tokens[:needed])
        tokens = tokens[needed:]

    if total_train < train_target and len(tokens) > 0:
        needed = train_target - total_train
        train_tokens.extend(tokens[:needed])
        total_train += len(tokens[:needed])

    if total_val >= val_target and total_train >= train_target:
        break

write_datafile(os.path.join(DATA_CACHE_DIR, f"{local_dir}_val_000000.bin"), val_tokens, args.model_desc)
write_datafile(os.path.join(DATA_CACHE_DIR, f"{local_dir}_train_000000.bin"), train_tokens, args.model_desc)

print(f"Val: wrote {len(val_tokens):,} tokens")
print(f"Train: wrote {len(train_tokens):,} tokens")
