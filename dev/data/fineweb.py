"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

Example of downloading the 100B dataset of FineWebEDU, from root directory:
python dev/data/fineweb.py -t edu -v 100B
"""

import argparse
import os
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data_common import write_datafile

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.add_argument("-t", "--type", type=str, default="classic", choices=["classic", "edu"])
parser.add_argument("-v", "--version", type=str, default="10B", choices=["10B", "100B"])
parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"])
parser.add_argument("--max-tokens", type=int, default=None, help="Total tokens for train+val (95% train, 5% val). None = full dataset")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in {"10B", "100B"}, "version must be one of: 10B, 100B"
assert args.type in {"edu", "classic"}, "type must be one of: edu, classic"

directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(args.type, args.version)]

# create the cache directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
if args.type == "classic":
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    name = "fineweb"
else:
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    name = "edu_fineweb"

# ----------------------------------------------------------------------------
# Tokenization functions
def tokenize_gpt2(doc):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>']
    tokens = [eot]
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint

def tokenize_llama(doc):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
    eot = tokenizer.encode('')[0]
    tokens = [eot]
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    tokens_np_uint = tokens_np.astype(np.uint32)
    return tokens_np_uint

tokenize = tokenize_gpt2 if args.model_desc == "gpt-2" else tokenize_llama
token_dtype = np.uint16 if args.model_desc == "gpt-2" else np.uint32

# ----------------------------------------------------------------------------
# Apply 95/5 split with max-tokens limit
if args.max_tokens is None:
    args.max_tokens = 1_000_000_000_000

val_target = int(args.max_tokens * 0.05)
train_target = args.max_tokens - val_target

print(f"Target split: {train_target:,} train + {val_target:,} val = {args.max_tokens:,} total")

val_tokens = []
train_tokens = []
total_val = 0
total_train = 0

nprocs = max(1, os.cpu_count() - 2)
with mp.Pool(nprocs) as pool:
    for tokens in tqdm(pool.imap(tokenize, fw, chunksize=16), desc="Tokenizing"):
        if total_val < val_target:
            needed = val_target - total_val
            val_tokens.extend(tokens[:needed])
            total_val += len(tokens[:needed])
            tokens = tokens[needed:]

        if total_train < train_target and tokens:
            needed = train_target - total_train
            train_tokens.extend(tokens[:needed])
            total_train += len(tokens[:needed])

        if total_val >= val_target and total_train >= train_target:
            break

# Write files
write_datafile(os.path.join(DATA_CACHE_DIR, f"{name}_val_000000.bin"), val_tokens, args.model_desc)
write_datafile(os.path.join(DATA_CACHE_DIR, f"{name}_train_000000.bin"), train_tokens, args.model_desc)

print(f"Val: wrote {len(val_tokens):,} tokens")
print(f"Train: wrote {len(train_tokens):,} tokens")
