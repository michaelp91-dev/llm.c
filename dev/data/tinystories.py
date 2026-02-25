"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is using either GPT-2 or LLaMA 3 tokenizer.

The output is written to a newly created tinystories/ folder.
The script prints:

For GPT-2:
Number of shards: 50
Tokenizing val split...
writing 19,043,638 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 925,653,391 tokens to tinystories/TinyStories_train.bin

For LLaMA 3:
Number of shards: 50
Tokenizing val split...
writing 18,660,516 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 907,021,844 tokens to tinystories/TinyStories_train.bin

And runs in few minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.
"""

import argparse
import os
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import tiktoken
from transformers import AutoTokenizer

from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_index, shard_filename, model_desc):
    if model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>'] # end of text token
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    else:
        raise ValueError(f"unknown model descriptor {model_desc}")

    with open(shard_filename, "r") as f:
        data = json.load(f)
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize(model_desc, max_tokens=None):
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if max_tokens is None:
        max_tokens = 1_000_000_000_000   # full dataset

    val_target = int(max_tokens * 0.05)
    train_target = max_tokens - val_target

    print(f"Target split: {train_target:,} train + {val_target:,} val = {max_tokens:,} total")

    # VAL split (5%) - take tokens sequentially until target
    print("Tokenizing val split...")
    val_tokens = []
    total_val = 0
    shard_idx = 0
    while total_val < val_target and shard_idx < len(shard_filenames):
        tokens = process_shard(shard_idx, shard_filenames[shard_idx], model_desc)
        needed = val_target - total_val
        val_tokens.extend(tokens[:needed])
        total_val += len(tokens[:needed])
        shard_idx += 1
    write_datafile(os.path.join(DATA_CACHE_DIR, "TinyStories_val.bin"), val_tokens, model_desc)
    print(f"Val: wrote {len(val_tokens):,} tokens")

    # TRAIN split (95%) - continue from where val stopped
    print("Tokenizing train split...")
    train_tokens = []
    total_train = 0
    for s in range(shard_idx, len(shard_filenames)):
        tokens = process_shard(s, shard_filenames[s], model_desc)
        needed = train_target - total_train
        train_tokens.extend(tokens[:needed])
        total_train += len(tokens[:needed])
        if total_train >= train_target:
            break
    write_datafile(os.path.join(DATA_CACHE_DIR, "TinyStories_train.bin"), train_tokens, model_desc)
    print(f"Train: wrote {len(train_tokens):,} tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Stories dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    parser.add_argument("--max-tokens", type=int, default=None,
                    help="Total tokens for train+val (95 percent train, 5 percent val). None = full dataset")
    args = parser.parse_args()
    download()
    tokenize(args.model_desc, args.max_tokens)
