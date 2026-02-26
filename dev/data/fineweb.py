import argparse
import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import tiktoken
from transformers import AutoTokenizer
from data_common import write_datafile

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, default="classic", choices=["classic", "edu"])
parser.add_argument("-v", "--version", type=str, default="10B", choices=["10B", "100B"])
parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"])
parser.add_argument("--max-tokens", type=int, default=None)
args = parser.parse_args()

local_dir = f"{'edu_' if args.type=='edu' else ''}fineweb{args.version}"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset(f"HuggingFaceFW/fineweb{'-edu' if args.type=='edu' else ''}", name=f"sample-{args.version}T", split="train")

def tokenize(doc):
    if args.model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens['<|endoftext|>']
        tokens = [eot] + enc.encode_ordinary(doc["text"])
        return np.array(tokens, dtype=np.uint16)
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        eot = tokenizer.encode('')[0]
        tokens = [eot] + tokenizer.encode(doc["text"], add_special_tokens=False, verbose=False, split_special_tokens=True)
        return np.array(tokens, dtype=np.uint32)

nprocs = max(1, os.cpu_count() - 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((10**8,), dtype=np.uint16 if args.model_desc=="gpt-2" else np.uint32)
    token_count = 0
    total_tokens = 0

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < 10**8:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            total_tokens += len(tokens)
        else:
            split = "val" if shard_index == 0 else "train"
            write_datafile(os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin"), all_tokens_np[:token_count].tolist(), args.model_desc)
            shard_index += 1
            token_count = len(tokens)
            all_tokens_np[:token_count] = tokens
            total_tokens += len(tokens)

        if args.max_tokens is not None and total_tokens >= args.max_tokens:
            break

    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        write_datafile(os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin"), all_tokens_np[:token_count].tolist(), args.model_desc)
