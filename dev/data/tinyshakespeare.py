import argparse
import os

import tiktoken
from transformers import AutoTokenizer

from data_common import write_datafile

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")

def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url}...")
        import urllib.request
        urllib.request.urlretrieve(data_url, data_filename)

def tokenize(model_desc, max_tokens=None):
    download()
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()

    if model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>']
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
        eot = tokenizer.encode('')[0]

    sections = text.split("\n\n")
    tokens = []
    for s in sections:
        tokens.append(eot)
        tokens.extend(encode(s + "\n\n"))

    if max_tokens is None:
        max_tokens = len(tokens)

    val_target = int(max_tokens * 0.05)
    val_tokens = tokens[:val_target]
    train_tokens = tokens[val_target:max_tokens]

    write_datafile(os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin"), val_tokens, model_desc)
    write_datafile(os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin"), train_tokens, model_desc)

    print(f"Val: wrote {len(val_tokens):,} tokens")
    print(f"Train: wrote {len(train_tokens):,} tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"])
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    tokenize(args.model_desc, args.max_tokens)
