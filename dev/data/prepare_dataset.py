import argparse
import subprocess
import os
import struct
from datasets import load_dataset
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument("--max-tokens", type=int, default=10_000_000_000)
parser.add_argument("--mix", type=str, required=True, help="e.g. fineweb:0.6,tinystories:0.25,conversational:0.1,code:0.05")
args = parser.parse_args()

DATA_DIR = "dev/data"
MIX_DIR = f"{DATA_DIR}/mixed"
os.makedirs(MIX_DIR, exist_ok=True)

parts = [p.strip() for p in args.mix.split(",")]
sources = {}
for p in parts:
    name, ratio = p.split(":")
    sources[name] = float(ratio)

total = sum(sources.values())
for k in sources:
    sources[k] /= total

print("Preparing mixture for 135M NanoGrok:", sources)
print("Total tokens:", args.max_tokens)

all_train = bytearray()
all_val = bytearray()

enc = tiktoken.get_encoding("gpt2")

for name, ratio in sources.items():
    tokens_needed = int(args.max_tokens * ratio)
    print(f"→ {name}: {tokens_needed:,} tokens")

    if name == "tinystories":
        cmd = ["python", f"{DATA_DIR}/tinystories.py", "--max-tokens", str(tokens_needed)]
        train_path = f"{DATA_DIR}/tinystories/TinyStories_train.bin"
        val_path = f"{DATA_DIR}/tinystories/TinyStories_val.bin"
    elif name == "fineweb":
        cmd = ["python", f"{DATA_DIR}/fineweb.py", "--max-tokens", str(tokens_needed)]
        train_path = f"{DATA_DIR}/fineweb10B/fineweb10B_train_000000.bin"
        val_path = f"{DATA_DIR}/fineweb10B/fineweb10B_val_000000.bin"
    elif name == "conversational":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        count = 0
        for example in ds:
            text = ""
            for msg in example["messages"]:
                text += msg["content"] + "\n"
            tokens = enc.encode(text)
            for t in tokens:
                all_train.extend(struct.pack("H", t))
            count += len(tokens)
            if count >= tokens_needed:
                break
        continue
    elif name == "code":
        ds = load_dataset("codeparrot/codeparrot", split="train", streaming=True)
        count = 0
        for example in ds:
            text = example["content"]
            tokens = enc.encode(text)
            for t in tokens:
                all_train.extend(struct.pack("H", t))
            count += len(tokens)
            if count >= tokens_needed:
                break
        continue

    subprocess.run(cmd, check=True)

    with open(train_path, "rb") as f:
        f.read(8)
        all_train.extend(f.read())
    with open(val_path, "rb") as f:
        f.read(8)
        all_val.extend(f.read())

def write_bin(path, data):
    num_tokens = len(data) // 2
    header = [0] * 256
    header[0] = 20240520
    header[1] = 1
    header[2] = num_tokens
    with open(path, "wb") as f:
        f.write(struct.pack("<256I", *header))
        f.write(data)

write_bin(f"{MIX_DIR}/train.bin", all_train)
write_bin(f"{MIX_DIR}/val.bin", all_val)

print("✅ Mixed dataset ready!")
print("Train tokens:", len(all_train)//2)
print("Val tokens:", len(all_val)//2)
