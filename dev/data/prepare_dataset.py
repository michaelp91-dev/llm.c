import argparse
import subprocess
import os
import shutil
import struct

parser = argparse.ArgumentParser(description="Prepare single or mixed dataset for llm.c")
parser.add_argument("--max-tokens", type=int, default=None)
parser.add_argument("--source", type=str, choices=["tinystories", "tinyshakespeare", "fineweb"])
parser.add_argument("--mix", type=str, help="Mixture like 'fineweb:0.5,tinyshakespeare:0.3,tinystories:0.2'")
args = parser.parse_args()

if args.max_tokens is None:
    args.max_tokens = 1_000_000_000_000

DATA_DIR = "dev/data"
MIX_DIR = os.path.join(DATA_DIR, "mixed")
os.makedirs(MIX_DIR, exist_ok=True)

if args.source:
    # Single source mode
    print(f"Preparing single source: {args.source} ({args.max_tokens:,} tokens)")
    if args.source == "tinystories":
        cmd = ["python", f"{DATA_DIR}/tinystories.py", "--max-tokens", str(args.max_tokens)]
        train_path = f"{DATA_DIR}/tinystories/TinyStories_train.bin"
        val_path = f"{DATA_DIR}/tinystories/TinyStories_val.bin"
    elif args.source == "tinyshakespeare":
        cmd = ["python", f"{DATA_DIR}/tinyshakespeare.py", "--max-tokens", str(args.max_tokens)]
        train_path = f"{DATA_DIR}/tinyshakespeare/tiny_shakespeare_train.bin"
        val_path = f"{DATA_DIR}/tinyshakespeare/tiny_shakespeare_val.bin"
    elif args.source == "fineweb":
        cmd = ["python", f"{DATA_DIR}/fineweb.py", "--max-tokens", str(args.max_tokens)]
        train_path = f"{DATA_DIR}/fineweb10B/fineweb10B_train_000000.bin"
        val_path = f"{DATA_DIR}/fineweb10B/fineweb10B_val_000000.bin"
    subprocess.run(cmd, check=True)
    final_train = os.path.join(MIX_DIR, "train.bin")
    final_val = os.path.join(MIX_DIR, "val.bin")
    shutil.copy(train_path, final_train)
    shutil.copy(val_path, final_val)

else:
    # Mixture mode
    parts = [p.strip() for p in args.mix.split(",")]
    sources = {}
    for p in parts:
        name, ratio_str = p.split(":")
        sources[name] = float(ratio_str)
    total = sum(sources.values())
    for k in sources:
        sources[k] /= total

    print(f"Preparing mixture: {sources}")
    print(f"Total tokens: {args.max_tokens:,}\n")

    all_train = bytearray()
    all_val = bytearray()

    for name, ratio in sources.items():
        tokens = int(args.max_tokens * ratio)
        print(f"→ {name}: {tokens:,} tokens ({ratio*100:.1f}%)")

        if name == "tinystories":
            cmd = ["python", f"{DATA_DIR}/tinystories.py", "--max-tokens", str(tokens)]
            train_path = f"{DATA_DIR}/tinystories/TinyStories_train.bin"
            val_path = f"{DATA_DIR}/tinystories/TinyStories_val.bin"
        elif name == "tinyshakespeare":
            cmd = ["python", f"{DATA_DIR}/tinyshakespeare.py", "--max-tokens", str(tokens)]
            train_path = f"{DATA_DIR}/tinyshakespeare/tiny_shakespeare_train.bin"
            val_path = f"{DATA_DIR}/tinyshakespeare/tiny_shakespeare_val.bin"
        elif name == "fineweb":
            cmd = ["python", f"{DATA_DIR}/fineweb.py", "--max-tokens", str(tokens)]
            train_path = f"{DATA_DIR}/fineweb10B/fineweb10B_train_000000.bin"
            val_path = f"{DATA_DIR}/fineweb10B/fineweb10B_val_000000.bin"

        subprocess.run(cmd, check=True)

        with open(train_path, "rb") as f:
            f.read(8)
            all_train.extend(f.read())
        with open(val_path, "rb") as f:
            f.read(8)
            all_val.extend(f.read())

    final_train = os.path.join(MIX_DIR, "train.bin")
    final_val = os.path.join(MIX_DIR, "val.bin")

    def write_bin(path, data):
        num_tokens = len(data) // 2
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", num_tokens))
            f.write(data)

    write_bin(final_train, all_train)
    write_bin(final_val, all_val)

print(f"\n✅ Done!")
print(f"Final train.bin: {os.path.getsize(final_train)/1_048_576:.1f} MB")
print(f"Final val.bin:   {os.path.getsize(final_val)/1_048_576:.1f} MB")
print(f"\nFiles saved to: {MIX_DIR}/")
print(f"\nTrain with:")
print(f"./train_gpt2fp32cu -i {MIX_DIR}/train.bin -j {MIX_DIR}/val.bin -t 512 -s 50")
