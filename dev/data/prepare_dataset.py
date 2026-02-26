import argparse
import subprocess
import os
import shutil
import struct

parser = argparse.ArgumentParser(description="Prepare mixed dataset for llm.c")
parser.add_argument("--max-tokens", type=int, default=None)
parser.add_argument("--mix", type=str, required=True,
                    help="Mixture like 'fineweb:0.5,tinyshakespeare:0.3,tinystories:0.2'")
args = parser.parse_args()

if args.max_tokens is None:
    args.max_tokens = 1_000_000_000_000

# Parse mixture
parts = [p.strip() for p in args.mix.split(",")]
sources = {}
for p in parts:
    name, ratio_str = p.split(":")
    sources[name] = float(ratio_str)

# Normalize
total = sum(sources.values())
for k in sources:
    sources[k] /= total

DATA_DIR = "dev/data"
MIX_DIR = os.path.join(DATA_DIR, "mixed")
os.makedirs(MIX_DIR, exist_ok=True)

print(f"Preparing mixture: {sources}")
print(f"Total tokens: {args.max_tokens:,}\n")

train_paths = []
val_paths = []

for name, ratio in sources.items():
    tokens = int(args.max_tokens * ratio)
    print(f"→ {name}: {tokens:,} tokens ({ratio*100:.1f}%)")

    if name == "tinystories":
        cmd = ["python", f"{DATA_DIR}/tinystories.py", "--max-tokens", str(tokens)]
        train_path = f"{DATA_DIR}/tinystories/TinyStories_train.bin"
        val_path   = f"{DATA_DIR}/tinystories/TinyStories_val.bin"
    elif name == "tinyshakespeare":
        cmd = ["python", f"{DATA_DIR}/tinyshakespeare.py", "--max-tokens", str(tokens)]
        train_path = f"{DATA_DIR}/tinyshakespeare/tiny_shakespeare_train.bin"
        val_path   = f"{DATA_DIR}/tinyshakespeare/tiny_shakespeare_val.bin"
    elif name == "fineweb":
        cmd = ["python", f"{DATA_DIR}/fineweb.py", "--max-tokens", str(tokens)]
        train_path = f"{DATA_DIR}/fineweb10B/fineweb10B_train_000000.bin"
        val_path   = f"{DATA_DIR}/fineweb10B/fineweb10B_val_000000.bin"

    subprocess.run(cmd, check=True)

    train_paths.append(train_path)
    val_paths.append(val_path)

final_train = os.path.join(MIX_DIR, "train.bin")
final_val   = os.path.join(MIX_DIR, "val.bin")

def concat_bin_files(output_path, input_paths):
    all_tokens = bytearray()
    for path in input_paths:
        with open(path, "rb") as f:
            f.read(8)  # skip the 8-byte header
            all_tokens.extend(f.read())
    total_tokens = len(all_tokens) // 2   # uint16 tokens

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", total_tokens))  # write correct header
        f.write(all_tokens)

print("\nConcatenating with correct header...")

concat_bin_files(final_train, train_paths)
concat_bin_files(final_val, val_paths)

print(f"\n✅ Success!")
print(f"Final train.bin: {os.path.getsize(final_train)/1_048_576:.1f} MB")
print(f"Final val.bin:   {os.path.getsize(final_val)/1_048_576:.1f} MB")
print(f"\nReady to train:")
print(f"./train_gpt2fp32cu -i {MIX_DIR}/train.bin -j {MIX_DIR}/val.bin -t 512 -s 50")
