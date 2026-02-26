import argparse
import subprocess
import os
import shutil

parser = argparse.ArgumentParser(description="Prepare single or mixed dataset for llm.c")
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

# Normalize ratios
total = sum(sources.values())
for k in sources:
    sources[k] /= total

DATA_DIR = "dev/data"
MIX_DIR = os.path.join(DATA_DIR, "mixed")
os.makedirs(MIX_DIR, exist_ok=True)

print(f"Preparing mixture: {sources}")
print(f"Total tokens: {args.max_tokens:,}\n")

train_files = []
val_files = []

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

    train_files.append(train_path)
    val_files.append(val_path)

# === Concatenate into final mixed files ===
final_train = os.path.join(MIX_DIR, "train.bin")
final_val   = os.path.join(MIX_DIR, "val.bin")

print("\nConcatenating files...")

with open(final_train, "wb") as f:
    for path in train_files:
        with open(path, "rb") as src:
            shutil.copyfileobj(src, f)

with open(final_val, "wb") as f:
    for path in val_files:
        with open(path, "rb") as src:
            shutil.copyfileobj(src, f)

print(f"✅ Final train.bin: {os.path.getsize(final_train)/1_048_576:.1f} MB")
print(f"✅ Final val.bin:   {os.path.getsize(final_val)/1_048_576:.1f} MB")
print(f"\nFiles ready at: {MIX_DIR}/")
print("You can now train with:")
print(f"./train_gpt2fp32cu -i {MIX_DIR}/train.bin -j {MIX_DIR}/val.bin -t 512 -s 50")
