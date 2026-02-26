import argparse
import subprocess
import os

parser = argparse.ArgumentParser(description="Prepare mixed or single dataset for llm.c training")
parser.add_argument("--max-tokens", type=int, default=None, 
                    help="Total tokens for train+val (95% train, 5% val)")
parser.add_argument("--source", type=str, default=None,
                    choices=["tinystories", "fineweb", "tinyshakespeare"],
                    help="Single source (use instead of --mix)")
parser.add_argument("--mix", type=str, default=None,
                    help="Mixture like 'fineweb:0.5,tinyshakespeare:0.3,tinystories:0.2'")
args = parser.parse_args()

if args.max_tokens is None:
    args.max_tokens = 1_000_000_000_000  # very large number = full dataset

DATA_DIR = "dev/data"

def run_prepro(script, max_tokens):
    cmd = ["python", f"{DATA_DIR}/{script}", "--max-tokens", str(max_tokens)]
    subprocess.run(cmd, check=True, capture_output=False)

if args.mix:
    # Parse mix string: "fineweb:0.5,tinyshakespeare:0.3,tinystories:0.2"
    parts = args.mix.replace(" ", "").split(",")
    sources = {}
    total_ratio = 0.0
    for p in parts:
        name, ratio_str = p.split(":")
        ratio = float(ratio_str)
        sources[name] = ratio
        total_ratio += ratio

    if abs(total_ratio - 1.0) > 0.001:
        print("Warning: ratios do not sum to 1.0, normalizing...")
        for k in sources:
            sources[k] /= total_ratio

    print(f"Mixing datasets: {sources}")
    print(f"Total tokens: {args.max_tokens:,}")

    # Calculate tokens for each source
    for name, ratio in sources.items():
        tokens_for_this = int(args.max_tokens * ratio)
        print(f"→ {name}: {tokens_for_this:,} tokens ({ratio*100:.1f}%)")
        run_prepro(f"{name}.py", tokens_for_this)

    print("\nAll datasets prepared with clean 95/5 splits.")
    print("You can now manually concatenate the .bin files if needed, or train on one at a time.")

else:
    # Single source mode (original behavior)
    if args.source is None:
        print("Error: Use either --source or --mix")
        exit(1)
    
    script = f"{args.source}.py"
    print(f"Preparing single dataset: {args.source} ({args.max_tokens:,} tokens)")
    run_prepro(script, args.max_tokens)

print("\nDone!")
