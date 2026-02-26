import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--max-tokens", type=int, default=None)
parser.add_argument("--source", type=str, required=True, choices=["tinystories", "fineweb", "tinyshakespeare"])
args = parser.parse_args()

if args.source == "tinystories":
    cmd = ["python", "dev/data/tinystories.py", "--max-tokens", str(args.max_tokens)]
elif args.source == "tinyshakespeare":
    cmd = ["python", "dev/data/tinyshakespeare.py", "--max-tokens", str(args.max_tokens)]
elif args.source == "fineweb":
    cmd = ["python", "dev/data/fineweb.py", "--max-tokens", str(args.max_tokens)]

subprocess.run(cmd, check=True)
