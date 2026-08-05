"""
Generic124M education-heavy mix for llm.c pretraining.

Sources (all open licenses):
  1. FineWeb-Edu sample-10BT  (ODC-By)  – educational web, filterable by score
  2. Cosmopedia-v2            (ODC-By)  – synthetic textbooks / stories
  3. Python-Edu               (ODC-By)  – educational Python from Stack (optional;
                                          requires Software Heritage S3 access)

Target mix (approx):
  FineWeb-Edu  ~55%
  Cosmopedia   ~25%
  Python-Edu   ~15%
  (OpenHermes  ~5% can be added later as a small annealing stage)

Example (Colab + Drive):

  from google.colab import drive
  drive.mount('/content/drive')

  # Strict FineWeb-Edu only (score >= 4), write to Drive, cap at 2B tokens:
  !python dev/data/generic124m_mix.py \\
      --source fineweb-edu \\
      --output-dir /content/drive/MyDrive/llm_c_data/generic124m \\
      --min-score 4 \\
      --max-tokens 2000000000

  # Cosmopedia slice:
  !python dev/data/generic124m_mix.py \\
      --source cosmopedia \\
      --output-dir /content/drive/MyDrive/llm_c_data/generic124m \\
      --max-tokens 800000000

  # Then point train_gpt2cu at the resulting train_*.bin / val_*.bin shards.
"""

import argparse
import os
import multiprocessing as mp
from functools import partial

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from data_common import write_datafile

# -----------------------------------------------------------------------------
# Tokenization (GPT-2 only for now – matches train_gpt2cu)

def _get_encoder():
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    return enc, eot


# Per-process tokenizer cache (multiprocessing-safe)
_ENC = None
_EOT = None

def _init_tokenizer():
    global _ENC, _EOT
    if _ENC is None:
        _ENC, _EOT = _get_encoder()

def tokenize_doc(doc, text_key="text"):
    """Tokenize one document → uint16 numpy array with leading EOT."""
    _init_tokenizer()
    text = doc.get(text_key) or ""
    if not text or not isinstance(text, str):
        return np.array([_EOT], dtype=np.uint16)
    tokens = [_EOT]
    tokens.extend(_ENC.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.int64)
    # GPT-2 vocab fits in uint16
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)


# -----------------------------------------------------------------------------
# Shard writer

def write_shards(token_iter, output_dir, name_prefix, shard_size, max_tokens=None):
    """
    Consume an iterator of uint16 token arrays and write llm.c .bin shards.
    First shard is always named *_val_*.bin (validation), rest are train.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_index = 0
    all_tokens = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    total_written = 0
    progress = None

    def flush(final=False):
        nonlocal shard_index, token_count, progress, total_written
        if token_count == 0:
            return
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(
            output_dir, f"{name_prefix}_{split}_{shard_index:06d}.bin"
        )
        write_datafile(filename, all_tokens[:token_count].tolist(), "gpt-2")
        total_written += token_count
        shard_index += 1
        token_count = 0
        if progress is not None:
            progress.close()
            progress = None

    for tokens in token_iter:
        if max_tokens is not None and total_written + token_count >= max_tokens:
            break

        # truncate last doc if we would exceed max_tokens
        if max_tokens is not None:
            remaining = max_tokens - (total_written + token_count)
            if remaining <= 0:
                break
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

        if token_count + len(tokens) < shard_size:
            all_tokens[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress is None:
                progress = tqdm(
                    total=shard_size, unit="tok", desc=f"Shard {shard_index}"
                )
            progress.update(len(tokens))
        else:
            remainder = shard_size - token_count
            if progress is not None:
                progress.update(remainder)
            all_tokens[token_count : token_count + remainder] = tokens[:remainder]
            token_count += remainder
            flush()
            # leftover of this document starts the next shard
            leftover = tokens[remainder:]
            if len(leftover) > 0:
                all_tokens[: len(leftover)] = leftover
                token_count = len(leftover)
                progress = tqdm(
                    total=shard_size, unit="tok", desc=f"Shard {shard_index}"
                )
                progress.update(len(leftover))

    flush(final=True)
    print(f"Done. Wrote {total_written:,} tokens across {shard_index} shard(s) → {output_dir}")
    return total_written


# -----------------------------------------------------------------------------
# Source loaders

def iter_fineweb_edu(min_score, max_tokens, num_proc):
    """
    Stream FineWeb-Edu sample-10BT, keep docs with int_score >= min_score.
    FineWeb-Edu is already filtered to score >= 3; raising min_score to 4
    keeps only the highest-educational tail.
    """
    print(f"Loading HuggingFaceFW/fineweb-edu (sample-10BT), min_score={min_score} ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    kept = 0
    skipped = 0
    for doc in ds:
        score = doc.get("int_score", doc.get("score", 0))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0
        if score < min_score:
            skipped += 1
            continue
        text = doc.get("text") or ""
        if len(text) < 64:  # drop near-empty
            skipped += 1
            continue
        kept += 1
        yield {"text": text}
        if max_tokens is not None and kept % 10000 == 0:
            # soft progress; actual token limit enforced in write_shards
            pass
    print(f"FineWeb-Edu filter: kept={kept:,}  skipped={skipped:,}")


def iter_cosmopedia(max_tokens, num_proc):
    """
    Stream Cosmopedia-v2 (synthetic textbooks / stories).
    Text column is present; no extra download needed.
    """
    print("Loading HuggingFaceTB/smollm-corpus (cosmopedia-v2) ...")
    # streaming avoids the ~122 GB materialization on Colab disk
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )
    for doc in ds:
        text = doc.get("text") or ""
        if len(text) < 64:
            continue
        yield {"text": text}


def iter_python_edu(max_tokens, num_proc):
    """
    Python-Edu from SmolLM-Corpus.

    NOTE: The public dataset only contains blob_id metadata. Actual file
    contents live on Software Heritage's S3 bucket and require boto3 +
    network access. This loader attempts the download; if it fails it
    yields nothing and prints a warning so the rest of the mix still works.
    """
    print("Loading HuggingFaceTB/smollm-corpus (python-edu) ...")
    try:
        import boto3
        from botocore.exceptions import ClientError
        import gzip
    except ImportError:
        print(
            "WARNING: boto3 not installed – skipping python-edu. "
            "Install with: pip install boto3"
        )
        return

    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "python-edu",
        split="train",
        streaming=True,
    )
    s3 = boto3.client("s3")
    bucket = "softwareheritage"
    ok, fail = 0, 0

    for row in ds:
        blob_id = row.get("blob_id")
        if not blob_id:
            fail += 1
            continue
        # optional score filter (dataset is already score >= 4)
        int_score = row.get("int_score", 4)
        if int_score is not None and int_score < 4:
            continue
        try:
            obj = s3.get_object(Bucket=bucket, Key=f"content/{blob_id}")
            with gzip.GzipFile(fileobj=obj["Body"]) as fin:
                content = fin.read().decode("utf-8", errors="ignore")
            if len(content) < 64:
                fail += 1
                continue
            ok += 1
            yield {"text": content}
        except Exception:
            fail += 1
            continue
    print(f"Python-Edu download: ok={ok:,}  fail={fail:,}")


# -----------------------------------------------------------------------------
# Main

SOURCES = {
    "fineweb-edu": iter_fineweb_edu,
    "cosmopedia": iter_cosmopedia,
    "python-edu": iter_python_edu,
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Generic124M education-heavy mix for llm.c"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="fineweb-edu",
        choices=list(SOURCES.keys()) + ["all"],
        help="Which source to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write .bin shards (default: dev/data/generic124m_mix/<source>)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=4.0,
        help="FineWeb-Edu: keep docs with int_score >= this (default 4.0 = strict)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Stop after writing this many tokens (useful for mix ratios)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10**8,
        help="Tokens per .bin shard (default 100M)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="Worker processes for tokenization",
    )
    args = parser.parse_args()

    sources_to_run = list(SOURCES.keys()) if args.source == "all" else [args.source]

    for src in sources_to_run:
        out_dir = args.output_dir
        if out_dir is None:
            out_dir = os.path.join(
                os.path.dirname(__file__), "generic124m_mix", src.replace("-", "_")
            )
        else:
            # keep sources in subfolders when writing a shared Drive root
            out_dir = os.path.join(out_dir, src.replace("-", "_"))

        print("=" * 60)
        print(f"Source: {src}")
        print(f"Output: {out_dir}")
        print(f"min_score={args.min_score}  max_tokens={args.max_tokens}  shard_size={args.shard_size}")
        print("=" * 60)

        if src == "fineweb-edu":
            doc_iter = iter_fineweb_edu(args.min_score, args.max_tokens, args.num_proc)
        elif src == "cosmopedia":
            doc_iter = iter_cosmopedia(args.max_tokens, args.num_proc)
        elif src == "python-edu":
            doc_iter = iter_python_edu(args.max_tokens, args.num_proc)
        else:
            raise ValueError(src)

        # tokenize in a process pool
        nprocs = max(1, args.num_proc)
        tokenize = partial(tokenize_doc, text_key="text")

        def token_iter():
            # For streaming HF datasets, tokenize in-process by default.
            # Multiprocessing + streaming iterators is fragile on Colab.
            # tiktoken is fast enough single-process for this scale.
            _init_tokenizer()
            for doc in doc_iter:
                tokens = tokenize_doc(doc)
                if tokens is not None and len(tokens) > 1:
                    yield tokens

        write_shards(
            token_iter(),
            output_dir=out_dir,
            name_prefix=src.replace("-", "_"),
            shard_size=args.shard_size,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
