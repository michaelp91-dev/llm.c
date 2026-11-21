"""
Downloads the latest llm.c checkpoint pushed by train_gpt2cu -hf <repo_id> (see dev/hf_push.py)
from ckpt/ of the Hugging Face repo into an output dir, so that `train_gpt2cu -o <out_dir> -y 1`
resumes from it, e.g. after a Colab disconnect:

    python dev/hf_pull.py --repo user/llmc-d16 --out_dir /content/models/d16

Does nothing (and exits 0) if the repo or a checkpoint doesn't exist yet, or if the output dir
already has that checkpoint or a newer one. DONE_XXXXXXXX is written last, so an interrupted
download is never mistaken for a complete checkpoint. Token: HF_TOKEN env or cached login.
"""

import argparse
import glob
import os
import re
import shutil


def log(*a):
    print("[hf_pull]", *a, flush=True)


def local_max_step(out_dir):
    steps = [int(m.group(1)) for p in glob.glob(os.path.join(out_dir, "DONE_*"))
             if (m := re.match(r"DONE_(\d+)$", os.path.basename(p)))]
    return max(steps, default=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dry_run", action="store_true", help="only print what would be downloaded")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi()
    try:
        files = api.list_repo_files(args.repo)
    except RepositoryNotFoundError:
        log(f"repo {args.repo} not found (or no access): fresh start")
        return
    ckpt = [f for f in files if f.startswith("ckpt/")]
    steps = [int(m.group(1)) for f in ckpt if (m := re.match(r"ckpt/DONE_(\d+)$", f))]
    if not steps:
        log(f"no checkpoint in {args.repo}/ckpt: fresh start")
        return
    step = max(steps)
    have = local_max_step(args.out_dir)
    if have >= step:
        log(f"{args.out_dir} already has step {have} (hub has {step}): nothing to do")
        return
    tag = f"{step:08d}"
    wanted = [f for f in ckpt if re.match(rf"ckpt/(model_{tag}\.bin|state_{tag}_\d+\.bin|main\.log)$", f)]
    if not any(f.startswith("ckpt/model_") for f in wanted):
        raise SystemExit(f"ckpt/DONE_{tag} exists but ckpt/model_{tag}.bin is missing")
    wanted.append(f"ckpt/DONE_{tag}")  # last
    if args.dry_run:
        log(f"[dry-run] would download step {step}: {wanted} -> {args.out_dir}")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    staging = os.path.join(args.out_dir, ".hf_pull")
    for f in wanted:
        log(f"downloading {f}")
        p = hf_hub_download(args.repo, f, local_dir=staging)
        os.replace(p, os.path.join(args.out_dir, os.path.basename(f)))
    shutil.rmtree(staging, ignore_errors=True)
    log(f"restored checkpoint step {step} into {args.out_dir}; resume with -y 1")


if __name__ == "__main__":
    main()
