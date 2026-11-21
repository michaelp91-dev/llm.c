"""
Uploads an llm.c checkpoint to a Hugging Face model repo. train_gpt2cu runs this in the
background when given -hf <repo_id>: it stages model_XXXXXXXX.bin, state_XXXXXXXX_*.bin,
main.log and DONE_XXXXXXXX into <output_log_dir>/.hf_push/step_XXXXXXXX (hardlinks, so
-nk deletions can't break an upload in flight) and then calls

    python dev/hf_push.py --repo <repo_id> --staging <output_log_dir>/.hf_push/step_XXXXXXXX

The files land in ckpt/ of the repo (created private if missing) and older ckpt/ files are
deleted in the same commit, so the repo only ever holds the latest checkpoint. Pushes are
serialized with a lock file; if a newer checkpoint was staged while waiting, only the
newest one is uploaded. By default the repo history is squashed after each push so the
deleted checkpoints stop counting against your storage.

The token comes from the HF_TOKEN environment variable (or a cached `hf auth login`);
it is never printed.

Wait for all pending pushes to finish (e.g. at the end of a Colab notebook):

    python dev/hf_push.py --wait <output_log_dir>
"""

import argparse
import fcntl
import glob
import os
import re
import shutil
import sys
import time

CKPT_PATTERNS = ["model_*.bin", "state_*.bin", "DONE_*", "main.log"]


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[hf_push]", *a, flush=True)


def staged_steps(push_dir):
    # complete staging dirs (DONE_ is staged last), oldest first
    out = []
    for d in glob.glob(os.path.join(push_dir, "step_*")):
        m = re.match(r"step_(\d+)$", os.path.basename(d))
        if m and glob.glob(os.path.join(d, "DONE_*")):
            out.append((int(m.group(1)), d))
    return sorted(out)


def upload(repo, folder, step, squash, dry_run):
    files = sorted(os.listdir(folder))
    if dry_run:
        log(f"[dry-run] would create private repo {repo} (if missing), upload {files} to ckpt/ "
            f"and delete other ckpt/ files matching {CKPT_PATTERNS}" + (", then squash history" if squash else ""))
        return
    from huggingface_hub import HfApi
    api = HfApi()  # token from HF_TOKEN / cached login
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo="ckpt",
                      commit_message=f"checkpoint step {step}",
                      delete_patterns=CKPT_PATTERNS)  # files being uploaded are kept
    if squash:
        try:
            api.super_squash_history(repo_id=repo, commit_message=f"latest checkpoint: step {step}")
        except Exception as e:
            log(f"squash failed (not fatal): {type(e).__name__}: {e}")


def push(args):
    push_dir = os.path.dirname(os.path.abspath(args.staging))
    os.makedirs(push_dir, exist_ok=True)
    with open(os.path.join(push_dir, "lock"), "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)  # one upload at a time
        steps = staged_steps(push_dir)
        if not steps:
            log("nothing staged (already pushed by a later run of this script)")
            return 0
        step, folder = steps[-1]
        for _, old in steps[:-1]:
            log(f"skipping older staged checkpoint {os.path.basename(old)}, a newer one is ready")
            shutil.rmtree(old, ignore_errors=True)
        for attempt in range(1, 4):
            try:
                t0 = time.time()
                upload(args.repo, folder, step, not args.no_squash, args.dry_run)
                log(f"pushed step {step} to hf://{args.repo}/ckpt in {time.time() - t0:.0f}s")
                shutil.rmtree(folder, ignore_errors=True)
                return 0
            except Exception as e:
                log(f"push of step {step} failed (attempt {attempt}/3): {type(e).__name__}: {e}")
                time.sleep(30 * attempt)
        log(f"giving up on step {step} for now; it stays staged and is retried by the next push or --wait")
        return 1


def wait(args):
    push_dir = os.path.join(args.wait, ".hf_push")
    if not os.path.isdir(push_dir):
        return 0
    # take the lock (waits for any running upload), then retry anything still staged
    while True:
        with open(os.path.join(push_dir, "lock"), "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            steps = staged_steps(push_dir)
        if not steps:
            log("no pending pushes")
            return 0
        if not args.repo:
            log(f"{len(steps)} staged checkpoint(s) left and no --repo given to retry them")
            return 1
        args.staging = steps[-1][1]
        if push(args) != 0:
            return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="Hugging Face repo id, e.g. user/llmc-d16")
    ap.add_argument("--staging", help="staged checkpoint dir (<out>/.hf_push/step_XXXXXXXX)")
    ap.add_argument("--wait", metavar="OUT_DIR", help="wait until pending pushes for this output dir are done")
    ap.add_argument("--no_squash", action="store_true", help="keep the full commit history (uses more storage)")
    ap.add_argument("--dry_run", action="store_true", default=os.environ.get("LLMC_HF_DRY_RUN") == "1",
                    help="print what would be uploaded, touch nothing remote (or set LLMC_HF_DRY_RUN=1)")
    args = ap.parse_args()
    if args.wait:
        sys.exit(wait(args))
    if not args.repo or not args.staging:
        ap.error("--repo and --staging are required (or use --wait)")
    sys.exit(push(args))


if __name__ == "__main__":
    main()
