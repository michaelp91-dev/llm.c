"""
Streams karpathy/climbmix-400b-shuffle from the Hugging Face Hub, tokenizes it with the
GPT-2 tokenizer and writes llm.c .bin shards *as it goes*, so training can start as soon
as the val shard and the first train shard exist (see train_gpt2cu -W 1).

Output (in --out_dir):
    climbmix_val_000000.bin          first --val_tokens tokens of the stream
    climbmix_train_000000.bin, ...   then train shards of --shard_size tokens each
    climbmix_state.json              progress, used to resume tokenizing after a restart
    climbmix_carry.bin               tokens tokenized past the last shard (part of the state)
    STREAM_DONE                      written last, contains the number of train shards

Every shard is written to a temp name and atomically renamed when complete, so a reader
never sees a partial shard. Memory use is constant (one shard-sized numpy buffer).

Optionally (--hf_repo) every finished shard is also committed to a Hugging Face dataset
repo together with the state, so a later run (e.g. a new Colab VM) downloads the existing
shards instead of re-tokenizing them, and continues from where the data left off.
The token comes from the HF_TOKEN env var (or your huggingface-cli login).

Example (5.4B tokens, 45 train shards of 120M tokens):
    python dev/data/climbmix_stream.py --out_dir dev/data/climbmix -n 5400000000
then train with:
    ./train_gpt2cu -i "dev/data/climbmix/climbmix_train_*.bin" \
                   -j "dev/data/climbmix/climbmix_val_*.bin" -W 1 -x <steps> ...
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

import numpy as np

HEADER_MAGIC = 20240520
HEADER_VERSION = 1
PREFIX = "climbmix"
STATE_NAME = f"{PREFIX}_state.json"
CARRY_NAME = f"{PREFIX}_carry.bin"
DONE_NAME = "STREAM_DONE"


def val_name(): return f"{PREFIX}_val_{0:06d}.bin"
def train_name(i): return f"{PREFIX}_train_{i:06d}.bin"


def log(*a):
    print("[climbmix_stream]", *a, flush=True)


def write_atomic(path, data_fn):
    # write via a temp file in the same directory, then rename (atomic on POSIX)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        data_fn(f)
    os.replace(tmp, path)


def write_shard(path, tokens):
    # llm.c .bin format: 256 int32 header (magic, version, ntok), then uint16 tokens
    assert tokens.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = HEADER_MAGIC, HEADER_VERSION, len(tokens)
    def fn(f):
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    write_atomic(path, fn)


def shard_ok(path, ntok):
    # a complete shard has the right header and file size
    try:
        if os.path.getsize(path) != 256 * 4 + ntok * 2:
            return False
        header = np.fromfile(path, dtype=np.int32, count=3)
        return header[0] == HEADER_MAGIC and header[1] == HEADER_VERSION and header[2] == ntok
    except OSError:
        return False


# -----------------------------------------------------------------------------
# optional Hugging Face dataset repo sync

class HubSync:
    def __init__(self, repo, out_dir, dry_run=False):
        self.repo, self.out_dir, self.dry_run = repo, out_dir, dry_run
        self.q = queue.Queue()
        self.failed = False
        self.thread = None
        if dry_run:
            self.api = None
            return
        from huggingface_hub import HfApi
        self.api = HfApi()  # picks up HF_TOKEN from the environment
        self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    def remote_files(self):
        if self.dry_run:
            return set()
        return set(self.api.list_repo_files(self.repo, repo_type="dataset"))

    def read_bytes(self, filename):
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(self.repo, filename, repo_type="dataset"), "rb") as f:
            return f.read()

    def download(self, filename):
        # download into a staging dir, then atomically move into out_dir
        if self.dry_run:
            log(f"[dry-run] would download {filename}")
            return
        from huggingface_hub import hf_hub_download
        staging = os.path.join(self.out_dir, ".hf_download")
        p = hf_hub_download(self.repo, filename, repo_type="dataset", local_dir=staging)
        os.replace(p, os.path.join(self.out_dir, filename))

    def start(self):
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, files, blobs, message):
        # files: names in out_dir that won't change (shards). blobs: {name: bytes} snapshots (state)
        self.q.put((files, blobs, message))

    def _worker(self):
        from huggingface_hub import CommitOperationAdd
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            files, blobs, message = item
            if self.failed:  # never upload a later state if an earlier shard is missing remotely
                self.q.task_done()
                continue
            if self.dry_run:
                log(f"[dry-run] would commit {files + list(blobs)}: {message}")
                self.q.task_done()
                continue
            ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(self.out_dir, f)) for f in files]
            ops += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in blobs.items()]
            for attempt in range(5):
                try:
                    self.api.create_commit(self.repo, ops, commit_message=message, repo_type="dataset")
                    log(f"uploaded to hf://datasets/{self.repo}: {message}")
                    break
                except Exception as e:  # network hiccups etc.
                    log(f"upload failed (attempt {attempt + 1}/5): {type(e).__name__}: {e}")
                    time.sleep(30 * (attempt + 1))
            else:
                log("giving up on uploads for this run; local shards are unaffected")
                self.failed = True
            self.q.task_done()

    def finish(self):
        if self.thread is not None:
            self.q.put(None)
            self.thread.join()


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out_dir", default="dev/data/climbmix")
    ap.add_argument("-n", "--target_tokens", type=float, default=5.4e9, help="train tokens to produce (rounded up to whole shards)")
    ap.add_argument("-s", "--shard_size", type=int, default=120_000_000, help="tokens per train shard")
    ap.add_argument("-v", "--val_tokens", type=int, default=10_000_000, help="tokens in the val shard (taken first)")
    ap.add_argument("--dataset", default="karpathy/climbmix-400b-shuffle")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--no_eot", action="store_true", help="don't prepend <|endoftext|> to each document (llm.c convention is to prepend it)")
    ap.add_argument("--batch_docs", type=int, default=1024, help="documents tokenized per batch")
    ap.add_argument("--num_threads", type=int, default=os.cpu_count() or 4, help="tiktoken threads")
    ap.add_argument("--hf_repo", default=None, help="HF dataset repo to upload shards to / download existing shards from")
    ap.add_argument("--download_only", action="store_true", help="only download what exists in --hf_repo, don't tokenize")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing local state and start over")
    ap.add_argument("--dry_run", action="store_true", help="with --hf_repo: print hub actions instead of doing them")
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    num_train_shards = int(-(-int(args.target_tokens) // args.shard_size))  # ceil
    eot = not args.no_eot
    config = dict(dataset=args.dataset, split=args.split, shard_size=args.shard_size,
                  val_tokens=args.val_tokens, eot=eot)
    log(f"target: {num_train_shards} train shards x {args.shard_size:,} tokens -> {out}")

    def read_state():
        p = os.path.join(out, STATE_NAME)
        if args.fresh or not os.path.exists(p):
            return None
        with open(p) as f:
            st = json.load(f)
        for k, v in config.items():
            if st["config"].get(k) != v:
                sys.exit(f"existing state in {out} was made with {k}={st['config'].get(k)!r}, not {v!r}; use --fresh or another --out_dir")
        return st

    # ---- reuse what already exists: hub first (if asked), then local ----
    hub = HubSync(args.hf_repo, out, args.dry_run) if args.hf_repo else None
    if hub is not None:
        remote = hub.remote_files()
        local_st = read_state()
        if args.dry_run:
            log(f"[dry-run] hub files: {sorted(remote)}")
        if STATE_NAME in remote and not args.dry_run:
            remote_st = json.loads(hub.read_bytes(STATE_NAME))
            local_n = local_st["num_train_shards"] if local_st else -1
            if remote_st["config"] != config:
                log(f"hub state was made with a different config {remote_st['config']}; not using it")
            elif remote_st["num_train_shards"] > local_n:
                # val + shards in order, so training can start as soon as the first ones land
                names = [val_name()] + [train_name(i) for i in range(min(remote_st["num_train_shards"], num_train_shards))]
                for name in names:
                    ntok = args.val_tokens if name == val_name() else args.shard_size
                    if not shard_ok(os.path.join(out, name), ntok):
                        log(f"downloading {name} from hf://datasets/{args.hf_repo}")
                        hub.download(name)
                # adopt the hub's state only once its shards are all here
                if CARRY_NAME in remote:
                    hub.download(CARRY_NAME)
                elif os.path.exists(os.path.join(out, CARRY_NAME)):
                    os.remove(os.path.join(out, CARRY_NAME))
                hub.download(STATE_NAME)
                log(f"have {len(names) - 1} train shards from the hub")
        if args.download_only:
            st = read_state()
            if st and st["num_train_shards"] >= num_train_shards:
                with open(os.path.join(out, DONE_NAME), "w") as f:
                    f.write(f"{num_train_shards}\n")
            else:
                log(f"hub has {st['num_train_shards'] if st else 0} of {num_train_shards} train shards; run without --download_only to make the rest")
            return

    st = read_state()
    if st is not None:
        # trust the state only as far as the local shards actually exist
        have = 0
        while have < st["num_train_shards"] and shard_ok(os.path.join(out, train_name(have)), args.shard_size):
            have += 1
        if not st["val_done"] or not shard_ok(os.path.join(out, val_name()), args.val_tokens) or have < st["num_train_shards"]:
            log(f"state says {st['num_train_shards']} shards but only {have} are complete locally; starting over")
            st = None
    if st is not None and st["num_train_shards"] >= num_train_shards:
        with open(os.path.join(out, DONE_NAME), "w") as f:
            f.write(f"{num_train_shards}\n")
        log(f"all {num_train_shards} train shards already exist, nothing to do")
        return
    if os.path.exists(os.path.join(out, DONE_NAME)):
        os.remove(os.path.join(out, DONE_NAME))

    # ---- set up the token stream ----
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot_id = enc._special_tokens["<|endoftext|>"]
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    cap = max(args.shard_size, args.val_tokens)
    buf = np.empty(cap, dtype=np.uint16)  # the one big buffer, reused for every shard
    fill = 0
    extra = []  # tokens tokenized after the last shard of the target was written (kept in the carry)
    t0 = time.time()
    tokens_this_run = 0

    def push(arr):
        # append tokens to the buffer, writing out each shard as it fills up. returns new shard names
        nonlocal fill, tokens_this_run
        new_files = []
        i = 0
        while i < len(arr):
            if st["num_train_shards"] >= num_train_shards:
                extra.append(arr[i:].copy())
                break
            target = args.shard_size if st["val_done"] else args.val_tokens
            n = min(target - fill, len(arr) - i)
            buf[fill:fill + n] = arr[i:i + n]
            fill += n
            i += n
            if fill == target:
                if not st["val_done"]:
                    name = val_name()
                    st["val_done"] = True
                else:
                    name = train_name(st["num_train_shards"])
                    st["num_train_shards"] += 1
                write_shard(os.path.join(out, name), buf[:fill])
                new_files.append(name)
                tokens_this_run += fill
                fill = 0
                dt = time.time() - t0
                log(f"wrote {name} ({st['num_train_shards']}/{num_train_shards} train shards, "
                    f"{tokens_this_run / max(dt, 1e-9) / 1e6:.2f}M tok/s)")
        return new_files

    if st is None:
        st = dict(config=config, docs_consumed=0, ds_state=None, val_done=False, num_train_shards=0)
        pending = []
    else:
        carry_path = os.path.join(out, CARRY_NAME)
        carry = np.fromfile(carry_path, dtype=np.uint16) if os.path.exists(carry_path) else np.empty(0, np.uint16)
        resumed = False
        if st.get("ds_state") is not None and hasattr(ds, "load_state_dict"):
            try:
                ds.load_state_dict(st["ds_state"])
                resumed = True
            except Exception as e:
                log(f"could not restore dataset position ({e}); skipping documents instead")
        if not resumed and st["docs_consumed"] > 0:
            ds = ds.skip(st["docs_consumed"])
        log(f"resuming after {st['docs_consumed']:,} documents, {st['num_train_shards']} train shards, {len(carry):,} carried tokens")
        pending = push(carry)

    if hub is not None:
        hub.start()

    def save_state(new_files):
        # state + carry describe "everything up to the last finished batch of documents"
        carry = np.concatenate([buf[:fill]] + extra)
        write_atomic(os.path.join(out, CARRY_NAME), lambda f: f.write(carry.tobytes()))
        state_bytes = json.dumps(st, indent=1).encode()
        write_atomic(os.path.join(out, STATE_NAME), lambda f: f.write(state_bytes))
        if hub is not None:
            hub.submit(new_files, {STATE_NAME: state_bytes, CARRY_NAME: carry.tobytes()},
                       f"{', '.join(new_files)} (train shards: {st['num_train_shards']})")

    batch = []
    it = iter(ds)
    exhausted = False
    while st["num_train_shards"] < num_train_shards:
        # grab a batch of documents and tokenize them in parallel
        batch.clear()
        for doc in it:
            batch.append(doc.get(args.text_field) or "")
            if len(batch) >= args.batch_docs:
                break
        if not batch:
            exhausted = True
            break
        toks = enc.encode_ordinary_batch(batch, num_threads=args.num_threads)
        new_files, pending = pending, []
        for t in toks:
            new_files += push(np.asarray([eot_id] + t if eot else t, dtype=np.uint16))
        st["docs_consumed"] += len(batch)
        if new_files:
            if hasattr(ds, "state_dict"):
                try:
                    st["ds_state"] = ds.state_dict()
                except Exception:
                    st["ds_state"] = None
            save_state(new_files)

    if pending:
        save_state(pending)
    if exhausted and fill > 0 and st["val_done"]:
        # the dataset ran out (only happens for small datasets): flush a final, shorter shard
        name = train_name(st["num_train_shards"])
        st["num_train_shards"] += 1
        write_shard(os.path.join(out, name), buf[:fill])
        fill = 0
        save_state([name])

    # done marker last: readers treat it as "every shard is on disk"
    with open(os.path.join(out, DONE_NAME), "w") as f:
        f.write(f"{st['num_train_shards']}\n")
    if hub is not None:
        hub.submit([DONE_NAME], {}, "STREAM_DONE")
        hub.finish()
    log(f"done: {st['num_train_shards']} train shards in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
