"""
Offline tests for dev/hf_push.py, dev/hf_pull.py and the --hf_repo path of
dev/data/climbmix_stream.py. The Hugging Face Hub and the dataset are replaced by
local fakes, so no token or network is needed.

run from the llm.c root:  python dev/test/test_hf_helpers.py
"""
import fnmatch
import hashlib
import os
import runpy
import shutil
import sys
import tempfile

import huggingface_hub
import datasets

ROOT = tempfile.mkdtemp(prefix="llmc_hf_test_")
HUB = os.path.join(ROOT, "hub")


class FakeApi:
    """the subset of HfApi the scripts use, backed by a local directory per repo"""
    commits = []

    def __init__(self, *a, **k): pass

    def _dir(self, repo): return os.path.join(HUB, repo)

    def create_repo(self, repo, repo_type=None, private=False, exist_ok=False):
        assert private, "repos must be created private"
        os.makedirs(self._dir(repo), exist_ok=True)

    def list_repo_files(self, repo, repo_type=None):
        d = self._dir(repo)
        if not os.path.isdir(d):
            raise _NotFound()
        return sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs)

    def upload_folder(self, repo_id, folder_path, path_in_repo, commit_message="", delete_patterns=()):
        d = os.path.join(self._dir(repo_id), path_in_repo)
        os.makedirs(d, exist_ok=True)
        new = set(os.listdir(folder_path))
        for f in os.listdir(d):  # delete matching remote files that are not being uploaded
            if f not in new and any(fnmatch.fnmatch(f, p) for p in delete_patterns):
                os.remove(os.path.join(d, f))
        for f in new:
            shutil.copy(os.path.join(folder_path, f), os.path.join(d, f))
        FakeApi.commits.append((repo_id, commit_message))

    def create_commit(self, repo, ops, commit_message="", repo_type=None):
        for op in ops:
            dst = os.path.join(self._dir(repo), op.path_in_repo)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if isinstance(op.path_or_fileobj, (bytes, bytearray)):
                open(dst, "wb").write(op.path_or_fileobj)
            else:
                shutil.copy(op.path_or_fileobj, dst)
        FakeApi.commits.append((repo, commit_message))

    def super_squash_history(self, repo_id, commit_message=""):
        FakeApi.commits.append((repo_id, "squash"))


class _NotFound(Exception):
    pass


def fake_download(repo, filename, repo_type=None, local_dir=None, **k):
    src = os.path.join(HUB, repo, filename)
    dst = os.path.join(local_dir or os.path.join(ROOT, "cache", repo), filename)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
    return dst


huggingface_hub.HfApi = FakeApi
huggingface_hub.hf_hub_download = fake_download
huggingface_hub.utils.RepositoryNotFoundError = _NotFound


def fake_docs():
    for i in range(200_000):
        yield {"text": f"document {i}: " + " ".join(f"word{(i * 7 + j) % 997}" for j in range(i % 50 + 5))}


datasets.load_dataset = lambda *a, **k: datasets.IterableDataset.from_generator(fake_docs)


def run(script, *args):
    sys.argv = [script] + [str(a) for a in args]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()


def test_push_pull():
    print("test_push_pull...")
    out = os.path.join(ROOT, "run")
    repo = "me/llmc-test"
    for step in (250, 500):
        st = os.path.join(out, ".hf_push", f"step_{step:08d}")
        os.makedirs(st)
        for name in (f"model_{step:08d}.bin", f"state_{step:08d}_00000.bin", "main.log", f"DONE_{step:08d}"):
            open(os.path.join(st, name), "wb").write(os.urandom(1000) if name.endswith(".bin") else b"x")
    # both staged before the push runs: only the newest gets uploaded
    run("dev/hf_push.py", "--repo", repo, "--staging", os.path.join(out, ".hf_push", "step_00000250"))
    files = FakeApi().list_repo_files(repo)
    assert files == ["ckpt/DONE_00000500", "ckpt/main.log", "ckpt/model_00000500.bin", "ckpt/state_00000500_00000.bin"], files
    assert not [d for d in os.listdir(os.path.join(out, ".hf_push")) if d.startswith("step_")]
    # a later push replaces the old checkpoint on the hub
    st = os.path.join(out, ".hf_push", "step_00000750")
    os.makedirs(st)
    for name in ("model_00000750.bin", "state_00000750_00000.bin", "main.log", "DONE_00000750"):
        open(os.path.join(st, name), "wb").write(b"750")
    run("dev/hf_push.py", "--repo", repo, "--staging", st)
    files = FakeApi().list_repo_files(repo)
    assert files == ["ckpt/DONE_00000750", "ckpt/main.log", "ckpt/model_00000750.bin", "ckpt/state_00000750_00000.bin"], files
    run("dev/hf_push.py", "--wait", out)
    # pull into a fresh dir (as after a Colab disconnect)
    fresh = os.path.join(ROOT, "fresh")
    run("dev/hf_pull.py", "--repo", repo, "--out_dir", fresh)
    assert sorted(os.listdir(fresh)) == ["DONE_00000750", "main.log", "model_00000750.bin", "state_00000750_00000.bin"], os.listdir(fresh)
    run("dev/hf_pull.py", "--repo", repo, "--out_dir", fresh)  # already there: no-op
    run("dev/hf_pull.py", "--repo", "me/does-not-exist", "--out_dir", os.path.join(ROOT, "x"))  # fresh start, no error
    print("OK")


def test_stream_hub_resume():
    print("test_stream_hub_resume...")
    common = ["-s", 20000, "-v", 4096, "--batch_docs", 64, "--num_threads", 2]
    ref = os.path.join(ROOT, "ref")
    run("dev/data/climbmix_stream.py", "-o", ref, "-n", 100000, *common)  # 5 shards straight through
    a = os.path.join(ROOT, "a")
    run("dev/data/climbmix_stream.py", "-o", a, "-n", 60000, "--hf_repo", "me/data", *common)  # 3 shards, uploaded
    hub_files = FakeApi().list_repo_files("me/data")
    assert "climbmix_train_000002.bin" in hub_files and "STREAM_DONE" in hub_files, hub_files
    b = os.path.join(ROOT, "b")  # new machine: reuse the 3 hub shards, tokenize 2 more
    run("dev/data/climbmix_stream.py", "-o", b, "-n", 100000, "--hf_repo", "me/data", *common)
    for f in sorted(os.listdir(ref)):
        if f.endswith(".bin"):
            assert md5(os.path.join(ref, f)) == md5(os.path.join(b, f)), f"{f} differs"
    assert open(os.path.join(b, "STREAM_DONE")).read().strip() == "5"
    c = os.path.join(ROOT, "c")  # download only
    run("dev/data/climbmix_stream.py", "-o", c, "-n", 100000, "--hf_repo", "me/data", "--download_only", *common)
    assert md5(os.path.join(ref, "climbmix_train_000004.bin")) == md5(os.path.join(c, "climbmix_train_000004.bin"))
    print("OK")


if __name__ == "__main__":
    try:
        test_push_pull()
        test_stream_hub_resume()
        print("all hf helper tests passed")
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
