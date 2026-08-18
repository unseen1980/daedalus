"""Live check of daedalus/publisher.py against the real Hub.

The offline suite fakes HfApi, so it cannot exercise the things that actually
break a publish: LFS (files over 10 MB take a different upload path),
`upload_folder` on a directory containing a 321 MB safetensors, real repo
creation, or privacy defaults. `hub-restore.md` exists because exactly that gap
hid problems in the *checkpoint* path; this is the same check for the
*deliverable* path.

Uses a random-init `daedalus-150m`, so the weights are meaningless -- the point
is the transfer, and the byte sizes are the real ones. Publishes to a throwaway
repo, verifies by listing the repo back, then deletes it, so the release name is
not squatted with junk.

Run:  python runs/preflight/publish_live.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

import export
from daedalus import publisher
from daedalus.config import PRESETS
from daedalus.model import Daedalus
from daedalus.muon import build_optimizers
from train import save_checkpoint

SMOKE_REPO = "Unseen1980/daedalus-publish-smoke"
# A real llama.cpp Q4_0 of a daedalus-150m, left by the quantization dry run.
# Using a real one rather than random bytes keeps the LFS path honest.
GGUF_CANDIDATES = ("/tmp/q4scratch/model-Q4_0.gguf",
                   "/tmp/dense-gguf-dryrun/dense-q4_0.gguf")


def main():
    token = os.environ.get("HF_TOKEN_WRITE")
    if not token:
        print("HF_TOKEN_WRITE not set; cannot run the live check")
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    work = tempfile.mkdtemp(prefix="publish-live-")
    result = {"repo": SMOKE_REPO, "phases": {}}
    try:
        # --- A. produce a real export, at real size -------------------------
        t0 = time.time()
        run_dir = os.path.join(work, "runs", "publish-smoke")
        os.makedirs(run_dir)
        cfg = PRESETS["daedalus-150m"]
        model = Daedalus(cfg)
        muon, adamw, _ = build_optimizers(model)
        ckpt = save_checkpoint(os.path.join(run_dir, "checkpoint.pt"), model,
                               muon, adamw, step=1234, tokens_seen=5_000_000_000,
                               cfg=cfg)
        with open(os.path.join(run_dir, "milestone.json"), "w") as f:
            json.dump({"revision": "publish-smoke-stable-end-step1234",
                       "step": 1234, "tokens_seen": 2_750_000_000,
                       "decay_frac": 0.45, "lr_mult_at_branch": 1.0,
                       "muon_lr": 0.02, "adam_lr": 3e-4,
                       "config": "daedalus-150m",
                       "repo": "Unseen1980/daedalus-checkpoints",
                       "path_in_repo": "milestone/publish-smoke/checkpoint.pt"}, f)

        hf_dir = os.path.join(work, "export", "hf")
        export.export_hf_model(ckpt, "daedalus-150m", hf_dir)
        export.export_tokenizer(hf_dir)
        sizes = {n: os.path.getsize(os.path.join(hf_dir, n))
                 for n in sorted(os.listdir(hf_dir))}
        result["phases"]["A_export"] = {
            "seconds": round(time.time() - t0, 1), "files": sizes,
            "publishable": publisher.check_publishable(hf_dir) or "yes"}
        print(f"A: exported {len(sizes)} files, "
              f"{sum(sizes.values()) / 1e6:.1f} MB total")

        gguf = next((g for g in GGUF_CANDIDATES if os.path.exists(g)), None)
        print(f"A: gguf = {gguf} "
              f"({os.path.getsize(gguf) / 1e6:.1f} MB)" if gguf else "A: no gguf")

        # --- B. publish -----------------------------------------------------
        t0 = time.time()
        published = publisher.publish_model(
            hf_dir, SMOKE_REPO, gguf_paths=[gguf] if gguf else [],
            token=token, private=True,
            commit_message="publish_live.py smoke")
        mb = (sum(sizes.values()) + (os.path.getsize(gguf) if gguf else 0)) / 1e6
        elapsed = time.time() - t0
        result["phases"]["B_publish"] = {
            "seconds": round(elapsed, 1), "MB": round(mb, 1),
            "Mbit_per_s": round(mb * 8 / max(elapsed, 1e-9), 1),
            "result": published}
        print(f"B: published {mb:.1f} MB in {elapsed:.1f}s")

        # --- C. verify by listing the repo back, not by trusting the return --
        info = api.repo_info(SMOKE_REPO, repo_type="model", files_metadata=True)
        remote = {s.rfilename: {"size": s.size, "lfs": s.lfs is not None}
                  for s in info.siblings}
        expected = set(sizes) | ({f"gguf/{os.path.basename(gguf)}"} if gguf else set())
        missing = sorted(expected - set(remote))
        result["phases"]["C_verify"] = {
            "private": info.private, "remote": remote, "missing": missing,
            "lfs_files": sorted(k for k, v in remote.items() if v["lfs"])}
        print(f"C: {len(remote)} files on the Hub, private={info.private}, "
              f"missing={missing or 'none'}")
        result["ok"] = (not missing) and info.private is True
    finally:
        # --- D. clean up: do not squat the release name with junk weights ---
        try:
            api.delete_repo(SMOKE_REPO, repo_type="model", missing_ok=True)
            result["phases"]["D_cleanup"] = "deleted"
            print("D: smoke repo deleted")
        except Exception as e:
            result["phases"]["D_cleanup"] = f"FAILED: {e!r}"
            print(f"D: could not delete the smoke repo: {e!r}")
        shutil.rmtree(work, ignore_errors=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "publish-live.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {out}  ok={result.get('ok')}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
