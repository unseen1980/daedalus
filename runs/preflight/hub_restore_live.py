"""Live end-to-end check of the Hub checkpoint durability path.

This is the half the offline suite cannot cover. `tests/test_ckpt_uploader.py`
and the Hub tests in `tests/test_train.py` fake `HfApi`, which means they never
exercise the three things most likely to fail in production:

  * **LFS.** Files above 10 MB go through git-lfs multipart upload; the tiny
    models the suite uses do not. A restore path proven only on a 5 MB file is
    not proven for a 321 MB one.
  * **Branch creation on a real repo**, including the second call that has to
    be a no-op.
  * **`hf_hub_download` into a clean directory** -- the actual call made on the
    day the box has been lost, rather than a monkeypatched stand-in.

Requires HF_TOKEN_WRITE and network. Not part of the default suite.

    python runs/preflight/hub_restore_live.py --repo Unseen1980/daedalus-checkpoints
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def load_env(path=".env"):
    """Read .env without printing any value -- never echo secrets (SS0.2)."""
    for line in open(path):
        m = re.match(r'^(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)=(.*)$', line.strip())
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--workdir", default="/tmp/hub-restore-live")
    args = p.parse_args()

    load_env()
    assert os.environ.get("HF_TOKEN_WRITE"), "HF_TOKEN_WRITE required"

    import torch
    from daedalus import ckpt_uploader as cu
    from train import TrainArgs, Trainer

    shutil.rmtree(args.workdir, ignore_errors=True)
    os.makedirs(args.workdir, exist_ok=True)
    report = {"repo": args.repo, "phases": []}

    def base(run_dir, run_name, config, **kw):
        return TrainArgs(
            run_name=run_name, config=config, device="cpu", compile=False,
            micro_batch=1, seq_start=16, seq_end=16, tok_start=16, tok_end=16,
            wandb_enabled=False, run_dir=run_dir, ckpt_every_sec=1e9,
            push_every_sec=1e9, log_every_steps=10 ** 9,
            metrics_every_steps=10 ** 9, hub_repo=args.repo,
            hub_uploader=False,  # the exit drain does the transfer, synchronously
            **kw)

    # -- Phase A: production-size rolling checkpoint, uploaded for real -------
    run_name = "hubsmoke"
    run_a = os.path.join(args.workdir, "a")
    print(f"\n=== A: train daedalus-150m 2 steps, stage + upload rolling bf16 ===")
    t0 = time.time()
    a = Trainer(base(run_a, run_name, "daedalus-150m", max_steps=2))
    a.fit()
    dt = time.time() - t0
    pending = cu.pending_uploads(cu.outbox_dir(run_a))
    ref = {k: v.detach().clone() for k, v in a.model.state_dict().items()}
    phase_a = {"phase": "A", "seconds": round(dt, 1),
               "step": a.step, "tokens_seen": a.tokens_seen,
               "pending_after_drain": len(pending)}
    print(json.dumps(phase_a, indent=2))
    assert not pending, f"outbox did not drain: {pending}"
    report["phases"].append(phase_a)

    # -- Phase B: restore from the Hub into a clean directory ----------------
    uri = f"hub://{args.repo}/rolling/{run_name}/weights.pt?rev=rolling"
    print(f"\n=== B: restore from {uri} into a clean directory ===")
    run_b = os.path.join(args.workdir, "b")
    t0 = time.time()
    b = Trainer(base(run_b, run_name + "-restored", "daedalus-150m",
                     max_steps=3, resume=uri))
    dt = time.time() - t0
    assert b.step == a.step, (b.step, a.step)
    assert b.tokens_seen == a.tokens_seen
    # bf16 round trip: equal to bf16's precision, not bit-identical fp32.
    worst = max(float((v - ref[k].to(v.dtype)).abs().max())
                for k, v in b.model.state_dict().items() if v.is_floating_point())
    worst_rel = max(
        float(((v - ref[k].to(v.dtype)).abs() / ref[k].abs().clamp(min=1e-6)).max())
        for k, v in b.model.state_dict().items() if v.is_floating_point())
    b.fit()  # and it actually keeps training
    phase_b = {"phase": "B", "seconds": round(dt, 1), "restored_step": b.step,
               "worst_abs_delta": worst, "worst_rel_delta": worst_rel,
               "trained_on_to_step": b.step}
    print(json.dumps(phase_b, indent=2))
    assert worst_rel < 2 ** -7, worst_rel
    report["phases"].append(phase_b)

    # -- Phase C: milestone on its own revision, with optimizer state --------
    # `tiny` here on purpose: this phase is about branch isolation and the
    # optimizer-state payload, and a full-size milestone is ~1.9 GB of
    # bandwidth that dataprep is currently sharing. Phase A already proved LFS.
    print("\n=== C: milestone -> own revision, restored with optimizer state ===")
    run_c = os.path.join(args.workdir, "c")
    c = Trainer(base(run_c, "hubsmoke-ms", "tiny", max_steps=10,
                     decay_frac=0.45, warmup_steps=1))
    c.fit()
    record = json.load(open(os.path.join(run_c, "milestone.json")))
    assert not cu.pending_uploads(cu.outbox_dir(run_c))

    ms_uri = (f"hub://{args.repo}/{record['path_in_repo']}"
              f"?rev={record['revision']}")
    run_d = os.path.join(args.workdir, "d")
    d = Trainer(base(run_d, "hubsmoke-ms-restored", "tiny", max_steps=11,
                     resume=ms_uri))
    # The point of the milestone: optimizer state comes back too, so a branch
    # does not have to rebuild Muon momentum and AdamW moments.
    muon_state = d.muon.state_dict()["state"]
    n_buffers = sum(1 for s in muon_state.values() if s)
    phase_c = {"phase": "C", "milestone_step": record["step"],
               "revision": record["revision"],
               "lr_mult_at_branch": record["lr_mult_at_branch"],
               "restored_step": d.step,
               "muon_entries_with_state": n_buffers}
    print(json.dumps(phase_c, indent=2))
    assert d.step == record["step"] == c.milestone_step
    assert n_buffers > 0, "milestone restored without Muon momentum buffers"
    report["phases"].append(phase_c)

    # -- Phase D: the branch revision is isolated from the rolling slot ------
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN_WRITE"])
    refs = api.list_repo_refs(args.repo, repo_type="model")
    branches = sorted(b.name for b in refs.branches)
    files_main = api.list_repo_files(args.repo, repo_type="model")
    phase_d = {"phase": "D", "branches": branches,
               "pointers_on_main": [f for f in files_main if f.startswith("latest-")]}
    print(json.dumps(phase_d, indent=2))
    assert "rolling" in branches and record["revision"] in branches
    report["phases"].append(phase_d)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "hub-restore-live.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nALL PHASES PASSED -> {out}")


if __name__ == "__main__":
    main()
