# `hero.py` run end to end before the four-day job

`hero.py` drives the largest line in the budget (~$43.7, four days) and had
never been executed — only unit-tested with mocked subprocesses. Smoke-run
2026-08-09 ~20:35-20:40Z against a synthetic two-source mixture in `/tmp`,
with **production flags on** (W&B enabled, `torch.compile` on, CUDA) — a
W&B-disabled validation has hidden real fork-safety bugs on this project
before.

## Result: exit 0, one attempt, full path exercised

`daedalus-150m`, 1M tokens, micro-batch 2. Verified:

- the per-source holdout split is carved from `--mixture-dir` into
  `--split-root`;
- `train.py` is launched as a subprocess with the right argv, including
  `--val-dir` pointing at the holdout root and `--qat-frac 0.05`;
- `watchdog.py` starts alongside and is **terminated by the `finally` block** —
  no stray process left behind (checked by name after the run);
- the run's W&B run appears (`0l27na4j`);
- `note_in_status` prepends a dated line to `STATUS.md` on success.

The failure path was exercised too, by accident and usefully (below): on
terminal failure it prepends **"stopped after 2 failed attempts (exit codes
[1, 1]). Checkpoint at ...; nothing was deleted. The box is idle until this is
looked at."** That is the "silence is the expensive failure mode" path working.

## What the smoke run does *not* cover, and why

At 1M tokens the run is ~8 optimizer steps (tok/step ramps 128K→512K), which is
below `val_every_steps=500`, `metrics_every_steps` and the final-5% QAT trigger
— so no `metrics.jsonl`, no val, no QAT. **`hero.py` cannot forward those
cadence flags**, so it cannot be made to exercise them at small scale; reaching
500 steps needs ~64B tokens. Covered instead by driving `train.py` directly
(below) and by unit tests.

## `val_bpb` on a mixture holdout root, live

The fix committed earlier today, validated outside its unit tests — `train.py`
run directly with `--val-dir` on a real mixture root, `--val-every-steps 1`:

| step | loss | val_bpb |
|---|---|---|
| 1 | 15.1913 | 8.1371 |
| 2 | 15.1911 | 8.1349 |
| 3 | 15.1726 | 8.1258 |

Non-null and decreasing, with no `val_bpb failed` warning. Before the fix every
one of these would have been `null`. (The absolute value is meaningless — the
synthetic shards are uniform random token ids — but the mechanism and the trend
are the point.) Initial loss 15.19 also matches the `hidden * init_std = 15.36`
prediction recorded in `dense-150m-trains.md`.

## Bug found: a too-short RoPE cache failed unreadably

The first smoke attempt used the `tiny` preset, whose
`max_position_embeddings=256` gives a 1024-position cache while `train.py`'s
sequence ramp ends at 2048. The run trained fine at seq 1024, checkpointed,
then died when the ramp stepped — with a `torch._dynamo` fake-tensor broadcast
error ("Attempting to broadcast a dimension of length 1024 at -2") naming
neither the config nor the sequence length, because `cos[off:off + T]` past the
end silently returns a *shorter* tensor instead of raising.

Both production presets are safe (`max_position_embeddings=2048` → 8192-position
cache), so this was never going to hit `hero` — but it cost real minutes to
diagnose from the message alone. `Daedalus.forward` now raises a `ValueError`
naming the cache length, the required length, the offset and the config value.
Pinned by a test.
