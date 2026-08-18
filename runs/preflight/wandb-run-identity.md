# A `hero` restart would have stranded the operator on a dead dashboard

`supervise.run_with_resume` exists because a ~92 h run is expected to be
interrupted — it relaunches `train.py --resume` up to 10 times. W&B mints a
**fresh random run id on every `init()`**; the run *name* is only a display
label. So every restart started a new run at a new URL.

The URL published in `STATUS.md` and in the `[ASK HUMAN] ready for hero` issue
is the one the operator watches from a phone. After a crash it would have sat
frozen at the crash point while training carried on, unseen, somewhere else —
and **a frozen dashboard is indistinguishable from a dead run**, which is the
alarm W&B exists to raise (AGENT.md §5.1).

This is the same failure mode fixed this morning for dropped packets, reached by
a different route. That one was probabilistic; this one was **guaranteed on any
restart**.

## The bug, measured rather than reasoned about

Two `wandb.init()` calls with identical project/name/tags, real client:

```
RUN IDS: ['ma562epj', 'vlr1n10i']
SAME RUN? False
```

`wandb_logger.py:55` passed no `id` and no `resume`. Nothing in the test suite
pinned the init arguments, so the omission was invisible.

## The fix, and the case it deliberately does *not* cover

`resolve_wandb_run_id(run_dir, resumed)` persists the id to
`runs/<run>/wandb-run-id.txt` and re-attaches with `resume="allow"` — but only
when **this process actually resumed a checkpoint**, not merely when the run
name matches.

That distinction is not hypothetical. `sweep` was thrown out and re-run under
the same run names after the WSD bug. Keying off the name would have appended
the good curve to the discarded one **at the same step numbers**, producing a
single chart that silently interleaves two different experiments — a worse
outcome than the bug being fixed.

Resume with no id file (run dir rebuilt, or a checkpoint restored from the Hub
onto a clean box — the disaster-recovery case) cannot re-attach to an unknown
id. It mints a fresh one and says so, rather than failing.

## Verified in a real process, not only against the fake

The fake `wandb` module cannot tell us whether *this* client version accepts
`id` + `resume="allow"`. Two real `train.py` subprocesses, tiny config on CPU,
W&B enabled in offline mode, second one resuming the first:

```
first rc=0
second rc=0
runs/e2ewb/wandb-run-id.txt -> ae12d846
wandb/offline-run-20260810_072314-ae12d846
wandb/offline-run-20260810_072329-ae12d846   <- same id, not a sibling run
resumed from runs/e2ewb/checkpoint.pt: step=4 tokens_seen=128
W&B: re-attaching to run id ae12d846 (resumed)
```

Both processes exited 0, so the kwargs are accepted by the installed client.

## What it does not fix — checked against the real backend, not assumed

A resumed run re-logs the steps between the checkpoint and the last point W&B
received (checkpoints are 30-minutely, metrics 20-steply). Step-ordering is
enforced by the *backend*, not the client, so this could not be settled offline
or against the fake. A throwaway online run — log steps 1–10, resume the same
id, log steps 5–15:

```
resumed flag: True
history rows: 15
[(1,1.0), (2,0.5), ... (10,0.1), (11,99), (12,99), (13,99), (14,99), (15,99)]
```

The overlap (5–10) **keeps its original values** — the re-logged 99s are
discarded — and logging continues cleanly once past the previous maximum. So a
restart shows as a brief flat spot in a **continuous** history, not as a fork
and not as corruption. That is the intended trade and is documented in
`resolve_wandb_run_id`. Probe run deleted afterwards; it ran in a scratch
project so the operator's `daedalus` dashboard was never touched.

6 tests in `tests/test_train.py`, including the recycled-name counter-case and
an assertion that callers passing neither kwarg do not start sending `id=None`.
**788 passed, 7 skipped** (the 7 are QAT CUDA probes, skipped because `abl-arch`
owns the GPU).

## Then smoked on the real path, because arm 2 loads this file at ~17:13Z

Unit tests and a `tiny`-config subprocess both avoid the production shapes. A
final check ran `train.py` against the **real 9-source mixture root** at the real
`daedalus-150m` config, CPU, 2 steps, cwd in `/tmp` so nothing could touch the
repo's `runs/`:

```
rc=0
MixtureBatchSource: mixture L1 skew 6.10 pts from target; capped: [...]   <- no WARNING, correctly under 10.0
runs/mixsmoke/wandb-run-id.txt -> efcd1566
[milestone] end of stable phase at step 1 -> revision mixsmoke-stable-end-step1
ckpt-uploader: milestone/mixsmoke/checkpoint.pt  (1435 MB)
ckpt-uploader: rolling/mixsmoke/weights.pt       (321 MB)
```

Worth noting the skew figure: **6.10 pts on the holdout-split root**, against
3.99 on the full corpus, because the split holds 2% back and drops
`everyday-conversations` entirely. Still comfortably under the threshold, which
is the behaviour wanted — the guard fires on a collapsed mixture, not on a
holdout carve. The `git commit/push failed` lines in that log are the `/tmp`
isolation working, not a fault.


`abl-arch` arm 1 is running the old code; arm 2 launches a fresh `train.py` and
picks this up. It lands for `hero`, which is where restarts are expected.
