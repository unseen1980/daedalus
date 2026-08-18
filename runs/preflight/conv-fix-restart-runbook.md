# Restarting `hero` with the conv-death fix — the runbook

Written 2026-08-11 ~17:50 UTC, while issue #8 is open and **before** any decision,
so that a "go" costs no implementation latency and, more importantly, so the
hazards are found now rather than during a stop/start on a live $63 run.

Nothing here has been executed against `hero`. The service behaviours below were
read out of the running configuration and the scripts themselves.

## The hazards, found by reading the live system rather than assuming

Four things would fight a naive `kill <trainer pid>`:

| # | what | why it bites |
|---|---|---|
| 1 | **`hero.py` (pid 1133711) is a supervisor** | `run_with_resume` relaunches `train.py` on a non-zero exit, `max_attempts: 10` with backoff. Kill the trainer alone and it comes straight back — on the **old** config. Kill the supervisor *first*. |
| 2 | **`daedalus_after_hero` (RUNNING)** | It is waiting for the run to end. A deliberate stop reads as `INCOMPLETE`, and it re-checks 4× at 900 s. If nothing is running again within **1 hour** it exits 1 and the after-run insurance is gone for the rest of the project. |
| 3 | **`runs/hero/checkpoint.pt`** | Both recovery paths resume from it (`supervise.py:311`, `abl_arch.py:110`). A relaunch into the existing run dir is a *resume*, not a restart. |
| 4 | **`inflight.json` has `completed: false`** | `scripts/boot_resume.py` re-enters an incomplete run on box restart. Left as-is it would resurrect the abandoned run. |

Hazard 3 fails **loudly**, which is the one piece of luck here: `--conv-proj-wd`
adds a Muon parameter group, so resuming an old checkpoint with the flag raises
`ValueError: loaded state dict has a different number of parameter groups`, rc 1.
It cannot silently continue the unfixed run. Hazards 1, 2 and 4 are all silent.

Note also that `after_hero`'s `JOB_PAT` is
`bin/python[0-9.]* (hero|train)[.]py` — it matches **any** `train.py`, not just
`--run-name hero`. So the validation run in step 2 below looks like `hero` to it.
Benign in this direction (it keeps waiting), but it is why the chain is stopped
explicitly rather than relied on to reason correctly.

## Step 1 — stop, in this order

```bash
supervisorctl stop daedalus_after_hero          # hazard 2, BEFORE anything else
kill 1133711                                    # hero.py supervisor  (hazard 1)
sleep 5
kill 1133733 1133734                            # watchdog, then train.py
# confirm nothing came back:
pgrep -af "bin/python[0-9.]* (hero|train)[.]py"  # must print nothing
```

Then close out the abandoned run so nothing resurrects it, and keep its evidence:

```bash
$DAEDALUS_PYTHON -c "import json,io; p='runs/hero/inflight.json'; \
d=json.load(open(p)); d['completed']=True; d['outcome']='abandoned-for-conv-fix'; \
json.dump(d, open(p,'w'), indent=2)"                      # hazard 4
mv runs/hero runs/hero-nofix-abandoned                    # hazard 3
```

`mv` rather than `rm`: those 7.3 h of `metrics.jsonl` are the only 150M-scale
loss curve this project has under the shipped optimizer split, and they are the
control the fix gets compared against.

## Step 2 — validate at real scale, against a baseline that already exists

**This is the step that decides whether the relaunch carries the fix.** The fix
has never trained a 150M model; only a 9-block/hidden-256 CPU probe.

The cheap, *paired* way to check it is not a fresh experiment. `sweep-wsdfix-lr0.04`
already measured **15.4% dead at 1,040 steps / 0.50B tokens** on `daedalus-150m`
(`runs/preflight/conv-channel-death.md:39`), and its split still exists on disk
(`data/shards-sweep-split/train`, 3 shards). So re-run *that arm* with the fix and
nothing else changed:

```bash
$DAEDALUS_PYTHON train.py \
  --run-name convfix-lr0.04 --config daedalus-150m \
  --data-dir data/shards-sweep-split/train \
  --total-tokens 500000000 --muon-lr 0.04 \
  --tags convfix --hub-repo "" \
  --conv-proj-wd 0.0133
$DAEDALUS_PYTHON scripts/conv_death_watch.py --run-dir runs/convfix-lr0.04
```

- lr **0.04**, not `hero`'s 0.02, deliberately: at 0.02 the shipped config still
  reads 0.0% at 1,040 steps, so a 0.02 arm could not tell a working fix from a
  clock that has not run yet. 0.04 is the lowest lr on this project's own grid at
  which the phenomenon is *visible* at 0.5B tokens.
- ~68 min, **~$0.51**, one arm. The comparison arm was paid for in `sweep`.
- **Pass:** dead fraction ≈ 0% where the paired baseline read 15.4%.
- **Fail:** materially above 0 → relaunch `hero` **unchanged**, having spent
  $0.51 to avoid a 6-day mistake.

## Step 3 — relaunch

```bash
$DAEDALUS_PYTHON hero.py --muon-lr 0.02 --micro-batch 16 \
    --total-tokens 59900000000 --conv-proj-wd 0.0133      # omit the flag on a FAIL
supervisorctl start daedalus_after_hero
$DAEDALUS_PYTHON scripts/verify_hero_launched.py
```

Both flags are confirmed present on the real parsers: `hero.py --help` and
`train.py --help` each list `--conv-proj-wd`, defaulting to `None` = the shipped
single-Muon-group split.

## What does not change, and one thing that does

Unchanged: 59.9B budget, 124,476 steps, decay/milestone at 68,461, QAT final 5%,
mixture, Hub cadence. The schedule derives from the budget, which is untouched.

Changed: **a new W&B run**, so the dashboard link in `STATUS.md` and issue #8 must
be repointed, and the pre-restart curve lives under `hero-nofix-abandoned`. The
conv-death sampler (`daedalus_conv_death`) starts a fresh `conv-death.jsonl` — and
that file is now the *primary evidence* that the fix works at scale over a full
run, which the 6,000-step CPU probe can only foreshadow.

## Cost

$3.29 sunk at 17:41Z, growing at $0.449/h, plus $0.51 for step 2. Credit at
`hero`'s end ≈$27.85 after the $5.50 tail (≈$31.06 if we do not restart).
