# The supervisor loop, rehearsed live — and the false positive it exposed

2026-08-11 ~03:50 UTC. CPU-only, `CUDA_VISIBLE_DEVICES=""`, in a scratch tree at
`/tmp/hero-rehearsal`. `abl-arch` arm 2 held the GPU throughout and was never
touched.

## Why this was worth doing

`supervise.run_with_resume` is the loop that keeps the ~5.6-day `hero` run alive
across crashes. `grep` says it is imported by `hero.py` and by nothing else —
`abl_arch.py` takes only `start_watchdog`/`stop_watchdog` — so **neither
`abl-arch` arm exercised it**. Every existing test injects `runner=`, which
means the one thing never executed was the default `runner`: a real
`subprocess.run`, a real SIGKILL, a real `--resume` handed to a real `train.py`.

That is the shape of every expensive bug this project has had. The WSD schedule,
the 65,536-character gate body, the mixture model that imitated the split
instead of calling it — green suite, unmeasured artifact.

## What was run

A real `train.py` under the real supervisor, SIGKILLed mid-run, on a tiny
corpus in the mixture-root shape `hero` passes (`data/train/{fineweb-edu,
dclm-baseline}`, per-source manifests).

Deviations from `hero`'s command, each deliberate and each stated:

| deviation | why |
|---|---|
| `--config tiny`, `--seq-start 128 --seq-end 256`, small token ramp | `tiny.max_position_embeddings` is 256, so hero's 1024→2048 default cannot apply. CPU-only. |
| `--hub-repo ""` | the Hub path is already proven live on both arms; a rehearsal must not push junk into the repo `hero` resumes from |
| `backoff_sec=5` instead of 60 | sleeping is not the thing under test |
| `WANDB_PROJECT=daedalus-rehearsal` | W&B stays **on** — the restart path is exactly what is being tested — but off the dashboard the operator reads |

## Result: the loop works, including a case I did not set out to test

```json
{"report": {"attempts": 3, "resumed": true, "returncodes": [-9, -9, 0]},
 "final_step": 381, "final_tokens": 6002688, "target_tokens": 6000000}
```

Attempt 1 was SIGKILLed at step 110 with the rolling checkpoint at step 1. The
supervisor logged `attempt 1 exited -9; retrying in 5s`, relaunched with
`--resume runs/hero-rehearsal/checkpoint.pt`, and `train.py` printed
`resumed from …: step=1 tokens_seen=8192` — it replayed the 109 lost steps and
went on to the target. **W&B re-attached to the same run id** (`d93ef4d1
(resumed)`) rather than opening a second run.

The second kill was accidental and is the more interesting one. It landed after
attempt 2 had reached the target and written its final forced checkpoint, but
before the process exited. Attempt 3 resumed from that completed checkpoint,
saw the target already met, and **exited 0 without retraining anything**. A
crash in the last seconds of `hero` therefore costs seconds, not a re-run.

Cost of a crash, measured: everything since the last checkpoint is replayed.
Here 109 steps / 1.55M tokens. In production the rolling gate is 30 minutes, so
the worst case is ~30 minutes of GPU (~$0.22).

## The bug this exposed

Getting the resume to work is what made the next problem visible.

`IntervalGate.ready()` fires on its **first** call (`train.py:708`, `last is
None`), so the first rolling checkpoint of any run is written at **step 1**, and
the second 30 minutes later. A hard crash inside that first half hour therefore
resumes correctly from step 1 — and then `watchdog.detect_divergence` compares
the resumed loss against the running mean of the records from just before the
crash, because `read_metrics` reads the whole file and metrics.jsonl spans
attempts.

Replayed against `runs/abl-arch-daedalus-150m/metrics.jsonl`, the real curve:

| quantity | real value |
|---|---|
| loss the resumed attempt reports at step 20 | **9.3150** |
| running mean of the 20 records before a 30-minute crash | **3.7088** |
| divergence threshold (`2.0 x` mean) | 7.4176 |
| verdict | **9.3150 > 7.4176 → divergence** |

Divergence is deliberately terminal. `check_once` SIGTERMs the trainer
(`watchdog.py:311`), writes a `WATCHDOG HALT` to `STATUS.md`, and writes the
halt marker; `run_with_resume` reads that marker and **refuses to resume past
it** (`supervise.py:146-153`), by design — retrying is the right answer to a
crash and the wrong one to a real divergence.

So: `hero` crashes at minute ~25, recovers correctly, and is then killed by its
own watchdog at minute ~26 of a 5.6-day run, with `STATUS.md` blaming the
learning rate. The box bills $0.449/h until someone notices — at a 07:52Z launch
that is a night. The exposure window is roughly from where the loss first falls
below ~4.66 (early — arm 1 was at 3.40 by 0.52 h) until the second checkpoint at
t+30 min.

This is not hypothetical for this project: arm 2 OOM'd twice at startup on
2026-08-10, which is exactly a crash in the opening minutes.

## Fix

`watchdog.records_since_resume` scopes the loss history to the current attempt
by finding the most recent point where `step` goes backwards — which is what a
resume *is* in metrics.jsonl. `detect_divergence` builds its window from that
tail only.

Truncating rather than widening `factor` keeps a genuine divergence terminal:

- the **NaN/inf check is upstream of any history**, so the dominant real failure
  mode is still caught on the first record, resume or no resume;
- a real blow-up inside one attempt still trips on the very next record;
- immediately after a resume the detector reports "not enough history to judge"
  for 5 records and then re-arms — the same doctrine the docstring already
  stated ("returns None too early to tell, not a false positive").

## Evidence

Eight tests in `tests/test_watchdog.py`, two of them verified failing against
the pre-fix file with exactly the predicted message:

```
tests/test_watchdog.py::test_a_resume_is_not_scored_against_the_curve_it_rolled_back_from
E  assert 'loss diverged at step 20: 9.3150 > 2.0x running mean 3.7088 (over last 20 points)' is None
tests/test_watchdog.py::test_a_resumed_run_is_not_halted_by_check_once
E  assert {'kind': 'divergence', ...} is None or 'divergence' != 'divergence'
```

`test_the_real_abl_arch_arms_still_never_trip` replays `detect_divergence`
record by record over **both** completed arms — 519 and 419 real records,
including arm 2's two genuine resume boundaries at step 1600 → 1580 — and
asserts it never fires. That is the regression floor: neither arm tripped before
the fix and neither may after it.

61 tests in `tests/test_watchdog.py`, 142 across
`test_supervise`/`test_hero`/`test_hero_gate_safety`, all passing.

## Checked and clean, in passing

- **`--hub-repo` reaches `hero`.** `build_train_cmd` never passes it, which
  would have silently voided blocking precondition 1; it survives because
  `train.py:1556` defaults it to `ckpt_uploader.DEFAULT_MODEL_REPO`.
- **Concurrent uploaders cannot race.** `ckpt_uploader` takes a PID-file lock on
  the outbox with a liveness check (`ckpt_uploader.py:375-406`) and the incumbent
  keeps it, so the up-to-10 restarts `hero` allows cannot produce competing
  uploaders. Arm 1's uploader is still alive 11 h after its trainer exited — an
  orphan from the 18:31Z kill, not a leak in the normal path, and harmless.
- **`save_checkpoint` is atomic** (`train.py:155-157`, tmp + `os.replace`), so a
  SIGKILL during a checkpoint write cannot leave the file `hero` resumes from
  truncated.
- **`detect_stall` was already resume-aware** — it clocks from the newer of
  `metrics.jsonl` and `train.pid`, and takes the watchdog's own start time, both
  documented as restart handling. Only the divergence path had the gap.

---

## Two more, from asking the same question of the rest of attempt two

### The gate's *measured* `abl-arch` cost dropped every attempt but the last

`cost_usd` and `elapsed_h` come from `Trainer._elapsed_h`, which clocks from the
process's own start, so a restart resets both to zero.
`gate_body._render_abl_cost` read the last metrics line.

| arm | last line | actually metered | dropped |
|---|---|---|---|
| `abl-arch-daedalus-150m` (1 attempt) | $5.09 / 11.33 h | $5.09 / 11.33 h | — |
| `abl-arch-dense-150m` (**3 attempts**) | $4.12 / 9.20 h | **$4.64 / 10.32 h** | **$0.52 / 1.09 h** |

In a row the gate labels *measured*, in the direction that makes a budget look
affordable, in the document asking for $59.85. `hero` allows ten attempts over
5.9 days, so it compounds there.

**`COSTS.md` was never affected.** It prices from wall clock since the first
commit, on the explicit principle that the balance loses every rented hour
whether or not it is attributed to a job. That was the right basis, and it is
why the funding number was sound while the attribution was not.

Writing the test caught a second bug, in my own first implementation of the fix:
segmenting on *the counter* going backwards merges an attempt that died after
two minutes ($0.02) with the next one, which climbs straight past it without
ever dipping. It reported 2 attempts where the fixture had 3. The boundary has
to be **`step` going backwards** — the same signal as the watchdog fix. Pinned by
`test_metered_totals_sums_every_attempt`.

### Nothing ever checked the Hub uploader was still alive

Hard precondition 1 is weights-only uploads on a ~2 h cadence, out-of-band.
`_start_uploader` runs once at startup and **no call site reads
`self.uploader_proc` again** until `_stop_uploader` at exit. An uploader that
died mid-run — OOM-killed, or crashed on a payload — therefore took the run's
insurance with it, silently, for the rest of six days.
`_warn_if_hub_stalled` would eventually print, but a warning inside a 5.9-day
log is not a repair, and `AGENT.md` §0.2 is blunt about the consequence.

`Trainer._ensure_uploader` respawns on the same 2 h gate as the upload it
protects, so detection latency matches the cadence.

The check has to be `ckpt_uploader.uploader_is_live(outbox)` and **not**
`self.uploader_proc.poll()`. After any crash the process actually delivering is
the orphan the previous attempt left holding the outbox lock — `acquire_lock`'s
own docstring says the incumbent keeps it — and *this* attempt's uploader exited
on that lock immediately and by design. So the child handle reads "dead" for the
rest of the run while uploads are entirely healthy, and respawning on it would
fork a redundant uploader every two hours for six days. Arm 1's uploader, still
alive 11 h after its trainer was killed, is that orphan in the flesh.

`uploader_is_live` is read-only on purpose: a trainer that stamped its own pid
into the lockfile would lock out every real uploader for the rest of the run.
Also tested.

11 tests, all verified failing on the pre-fix files.
