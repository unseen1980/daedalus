# `hero` rehearsed end to end at 40M tokens — 2026-08-11 08:08–08:27Z

## Why

`hero._cli`'s tail — `build_train_cmd` → `start_watchdog` → `run_with_resume` —
had never run in sequence. The prefix was executed in isolation at 06:38Z
(`hero-cli-prefix-executed.md`), the supervisor loop was rehearsed on CPU
(`supervisor-rehearsal.md`), and `build_train_cmd` is unit-tested; the three had
never been run *by the launcher*, one after the other, on the real GPU.

The gate was posted and the box was idle waiting for a reply, so this cost
~$0.14 of otherwise-idle rent and used the exact production configuration:
`daedalus-150m`, Muon lr 0.02, micro-batch 16, `--qat-frac 0.05`, watchdog on,
W&B on, Hub uploads on. Only `--total-tokens` (40M) and `--run-name`
(`hero-rehearsal`) differed. Launched detached, exactly as the gate instructs.

## What it proved

**The launcher path runs.** All three preflight gates passed against live state
and the trainer started:

```
[hero] mixture preflight: 16,932,674,383 tokens on disk, l1_skew 0.00 pts …
[hero] QAT preflight passed (runs/preflight/qat-gate-evidence.md)
[supervise] watchdog pid 1112786: … watchdog.py --run-name hero-rehearsal …
[supervise] attempt 1/2: … train.py --run-name hero-rehearsal --config daedalus-150m …
```

This is also the first real use of the QAT evidence file the chain wrote at
07:39Z — the gate's blocking precondition 2, consumed by the code that enforces
it rather than asserted in a document.

**`verify_hero_launched.py` works on a real launch.** Written earlier today
against synthetic logs; run here against an actual detached `setsid nohup`
launch it returned `OK: a trainer for …/runs/hero-rehearsal is alive (pid
1112787)`, rc 0. The shell had reported rc 0 for the backgrounded job before the
trainer existed, which is the whole reason the step was added.

**QAT activates on the token boundary, and its memory cost is now measured on
the real path.** This is the headline: QAT is on for `hero` and was off for
*both* `abl-arch` arms, so the only prior evidence was a CPU estimate.

| step | tokens | % of budget | `qat_active` | peak_mem_GB |
|---:|---:|---:|---:|---:|
| 20 | 6,533,120 | 16.3% | 0 | 24.29 |
| 40 | 17,018,880 | 42.5% | 0 | 24.29 |
| 60 | 27,504,640 | 68.8% | 0 | 24.29 |
| **80** | **37,990,400** | **95.0%** | **0** | **24.29** |
| **84** | **40,087,552** | **100.2%** | **1** | **25.95** |

- The switch is **token-derived**, not step-derived: still off at 95.0%, on by
  the next logged row. That is what the gate's schedule table claims.
- **24.29 → 25.95 GB is +1.66 GB.** The projection in `STATUS.md` was
  *"24.29 + 1.65 = ~25.9 GB"*, derived from a batch-1 CPU measurement on the
  argument that fake-quant's cost is in weights rather than activations and so
  does not scale with micro-batch. **Measured delta +1.66 GB against +1.65
  predicted.** `hero`'s final 5% has **~6.65 GB of headroom** on a 32.6 GB card.
- 24.29 GB before activation also reproduces `abl-arch` arm 1's peak exactly,
  at the same config and micro-batch.

**The milestone fires where it should.** `decay_frac` 0.45 → branch at 55% of
84 steps = 46.2 → written at **step 46**, revision
`hero-rehearsal-stable-end-step46`, with full optimizer state (1.44 GB) and the
`Unseen1980/daedalus-checkpoints` repo resolved from `train.py`'s default
without `build_train_cmd` passing `--hub-repo`.

**The watchdog called completion correctly**, on tokens rather than steps:
`completion -- reached target: 40,087,552 / 40,000,000 tokens at step 84`.

## What it did NOT prove — read this before quoting the above

**The run never left warmup, so there was no stable phase and no decay.** Warmup
is 300 steps; this run was 84. The Muon lr climbed monotonically and finished at
its highest value:

| step | 20 | 40 | 60 | 80 | 84 |
|---|---|---|---|---|---|
| lr | 1.267e-3 | 2.600e-3 | 3.933e-3 | 5.267e-3 | **5.533e-3** |

5.533e-3 is `(83/300) × 0.02` — pure warmup ramp. So:

- **The WSD decay was not exercised here at all.** It is covered by tests at the
  real budget (final-step lr multiplier 0.00002 at 58B), and that remains the
  only evidence for it. Nothing in this rehearsal supports or contradicts it.
- The milestone's `lr_mult_at_branch` reads **0.153**, not ~1.0, for the same
  reason: at step 46 of 300 warmup steps the schedule is still ramping. The
  *step* it chose (46 = 55%) is correct; the "end of stable phase" **semantics**
  are meaningless at this budget. At `hero`'s 120,528 steps, warmup is 0.25% of
  the run and the branch point sits deep in the stable phase.
- 84 steps of a batch ramp also make `tok_per_sec` (52,676) meaningless. It is
  not a throughput measurement and must not be compared with the 117,325 the
  budget is built on.

## What it found — Hub upload is slow, and my first reading of it was wrong

**Correction, written after the first version of this note.** I originally
recorded *"the trainer reached its target and did not exit — a hang on hero's
critical path"*. **That conclusion was wrong and the claim is withdrawn.** What
I saw was real; what I concluded from it was not, because I did not check
whether the drain was bounded before calling it a hang. It is:

```
daedalus/ckpt_uploader.py:87   DEFAULT_PASS_DEADLINE_S = 900.0
```

`_stop_uploader` drains through `upload_once_bounded`, which runs the pass in a
**child process** precisely so a wedged transfer can be SIGKILLed, and the
module docstring documents the identical 2026-08-10 wedge (socket in CLOSE-WAIT,
blocking send, SIGTERM ignored because CPython cannot run a handler while the
main thread sits in a C-level socket call). The two `--once` processes I found
and reported as *"duplicate uploaders spawned by `_ensure_uploader`"* are that
mechanism working: `upload_once_bounded` spawns exactly that command
(`ckpt_uploader.py:316`).

**I killed it at 08:27:18, ~70 seconds before its own 900 s deadline would have
fired.** Left alone it would have SIGKILLed the transfer, logged the failed pass,
left the payload in the outbox for the next attempt, and exited. The
insurance behaved as designed; I interrupted it and mistook the design for the
defect.

Two things remain unresolved and are recorded as unknowns rather than findings:
the reason **two** bounded passes overlapped (08:13:25 and 08:14:27) when
`subprocess.run` blocks its caller, and whether the second was a leftover child
of the terminated daemon. I destroyed the evidence by killing the processes, so
this cannot be settled from here.

### What is still true, and measured

The transfer really was slow, and that part is worth carrying forward.

**Upstream to the Hub is very slow and times out.** From the Xet client's own
log (`/workspace/.hf_home/xet/logs/xet_*_1113230.log`):

```
predicted bandwidth = 223251            # bytes/s -- ~223 KB/s, ~1.8 Mbit/s
Retryable Client Error: cas::upload_xorb api call failed (retry 1)
Request error … source: TimedOut
Concurrency control: Decreased concurrency from 1 to 1; reason: transfer failed
```

Measured directly: **0 bytes transmitted over 60 s** while two uploaders were
alive, and the last Xet log line was 9 minutes stale. Progress had been real
earlier (162 MB of the milestone sent, xorbs returning 200) — so this is a stall
under retry, not a dead path. At ~223 KB/s a 321 MB rolling checkpoint is
~24 min and the 1.4 GB milestone is ~1.8 h.

### How much of this matters for `hero`

| | severity |
|---|---|
| slow uploads **during** the run | **low** — uploads are out-of-band by design, so the training loop is unaffected, and staleness is exactly what `scripts/hub_watch.py` alarms on. A 2 h cadence absorbs a ~24 min upload |
| the 1.4 GB milestone at 55% | **low** — ~1.8 h at the rate measured tonight, out-of-band; it does not block training. Both `abl-arch` milestones are already on the Hub, so the path works |
| the end-of-run drain | **low** — bounded at 900 s, in a killable child, by design. Worst case it adds ≤15 min to a 133 h run and the payload waits in the outbox |

**None of this is a reason not to launch**, and the honest summary is the
opposite of my first draft: the one part of the system that had been built
specifically for a wedged upload is the part that behaved correctly under one.

The single number worth carrying into `hero` is the **~223 KB/s upstream**. If
it holds for six days, a 321 MB rolling checkpoint takes ~24 min of a 2 h
cadence — comfortable but not generous, and `hub_watch`'s staleness threshold
(3× cadence) is the right place to catch it degrading further.

## Cost and cleanup

~19 minutes of otherwise-idle box time, ~$0.14. The 3.0 GB of local artifacts
(checkpoint + outbox payloads) were deleted; `metrics.jsonl` and `milestone.json`
are kept as the evidence above. One artifact was **not** removed: the
`rolling/hero-rehearsal/weights.pt` branch (321 MB) reached
`Unseen1980/daedalus-checkpoints` before the stall. Deleting from the operator's
account is their call, and 321 MB against 42.6 GB free changes nothing.
