# What should happen when `abl-arch` arm 1 exits (~16:32Z)

Written 2026-08-10 13:00Z, with arm 1 at step 7,260 / 67.2%, so the check at
~16:32Z is a comparison and not a rationalisation. Two pieces of code run in
production for the first time tonight — `scripts/guard_exit_drain.py`, and the
eval waiter's new 99.5% completion tolerance — and both are *quiet* when they
work, which is exactly the shape that gets misread later.

## The arithmetic everything below turns on

Arm 1's loop breaks at **step 10,391** (5,000,083,456 tokens ≥ 5,000,000,000).
10,391 is **not** a multiple of `metrics_every_steps=20`, and arm 1 is running
the pre-fix `train.py` from 05:13Z, so **no final metrics row is forced**. The
durable record therefore ends at:

| | |
|---|---|
| last metrics row | **step 10,380** |
| tokens on that row | **4,994,316,288** |
| shortfall vs target | 5,683,712 tokens (**0.11%**) |
| waiter floor (99.5%) | 4,975,000,000 — **cleared** |

Arm 2 launches fresh and *does* get the forced row, so it should end at step
10,391 / 5,000,083,456. **That difference between the two arms is expected and
is not an asymmetry in the training.**

## Predicted sequence

1. **~16:31Z** — step 10,380 metrics row lands. From this moment the guard's
   token condition is satisfied while training is still running.
2. **~16:32Z** — 11 more steps (~47 s at 124k tok/s), loop breaks at 10,391.
   Throughout, the GPU is at ~97%, so the guard reads **`busy`** and its clock
   stays reset. *This is the load-bearing part of the design: the token test
   alone would already be true here.*
3. Forced checkpoint (~7.4 s), stage to outbox, `maybe_push`, `wandb.finish()`.
   GPU now idle → the guard flips to **`SUSPECT`** and starts its 12-minute
   clock. **A `SUSPECT` line is expected on the healthy path and is not a
   problem.**
4. `_stop_uploader` → the **unbounded** `upload_once` (arm 1 predates f99dd5a).
   ~321 MB, ~50 s at the 53 Mbit/s seen this morning.
5. Trainer exits 0, ~90 s after step 10,391. The guard's next poll sees no pid
   and prints **`guard: exited -- no trainer for this run is alive`**, rc 0.
6. The waiter sees no trainer, reads 4,994,316,288 ≥ 4,975,000,000, breaks its
   wait loop, checks ≥6 GB RAM free, and runs `eval.py` on arm 1's checkpoint →
   `runs/eval/ours-abl-arch-hybrid-5B.json`.
7. `abl_arch` runs its own full-holdout `eval_val_bpb` (the number the ablation
   is decided on) and then `export_and_bench`, now at **depths 0 / 512 / 2048**
   — verified today that `abl_arch.py:248` imports `export` lazily, so the
   running process picks up today's `DECODE_DEPTHS` rather than the depth-0
   default it started with.

## What each failure would look like, and what to do

| symptom | reading | action |
|---|---|---|
| `guard: WEDGED ... halted pid N` | the drain hung; the guard beat the 30-min stall marker | none — `abl_arch` retries with `--resume`, that process has the bounded drain and exits in seconds |
| `guard: blocked` | a halt marker already exists, so a kill cannot help | investigate by hand; the chain will **not** retry the arm |
| guard exits 0 with no `SUSPECT` line | trainer exited before the first idle poll | fine, just faster than predicted |
| waiter logs `... < 4975000000; waiting` | the tolerance is still too tight, or the run really did die short | check `tokens` on the last row before concluding |
| waiter logs `arm 2 has started` | the window closed before the eval finished | redo the eval during the gate wait, when the GPU is genuinely free |

## If the guard fires, `results.json` will carry a caveat that does not apply

This is the one way tonight could cost real money through a *wrong reaction*
rather than a wrong outcome, so it is written down before the fact.

`abl_arch.run_training` sets `resumed=True` on any retry and attaches:

> *"this arm was resumed after a failure, so from the restart point it drew a
> different data stream than an arm that ran straight through; the head-to-head
> is no longer byte-identical in data order"*

and the module docstring goes further — **"if an arm dies, restart both from
scratch"**. Following that after a guard kill would throw away a perfect 11.3 h
arm and ~$5.40 to re-run it, and then do the same to arm 2.

**The caveat does not apply to a guard kill, and the reason is structural.** The
guard only acts once the token target is already met, so the retry's `fit` breaks
at the top of its first iteration and trains **zero steps** — it draws no
batches, so `set_position` repositions nothing that is ever sampled. The weights,
the data order and the val_bpb are those of an uninterrupted run; only the
post-training upload was interrupted. Pinned by
`test_resume_from_an_already_complete_checkpoint_exits_cleanly`, which asserts
the resumed trainer's step count is unchanged.

Note `abl_arch.py` has been running since 00:19Z, so **editing it now would not
change tonight's behaviour** — it is the in-memory copy that runs. Hence a note
here rather than a code change. If the guard fires, read `results.json`'s
`resumed: true` together with `/tmp/guard-exit-drain.log`: a `WEDGED` line there
means the retry was post-completion and the caveat should be **struck from the
writeup**, not repeated. If there is no such line, the resume was a real
mid-training failure and the caveat stands.

## One cleanup item, after the final upload lands

The uploader serving arm 1 (**pid 795860**, started by hand at 11:30Z after the
wedge) has **PPID 1**, so it is not the trainer's child and `_stop_uploader`'s
`proc.terminate()` will not touch it. It will keep polling
`runs/abl-arch-daedalus-150m/hub_outbox` every 300 s forever — through arm 2 and
through `hero`.

Harmless but not free: ~0.5 GB RSS against ADDENDUM 2's 20 GB ceiling, for days.
`upload_once` returns before constructing `HfApi` when nothing is pending
(`ckpt_uploader.py:209-211`), so it makes no network calls once the outbox
drains. **Kill it once arm 1's final 321 MB is confirmed on the Hub**, not
before.

Arm 2 needs no equivalent: `_start_uploader` spawns its own child, and
`--pass-deadline-s` defaults to `DEFAULT_PASS_DEADLINE_S = 900.0`, so the
bounded behaviour applies without the launcher passing anything. Checked rather
than assumed — the 11:30Z restart passed it explicitly, which made it look like
an opt-in.

## Where this prediction could be wrong

- **Throughput drift.** The ETA is from a 124.4k tok/s window; the box has been
  observed between 121.7k and 124.3k, so ~16:32Z carries roughly ±10 minutes.
- **Step 10,391 assumes the steady-state stride holds to the end.** It has held
  at exactly 524,288 tokens/step since the ramp finished, and the schedule
  replay matched all 184 rows checked earlier, so this is the firmest number
  here.
- **The wedge is transient, so it may simply not happen.** The guard doing
  nothing is still the likelier outcome, not evidence it is broken — which is
  precisely why its behaviour is written down before the fact.

## Correction, 15:50Z — the size hypothesis in this document is falsified

This section previously read *"both observed wedges were on the 1.4 GB
milestone; all five 321 MB rolling uploads have succeeded"*, and used that to
argue the exit drain (a 321 MB payload) was low risk.

**A 321 MB rolling upload wedged at 15:18Z** — identical signature, `CLOSE-WAIT`
and 800 bytes moved per 25 s, killed by the 900 s deadline at 15:33Z and landed
on retry at 15:43:30Z (`runs/preflight/uploader-wedge-recovery.md`). Size is not
the discriminator. The honest rate is **3 wedges in 9 attempts today**, so the
chance of one at arm 1's exit is roughly **1 in 3**, not the "unlikely" this
document implied.

**But the consequence is smaller than that makes it sound, for a reason worth
stating.** The final checkpoint has **two independent routes to the Hub**:

1. the trainer's own synchronous drain — arm 1 holds the pre-fix *unbounded*
   `upload_once`, so this is the one that can hang the process; and
2. **daemon 795860**, which polls the same outbox every 300 s under a 900 s
   deadline, is PPID 1, and is *not* touched by `_stop_uploader` (it terminates
   only the child it spawned at 05:13Z — now the zombie at pid 435972).

`upload_once` never raises, and a payload survives a killed pass because
`_discard` (`ckpt_uploader.py:190`) removes the staged file only *after* a
successful upload. So if route 1 wedges, route 2 delivers the same checkpoint
anyway, bounded, within ~20 minutes. **A wedge at exit costs a hung trainer
process — which the guard kills and `abl_arch` retries with `--resume` — and
does not put the final checkpoint at risk.**

The cleanup item above is unaffected but its ordering now matters more: do not
kill 795860 until arm 1's final upload is confirmed on the Hub, because it may
be the thing that performed it.
