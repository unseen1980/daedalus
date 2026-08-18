# Measured throughput for both arms, and what the remaining jobs cost

Run 2026-08-09 ~22:45Z via `scripts/preflight_batch.sh 16,12,8
daedalus-150m,dense-150m 2048` — the exact command the overnight chain runs
before `abl-arch`. Production settings (`torch.compile` on, bf16, loss chunk
1024, no block checkpointing). **Measured while `dataprep` was still running**,
so a quiet box should be no worse.

| arm | micro-batch | tok/s | ms/step | peak VRAM | of 32.6 GB |
|---|---|---|---|---|---|
| `daedalus-150m` (hybrid) | 16 × 2048 | **115,692** | 283.2 | **25.29 GB** | 78% |
| `dense-150m` (twin) | 16 × 2048 | **102,046** | 321.1 | **29.55 GB** | 91% |

`preflight_batch.sh` chose **16** — the top candidate — and exited 0.

## Cost at these rates ($0.449/hr)

| job | tokens | GPU-hours | cost |
|---|---|---|---|
| `abl-arch` hybrid arm | 5B | 12.0 | $5.39 |
| `abl-arch` dense arm | 5B | 13.6 | $6.11 |
| `abl-arch` full-pass val, both arms | — | ~1.3 | ~$0.58 |
| **`abl-arch` total** | | **~26.9** | **~$12.08** |
| **`hero`** (hybrid, 40B) | 40B | **96.0** | **$43.12** |

Both land within the figures already quoted to the operator (~$11.4 and
~$43.70); the small overshoot on `abl-arch` is the val pass, which was
priced separately.

## The thing this was built to catch

The dense twin had never been memory-checked. It is param-matched to the
hybrid but not activation-matched — 24 attention layers against 6, FF 24×2304
against 18×2048 — and the header of `preflight_batch.sh` estimated it at
"essentially on the OOM line" at batch 16. It fits, but at **29.55 GB of 32.6,
about 3 GB of headroom**. That is the tightest number in the plan.

Two things make it safe rather than lucky:

- ~~`abl_arch.py` does **not** pass `--val-dir`~~ — **superseded 2026-08-10,
  before `abl-arch` launched.** `abl_arch.py:445` now passes `--val-dir` to
  *both* arms, because the alternative was training ~25 h with no validation
  curve at all. That removes the premise this bullet rested on, so the 3 GB
  margin needs a different argument. It has one, and none of it is new code:
  the in-process val runs at `val_batch_size=8` (half the training
  micro-batch), under `torch.no_grad()` with no backward graph, and against
  `self.model` — the **uncompiled** module, chosen at `train.py:889` precisely
  so an eval batch shape cannot trigger a `torch.compile` recompile and a
  fresh autotune workspace. Its live allocation is therefore far below the
  training step's, and the freed training-activation blocks it reuses are
  already in the caching allocator. If that reasoning is nevertheless wrong,
  `_val_bpb` catches **every** exception (`train.py:908`) and logs a warning,
  so a val OOM costs this run its val curve, not the arm. **Confirm against
  the dense arm's `peak_mem_GB` at its first val step (500) rather than
  trusting this paragraph** — that is ~30 min into the arm, not 13 h.
- Sequence length is already at its maximum here (2048), so the ramp cannot
  push the peak higher later in the run. `torch.compile`'s autotune workspace,
  which is what OOM'd the hybrid at batch 24, is allocated in the first steps
  and is included in this measurement.

`hero` runs the hybrid arm at 25.29 GB, 7.3 GB of headroom, and *does* pass
`--val-dir` — comfortable.

## What this does not prove

Five steps, synthetic random tokens, no data pipeline. It measures compute and
memory, not the shard reader or long-run stability.

---

## Superseded for the hybrid arm — 2026-08-10 06:35Z

The table above is a *benchmark*: five synthetic steps, taken while `dataprep`
competed for CPU. `abl-arch` arm 1 has now passed its ramp and is training the
same arm at the same micro-batch on a quiet box, so the hybrid row has a
production replacement (`steady-state-throughput.md`):

| | benchmark | measured in production | |
|---|---|---|---|
| `daedalus-150m` (hybrid), seq 2048, batch 512k | 115,692 | **121,994** (sd 0.11%) | **+5.4%** |

Re-pricing the jobs from it:

| job | GPU-h | cost | basis |
|---|---|---|---|
| `abl-arch` hybrid arm (5B) | 11.4 | $5.10 | measured |
| `abl-arch` dense arm (5B) | 12.9 | $5.78 | **inferred, see below** |
| `abl-arch` full-pass val, both arms | ~1.3 | ~$0.58 | as before |
| **`abl-arch` total** | **~25.5** | **~$11.46** | was ~$12.08 |
| **`hero`** (hybrid, 40B) | **91.9** | **$41.26** | measured |

**The dense row is an inference and should be read as one.** Arm 2 has not run.
It assumes the benchmark's hybrid:dense ratio (102,046/115,692 = 0.882) carries
over to the quiet box — reasonable, since both rows were taken under the same
contention in the same script, but it is a ratio applied to a measurement, not a
measurement. Arm 2 starts in ~11 h and replaces it with the real number.

That correction brings `abl-arch` back under the ~$11.4 originally quoted to the
operator, which the benchmark-based $12.08 had slightly overshot.

## 2026-08-10 07:45Z — the dense arm's memory margin, measured instead of argued

The thinnest number in the plan is the dense twin at **29.55 GB of 32.6** in the
batch-16 preflight, and an OOM there lands at hour 8 of arm 2, after arm 1 has
already burned 12 h. Two claims were carrying that risk, both arguments rather
than measurements. Arm 1 has now run long enough to settle both.

**1. `smoke.py` reads conservative — confirmed at 4.0%.** The hybrid measured
**25.29 GB** in the preflight and peaks at **24.288 GB** in the live arm at the
same seq 2048 and micro-batch 16. Applying that ratio to the dense row puts its
real peak near **28.4 GB, ~4.2 GB spare (13%)** rather than 3.05 GB (9%).

**2. The val pass costs nothing at the peak — measured, not reasoned.**
`--val-dir` was added to `abl_arch.py` *after* the memory preflight was taken,
so no measurement covered it, and the safety case was the argument that
forward-only work under `no_grad` at `val_batch_size` 8 must allocate less than
the training step at micro-batch 16. Five val passes have now run (steps 500,
1000, 1500, 2000, 2500) and **not one moved `peak_mem_GB`**:

| | step 480 | step 500 (val) | step 520 |
|---|---|---|---|
| `peak_mem_GB` | 16.386 | 16.386 | 16.386 |

The peak tracks the seq ramp alone — 14.547 → 16.386 → ... → 24.288 — and has
been flat at 24.288 since seq reached 2048. So the argument was right, and it is
now a measurement on this hardware rather than a claim about allocator
behaviour.

Both corrections run in the same direction: the dense arm has more headroom than
the number that was worrying us. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
stays unset, as decided — introducing an untested allocator flag hours before a
25 h unattended run is the worse trade, and `abl_arch.py` already retries a
failed arm 3× with `--resume`.
