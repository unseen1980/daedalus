# How much GPU is actually free while `hero` runs, and for how long it isn't

2026-08-11 15:25 UTC. Read-only: `nvidia-smi`, `runs/hero/metrics.jsonl`, and
`train.py`'s own schedule functions replayed against the live process's launch
arguments. `hero` was not touched.

## The question

I was about to run a small concurrent GPU experiment (the conv-death fix,
issue #7) in what looked like 9.7 GB of spare VRAM. Before taking it: is it
actually spare?

## The answer: no, not for another 8.5 hours

**The ramp is measured in tokens, not steps.** `ramp_tokens = total_tokens *
ramp_frac` (`train.py:924`) = 5.99B of 59.9B. `hero` is at 2.22B tokens — **37%
of the ramp**, not the 93% that reading it as "10% of 124,476 steps" suggests.

Five sequence-length step-ups are still to come, each a fresh allocation:

| at tokens | seq | batch tokens | grad accum | |
|---|---|---|---|---|
| 0 | 1024 | 131,072 | 8 | passed |
| 479.2M | 1152 | 162,529 | 9 | passed |
| 1.198B | 1280 | 209,715 | 10 | passed |
| 1.917B | 1408 | 256,901 | 11 | passed |
| **2.636B** | **1536** | 304,087 | 12 | **~0.9 h away** |
| 3.474B | 1664 | 359,137 | 13 | ~2.8 h |
| 4.193B | 1792 | 406,323 | 14 | ~4.4 h |
| 4.912B | 1920 | 453,509 | 15 | ~6.0 h |
| 5.631B | 2048 | 500,695 | 15 | ~7.7 h |

Peak VRAM has tracked those step-ups exactly — 14.55 GB (step 20) → 15.92
(2,700) → 16.31 (6,980) → **17.91 GB** (10,340) — and will keep climbing until
~5.99B tokens, **8.5 h from now**.

## Where it lands, and why that is already known to be safe

The ceiling is not a guess. `abl-arch` ran the *same* architecture at the *same*
`--micro-batch 16` through the full ramp to seq 2048:

| | peak allocated |
|---|---|
| `abl-arch` hybrid (= `hero`'s arch) | **24.29 GB** |
| `abl-arch` dense twin | 28.29 GB (29.55 GB reserved — the thinnest margin in the plan) |
| `hero` now, at seq 1408 | 17.91 GB |

So `hero` needs **~6.4 GB more than it holds right now**, and lands ~8.3 GB
under the card. It survives its own ramp on precedent, not on optimism.

**QAT does not move this.** Measured at micro-batch 16, seq 2048, compile and
bf16 live: 24.01 GB with QAT off, **24.01 GB with it on**
(`qat-compile-lattice.md`). The final 5% is not a second memory event.

## The rule this sets for the next six days

1. **Nothing else touches the GPU before ~00:00Z** (ramp end). Taking 5 GB now
   would meet a step-up that needs it and OOM a run that is five hours old.
2. **After that, a concurrent job must stay under ~6 GB** and be killable, to
   keep a margin against the reserved-vs-allocated gap (`nvidia-smi` reads
   22.9 GB reserved against 17.91 GB allocated — the caching allocator holds
   ~5 GB more than the peak the metrics report, so the metrics understate the
   footprint by about that much).
3. The conv-death fix experiment (issue #7) is therefore **not runnable today**.
   It is not blocked on anything technical — it is blocked on `hero` needing
   the memory first, which is the correct priority.

## What this corrects

The reasoning I nearly acted on — "peak_mem has been flat at 17.91 GB for
1,300 steps, the ramp ends at 10% of the run and we are at 8.6%, so the ramp is
essentially done and the spare 9.7 GB is spare" — was wrong in the one way that
matters. 8.6% was the fraction of *steps*; the ramp is paced by *tokens*, and by
tokens the run is 3.7% in. The flat 17.91 GB was the plateau between two
step-ups, not the ceiling.
