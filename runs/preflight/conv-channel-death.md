# 48% of the short-conv channels are dead, and it is provable

2026-08-11 ~14:30-14:50 UTC. CPU only, read-only against checkpoints on disk.
`hero` was not touched, paused or slowed. Found while checking something else
(the export dtype, `qat-survives-export.md`): 8% of the model's weights sat at
~1e-9, which is not a number trained weights have.

## The finding

In `hero`'s live checkpoint at step 9,896 (1.77B tokens, 8% of the run),
**4,417 of the 9,216 short-conv channels — 47.9% — contribute exactly nothing
to the model's output.**

Not "contribute little". Zeroing all 4,417 channels outright changes the loss
by **0.0**:

| held-out source | loss as trained | dead channels zeroed | delta |
|---|---|---|---|
| cosmopedia-v2 | 2.7356204987 | 2.7356204987 | **0.0** |
| dclm-baseline | 4.1978883743 | 4.1978878975 | 4.8e-07 |
| finemath-3plus | 2.7524709702 | 2.7524709702 | **0.0** |

(Real held-out shards, real `Daedalus` forward, checkpoint loaded with 0 missing
and 0 unexpected keys. The one non-zero delta is fp32 summation order.)

A channel is dead when rows `B[j]`, `C[j]`, `x[j]` of `conv.in_proj` and column
`j` of `conv.out_proj` have all collapsed. `ShortConv` computes
`out_proj(C * conv(B * x))` (`daedalus/model.py:62`) — a **double** multiplicative
gate, so a channel whose product goes to zero receives zero gradient in all
three projections, and zero gradient is an absorbing state. It can never come
back.

## It is not confined to one run, and it is not the ablation arm

| run | lr | step | tokens | dead conv channels |
|---|---|---|---|---|
| sweep-wsdfix-lr0.01 | 0.01 | 1,040 | 0.50B | 0.0% |
| sweep-wsdfix-lr0.02 | 0.02 | 1,040 | 0.50B | 0.0% |
| sweep-wsdfix-lr0.04 | 0.04 | 1,040 | 0.50B | **15.4%** |
| **hero** | 0.02 | 9,896 | 1.77B | **47.9%** |
| abl-arch arm 1 | 0.02 | 10,391 | 5.00B | 45.0% |

Per layer in `hero`, deadness rises with depth and then flattens:
L0 5%, L1 42%, L2 45%, L3 45%, L5 49%, L6 52%, L8 56%, L10 58%, L12 57%,
L14 55%, L15 58%, L17 54%.

**Measure deadness relatively, not absolutely.** Dead weights decay as
`init * exp(-lr * wd * steps)`, so an absolute cutoff measures how long weight
decay has been running rather than how many channels died — which is exactly why
the 1,040-step sweep checkpoints read 0.0% under a 1e-6 cutoff. The table uses
"below 1% of the p95 channel in the same tensor".

## The mechanism, checked rather than assumed

Muon carries `weight_decay=0.1`, deliberately high — its own docstring says
decay "is what keeps Muon stable" (`daedalus/muon.py:48`) — and it applies to
every 2D matrix, `conv.in_proj` and `conv.out_proj` included. A row receiving no
gradient is then multiplied by `(1 - lr*wd)` every step: at lr 0.02 that is
0.998, and after 9,896 steps `0.998^9896 = 2.5e-9`, against an init scale of
~0.02. **The observed dead weights sit at ~1e-9 to 1e-11.** The clock matches.

The discriminating evidence is the depthwise kernel. It is 1D, so
`build_optimizers` routes it to AdamW with **weight decay 0** — it *cannot* be
shrunk by decay. If the channels were being ground down by decay while still
carrying signal, their kernels would look trained. They do not:

| | mean \|w\| of `conv.conv.weight` |
|---|---|
| dead channels | **0.0501** |
| alive channels | 0.0221 |
| ratio | **2.27×** |

The dead channels' kernels are *larger* — sitting where initialization put them,
never updated. So those channels stopped receiving gradient, and the decay that
followed is the consequence, not the cause.

## Scope: the double gate is what is special

Every 2D tensor family in the checkpoint, dead-unit fraction:

| family | dead |
|---|---|
| `conv.in_proj` (out-units) / `conv.out_proj` (in-units) | **47.9%** |
| `feed_forward.w1` / `w2` / `w3` | 0.4% |
| everything else (attention q/k/v/o, embeddings) | <0.05% |

SwiGLU is also multiplicatively gated and barely dies (0.4%). The short conv
has **two** gates in series (`B * x`, then `* C`), which makes the dead zone far
easier to fall into and impossible to climb out of.

## What it costs

| | |
|---|---|
| dead parameters | 4,417 × (3×768 + 768 + 3) = **13.58M** |
| of the 160.49M model | **8.5%** |
| of the 122.68M weights that ship as Q4_0 | **11.1%** |
| of the 102 MB Q4_0 file | **~7.7 MB** |

Those bytes are read on every decoded token — llama.cpp has no idea they are
zero — so they are also ~11% of the quantized weight traffic behind the CPU
decode number this project leads with.

## What it does *not* show

- **No evidence that quality is worse.** The obvious counterfactual — the same
  architecture without the collapse — has never been trained. What we do have is
  that `abl-arch` arm 1 carried 45% dead conv channels and still beat the
  param-matched dense twin on the pre-registered metric (val_bpb 0.910398 vs
  0.917774). The hybrid wins *with* this handicap.
- **No evidence about where it ends.** Nothing on this project has trained past
  ~10.4k steps. `hero` is at 9,896 of 124,476. Whether 48% is a plateau or a
  waypoint is unknown, and it is the number that decides how much this matters.
- **Nothing here is recoverable within this run.** Zero gradient is absorbing;
  the 4,417 channels are gone for the remaining 5.5 days whatever is decided.

## What was done about it

1. `scripts/conv_death_watch.py` samples the fraction from `hero`'s own rolling
   checkpoint (throttled to one sample per 2,000 steps, skips a torn read,
   ~10 s), appending to `runs/hero/conv-death.jsonl`. By the milestone at step
   68,461 there will be a real trajectory instead of two points from different
   runs. 10 tests; writing them found a defect where the very first sample of
   any run was silently throttled away.
2. Reported as issue #7 with the options and their costs. **No change to
   `hero`** — see the issue for why restarting on an untested hypothesis is the
   worse trade, and why the stable-phase milestone is the natural decision point
   if the trajectory turns out to be climbing.

---

# Addendum, 2026-08-11 15:55Z — the measurement has no threshold ambiguity, and the first two points are flat

Two things needed checking before the trajectory could be trusted to answer
"plateau or waypoint", both read off `hero`'s live checkpoint at step 11,626
(2.207B tokens), CPU, read-only.

## 1. The relative threshold has a moving part, and it turns out not to matter

`DEAD_REL` is 1% of the p95 channel *in the same tensor*, so if healthy channels
grow over training the bar grows with them and channels can cross it without
anything having died. **That confound is real here — p95 has more than doubled,
from ~0.020 at init to 0.042190 now.**

It changes nothing, because the distribution is bimodal with an enormous gap:

| threshold (× p95) | mean dead |
|---|---|
| 0.1 | 47.96% |
| 0.01 (the one used) | **47.96%** |
| 0.001 | 47.94% |
| 0.0001 | 47.91% |

Four orders of magnitude of threshold move the reading by 0.05 points. Layer 10
shows why: the largest "dead" channel is **2.195e-11**, the smallest alive one
is **3.240e-02** — a gap of **1.5 × 10⁹**. Nothing sits in between.

**So there is no such thing as a partly-dead channel here.** Death is binary and
complete, which is what an absorbing state predicts and is stronger than the
original note claimed. It also means any future change in the number is a real
change in the *count* of dead channels, not an artifact of where the bar sits.

`conv_death_watch.py` now records all three thresholds plus `p95_mean` on every
row, so this stays checkable over the run rather than being established once.

The decay clock matches too: `0.02 × 0.998^11626 = 8e-11` against an observed
~2e-11, i.e. these channels died early and have been decaying ever since.

## 2. First read of the trajectory: flat

| step | tokens | dead |
|---|---|---|
| 9,896 | 1.766B | 47.93% |
| 11,626 | 2.207B | **47.96%** |

**~3 channels died in 1,730 steps, against 4,417 in the first 9,896.**

That is what a plateau looks like, and it is consistent with the mechanism: the
collapse happens in early training when gradients are noisy, and the survivors
are stable. **But two points is not a trajectory** — they are 1,730 steps apart
in a 124,476-step run, and the seq/batch ramp is still changing the optimization
underneath them (37% through, `vram-headroom-during-hero.md`). The sampler runs
every 15 min for the rest of the run; the milestone at step 68,461 is where this
gets decided, on ~50 points rather than 2.

Nothing about `hero` changes on this. It is early evidence pointing at the
cheaper of the two answers, recorded so the direction is on the record before
the data that settles it arrives.
