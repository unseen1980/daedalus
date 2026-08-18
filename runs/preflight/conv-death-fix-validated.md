# The conv-death fix, validated — including the check that would have broken it

Written 2026-08-11 ~18:00 UTC, with `hero` at ~step 14,500 of 124,476 (5% in).
`hero` was not touched at any point; every measurement here is CPU-only.

This note is the evidence behind the restart decision in issue #7. It exists
because the headline result — "removing weight decay from the conv projections
eliminates channel death" — had a failure mode that would have looked exactly
like success, and the interesting part of the work is ruling that out.

## What was measured

`scripts/conv_death_mechanism.py`, three arms, the real `Daedalus` class, the real
`build_optimizers`, real tokens from `hero`'s own training shards. Identical seed
and identical data order across arms, so the arms differ in one thing each.

Probe: hidden 256, 9 layers (2:1 conv:attention, matching `daedalus-150m`'s 12:6),
vocab 8,192 by frequency rank, seq 256, batch 8, 600 steps, Muon lr **0.15**.

The lr is deliberately 7.5× `hero`'s 0.02. Muon's decay is `w *= (1 - lr*wd)`, so
lr scales the decay pressure per step; 0.15 makes a phenomenon that takes `hero`
~10,000 steps visible in ~600. This is an accelerant, not a different mechanism —
and the void-gate below is what keeps that honest.

| arm | the one thing that differs |
|---|---|
| `baseline` | nothing — the shipped code path |
| `no_wd_conv` | conv `in_proj`/`out_proj` in a second Muon group at wd 0 |
| `nonzero_out` | `conv.out_proj` initialised `normal(0, std)` instead of zeros |

## The result

| arm | dead @600 | held-out loss | Δ ablate flagged | Δ ablate matched (1,030) |
|---|---|---|---|---|
| `baseline` | **67.06%** | 5.1619 | +0.0000 | +0.0000 |
| `no_wd_conv` | **0.00%** | **5.1123** | (none flagged) | **+0.6008** |
| `nonzero_out` | 65.04% | 5.1538 | +0.0000 | +0.0151 |

Weight decay on the conv projections is the cause. Removing it eliminates the
death and *improves* held-out loss by 0.96% on identical windows.

**Reproducibility**: the whole thing was run twice, independently. Every
trajectory number matched bit-for-bit (run 1 preserved as
`runs/conv-death-mechanism/*-run1.*`).

## Why my earlier explanation was wrong

Issue #7 and the morning `STATUS.md` entry argued the channels stopped receiving
gradient *first* and decay was the consequence, via a race started by
`conv.out_proj`'s zero-init (`model.py:204`): at step 0 the gradient reaching
`in_proj` is `out_proj.T @ grad` = exactly zero, so decay acts before any upstream
gradient arrives.

`nonzero_out` tests that story directly by removing the zero-init. **65.04% still
die** — barely better than baseline's 67.06%. So the zero-init is at most a minor
runner in the race. The decay is the mechanism, and the causal direction in issue
#7 was backwards.

## The check that could have made this a false positive

`dead_fraction` measures mean `|out_proj|` down each column and calls a channel
dead below 1% of the layer's p95. It is a **weight-magnitude proxy**. The fix
**removes weight decay**. So there was an innocent explanation for `no_wd_conv`'s
0.00% that is indistinguishable from the real one at the level of that metric:

> nothing shrank, so the metric never fired.

A fix passing for that reason would be worthless and would look flawless. The
metric is relative to the layer's own p95, which helps — a uniform rescaling
cannot move it — but it does not settle the question, because the criterion is
still about weights rather than about function.

So `functional_check` scores each arm three ways on **fixed** held-out windows
(deterministic, never sampled — a 0.01-nat effect must not be confusable with
which batches were drawn), taken from tokens *after* the training slice under the
**training** remap:

1. **held-out loss** — the arm comparison. The trajectory's `loss` is one training
   minibatch and far too noisy for this.
2. **ablate the flagged channels** — zero them, re-score. **~0 confirms** they
   contribute nothing, which is what "dead" has to mean.
3. **ablate the same *number* of the arm's weakest channels** — the matched
   control. On the fix arm this is the load-bearing one, since its flagged set is
   empty and ablating nothing proves nothing.

Both directions came back clean:

- Baseline's 1,030 flagged channels move held-out loss by **exactly +0.0000**.
  The ruler is functionally valid, and this reproduces `hero`'s own finding
  (zeroing 4,417 channels → loss delta 0.0) at probe scale.
- The fix arm's weakest 1,030 channels are worth **+0.6008 nats**. Its 0% is real
  capacity in use.

### The instrument is itself tested

Two versions of the negative-control test were wrong before it was right, and both
failures were informative:

| version | result | what it revealed |
|---|---|---|
| ablate live channels on an **untrained** model | `-0.0` | an untrained model uses no channel, so *any* ablation is free — this would have certified an instrument that cannot detect anything |
| threshold at 0.01 on a briefly-trained model | 0.0033 | `lowest_k` picks the **weakest** live channels by construction, so the effect is intrinsically small — the threshold was arbitrary, not the instrument wrong |

The test now trains the probe on a sequence whose next token is a function of the
previous one (so the short conv is the only path that can predict it), requires the
loss to have actually fallen, and asserts a **ladder**: ablating the strongest
channels must cost >5× the weakest. The comparison is against an *exact* zero
rather than against noise, because the holdout pass is deterministic.

## The risk this does NOT retire

Decay 0 has **no equilibrium**. Measured over 600 steps:

| mean \|w\| | `baseline` | `no_wd_conv` | ratio |
|---|---|---|---|
| conv `in_proj` | 0.079595 | 0.541407 | **6.8×** |
| conv `out_proj` | 0.051267 | 0.535837 | **10.5×** |

(Baseline's figures are dragged down by its own ~1,030 near-zero columns, so the
true alive-channel ratio is smaller than these. Alive-only norms are reported for
the sweep below, which ran after that was added.)

600 steps is 0.5% of `hero`'s 124,476. Two consequences that the probe cannot
speak to:

1. **Late-training stability.** `muon.py:48` gives the decay's purpose as keeping
   Muon ahead of AdamW *in the heavily-overtrained regime* — which is precisely
   where `hero` sits (59.9B tokens / 160M params = 374 tokens/param) and precisely
   where 600 steps has no information.
2. **Q4_0 damage.** Wider dynamic range quantizes worse. `abl-arch` measured 2.53%
   fp16→Q4_0 against a 1% bar, with QAT expected to close it.

The conv projections are 28,311,552 of Muon's 122,683,392 parameters, so decay
stays at 0.1 on the other **76.9%** and this is not "turn off regularisation" —
but the residual risk is real and is stated as unhedged in issue #7 rather than
argued away.

## The decay sweep

If a decay weak enough to lose the early race but strong enough to bound growth
exists, it is a better fix than 0.

**A correction to how these arms should be read.** I first set `conv_wd = 0.0133`
on the reasoning that at the probe's lr 0.15 it reproduces `hero`'s per-step shrink
exactly (0.15 × 0.0133 = 0.02 × 0.1 = 0.002). **That calibration is the wrong
one.** Muon's update is `w -= lr*update + lr*wd*w`: *both* terms scale with lr, so
lr cannot tilt the race. What decides which side wins is the ratio of decay to
update, which is **`wd` alone**. lr only sets the clock — which is exactly why the
death rate tracks `lr × steps` (lr 0.04 reached 15% dead by step 1,040 where lr
0.02 was still at 0%), and why lr 0.15 for 600 steps is a legitimate accelerant
for ~4,500 steps at 0.02.

Read as they should be, then, the arms need no calibration at all: they test
`hero`'s candidate values **directly**. `conv_wd=0.0133` is a 7.5× weaker decay
than the shipped 0.1, and `conv_wd=0.001` is 100× weaker. The prediction is that
0.0133 still dies but ~7.5× later in `lr × steps`, so 600 probe steps may catch it
partway.

### How the sweep will be read — written before its arms finished

Recorded in advance, on this project's own precedent (`sweep`'s tie rule fired and
saved a $41 decision from a 0.05% noise winner). At the time of writing,
`conv_wd=0.0133` had reported **0.00% dead at step 400**, where baseline was at
47.4% — so the interesting branch is already the live one, and the risk of reading
it favourably after the fact is exactly what this paragraph exists to remove.

The two candidate fixes fail in *opposite* directions, and the sweep alone cannot
separate them:

| candidate | its failure mode | can 600 steps see it? |
|---|---|---|
| `conv_proj_wd = 0` | no equilibrium → unbounded weight growth over 124,476 steps | **no** |
| `conv_proj_wd = 0.0133` | decay still wins the race, just ~7.5× later in `lr × steps` — so the death is **postponed**, perhaps to `hero`'s step ~70,000, not prevented | **no** |

So a `0.0133` arm reading 0.00% at step 600 is **not** sufficient to prefer it.
Its own mechanism predicts death at roughly probe-equivalent step 4,500, which is
past where this probe stops. The rule:

1. If `conv_wd=0.0133` shows **material death** (≥5%) at 600 steps, weaker decay
   only delays and **`conv_proj_wd = 0` is the recommendation**.
2. If it shows **~0% dead**, that is promising but unresolved, and the decision
   requires one more measurement rather than a preference: **a longer single-arm
   run at `conv_wd=0.0133`** (1,800 steps, ~16 min CPU, ~$0.12 of `hero`
   wall-clock) to see whether the death appears once the probe passes the point
   its mechanism predicts. Only a `0.0133` arm that is still clean *there* earns
   the recommendation over 0, because it would then stop the death **and** retain
   an equilibrium.
3. Either way the loss comparison is secondary. These are single 600-step arms and
   a 0.01-nat difference between two fix candidates is not a basis for choosing.

SWEEP_RESULT

## Scope

Nothing here was applied to `hero`. `build_optimizers(conv_proj_wd=...)`,
`train.py --conv-proj-wd` and `hero.py --conv-proj-wd` all default to `None`,
which reproduces the shipped single-Muon-group split byte-identically, including
the optimizer `state_dict` layout — so a `hero` crash-resume onto this code
trains identically. Verified by `test_conv_proj_wd_defaults_to_the_shipped_
single_group_behaviour`, which asserts the group *count*, not just the numbers.

The fix cannot be adopted mid-run: it changes the number of Muon param groups, and
resuming across that boundary exits **rc 1** with `ValueError: loaded state dict
has a different number of parameter groups` — demonstrated through the real
`train.py` CLI, not only in a unit test. That is why the decision in issue #7 is
restart-or-not rather than apply-or-not.
