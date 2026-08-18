# The conv-death decision rule, pre-registered before the trajectory lands

Written 2026-08-11 ~16:20 UTC, with `hero` at step ~13,000 of 124,476 and the
conv-death trajectory holding **exactly two points**. The decision this governs
is taken at the stable-phase milestone, **step 68,461 (~Aug 14)**, which is
~55,000 steps and ~2.6 days away. Neither the milestone's dead fraction nor the
mechanism experiment's result is known as this is written.

The precedent is this project's own, twice: `sweep` had a tie rule written
before its probes scored, and it fired — the winner beat the runner-up by 0.05%,
the rule said that is noise, and the blueprint's 0.02 was taken instead of a
noise winner. `abl-arch` had one for the same reason. Issue #7 named the
milestone as "the natural decision point" and then left the reading of it to
whoever was looking at the number at the time. That is the gap this closes.

## What is actually being decided

At step 68,461 `train.py` writes a milestone checkpoint with full optimizer
state, under its own Hub revision, that the rolling checkpoint cannot overwrite.
That artifact exists so the run can be *branched* rather than repeated. The
decision is whether to use it:

- **A. Do nothing.** `hero` runs on to 59.9B and ships. Cost **$0**.
- **B. Branch with the fix.** Restart from the milestone with the conv
  projections excluded from Muon's weight decay, retraining the remaining 45%.
- **C. Prune the dead channels at export.** Independent of A/B, touches no
  training, and is *not* in competition with them — see below.

A and B are exclusive. C is separable and is deliberately kept out of this rule's
either/or.

## The quantity that decides it, and why it is not the 48%

The headline number — 47.9% dead — is **not** the decision variable, because
**none of it is recoverable by branching.** Zero gradient is an absorbing state;
a channel dead at step 68,461 is dead in every continuation of that checkpoint,
with or without weight decay. Removing decay at the milestone cannot resurrect a
single one of the ~4,400 already gone.

What branching can buy is only the **additional** death that would otherwise
accrue over the remaining 45% of the run. So the decision variable is the
trajectory's **slope**, not its level. Reading the level would recommend
branching every time, since 48% always looks alarming.

Measured slope on the two points that exist:

| | step | dead |
|---|---|---|
| first | 9,896 | 47.928% |
| second | 12,429 | 47.960% |
| slope | | **+0.0129 pp per 1,000 steps** |

Linear projection: **48.68%** at the milestone, **49.40%** at the end of the run.
That is **+1.44 pp** from today, and one pp is 92 channels = 0.283M parameters =
**0.177% of the model**. So on the evidence available today, the entire prize
for branching is **~0.25% of parameters** — and buying it means perturbing the
optimizer of a 4-day, $63.78 run at 55% completion on a change that has never
trained a 150M model.

## The rule

**Default is A (do nothing).** B requires *all four* of the following to hold at
the milestone. Any one failing selects A.

1. **Slope is material.** Projected end-of-run dead fraction, fitted over all
   trajectory points from step ≥ 20,000 (excluding the early race, which is over
   by then), is **≥ 55.0%** — i.e. at least ~7 pp of *further* death, ~2M
   parameters, ~1.3% of the model. Below that, the prize is smaller than the
   risk of touching the run and A wins by inspection.
2. **The fix is validated at small scale**, by an experiment that has a
   **reproduced positive control**: a baseline arm that actually exhibits the
   death, and a fix arm that materially reduces it, at comparable or better
   final loss. A fix arm that looks clean because *nothing* died in the probe is
   not validation and does not satisfy this clause.
3. **The fix does not cost stability.** `muon.py:48` states weight decay "is
   what keeps Muon stable" and is deliberately high; the validation must show
   the fix arm's loss curve is not worse than baseline's. A fix that trades 48%
   dead channels for a divergence risk on the one job that must not fail is not
   a fix.
4. **The branch is affordable at the moment it is taken**, counting the
   post-milestone tokens discarded plus the re-run of the decay phase, against
   the credit left at that time with the `post` and eval jobs still to fund.

If 1 fails, record the finding and move on — the trajectory plateauing *is* the
result, and it is the one the writeup wants either way.

## What the rule predicts today, stated so it can be wrong

On the two points available, clause 1 fails by a wide margin: 49.40% projected
against a 55.0% bar. **The rule as written selects A, and I expect it to select
A at the milestone.** Recording that expectation now is the point — if the
trajectory bends upward and clause 1 passes, that is a genuine surprise that
earns the branch, rather than a number reinterpreted after the fact.

The mechanism supports the same prediction for an independent reason. `hero`'s
per-layer fractions are **bit-identical** between the two samples on 9 of 12
conv layers across 660M tokens; only layers 0, 5 and 6 moved, each by ~0.1 pp.
A process still killing channels does not leave nine layers unchanged to four
decimal places. This is consistent with the death being decided by an early race
that is already over — see `conv-death-mechanism.md` for the experiment testing
that directly.

## Option C, which this rule deliberately does not gate

Structurally pruning the dead channels at export would make the shipped GGUF
~7.7 MB smaller and ~11% lighter on decoded-token weight traffic at
**bit-identical output** — it removes weights that provably contribute nothing.
That is an improvement to the CPU-decode number this project leads with, and it
touches no training and no decision above.

It is gated only on whether llama.cpp's `lfm2` graph tolerates conv width ≠
hidden size, which is an export question answerable on CPU at any time. Kept out
of the A/B rule so that a "no" on branching does not read as a "no" on this.

> **ANSWERED 2026-08-11 18:30Z — option C is CLOSED. `INFEASIBLE`.**
>
> `src/models/lfm2.cpp` creates all three short-conv tensors at fixed `n_embd`
> and `create_tensor` shape-checks them. Tested rather than argued: the real
> `abl-arch` GGUF narrowed 768 → 640 is **rejected** by a real `llama-perplexity`
> (`check_tensor_dims: ... expected 3,768, got 3,640`), while both the unmodified
> file **and** the same file rebuilt at full width through the identical writer
> load fine. The second control is what makes it conclusive.
>
> The 13.6M dead parameters cannot be reclaimed at export. Patching llama.cpp
> would forfeit stock-binary compatibility — the basis of the CPU-decode claim —
> for 7.7 MB, which is not a trade worth making. Q4_0 spends 4 bits on a zero
> like any other weight, so leaving them in saves nothing either.
>
> Evidence: `runs/preflight/conv-prune-feasibility.md`, reproduce with
> `scripts/conv_prune_feasibility.py`.

## Scope

This rule governs the conv-death decision only. It does not authorise any change
to the running job: nothing in A, B or C is to be applied to `hero` before the
milestone, and B is a branch from the milestone checkpoint rather than an edit
to the live process.
