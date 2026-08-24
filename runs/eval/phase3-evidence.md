# Phase 3 QAT recovery: what was found before a single probe ran

Written 2026-08-24 on the Vast box. This file is the reading of Phase 3; the
machine-readable plan is `runs/qat-recovery/preregistration.json` and the
verdicts land in `runs/qat-recovery/`.

The phase has not yet produced a recovered model. What it has produced is two
defects that would each have made the recovery meaningless, and the evidence
that the path is now clean. Recording that first is the point: a phase that
only reports its headline number hides the part where the number nearly came
out of a broken pipeline.

## The released model could not be quantization-aware trained at all

The first smoke run — released checkpoint, real shards, `qat_frac=1.0`,
`--max-steps 3` — produced a **NaN loss on its very first step** and then span
for 0.18 GPU-hours writing 2,794 identical metrics rows before it was killed by
hand.

Two independent defects, which looked like one because they share a symptom.

### 1. A dying channel poisoned the tensor it lived in

`daedalus/qat.py::_safe_reciprocal` guarded `d == 0`, which is where a *fully*
dead channel lands, and its docstring reasons carefully about exactly that
case. A channel on its way to zero passes through a window it did not cover:
the block absmax is denormal-small but not zero, so `d = absmax / -8` is
representable and `1/d` is **not**. It overflows fp32 to `inf`, and the block's
exactly-zero elements — of which a dying channel has many — compute
`0 * inf = NaN`. `floor(NaN + 8.5)` stays NaN, `clamp` propagates it, and one
such block poisons the whole tensor.

Measured on `/root/daedalus/final/hero/checkpoint.pt`, whose every **stored**
weight is finite:

| tensor | NaNs after `q4_0_qdq` | block absmax |
|---|---|---|
| `layers.1.feed_forward.w1` | 1,418 | 0.2296 |
| `layers.1.feed_forward.w3` | 1,674 | 0.1628 |
| `layers.13.feed_forward.w1` | 3 | 0.7870 |

This also made `qat_rel_rmse` NaN, which is why the failure was ambiguous
between a weight fault and a data fault: on CUDA an out-of-bounds embedding
gather corrupts the context and makes *every* later kernel return garbage, so a
bad shard produces the same pair of NaNs.

The guard is now "is the reciprocal representable", which subsumes `d != 0`. It
matches the C reference wherever the C reference is defined — for a denormal
`d` under flush-to-zero, `d ? 1.0f/d : 0.0f` takes the zero branch too — and it
cannot move a lattice that matters, because a block reaching that branch has an
absmax below 2.4e-38, so `fp16(d)` is 0 and the block dequantizes to zero
whichever way `q` lands.

**This is a real quality result, not only a bug fix.** The released model
carries dead and dying ShortConv channels; the phase brief anticipates them
("inherited dead channels"). Any future QAT on this checkpoint family — Phase 8
Daedalus-Code included — would have hit the same wall.

### 2. A run that skips every step never ends

`train_step` returns early on a non-finite loss *without* advancing `step` or
`tokens_seen`. That is correct on its own terms: a skipped update trained
nothing and should not be billed against the budget. But both of `fit`'s break
conditions are thresholds on those two counters, so a run whose every step is
skipped can reach neither. It spins.

`--max-steps 3` was set. Nothing in the process would have stopped it before
the 144-hour deadline. `fit` now raises `NonFiniteStall` after
`max_consecutive_skips` (25 by default), handing the decision to the supervisor
that knows whether to retry or move on. The count is *consecutive*, so an
occasional bad batch never accumulates toward it.

### 3. The grid tests were never running on this box

`tests/test_qat.py::_find_libggml` searched `/tmp/llama.cpp` and
`vendor/llama.cpp`. The Vast image builds llama.cpp at `/opt/llama.cpp`, which
no pattern matched, so the four tests that certify **our Q4_0 grid is
llama.cpp's grid** — the single claim QAT rests on — skipped silently through
Phase 2 and into a change to the quantizer itself.

With the path added all four run and pass against the real
`libggml-base.so`. That is what makes "the fix moved nothing" an observation
rather than an argument.

## The inputs are now verified, separately

`scripts/recovery_preflight.py` checks weights and data by different means on
purpose, because their failures are indistinguishable downstream.

| check | result |
|---|---|
| every source's token ids | max id **49,151** against `vocab_size` 49,152 — in range |
| stored weights finite | yes, 0 tensors non-finite |
| weights survive `q4_0_qdq` | yes, 0 tensors non-finite (was 3) |
| `qat_rel_rmse` on the released model | **0.0349** |

That 0.0349 is the quantity QAT exists to drive down, measured on the released
weights before any recovery step.

## The recovery path, measured

Second smoke run, same command, after both fixes:

| | |
|---|---|
| weights loaded | `--init-from`, `step=0 tokens_seen=0`, fresh optimizers |
| QAT engaged | step 0, **103 tensors** on the `q4_0/q8_0` grid |
| losses | 2.459, 2.588, 2.461 — finite |
| grad norms | 2.68, 2.70, 2.53 — finite |
| skipped updates | **0** |
| peak memory | **17.4 GB** of 24 GB |
| throughput | **~25,000–27,000 tok/s** |

At 25k tok/s a 100M-token arm costs about **1.1 hours**, so three arms plus the
300M follow-up is roughly 6.7 hours and a 1B escalation would add about 11.
That fits the remaining budget with room, which means the deadline is not what
decides how far this phase goes — the gates are.

### The mixture the probes train on

The corpus is a proportional slice of `Unseen1980/daedalus-corpus`, the same
tokenized shards the released model was pretrained on. 1.6B tokens on disk
across all ten sources. `MixtureBatchSource` reports **L1 skew 0.77 points**
from the target mixture, with only `everyday-conversations` capped — it holds
403,573 tokens and would need 5.0 epochs to hit its 2% share, so the 4.0-epoch
limit binds. That is the same shortfall the original run recorded for that
source, not a new one.

## Preregistered deviation: the retrieval gate

The brief's retrieval gate is "no more than 1 point absolute at any depth". The
Phase 2 baseline was measured at **10 items per depth**, where one item is 10
points — so as written the gate admits only *exact equality*. It is a
zero-tolerance gate wearing a 1-point label.

`RETRIEVAL_PER_DEPTH` raises the count to 100, the smallest count at which one
item is one point, and the baseline is re-measured at that count so the
comparison is like for like. This was decided and written into the plan
**before any probe ran**; doing it afterwards would have been the
threshold-tuning the phase forbids.

It should not be oversold. 100 items makes the gate *arithmetically
expressible*, not *statistically resolvable*: at an exact-match rate near 0.85,
binomial noise alone is about 3.6 points, so a 1-point move is well inside it.
Resolving one point would need thousands of items per depth, which this phase
has no budget for. What the gate can honestly do is catch a model that stopped
retrieving; what it cannot do is certify retrieval is unchanged to within a
point. Verdicts report the observed drop next to the limit rather than implying
a precision the instrument does not have.

## The controller lease serializes work that does not contend

`run_phase` takes a single program-wide lease, so only one phase command can
run at a time. That is right for two trainers and wrong for the mix this phase
actually has: exporting a checkpoint, converting it to GGUF and measuring
perplexity are **CPU** jobs that could run beside a **GPU** training arm, and
instead they queue behind it.

Attempting an export while arm 1 was training returns
`ControllerLeaseError: active controller pid 63637 owns
runs/vast-program/controller.lock`, which is the lease doing its job — there is
just no way to say "this phase wants the CPU, not the GPU". The practical cost
here is that scoring an arm (~45 min, mostly `llama-perplexity` on 16 threads)
cannot overlap the next arm's training (~1 h on the GPU), so the phase runs
about 2.2 hours longer than the resources require.

Not worth fixing mid-phase — a second lease class is a control-plane change,
and Phase 1's drills are what certify that layer. Recorded because the same
cost recurs in Phase 6 and Phase 8, where the proxy sweeps have far more
CPU-side evaluation per GPU-hour than this one does.

## A control-plane wrinkle worth an operator's attention

`scripts/vast_program.py::run_phase` writes `halted` — a **terminal** status —
whenever its command exits non-zero. The preflight exiting 1 because it had
found the NaN was therefore read as the program being unable to continue, and
the next phase call refused with `TerminalStateError: program already halted`.

A diagnostic reporting a problem is not a program that has stopped. The halt
was cleared per the runbook and the cause recorded in the event log. A
concurrent change (`3ba742b`) fixed the keeper's half of this — it no longer
treats a failed phase command as terminal — but the controller still writes the
same status for both meanings, so the next operator to run a check that
correctly fails will hit it again.

## The arms, as trained

Preregistered in `runs/qat-recovery/preregistration.json` before any of them
started. All three share data, order, seed, schedule shape and budget; only the
learning rates differ, with Adam following the shipped 0.015 Muon:Adam ratio.

| arm | Muon LR | Adam LR | tokens | steps | wall | skipped updates |
|---|---|---|---|---|---|---|
| `qat-recovery-lr0.0002` | 2e-4 | 3e-6 | 100,139,008 | 191 | 1.03 h | **0** |
| `qat-recovery-lr0.0005` | 5e-4 | 7.5e-6 | — | — | — | — |
| `qat-recovery-lr0.001` | 1e-3 | 1.5e-5 | — | — | — | — |

Arm 1 ran in one attempt with no resume, reached lr 1.3e-6 at step 191 (the
schedule is fully decayed as required), and held peak memory at 17.4 GB of 24.

### `qat_rel_rmse` rose, and that is not obviously wrong

| step | 0 (released) | 20 | 40 | 120 | 191 |
|---|---|---|---|---|---|
| `qat_rel_rmse` | 0.034945 | 0.034979 | 0.035051 | 0.035285 | 0.035351 |

The metric QAT exists to drive down went **up** by 1.2% relative over the arm,
while the training loss fell from 2.578 to 2.441.

That combination is consistent rather than contradictory, and it is worth being
precise about why. The straight-through estimator makes the *forward* pass
quantized, so what the run minimises is the loss of the **quantized** model.
Nothing in that objective asks the float master weights to sit near the grid;
`qat_rel_rmse` measures only how far the masters are from it. Masters drifting
slightly off-grid while the quantized model improves is exactly what optimizing
the stated objective looks like.

It does have a consequence the gates need to catch. The shipped FP16 artifact
*is* the master, so masters drifting away from the grid widens the gap between
the FP16 and Q4_0 artifacts — and the Q4 penalty is a ratio between them. A run
could therefore shrink the penalty by degrading FP16 rather than by improving
Q4, which would satisfy the improvement gate while making the model worse. That
is precisely what the mandatory FP16-retention gate (regression at most 0.5%)
is there to refuse, and it is why acceptance requires both gates rather than
either.

Nothing is concluded from `qat_rel_rmse` here. The gate reads the paired Q4
perplexity from the exported artifacts, which is measured after training.

## Scoring order, decided before any Q4 number was seen

Scoring one arm costs roughly an hour: GGUF export plus paired perplexity on
CPU, then the five-task battery and 100-item retrieval on GPU. Three arms is
three hours on top of three hours of training, and the controller lease
(above) means none of it overlaps.

The preregistered selection order is lexicographic — paired Q4 reduction first,
then FP16 retention, then BPB, then the five-task mean, then retrieval — so the
later criteria only matter for ties. Q4 penalty and FP16 perplexity both come
out of the *same* CPU-side paired measurement, which is the cheap half. So all
three arms are measured on that half, and the full retention battery runs on
the leader; if the leader fails a mandatory gate, it runs on the next.

This is a resource-allocation decision, not a threshold change, and it is
recorded here before any arm's Q4 penalty was measured. Every candidate that is
*selected* still has to clear every mandatory gate — the staging changes which
arms get measured on the criteria that cannot change their rank, not what any
of them has to pass.

## Result: the Q4 penalty is gone, and no arm was accepted

`runs/qat-recovery/verdict.json`, produced by the preregistered scorer:

> no 100M probe passed both the improvement and retention gates; reporting the
> negative result rather than escalating

That verdict is mechanical — `select_winner` returned `None`, so
`escalation_decision` refused the 300M follow-up and the 1B escalation. The
numbers behind it are more interesting than the word "negative" suggests.

| | released base | lr 2e-4 | lr 5e-4 | lr 1e-3 |
|---|---|---|---|---|
| FP16 perplexity | **6.6135** | 6.7126 | 6.7034 | 6.7305 |
| Q4_0 perplexity | **6.9798** | 6.7057 | **6.6873** | 6.7054 |
| Q4 penalty | **+5.539%** | −0.103% | −0.240% | −0.373% |
| penalty reduction | — | 101.9% | 104.3% | **106.7%** |
| FP16 regression | — | +1.498% | +1.359% | +1.769% |
| five-task mean | **47.374** | 47.050 | 47.632 | **48.082** |
| five-task change | — | −0.324 | +0.258 | **+0.708** |
| retrieval, worst depth | — | −7.0 | −23.0 | −7.0 |
| skipped updates | — | **0** | **0** | **0** |

### What worked

**The quantization penalty is not merely halved, it is inverted.** Every arm
lands between −0.10% and −0.37%: after recovery the Q4_0 artifact scores
*better* than its own FP16 parent, which is the expected end state when a model
has been trained to expect the lattice it ships on. Against the improvement
gate — halve the 5.539% — the reductions are 102–107%. Target (≤3%) and stretch
(≤1%) are both cleared with room.

**The artifact that actually ships got better.** Q4_0 perplexity falls from
6.9798 to 6.6873 at the best arm: a **4.19% improvement on the ship format**,
from 100M tokens, 1.03 GPU-hours and about $0.46. The paired view backs it: 164
of 292 chunks improve.

**Task quality held, and mostly improved.** Two of three arms *raised* the
five-task mean, by 0.26 and 0.71 points. No arm came close to the 0.5-point
drop limit in the wrong direction except lr 2e-4 at −0.324, which still passed.

**Nothing diverged.** Zero skipped updates and zero non-finite rows across all
three arms and 573 steps.

### What failed, and which failure is the real one

Two gates blocked every arm, and they are not equally serious.

**FP16 perplexity regressed 1.36–1.77% against a 0.5% limit.** This is
inherent to what QAT does rather than a symptom of a bad run. The
straight-through estimator optimizes the *quantized* model; the FP16 artifact is
the float master, which nothing in the objective asks to stay put. The gate is
doing exactly what it was written to do — refusing a candidate that shrank the
penalty ratio by moving its FP16 numerator — but here the ratio closed from
*both* ends: Q4 improved 4.19% in absolute terms while FP16 gave up 1.36%. A
model that ships as Q4_0 is better after this trade, not worse, and the
five-task mean agrees.

That is a finding about the gate, not a reason to move it. The gate stays as
preregistered and every arm stays rejected. Whether a 0.5% FP16 limit is the
right constraint on a *ship-format* recovery is a question for the operator,
and it is the single most consequential open item in this phase.

**Retrieval degraded, and this one is real.** Per depth, against the 100-item
baseline:

| task, depth | base | lr 2e-4 | lr 5e-4 | lr 1e-3 |
|---|---|---|---|---|
| passkey d256 | 0.83 | 0.76 | **0.60** | 0.79 |
| passkey d512 | 0.81 | 0.81 | 0.80 | 0.82 |
| passkey d1024 | 0.86 | 0.83 | 0.80 | 0.87 |
| passkey d2048 | 0.88 | 0.88 | 0.82 | 0.81 |
| MQAR d256 | 0.99 | 0.99 | 0.99 | 0.99 |
| MQAR d512 | 0.96 | 0.94 | 0.94 | 0.93 |
| MQAR d1024 | 0.91 | 0.87 | 0.85 | 0.88 |
| MQAR d2048 | 0.86 | 0.81 | 0.83 | 0.85 |

Copy-control stays at 1.000 for all three, so the prompt formatter is intact
and these are model differences.

The honest reading separates two things. **MQAR degrades consistently** — every
arm is worse at d512, d1024 and d2048, by 2 to 6 points, with the damage growing
with depth. Three independent arms moving the same way at the same depths is a
pattern, not noise. **The −23 points at passkey d256 for lr 5e-4 is one cell**,
it is not reproduced by the neighbouring rates (−7 and −4), and at n=100 with
p≈0.83 one cell carries about 3.8 points of binomial noise. It should not be
quoted as the headline; the consistent 2–6 point MQAR decline should.

Either way the gate fails: even 2 points is twice the 1-point limit. And unlike
the FP16 result, this one is a genuine capability loss on the format that
ships, in the dimension the plan singles out for protection.

### Why this is a stop and not a setback

The preregistered stop rule exists for exactly this: a result that is
*measurably good on the headline number* and fails a protection gate should not
spend 1B tokens before anyone looks at it. Escalating would have cost about 11
GPU-hours to make a retrieval regression larger.

The recovery recipe works. What is not yet established is whether it can be made
retrieval-safe — and that is a question about the recipe (replay mixture, LR
floor, which tensors QAT touches), not about the budget.

## Artifacts

Kept on the box, hashed, and **not published**. Publishing implies endorsement,
and the preregistered verdict endorses none of these.

| artifact | sha256 |
|---|---|
| `runs/qat-recovery-lr0.0002/checkpoint.pt` | `ea4719e62998e842df969bf9dc716eabf13a469fb825ce602cd7fb12c7297a6a` |
| `runs/qat-recovery-lr0.0005/checkpoint.pt` | `1b7b65644c18bece548afe8b8f2cf4072b93b854de16cb5874fa20ccd821ed90` |
| `runs/qat-recovery-lr0.001/checkpoint.pt` | `fc01bea145d5e339e0906409d93996b26ef3aeb6b0c49a86eedded34d1e0194c` |
| `runs/qat-recovery/export/lr0.0005/model-f16.gguf` | `35fb7a2b33f75fb5980c2ac30793e035410904cd975b658e9edcf9e0eabd79da` |
| `runs/qat-recovery/export/lr0.0005/model-q4_0.gguf` | `30dd89dcef47392cfda5027286bb01b0ef8128ef6d4f52a61300b1c0db772d9a` |

The released repositories are untouched. The input checkpoint
(`cfbf27dc…`) was opened read-only and every arm started from it with
`--init-from`.

## What the operator has to decide

1. **Is a 0.5% FP16 perplexity limit the right gate for a ship-format
   recovery?** As written it rejects a model whose Q4_0 artifact is 4.19%
   better and whose five-task mean is up 0.26–0.71 points. If the answer is
   "score the ship format against the ship format", the correct comparison is
   Q4_0 6.6873 against the released Q4_0 6.9798, and lr 5e-4 passes
   comfortably. That is a change to a preregistered gate and is therefore the
   operator's call, not this session's.
2. **Is the MQAR decline acceptable, or does the recipe need fixing first?**
   The consistent 2–6 point loss at depth is the one result here that is
   unambiguously a regression.

## Still open

- **The 300M follow-up and the 1B escalation were not run**, per the
  preregistered stop rule. Roughly 14 GPU-hours were deliberately not spent.
- **Nothing is published.** The artifacts above are on the box with hashes; the
  decision to publish any of them privately waits on the two questions above.
- **No retrieval-safe variant has been tried.** The obvious candidates —
  raising the general-replay share, flooring the LR earlier, or leaving
  attention projections out of the QAT plan so the retrieval path is not
  retrained on the lattice — are untested and would each need a fresh
  preregistration.
- **The `_safe_reciprocal` fix should be carried into Phase 8.** Daedalus-Code
  starts from the same checkpoint family and would hit the identical NaN.
- **Full-pass BPB was not measured**, and it is third in the preregistered
  selection order. `scripts/bpb_eval.py` needs a `--holdout-root`, and the
  corpus slice fetched here is train shards only — `make_mixture_holdout_split`
  was never run against it, so there is no held-out split to score. It could
  not have changed this verdict: BPB sits below the Q4 penalty and FP16
  retention in the lexicographic order, both of which already separated the
  arms, and no arm was accepted, so no tie needed breaking. It is recorded as
  not-measured rather than left to look measured, and building the holdout is
  a prerequisite for any rerun that expects BPB to arbitrate.
