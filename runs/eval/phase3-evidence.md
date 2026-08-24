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

## Still open

- The three 100M arms have not run. Nothing below the smoke run is measured
  yet, and no claim about recovered quality appears in this file.
- The retrieval baseline at 100 items per depth is in flight; the
  preregistration is written once it lands, and the arms start after that.
