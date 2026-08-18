# Finishing `hero` early *with* its cooldown — rehearsed 2026-08-11 10:5xZ

**The gap this closes.** The credit watch escalates when the balance cannot
cover the rest of the run. Until now that escalation named a problem and no
action: top up, or lose it. There was a third option nobody had implemented, and
it is the one that makes WSD worth using in the first place.

A run stopped mid-stable-phase hands back a checkpoint sitting at **full learning
rate**. That is not "the same model with fewer tokens" — the linear decay to zero
is where a large share of the final quality is made (D2Z, arXiv 2502.15938,
cited in the blueprint for exactly this schedule). Deciding the cooldown late is
WSD's defining property, so a run that is going to be cut short should be
*retargeted* to a budget the balance covers and annealed properly, not killed.

## What it looks like at hero scale

Priced by `scripts/early_finish.py` against the real `estimate_total_steps`, at
123,000 tok/s and $0.449/h:

| trained when the money runs out | retarget to | cooldown | lr at resume | still to run | naive waste |
|---|---|---|---|---|---|
| 20B (33%) | **28B** | 8B (29%) | 0.533 → 0 | 18 h, $8 | 5,445 steps, $2.83 |
| 30B (50%) | **38B** | 8B (21%) | 0.410 → 0 | 18 h, $8 | 3,738 steps, $1.94 |
| 40B (67%) | **46B** | 6B (13%) | 0.260 → 0 | 14 h, $6 | 2,373 steps, $1.23 |
| 45B (75%) | **50B** | 5B (10%) | 0.201 → 0 | 11 h, $5 | 1,690 steps, $0.88 |

lr **steps down** onto the short ramp at resume and anneals from there. The
cooldown fractions (10–29%) sit in the range the cooldown literature finds
recovers most of the annealing gain.

## The part that had to be fixed first — the last column

`train.py` derives `ramp_tokens = total_tokens × ramp_frac` and then *replays*
the batch ramp to count steps exactly (the fix for the WSD bug in `95d7bc4`).
Retarget the budget naively and the ramp shrinks with it, so the replay believes
the run reached its current token count in fewer steps than it really did. The
live step counter is then **ahead** of the schedule: lr reaches zero before the
tokens run out, and the run spends its last stretch training at lr exactly 0 —
paying full price to change no weights.

That is the "naive waste" column: **1,690–5,445 steps, $0.88–$2.83**, charged
against the **$5 margin that triggered the alarm in the first place**. The
emergency procedure would have spent up to half its own headroom.

`--ramp-frac` (new; additive, default 0.1, behaviour unchanged when omitted)
lets the retarget pass the *original* run's `ramp_tokens` as a fraction of the
*new* budget, so the replay reproduces the ramp that actually happened and the
schedule's last step is the step the token budget really ends on. It changes
nothing else: by the time this matters the run is tens of billions of tokens past
the ramp, so seq_len and batch_tokens are pinned at their end values either way
and `ramp_frac` only feeds the step-count replay.

## Rehearsed, not just tested

`tests/test_early_finish.py` (26 tests) covers the arithmetic. The plumbing —
the part that fails at 3am on day four — was **executed**:
`scripts/early_finish_rehearsal.py` trains a toy run under an "original" budget,
interrupts it mid-flight, and resumes it exactly as the tool prints:

```
interrupted at   step 70,  121,088 tokens, lr multiplier 1.000   (full lr)
planned          89 steps, decay from 48, ramp_frac 0.15
resumed at       step 70,  121,088 tokens, lr multiplier 0.463   <- stepped down
finished at      step 89,  160,000 tokens, lr multiplier 0.000
```

All eight checks pass: it resumed from the checkpoint rather than from zero, the
`--ramp-frac` reproduced the original `ramp_tokens` exactly, the trainer's
schedule was the planned one, lr stepped down and annealed to zero, the run
stopped on the **new** budget rather than the old one, and the schedule and the
token target ended on the same step — no step at lr 0. ~10 s on CPU; it never
touches the GPU. It runs in the suite.

**A bug the rehearsal caught, in the rehearsal itself:** the first run reported
the interrupt at step 140 for 121,088 tokens — half the tokens per step the
replay predicts. `train_step()` increments `self.step` itself (`train.py:1044`)
and the harness incremented it again. Two of the eight checks failed and named
it. Worth recording because it is the reason to execute these things: the
arithmetic tests were green throughout, and a harness that lies about the step
count is precisely how a schedule bug hides.

## How it is triggered

`scripts/credit_watch.py`'s WARN and CRITICAL blocks — in `STATUS.md` and in the
GitHub issue — now carry the command and the budget that is actually available
(balance minus the $5.50 `post`/eval/GGUF tail; if that is negative it says so
instead of offering a retarget it cannot fund). The watch deliberately does
**not** import the planner: it polls every 30 minutes and the planner pulls in
torch and reads a 1.4 GB checkpoint, and a watch that dies computing its own
advice is worse than one that just names the command.

```bash
/venv/main/bin/python scripts/early_finish.py --run-dir runs/hero --budget-usd 18
```

It reads the resume point from **`checkpoint.pt`, not from `metrics.jsonl`** —
metrics run up to 30 minutes (~200M tokens) ahead of the rolling checkpoint, and
planning off the optimistic number would understate what is left to train, in the
direction that makes an unaffordable plan look affordable. It prices against the
run's own **median** throughput (one checkpoint or val stall must not set the
budget), refuses a budget at or below the tokens already trained, and prints a
relaunch that goes through **`hero.py`** rather than `train.py` — an emergency
cooldown is the worst possible moment to also give up the run's crash
supervision, watchdog and `inflight.json`.

## Not armed, and deliberately so

Nothing runs this automatically. It changes the token budget of a run the
operator approved at a specific number, so it is a decision, not a repair. What
is automatic is that the alarm now arrives with the option, the price and the
command.
