# How much room is there between the corpus and `hero`'s refusal at 60B?

2026-08-10 ~21:00 UTC. Cost: nothing — CPU only, manifests only, no GPU.

Reproduce any row with `scripts/mixture_margin.py`, which goes through
`hero.check_mixture` rather than reimplementing the skew maths.

## Why this was worth asking

`hero._cli` refuses with rc 2 when the sampled mixture drifts more than
**10.0 pts** from the blueprint target. That is a boolean at one budget, and it
answers the wrong question for a decision: not *"would 60B launch"* but *"by how
much, and what would have to go wrong for it to stop"*.

The prompt for asking was `stack-edu-python`, which **exhausted its stream** at
1,210,964,651 of a 1,350,000,000 budget — 139M short, permanently. A source
running out of documents is no longer hypothetical on this corpus, so the
question is what happens if another one does.

## The answer, on the split `hero` actually gates on

`hero` reads the **train** split, not the whole corpus, so every number below
carves the 2% per-source holdout first.

| scenario | split corpus | l1_skew | largest budget that launches | verdict at 60B |
|---|---|---|---|---|
| corpus as it stands (21:00Z) | 15.38B | 33.61 | **45.89B** | REFUSE — top-up still running |
| **projected final**, every remaining source reaches budget | 17.23B | **9.01** | **60.39B** | **LAUNCH, by 0.64%** |
| `finepdfs-edu` also exhausts where it is now (885.7M) | 16.92B | 13.12 | **58.72B** | **REFUSE** |

Two things follow, and the second is the one that matters.

**1. The 60B budget clears by 0.99 pts of skew — 0.64% of budget.** The operator
chose 60B knowing the credit buffer was thin; this is a *second* thin margin
they had not been told about, in a different currency. Nothing is wrong with the
plan, but 60B is much closer to the refusal line than "it launches" suggests.

**2. `finepdfs-edu` is the single point of failure, and it is measurable.**

> **`finepdfs-edu` must reach ≥ 1,124,340,092 tokens** — 93.7% of its 1.20B
> budget — for `hero` to launch at 60B. It stood at **885,692,493**, so it needs
> **+238,647,599 more**.

It is not stalled: it soft-RSS-stopped at 5.05 GB, resumed via O(1)
stream-position restore, and was actively writing `finepdfs-edu_00027.bin` at
20:53Z. At its measured 0.255 B/h it needs ~0.94 h. The risk is **exhaustion,
not time** — exactly what `stack-edu-python` did — and there is no way to know
in advance how many documents the source has left.

## What to do if it falls short

Ranked by what it costs.

| option | effect | cost |
|---|---|---|
| **launch at 58B** instead of 60B | launches even if `finepdfs-edu` stops dead now | −2B tokens, ~−$2.1 |
| let the top-up run longer | may or may not help; the source may be empty | free (runs beside arm 2) |
| holdout 2% → 1% | ceiling 60.39B → **61.00B** | val_bpb on 325M instead of 650M tokens |
| `--allow-skewed-mixture` | launches on a mixture we measured as wrong | the thing the gate exists to prevent |

**The holdout lever is weak and should not be relied on**: halving the holdout
buys +0.61B of budget headroom, which does not rescue the `finepdfs-edu`-stalls
case (58.72B → 59.32B, still under 60B). It is listed to record that it was
measured, not because it is a plan.

**The recommendation is 58B if `finepdfs-edu` has not reached 1.124B by gate
time, and 60B if it has.** That is a 3.3% token cut in the bad case, against a
refusal after the operator has already said go.

## Re-run this at gate time rather than trusting the projection

Every row above is a projection of a corpus still being built. At ~07:05Z the
corpus is final and the question is answerable exactly:

```bash
python scripts/mixture_margin.py --data-dir data/shards --budget 60e9
```

rc 0 = launches, rc 2 = refuses, and the "largest budget that launches" line is
what the gate should quote. 11 tests in `tests/test_mixture_margin.py` pin that
the ceiling is the real crossing point (just below launches, just above
refuses), that the holdout is applied, and that the verdict agrees with
`hero.check_mixture` on the same corpus.
