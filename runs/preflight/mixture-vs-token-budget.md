# 40B is the largest budget at which the blueprint mixture survives

2026-08-10 07:55Z, written for the `[ASK HUMAN] ready for hero` gate, which asks
the operator to choose between **40B and 30B tokens**. The choice has a mixture
consequence that was not in the draft, and there is a third option — "train
longer than 40B" — that looks free and is not.

The operator's own framing: *"corpus mixture balance matters more than corpus
size"* — a skewed mixture hurts benchmarks more than a smaller balanced one.

## The realised mixture, computed with the trainer's own code

`cap_weights_by_epochs` (`train.py:316`) clamps each source so nothing is
repeated more than 4 epochs, then water-fills the freed mass onto whatever still
has headroom. A source's *effective share therefore depends on the size of the
run*. Run against the finished 14,217,552,718-token corpus:

| budget | L1 skew vs blueprint | fineweb-edu epochs | fineweb-edu share | dclm share |
|---|---|---|---|---|
| 10B | 3.97 pts | 1.02 | 38.26% | 22.96% |
| 20B | 3.98 pts | 2.04 | 38.26% | 22.96% |
| 30B | 3.99 pts | 3.06 | 38.26% | 22.96% |
| **40B** | **3.99 pts** | **4.00** | **37.50%** | **22.50%** |
| 50B | **29.91 pts** | 4.00 | **30.00%** | **18.00%** |
| 60B | — | 6.00 | 37.50% | 22.50% (cap abandoned, warns) |

**At 40B the two largest sources land exactly on the 4-epoch cap.** That is not
a coincidence: the corpus was sized as `blueprint_share × 40B / 4`, so 40B is
the budget the data was built for. It is the ceiling, not an arbitrary pick.

## The three readings that matter for the gate

**1. 30B costs nothing in mixture fidelity.** Skew is 3.99 pts either way, and
~all of it is the already-recorded `everyday-conversations` deviation (2.00 pts
of it) plus the water-fill of that freed 2% onto everything else (+0.16 to +0.47
each). So the 40B-vs-30B decision is purely quality-per-dollar; it is *not* a
trade against data balance. That strengthens the cheaper option rather than
weakening it.

**2. Going above 40B silently wrecks the mixture.** At 50B the cap binds hard
and the web backbone collapses while small sources inflate:

| | 40B | 50B |
|---|---|---|
| fineweb-edu | 37.5% | **30.0%** |
| dclm-baseline | 22.5% | **18.0%** |
| finephrase | 7.4% | **13.1%** |
| cosmopedia-v2 | 5.3% | 7.6% |

`finephrase` (Table/FAQ/Tutorial text) nearly doubles and the two highest-quality
web sources give up 12 points between them. **This is the one to flag**, because
"the run is going well, let's extend it" is the natural request after a good
`hero`, it costs money to act on, and the lever is a *budget* change that does
not look like a data change.

**Correcting myself on how visible this was.** I first wrote that "nothing in
the logs would say the mixture had moved". That is wrong: `train.py` already
computed `l1_skew_pts`, put the full `data_mixture` breakdown in the W&B run
config, and printed a line naming the capped sources. The real gap was narrower
and worth stating exactly — **the signal was ungraded**. 3.99 pts at 40B and
29.91 pts at 50B produced the same shape of line, so nothing distinguished
"normal" from "the web backbone just collapsed".

Now graded: above `MAX_MIXTURE_SKEW_PTS = 10.0` the run prints a WARNING naming
the budget, the epoch limit and the remedy. The threshold is quiet at every
budget the corpus was built for (3.97–3.99 pts from 10B to 40B) and loud at the
first one that breaks it. Two tests pin both sides, and one brackets the
constant against those real numbers so it cannot be widened past the budget it
exists to protect.

**3. Above ~55B the cap gives up entirely** and prints a `build more data`
warning, keeping the target mixture with 6× repetition instead. That is the
correct fallback and it is loud, so it is not a trap — unlike 50B, which is
silent.

## Caveat on "epochs"

The loader samples **with replacement**, so "4 epochs" means 4× as many token
draws as the source holds, not 4 complete passes: at 4 epochs ~1.83% of a
source's tokens are never drawn (Poisson, verified empirically at 6.38% vs a
6.00% prediction for `hero`'s 2.81 mean — see `train.py`'s `make_loader`
docstring). The cap arithmetic is unaffected, since it is defined on draws.

## No code change

The cap behaves correctly at every budget; this is a statement about which
budgets to *choose*. Recorded so the gate can carry it and so a later "just
train it longer" is a decision rather than an accident.
