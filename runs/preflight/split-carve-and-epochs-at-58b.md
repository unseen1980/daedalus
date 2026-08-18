# The holdout carve moved the mixture, and 60B stopped launching

**Supersedes `runs/preflight/epochs-at-60b.md`**, which was computed against a
modelled split. Written 2026-08-10 23:50Z, after the corpus top-up finished at
23:29Z and the real `hero` split could be carved for the first time.

## The finding

`hero._cli` refuses when the sampled mixture drifts more than 10.0 pts from the
blueprint target, and it evaluates that on `data/shards-hero-split/train` — the
directory `train.py` actually samples from. Carving that directory for real and
running `hero.check_mixture` on it:

    l1_skew 10.21 pts (limit 10.0)  ->  REFUSE at 60B

The gate document had said **LAUNCH, by 0.64%**. Replying `go` would have
produced rc 2 and an unstarted run.

## Why the projection disagreed with the split

Both figures came from `scripts/mixture_margin.py`. Its `_tree` built a
manifest-only mirror of the corpus with

```python
{"total_tokens": int(n * (1.0 - holdout_frac))}      # a flat 2% haircut
```

under a docstring claiming it was "carved like the real split". It is not what
`make_mixture_holdout_split` does. That function reserves **whole shard files**
— deliberately, so no `PackedTokenDataset` window can straddle the train/holdout
boundary — taking shards from the end until the 2% target is covered. When a
source's final shard is small relative to that target, it has to take the full
~100M shard before it as well:

| source | shards | tail shard | 2% target | real carve |
|---|---|---|---|---|
| `stack-edu-python` | 13 | 10,964,651 | 24,219,293 | **110,964,651 = 9.16%** |
| `cosmopedia-v2` | 26 | 14,592,179 | 19,000,011 | 50,573,683 = 5.32% |
| `finepdfs-edu` | 33 | — | — | 5.20% |
| `finemath-3plus` | 14 | — | — | 4.76% |
| `finephrase` | 21 | — | — | 4.37% |
| `infiwebmath-3plus` | 14 | — | — | 4.23% |
| `finewiki-en` | 53 | — | — | 3.75% |
| `dclm-baseline` | 56 | — | — | 2.26% |
| `fineweb-edu` | 113 | 10,831,608 | 112,500,000 | 115,717,463 = **2.06%** |
| `everyday-conversations` | **1** | — | — | **100% — dropped** |

**3.67% carved overall against a 2.0% target, and unevenly.** `fineweb-edu` is
close to target because it has 113 shards and a large tail; `stack-edu-python`
is 4.6× over because it has 13 and a small one. That asymmetry is the whole
problem: an even carve would not move the mixture at all, and an uneven one
moves it in proportion to how much each source over-gives.

Consequences on the real split at 60B: `stack-edu-python` 9.2% → 7.3%,
`finephrase` 7.1% → 9.4%, `fineweb-edu` 38.3% → 36.7%.

`everyday-conversations` is a second, cleaner effect. One shard of 403,573
tokens cannot be split without emptying one side, so
`make_mixture_holdout_split` skips it with a warning and it is **absent from
what `hero` trains on**. The modelled tree carried it at 395,501 tokens. This
is not a loss worth fixing — it is the same source that took 2,973 epochs of
~2,200 conversations at 60B before issue #5 — but it has to be in the model,
because its 2% target is redistributed over the other nine.

## The fix

`select_holdout_shards` extracted from `make_holdout_split` as a pure function
(same loop, no behaviour change), and `mixture_margin._tree` now calls **that**
rather than modelling it. The mirror is the selector.

`test_the_model_agrees_with_the_split_hero_will_actually_read` compares the
modelled train total against the materialized split source by source, so the two
cannot drift again. That test is the one that should have existed before the
gate was drafted: nothing *inside* the projection could ever have caught this,
because the error was in what it modelled, not in how it computed.

## Corrected budget curve, measured on the real split

`data/shards-hero-split/train`, 16,932,674,383 tokens:

| budget | l1_skew | verdict |
|---|---|---|
| 40B | 0.00 | LAUNCH |
| 45B | 0.00 | LAUNCH |
| 50B | 0.77 | LAUNCH |
| 51B | 1.11 | LAUNCH |
| 55B | 2.37 | LAUNCH |
| **58B** | **4.94** | **LAUNCH — the ask** |
| 59.92B | ~10.0 | LAUNCH — the exact ceiling |
| **60B** | **10.21** | **REFUSE** |

60B misses by 0.14%. The curve is a cliff over the last 5B because each extra
billion pushes another source onto the 4-epoch cap and redistributes its mass to
the three that still have headroom.

## Per-source epochs at 58B — the operator's pre-launch requirement

> *"note the exact per-source epoch counts in `STATUS.md` before launch, and
> flag any single source that lands far above the rest."*

**No source lands far above the rest.** Max 4.00×, median 4.00×, minimum 1.56×,
and the maximum *is* the cap rather than an outlier past it.

| source | target → effective | epochs |
|---|---|---|
| `fineweb-edu` | 38.3% → 38.0% | 4.00× (capped) |
| `dclm-baseline` | 23.0% → 22.8% | 4.00× (capped) |
| `stack-edu-python` | 9.2% → 7.6% | 4.00× (capped) |
| `finepdfs-edu` | 8.2% → 7.8% | 4.00× (capped) |
| `finewiki-en` | 3.1% → 3.0% | 4.00× (capped) |
| `cosmopedia-v2` | 5.1% → 5.8% | 3.73× |
| `finephrase` | 7.1% → 8.1% | 2.38× |
| `finemath-3plus` | 3.1% → 3.5% | 1.57× |
| `infiwebmath-3plus` | 3.1% → 3.5% | 1.56× |

Targets are renormalized over the nine sources present, which is why
`fineweb-edu` reads 38.3% against its 37.5% blueprint share:
`everyday-conversations`' 2% is redistributed.

Corpus-average repetition is **3.43 epochs** (58B over 16.93B), against the
~4.3 the 60B decision was taken on — the top-up bought repetition headroom as
well as tokens.

## What was deliberately not done

Making the carve *exact* (2% per source) would need the boundary shard to be
split, and `PackedTokenDataset` sizes each shard from `len(memmap)` — the file
— not from the manifest's token count (`daedalus/data.py`, `__init__`). So a
manifest that claimed fewer tokens than the file holds would be **ignored**, and
training would silently read the holdout. Doing it properly means physically
writing truncated boundary shards, which is new code on the launch path hours
before a 5.6-day run, to buy 2B tokens. Not taken. Recorded here because it is
the obvious next idea and the reason it is not free is not obvious.
