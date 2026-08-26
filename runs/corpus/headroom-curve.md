# Phase 7: unique-token headroom, as a curve across budgets

Measured 2026-08-26, from `scripts/source_headroom.py epochs --hub`. Exact
numbers in `runs/corpus/headroom-curve.json`; `headroom-curve-realized.json` is
the same computation with the Hub file metadata withheld, i.e. the floor that
credits only tokens the released build actually produced.

The operator has not fixed a successor size and a month on one RTX 5090 reaches
somewhere between 60B and 200B tokens, so this reports a curve rather than a
verdict on one budget. Any later target reads off the same measurement.

## Headline

The corpus as built holds **17.15B unique tokens**, which is where the released
run's 59.9B budget and its ~3.5 epochs came from. Counting the files the build
never opened, the same ten sources can still supply **6,582B unique tokens** --
but that supply is extremely uneven, and only three sources ever bind.

**The mixture supports a total budget of ~53.8B tokens at four epochs.** That
ceiling is set by a single source, `stack-edu-python`, which is out of
documents. Dropping `everyday-conversations` -- which the phase plan already
directs, for unrelated reasons -- removes a ceiling that sits at 80.7M tokens.
Below ~900B nothing else binds at all.

## How supply was measured

Per source, supply is the tokens the released build realized **plus** a lower
bound on what is still reachable in the files its stream never opened, credited
at the density the source itself achieved.

The density is measured, not assumed: the source's own realized tokens over the
bytes of every file it touched *including* the partly-read one, while the
reachable remainder *excludes* that file's tail. Both roundings point down, so
the reach cannot be an artefact of optimistic arithmetic. Per-source filters
(`fineweb-edu`'s `int_score >= 3`), the 100k-char document cap, dedup and
decontamination are all already priced in, because the numerator is what the
build actually emitted.

Two sources record no stream position, so they are credited with what they
produced and nothing more -- placing an unknown position at file 0 would divide
a whole build's tokens by one file's bytes and read `stack-edu-python`'s 1.2B as
roughly 13B. For that source the conservatism happens to be exactly right: the
Hub resolves its config to 10 files and the original build's recorded position
was file 10. Its 139M shortfall on 2026-08-10 was permanent, and it still is.

Two independent cross-checks that the extrapolation lands in the right place:
the measured reach for `fineweb-edu` is 1,424B against a published ~1.3T for
that dataset, and for `dclm-baseline` 3,746B against a published multi-trillion
corpus.

## Table 1: the largest total budget each source can feed at four epochs

`4 x unique / share`. Worst first, so the first row is the corpus ceiling and
the rest are the order in which sources fail as a successor grows.

| source | share | unique tokens | supports a total budget of |
| --- | ---: | ---: | ---: |
| everyday-conversations | 0.020 | 0.0004B | **0.08B** |
| stack-edu-python | 0.090 | 1.21B | **53.8B** |
| finewiki-en | 0.030 | 6.76B | 902B |
| cosmopedia-v2 | 0.050 | 24.72B | 1,977B |
| finepdfs-edu | 0.080 | 56.94B | 2,847B |
| infiwebmath-3plus | 0.030 | 21.54B | 2,873B |
| finemath-3plus | 0.030 | 34.51B | 4,602B |
| fineweb-edu | 0.375 | 1,424.32B | 15,193B |
| dclm-baseline | 0.225 | 3,745.79B | 66,592B |
| finephrase | 0.070 | 1,266.27B | 72,359B |

## Table 2: the curve

| budget | sources over the cap | unique tokens to add | ... excluding dialogue |
| ---: | ---: | ---: | ---: |
| 30B | 1 | 0.15B | **0** |
| 60B | 2 | 0.44B | 0.14B |
| 100B | 2 | 1.54B | 1.04B |
| 200B | 2 | 4.29B | 3.29B |
| 500B | 2 | 12.54B | 10.04B |
| 1,000B | 3 | 27.03B | 22.03B |

At 1T the 22.03B that is not dialogue is 21.29B of code and 0.74B of wiki.
Every other source still has slack at 1T.

## What would have to grow, and by how much

1. **Code is the whole constraint.** `stack-edu-python` holds 1.21B unique
   tokens and cannot yield another document. At a 9% share it needs 1.35B at
   60B, 2.25B at 100B, 4.5B at 200B, 11.25B at 500B and 22.5B at 1T to stay at
   four epochs. Phase 8's code corpus -- permissively licensed, eight languages,
   repository-split -- is where that growth has to come from, and this is the
   number it has to hit. It is the reason to build it even for a general
   successor.
2. **`everyday-conversations` cannot be fixed, only removed.** 403,573 unique
   tokens against a 2% share is 1,487 epochs at 30B and 49,557 at 1T. The plan
   already moves dialogue to SFT; this is the measurement that says the cap
   is not a tuning question.
3. **`finewiki-en` needs +0.74B, and only past ~900B.** Below that it is fine.
4. **Nothing else needs a new source below 1T.** Web, PDF and math need
   *re-streaming*, not new datasets -- a throughput and disk question rather
   than a data-availability one.

## Two things this does not say

**The aggregate is not the number to steer by.** At 1T the corpus-level ratio
is 0.2 epochs while three sources are over the cap, because a mixture cannot
spend `dclm-baseline`'s headroom on `stack-edu-python`'s shortfall. The report
carries the binding source beside the aggregate for that reason.

**Stronger dedup will lower these numbers, not raise them.** Supply is counted
after the released build's dedup and decontamination, which were memory-bounded
and reset periodically -- so exact duplicates outside a reset window were kept.
This phase's own mandate is to make those hashes persistent across the build and
to share near-duplicate groups across overlapping web sources, which by
construction keeps *fewer* tokens. The reach figures above should be re-measured
after the rebuild rather than carried forward.

Finally, the reachable component is a lower bound in construction but an
extrapolation in substance: it assumes untouched files resemble the touched
prefix in density and filter pass rate. A source is only *proved* to hold what
it realized, which is the floor recorded in `headroom-curve-realized.json`.
