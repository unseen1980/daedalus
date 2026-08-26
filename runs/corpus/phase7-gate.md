# Phase 7: the acceptance list, decided from the corpus on disk

Measured 2026-08-26 with `scripts/corpus_gate.py`. Exact numbers in the
`runs/corpus/phase7-gate-*.json` verdicts — one per budget, named for it, with
`-no-dialogue` for the arms that drop `everyday-conversations`. The exit status
of each run is the verdict, so a controller gates on it without reading this
page.

Phase 7's acceptance is five claims. Each is a property of files that exist — a
frozen n-gram index, a scan artifact, ten `manifest.json`s — so each is read
rather than asserted. Two of the five pass today, one is a measured finding, and
two fail in ways this phase's own remaining work closes.

## Verdicts on the corpus as built, at the released run's 59.9B budget

`runs/corpus/phase7-gate-59.9b.json`, blueprint mixture, four-epoch cap.

| criterion | verdict | what decided it |
| --- | --- | --- |
| `decontam-index-complete` | **PASS** | 1,371,773 n-grams over all five scored tasks at their scored splits, no per-task limit |
| `corpus-contamination` | **FAIL** | 1 document hit `split_gap`, 1 hit `limit_gap`; and 157,921,561 `fineweb-edu` tokens were never in front of the scanner |
| `epoch-cap` | **PASS** | worst source 4.000 epochs; 7 of 10 sources pinned at the cap |
| `mixture-skew` | **FAIL** | 11.4593 pts from the blueprint, against a 5-pt bound |
| `manifest-provenance` | **FAIL** | 10 of 10 manifests carry no revision, no filters block and no builder sha |

## The two failures phase 7's own work closes

**Contamination.** The two hits are not a defect in the pipeline; they are the
two gaps this phase was written to close. `docs_filtered` is **0** — the negative
control — so `dataprep` removed everything it indexed. What it did not index was
ARC-Easy and OpenBookQA `test` (it indexed their `validation` splits) and
everything past `--eval-task-limit 2000` (HellaSwag was 19.9% covered). Those are
the `split_gap` and `limit_gap` rows, and the frozen index built this phase —
which the first criterion passes on — has neither. The hits are therefore a
property of the corpus **as built**, and are cleared by the rebuild in step 9,
not by any further indexing work.

The scan-coverage failure is new information. A scan artifact names no shard
tree, so nothing linked `runs/preflight/contam-exposure.json` to the corpus it is
cited about except the extent it recorded per source. Nine of ten sources match
their manifests exactly. `fineweb-edu` does not: the scan saw 5,035,572,292
tokens where the source now holds 5,193,493,853, so **158M tokens — 3.0% of the
largest source — are covered by a clean verdict that never read them.** That gap
was invisible until the two artifacts were compared, and it is now a criterion.

**Provenance.** All ten manifests predate `dataprep.source_provenance`. They name
a dataset and a stream position and say nothing about which revision of that
dataset was read, which filters ran, or which tree ran them. That is precisely
the hole `source_provenance` was added to close this phase; the next build
satisfies the criterion and this one cannot be made to retroactively.

## The mixture finding, which is not closed by anything in phase 7

The skew is the one criterion whose failure is a *measurement* rather than a
known gap, so it is worth the detail.

| budget | blueprint | dialogue dropped |
| ---: | ---: | ---: |
| 30B | 3.9892 | **0.0000** |
| 40B | 3.9919 | |
| 45B | 3.9928 | |
| 50B | 3.9935 | **0.0000** |
| 55B | 4.3801 | 1.7422 |
| 56B | **5.5020** | |
| 57B | | **5.0108** |
| 58B | | 6.5606 |
| **59.9B** | **11.4593** | 10.6180 |

L1 skew in points; **bold** is the first budget over the 5-pt bound in each
column. Every row passes `epoch-cap` — the worst source sits at exactly 4.000
epochs from 30B up, and at 2.275 at 30B with dialogue dropped.

**The corpus as built delivers the blueprint within the bound to ~55.4B, and to
~56.9B with the dialogue source removed.** The released run's 59.9B is past both.

Three things this says.

1. **The released run's own budget is outside the bound this phase
   preregisters.** At 59.9B the corpus delivers 11.46 pts of skew: seven of ten
   sources are pinned at four epochs, `fineweb-edu` falls from a 37.5% target to
   34.7% and `everyday-conversations` from 2% to 0.003%, and that mass
   water-fills onto `finephrase` (+2.33 pts) and `cosmopedia-v2` (+1.34 pts).
   Repetition is still bounded — the cap worked — but the blueprint is not what
   gets sampled. This is an independent measurement of the same phenomenon
   `daedalus/data.py::select_holdout_shards` records at 10.21 pts for the
   post-carve mixture at 60B; this figure is pre-carve, so the two are not the
   same number and should not be quoted as one.

2. **Below ~53B the entire skew is one source.** The 3.99 pts floor at every
   budget from 30B to 50B is `everyday-conversations` alone: 403,573 tokens
   cannot fund a 2% share, so 2 points leave and 2 points redistribute. Drop it
   and the skew is **exactly 0.0000** at both 30B and 50B, with the worst source
   at 2.275 epochs at 30B. That is the plan's step 4 measured rather than
   argued, and it is a complete fix while it applies.

3. **It stops being a fix at almost exactly the point the code source runs
   out.** Removing dialogue moves the ceiling from ~55.4B to ~56.9B — about
   1.5B of budget — and at 59.9B buys 0.84 pts of the 6.46 it would need. The
   2% it frees is renormalized onto sources that are themselves at the cap and
   cannot absorb it, so one kind of skew converts into another. That knee sits
   just above the 53.8B ceiling `runs/corpus/headroom-curve.md` derives from
   `stack-edu-python` having no more documents, and the two are the same fact:
   past it the binding constraint is supply, which is step 5's top-up and phase
   8's code corpus, not a source removal.

## The trap this gate is built around

`l1_skew_pts` sees only one of `cap_weights_by_epochs`'s two failure modes. When
the cap binds it reweights and the skew rises. When *no* allocation satisfies the
cap — every source over the limit — the target shares are returned unchanged and
the skew is **0.00 by construction**, its best possible value, at the one budget
where repetition is bounded by nothing. So the skew criterion carries a fallback
guard: after a successful cap no source exceeds the limit, making
`max_epochs_seen > max_epochs` true precisely in the unbounded case.

`runs/corpus/phase7-gate-as-built.json` is that guard firing on real data. It is
the same 59.9B question asked with `--local-supply`, i.e. counted against the
shards physically present on this box rather than the sources they were fetched
from. Every source is then over the cap, the worst at **2,968 epochs**, and the
skew reads **0.0000**. A gate reading the skew alone would have given that its
cleanest verdict.

That file is also why `--local-supply` is not the default. A shard directory here
is a fetch — `subset_of` records what from — and counting the local shards
understates supply by 10x to 30x, which is what turned a corpus comfortably
inside the epoch cap into one that appeared to blow through it tenfold on the
first run of this gate.

## What this does not say

The corpus-contamination criterion is decided from a **sampled** scan: 1.32% of
corpus tokens by systematically-spaced windows. A hit is decisive, a zero is not
— it bounds the document rate at 3.2e-05 (95% upper) rather than proving
absence. Every passing verdict carries that bound in its `detail`.

Nothing here is a statement about the corpus phase 7 recommends. It is the
corpus as built, which is the only one that exists; the rebuild is step 9 and
clears two of these three failures by construction.
