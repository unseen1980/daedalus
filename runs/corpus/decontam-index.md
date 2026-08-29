# Phase 7: the decontamination index, frozen and complete

Built 2026-08-26 by `scripts/decontam_index.py build`. Provenance in
`data/decontam/eval-index-13gram.txt.gz.json`; the verification pass is
`runs/corpus/decontam-index.json`. The index itself is 31 MB gzipped and is not
in git — its digest makes it reproducible from the command above.

## Headline

**1,371,773 13-grams over all 16,023 scored items, on all five scored splits**,
against the **214,682** the same code produces at the build's `limit=2000`.
6.39x the n-grams from 2.11x the items, because the coverage that was missing
was HellaSwag's, and HellaSwag items are the long ones.

| | items indexed | items in the scored split | covered |
| --- | ---: | ---: | ---: |
| hellaswag | 2,000 → **10,042** | 10,042 | 19.9% → **100%** |
| arc_easy | 2,000 → **2,376** | 2,376 | 84.2% → **100%** |
| piqa | 1,838 | 1,838 | 100% |
| openbookqa | 500 | 500 | 100% |
| winogrande | 1,267 | 1,267 | 100% |
| **total** | 7,605 → **16,023** | 16,023 | 47.5% → **100%** |

Against what the released corpus was *actually* filtered with — 183,359 grams,
with ARC-Easy and OpenBookQA on `validation` — the frozen index is **7.5x**
larger and is the first one built on the splits the model is scored on.

The 214,682 figure reproduces the number `scripts/contam_scan.py` records for
the same limit, from different code written for a different purpose. That
agreement is the reason the 1,371,773 can be read as a coverage measurement
rather than as this tool's opinion of itself.

## The gap that was not the coverage gap

The 2,000-item limit is the visible hole and `contam_scan.py` already measured
what came through it. The one underneath it is that **nothing recorded which
index a source was filtered against**. `334c86c` (2026-08-09 18:09Z) moved
ARC-Easy and OpenBookQA onto their scored `test` splits mid-corpus; establishing
which sources predate it meant rebuilding the index at the old splits and
matching a gram count that happened to have been logged. That worked once. It
is not a procedure.

The cause is that the index was derived at the start of every build: it is a
function of what `datasets` returned that day and what `TASK_SPLITS` said that
week, and neither was written down. So it is now built once, sorted,
content-addressed, and recorded beside the item counts, splits and revisions it
came from. `run_dataprep --eval-index PATH [--eval-index-digest SHA]` loads it
and writes `decontam_index` into the corpus manifest, so the question is
answered by artifacts.

Two properties make the digest usable as an identity: the file is sorted, and
gzip's header carries neither the mtime nor the source filename — without the
second, the same index written to two paths produced two different files and
the digest would have named the write rather than the filter.

## Four refusals, which are the actual deliverable

`eval.load_all_tasks` skips a benchmark it cannot load, with a warning. That is
right for scoring and wrong for an index: a HellaSwag outage would produce an
index that looks fine, filters nothing against HellaSwag, and leaves no trace in
the corpus — the released failure arriving by a different road. So `build_index`
refuses, naming every problem at once:

1. **a missing task** — `load_all_tasks` returns no entry for one that failed;
2. **a task at zero items**;
3. **a split other than the one that task is scored on** — 334c86c as a rule
   rather than as archaeology;
4. **a task that merely came back short**. The Hub is read unauthenticated on
   this box, so a rate-limited split returns fewer items rather than failing,
   and none of the first three guards sees it. `EXPECTED_ITEMS` pins the five
   sizes; an index built from a truncated split would otherwise be smaller,
   marked complete, digested, and used, with its own provenance asserting the
   opposite.

A limit is refused too, unless asked for twice (`--limit` plus
`--allow-partial`), and a partial index stops `run_dataprep` rather than being
recorded and used. Freezing an index does not make it complete.

`coverage_problems` asks the same questions of a frozen file months later,
which is a different question and the one that goes stale: a task added to
`TASK_LOADERS`, or a split that grows, is invisible to the build that used the
index and visible to `verify`.

## What it costs the rebuild

**241 MB resident, once per dataprep worker** (measured by
`decontam_index.py verify`, not estimated). `run_dataprep` forks one worker per
source group against a 4.0 GB per-worker cap on a measured ~2.6–2.9 GB
baseline, so this spends roughly 200 MB more than the old index did and leaves
~0.9–1.2 GB of margin at the default `--max-workers 4`. It fits, but it is
close enough to the cap to be a number the rebuild should have in advance
rather than discover — see `run_dataprep`'s RAM discipline note.

## What this does not say

It does not say the released corpus is clean. It says the *next* build filters
against every scored item on every scored split, and records which index it
used. What the released corpus let through is `contam_scan.py`'s measurement
and is unchanged by this.

It also does not rebuild anything on its own. The index is an input; the
rebuild that consumes it is the rest of phase 7.
