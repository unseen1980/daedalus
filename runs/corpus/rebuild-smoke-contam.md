# Contamination exposure of the built corpus — measured, not argued

Written by `scripts/contam_scan.py`. Window 32,768 tokens, up to 120 windows per source, 13-grams.

## Verdict

| index | what a hit means | 13-grams | rate over sampled training tokens | docs hit | 95% upper bound |
|---|---|---|---|---|---|
| `filtered` | **negative control** — n-grams `dataprep` did remove | 183,359 | **0.0000%** | 0 / 375 | 1.0140% |
| `split_gap` | scored splits the build never indexed (ARC-Easy/OpenBookQA `test` vs `validation`, pre-`334c86c`) | 47,549 | **0.0000%** | 0 / 375 | 1.0140% |
| `limit_gap` | eval items beyond `--eval-task-limit 2000` (HellaSwag 19.9% covered) | 1,157,091 | **0.0000%** | 0 / 375 | 1.0140% |

`filtered` is the load-bearing row and the reason the other two can be believed: those n-grams *were* removed at build time, so a non-zero rate there would mean this scan and the pipeline disagree about what a document is — and the exposure rows would be measuring that disagreement rather than the corpus.

## Coverage the build actually had

| task | items indexed | items in the scored split | covered | split the build indexed |
|---|---|---|---|---|
| arc_easy | 2,000 | 2,376 | 84.2% | validation |
| hellaswag | 2,000 | 10,042 | 19.9% | (as scored) |
| openbookqa | 500 | 500 | 100.0% | validation |
| piqa | 1,838 | 1,838 | 100.0% | (as scored) |
| winogrande | 1,267 | 1,267 | 100.0% | (as scored) |

## Per source

| source | source tokens | sampled docs | doc tokens | `filtered` | `split_gap` | `limit_gap` |
|---|---|---|---|---|---|---|
| everyday-conversations | 69,438 | 375 | 64,815 | 0 | 0 | 0 |

Sampled 64,815 tokens of 69,438 (93.342% of the corpus). 344 of 65,536 scanned tokens (0.52%) sat in a partial document at a window edge and were excluded — that is the size of the documented downward lean on long documents.
