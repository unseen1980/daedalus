# Contamination exposure of the built corpus — measured, not argued

Written by `scripts/contam_scan.py`. Window 131,072 tokens, up to 200 windows per source, 13-grams.

## Verdict

| index | what a hit means | 13-grams | rate over sampled training tokens | docs hit | 95% upper bound |
|---|---|---|---|---|---|
| `filtered` | **negative control** — n-grams `dataprep` did remove | 183,359 | **0.0000%** | 0 / 174,932 | 0.0022% |
| `split_gap` | scored splits the build never indexed (ARC-Easy/OpenBookQA `test` vs `validation`, pre-`334c86c`) | 47,549 | **0.0004%** | 1 / 174,932 | 0.0032% |
| `limit_gap` | eval items beyond `--eval-task-limit 2000` (HellaSwag 19.9% covered) | 1,157,091 | **0.0012%** | 1 / 174,932 | 0.0032% |

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
| fineweb-edu | 5,035,572,292 | 24,828 | 25,281,764 | 0 | 0 | 0 |
| dclm-baseline | 3,375,002,145 | 20,246 | 25,231,808 | 0 | 0 | 0 |
| finephrase | 2,066,170,024 | 25,090 | 25,535,537 | 0 | 0 | 0 |
| finemath-3plus | 1,350,008,857 | 15,233 | 24,479,053 | 0 | 0 | 1 |
| infiwebmath-3plus | 1,350,000,198 | 15,761 | 24,962,841 | 0 | 1 | 0 |
| stack-edu-python | 1,210,964,651 | 13,155 | 24,375,991 | 0 | 0 | 0 |
| finepdfs-edu | 1,200,001,535 | 6,140 | 23,433,317 | 0 | 0 | 0 |
| cosmopedia-v2 | 950,000,576 | 36,082 | 26,016,369 | 0 | 0 | 0 |
| finewiki-en | 450,005,019 | 16,199 | 24,986,143 | 0 | 0 | 0 |
| everyday-conversations | 403,573 | 2,198 | 390,603 | 0 | 0 | 0 |

## Every hit, in full

A rate cannot tell a document that reproduces an eval item from one that shares thirteen words of ordinary prose with it, and the two mean very different things for a benchmark claim.

**`limit_gap` in finemath-3plus** — 48 matching 13-gram(s)

> matched: `(30 cm) to 2 feet (61 cm) apart. Be sure your cuts are`

> context: …direction the wood is laying. Using a circular saw, make single cuts straight across the flooring that are about 1 foot (30 cm) to 2 feet (61 cm) apart. Be sure your cuts are perpendicular to the direction of the wood. Start on 1 side of the room and work your way systematically to the other side, spacing each cut about 1 foot (30 cm) to 2 feet (61 cm) apart.[3] My advice is to bring in a professional to make sure the floors can be refi…

**`split_gap` in infiwebmath-3plus** — 1 matching 13-gram(s)

> matched: `acceleration of an object is determined by the mass of the object and`

> context: …# 2.1 Understanding acceleration   Page 1 / 3 Acceleration exists where net external force exists. The acceleration of an object is determined by the mass of the object and net external force applied. Most important aspect of acceleration is that it is independent of motion i.e. velocity.  The relationship among velocity, acceleration and…

Sampled 224,693,426 tokens of 16,988,128,870 (1.323% of the corpus). 11,452,655 of 236,322,816 scanned tokens (4.85%) sat in a partial document at a window edge and were excluded — that is the size of the documented downward lean on long documents.

