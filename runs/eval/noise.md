# Eval sampling noise

Standard error of each task's accuracy (`sqrt(p(1-p)/n)`) and of the 5-task mean. **Benchmark sampling only** — this does not bound seed variance, which nothing in this project has measured.

| model | hellaswag | arc_easy | piqa | openbookqa | winogrande | **5-task mean** |
|---|---|---|---|---|---|---|
| us @0.5B | 27.3 ± 0.44 | 38.7 ± 1.00 | 56.0 ± 1.16 | 28.4 ± 2.02 | 50.7 ± 1.40 | **40.22 ± 0.59** |
| Pythia-160M | 30.4 ± 0.46 | 37.5 ± 0.99 | 60.6 ± 1.14 | 25.0 ± 1.94 | 51.5 ± 1.40 | **41.00 ± 0.57** |

`us @0.5B`: OpenBookQA is 500 items, so one question is 0.2 points of that column alone.

**Pythia-160M − us @0.5B = +0.78 points**, against an unpaired standard error of ±0.82 (1.0σ).

That σ is an **upper bound**: both models answer the same items, so the paired error is smaller by an amount only per-item outputs could quantify. A difference that clears it is real; one that does not is simply unresolved by this measurement.

For scale, a difference must reach **1.64 points** to be two of these sigmas.
