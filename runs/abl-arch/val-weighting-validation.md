# val_bpb: validated on real data before `abl-arch` runs

Run 2026-08-09 ~20:40Z while `dataprep-13b-attempt10` was still writing, against
the real corpus and a real `daedalus-150m` checkpoint (`runs/sweep-lr0.01`,
0.5B tokens). Bounded to 6 batches/source to stay clear of dataprep's RAM.

## The weighting bug, quantified

`eval_val_bpb` weighted each source by its **holdout token count**.
`make_holdout_split` reserves whole shard *files*, so that share is set by the
arbitrary size of a source's trailing partial shard, not by the mixture.

| source | val bpb | mixture weight (new) | holdout weight (old) |
|---|---|---|---|
| fineweb-edu | 1.0928 | **38.27%** | 13.46% |
| dclm-baseline | 1.3944 | **22.96%** | 9.72% |
| stack-edu-python | 1.9902 | 9.18% | **18.22%** |
| finepdfs-edu | 1.3337 | 8.16% | 6.83% |
| finephrase | 1.1140 | 7.14% | 14.82% |
| cosmopedia-v2 | 1.0617 | 5.10% | 8.30% |
| finemath-3plus | 1.5651 | 3.06% | 10.55% |
| finewiki-en | 1.2853 | 3.06% | 8.73% |
| infiwebmath-3plus | 1.7565 | 3.06% | 9.39% |

Headline effect on the same checkpoint:

- old, holdout-token weighted: **1.4315**
- new, mixture weighted: **1.3047**
- **+0.1268 bpb, +9.7%** — the old number was inflated by over-weighting the
  hardest sources (code 1.99, math 1.76) far above their training share.

Both `abl-arch` arms would have shared the distortion, so the hybrid-vs-dense
*ranking* was never at risk. The reported figure simply did not describe the
blend either arm trained on.

The new weights reproduce the blueprint mixture as intended — fineweb-edu
38.3% against its 37.5% target, dclm-baseline 23.0% against 22.5% — which is
the check that `mixture_sampling_weights` really mirrors `MixtureBatchSource`.

## Sanity of the per-source numbers

Ordering is what a 0.5B-token model should give: code hardest (1.99), then
math (1.76 / 1.57), synthetic textbook prose easiest (1.06), web in between.
Nothing is NaN, inf, or suspiciously equal across sources.

## Cost of the step, for the `abl-arch` schedule

265K tok/s over the 609M-token holdout (bf16 autocast, batch 8, seq 2048,
GPU peak 2.33 GB, sharing the box with dataprep):

- **~38 min per arm, ~77 min for both** — `abl_arch.eval_val_bpb` runs a full
  pass (`max_batches=None`).
- Add ~1.3 h to the ~24 h `abl-arch` estimate: **~$0.58**. Left as a full pass;
  a bounded sample would save under a dollar and cost precision on the number
  the writeup quotes.

`train.py`'s periodic val stays bounded at `val_batches=8` — now 8 batches
*per source* (~72 batches, ~1.2M tokens, a few seconds), so a `hero` eval
interval samples every source instead of whichever one it was pointed at.
