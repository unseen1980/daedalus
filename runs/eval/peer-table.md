### Measured on this harness, full validation splits, fp32

| model | hellaswag | arc-easy | piqa | openbookqa | winogrande | our 5-task mean | their published 8-task avg | tokens |
|---|---|---|---|---|---|---|---|---|
| EleutherAI/gpt-neo-125m | 30.4 | 39.4 | 62.6 | 26.2 | 51.0 | **41.9** ± 0.58 | 42.9 | 300B |
| EleutherAI/pythia-160m | 30.4 | 37.5 | 60.6 | 25.0 | 51.5 | **41.0** ± 0.57 | 42.5 | 300B |
| facebook/opt-125m | 31.4 | 39.9 | 62.0 | 27.8 | 49.4 | **42.1** ± 0.58 | 42.6 | 180B |
| HuggingFaceTB/SmolLM2-135M | 43.2 | 58.8 | 68.7 | 33.0 | 52.2 | **51.2** ± 0.59 | 50.7 | 2T |
| openai-community/gpt2 | 31.2 | 40.1 | 62.1 | 27.6 | 49.8 | **42.2** ± 0.58 | - | - |
| **Daedalus-150M (sweep probe, undertrained)** | 27.3 | 38.7 | 56.0 | 28.4 | 50.7 | **40.2** ± 0.59 | - | 0.5B |

± is the binomial standard error of the 5-task mean (`sqrt(p(1-p)/n)` per task, `sqrt(sum)/5` for the mean). A difference must reach ~1.65 points to be two of these sigmas, so most gaps in this table are ties. Sampling error only — it does not bound seed variance, which is unmeasured here and larger. See `scripts/eval_noise.py` and `scripts/mcnemar.py` for the paired test, which is tighter.

**Read the Daedalus row as a floor, not a claim.** It is `sweep` probe 2's
0.5B-token checkpoint — **1.25% of `hero`'s 40B budget** — scored only to prove
the harness end to end on our own weights before the gate asks for $43.70. The
finished model is not this. What it establishes is that the pipeline is not
broken: chance on this metric is **35.0**, and 40.2 sits 5.2 points above it,
with real movement on the three tasks that discriminate (ARC-Easy +13.7 over
chance, PIQA +6.0, HellaSwag +2.3) and none on the two that do not (OpenBookQA,
WinoGrande — where the 300B peers are also at chance).

Do not read the ARC-Easy row as being ahead of schedule. Across 0.5B→300B — a
600× token increase — this column barely moves (43.6 acc ours, 44.2 GPT-2, 43.7
GPT-neo, 43.5 OPT, 37.5 Pythia), so it is ordered by data and recipe rather than
by budget; only SmolLM2's 2T breaks the cluster. The gap that more tokens have
to close lives in **HellaSwag and PIQA**.


### Our measurement minus the published table (points)

| model | hellaswag | arc_easy | piqa | openbookqa | winogrande |
|---|---|---|---|---|---|
| EleutherAI/gpt-neo-125m | +0.7 | -1.3 | +0.1 | -5.4 | +0.3 |
| EleutherAI/pythia-160m | +0.5 | -2.5 | -1.4 | -6.2 | +0.6 |
| facebook/opt-125m | +0.3 | -1.4 | -0.0 | -3.4 | -1.4 |
| HuggingFaceTB/SmolLM2-135M | - | - | - | - | - |
| openai-community/gpt2 | - | - | - | - | - |
| **Daedalus-150M (sweep probe, undertrained)** | - | - | - | - | - |

### CPU decode, Q4_0, llama.cpp defaults — measured 2026-08-10

Quality is only half the stated bar; the other half is CPU decode, and this
table had no decode number for any peer. First one measured
(`decode-vs-smollm2.md`, raw in `decode-vs-smollm2.json`), 8 threads,
3 alternating rounds, `llama-bench -p 0 -n 128`:

| decode at depth | Daedalus-150M (160.5M, 102.0 MB) | SmolLM2-135M (135M, 91.7 MB) | ratio |
|---|---|---|---|
| 0 | 960.9 ± 3.4 | 908.1 ± 37.1 | 1.06x |
| 512 | 933.7 ± 28.5 | 625.7 ± 38.5 | 1.49x |
| 2048 | 648.6 ± 12.6 | 312.4 ± 7.2 | **2.08x** |

Read the depth column, not the headline: at depth 0 -- what a default
`llama-bench` invocation measures -- the gap is 1.06x and not worth claiming.
It opens with context because only 6 of our 18 blocks keep a KV cache against
all 30 of theirs, i.e. 3.75x less cache re-read per generated token. Absolutes
were taken while `sweep` trained on the GPU and are comparable only within that
invocation; the ratio is the durable part. Caveats, including the Q6_K vs Q8_0
embedding asymmetry that `llama-quantize` chooses on its own, are in the
writeup.

## The paired instrument has no peer side yet

`README.md` and `STATUS.md` both lean on a paired comparison as *"the one a
~1-point claim needs"* — correctly, because this table's four beatable peers sit
inside a 1.2-point band and the unpaired 2σ threshold is 1.65 points, i.e. wider
than the entire band. `eval.py` writes the per-item sidecar that makes pairing
possible, and `scripts/mcnemar.py` consumes it.

**But the five peer results in this table predate that feature.** They were
scored 2026-08-09 18:18–18:49; the sidecar landed this morning. So:

```
runs/eval/peer-*.json         5 files, present
runs/eval/peer-*.items.json   none
```

**A paired Daedalus-vs-peer test cannot be run today**, and nothing would have
said so until the moment the final claim was made — the tool exists, the
Daedalus side will have its sidecar, and the failure would look like a missing
file at the end of a four-day run rather than a decision not taken now.

**Cost to close: ~25 min of GPU and ~$0.19** (the six models took 18:18→18:49
sequentially), and no code change — `--per-item` is derived from `--out`, so
re-running `scripts/eval_peers.sh` writes the sidecars by itself.

**It needs an idle GPU**, which is the constraint that schedules it. Arm 1 holds
32.4 GB of 32.6 (236 MiB free — measured, see `STATUS.md`), so this belongs in
the same post-arm-2 idle window already reserved for the 0.5B paired re-score,
not alongside a training arm.

Two things to do at the same time, since the re-run is what makes them free:

- **Score `openai-community/gpt2` from the script rather than by hand.** It was
  missing from the loop until 2026-08-10 and is the peer that *sets* the 42.2
  bar, so the documented reproduction command rebuilt every peer except that
  one. Fixed in `scripts/eval_peers.sh`.
- **Drop `--no-wandb` for that run** if the peer bar is wanted on the operator's
  phone alongside the training runs. Not changed unilaterally: it is a
  reporting preference, not a correctness bug, and the flag is deliberate
  (these are baselines, not project runs).

Until this is done, every peer comparison in this repo is **unpaired**, and the
±0.59 / 1.65-point arithmetic quoted beside it is the honest instrument.
