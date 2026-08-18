# What Q4_0 costs on a *task*, not on perplexity — measured 2026-08-10

Every quality number this project quotes describes the fp32/bf16 PyTorch model.
The artifact anyone actually runs is a Q4_0 GGUF, and `eval.py` has no GGUF
path, so the gap between **what we benchmark** and **what we ship** had never
been measured on a task. `scripts/gguf_task_delta.py` existed for this and had
never been run against a real checkpoint.

Run on `runs/sweep-wsdfix-lr0.02/checkpoint.pt` — 0.5B tokens, fully annealed,
the architecture and tokenizer `hero` will use — through the real `export.py`
and the real `llama-quantize`. CPU only, 6 of 16 threads, alongside `abl-arch`
arm 1; it cost about 2% of training throughput for an hour (~$0.01) and no GPU.

## The result

| | perplexity (finewiki 150k, `-c 512`) | HellaSwag, all 10,042 items |
|---|---|---|
| fp16 | 29.1321 | 26.827 (CI 25.970–27.703) |
| Q4_0 | 29.5833 | 27.056 (CI 26.196–27.934) |
| **delta** | **+1.549%** | **+0.229 points** |

**Q4_0's perplexity cost does not show up as task damage here.** It is nominally
*better*, which is another way of saying the difference is not resolvable: the
two confidence intervals overlap across almost their whole width.

## The limitation, which is large enough that it changes the conclusion

This checkpoint scores **26.8 on a task whose chance floor is 25.0**. There is
almost no signal in that gap for quantization to destroy, so the test has very
little power — "no measurable damage" here is close to "no measurable anything
here". It would be wrong to read this as evidence that Q4_0 is free.

What it does rule out is a *large* task-level collapse — the failure mode where
a model that looks fine on perplexity emits degraded answers — and that was
worth ruling out before shipping.

**Repeat it on the 5B checkpoint.** `abl-arch` arm 1's export produces exactly
these two GGUFs at ~17:13Z today for free, on a model with roughly ten times the
tokens and further above the floor, where the test has real power. That is the
number to quote, and this one exists mainly so the method is proven and the
tooling is exercised before it matters.

## Two things this does not say

**It does not weaken the case for QAT.** The perplexity cost is real and
measured (+1.549%, above `export.py`'s 1.0% gate), and QAT is what closes it.
What it does is stop the perplexity number being *over*-read: a 1.5% perplexity
delta is not yet evidence of a 1.5-point benchmark delta, and nobody should
present it as one.

**It is not comparable to the peer table.** llama.cpp normalises each
candidate's summed logprob by **token count** (`perplexity.cpp`,
`log_prob / count`); lm-evaluation-harness `acc_norm` — which every published
peer number uses, and which `eval.py` reproduces — normalises by the **byte
length of the choice**. Different denominator, different metric. That is why
26.827 here sits below `eval.py`'s 27.3 for the same checkpoint, and why only
the *paired delta between two GGUFs of one model* means anything in this file.

## Correcting an expectation that was written down

`runs/preflight/gguf-tokenizer-and-q4-damage.md` recorded +2.576% on a
**non-annealed** 500M checkpoint at lr 0.02 and predicted that annealing would
reduce it, since settled weights quantize better. The like-for-like re-run —
same lr, same token count, only the WSD fix — gives **+1.549%**, a 40%
reduction, in the predicted direction.

**But one pair does not establish the mechanism**, and the third measurement on
disk says so: `runs/preflight/token-embd-quant-grid.md` recorded **+1.558%** on
a *non-annealed* checkpoint at lr 0.01. So the non-annealed population spans
1.56–2.58% and the annealed point lands at the bottom of it rather than below
it. Checkpoint-to-checkpoint variation is of the same order as the effect.

The conclusion that survives all three: **every real checkpoint measured so far
exceeds the 1.0% gate before QAT**, which is the load-bearing fact and is what
made QAT worth protecting.
