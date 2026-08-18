# The shipped GGUF, checked against the corpus that trained it

2026-08-09 ~23:35 UTC. Cost: ~5 min of an idle GPU plus ~2 min of CPU, under $0.05.
Artifacts: `runs/sweep-lr0.02/checkpoint.pt` (real, 500M tokens) through the
real `export.py` and the real `llama-quantize`.

> **Correction, 2026-08-10 ~14:40Z.** Section 2's **+2.576%** was measured on an
> eval text that then contained 153 literal `<|endoftext|>` strings, which
> `llama-perplexity` does not parse and instead spelled out as ~7 junk tokens
> each. That made absolute perplexity read ~9% high and the Q4_0 delta read
> **~6% low**. So +2.576% is an **underestimate** — the conclusion below (Q4_0
> damage exceeds the 1.0% gate, QAT is load-bearing) holds and is strengthened.
> The file is fixed; the numbers here are not re-measured, because the
> checkpoint they used is superseded. → `gguf-vs-pytorch-fidelity.md`

## 1. Why the tokenizer needed checking, when every gate already passed

`export.py`'s Q4_0 gate compares fp16 perplexity against Q4_0 perplexity. Both
sides use the **same** tokenizer, so a wrong vocab cancels out exactly and the
gate passes. `llama-bench` only times decode, so it passes too. Nothing in the
pipeline compares the ids the GGUF assigns against the ids the *corpus* was
built with — and a mismatch there ships a model that emits fluent garbage while
every number in the writeup looks healthy.

Read straight out of the produced `model-q4_0.gguf`:

| | |
|---|---|
| architecture | `lfm2` |
| tokenizer model / pre | `gpt2` / **`smollm`** |
| vocab size | 49,152 |
| bos / eos / unk | 0 / 0 / 0 |
| **id -> token mismatches vs `HuggingFaceTB/SmolLM2-135M`** | **0 of 49,152** |

The vocabularies are identical, id for id, and the registered `smollm`
pre-tokenizer is the one llama.cpp applies. So a token id means the same thing
to `llama.cpp` as it did to `dataprep`. This was the last unverified link
between the corpus and the file the operator will actually run.

## 2. Q4_0 costs 2.58% perplexity on a real model, not the 0.03% smoke suggested

The `abl-arch` end-to-end run reported fp16-vs-Q4_0 deltas of **0.035%** and
**0.022%** and I recorded those as the gate passing. They are not evidence of
anything. Those arms had trained for a handful of smoke steps, so their weights
were still near-Gaussian at initialisation scale, and near-Gaussian weights
quantize almost perfectly. A genuinely trained model has structured,
heavier-tailed weights that do not.

On a real checkpoint (500M tokens), `--ppl-text-file data/eval/ppl-finewiki-150k.txt`,
`n_ctx` 512:

| | |
|---|---|
| fp16 perplexity | 33.8535 |
| Q4_0 perplexity | 34.7257 |
| **delta** | **+2.576%** |
| `passes_threshold` (< 1.0%) | **false** |

### What this changes

**It makes QAT load-bearing rather than marginal.** The operator's approved
§4.4 plan protected QAT; this is the first measurement that says why. Gemma-3's
QAT playbook reports a ~54% cut in Q4_0 damage, which would put this at ~1.2% —
still near the threshold, but a different model from shipping 2.6%.

**It predicts tonight's `abl-arch` output, so a warning is not read as a
failure at 06:00Z.** Both arms run `qat_frac=0` by design — a quantized forward
would invalidate the hybrid-vs-dense comparison — so both will very likely
report `passes_threshold: false`. **This aborts nothing**: `passes_threshold`
is recorded in `results.json` and printed as a WARNING, and is fatal in neither
`export.py:329` nor `abl_arch.py:234`. Verified by reading both, and by this
run exiting **0** with the gate failed.

### Do not over-read 2.58% as hero's number

This checkpoint comes from the **first** sweep, trained under the WSD schedule
that never annealed — its weights are still hot from a constant high LR.
Annealing to lr~0 is exactly what settles weights, and settled weights quantize
better. So 2.58% is most likely an over-estimate of what `hero` will see, and
the honest statement is: the real figure is **not** the 0.03% the smoke implied,
and it is not yet known. `hero` + QAT produces the number that counts.

## 3. Two things confirmed in passing

**The decode headline reproduces on trained weights.** 968.2 ± 15.4 tok/s at 8
threads, against the 966.0 ± 29.6 measured on random init at the same thread
count. The speed claim does not depend on the weights, as expected — but it had
only ever been measured on random init, and now it has not.

**`export.py` skips GGUF conversion entirely without `--ppl-text-file`.**
Running it without that flag prints `no --ppl-text-file given; skipping GGUF
conversion/quantization check` and produces only the HF directory — the ship
artifact is gated behind a flag whose name suggests it controls a *check*. It
is explicit rather than silent, and every live caller passes it (the chain
passes it to `abl_arch.py`), so nothing is broken today. Recorded because the
final deliverable export is the one invocation that must not get this wrong.
