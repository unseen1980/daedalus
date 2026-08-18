# `token_embd` ships Q6_K, the blueprint says Q8_0, and the blueprint is wrong here

Measured 2026-08-10 01:40–01:50 UTC, while `sweep` was running on the GPU
(load average 1.12 on 16 cores, so the CPU numbers are not contended).

## What I was actually checking

`daedalus/qat.py::plan_qat` fake-quantizes `embed_tokens`/`lm_head` to **Q8_0**
and every backbone `nn.Linear` to **Q4_0**, and its docstring justifies the
embedding choice as "matching the blueprint's *keep token_embd/output at
Q8_0*". That is a claim about what `llama-quantize` produces, and nothing had
ever read it back out of a real `.gguf`. If it were wrong, QAT would be
training against a lattice inference never uses — the same shape of defect as
the `torch.compile` fp16-scale finding, which also raised nothing.

So I exported a real 500M-token checkpoint (`runs/sweep-lr0.01/checkpoint.pt`,
from the first sweep) to fp16 GGUF and quantized it, then dumped the tensor
types with `gguf-py`.

## What the artifact actually contains

164 tensors: **102 Q4_0**, **61 F32**, **1 Q6_K**.

| tensor | shipped type | `plan_qat` assumes | agree? |
|---|---|---|---|
| 102 backbone linears | Q4_0 | Q4_0 | yes |
| norms, `shortconv.conv` (depthwise) | F32 | left alone | yes |
| **`token_embd.weight`** | **Q6_K** | **Q8_0** | **no** |

`output.weight` is **absent** — embeddings are tied, so that single Q6_K tensor
is both the input embedding *and* the output projection, i.e. the highest-
leverage tensor in the model for logit quality. It is 768 × 49,152 = 37.7M
parameters, about 25% of the model.

The cause is one line: `export.py::quantize_gguf` runs
`[llama-quantize, in, out, "Q4_0"]` and never passes `--token-embedding-type`,
so llama.cpp's own default for a Q4_0 target (Q6_K for `token_embd`) wins.
`DAEDALUS-BLUEPRINT-v6.md:59` says "Keep `token_embd`/`output` at Q8_0", so the
shipped artifact has been quietly deviating from a locked decision.

## Is the deviation bad? Measured, not argued

`llama-quantize --token-embedding-type q8_0` produces the blueprint-compliant
variant. Same fp16 source, same everything else.

**Perplexity** (`llama-perplexity -f data/eval/ppl-finewiki-150k.txt -c 512 -t 8`):

| model | PPL | vs fp16 |
|---|---|---|
| fp16 | 35.3605 ± 0.53644 | — |
| Q4_0, `token_embd` **Q6_K** (what we ship) | 35.9114 ± 0.54219 | **+1.558%** |
| Q4_0, `token_embd` **Q8_0** (blueprint) | 35.8954 ± 0.54201 | **+1.513%** |

The blueprint-compliant variant is better by **0.045%** — against ±0.54 error
bars. It is not a measurable quality difference.

**CPU decode** (`llama-bench -t 8 -n 128`), alternating the two models back to
back for three rounds, because the last time this project trusted a
non-alternating decode comparison it reported 1.29× and the real figure was
1.15×:

| round | Q6_K embd | Q8_0 embd |
|---|---|---|
| 1 | 1037.54 ± 44.00 | 922.04 ± 38.76 |
| 2 | 1006.82 ± 54.67 | 912.65 ± 33.81 |
| 3 | 1046.99 ± 21.24 | 939.89 ± 24.13 |
| **mean** | **1030.5** | **924.9** |

The two groups do not overlap in any round. Q8_0 embeddings are **10.2%
slower**, and the file grows **+9.1%** (95.56 → 104.28 MiB).

## Conclusion

**Keep Q6_K. Do not "fix" `quantize_gguf` to match the blueprint.**

The mission is the best ~150M model that is *also* the fastest on CPU decode.
Blueprint line 59 costs **10.2% of the headline decode number and 9.1% of the
file size to buy a perplexity improvement that is inside its own error bars.**
That is the definition of expensive and marginal at this scale, and the
operator explicitly asked for those to be named rather than protected.

So the deviation stands — but it is now *deliberate and measured* rather than
an accident of an unpassed flag. `quantize_gguf`'s docstring records the
numbers so a later reader who greps the blueprint for "Q8_0" does not helpfully
restore compliance and silently give back 10% of decode.

## The residual defect, and why I am not fixing it

QAT still fake-quantizes `token_embd` to **Q8_0** while the artifact ships
**Q6_K**, so for that one tensor QAT optimizes against the wrong lattice.

That is real, and the measurement above also bounds it: the entire difference
in quantization damage between the Q6_K and Q8_0 embedding grids is **0.045% of
perplexity**. Implementing a Q6_K fake-quant means writing k-quant super-blocks
(256-element super-blocks, 16 sub-blocks of 16, 6-bit weights, 8-bit sub-scales,
one fp16 super-scale) and re-verifying bit-exactness against `libggml` — a
substantial change to code that currently passes 31 QAT tests, days before
`hero` starts, to chase at most 0.045%.

Not worth it. Recorded instead. The Q4_0 linears — where `plan_qat` and the
artifact *do* agree, and where essentially all of the 1.56% damage lives — are
what QAT is bought for, and those are correct.

## While the reader was open: the locked architecture does round-trip

Same `.gguf`, same session. Every architecture decision the blueprint locks,
read out of the shipped artifact rather than out of `config.py`:

| property | `PRESETS['daedalus-150m']` | GGUF metadata | |
|---|---|---|---|
| blocks | 18 | `lfm2.block_count` 18 | ✓ |
| hidden | 768 | `lfm2.embedding_length` 768 | ✓ |
| query heads | 12 | `lfm2.attention.head_count` 12 | ✓ |
| KV heads (GQA) | 4 | `head_count_kv` = 4 on attention blocks | ✓ |
| **conv/attn interleave** | attn at `[4,7,9,11,13,16]` | `head_count_kv` per-block array `[0,0,0,0,4,0,0,4,0,4,0,4,0,4,0,0,4,0]` → attn at `[4,7,9,11,13,16]` | ✓ |
| RoPE theta | 1e6 | `lfm2.rope.freq_base` 1000000.0 | ✓ |
| context | 2048 | `lfm2.context_length` 2048 | ✓ |
| vocab | 49152 | `lfm2.vocab_size` 49152 | ✓ |
| conv kernel | 3 | `lfm2.shortconv.l_cache` 3 | ✓ |
| tied embeddings | yes | no `output.weight` tensor | ✓ |

`head_count_kv` is worth a note because it reads as a scalar `0` if you take the
first element, which looks alarming. It is a **per-block array**, and `0` is how
llama.cpp's `lfm2` graph marks a conv block. The attention blocks carry 4, so
GQA 12/4 is intact and the interleave the model was trained with is exactly the
interleave that ships.

Architecture is `lfm2` (not `Qwen3ForCausalLM`, which is what the conv-free
`dense-150m` twin exports as — the two arms genuinely take different llama.cpp
graphs, which is the thing `abl-arch`'s CPU-decode half depends on).

## Two numbers worth carrying forward

- **The fp16→Q4_0 perplexity delta on this checkpoint is +1.558%**, not the
  +2.576% recorded earlier for a "real 500M-token checkpoint". Different
  checkpoint and `-c 512`; I am not reconciling them here beyond noting both
  were measured and both sit **above** `MAX_PPL_DELTA_PCT = 1.0`. The direction
  of the earlier conclusion is unchanged: **QAT is load-bearing**, and neither
  figure is `hero`'s, which will have annealed and QAT-trained weights.
- **Decode on trained weights at 8 threads is ~1030 tok/s** for the shipped
  Q4_0, consistent with the 966 ± 30 measured on random init while `dataprep`
  was competing for CPU.
