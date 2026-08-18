# CPU decode: Daedalus vs SmolLM2-135M — the half of the bar nothing had measured

**2026-08-10 03:19Z.** The stated success definition is *"concede SmolLM2-135M
on quality while beating it decisively on CPU decode."* Until now the second
half had **no measurement at all**: `runs/eval/peer-table.md` carries quality
for five peers and a decode number for none of them, and `abl-arch` only ever
compares our two arms to each other.

Both models quantized to Q4_0 with llama.cpp's own defaults, benchmarked with
`llama-bench -p 0` at matched thread counts, alternating rounds
(`scripts/decode_bench.py`). Raw: `runs/eval/decode-vs-smollm2.json`.

## Result

8 threads, 3 alternating rounds each, `n_gen` 128:

| decode at context depth | Daedalus-150M | SmolLM2-135M | ratio |
|---|---|---|---|
| 0 | 960.9 ± 3.4 | 908.1 ± 37.1 | **1.06×** |
| 512 | 933.7 ± 28.5 | 625.7 ± 38.5 | **1.49×** |
| **2048** (what we train for) | **648.6 ± 12.6** | **312.4 ± 7.2** | **2.08×** |

**At the context this model is built for, it decodes 2.08× faster than
SmolLM2-135M — while carrying 19% more parameters** (160.5M vs 135M) in a 11%
larger file (102.0 MB vs 91.7 MB).

## Why depth is the whole story, and why a default benchmark hides it

A default `llama-bench` run measures decode at **depth 0** — into an empty
context — and there the answer is 1.06×, i.e. nothing worth claiming. That is
the number we would have reported if we had benchmarked the obvious way.

The gap opens with context because that is the mechanism:

| | Daedalus-150M | SmolLM2-135M |
|---|---|---|
| blocks | 18 (12 gated short-conv + 6 GQA attention) | 30, all attention |
| hidden | 768 | 576 |
| KV heads × head_dim | 4 × 64 | 3 × 64 |
| **KV bytes read per decoded token, per context token** | 6 × 4 × 64 × 2 × 2 B = **6,144 B** | 30 × 3 × 64 × 2 × 2 B = **23,040 B** |

**3.75× less KV traffic per token**, because only 6 of our 18 blocks keep a KV
cache at all while all 30 of theirs do. At depth 2048 that is ~12.6 MB versus
~47.2 MB of cache re-read for every single token generated, and on a
memory-bandwidth-bound CPU decode that is the cost that matters. A gated short
conv's decode cost is flat in context length; attention's is not.

## What this does and does not establish

**Does:** the Pareto claim has a number behind it for the first time, and
"decisively" survives contact with a measurement — at the context we train for.

**Does not:**

- **This is not a clean conv-vs-attention result.** SmolLM2-135M is deeper (30
  vs 18 blocks) and narrower (576 vs 768), so the depth-0 figure of 1.06×
  mixes layer count, width, file size and embedding quantization. The *scaling*
  from 1.06× to 2.08× is the architecture effect; the baseline is not.
  `abl-arch` is what isolates it: same params, same data, same steps, hybrid vs
  dense.
- **The embedding tensors are not quantized alike, and not by our choice.**
  `llama-quantize` gives our `token_embd` **Q6_K** and SmolLM2's **Q8_0** — it
  falls back for them because Q6_K wants rows that are a multiple of 256 and
  their hidden size is 576. Our own grid (`token-embd-quant-grid.md`) measured
  Q8_0 as ~10% slower than Q6_K on our model, so some of the *depth-0* 1.06×
  is that fallback rather than us. It cannot explain a 2.08× at depth, and both
  files are what `llama-quantize` actually produces for each model with default
  flags, which is the comparison a user would get.
- **Weights are not final.** Decode speed depends on shapes and quant types,
  not on the values in them, so this used an existing exported hybrid GGUF
  rather than a trained one. The `hero` model will have identical shapes.
- **The box was busy.** `sweep` probe 2 trained on the GPU throughout;
  absolutes are depressed and are comparable only within this invocation. The
  trainer's throughput was unchanged at ~123k tok/s across the benchmark, and
  rounds alternate precisely so that shared load cancels out of the ratio.

## Why only SmolLM2 is in this table

The other four quality peers were tried and **none of them converts to GGUF in
this environment**, so there is no like-for-like decode number for them. Each
was attempted from its cached snapshot with the same
`convert_hf_to_gguf.py`; the failures are recorded rather than quietly omitted:

| peer | outcome |
|---|---|
| `EleutherAI/gpt-neo-125m` | `Model GPTNeoForCausalLM is not supported` — llama.cpp has no such architecture |
| `EleutherAI/pythia-160m` | `KeyError: 'rotary_pct'` — the converter reads a config key this revision does not carry |
| `openai-community/gpt2` | `Can not map tensor 'h.0.attn.bias'` — the checkpoint ships the causal-mask buffer this converter version does not skip |
| `facebook/opt-125m` | cached as `pytorch_model.bin` only; the converter wanted safetensors |

Two of these could be forced — patching a peer's config or dropping a mask
tensor — and neither was, because a benchmark that depends on me hand-editing
the opponent's configuration is not a benchmark anyone should trust. The
outcome is worth stating on its own: of the five published peers this project
is measured against, **only SmolLM2-135M actually runs under llama.cpp
out of the box**, which is a real deployability difference and not just an
inconvenience for this table. (GPT-2 124M could not be compared at depth 2048
in any case — its context limit is 1024.)

## Method note

Rounds alternate A, B, A, B, … rather than running one model three times and
then the other. That is not pedantry: on this box a non-alternating
hybrid-vs-dense comparison reported **1.29×** where alternating rounds put the
truth at **1.15×** — the machine's own load drifted underneath the measurement.
Every round is recorded in the JSON, not just the mean.
