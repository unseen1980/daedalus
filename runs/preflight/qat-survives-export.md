# Does QAT's work actually reach the shipped Q4_0 file?

2026-08-11 ~14:40 UTC. CPU only, isolated outputs under `/tmp`; `hero` untouched.
Cost: ~10 min of CPU on a box already rented for the GPU.

## The question nobody had asked

`tests/test_qat.py` proves `daedalus/qat.py`'s Q4_0 grid **is** llama.cpp's
grid, bit for bit, against the real `libggml`. That is the claim the QAT design
rests on and it is correct.

It is also not sufficient. Between the weight QAT trains and the weight
`llama-quantize` sees sits `export.py`, which writes the HF tensors in a
reduced dtype and then calls `convert_hf_to_gguf.py --outtype f16`. **Nothing
tested that composition.** The suite was green over a two-step path where only
the second step had ever been checked.

`hero` spends its final 5% — ~3B tokens, ~5 GPU-hours — on QAT, and the Q4_0
GGUF is the shipping artifact and the entire basis of the CPU-decode claim. So
the question is worth answering before, not after.

## What was wrong

`export.py:133` wrote the HF model as **bfloat16**.

Q4_0 values are `(q - 8) * fp16(d)`. bf16 keeps **8 mantissa bits**; fp16 keeps
**11**. Rounding an on-grid tensor to bf16 perturbs the block extreme, so
`llama-quantize` re-derives the scale as `bf16(d)` rather than `fp16(d)`, and
every weight in the block shifts with it. QAT converges to one lattice;
shipping lands on a neighbouring one.

## Measured, through the real path

Not simulated. A real 150M checkpoint (`abl-arch` arm 1, 5B tokens) was
projected onto the QAT grid exactly as a converged QAT run would leave it —
`q4_0_qdq` on every linear, `q8_0_qdq` on the embedding, following
`qat.plan_qat` — then run through the **real** `export_hf_model` →
`convert_hf_to_gguf.py` → `llama-quantize Q4_0`, and the Q4_0 blocks were read
back out of the GGUF with `gguf.GGUFReader` and compared against the reference.

| HF export dtype | Q4_0 tensors | weights compared | off the QAT grid by ≥½ level | ‖shipped − grid‖ / ‖grid‖ | file |
|---|---|---|---|---|---|
| **bfloat16** (what shipped) | 102 | 122,683,392 | 0 | **0.1712 %** | 102.0 MB |
| **float16** (now the default) | 102 | 122,683,392 | 0 | **0.0000 %** | 102.0 MB |

For scale: RTN damage on this checkpoint — the thing QAT is bought to remove —
is `‖q4_0(w) − w‖/‖w‖` = **6.2708 %**. So bf16 export left **2.7% of that
damage in place**; fp16 leaves none. QAT still did ~97% of its job under bf16,
so this is an erosion, not a catastrophe — but it is a free 3%, and the file is
the same size either way.

It is the same *class* of defect as the `torch.compile` lattice bug
(`qat-compile-lattice.md`), which this project already treated as real and
fixed: there the training grid drifted off the shipping grid by ~0.4% of one
Q4_0 level; here the shipping grid drifts off the trained grid, and by more.

## Why fp16 rather than fp32

fp32 is also bit-exact and doubles the published artifact for nothing. fp16 is
the same size as bf16 today.

fp16's narrower exponent range is not a risk for these weights, checked on the
real checkpoint rather than assumed:

| | |
|---|---|
| max \|w\| | **2.19** vs fp16's 65504 — ~30,000× headroom |
| smallest non-zero \|w\| | 7.17e-16, below fp16's 5.96e-08 subnormal floor |
| weights that flush to zero in fp16 | 12.9M of 160.5M |

That last row looks alarming and is not. Those weights are ≥5 orders of
magnitude below **one Q4_0 level** (level ≈ 0.028 in a typical block), so they
quantize to zero in the shipped file whatever dtype carries them there — which
is exactly why the measured drift is **0.0000%** and not merely small. They are
also not random: they are the dead short-conv channels documented in
`conv-channel-death.md`.

## The fix

`export_hf_model(dtype=torch.float16)`, and `to_qwen3_config`'s `torch_dtype`
declaration moved to match (it said `bfloat16` while the weights are now fp16;
a wrong hint in `config.json` makes `from_pretrained` cast against the wrong
dtype).

Three tests, in `tests/test_qat.py`:

- `test_fp16_export_keeps_qat_weights_exactly_on_the_grid` — through the real
  `libggml` quantizer, on-grid weights round-trip **bit-identically**.
- `test_bf16_export_would_move_the_weights_off_the_qat_grid` — pins *why*, so
  the default cannot be quietly restored to the training dtype.
- `test_export_defaults_to_fp16_because_bf16_erodes_qat` — **verified failing**
  against a pre-fix copy of `export.py`.

115 pass across `test_export.py`, `test_qat.py` and `test_publisher.py`.

## Scope, stated so it is not over-read

- **`hero` is unaffected while it trains.** This is downstream of the run; the
  export happens when it finishes, on the fixed code.
- `abl-arch`'s published Q4_0 deltas (2.53% / 2.38%) were measured through the
  bf16 path. Neither arm ran QAT, so their weights were never on the grid and
  the bf16 rounding is buried in a much larger RTN error; the numbers stand and
  are not worth re-measuring. `hero`'s delta is the one this changes.
- The `token_embd` tensor ships as Q6_K by llama.cpp's default (a deliberate,
  measured deviation — see `token-embd-quant-grid.md`), so the table above
  covers the 102 Q4_0 tensors, which is where the Q4_0 damage lives.
