# Hybrid vs dense twin: CPU decode against context depth

**Measured 2026-08-10 ~09:55Z.** Raw data: `runs/eval/decode-hybrid-vs-dense.json`.

The project's headline Pareto claim is that Daedalus's conv/attention hybrid
decodes faster on CPU than a param-matched all-attention model. Until now that
claim rested on a single number — **1.15×** — taken at **context depth 0**, which
is the one regime where the architecture has almost nothing to gain.

At the context these models are trained for it is **~1.8–2.0×** (three
measurements today: 1.83× / 1.93× paired, 2.03× unpaired — see "Repeated" below;
the spread is why this is stated as a range).

| depth | `daedalus-150m` (hybrid) | `dense-150m` (all-attention twin) | ratio |
|---|---|---|---|
| 0 (empty context) | 951.9 ± 37.8 tok/s | 825.3 ± 5.0 tok/s | 1.15× |
| 512 | 869.5 ± 22.9 tok/s | 593.6 ± 29.6 tok/s | 1.46× |
| **2048** (the trained context) | **648.5 ± 7.0 tok/s** | **354.1 ± 33.8 tok/s** | **1.83×** |

## Why depth 0 understates it, and by how much

The hybrid's advantage *is* the KV cache it does not keep. At depth 0 there is no
KV cache to re-read, so the measurement is taken exactly where the mechanism is
switched off. Per decoded token, with an fp16 KV cache:

| | KV blocks | KV heads | bytes/token | at depth 2048 |
|---|---|---|---|---|
| hybrid | 6 of 18 | 4 | 6,144 | ~12.6 MB re-read |
| dense twin | 24 of 24 | 2 | 12,288 | ~25.2 MB re-read |

**Exactly 2× the KV traffic**, and on a memory-bandwidth-bound CPU that is the
cost that dominates as context grows. The measured ratio climbing 1.15 → 1.46 →
1.83 with depth is that mechanism showing up; a flat ratio would have meant the
claim was wrong.

Note the dense twin is not a naive baseline — it uses *more* aggressive GQA than
the hybrid (2 KV heads against 4), so per attention layer it is the cheaper of
the two. It loses on having 24 attention layers where the hybrid has 6.

## Method

```
python scripts/decode_bench.py \
  --models hybrid=…/hybrid-q4_0.gguf dense=…/dense-qwen3-q4_0.gguf \
  --threads 8 --rounds 3 --n-gen 128 --depths 0 512 2048
```

Q4_0, `-p 0` (decode only, no prefill row), **matched thread counts**, and rounds
**alternating** hybrid/dense/hybrid/dense — both precautions exist because this
project has already been burned by their absence: a non-alternating comparison
reported 1.29× where alternating rounds put the same measurement at 1.15×.

## What this does and does not establish

- **Random-init weights.** Valid for speed only: decode speed is a property of
  architecture and size, not weight values — separately confirmed at 968.2 ± 15.4
  tok/s on trained weights against 966.0 ± 29.6 on random init. It says nothing
  about quality.
- **The two arms use different llama.cpp graphs.** The hybrid exports as `lfm2`;
  the dense twin exports as `qwen3`, because llama.cpp's `lfm2` graph is hybrid
  by construction and aborts in `llama_decode` on a conv-free model. So some of
  the gap could be per-architecture implementation quality rather than
  architecture. That confound runs **against** this result rather than for it:
  `qwen3` is one of the most heavily exercised graphs in llama.cpp and `lfm2` one
  of the least, so if anything the dense twin has the better-optimised kernels.
- **Absolutes are depressed and noisy** — `abl-arch` arm 1 was training on the
  GPU throughout, and its dataloader competes for the same cores. Only the ratio
  within one invocation is durable, which is why every row above was taken in one
  alternating pass.
- **n = 3 rounds**, so the stddevs are thin. The dense twin's ±33.8 at depth 2048
  is 9.5% of its own mean; the ratio is nowhere near that margin, but a tighter
  number would want more rounds.
- **It does not extrapolate past 2048.** The mechanism says the ratio keeps
  climbing with context and nothing here measures that.

## Repeated, and the honest number is a range: **~1.8–2.0× at depth 2048**

The table above is one invocation. Two more were taken within half an hour, on
the same GGUFs, with arm 1 training throughout:

| run | method | depth 0 | depth 512 | **depth 2048** |
|---|---|---|---|---|
| A — 09:55Z | paired, alternating | 1.15× | 1.46× | **1.83×** |
| B — 10:22Z | **unpaired**, each model measured in its own invocation | 1.20× | 1.32× | **2.03×** |
| C — 10:24Z | paired, alternating | 1.27× | 1.50× | **1.93×** |

Two things follow, and the second is the more useful.

**Pairing matters, and now it is measured rather than argued.** Run B is the
method `abl_arch.py` uses — each arm benchmarked in its own invocation — and it
read **2.03×** where the paired runs either side of it read 1.83× and 1.93×.
That is an ~8% overstatement from invocations *minutes* apart on one box; arm 1
and arm 2 will be benchmarked **~12 hours** apart, which is strictly worse. This
is the same failure that once turned a real 1.15× into a published 1.29×.

**But pairing does not make it precise.** Runs A and C are the same method on the
same files 29 minutes apart and differ by **5.5%** at depth 2048 (1.83 vs 1.93)
and by 10% at depth 0. So a single decimal place here is false precision. The
honest statement of this result is **~1.8–2.0× at the trained context**, and the
number worth quoting will be `abl-arch`'s own paired measurement on **trained
weights and an idle box** — this one was taken while a 24 GB training job had the
GPU and its dataloader had the cores.

What is stable across all three is the shape: the ratio is smallest at depth 0
and grows monotonically with context, in every run, by a wide margin. That is the
claim the architecture makes, and it does not depend on which invocation you take.

## Cross-check

Our hybrid reads **648.5 ± 7.0** tok/s at depth 2048 here, against **648.65 ±
12.57** measured on 2026-08-09 in `runs/eval/decode-vs-smollm2.json` — a
separate invocation on a different day under different box load, reproducing to
0.02%. The depth-2048 measurement is stable even though the absolute depth-0
numbers move by ~10% with load.

## What changed as a result

1. `export.measure_decode_speed` now sweeps `DECODE_DEPTHS = (0, 512, 2048)`
   instead of measuring depth 0 alone, so `abl-arch`'s own export produces this
   table on **trained** weights for both arms. Depth 0 keeps the top-level keys,
   so every number already on record still means what it meant. The deep
   measurements are best-effort — a failure is recorded, never raised, because
   this runs inside an export step that a 12-hour training arm depends on.
2. `scripts/abl_table.py` quotes the trained context as the headline ratio and
   prints depth 0 beside it, rather than reporting depth 0 as the result.
3. `README.md`'s hybrid-vs-dense table carries all three depths.

Had this not been caught, `abl-arch` — the experiment the blueprint calls the
reason the project is interesting — would have reported its headline result at
roughly 60% of its true size, in the gate issue and the writeup both.
