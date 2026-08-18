# Option C is closed: llama.cpp will not load a pruned short-conv model

2026-08-11 ~18:30 UTC. CPU only, read-only against an existing `abl-arch` GGUF.
`hero` was not touched.

## The question, and why it was worth answering

`runs/preflight/conv-death-decision-rule.md:111` left exactly one thing open:

> "[Option C] is gated only on whether llama.cpp's `lfm2` graph tolerates conv
> width ≠ hidden size, which is an export question answerable on CPU at any
> time."

It was worth answering because it is the only route by which today's negative
finding becomes a positive deliverable. ~48% of `hero`'s short-conv channels
contribute exactly nothing (issue #7); structurally removing them was costed at
**~7.7 MB off the GGUF and ~11% less weight traffic per decoded token, at
bit-identical output** — an improvement to the CPU-decode number this project
leads with, for no training money.

Nobody had answered it. `grep` finds "prune" in three markdown files and **zero
Python files**.

## Answer: no. `INFEASIBLE`.

`src/models/lfm2.cpp:77-79` creates all three short-conv tensors at fixed
`n_embd`, and there is no conv-width hparam to override:

```cpp
layer.shortconv.conv     = create_tensor(..., {hparams.n_shortconv_l_cache, n_embd}, 0);
layer.shortconv.in_proj  = create_tensor(..., {n_embd, 3 * n_embd}, 0);
layer.shortconv.out_proj = create_tensor(..., {n_embd, n_embd}, 0);
```

`create_tensor` shape-checks against the GGUF and throws
(`llama-model-loader.cpp:897`, "wrong shape").

That is a code reading, so it is not the evidence. The evidence is the artifact:
`scripts/conv_prune_feasibility.py` takes the **real** `abl-arch` hybrid GGUF,
rewrites every short-conv tensor from 768 channels down to 640 — a *uniform*
prune, the easiest possible case for a loader — and hands it to a **real
`llama-perplexity`**:

| build | loads? | |
|---|---|---|
| unmodified `model-f16.gguf` | **yes**, rc 0 | control 1 |
| **rebuilt through this script's writer at full width 768** | **yes**, rc 0, PPL 1.0077 | control 2 |
| short-conv narrowed to 640 | **no, rc 1** | `check_tensor_dims: tensor 'blk.0.shortconv.conv.weight' has wrong shape; expected 3,768, got 3,640` |

**Control 2 is the one that makes this conclusive.** Without it, the rejected
narrow build is ambiguous — it could mean llama.cpp refuses the narrowing, or
merely that my GGUF writer emits a broken file. The full-width rebuild goes
through the identical writer path and loads and computes perplexity, so the only
difference left is the narrowing. The first version of this probe did not have
that control and its verdict would not have been trustworthy.

Uniform narrowing is also the right case to test first: a real per-layer prune
gives each conv block its own width, so a loader that rejects the uniform case
rejects the per-layer case a fortiori.

## What this rules out, and what it leaves

The 13.6M dead parameters (8.47% of the model) **cannot be reclaimed at export**:

- **Patch llama.cpp** — would ship a GGUF that only runs on a custom build. That
  forfeits "runs on stock llama.cpp", which is both the credibility of the
  CPU-decode comparison and the reason the artifact is usable at all. Not worth
  7.7 MB.
- **Shrink `n_embd`** — impossible; it is the residual stream width, shared with
  every attention and FFN block.
- **Leave the zeros in** — Q4_0 spends 4 bits on a zero exactly as on any other
  weight. No saving, which is why "they're already zero" is not a workaround.
- GGUF has no sparse tensor format.

So the dead channels cost their full weight bytes and their full decode traffic,
and there is no export-side fix.

## What it means

Three things, all for the writeup rather than for any decision:

1. **Option C is closed.** The decision rule deliberately kept it out of the A/B
   either/or so that a "no" on branching would not read as a "no" on this. The
   answer is now "no" on its own merits, and it is a llama.cpp limitation rather
   than anything about this model.
2. **`abl-arch`'s hybrid result is stronger than it looks.** The hybrid beat the
   dense twin while carrying ~8.5% of its parameters as dead weight *and* paying
   full decode traffic for them. It is not merely handicapped on quality — it is
   handicapped on the speed axis it wins.
3. **The conv-death fix is a training-side quality lever, not a speed lever.** A
   future run with `conv_wd < 0.0024` would get ~8.5% more *effective* parameters
   at identical file size and identical decode cost. That is worth stating
   plainly, and it is not something this budget can buy.

Reproduce: `python scripts/conv_prune_feasibility.py`
Machine-readable result: `runs/preflight/conv-prune-feasibility.json`
