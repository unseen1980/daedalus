# The depth sweep, run for real before it runs unattended

**2026-08-10 13:52Z.** `export.measure_decode_speed` grew a context-depth sweep
this morning, and every test of it stubs `subprocess.run`. It had never been
executed against a real `llama-bench`. It runs unattended three times — at the
end of `abl-arch` arm 1 (~16:31Z tonight), arm 2 (~06:04Z tomorrow), and `hero`
— and it produces the project's **headline Pareto number**.

The depth measurements already on record (`runs/eval/decode-hybrid-vs-dense.md`)
came from `scripts/decode_bench.py`, a **different code path** with a different
command line. So the numbers were real and the function that will report them
was not exercised.

This is the shape that has cost this project time twice already: `check_milestone.py`
was green in 17 tests and died on its documented invocation, because the test
environment supplied what the real one did not.

## What was run

The live arm-1 rolling checkpoint — real trained weights, **step 8,140,
3.82B tokens**, not a random init — through `abl_arch.export_and_bench`, the
exact function tonight calls, into `/tmp` so nothing touches run state:

```
checkpoint.pt -> HF dir -> model-f16.gguf -> model-q4_0.gguf -> decode sweep
```

Thread-capped at 4 so it could not steal the live trainer's dataloader cores;
the perplexity step (already proven on real GGUFs) was skipped.

## Result: it works, and it is fast

**Whole pipeline: 14.8 s.** 306.22 MiB f16 → **95.56 MiB Q4_0** (4.99 BPW),
164 tensors, quantize time 485 ms.

| depth | tok/s (4 threads) | stddev |
|---|---|---|
| 0 | 720.24 | 17.13 |
| 512 | 625.51 | 6.67 |
| 2048 | 483.03 | 4.12 |

All three depths returned, correctly keyed, top-level keys still depth 0. The
absolute numbers are **not comparable** to anything else on record — 4 threads
beside a live trainer, versus the matched-thread alternating rounds that
produced the 1.83× figure. What this run establishes is that the code path
works and the JSON parses; ratios come from `scripts/rebench_arms.py`.

The depth trend is the expected one (720 → 626 → 483, i.e. 0.67× from 0 to
2048), which is a weak confirmation that `-d` is doing what it claims.

## The one hole it left, now closed

`_bench_one_depth` called `subprocess.run` with **no timeout**. The function's
contract is "deep measurements are best-effort, record the error, never raise" —
and that is written against *exceptions*. **A hang is not an exception.** With no
timeout the export step never returns, `abl_arch.py` waits on it with no timeout
of its own, and the chain stalls silently with the GPU idle at $10.78/day.

That is precisely how the Hub uploader failed twice this morning: wedged in
CLOSE-WAIT, `except Exception` everywhere, nothing could see it.

`BENCH_TIMEOUT_S = 900.0` now bounds every bench invocation — 60× the measured
14.8 s sweep, so it cannot fail a healthy run on a loaded box. A timeout at a
deep depth is recorded like any other failure; at depth 0 it raises, which is
the documented behaviour and is strictly better than hanging.

Two tests. `test_every_bench_invocation_is_time_bounded` is a genuine regression
test — **mutation-checked**: reverted against the pre-fix line it fails with
*"depth 0 bench has no timeout"*. `test_a_wedged_deep_bench_is_recorded_like_any_other_failure`
pins the handling contract and would pass either way; it is documentation of
behaviour, not proof of the fix, and is labelled that way rather than counted
as two.

## The dense arm ships through a different converter, so it was checked too

Arm 2 exports at ~06:04Z as **`qwen3`**, not `lfm2` — llama.cpp's `lfm2` graph
aborts on a conv-free model — so it is a different converter branch, and the
chat-template change touches the tokenizer export both arms share. Verified on a
random-init `dense-150m` through the same `export_and_bench`: **266 tensors**,
`model_type: qwen3`, quantize clean, decode sweep returns both depths.

**Two things fell out that belong in the ablation writeup rather than here.**

*The dense twin quantizes worse.* `llama-quantize` reported
*"1 of 266 tensor(s) required fallback quantization"* on dense and **no such
warning on the hybrid**:

| | f16 | Q4_0 | BPW |
|---|---|---|---|
| `daedalus-150m` (hybrid) | 306.22 MiB | **95.56 MiB** | **4.99** |
| `dense-150m` (twin) | 307.63 MiB | 101.62 MiB | 5.29 |

Param-matched at f16 to within 0.5%, the hybrid's Q4_0 is **6.3% smaller**. On a
project whose stated bar is CPU inference, file size is part of the claim, and
this one is free — it comes from the shapes, not the training.

*The decode ratio is where it should be.* Unpaired and at 4 threads beside a
live trainer, so **indicative only** — `rebench_arms.py` is what produces the
real figure — but hybrid/dense reads **≈1.14× at depth 0 and ≈2.00× at depth
2048**, against 1.15× and 1.83× measured properly at 8 threads. The trend
survives a change of thread count and a change of day, which is the part worth
knowing before tomorrow morning.

## The other two scripts that produce headline numbers unattended

Same question, same answer needed before ~06:00Z tomorrow, when
`rebench_arms.py` and `abl_table.py` run with nobody watching and between them
emit **the** hybrid-vs-dense claim. Their tests cover the refusals (a missing
arm, a recorded-but-deleted GGUF, a corrupt sidecar); neither had been executed
against real GGUFs.

Both were, on the two real files above, in an isolated `/tmp` dir with a
fabricated `results.json` (the f16 GGUF standing in for the dense arm, so the
two models genuinely differ in cost):

- **`rebench_arms.py`** — 3 depths × 2 alternating rounds × 2 models, sidecar
  written to `decode-paired.json`, summary printed. Round-to-round spread was
  ≤2.4%, so the alternating design is doing its job.
- **`abl_table.py`** — picked the sidecar up **by convention, with no flag**,
  which is the property that matters at 06:00Z. It rendered the depth table,
  took **depth 2048 as the headline** with depth 0 shown beside it, and on the
  deliberately incomplete input **degraded rather than raised**: *"Quality: not
  comparable — an arm is missing val_bpb"*, `hero` falling back to the
  blueprint default, **rc 0**.

That last part is the one worth having seen: the pre-registered decision rule
says a missing score goes to the hybrid, and it now has an execution behind it
rather than only a test.

(The 3.66×/2.79× ratios printed are Q4_0 against f16 — an artifact of the
stand-in, not a hybrid-vs-dense result. What was under test is the plumbing.)

## Separate finding: `setup_llama_cpp` does not build `llama-cli`

`export.py:494` builds `llama-quantize`, `llama-perplexity`, `llama-bench` — the
three the pipeline needs. **Not `llama-cli`**, so a fresh box following this
repo's own setup gets no tool to actually *run* the model, and no Daedalus GGUF
has ever generated a token of text. Perplexity (29.13 fp16, measured earlier)
already proves the logits are right, so this is a packaging gap rather than a
correctness one — but it is on the path between the operator and the deliverable.

Building it turned up a second wrinkle: `llama-cli` now depends on `llama-ui`,
which **downloads** a prebuilt UI tarball at build time and fails without
network access to that bucket. `-DLLAMA_BUILD_UI=OFF` is required. Worth
knowing before the final artifact is handed over.
