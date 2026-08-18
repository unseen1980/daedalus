# QAT under `torch.compile`: the grid moved, and nothing said so

2026-08-09 ~23:20 UTC. Cost: ~10 min of an otherwise idle GPU, under $0.08.

## Why look here at all

QAT is the only part of the plan that runs **once, near the end of the single
most expensive job**, and never before. `hero` turns it on at 95% of 83,123
steps — hour ~90 of ~95. Every test covering it ran `compile=False,
device="cpu"` (`tests/test_qat.py:401`), which is the one configuration `hero`
never uses. So the QAT phase was scheduled to make its debut with $43.70
already spent.

## What was wrong

`train.py` compiles the model at construction (`train.py:657`) and registers
the QAT parametrizations ~90 hours later, so the quantizer runs *inside* an
inductor kernel.

`daedalus/qat.py` derives the Q4_0 scale as `d`, then rounds it to fp16 because
that is what llama.cpp stores. Written plainly as `d.half().float()`, inductor
folds the round trip away as redundant and the fused kernel keeps the
full-precision scale.

Measured here (torch 2.12.0+cu130, sm_120), 512x256 weights:

| | elements differing from eager | recovered scale |
|---|---|---|
| Q4_0 | 115,957 / 131,072 (**88.5%**) | fp32 `d` = 0.0066096145 |
| Q8_0 | 130,184 / 131,072 (**99.3%**) | fp32 `d` |
| what llama.cpp stores | — | fp16 `d` = 0.0066108704 |

`torch._inductor.config.emulate_precision_casts = True` does **not** prevent
it (tested).

## Why it is worth fixing but not worth panicking about

The premise of QAT is that the training grid *is* the shipping grid — that is
the entire reason `test_q4_0_matches_real_llama_cpp_bit_for_bit` exists and
checks against the real `libggml`, bit for bit. That guarantee is measured in
eager and silently does not hold in the compiled path.

The magnitude is small: fp16 carries ~5e-4 relative on the scale, so the
lattice offset is ~0.4% of one Q4_0 level, and the quantization *level* `q` is
unchanged for essentially every element (measured: 0.001 grid steps). The STE
pushes weights toward grid centres, which are far from level boundaries, so
level flips at export would be rare.

So: not a disaster averted. A quiet erosion of the one property QAT is bought
for — idempotency at export, i.e. `llama-quantize` being a no-op on weights
that already sit on the grid.

**The failure is invisible.** Nothing raises. The loss curve is unchanged.
`qat_rel_rmse` — the metric that exists to prove QAT is working — is computed
off the master weight in eager, so it keeps falling and keeps reporting
success.

## The fix

An opaque custom op (`daedalus::round_fp16`). Inductor cannot see inside it, so
it cannot fold the cast, and unlike `torch.compiler.disable` it stays *in* the
graph rather than breaking it once per parametrized module (~50 of them). The
op needs no backward: `q4_0_qdq`/`q8_0_qdq` are consumed only inside the STE's
`.detach()` or under `no_grad`.

After the fix, compiled output is **bit-identical** to eager for both grids,
and the eager-vs-`libggml` bit-exactness tests still pass unchanged.

## Tests added (7, all CUDA-gated)

- `test_enabling_qat_mid_run_reaches_an_already_compiled_graph` — Dynamo does
  retrace after `register_parametrization`; QAT is not a no-op. This half was
  never in doubt, but had never been checked.
- `test_compiled_qat_lands_on_the_same_lattice_as_eager` — Q4_0 and Q8_0.
- `test_compiled_qat_scale_is_the_fp16_one_llama_cpp_stores` — pins the
  specific failure mode, not just the symptom.
- `test_compiled_qat_trains_the_master_weight_through_the_ste`
- `test_qat_phase_pulls_weights_toward_the_grid_under_compile`
- `test_checkpoint_written_mid_qat_under_compile_loads_into_a_plain_model`
- `test_trainer_qat_phase_on_the_real_compiled_cuda_path` — the real `Trainer`
  on CUDA with compile and bf16 autocast, crossing into the phase mid-loop,
  checking the switch does not orphan Muon/AdamW state.

643 tests pass.

---

# While here: does the QAT switch trip the watchdog?

`hero.py` runs `watchdog.py` with `--stall-min 30`, and the watchdog halts on
`loss > 2.0 x` the running mean of the last 20 points
(`watchdog.py:42`). Turning QAT on raises the loss *discontinuously* — it is
the only deliberate upward step in the run, and it lands at 95% of ~$43.70.
A false halt there would waste the decay phase.

Measured on a **real** `daedalus-150m` checkpoint (`runs/sweep-lr0.02`, 500M
tokens, step 1040), 8 x [4, 2048] windows of `finewiki-en`:

| | |
|---|---|
| loss before QAT | 3.5863 |
| loss after QAT | 3.5954 |
| ratio | **1.0025x** (halt at 2.00x) |
| pre-QAT relative RMSE | 0.0801 |

Margin is ~800x. The QAT switch cannot trip the divergence rule.

Both watchdog thresholds also checked against real run logs rather than
arithmetic:

| | observed on real runs | threshold |
|---|---|---|
| metrics.jsonl write gap | median 1.40 min, max 2.13 min | 30 min stall |
| loss / running-mean-20 | max **0.972x** | 2.00x divergence |

Cadence is consistent with `hero`'s own arithmetic (83,123 steps / ~95 h ->
4.1 s/step -> 20 steps = 1.4 min), so the ~14x stall margin carries over.

**Noted, not changed:** `abl_arch.py` starts no watchdog at all — only
`hero.py` does. Tonight's ~24 h ablation therefore runs with the chain's exit
status as its only supervision. That is defensible (the chain gates on exit
code, and a stuck arm shows up as a flat W&B curve) but it is worth stating
plainly rather than assuming coverage that is not there.

---

# What the QAT phase costs

The fix adds a kernel boundary inside the fused graph, and `hero` spends ~4.75 h
(5% of 83,123 steps) with QAT on, so the overhead is a real budget line rather
than a curiosity. Nobody had measured it.

`daedalus-150m`, micro-batch 16, seq 2048, fwd+bwd+step, torch.compile and bf16
autocast live, same optimizer both sides, 12 timed iterations after 4 warmup:

| | ms/step | tok/s | peak VRAM |
|---|---|---|---|
| QAT off | 257.4 | 127,306 | 24.01 GB |
| QAT on | 255.0 | 128,509 | 24.01 GB |

**Overhead: -0.9%, i.e. free within noise.** That is physically what one should
expect — the quantizer is a memory-bound elementwise pass over ~600 MB of
weights, ~1-2 ms on this card against a 257 ms step — but it is worth having
measured rather than assumed, and it confirms the custom op did not cost what
`torch.compiler.disable` would have.

`hero`'s cost projection is unchanged by the QAT phase.

(These numbers are AdamW-only on random batches, which is why they sit above
the 115,692 tok/s the chain preflight measured for the same arm with Muon and a
real loader. The comparison here is like-for-like on both sides, so the *ratio*
is the trustworthy part — the same rule that applies to the CPU decode figures.)

Verified QAT was genuinely engaged during the measurement, not silently off:
**103 tensors** registered on the grid, and the compiled loss tracked eager
through the switch to six decimals (15.187711 -> 15.188095, both paths).

---

# The QAT phase run inside `train.py` on the real model, and one thing I did not measure

A 600-step QAT fine-tune was started from `runs/sweep-lr0.02/checkpoint.pt`
(real, 500M tokens) via `train.py --init-from ... --qat-frac 1.0` against real
`finewiki-en` shards. It reached step 100 before I stopped it. What that
establishes:

- **`train.py`'s QAT path works on the real 150M config with real data** —
  `qat_active: 1` from the first step, `qat_rel_rmse` logged at 0.0802, loss
  finite and falling (2.9883 -> 2.9429 over steps 80->100), **97-98K tok/s** at
  micro-batch 8 with QAT on, peak VRAM 16.70 GB. Until now QAT had only run
  inside `Trainer` on the `tiny` config.
- **`--init-from` behaves on the real model**, not just in the tmp_path tests:
  fresh WSD schedule (still in warmup at step 100, lr 6.6e-4) and `tokens_seen`
  from zero. That is the `post.py` fix validated at production scale.

**What I deliberately did not do: claim a number for QAT's benefit.** Each
optimizer step here is ~494K tokens (the batch-token ramp with accumulation),
so `hero`'s QAT phase is 4,156 steps x ~494K = **~2B tokens**, against this
probe's 33M — about 60x. A short probe that showed no reduction in Q4_0 damage
would say nothing about `hero`, and reporting it as evidence either way would be
worse than not running it. The measurement that counts is `hero`'s own
fp16-vs-Q4_0 delta with QAT, against the **2.576%** baseline recorded in
`gguf-tokenizer-and-q4-damage.md`.

**Checked while here, and fine:** with `--no-hub-uploader` the outbox keeps its
321 MB payload rather than superseding it, because supersession happens *in*
the uploader (`upload_once` drops older pending payloads for the same repo
path). If `hero`'s uploader died and never restarted, payloads would accumulate
at ~321 MB / 2 h, ~15 GB over the full run — survivable against 203 GB free,
and visible from a phone because `hub_pending` goes to W&B. The design is
sound; no change made.
