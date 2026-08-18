# `hero` precondition #2, predicted before it happens

Precondition #2 says a milestone checkpoint **with full optimizer state** must be
written and uploaded at the WSD decay-start step, on its own revision the rolling
copy cannot overwrite. Today it is proven by tests and by a hub smoke. It has
never fired inside a real multi-hour training job.

`abl-arch` arm 1 fires it today. Writing the prediction down first, so the check
at the time is a comparison rather than a rationalisation.

## The prediction

Computed from `train.py`'s own functions against arm 1's exact launch args
(5B tokens, `micro_batch=16`, `ramp_frac=0.1`, `decay_frac=0.45`, seq 1024→2048,
batch 128,000→512,000):

| | |
|---|---|
| `estimate_total_steps` | **10,391** |
| `decay_start_step` | **5,715** (55.0%) |
| tokens at that step | ~2.996B |
| **expected wall clock** | **~11:04Z**, from the measured 121,994 tok/s |
| revision | `abl-arch-daedalus-150m-stable-end-step5715` |
| path in repo | `milestone/abl-arch-daedalus-150m/checkpoint.pt` |
| size | ~1435 MB (fp32 weights + Muon and AdamW state) |

What to check at ~11:04Z: `runs/abl-arch-daedalus-150m/milestone.json` exists
with `step: 5715`; the `[milestone] end of stable phase` line in the run log;
`milestone_step` in W&B; and — the part that matters — the revision listed on the
**real Hub repo**, verified by listing it rather than by trusting local
bookkeeping.

Three things make it likely to fire rather than skip quietly:

- `IntervalGate.ready()` returns True on its first call (`self.last is None`,
  `train.py:572`), so the 600 s retry backoff does not delay the first attempt.
- The test is `self.step < self.milestone_step` (`train.py:1011`), i.e. `>=`, so
  a resumed run that lands *past* the step still produces the artifact.
- 190 GB free; the outbox is drained to `uploaded.json`; uploader pid 435972
  alive and having already delivered the step-1 rolling checkpoint at 05:18:32Z.

And if it fails it will not take the run with it: the exception path warns and
retries on the gate rather than raising (`train.py:1044-1049`).

## A correction to my own arithmetic, not to the code

Checking this, I first computed hero's schedule as **82,792 steps / decay at
45,535** and flagged it as disagreeing with the **83,123 / 45,717** in
`hero-schedule.md` and the gate draft. **The documented numbers are right and my
check was wrong**: I passed power-of-two batch tokens (131,072 → 524,288) where
`TrainArgs` actually defaults to **128,000 → 512,000** (`train.py:590-591`).

Worth recording because the realized step size is *identical* either way —
`grad_accum_steps` rounds 512,000 up to 16 × 16 × 2048 = **524,288 tokens**, which
is exactly what the live metrics show — so the error is invisible in steady state
and only shows up in the step *count*, which is the denominator `wsd_lr` decays
over. That is the same quantity the earlier WSD bug got wrong at a cost of four
days and ~$44. The lesson is the one already learned: read the schedule inputs
off `TrainArgs`, never retype them.

## Also checked: arm 1's export at ~17:00Z

`render_model_card` was added this morning and is wired into `export_hf_model`
(`export.py:171`), so it runs unattended when arm 1 exports. It had only ever
been exercised on the `tiny` config.

Its failure path is deliberately non-fatal (`export.py:445`) — which means a
failure would be **silent**, and every artifact would ship without provenance,
the exact gap that was closed this morning. So it was worth rendering for the
real configs rather than assuming:

- `daedalus-150m` — 3,536 chars, carries the `ccccAccAcAcAcAccAc` interleave, the
  success bar, tokens seen, and a branch command naming the revision above.
- `dense-150m` — 2,770 chars, renders clean (arm 2 exports as `Qwen3ForCausalLM`).

Confirmed the emitted branch command is runnable as written:

```
python train.py --run-name abl-arch-daedalus-150m-ext \
  --resume 'hub://Unseen1980/daedalus-checkpoints/milestone/abl-arch-daedalus-150m/checkpoint.pt?rev=abl-arch-daedalus-150m-stable-end-step5715'
```

Separately confirmed that an export failure cannot cost the training:
`abl_arch.py:461` catches it per-arm into `entry["error"]` and still writes
`results.json`, so arm 2 runs regardless.

## The unsupervised window between the two arms, bounded rather than fixed

Arm 1 stops training ~16:40Z, then evaluates and exports before arm 2 starts.
Two things about that window are worth having checked rather than assumed.

**The watchdog is deliberately off for it, and that is correct.** `run_arm` stops
it in a `finally` before the eval/export returns (`abl_arch.py:151-155`), because
a watchdog left running would read the *finished* run's own static
`metrics.jsonl` as a 30-minute stall and halt a healthy job. This is the concern
recorded on 2026-08-09 — *"abl-arch's ~38 min full-pass validation is precisely
the quiet stretch that would trip a 30 min stall rule"* — and it was handled when
the watchdog was added rather than left as a live hazard. No change needed.

**The consequence is that export runs with nothing watching it, and none of its
subprocess calls has a timeout.** `convert_to_gguf`, `quantize_gguf`,
`measure_perplexity` and `measure_decode_speed` all call
`subprocess.run(..., check=True)` with no `timeout=` (`export.py:489, 520, 538,
561`), so a hung llama.cpp binary would block forever with the GPU idle.

**Recorded, not fixed, and the reasoning is the exposure rather than the
likelihood.** The path is not novel — it has run end to end at least three times
on this box (the `abl-arch` smoke produced `decode_speed` and `delta_pct` for
*both* arms, plus the `dense-150m` Q4_0 dry run and the token-embd quant grid) —
and the damage is bounded by the ~10-minute check-in cadence plus `HEARTBEAT.md`'s
stall verdict, so a hang costs roughly **$0.10** before it is seen. Choosing a
timeout value that is generous enough not to kill a legitimately slow
`llama-perplexity` run, hours before a 25 h unattended job, is a worse trade than
catching it by polling. Worth doing before `hero`'s own export, where the same
call sites run at the end of a four-day run.

## The checker itself was broken, and would have failed the milestone that landed

Found at 10:45Z, ~17 minutes before the check was due to run. `check_milestone.py`
authenticated its Hub listing with `os.environ.get("HF_TOKEN")`
(`scripts/check_milestone.py:124` before the fix). **`HF_TOKEN` is not set on this
box** — every other component reads `HF_TOKEN_WRITE` (`ckpt_uploader.py:431`,
`shard_uploader.py:198`, `publisher.py:101`, `train.py:740`) — and
`Unseen1980/daedalus-checkpoints` is **private**, so the listing was an anonymous
request against a repo requiring auth. Measured both ways rather than reasoned
about:

```
anonymous (what check_hub did):  FAIL RepositoryNotFoundError: 401 Client Error
HF_TOKEN_WRITE:                  OK, 5 files
```

So the check would have reported `could not list ...: 401` — i.e. **the milestone
is not on the Hub** — for a milestone sitting on the Hub. Precondition #2 is the
last unproven one of the four, this script is the evidence for it, and it fires
**once, unattended, at 55% of a run**. The failure would have arrived as a
credible-looking negative on the exact artifact a $41.26 gate turns on, with the
plausible next move being to go debugging the uploader instead of the checker.

**Why the tests did not catch it.** All three existing hub tests fake `HfApi`
with `def __init__(self, token=None): pass` and never assert anything about the
token, so a `None` flowed through green. The regression test now asserts the
token reaching `HfApi` is the uploader's own variable
(`test_the_hub_check_authenticates_with_the_uploaders_own_variable`), plus
precedence and explicit-override cases. Fixed by resolving
`HF_TOKEN_WRITE` → `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN`.

**Then exercised against the real repo, since that is the half tests cannot
reach**: a record pointing at a file known to exist returns clean, and one
pointing at a missing path returns `is not in ...@rolling (found 5 files)`. Both
now come back authenticated.

**Audited for the same bug elsewhere, because a token read from the wrong
variable fails identically in worse places** — the restore path is what runs
*after* the box is lost. Every other call site already threads `HF_TOKEN_WRITE`,
including `download_checkpoint` via `resolve_resume` (`train.py:740, 756`). This
script was the only outlier: the newest file, and the only one not written
against the uploader it checks.

**A by-product worth recording.** The authenticated listing confirms
`rolling/abl-arch-daedalus-150m/weights.pt` is on the Hub — precondition #1
verified from outside this box rather than from `uploaded.json`, which is the
distinction this script exists to make. It also showed
`runs/preflight/hub-restore.md`'s claim that the smoke branches and files "were
deleted afterwards" to be false on both counts; corrected there.

### How the export timeout gets chosen, without touching the live path

The paragraph above defers `subprocess` timeouts because picking a value hours
before a 25 h unattended job risks killing a legitimately slow
`llama-perplexity`, and no step's wall clock has ever been recorded — the smoke
results carry `delta_pct` and `decode_speed` but no durations.

Instrumenting `export.py` to get them would put new code in the path that arm 1's
$11 export runs tonight, for a measurement obtainable for free. Arm 1's export
writes `hf/`, `model-f16.gguf`, `model-q4_0.gguf` and finally its entry in
`runs/abl-arch/results.json`, each at a distinct moment, so **mtime differences
give convert, quantize, and the perplexity-plus-decode tail with no code change
at all.** Two arms give two samples of each.

That is the input for a timeout on `hero`'s export — where the same call sites
run at the end of a four-day run — set from measured durations with a wide
multiplier, rather than from a guess made today.
