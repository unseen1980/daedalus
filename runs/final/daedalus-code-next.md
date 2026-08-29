# Daedalus-Code: what exists, and exactly what to do next

Numbers in `runs/final/improvement-report.json`, section `daedalus-code`. Gate
detail in `runs/code-probes/branch-1b-verdict.json` and `branch-1b-stop.json`.

## What exists

**`runs/code-branch-1b/checkpoint.pt`** — sha256 `52c451a1768f17a4…`, exported
to `runs/code-branch-1b/export/model-f16.gguf` and `model-q4_0.gguf`, both
loading and generating under stock llama.cpp.

- Continued pretraining from `hero-base-f16` (sha256 `cfbf27dccf93a07c…`), never
  from the SFT/DPO checkpoint.
- 1,000,341,504 tokens, `daedalus-150m`, Muon lr 1e-3 / Adam lr 1.5e-5, warmup
  95 steps, decay fraction 0.8, seq 2048, seed 20260824.
- Mixture 65% code / 15% technical-math prose / 20% general replay; within code,
  Python 55% and JavaScript-TypeScript 45%. Split by repository, permissive
  licences only, decontaminated against HumanEval+ and MBPP+.
- **It is a base model.** No SFT ran, so it has no chat template behaviour and
  should not be given one by implication.

**It is labelled V1 and cannot be otherwise.** It inherits the released model's
49,152 vocabulary — Phase 4 selected 32,768, and a vocabulary cannot be
transplanted into trained weights — and its dead ShortConv channels, which Phase
5's schedules could only have prevented from initialization.

## Why it stopped

The `branch_1b` gate is five checks. Three passed, two failed:

| check | limit | observed | |
| --- | ---: | ---: | :--- |
| code BPB improvement | ≥ 2% | **+31.46%** | pass |
| execution regression | none | MBPP+ moved, nothing regressed | pass |
| five-task mean drop | ≤ 1.0 pt | 0.73 pt | pass |
| general-replay BPB regression | ≤ 1.5% | **+2.26%** | **fail** |
| retrieval drop, any depth | ≤ 2.0 pt | **8.0 pt** at passkey d2048 | **fail** |

The gate returned *stop*, so the 2B extension, the code/general SFT stage, the
execution-grounded preference stage and the QAT pass over the final checkpoint
did not run. That was the preregistered consequence and it was applied as
written.

## The three things to understand before continuing

### 1. The general-BPB cost is still accruing, and that is the argument

The selected probe measured **1.48%** regression at 250M tokens — inside the
1.5% bound, which is how it was selected. Four times the tokens took it to
**2.26%**. The cost did not plateau; it grew roughly with the log of tokens.

This is the strongest single reason not to have run the 2B extension, and it is
the first thing a continuation must address. The obvious lever is untried: the
replay floor is 20% and was never varied.

### 2. The aggregate code-BPB number is carried by one bucket

| source | weight | base BPB | branch BPB | improvement |
| --- | ---: | ---: | ---: | ---: |
| Python | 0.550 | 0.5212 | 0.4888 | **+6.22%** |
| JavaScript | 0.193 | 0.7624 | 0.5062 | +33.60% |
| TypeScript | 0.257 | 0.5963 | **0.1391** | **+76.67%** |

TypeScript is 25.7% of the weight and contributes **~20 of the 31.46 aggregate
points** — about two thirds of the headline from a quarter of the mixture. A
held-out BPB of 0.139 is extraordinarily low in absolute terms.

**Do not quote 31.46% as a Python-first result.** The Python number is 6.22%,
and Python is what both gate benchmarks measure.

**This is the top open question and it should be answered before anything else
is trained.**

What the build records already rule out: TypeScript came from its own
`TypeScript-all` directory of `codeparrot/github-code`, not from the interleaved
fallback stream (`runs/codeprep/source-plan.json`); the split is a salted hash of
the repository name, stable across processes; and `no_repository` is **0** in
every probe (`runs/codeprep/source-probe.json`), so no row was admitted without a
repository identifier. File-level leakage is excluded, and it was excluded by a
mechanism that was measured rather than assumed.

What they do not rule out, and what to check:

- **Holdout narrowness.** In the 2,000-row probe, `TypeScript-all` put 25
  repositories on the holdout side against 1,380 on the train side. The
  TypeScript holdout is 1,160,534 tokens drawn from a proportionally small
  repository pool — half the size of the Python holdout from a narrower base. A
  handful of large files can dominate it.
- **Content character.** Generated and vendored TypeScript — `.d.ts` bundles,
  transpiler output, minified or committed build artifacts — is low-entropy
  machine-written text, and a small holdout dominated by it would produce a
  0.139 honestly and mean almost nothing about the model's TypeScript ability.
  Inspect the holdout files directly; this is a read, not an experiment.
- **Independent re-derivation.** Re-measure TypeScript BPB on a holdout drawn
  from a different repository pool entirely. If 0.139 survives that, it is real.

Until this is settled, the defensible claim is "code BPB improved, +6.2% on
Python", not the aggregate.

### 3. The retrieval failure is probably not yours to fix

The 8.0-point drop at passkey d2048 is real — paired McNemar p=0.013, 8 of 8
discordant items moving against the branch. But Phase 3's QAT recovery, which
shares nothing with this branch except its starting weights, failed the *same*
cell by 7.0 points.

See `runs/final/v2-recommendation.md` §6, which is on the parent branch
`vast/daedalus-improvements-20260824` — the Phase 9 outputs are split across the
stack because the evidence is. If deep retrieval is a property of the
released checkpoint rather than of code training, then no code mixture will fix
it, and a continuation that spends its budget trying will spend it for nothing.
**Run the isolating arm before treating this as a data problem.**

## The continuation plan, in order

Each step names its own stop condition. None of them should be started before
step 0.

**0. Settle the TypeScript question.** Cheap, CPU-only, no training. Until it
resolves, the artifact's headline is unsafe to publish. If it turns out to be
generated content, re-derive the code-BPB gate on a cleaned holdout and re-read
every downstream number against it.

**1. Run the isolating retrieval arm.** Continue two different checkpoints on
identical data and measure passkey d2048 on both. One short run. It decides
whether steps 2–4 are worth attempting at all, and it is also the single highest
-value experiment for V2. Do this before spending another token on the mixture.

**2. Raise the replay floor and re-probe at 250M.** The general-BPB failure is
the one clearly attributable to this program's own choices. Sweep replay at 20%
(control), 30% and 40%, 250M tokens each, same seed and data order. Select on
general-BPB regression at ≤1.5% *projected to 1B* — fit the growth rather than
reading the 250M point, since that is exactly what misled the first selection.
Stop if no arm projects inside the bound: it means 150M cannot hold both, and
the answer is a larger base, not a different mixture.

**3. Re-run the 1B branch with the selected replay.** Same gate, unchanged. Only
continue past it if all five checks pass — including retrieval, unless step 1
established the d2048 loss is a property of the base, in which case record the
exemption explicitly and with its evidence rather than relaxing the threshold.

**4. Then, and only then, the post-training this program never reached.**
SFT on syntax-checked and execution-tested conversations; the preference stage
only if held-out preference accuracy *and* execution pass@1 both improve, else
publish the SFT model and record DPO as rejected.

**5. QAT last.** The branch exports at a **6.23%** Q4_0 penalty against the
released base's 5.54% on the identical text — it inherits the base's damage and
adds a little, having been trained in full precision. Phase 3 selected no recipe
to inherit, but it did establish that 100M tokens of exact-grid QAT closes the
gap completely; what it could not do was close it without moving FP16. Applied to
a *final* code checkpoint that is about to ship as Q4_0 anyway, that trade reads
differently than it did for the released model, and should be re-priced rather
than assumed.

## Change the success metric before scaling

HumanEval+ pass@1 is **0.000** for both the base and the branch. That is the
model, not the harness — the canonical-solution oracle returns 1.000 through the
identical sandbox. MBPP+ pass@1 moved from 3 items of 378 to 8.

A 150M model does not do pass@1. The signal that actually moved was **MBPP+
syntax validity, 0.238 → 0.386**, which is why it was preregistered as the more
sensitive criterion. A continuation at this scale should keep selecting on
syntax validity and per-language BPB and stop treating pass@1 as a target it can
reach. If pass@1 is the goal, the honest route is a larger base, not more code
tokens into 150M.

## Publication

Nothing has been published. The artifacts are local, hashed, and manifested in
`runs/final/improvement-report.json`. Any publication must be to a **private**
experiment repository, must not resolve to a released-model path, and needs a
code-specific model card — the one `export.py` generated is the general-models
template and would misdescribe this artifact in three places at once (it is a
base model, it failed its gate, and it carries an unexplained TypeScript
number).

Given step 0 is unresolved, the right order is: settle TypeScript, write the
card, then publish privately.
