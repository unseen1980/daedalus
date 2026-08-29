# Daedalus improvement and code program: final report

Six days on one RTX 3090 Ti. A deterministic controller owned the phases, the
deadline and the gates; bounded Claude Code sessions did the engineering.

**The headline is a negative one.** This program ships no improved V1. Every
preregistered improvement gate that could have produced a better released model
returned *stop*, and the reasons are more useful than a win would have been.
What it does hand over is a Daedalus-Code V1 base checkpoint that failed its own
continuation gate, four proxy phases that each narrowed the V2 design space, and
one cross-phase finding that changes what a V2 should start from.

Every number below is re-read at finalization from the artifact that produced
it, and the artifact's SHA-256 is re-hashed and compared against the digest the
measuring scorecard recorded. All bits-per-byte figures are full-pass, never the
evaluator's bounded sample.

## How to read a number in this report

Every claim carries a scope. The scope decides what the number is evidence *about*, and the four are not interchangeable:

- **released model** -- measured on the released 150M weights, or on an artifact this program derived from them. Actionable for the shipped model.
- **Daedalus-Code** -- measured on the Daedalus-Code branch, which starts from the released base weights. Actionable for that artifact only.
- **proxy evidence** -- measured on a smaller stand-in trained for the comparison. Evidence about a *decision*, never a property of a shipped model.
- **projection** -- an extrapolation. It carries no measurement of its own and names the ones it extrapolates from.

No proxy result in this report is a statement about the released 150M model. Nothing here measured one on the other, and the report's own validator refuses the combination.

## 1. Released V1 baseline

*Scope: released model.*

Re-measured at finalization from the released GGUFs, through stock llama.cpp, against the same 292-chunk text at the same context size as the Phase 0 baseline five days earlier.

| finding | value | read from |
| --- | --- | --- |
| FP16 perplexity | 6.6135 | `runs/final/quant/released-base/quant-comparison.json` |
| Q4_0 perplexity | 6.9798 | `runs/final/quant/released-base/quant-comparison.json` |
| Q4_0 perplexity penalty over FP16 | 5.53867 % | `runs/final/quant/released-base/quant-comparison.json` |
| five-task mean (full splits, no per-task limit) | 47.3744 | `runs/eval/baseline-hero-tasks.json`, `runs/qat-recovery/baseline.json` |
| finalization re-measurement reproduced Phase 0 exactly (identical perplexities, chunk counts and artifact digests) | yes | `runs/final/quant/released-base/quant-comparison.json`, `runs/eval/quant-base/quant-comparison.json` |

Caveats, by finding:

- **Q4_0 perplexity penalty over FP16**
  - Paired per-chunk: Q4_0 is worse on 285 of 292 chunks, so the penalty is a consistent shift and not an aggregate artefact.

## 2. QAT recovery of the released model

*Scope: released model.*

Three 100M-token probes from `--init-from` on the released base, exact-grid QAT from step one, identical data order and seed. All three eliminated the Q4_0 penalty. All three failed retention. **Negative result: no recovery checkpoint is recommended and the released weights are untouched.**

| finding | value | read from |
| --- | --- | --- |
| selected recovery checkpoint | none -- all three arms rejected | `runs/qat-recovery/verdict.json` |
| Q4_0 penalty after recovery at Muon lr 1e-3 (baseline 5.539%) | -0.372929 % | `runs/qat-recovery/scored/qat-recovery-lr0.001.json` |
| Q4_0 absolute perplexity after recovery, against the released Q4_0's 6.9798 | 6.7054 | `runs/qat-recovery/scored/qat-recovery-lr0.001.json`, `runs/qat-recovery/baseline.json` |
| FP16 perplexity regression (gate: at most 0.5%) | 1.76911 % | `runs/qat-recovery/scored/qat-recovery-lr0.001.json` |
| worst retrieval drop, at passkey d2048 (gate: at most 1 point) | 7 points | `runs/qat-recovery/scored/qat-recovery-lr0.001.json` |
| five-task mean after recovery, against the baseline's 47.37 | 48.0823 | `runs/qat-recovery/scored/qat-recovery-lr0.001.json` |
| escalation to 300M and 1B tokens | refused -- no 100M probe passed both the improvement and retention gates; reporting the negative result rather than escalating | `runs/qat-recovery/verdict.json` |

Caveats, by finding:

- **selected recovery checkpoint**
  - The released weights are unchanged. This program ships no improved V1.
- **Q4_0 penalty after recovery at Muon lr 1e-3 (baseline 5.539%)**
  - Negative means Q4_0 scored marginally *better* than its own FP16 parent. All three arms cleared the improvement gate outright; the phase failed on retention, not on the quantization target.
- **Q4_0 absolute perplexity after recovery, against the released Q4_0's 6.9798**
  - The shipping artifact did get better in absolute terms (-3.93%). It is rejected because of what it cost elsewhere, which is the trade the retention gates exist to price.
- **five-task mean after recovery, against the baseline's 47.37**
  - General task scores rose. The damage is concentrated in long-context retrieval, which the five-task suite does not measure.

## 3. Daedalus-Code V1

*Scope: Daedalus-Code.*

Continued pretraining from `hero-base-f16` on a 65/15/20 code/technical/replay mixture, Python 55% and JavaScript-TypeScript 45%, split by repository. Three 250M-token probes selected Muon lr 1e-3; the 1B branch then **failed its continuation gate** on general BPB and retrieval, so the 2B extension, the SFT stage and the preference stage did not run. The 1B checkpoint is the terminal artifact and it is a base model, not an instruct model.

| finding | value | read from |
| --- | --- | --- |
| held-out code BPB improvement (gate: at least 2%) | 31.4582 % | `runs/code-probes/branch-1b-verdict.json` |
| Python held-out BPB improvement, the 55% bucket | 6.22374 % | `runs/code-probes/branch-1b-verdict.json` |
| MBPP+ syntax validity, from a 0.238 base | 0.386243 | `runs/code-probes/branch-1b-verdict.json` |
| MBPP+ pass@1, from a 0.0079 base | 0.021164 | `runs/code-probes/branch-1b-verdict.json` |
| HumanEval+ pass@1, base and branch alike | 0 | `runs/code-probes/branch-1b-verdict.json` |
| general-replay BPB regression (gate: at most 1.5%) -- FAILED | 2.25919 % | `runs/code-probes/branch-1b-verdict.json` |
| worst retrieval drop, at passkey d2048 (gate: at most 2 points) -- FAILED | 8 points | `runs/code-probes/branch-1b-verdict.json`, `runs/code-probes/branch-1b-stop.json` |
| five-task mean drop (gate: at most 1 point) | 0.733762 points | `runs/code-probes/branch-1b-verdict.json` |
| Q4_0 penalty of the exported branch, against the released base's 5.54% on the identical text | 6.23006 % | `runs/final/quant/code-branch-1b/quant-comparison.json` |
| selected probe learning rate | code-probe-lr0.001 | `runs/code-probes/verdict.json` |
| stages the failed gate cancelled | 2B extension, code/general SFT, execution-grounded DPO, and the QAT pass over the final code checkpoint | `runs/code-probes/branch-1b-verdict.json`, `runs/code-probes/branch-1b-stop.json` |

Caveats, by finding:

- **held-out code BPB improvement (gate: at least 2%)**
  - Weighted over three sources whose individual improvements are 6.2% (Python), 33.6% (JavaScript) and 76.7% (TypeScript). TypeScript is 25.7% of the weight and contributes about 20 of the 31.46 points -- two thirds of the headline from a quarter of the mixture.
  - Its held-out BPB of 0.139 is low enough to need explaining. File-level leakage is excluded (own source directory, salted per-repository split, zero rows admitted without a repository), but the TypeScript holdout is narrow and generated or vendored content would produce this honestly and mean little. Unresolved; see runs/final/daedalus-code-next.md step 0.
  - The Python figure, 6.2%, is the one to quote against a Python-first claim -- and both gate benchmarks are Python-only.
- **MBPP+ syntax validity, from a 0.238 base**
  - The signal that moved. This was preregistered as the more sensitive alternative to pass@1 at a scale where pass@1 is near zero.
- **MBPP+ pass@1, from a 0.0079 base**
  - 8 of 378 items against 3. At a 150M base this is movement off the floor, not a usable coding model.
- **HumanEval+ pass@1, base and branch alike**
  - Zero before and after. The harness is not at fault: the canonical-solution oracle returns 1.000 through the identical sandbox.
- **general-replay BPB regression (gate: at most 1.5%) -- FAILED**
  - The selected probe measured 1.48% at 250M tokens, inside the bound. Four times the tokens took it to 2.26%: the cost is still accruing, which is the argument against the 2B extension.
- **worst retrieval drop, at passkey d2048 (gate: at most 2 points) -- FAILED**
  - Paired McNemar p=0.013, and 8 of 8 discordant items moved against the branch. Not noise.
- **Q4_0 penalty of the exported branch, against the released base's 5.54% on the identical text**
  - The branch inherits the base's quantization damage and adds a little. It was trained in full precision with no QAT, and Phase 3 selected no recipe to inherit.
- **selected probe learning rate**
  - code-probe-lr0.001 is the lowest-code-BPB qualifying arm that holds general BPB within 1.5%

## 4. The finding that spans two phases

*Scope: released model.*

Phase 3's QAT recovery and Phase 8's code branch share nothing but their starting weights. Different data, different objective, different token budget, independently preregistered gates -- and each cleared its own progress criterion outright. Both then failed retention in the same place.

| finding | value | read from |
| --- | --- | --- |
| passkey d2048 drop under two unrelated continuations of the released base | 7.0 points (QAT lr 1e-3) and 8.0 points (code 1B) | `runs/qat-recovery/scored/qat-recovery-lr0.001.json`, `runs/code-probes/branch-1b-stop.json` |

Caveats, by finding:

- **passkey d2048 drop under two unrelated continuations of the released base**
  - Two unrelated treatments breaking the same capability in the same place points at the checkpoint rather than at either treatment. The released model's deepest retrieval appears to sit on a narrow basin that ordinary continued training leaves.
  - This is a hypothesis with two supporting observations, not an established mechanism. It was not isolated -- doing so needs an arm that varies only the starting weights, which no phase ran.
  - It is the single most consequential result for V2: it says the shipped checkpoint is hard to build on, which is a different problem from any that phases 4 to 7 were scoped to find.

## 5. Tokenizer lab (V2 only)

*Scope: proxy evidence.*

Three byte-level BPE vocabularies trained on a deterministic source-balanced sample and scored against the shipped SmolLM2 49,152 by a rule fixed before the numbers existed. **32,768 cleared every clause.** Nothing here was transplanted into any trained weights, because a vocabulary cannot be.

| finding | value | read from |
| --- | --- | --- |
| selected V2 vocabulary size | 32768 | `runs/tokenizer-lab/verdict.json` |
| code bytes-per-token change at 32,768 (negative is better) | -3.04016 % | `runs/tokenizer-lab/verdict.json` |
| tiny-model held-out BPB regression under the worse of the two protocols (bar: 0.5%) | 0.0376228 % | `runs/tokenizer-lab/verdict.json` |
| embedding Q6_K bytes saved against the incumbent | 33.3 % | `runs/tokenizer-lab/verdict.json` |
| the shipped SmolLM2 vocabulary fails the byte round-trip the candidates were held to | missing 21 of 256 byte characters; cannot round-trip U+40000-U+FFFFF | `runs/tokenizer-lab/addendum.json` |

Caveats, by finding:

- **selected V2 vocabulary size**
  - 32768 cleared every clause and has the best tiny-model BPB among those that did
- **tiny-model held-out BPB regression under the worse of the two protocols (bar: 0.5%)**
  - Measured on equal-compute tiny models, not on the 150M shape. It ranks vocabularies; it does not predict what a 150M or larger V2 would score.
- **embedding Q6_K bytes saved against the incumbent**
  - 10.3 MB of a ~101 MB Q4_0 artifact. This half of the result is arithmetic and transfers to any shape; the quality half does not.
- **the shipped SmolLM2 vocabulary fails the byte round-trip the candidates were held to**
  - A real defect in the shipped tokenizer, found while measuring the reference. It is not fixable in V1 for the same reason 32,768 is not adoptable in V1.

## 6. ShortConv channel death (V2 only)

*Scope: proxy evidence.*

Four decay schedules at the shipped 150M shape over 500M tokens, read on a coupled in_proj x kernel x out_proj instrument rather than on a weight-magnitude proxy. **Negative result: no schedule cleared the preregistered rule.** No result here revives a dead channel in the released model, and nothing claims to.

| finding | value | read from |
| --- | --- | --- |
| dead conv-channel fraction under the shipped 0.1 decay, at the 150M shape over 500M tokens | 53.8628 % | `runs/conv-health/verdict-paired.json` |
| held-out loss cost of removing every channel the control flagged dead | 2.98023e-08 nats | `runs/conv-health/verdict-paired.json` |
| lowest dead fraction any arm reached (weak-0.0133), against a 1% bar | 14.5182 % | `runs/conv-health/verdict-paired.json` |
| that arm's out_proj norm against the alive-channel baseline (limit 2x) | 2.33093 x | `runs/conv-health/verdict-paired.json` |
| schedules recommended for V2 | none | `runs/conv-health/verdict-paired.json` |

Caveats, by finding:

- **held-out loss cost of removing every channel the control flagged dead**
  - Effectively zero. The honest framing of the opportunity is parameters paid for and not used, not quality lost.
- **lowest dead fraction any arm reached (weak-0.0133), against a 1% bar**
  - Fourteen times the bar. This is not a threshold a slightly different ramp would have cleared.
- **that arm's out_proj norm against the alive-channel baseline (limit 2x)**
  - 1.61x at the shorter screen, 2.33x here: the cost grows with the decay clock rather than settling. That is the equilibrium objection, now measured.

## 7. Architecture Pareto proxies (V2 only)

*Scope: proxy evidence.*

Fifteen shapes at Stage A over 101M tokens, four parameter-matched finalists at Stage B over 252M, against the shipped 18x768 / 6-attention / 4-KV control. **No shape cleared every preregistered column, so the phase recommends none** -- and the control itself fails the KV-bytes ceiling the plan set.

| finding | value | read from |
| --- | --- | --- |
| recommended Pareto set | empty | `runs/architecture/stageb-recommendation.json` |
| the shipped shape's KV bytes per context token, against the plan's 6,144 ceiling | 8192 bytes | `runs/architecture/stageb-recommendation.json` |
| held-out BPB spread across the four Stage-B finalists, all inside the 0.5% floor | 0.264453 % | `runs/architecture/stageb-recommendation.json` |
| worst passkey drop among the arms that beat the control on KV bytes | 42 points | `runs/architecture/stageb-recommendation.json` |
| Apple Silicon decode | pending the Mac run; the decode column here is this box's CPU | `runs/architecture/stageb-recommendation.json`, `runs/architecture/decode-stageb.json` |

Caveats, by finding:

- **recommended Pareto set**
  - no shape clears every preregistered column, so this phase recommends none. That is a statement about the evidence, not about the shapes: see `unproven` for the columns still to be measured.
- **the shipped shape's KV bytes per context token, against the plan's 6,144 ceiling**
  - The most transferable finding here. Parameter and byte accounting is arithmetic and holds at any scale; the quality ranking does not.
- **held-out BPB spread across the four Stage-B finalists, all inside the 0.5% floor**
  - Attention-layer count barely moves BPB at this scale. What separated the arms was retrieval, which is the column the plan was right to add.
- **worst passkey drop among the arms that beat the control on KV bytes**
  - Every shape that bought a smaller KV cache paid for it in retrieval. That trade is the phase's real content, even though no arm cleared the gate.

## 8. Corpus, decontamination and mixture (V2 only)

*Scope: proxy evidence.*

Measured against the corpus **as built** -- the only one that exists. Two of five criteria pass, one is a measurement worth keeping, and two are gaps a rebuild closes by construction, which a 200,000-token rebuild smoke then demonstrated.

| finding | value | read from |
| --- | --- | --- |
| frozen decontamination index: n-grams over all five scored tasks at their scored splits, no per-task limit | 1371773 | `runs/corpus/decontam-index.json` |
| documents in the as-built corpus hitting the previously-unindexed split and limit gaps | 2 | `runs/corpus/phase7-gate-59.9b.json`, `runs/preflight/contam-exposure.json` |
| mixture L1 skew at the released run's 59.9B budget, against a 5-point bound | 11.4593 points | `runs/corpus/phase7-gate-59.9b.json` |
| worst source epoch count at 59.9B under the four-epoch cap | 4 | `runs/corpus/phase7-gate-59.9b.json` |
| as-built manifests carrying a source revision, a filters block and a builder sha | 0 of 10 | `runs/corpus/phase7-gate-59.9b.json` |
| mixture weights selected by the proxy sweep | baseline | `runs/corpus/mixture-verdict-probe.json` |

Caveats, by finding:

- **frozen decontamination index: n-grams over all five scored tasks at their scored splits, no per-task limit**
  - The corpus as built was indexed against the wrong splits for ARC-Easy and OpenBookQA and truncated at 2,000 items. That is the gap this index closes.
- **documents in the as-built corpus hitting the previously-unindexed split and limit gaps**
  - `docs_filtered` is 0 -- the negative control -- so dataprep removed everything it indexed.
  - Decided from a 1.32% sampled scan. A hit is decisive; a zero bounds the document rate at 3.2e-05 rather than proving absence.
  - 158M fineweb-edu tokens, 3.0% of the largest source, were never in front of the scanner. That gap was invisible until the scan artifact and the manifests were compared, and it is now its own criterion.
- **mixture L1 skew at the released run's 59.9B budget, against a 5-point bound**
  - The corpus delivers the blueprint inside the bound to ~55.4B, and to ~56.9B with the dialogue source dropped. The released run's own budget is past both.
  - Below ~53B the entire skew is one source: dropping everyday-conversations takes it to exactly 0.0000 at 30B and 50B.
- **worst source epoch count at 59.9B under the four-epoch cap**
  - The cap worked. Repetition is bounded; what is not delivered is the blueprint.
- **as-built manifests carrying a source revision, a filters block and a builder sha**
  - They predate `dataprep.source_provenance`. The rebuild smoke's manifest carries all of it, so the criterion is closed by construction rather than retroactively.
- **mixture weights selected by the proxy sweep**
  - best admissible arm 'quality-heavy' gains 0.0810% of aggregate BPB, under the preregistered 0.5000%
  - The best re-weighting bought 0.08% of aggregate BPB against a 0.5% bar. Mixture weights are not where the headroom is; supply is.

## 9. How the program ran

*Scope: process.*

Preregistered gates, and what they cost. Three of the four experimental phases returned a negative result and one returned no recommendation; none of those thresholds moved after a number was seen.

| finding | value | read from |
| --- | --- | --- |
| phases that stopped on a preregistered gate rather than continuing | Phase 3 (no QAT winner), Phase 5 (no schedule), Phase 6 (no shape), Phase 8 (stopped at 1B) | `runs/qat-recovery/verdict.json`, `runs/conv-health/verdict-paired.json`, `runs/architecture/stageb-recommendation.json`, `runs/code-probes/branch-1b-stop.json` |
| program base SHA; the default branch is unchanged from it | 99232b4aaaeee6c507611094593f39e861178ebe | `runs/vast-program/state.json` |
| program start (UTC) | 2026-08-24T10:44:35.993665Z | `runs/vast-program/state.json` |

## Artifact manifest

SHA-256 recomputed at finalization and compared against the digest the scorecards recorded when they measured the file.

| artifact | role | sha256 | verified |
| --- | --- | --- | --- |
| `released-base-checkpoint` | the immutable baseline every gate is measured against | `cfbf27dccf93a07c` | match |
| `released-base-f16-gguf` | released FP16 artifact, the FP16 side of the Q4 penalty | `95f1e6795b715be8` | match |
| `released-base-q4_0-gguf` | released Q4_0 artifact, the one a user runs | `1d45ce41239b9e03` | match |
| `code-branch-1b-checkpoint` | Daedalus-Code V1, 1B continued-pretraining tokens; terminal | `52c451a1768f17a4` | match |
| `code-branch-1b-f16-gguf` | Daedalus-Code FP16 export, stock llama.cpp | `b087eb098bd575c1` | match |
| `code-branch-1b-q4_0-gguf` | Daedalus-Code Q4_0 export, stock llama.cpp | `3e8302cdd3bcb6c1` | match |
| `qat-recovery-lr0.001-f16-gguf` | rejected QAT arm, best penalty reduction of the three | `c12655413bb81fff` | match |
| `qat-recovery-lr0.001-q4_0-gguf` | rejected QAT arm, Q4_0 side | `1fc643d4a55c4d69` | match |
| `tokenizer-v32768` | Phase 4's selected V2 vocabulary; never transplanted into weights | `e023697825569546` | fingerprint only |
