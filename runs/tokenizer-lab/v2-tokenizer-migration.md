# V2 tokenizer migration report

Phase 4 asks whether a future from-scratch V2 should keep SmolLM2's 49,152-token vocabulary or adopt 24,576, 32,768 or 40,960.

**Scope.** A tokenizer cannot be transplanted into a trained model: every embedding row and every output logit is indexed by the vocabulary the model was trained under. Nothing here changes released V1 weights or Daedalus-Code, and no result below should be read as a gain on either.

## The preregistered rule

> A candidate is selected only if no domain regresses bytes per token by more than 5%, code improves or ties, tiny-model BPB improves or stays within 0.5%, and projected embedding bytes fall materially.

Rule digest `4557ac1d70b0f27be7f86395370cd41d`, written before any measurement. Thresholds: no domain worse than 5%, code <= 0%, tiny-model BPB <= 0.5%, embedding bytes down at least 5%.

## Bytes per token vs the incumbent (SmolLM2), by domain

Negative is better: the candidate needs fewer tokens for the same bytes. This is the comparison the rule reads.

| tokenizer | general | math | technical | dialogue | code | all |
|---|---|---|---|---|---|---|
| 24576 | +1.34% | -8.48% | +0.73% | +0.89% | +0.09% | +0.38% |
| 32768 | -1.23% | -11.89% | -1.79% | -0.72% | -3.04% | -2.32% |
| 40960 | -2.98% | -13.61% | -3.43% | -1.71% | -4.99% | -4.10% |
| 49152-matched | -4.22% | -15.08% | -4.74% | -2.38% | -6.33% | -5.37% |

## Bytes per token vs a size-matched control

`49152-matched` is 49,152 tokens trained on **this** sample with the same trainer. Candidate-vs-matched isolates the cost of shrinking the vocabulary; candidate-vs-SmolLM2 above mixes that with the gain from retraining on this corpus, and would otherwise credit *size* for a *corpus-match* effect.

| tokenizer | general | math | technical | dialogue | code | all |
|---|---|---|---|---|---|---|
| 24576 | +5.33% | +5.74% | +5.22% | +3.20% | +6.04% | +5.45% |
| 32768 | +2.87% | +2.77% | +2.82% | +1.62% | +3.10% | +2.89% |
| 40960 | +1.19% | +1.28% | +1.25% | +0.65% | +1.26% | +1.21% |
| 49152-smollm2 | +4.05% | +13.11% | +4.52% | +2.32% | +5.96% | +5.09% |

## Artifact cost

| tokenizer | embedding params | Q6_K MiB | vs incumbent | KV bytes/context-token |
|---|---|---|---|---|
| 24576 | 18,874,368 | 14.77 | +50.0% | 6,144 |
| 32768 | 25,165,824 | 19.69 | +33.3% | 6,144 |
| 40960 | 31,457,280 | 24.61 | +16.7% | 6,144 |
| 49152-matched | 37,748,736 | 29.53 | +0.0% | 6,144 |
| 49152-smollm2 | 37,748,736 | 29.53 | +0.0% | 6,144 |

`token_embd.weight` ships Q6_K and is tied, so that one tensor is both the input table and the output projection (`runs/preflight/token-embd-quant-grid.md`). The KV column is identical by construction: the cache is attention-shaped, so a vocabulary change moves the embedding tensor and nothing in it.

## Byte coverage and round trips

| tokenizer | byte characters | round trip |
|---|---|---|
| 24576 | 256/256 | pass |
| 32768 | 256/256 | pass |
| 40960 | 256/256 | pass |
| 49152-matched | 256/256 | pass |
| 49152-smollm2 | 235/256 | FAIL |

The incumbent's row is a finding, not a formatting error. SmolLM2's vocabulary is missing 21 byte-level characters; most stand for bytes that never occur in valid UTF-8, but 0xf1-0xf3 are four-byte lead bytes, so code points U+40000-U+FFFFF are silently dropped rather than rejected. Every candidate covers all 256.

## Held-out bits per byte

Bits per **byte**, never per-token perplexity: a larger vocabulary improves per-token likelihood by packing more bytes into each token, with no improvement in the model.

| arm | protocol | tokens trained | BPB | vs incumbent |
|---|---|---|---|---|
| 24576 | equal-bytes | 220,069,888 | 1.2126 | -0.44% |
| 24576 | equal-tokens | 200,015,872 | 1.2272 | -0.63% |
| 32768 | equal-bytes | 214,564,864 | 1.2184 | +0.04% |
| 32768 | equal-tokens | 200,015,872 | 1.2299 | -0.41% |
| 40960 | equal-bytes | 210,894,848 | 1.2221 | +0.34% |
| 40960 | equal-tokens | 200,015,872 | 1.2315 | -0.28% |
| 49152-smollm2 | equal-bytes | 218,497,024 | 1.2179 | +0.00% |
| 49152-smollm2 | equal-tokens | 200,015,872 | 1.2350 | +0.00% |

## Can stock llama.cpp convert these?

The constraint that decides whether any of this is actionable: unmodified stock llama.cpp is a fixed program decision. `conversion/base.py::get_vocab_base_pre` identifies a BPE pre-tokenizer by hashing the token **ids** a fixed probe string encodes to, checks that against a hard-coded list, and raises `NotImplementedError` for anything absent. The hash is over ids, so it moves with the vocabulary and the merges -- a newly trained tokenizer cannot match a registered hash however faithfully it copies SmolLM2's pre-tokenizer configuration.

| tokenizer | converts | pre-tokenizer recognised |
|---|---|---|
| 24576 | no | no |
| 32768 | no | no |
| 40960 | no | no |
| 49152-matched | no | no |
| 49152-smollm2 | yes | yes |

## Verdict

**Selected: 32768.** 32768 cleared every clause and has the best tiny-model BPB among those that did

**Not actionable as it stands.** Stock llama.cpp will not convert a model carrying 32768: its BPE pre-tokenizer hash is not in the converter's hard-coded list, and unmodified stock llama.cpp is a fixed program decision. `49152-matched` -- the incumbent's own vocabulary size, retrained on this sample -- fails identically, so the blocker is that the vocabulary is newly trained, not that it is smaller; choosing a different size cannot route around it. Adopting any new vocabulary for V2 therefore depends on that hash being registered upstream and reaching a release, which is outside this program, or on keeping SmolLM2's vocabulary and its measured costs. The fertility and BPB results above are unaffected -- they are what the rule read, and the rule's output is unchanged -- but they do not clear this constraint.

| candidate | selectable | failed clauses |
|---|---|---|
| 24576 | no | code-fertility |
| 32768 | yes | - |
| 40960 | yes | - |

## The sample

1.374 GB across 19 sources, split disjointly by SHA-256 of each document's bytes: holdout 28 MB, lm-train 905 MB, tokenizer-train 442 MB.

| domain | bytes |
|---|---|
| code | 126.5 MB |
| dialogue | 1.7 MB |
| general | 1,050.1 MB |
| math | 84.0 MB |
| technical | 112.0 MB |

Sources that could not fill their share, recorded rather than force-filled:

- `everyday-conversations`: 1.73 MB of 28.00 MB (6.2%)
