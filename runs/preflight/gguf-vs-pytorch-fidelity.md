# Does the GGUF compute the same function as the model we train?

**2026-08-10 ~14:40Z.** Nothing had ever checked this. Every quality number this
project reports describes the **fp32 PyTorch model**; the artifact the operator
runs is a **Q4_0 GGUF**. The links in between were each verified separately —
our weights load into the real `Lfm2ForCausalLM` and give bit-identical logits
(`test_export_hf_model_roundtrip_matches_original_logits`), and the GGUF's
vocabulary matches SmolLM2 id-for-id (`gguf-tokenizer-and-q4-damage.md`) — but
**nobody had compared the two ends numerically.**

Every existing gate is blind to a break here, by construction:

| gate | why it cannot see it |
|---|---|
| `passes_threshold` (fp16 vs Q4_0 ppl) | both sides are llama.cpp; a shared error cancels |
| `llama-bench` | times decode, never checks output |
| vocab check | compares id→token strings, not the graph |

## Answer: the GGUF is faithful. The scare was my own measurement.

Final paired comparison, same frozen checkpoint (step 8,430, 3.97B tokens),
same text, llama.cpp's exact scoring convention:

| | perplexity |
|---|---|
| PyTorch fp32 | 9.0578 |
| llama.cpp f16 GGUF | 8.8776 |
| **difference** | **−1.99%** |

The residual is chunk misalignment, not the graph: HF and llama.cpp still
disagree on ~40 tokens in 150,000 (0.027%) from merge edge cases, and once the
streams drift the two harnesses score slightly different spans. Recorded as a
known limit rather than chased.

## How the false alarm happened, because it is the interesting part

The first comparison read **+14.4%** — llama.cpp much worse — and the way that
number dissolved is worth writing down.

**1. A moving checkpoint.** The trainer rewrites `checkpoint.pt` every 30
minutes. I exported a GGUF at 13:52 (step 8,005) and compared it against a
PyTorch model loaded at 14:25, which was **step 8,430** — 220M tokens of extra
training, mid-decay. Everything after this used a frozen `/tmp` copy and an
export built from that same copy. *This project's own rule about isolating
run state from live state, and I broke it.*

**2. f16 rounding, ruled out by measurement rather than argument.** Rounding
the weights to f16 and back, with fp32 compute, moved perplexity from
**17.1454 to 17.1452**. So the gap was not precision.

**3. A context sweep that made no sense.** The gap by context length came out
**+12%, −2%, +3%, +14%** at n_ctx 64/128/256/512 — non-monotonic, and llama.cpp
*better* at 128. A real graph error (wrong RoPE style, dropped QK-norm) would
grow monotonically with position. That incoherence was the clue that the
harness, not the model, was the variable.

**4. The actual cause.** `data/eval/ppl-finewiki-150k.txt` contained **153
literal `<|endoftext|>` strings** as document separators. The HF tokenizer maps
each to the single id **0** — the ids the corpus was built with.
`tools/perplexity/perplexity.cpp:475` calls
`common_tokenize(ctx, params.prompt, true)`, whose `parse_special` defaults to
**false**, so llama.cpp spelled each one out as ~7 ordinary tokens.

That single fact explains every symptom: the token-count divergence (llama.cpp
read **294** chunks where HF read 292 — ~1,000 extra tokens, ≈153 × 6.5), the
inflated perplexity (~1,000 tokens of text the model has never seen in that
form), and the erratic context sweep (with only ~500 tokens scored, the answer
depended on whether a mangled separator happened to land inside the scored
window).

Removing the separators collapsed the gap from **+14.4% to −1.99%** and brought
the chunk counts into line: **292 vs 293**, where the residual is the 0.027%
merge difference above.

## The bug this exposed, and what it cost

The eval text is not a scratch file — it is the input to `export.py`'s Q4_0
quantization gate, and to every perplexity number this project has recorded.
Measured on one real GGUF over one identical text span:

| text | fp16 | Q4_0 | delta |
|---|---|---|---|
| with separators (as it was) | 9.6791 | 9.8413 | **+1.676%** |
| separators removed | 8.8776 | 9.0357 | **+1.781%** |

So the file made absolute perplexity read **~9% high**, and made the Q4_0 damage
— the number the QAT decision rests on — read **~6% low**. The gate survived
because it is a *ratio* and both sides ate the same junk, which is precisely why
nothing caught it.

**Direction matters here and it is the reassuring direction.** Q4_0 damage is
worse than recorded, not better, so *"QAT is load-bearing"* is strengthened.
`gguf-tokenizer-and-q4-damage.md`'s +2.576% was measured on the dirty file and
is therefore an underestimate; it was already above the 1.0% gate, and the
correction moves it further above.

Fixed by removing the 153 separators from the file (531,084 → 529,401 bytes).
`llama-perplexity` is its only consumer, so nothing else changes. Both `abl-arch`
arms export *after* this change, so the ablation stays internally consistent.

`test_the_perplexity_text_holds_no_literal_special_token_strings` pins it,
mutation-checked: reinserting one `<|endoftext|>` fails the test.

## What is now known, and what is still not

**Known.** The Q4_0 GGUF is a faithful rendering of the trained model to within
~2%, which is itself dominated by harness alignment rather than the model. The
export path — checkpoint → HF → f16 GGUF → Q4_0 — runs end to end on a real
trained checkpoint in 14.8 s and produces a model that generates fluent English
and scores sane perplexity.

**Not known.** This says nothing about *task* accuracy through the GGUF beyond
`gguf_task_delta.py`'s single HellaSwag run near the chance floor.

**The residual divergence, now characterised rather than left as a mystery.**
Walking both token streams over the cleaned file and resyncing after each
disagreement gives **55 divergences in 150,017 tokens — 0.037%**, in exactly
three shapes:

| n | text | HF | llama.cpp |
|---|---|---|---|
| 35 | `\n\n` | `[198, 198]` | `[1116]` |
| 14 | `' ½'` | `[3351, 138]` | `[216, 16738]` |
| 5 | `'  '`, `'\n\n\n'` | — | — |

Both are **context-sensitive**, not blanket disagreements: in isolation
`a\n\nb` tokenizes identically on both sides (`[81, 198, 198, 82]`), and it is
only in certain surroundings that llama.cpp takes the merged `\n\n` token. The
`' ½'` case is the general one — a space followed by a multi-byte UTF-8
character, where HF's byte-level BPE merges the space with the *first byte*
(`[' �', '�']`) and llama.cpp merges space-then-whole-character. That class
covers ` é`, ` —`, ` “`, ` €` and similar, which are not rare in web text.

**Which side is "right" is the wrong question — consistency is the point.** The
corpus was tokenized with HF, so HF's segmentation is what the model actually
learned, and any llama.cpp disagreement is a small train/inference mismatch on
the shipped artifact. Measured, that mismatch costs nothing detectable: the
paired gap is −1.99% and runs in llama.cpp's *favour*. Not worth touching a
pre-tokenizer registration over, but it does mean PyTorch and GGUF perplexities
should never be quoted as if they were the same measurement.

## Aside: the first text a Daedalus model ever generated

`setup_llama_cpp` builds `llama-quantize`, `llama-perplexity` and `llama-bench`
— **not `llama-cli`** — so no Daedalus model had produced a token of text.
Built it (needs `-DLLAMA_BUILD_UI=OFF`; `llama-cli` now pulls in `llama-ui`,
which downloads a prebuilt tarball at build time and fails without network
access to that bucket).

Greedy, temperature 0, from the step-8,430 checkpoint at 3.97B tokens:

```
The capital of France is   ->   Paris.\n\nThe city is located in the
Water boils at a temperature of  ->   100°C.
The largest planet in our solar system is  ->   the sun. ...
```

Two right, one wrong, which is about what 150M parameters and 4B tokens buys.
The PyTorch model puts **P(" Paris") = 0.183** as its top token. Worth noting
that `llama-cli`'s own output for the same prompt looked like a non-sequitur —
that was the depth-0 comparison against a **different (step-8,005) GGUF** during
phase 1 above, and it is not evidence of anything.
