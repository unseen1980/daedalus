# The peer token budgets, checked against primary sources

The `tokens` column of `runs/eval/peer-table.md` is not decoration. The hero
gate's central quality argument is that **token count does not explain the
variance among these peers** — that a 42.2 at a small budget sitting above a 41.0
at 300B is why 40B of modern curated data is a reasonable bet against 300B of
Pile-era data. That argument is read directly off this column, so a wrong entry
would misprice a $41 decision.

Every figure was checked against the primary source on **2026-08-10**. Two were
wrong.

| model | our table said | **actual** | source |
|---|---|---|---|
| Pythia-160M | 300B | **299,892,736,000** ✓ | EleutherAI model card |
| GPT-neo-125M | 300B | **300B over 572,300 steps** ✓ | EleutherAI model card |
| **OPT-125M** | 300B | **180B / 800 GB** ✗ | OPT card (Zhang et al. 2022) |
| **GPT-2 124M** | "~30B" in prose | **undisclosed** ✗ | OpenAI / HF card |
| MobileLLM-125M | 1T | 1T ✓ | MobileLLM paper |
| SmolLM2-135M | 2T | 2T ✓ | SmolLM2 paper |

## OPT-125M is 180B, not 300B

The OPT corpus is **"180B tokens corresponding to 800GB of data"**. The 300B in
`scripts/peer_table.py` appears to have been assumed from the neighbouring
Pile-era models rather than read off the source. Corrected.

Caveat kept rather than smoothed over: the card gives no **125M-specific** budget
— it details only the 175B model's compute — so 180B is the *corpus*, and whether
the 125M variant consumed all of it is not stated. It is the best-supported
figure, not a certainty.

## GPT-2's budget was never published, and it is the load-bearing row

The gate quotes GPT-2 as "~30B tokens" and already flags it as the softest row.
It is softer than that: OpenAI **did not disclose it at all**. The HF card is
explicit — *"The training duration was not disclosed, nor were the exact details
of training."* The GPT-2 paper gives the dataset (8 million documents, ~40 GB of
text) and no token count or epoch count.

So "~30B" is not a cited number and not even a derived one. The only derivable
quantity is **one epoch of WebText ≈ 40 GB ÷ ~4 bytes/token ≈ 10B tokens**, with
the number of epochs unpublished. Commonly-cited estimates land at ~8–10B for a
single pass.

**This strengthens the gate's argument rather than weakening it**, which is worth
saying plainly because it would be easy to present it the flattering way and be
accused of it later. The claim is "a small-budget model beats a 300B one, so
budget is not destiny". Replacing GPT-2's budget with something *smaller and
less certain* makes the gap it wins by larger, not smaller. But the honest
statement is now **"undisclosed, plausibly ~10–30B"**, and the argument should
lean on the *whole cluster* rather than on that single row:

| model | budget | our 5-task mean |
|---|---|---|
| GPT-2 124M | **undisclosed** (~10–30B?) | **42.2** |
| OPT-125M | 180B | 42.1 |
| GPT-neo-125M | 300B | 41.9 |
| Pythia-160M | 300B | 41.0 |
| SmolLM2-135M | 2T | 51.2 |

Read without GPT-2 at all, the argument still holds and is now *better* sourced:
**OPT at 180B scores 42.1 and Pythia at 300B scores 41.0.** A 1.7× token
difference, in the wrong direction, among models of the same era and size. The
1.2-point spread across 180B→300B is recipe and data, not budget. Only SmolLM2's
2T — a 6.7× jump *and* a modern corpus — breaks out, at +9.

## What this does not change

The bar. Daedalus must still clear **~42.2** to beat every 300B-class peer, and
that number is a *measurement on our harness*, not a published budget-dependent
claim. Nothing here moves any accuracy. It changes only how confidently the
tokens column can be cited in the writeup and the gate.

Sources:
- <https://huggingface.co/openai-community/gpt2>
- <https://huggingface.co/facebook/opt-125m>
- <https://huggingface.co/EleutherAI/gpt-neo-125m>
- <https://huggingface.co/EleutherAI/pythia-160m>
- <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
