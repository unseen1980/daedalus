# Document-aligned packing: measured on our corpus, then closed on published evidence

**Date:** 2026-08-10, during `abl-arch` arm 1 (CPU-only, no GPU cost).
**Question:** the blueprint locks *"document-aligned packing via FlexAttention"*
(`DAEDALUS-BLUEPRINT-v6.md:37` and `:93`, echoed in `AGENT.md` §2). It is **not
implemented**. `hero` is the last chance to add it. Should it be?

**Answer: no.** Implement nothing. The gap is real and larger than I expected,
but the best-matched published ablation — same eval suite, same context length,
near-identical data mixture — reports **no effect**. Cost avoided: ~4% of
`hero`'s throughput (~5.7 GPU-h, **~$2.55**) plus a change to the core attention
path four days before a 5.9-day run.

---

## 1. What is actually implemented

| piece | state | evidence |
|---|---|---|
| EOS-separated dense packing | implemented | `daedalus/data.py:8-14`, `:145-161` |
| per-token document ids | implemented but **unused** | `data.py:220-224`, docstring: *"for a **future** document-aware attention mask"* |
| document-aware attention mask | **missing** | `daedalus/model.py:133` is `F.scaled_dot_product_attention(..., is_causal=True)` |
| FlexAttention | **missing** | no reference anywhere in `model.py` |

So the loader already derives the doc ids; nothing consumes them. This is a
genuine deviation from a locked blueprint decision, not a mislabelling.

`AGENT.md` §4's `smoke` job also asks to verify *"FlexAttention compiles"* — that
check has never been meaningful, because FlexAttention is not on the path.

## 2. How much cross-document attention we actually have

Measured on the **real shards**, not modelled: 2,000 real 2048-token windows per
source (197 for `everyday-conversations`, which is all it has), doc ids computed
by `data.py`'s own rule (increment *after* an EOS crossing, `data.py:222-224`).

`cross_attn%` is the fraction of causal (query, key) pairs whose two tokens come
from different documents — i.e. exactly the attention mass an intra-document mask
would delete. `tok w/foreign%` is the fraction of positions with at least one
foreign token in context.

| source | corpus share | cross_attn % | tok w/foreign % | mean docs / 2048-seq |
|---|---:|---:|---:|---:|
| fineweb-edu | 26.4% | 43.2 | 54.7 | 3.02 |
| dclm-baseline | 15.8% | 35.8 | 46.9 | 2.62 |
| finephrase | 14.5% | 41.1 | 51.6 | 2.90 |
| finemath-3plus | 9.5% | 29.6 | 38.9 | 2.17 |
| infiwebmath-3plus | 9.5% | 30.1 | 39.5 | 2.18 |
| stack-edu-python | 8.5% | 23.4 | 31.4 | 2.02 |
| cosmopedia-v2 | 6.7% | 66.3 | 80.5 | 3.85 |
| finepdfs-edu | 6.2% | 17.3 | 24.6 | 1.63 |
| finewiki-en | 2.9% | 37.0 | 48.1 | 2.78 |
| everyday-conversations | 0.003% | 91.2 | 95.5 | 12.46 |
| **corpus-weighted** | **100%** | **37.3** | **47.8** | **~2.7** |

**37.3% of the attention this model computes is across a document boundary**, and
nearly half of all training positions see at least one foreign token. That is not
a rounding error, and it is why the question deserved measuring rather than
dismissing.

Raw numbers: `/tmp/docmask-measure.json` (regenerable; the script is inline in
this document's git history).

Sanity check that the two independent estimates agree: mean document length for
`fineweb-edu` is 1054 tokens, so 2048/1054 = 1.94 expected boundaries per window,
i.e. 2.94 documents — against 3.02 measured. Every source matches to ~3%.

## 3. What the published evidence says it would buy

Three sources, in increasing order of how well they match our setting.

### Zhao et al. 2024, *Analysing The Impact of Sequence Composition on LM Pre-Training* (ACL 2024)

The paper that introduced intra-document causal masking (INTRADOC). **1.3B
params, 150B tokens of SlimPajama, 2K context** — the same context length as us.

| metric | MixChunk baseline | IntraDoc | delta |
|---|---:|---:|---:|
| perplexity @2K | 9.172 | 8.410 | **−0.883 (−8.3% rel)** |
| in-context learning (7 text-classification sets) | 63.54% | 70.52% | **+6.98 pts** |
| knowledge memorisation (EM, NQ+TQA) | 10.33% | 11.60% | +1.27 pts |
| training efficiency | — | — | **−4.0%** |

Taken alone this looks compelling. Two reasons it is weaker than it looks for us:

- **It reports none of our five tasks.** Not HellaSwag, ARC-Easy, PIQA,
  OpenBookQA or WinoGrande. Our bar is defined on those and nothing else.
- **The +6.98 headline is few-shot ICL**, where the in-context examples *are* the
  neighbouring documents — the mechanism is maximally exposed. Our suite is
  zero-shot cloze, where it is not exposed at all.

### Llama 3

Trained with intra-document masking at 8,192 context. Reported as **limited
impact during short-context pretraining, significant benefit for long-context
extension.**

### HuggingFace, *The Smol Training Playbook* — the decisive one

This is the ablation to read, because it matches us on every axis that matters:

- **1B model, 45B tokens** (we are 160M / 60B — same order, same regime),
- data mixture **FineWeb-Edu + FineMath + Python-Edu** — three of our four
  largest sources,
- evaluated on **HellaSwag, ARC, PIQA, OpenBookQA, WinoGrande, MMLU** — our suite,
- run by the team whose tokenizer we use byte-identically and whose SmolLM2-135M
  is our stated peer.

Result: *"The results showed identical loss curves and downstream evaluation
scores compared to standard causal masking"*, with **a small improvement on PIQA**
the only notable difference, and the conclusion *"we don't observe a noticeable
impact on short context tasks."*

They adopted it for SmolLM3 anyway — explicitly because it is *"crucial when
scaling to long sequences to speed up the training"*, for their 4k→64k context
extension. **Daedalus does no context extension.** Our context is 2048 at
pretraining and 2048 at inference; the reason SmolLM3 adopted it does not apply
to us.

## 4. Why the two results are not in conflict

Zhao et al. measure perplexity on held-out documents and few-shot ICL; HF measure
zero-shot cloze accuracy. Cross-document distraction plausibly costs real
perplexity (Zhao et al. also report the effect is largest on GitHub code, −1.3
PPL, where document independence is strongest) while leaving zero-shot cloze
accuracy — a ranking of 4 short completions — untouched. Both can be true, and
**our success bar is written in the quantity HF measured**, not the one Zhao et
al. did.

This matters because our own `val_bpb` would probably *improve* if we implemented
it. That would look like a win on the dashboard and buy nothing on the bar.

## 5. Decision

**Do not implement before `hero`.** Costs and risks, all certain; benefit, not
demonstrated on our objective:

| | |
|---|---|
| throughput cost | 4.0% (Zhao et al.'s own measurement) = ~5.7 GPU-h of `hero` = **~$2.55** |
| implementation risk | a change to `model.py:133`, the attention path shared by both `abl-arch` arms, ~4 days before a 5.9-day $63.78 run |
| ablation consistency | `abl-arch` arm 1 has already trained 5B tokens without it; adding it for `hero` would mean `hero` is not the architecture the ablation compared |
| benefit on the five tasks in the bar | **not demonstrated** — the one ablation that measured them found none |

If the operator wants it anyway as a fidelity matter, the honest framing is that
it is a blueprint-conformance change, not a quality one, and it should be paid
for out of a separate budget rather than out of `hero`'s.

**What changes instead:** nothing in the code. The blueprint deviation is now
costed and evidenced rather than a one-line admission at the end of the gate
draft, and `AGENT.md` §4's "FlexAttention compiles" smoke check is recorded as
vestigial.

## 6. What this does not settle

- Zhao et al.'s perplexity gain is real and we are choosing not to take it. If a
  future Daedalus does long-context extension, this decision must be revisited —
  that is precisely the case both Llama 3 and HF call out.
- The HF ablation is at 1B, not 160M. Smaller models could in principle be more
  distractible. No evidence either way was found; stated as a residual, not
  argued away.
- Nobody has measured this on *our* model. A direct test would be two 5B-token
  runs (~$11 and ~23 h) — the same price as the entire `abl-arch`, to chase an
  effect the best-matched published ablation measured as zero.

### Sources

- [Zhao et al., *Analysing The Impact of Sequence Composition on Language Model Pre-Training*, ACL 2024](https://arxiv.org/abs/2402.13991) ([HTML](https://arxiv.org/html/2402.13991), [ACL Anthology](https://aclanthology.org/2024.acl-long.427/))
- [The Smol Training Playbook (HuggingFace)](https://huggingfacetb-smol-training-playbook.hf.space/)
- [karpathy/llm.c discussion #690](https://github.com/karpathy/llm.c/discussions/690) — request only, no data; cited so it is not mistaken for evidence
