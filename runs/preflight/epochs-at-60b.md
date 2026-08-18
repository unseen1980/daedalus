# Per-source epoch counts at 60B — required before launch

> **SUPERSEDED 2026-08-10 23:50Z by `runs/preflight/split-carve-and-epochs-at-58b.md`.**
> Every number below was computed against a *modelled* split — a flat 2% haircut
> per source — and the real carve reserves whole shard files, taking 3.67%
> unevenly and dropping `everyday-conversations` entirely. On the split `hero`
> actually reads, 60B has `l1_skew` 10.21 and **refuses**. The ask is now 58B.
> Kept as the record of how the 60B decision was analysed, not as a description
> of the corpus.


The operator's 60B decision attached this condition:

> `~4.3 epochs` is just past the ~4-epoch guideline the plan cites. That is
> acceptable and was decided knowingly, but note the exact per-source epoch
> counts in `STATUS.md` before launch, and **flag any single source that lands
> far above the rest.**

Measured through `train.mixture_preflight` on the **train split `hero` actually
reads** (2% per-source holdout carved), with `fineweb-edu` assumed to finish the
~1.17B it has left. Train split 17,226,005,442 tokens.

## Nothing lands far above the rest

| source | target | effective | on disk (split) | epochs | capped |
|---|---|---|---|---|---|
| `cosmopedia-v2` | 5.00% | 6.21% | 931,000,564 | **4.00** | YES |
| `dclm-baseline` | 22.50% | 22.05% | 3,307,502,102 | **4.00** | YES |
| `everyday-conversations` | 2.00% | **0.00%** | 395,501 | **4.00** | YES |
| `finepdfs-edu` | 8.00% | 7.84% | 1,176,001,504 | **4.00** | YES |
| `fineweb-edu` | 37.50% | 36.75% | 5,512,500,000 | **4.00** | YES |
| `finewiki-en` | 3.00% | 2.94% | 441,004,918 | **4.00** | YES |
| `stack-edu-python` | 9.00% | 7.91% | 1,186,745,357 | **4.00** | YES |
| `finephrase` | 7.00% | 8.78% | 2,024,846,623 | 2.60 | |
| `finemath-3plus` | 3.00% | 3.76% | 1,323,008,679 | 1.71 | |
| `infiwebmath-3plus` | 3.00% | 3.76% | 1,323,000,194 | 1.71 | |

**The answer to the operator's question is: no source lands far above the rest.**
The spread is 1.71 to 4.00 epochs and the maximum *is* the cap, so the worst case
is exactly the bound the blueprint chose, not an outlier past it. This is the
issue #5 fix working: at 60B on the old 14.2B corpus the cap could not be
satisfied at all, the guard silently disengaged, and
`everyday-conversations` took 2% of 60B as **2,973 epochs** of ~2,200
conversations. It is now pinned to **0.00%** effective share.

`everyday-conversations` showing "4.00 epochs" at a 0.00% share is not a
contradiction: 4 epochs of 395,501 tokens is 1.58M tokens, which rounds to 0.00%
of 60B. That 2.00-pt deviation is the already-recorded, unfixable one — only
~2,200 conversations exist in total.

## 60B sits exactly on the knee of the mixture curve

This is the part that was not previously measured, and it is why the margin
against `hero`'s 10.0-pt refusal is thin.

| budget | l1_skew | sources capped | newly capped at this budget |
|---|---|---|---|
| 30–51B | **3.99** | 1 | `everyday-conversations` |
| 52B | 3.99 | 2 | `stack-edu-python` |
| 53B | 4.08 | 2 | |
| 55B | 4.73 | 2 | |
| 57B | 5.34 | 2 | |
| 58B | 5.63 | 6 | `dclm-baseline`, `finepdfs-edu`, `fineweb-edu`, `finewiki-en` |
| 59B | 6.38 | 6 | |
| **60B** | **9.01** | **7** | `cosmopedia-v2` |

Read that curve carefully, because it changes how the budget choice looks:

* **Everything up to 51B costs nothing in mixture terms.** Skew is 3.99 —
  *identical to 40B* — and the only capped source is the one that cannot be
  fixed. The corpus is, in effect, sized for ~51B at 4 epochs.
* Above 51B the cap starts binding on real sources and the mass they give up is
  redistributed to the three with headroom. `finephrase` ends at **8.78%**
  against a 7.00% target — 25% more than intended.
* The 59B → 60B step alone is **6.38 → 9.01**, when `cosmopedia-v2` caps.

So 60B's 0.99 pts of margin against the refusal limit is not an accident or an
error: it is the direct cost of choosing a budget one notch past the knee.

## What this does and does not imply for quality

Being honest about the size of this effect, because it would be easy to overstate
it in either direction.

The skew is real but its *composition* is mild for the eval suite we are measured
on. What actually changes at 60B versus 51B: less code (`stack-edu-python`
9.00% → 7.91%), slightly less web (`fineweb-edu` 37.50% → 36.75%,
`dclm-baseline` 22.50% → 22.05%), and more synthetic and maths (`finephrase`
+1.78, `cosmopedia-v2` +1.21, the two maths sources +0.76 each). None of
HellaSwag, ARC-Easy, PIQA, OpenBookQA or WinoGrande is a code task, and the extra
maths share is plausibly mildly positive rather than negative.

Set against that, 60B is **9B more training tokens than 51B** — the lever the
operator picked deliberately, and the one with the better-established effect at
this scale.

**Recommendation: keep 60B.** The repetition is bounded at exactly the 4-epoch
guideline the plan cites (Muennighoff et al. 2023, arXiv 2305.16264), the skew
stays inside the gate's own limit, and its composition does not point at the
tasks we are judged on. 51B is the alternative worth *knowing about* — it saves
~$9.3 and 5.02 pts of skew for 9B fewer tokens — and it should be a stated option
at the gate rather than something discovered afterwards, but it is not the
recommendation.

The one thing that genuinely follows from the knee: **do not raise the budget
above 60B.** 60.39B is the refusal ceiling and the curve is climbing steeply
there, so there is no headroom left in this direction without more corpus.

## Reproduce

    python -c "import sys; sys.path.insert(0,'.'); \
      import importlib.util, train; \
      spec=importlib.util.spec_from_file_location('mm','scripts/mixture_margin.py'); \
      mm=importlib.util.module_from_spec(spec); sys.modules['mm']=mm; \
      spec.loader.exec_module(mm); \
      c=mm.read_corpus('data/shards'); c['fineweb-edu']=5_625_000_000; \
      print(train.mixture_preflight(mm._tree(c,0.02), 60_000_000_000))"
