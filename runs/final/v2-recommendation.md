# What a from-scratch Daedalus V2 should do differently

Read `runs/final/improvement-report.json` for the numbers; this document says
what to do about them. Every recommendation below names the evidence behind it
and, where the evidence is thin, says so instead of rounding up.

> **Where the pieces live.** Phase 9's outputs are split across the two source
> branches, because the report is assembled from evidence that is itself split.
> This document and the report generator are on
> `vast/daedalus-improvements-20260824`; `improvement-report.json`, its markdown
> rendering and `daedalus-code-next.md` are on `vast/daedalus-code-20260824`,
> which is the tip of the stack and the only branch carrying both the Phase 3-7
> verdicts and the Phase 8 ones. Paths under `runs/code-probes/` cited below
> resolve on the code branch or on the merged stack, not on this branch alone.

**One framing before the list.** Four phases here were scoped to find gains in
the tokenizer, the optimizer schedule, the shape and the mixture. Three returned
"no recommendation" and the fourth bought 0.08% against a 0.5% bar. That is not
four wasted phases — it is a fairly strong statement that *the knobs this
program was built to turn are not where the remaining headroom is*. The one
result that did land, and that nobody planned for, is in §6.

---

## 1. Adopt the 32,768 vocabulary — the one unambiguous change

**Evidence:** `runs/tokenizer-lab/verdict.json`. 32,768 cleared every clause of
a rule fixed before any number existed: full byte round-trip, no domain fertility
regression (worst domain −0.72%), code fertility −3.04%, tiny-model BPB
regression 0.038% against a 0.5% bar, and 33.3% fewer embedding bytes under
Q6_K.

**Do:** train V2 on the 32,768 vocabulary in
`data/tokenizer-lab/tokenizers/v32768/`.

**Confidence:** high for the byte accounting, moderate for the quality half. The
10.3 MB it saves out of a ~101 MB Q4_0 artifact is arithmetic and holds at any
shape. The BPB comparison is equal-compute tiny models and ranks vocabularies
without predicting what a 150M-or-larger V2 scores.

**Also fix while you are here:** the shipped SmolLM2 vocabulary is missing 21 of
the 256 byte-level characters and cannot round-trip U+40000–U+FFFFF
(`runs/tokenizer-lab/addendum.json`). Every candidate was held to a round-trip
precondition the incumbent fails. This is unfixable in V1 — a vocabulary cannot
be swapped into trained weights — which is most of the argument for V2 existing.

---

## 2. Do not copy a conv decay schedule from this program. Change the question.

**Evidence:** `runs/conv-health/verdict-paired.json`,
`runs/conv-health/phase5-conv-decay.md`. At the shipped 150M shape over 500M
tokens the shipped 0.1 decay left **53.86%** of ShortConv channels dead. No
tested schedule cleared the preregistered rule; the best dead fraction any arm
reached was 14.52% against a 1% bar, and it bought that by growing `out_proj` to
2.33× the alive-channel baseline against a 2× limit.

**The result that matters is not the death — it is the ablation.** Removing
every channel the control flagged cost **2.98e-08 nats** of held-out loss. Those
channels were carrying nothing. So the opportunity is *parameters that were paid
for and not used*, not *quality that was lost*, and the two imply different
work.

**Do:**

- Treat this as capacity allocation. The cheapest V2 action is not a better
  decay schedule; it is a **narrower conv width**, spending the freed parameters
  where an ablation shows they earn their place.
- If a schedule is pursued anyway, it must clear the dead-fraction bar **and**
  the norm bar at a decay clock at least as long as the escalation's. Muon
  decays per optimizer step, so what a channel experiences is `sum(lr_t) * wd`,
  not tokens: the same arm read 1.61× at the short screen and 2.33× at the long
  one. A screen-length sweep will pass arms that a real run fails.
- Any claimed gain from reviving channels has to be shown as a held-out
  improvement, never as a higher alive count.

**Confidence:** high that the shipped schedule kills channels and that the dead
ones are inert; both were measured with a positive control and a matched
ablation. No confidence at all in any particular replacement schedule — none was
found.

**Not established:** whether a narrower conv width actually recovers the
parameters usefully. That was never run.

---

## 3. The KV-cache ceiling binds, and nothing this program measured can pick the shape

**Evidence:** `runs/architecture/stageb-recommendation.json`. No shape cleared
every preregistered column, so Phase 6 recommends none — including the control.

Two findings survive that verdict:

- **The shipped 18×768 / 6-attention / 4-KV configuration costs 8,192 KV bytes
  per context token, over the plan's own 6,144 ceiling.** This is arithmetic,
  transfers to any scale, and binds harder as the model grows. It is the most
  reliable output of the phase.
- **Every shape that bought a smaller KV cache paid for it in retrieval.**
  `a6-kv4` at 6,144 bytes lost 20 points of passkey at d1024; `a4-kv4` at 4,096
  lost 42 at d256; `a3-kv4` at 3,072 lost 38. Meanwhile held-out BPB moved by at
  most 0.26% across all four finalists. **Attention-layer count is nearly free in
  BPB and expensive in retrieval**, which means a V2 shape search scored on
  perplexity alone will pick a shape that cannot retrieve.

**Do:** treat KV-bytes-versus-retrieval as *the* architecture axis for V2, and
score every candidate on retrieval at depth. Do not select on BPB.

**Do not:** copy `a6-kv4` because it is the cheapest arm that passed BPB and KV.
It failed retrieval by 20 points.

**Confidence:** high on the KV arithmetic and on the direction of the trade.
Low on the ranking — 159M-parameter proxies over 252M tokens do not extrapolate
cleanly, and the phase says so itself.

**Fix the instrument first.** MQAR scored 0.000 for the control at depths 1024
and 2048, so 2 of 8 task/depth cells could not carry a 2-point gate at all. A
V2 sweep needs either more items per cell or a depth ladder the proxy scale can
actually resolve; otherwise "retention demonstrated" will keep meaning "we could
not have detected a loss".

---

## 4. Supply is the binding data constraint, not mixture weights

**Evidence:** `runs/corpus/mixture-verdict-probe.json`,
`runs/corpus/phase7-gate.md`, `runs/corpus/headroom-curve.md`.

The mixture sweep is a clean negative: the best admissible re-weighting bought
**0.081%** of aggregate BPB against a preregistered 0.5% bar, so Phase 7 kept
the baseline weights. Re-weighting is not where the headroom is.

What is: **the released run's own 59.9B budget is outside the skew bound this
program preregistered.** At 59.9B the corpus delivers 11.46 points of L1 skew
against a 5-point bound — seven of ten sources pinned at four epochs, the
blueprint not actually sampled. The corpus delivers the blueprint inside the
bound only to ~55.4B, or ~56.9B with the dialogue source dropped.

**Do, in order:**

1. **Drop `everyday-conversations` from general pretraining.** Below ~53B it is
   the *entire* skew: 403,573 tokens cannot fund a 2% share, so 2 points leave
   and 2 points water-fill elsewhere. Removing it takes skew to exactly 0.0000 at
   both 30B and 50B. Reserve dialogue for SFT.
2. **Top up supply before raising the budget.** Dropping dialogue moves the
   ceiling ~1.5B and buys 0.84 of the 6.46 points needed at 59.9B. Past ~53B the
   binding constraint is supply — `stack-edu-python` runs out of documents —
   and no reweighting fixes that.
3. **Rebuild against the frozen index.** The as-built corpus was indexed against
   the wrong splits for ARC-Easy and OpenBookQA and truncated at 2,000 items per
   task. The frozen index (`runs/corpus/decontam-index.json`, 1,371,773 n-grams,
   all five scored tasks at their scored splits, no limit) closes both. Two
   documents in the as-built corpus hit exactly those gaps.
4. **Require `source_provenance` on every manifest.** All ten as-built manifests
   carry no source revision, no filters block and no builder sha. The rebuild
   smoke's manifest carries all of it.

**Confidence:** high — these are properties of files, read rather than modelled.

**Caveat that must travel with any "clean" claim:** contamination is decided
from a 1.32% sampled scan. A hit is decisive; a zero bounds the document rate at
3.2e-05, it does not prove absence. And 158M `fineweb-edu` tokens — 3.0% of the
largest source — were covered by a clean verdict that never read them.

---

## 5. QAT belongs in pretraining, not in recovery

**Evidence:** `runs/qat-recovery/verdict.json` and the three scored arms.

All three recovery probes **eliminated the Q4_0 penalty outright** — the best
went from +5.539% to −0.373%, and the shipping artifact's absolute perplexity
improved 3.93% (6.9798 → 6.7054). All three were rejected, because closing the
gap cost 1.77% of FP16 perplexity and 7 points of passkey at d2048.

Read that carefully: **QAT worked, and 100M tokens of it was enough.** What
failed was doing it *afterwards*, where the only way to close the gap is to move
FP16 down toward the Q4 lattice.

**Do:** enable exact-grid QAT during V2 pretraining rather than as a recovery
pass. The machinery exists and is validated against real llama.cpp Q4_0 and
Q6_K grids (`daedalus/qat.py`); the plan's own `qat_frac` was designed for a
tail fraction of a full run. A model that has never been anywhere else does not
have to be dragged to the lattice.

**Confidence:** high that post-hoc recovery trades FP16 and retrieval for the
gap — three arms, monotone in learning rate, one seed each. Untested that
in-pretraining QAT avoids the trade; it is the obvious hypothesis, not a result.

---

## 6. The finding nobody planned for: the released checkpoint is hard to continue

**Evidence:** `runs/qat-recovery/scored/qat-recovery-lr0.001.json` and
`runs/code-probes/branch-1b-stop.json`.

Phase 3's QAT recovery and Phase 8's code branch share nothing but their
starting weights — different data, different objective, 100M tokens against 1B,
independently preregistered gates. Each cleared its own progress criterion
outright. **Both then failed retention in the same cell: passkey at 2048
tokens, by 7.0 and 8.0 points**, the code branch at paired McNemar p=0.013 with
8 of 8 discordant items moving against it.

Two unrelated treatments breaking the same capability in the same place points
at the checkpoint rather than at either treatment. The working hypothesis is
that the released model's deepest retrieval sits in a narrow basin that ordinary
continued training leaves.

**Why this outranks everything above it.** Phases 4–7 each ask "what should V2
be built *from*". This one says something about whether a V2 can be built *on*
at all — and it was found by two phases that were not looking for it.

**Do:**

- **Instrument retrieval at depth throughout V2 pretraining**, not only at the
  end. If the basin is narrow, the training curve will show when it narrows, and
  that is a thing a schedule can respond to.
- **Treat continuability as a design property.** A model that cannot be
  fine-tuned without losing long-context retrieval is much less useful than its
  benchmark scores suggest, and no general benchmark in this program's suite
  detects it: the five-task mean *rose* on the QAT arm that lost 7 points of
  passkey.
- **Run the isolating arm this program did not.** Continue two different
  checkpoints on identical data and measure d2048 on both. That separates "this
  checkpoint is fragile" from "continued pretraining costs deep retrieval in
  general", and the two have completely different implications.

**Confidence:** this is a hypothesis with two supporting observations, not an
established mechanism. It was never isolated. It is reported first among the V2
recommendations because of its consequence if true, not because of its evidential
weight — and the arm that would settle it is one run.

---

## What this program did not establish

Stated plainly, so none of it is quoted as if it were:

- **No proxy result here is a measurement of the released 150M model.** Phases
  4–7 ran at 105M and 159M parameters over 101M–500M tokens. They rank
  decisions. The report's own validator refuses to let them be recorded as model
  gains.
- **No architecture shape is recommended**, including the shipped one.
- **No conv decay schedule is recommended.**
- **No V2 quality projection is offered.** Nothing here supports a number for
  what a 59.9B-token V2 would score, and one would have to come from a scaling
  study this program did not run.
- **Apple Silicon decode is unmeasured.** Every decode figure is this box's CPU,
  which fixes the shape of the curve and not the number a user feels.
- **The 32,768 vocabulary's quality benefit is a tiny-model result.** Its byte
  savings are not.

---

## The shortest version

If exactly one thing is carried into V2, carry §6 — run the isolating arm and
instrument retrieval at depth. If three: §6, the 32,768 vocabulary (§1), and
QAT-during-pretraining (§5). Those are the three findings whose evidence is
strong enough and whose cost is low enough to act on now.
