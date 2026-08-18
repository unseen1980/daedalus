# 60B breaks the epoch cap, and the alert that should say so reads *perfect*

The operator raised `hero` from 40B to 60B tokens and asked, specifically, for
the exact per-source epoch counts before launch, with any source landing far
above the rest flagged.

One does. The finding is not "60B is a bit repetitive" — it is that **at 60B the
repetition guard switches itself off**, and the graded signal that exists to
catch exactly this reports 0.00 pts of skew, its best possible value.

## The arithmetic

`cap_weights_by_epochs` (`train.py:373`) clamps every source to
`4 x tokens_on_disk / total_run_tokens`. The corpus is **14,217,552,718 tokens**,
so the most any budget can draw at ≤4 epochs/source is

```
4 x 14.218B = 56.87B
```

**60B is above that line.** When no allocation can satisfy every cap, the
function returns the target shares **unchanged** with a printed warning
(`train.py:428-437`). That is a deliberate choice and defensible on its own
terms — uniform over-repetition degrades more gracefully than renormalising to
whatever happens to be on disk — but it means the epoch limit is not applied at
all.

## What that does to the mixture

Run against the real manifests:

| budget | L1 skew | >10 pts alert | cap active? | `everyday-conversations` epochs |
|---|---|---|---|---|
| 40B | 3.99 | quiet | yes | 4.0 |
| 42B | 9.71 | quiet | yes | 4.0 |
| 44B | 14.90 | **fires** | yes | 4.0 |
| 50B | 29.91 | **fires** | yes | 4.0 |
| 56B | 42.55 | **fires** | yes | 4.0 |
| **57B+** | **0.00** | **quiet** | **no** | **2,825** |
| **60B** | **0.00** | **quiet** | **no** | **2,973** |

Two things to take from that table.

**The one source far above the rest is `everyday-conversations`, by three orders
of magnitude.** At 60B it takes its full 2% share — 1.2B tokens — from a dataset
that is exhausted at **403,573 tokens**. That is ~2,200 conversations repeated
**2,973 times**, as 2% of the entire run. At 40B the cap pins it to ~0.00% and
the deviation is recorded and accepted. At 60B it comes back at full strength.
Every other source sits between 1.33 and 6.00 epochs.

**The alert is non-monotonic, and 60B is on the quiet side of the cliff.** At
50B `l1_skew_pts` is 29.91 and `MAX_MIXTURE_SKEW_PTS` correctly fires. At 60B it
is 0.00 — because the targets were returned untouched — so the dashboard number
whose entire job is to say "the corpus is not delivering the blueprint mixture"
reads as a flawless mixture at the one budget where repetition is unbounded.
`cap_weights_by_epochs` does print a loud WARNING, so this is not silent in the
log; but the *graded* signal, the one on the phone, says everything is fine.

Per-source at 60B as things stand:

| source | target | 40B effective | 60B effective | 60B epochs |
|---|---|---|---|---|
| fineweb-edu | 37.50% | 37.50% | 37.50% | 6.00 |
| dclm-baseline | 22.50% | 22.50% | 22.50% | 6.00 |
| stack-edu-python | 9.00% | 9.47% | 9.00% | 4.46 |
| finepdfs-edu | 8.00% | 8.42% | 8.00% | 5.45 |
| finephrase | 7.00% | 7.37% | 7.00% | 2.03 |
| cosmopedia-v2 | 5.00% | 5.26% | 5.00% | 3.16 |
| finemath-3plus | 3.00% | 3.16% | 3.00% | 1.33 |
| infiwebmath-3plus | 3.00% | 3.16% | 3.00% | 1.33 |
| finewiki-en | 3.00% | 3.16% | 3.00% | 4.39 |
| **everyday-conversations** | 2.00% | **0.00%** | **2.00%** | **2,973** |

The operator's stated rationale — "~4.3 epochs over the 13.94B corpus" — is
right *on average* (60/14.218 = 4.22) and that average is the problem: 4.22 > 4.0
is precisely what pushes the allocation past the point where the cap can be met.

## The fix is cheap: ~3.5B more tokens, ~4.2 h, ~$1.89

Not "drop back to 40B". A top-up restores the guard, and it is small because
only the largest sources are short:

| source | needs (share x 60B / 4) | on disk | short |
|---|---|---|---|
| fineweb-edu | 5.62B | 3.75B | **1.87B** |
| dclm-baseline | 3.38B | 2.25B | **1.12B** |
| finepdfs-edu | 1.20B | 0.88B | **0.32B** |
| stack-edu-python | 1.35B | 1.21B | **0.14B** |
| finewiki-en | 0.45B | 0.41B | **0.04B** |
| everyday-conversations | 0.30B | 0.0004B | impossible — dataset exhausted |
| **buildable total** | | | **3.50B** |

At the measured build rate — **0.83B tokens/h** wall-clock across the whole
14.218B corpus, i.e. **$0.54/B** at $0.449/h — that is **~4.2 h and ~$1.89**.

Verified rather than argued: with those five sources topped up and
`everyday-conversations` left as it is, `cap_weights_by_epochs` at 60B

- does **not** hit the all-capped fallback,
- gives **L1 skew 3.99 pts** — identical to what 40B has today,
- puts every source at **≤4.00 epochs**,
- and caps `everyday-conversations` to 0.00% again, the already-recorded
  deviation and the entire content of those 3.99 pts.

So for ~$1.89 the operator's 60B decision becomes exactly as clean as 40B is
today, instead of training 1.2B tokens of the same 2,200 conversations.

`everyday-conversations` cannot be topped up — the source is 2.2k rows total —
which is why capping it, as the approved §4.4 plan already called for, is the
other half of this and is what keeps it from forcing the fallback.

## Timing

The top-up needs no GPU. `abl-arch` runs until ~06:00Z, then the
`[ASK HUMAN] ready for hero` gate. **Running it during the gate wait costs
nothing beyond the box time already being spent waiting** — it is the one slot
where 4.2 h of CPU work is free. It must not run concurrently with `abl-arch`
under ADDENDUM 2: the trainer holds ~9.5 GB RSS, and four dataprep workers at
4 GB each would put the total at ~25.5 GB against the 20 GB ceiling.

## The rest of the 60B checklist

- **WSD schedule at 60B, measured at 60B** (not assumed from the 40B fix):
  124,684 steps, decay from **68,576 = 55.000%**, lr multiplier **1.000000** at
  the branch point and **1.78e-5** on the final step. Pinned by
  `test_the_schedule_still_anneals_at_the_60b_budget` and added to the
  documented-schedules parametrization.
- **QAT window**: `qat_active_at(self._progress(), qat_frac)` (`train.py:1023`)
  is a fraction of progress, not an absolute token count, so the last 5% is the
  last 3.00B tokens at 60B automatically.
- **Milestone step**: `decay_start_step(self.total_steps, args.decay_frac)`
  (`train.py:828`), also derived — it fires at step 68,576, not a hardcoded one.
- `hero.py --total-tokens` now defaults to 60B.

## Update 2026-08-10 12:55Z — the corpus top-up is now backed by an alarm

The plan above depends on the $1.89 top-up actually happening before `hero`
launches. If it is skipped, forgotten, or only partly completes, the run trains
on the wrecked mixture — and **every graded signal reads perfect while it does.**

That is not a figure of speech. `l1_skew_pts` is the total absolute deviation
from the target mixture, and in the all-capped fallback `cap_weights_by_epochs`
returns the target shares *unchanged*, so the deviation is **0.00 — its best
possible value — at exactly the budget where repetition is bounded by nothing.**
Measured on the real manifests just now:

| budget | `l1_skew_pts` | skew alarm | worst source | `max_epochs_seen` | repetition alarm |
|---|---|---|---|---|---|
| 40B | 3.99 | quiet | `dclm-baseline` | 4.0 | quiet |
| **60B** | **0.00** | **quiet** | `everyday-conversations` | **2,973.4** | **FIRES** |

So the mixture has two failure modes and only one was graded:

- **the cap binds and reweights** — the web backbone collapses, skew rises,
  `MAX_MIXTURE_SKEW_PTS` catches it. This was covered.
- **no allocation can satisfy the cap** — nothing is reweighted, nothing is
  bounded, and the skew metric is silent by construction. This was not.

`mixture_summary()` now reports `max_epochs_seen` and `most_repeated_source`
into the W&B run config, and `train.py` grades them. The rule needs no tuning
constant: when the cap works, a capped source is pinned at *exactly*
`max_epochs` and every other source sits below it, so `max_epochs_seen >
max_epochs` is true precisely in the unbounded case and never otherwise. The
40B column above is the no-false-positive half of that claim, measured rather
than argued.

`cap_weights_by_epochs` did already print a "Build more data" warning in this
case, so it was not literally silent. What was missing is the part the operator
actually sees: a graded signal alongside the skew check, and a field on the
dashboard that expresses repetition at all.

Tests: `test_unbounded_repetition_warns_where_the_skew_number_reads_perfect`,
`test_bounded_repetition_stays_quiet`,
`test_the_repetition_alarm_brackets_heros_real_budgets`.
