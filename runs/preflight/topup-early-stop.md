# What stopping the 60B top-up early would cost, and the gate it exposed

**2026-08-10 ~19:15Z**, while `abl-arch` arm 2 trains and the top-up runs beside
it. Written because the top-up shares a box with a 12.8 h training arm and sits
on `hero`'s critical path, so "can we cut it short?" is a question that will be
asked under time pressure. The answer is **no, and the intuitive early-stop
point is the worst possible place to stop.**

## The question

Issue #5: at 60B the corpus (14.218B then, 14.683B now) is too small for
`cap_weights_by_epochs` to satisfy the 4-epoch cap on every source, so it hits
its all-capped fallback and returns the target shares **unchanged** — bounding
repetition not at all. `everyday-conversations`, exhausted at 403,573 tokens,
would be repeated **2,973x** as 2% of the run.

The running top-up adds 3.03B more tokens over ~5.4 h to reach **17.717B**. Does
it have to run to completion?

## The measured curve

Along the top-up's actual per-source fill order, at the real 60B budget, using
`train.cap_weights_by_epochs` and `train.summarize_mixture` rather than a
restatement of them:

|   corpus |   +h | l1_skew | worst_ep | fineweb% |
|---------:|-----:|--------:|---------:|---------:|
|  14.683B |  0.0 |    0.00 | 2973.44x |   37.50% |
|  15.008B |  0.6 |   40.11 |    4.00x |   28.43% |
|  15.483B |  1.4 |   33.78 |    4.00x |   31.59% |
|  15.883B |  2.1 |   28.44 |    4.00x |   34.26% |
|  16.283B |  2.9 |   23.11 |    4.00x |   36.93% |
|  16.669B |  3.5 |   17.96 |    4.00x |   37.50% |
|  17.069B |  4.3 |   12.63 |    4.00x |   37.50% |
|  17.268B |  4.6 |    9.98 |    4.00x |   37.50% |
|  17.717B |  5.4 |    3.99 |    4.00x |   37.50% |

Three points matter:

- **The cap engages after only +0.6 h**, at 15.008B — the necessary condition is
  just `corpus > 60B / 4 = 15.0B`, and today's 14.683B misses it by 317M.
- **The mixture is at its *worst* right there.** Skew **40.11 pts** against a
  10-pt limit: `fineweb-edu` drawn at **28.43%** against a 37.50% target,
  because every short source is pinned at its cap and the freed mass has
  nowhere good to go.
- **Skew only falls under the 10-pt limit at 17.268B, +4.6 h** — 85% of the way
  through. The last 0.45B buys 9.98 → 3.99 pts.

So there is no useful early stop. Between +0.6 h and +4.6 h the corpus is in a
regime where the dangerous failure mode is gone but the mixture is materially
wrong, and the operator's own audit ruling was that **mixture balance matters
more to benchmark scores than corpus size**.

## The trap this sets for anyone reading a dashboard

`l1_skew_pts` reads **0.00 today — its best possible value — and 40.11 after the
first 36 minutes of top-up.** The number gets ten times worse while the corpus
gets strictly better, because 0.00 is what the all-capped fallback produces by
construction. Anyone watching that metric would conclude the top-up broke the
mixture. It did the opposite.

This is the same inversion issue #5 documented, seen from the other side, and it
is why `max_epochs_seen` is reported next to the skew rather than instead of it.
Neither number is sufficient alone:

| | `l1_skew_pts` | `max_epochs_seen` |
|---|---|---|
| unbounded repetition (14.683B) | **0.00 — looks perfect** | 2973x — fires |
| bounded but skewed (15.0–17.3B) | 40.11 — fires | 4.00x — looks perfect |
| healthy (17.717B) | 3.99 | 4.00x |

## The gate this exposed, and the fix

`hero.py` had **no mixture preflight at all**. `train.py` prints a warning for
both modes (`train.py:887`, and the UNBOUNDED warning inside
`cap_weights_by_epochs`) and then trains anyway — right for `sweep` and
`abl-arch`, which are hours long and cheap to redo, wrong for a **$61.89,
~5.9-day** run whose only reader watches a phone. A line printed once at step 0
of 124,684 is indistinguishable from silence.

Both regimes above would have launched. The 15.0B one would have launched
*looking healthier* on the graded metric than the corpus it replaced.

`hero.check_mixture()` now gates on both modes and `_cli` refuses with **rc 2**
before the watchdog or the trainer starts, writing the reason to `STATUS.md`.
`--allow-skewed-mixture` overrides it and records that choice in `STATUS.md`, so
a knowingly-bad run is still traceable in the writeup.

The preflight reads one `manifest.json` per source and nothing else — no GPU, no
memmaps — so it costs nothing to run at launch.

### Why it shares code with the loader rather than reimplementing it

A preflight that computes the mixture differently from the thing that samples it
is worse than none: `hero` would gate on one answer and train on another.
`MixtureBatchSource.__init__` and `mixture_summary()` were factored into
module-level `resolve_mixture()` / `summarize_mixture()`, and
`mixture_preflight()` is the same two calls without the samplers.
`test_the_preflight_agrees_with_the_loader_it_is_standing_in_for` asserts the
two produce an identical dict on the same corpus.

## Tests

| test | pins |
|---|---|
| `test_check_mixture_catches_unbounded_repetition` | the dangerous mode, and that skew reads 0.00 there |
| `test_check_mixture_catches_a_bounded_but_skewed_mixture` | the mode the epoch check cannot see |
| `test_check_mixture_passes_a_corpus_that_can_deliver_the_mixture` | it is not simply always-refuse |
| `test_hero_refuses_to_launch_and_starts_nothing` | rc 2, **and** neither watchdog nor trainer was started first |
| `test_allow_skewed_mixture_launches_and_records_the_choice` | the override leaves a trace |
| `test_the_preflight_agrees_with_the_loader_it_is_standing_in_for` | anti-drift |
| `test_hero_gates_at_its_own_default_budget_not_a_test_sized_one` | 60B/4 = 15.0B; 14.218B fails, 17.717B clears |

`tests/test_hero.py`'s `_no_split` helper now builds a real corpus rather than
naming a directory that does not exist — a preflight a test can skip by pointing
at nothing is not a gate.

## The gate runs on the train split, not the whole corpus — and the margin is 1.8 pts

`_cli` calls the preflight **after** the holdout carve, on `args.data_dir`,
because that is what `train.py` will actually sample. A 2% holdout removes 2%
from every source, so the number the gate sees is smaller than the corpus:

| | on disk | skew | worst | verdict |
|---|---|---|---|---|
| whole corpus at the top-up target | 17.717B | 3.99 | 4.00x | passes |
| train split (`holdout_frac=0.02`) | **17.362B** | **7.19** | 4.00x | **passes** |

It passes, but the skew is 7.19 against a 10.0 limit rather than 3.99. Bisecting
for where it stops passing: the split must retain **≥96.2%** of the top-up
target (17.052B whole-corpus equivalent). The carve takes 2.0%, so the margin is
**1.8 percentage points** — i.e. the top-up may finish up to ~3.8% short of
17.717B and still clear the gate, and no further.

That is thin enough to be worth knowing in advance rather than discovering as a
refusal at 06:45Z. It is not worth loosening the limit for: 10.0 pts was chosen
to be quiet at every budget the corpus was built for and loud at the first one
that breaks it, and a gate tuned to whatever the corpus happens to deliver is
not a gate.

## Bottom line

Let the top-up finish. It ends ~**00:30Z on the 11th**, well inside arm 2's
window (ends ~06:04Z) and ahead of the gate (~06:45Z), so it costs `hero`
nothing. If it is ever interrupted, the corpus must reach **≥17.27B** before
`hero` starts, and `hero` now enforces that itself instead of trusting anyone to
remember it.
