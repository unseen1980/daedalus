## Correction to my own default: on no reply I will launch **59.9B**, not 58B

Twenty minutes ago I said that with no reply by ~10:21Z I would launch 58B. **That
was the wrong default and I am changing it.** You gave an explicit, considered
decision — *"Operator decision: 59.9B … Approved — start hero at 59.9B"* — and
defaulting to 58B would substitute my judgement for yours on a call that is
yours to make, after you had already accepted the risk once.

**So: no reply by ~10:21Z → I launch at 59.9B, as you instructed.**

The concern stands exactly as posted and is on the record in `STATUS.md` and in
the run's own notes — 9.96 pts of mixture skew against a 10.0 limit, for +3.3%
tokens. I still think 58B is the better run for the benchmark objective. But
"my read" is a judgement call with real uncertainty: I have no measurement of
how 9.96 pts of skew moves the 5-task mean against what +1.9B tokens buys, and
saying so plainly matters more than winning the point.

**Reply `go 58B` at any time before ~10:21Z and I launch 58B instead.** After
that the schedule is fixed — WSD total steps are computed from the token budget,
so the budget cannot be changed mid-run without invalidating the decay schedule.

### Everything is verified and launch is mechanical

Run against the real split at both budgets, using `hero`'s own gate rather than
the analysis wrapper:

| | 58B | **59.9B** |
|---|---|---|
| `hero.check_mixture` | ok, skew 4.94 | **ok, skew 9.96** |
| max epochs seen | 4.00 (at cap) | 4.00 (at cap) |
| total steps | 120,528 | **124,476** |
| milestone / branch point | 66,290 | **68,461** (55.00%) |
| final lr multiplier | 1.84e-5 | **1.79e-5** |
| QAT window | last 2.90B | **last 3.00B** |

Preflight gates all pass right now: GPU free (1 MiB used, 0%), QAT evidence
present (`runs/preflight/qat-gate-evidence.md`, 07:39Z), holdout split already
materialised at `data/shards-hero-split/` so no re-split at launch.

Suite **1503 passed, 1 skipped**.
