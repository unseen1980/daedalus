# `hero`'s schedule, computed against its real configuration

> **Superseded for the headline numbers, 2026-08-10.** The operator raised
> `hero` to **60B**, where the schedule is **124,684 steps with decay from
> 68,576** — measured at 60B specifically rather than scaled from here, in
> `runs/preflight/mixture-at-60b.md` and pinned by
> `test_the_schedule_still_anneals_at_the_60b_budget`. This document remains the
> record for the 40B option (still on the menu as `go 40B`) and its method is
> unchanged: the point below is that the estimator matches a simulation of the
> actual loop, and that property was re-verified at 60B.

Evidence for the `[ASK HUMAN] ready for hero` issue. Config: `daedalus-150m`,
40B tokens, `micro_batch=16`, `warmup_steps=300`, `decay_frac=0.45`,
seq ramp 1024→2048, batch ramp 128,000→512,000 tokens over the first 10%.

| quantity | value |
|---|---|
| `estimate_total_steps` | **83,123** |
| steps the loop actually takes (simulated) | **83,123** (off by 0) |
| milestone / decay-start step | **45,717** = 55.0% of the run |

LR multiplier through the run:

| point | step | multiplier |
|---|---|---|
| start | 0 | 0.000000 |
| end of warmup | 300 | 1.000000 |
| decay start (milestone) | 45,717 | 1.000000 |
| 90% of the run | 74,810 | 0.222237 |
| last real step | 83,122 | 0.000027 |
| estimated end | 83,123 | 0.000000 |

## Why this is worth pinning

`estimate_total_steps` is the denominator `wsd_lr` decays over *and* the input
to `decay_start_step`, so a wrong value rescales the entire schedule and moves
the branch point. The earlier WSD bug — a schedule that never actually
annealed — would have cost four days and ~$44. These three numbers are now
asserted against hero's exact configuration in `tests/test_hero.py`
(`test_hero_step_estimate_is_exact`, `test_hero_lr_actually_decays_to_zero_at_the_end`,
`test_hero_milestone_lands_at_the_documented_55_percent`), so a change to the
ramp cannot silently move them.

The 55.0% figure is the one quoted to the operator in `STATUS.md`, `README.md`
and the model card as the stable-phase branch point. It is now checked rather
than asserted.

## What this does not prove

Throughput, memory, or that 83,123 steps of real data behave well. It is a
statement about the schedule only, computed in 0.1 s of pure Python.
