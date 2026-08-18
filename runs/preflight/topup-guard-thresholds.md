# Picking the in-window top-up's abort thresholds from a real control trace

Measured 2026-08-10 17:20Z, before arming `scripts/topup_beside_arm2.py`.

The supervisor kills the top-up if `abl-arch` arm 2's throughput holds below
a fraction of baseline for N consecutive polls. Picking those two numbers by
intuition is how you get a guard that either never fires or fires on noise, so
they come from replaying the **exact algorithm** over `abl-arch` arm 1's real
519-row trace -- a dataprep-free control for the same job on the same box.

## The trace is noisier than 'steady state' suggests

Baseline (median of the first 15 rows, as the supervisor takes it): **124,257 tok/s**.
Steady state is reached in under a minute -- the batch ramp does not depress
throughput -- but there is one substantial transient:

| row | elapsed h | tok/s | frac of baseline | hub_pending |
|---|---:|---:|---:|---:|
| 0 | 0.007 | 98,077 | 0.789 | 1 |
| 55 | 0.517 | 95,321 | 0.767 | 0 |
| 56 | 0.537 | 80,801 | 0.650 | 0 |
| 57 | 0.559 | 80,204 | 0.645 | 0 |
| 58 | 0.581 | 79,908 | 0.643 | 0 |
| 59 | 0.604 | 79,867 | 0.643 | 0 |
| 60 | 0.626 | 80,268 | 0.646 | 0 |
| 61 | 0.647 | 81,186 | 0.653 | 0 |
| 62 | 0.670 | 84,759 | 0.682 | 0 |

A single contiguous ~9-minute episode at **~65% of baseline**, ~31 min in, with
**no dataprep running**. `hub_pending` is 0 throughout so it is not the
uploader; it lines up with the first 30-minute checkpoint write. Any guard
tight enough to catch a real regression will also see this, so the
discriminator has to be **duration**, not depth.

## Replaying the algorithm

Poll every 120 s, median of the last 8 metric rows, N consecutive breaches to
abort. Against the real trace it must stay silent; against a synthetic
sustained regression it must fire promptly.

| floor | breaches | real trace (must be silent) | sustained -15% | sustained -10% |
|---:|---:|---|---|---|
| 0.92 | 3 | **FALSE FIRE** at 0.67 h | (false fire first) | (false fire first) |
| 0.92 | 5 | **FALSE FIRE** at 0.73 h | (false fire first) | (false fire first) |
| 0.92 | 8 | silent | fires 20 min after onset | fires 20 min after onset |
| 0.92 | 10 | silent | fires 24 min after onset | fires 24 min after onset |
| 0.90 | 3 | **FALSE FIRE** at 0.67 h | (false fire first) | (false fire first) |
| 0.90 | 5 | **FALSE FIRE** at 0.73 h | (false fire first) | (false fire first) |
| 0.90 | 8 | silent | fires 20 min after onset | fires 514 min after onset |
| 0.90 | 10 | silent | fires 24 min after onset | fires 518 min after onset |
| 0.85 | 3 | **FALSE FIRE** at 0.70 h | (false fire first) | (false fire first) |
| 0.85 | 5 | silent | fires 124 min after onset | missed |
| 0.85 | 8 | silent | fires 514 min after onset | missed |
| 0.85 | 10 | silent | fires 518 min after onset | missed |

## Chosen: floor **0.92**, breaches **8**

The only row that is silent on the control *and* notices a sustained 10-15%
regression within ~20 minutes. `0.85 + 8` is also silent but takes ~8.5 hours
to notice a 15% regression, which is a guard in name only. `0.92 + 3` -- the
value this was first written with -- false-fires on the control at 0.67 h.

Eight consecutive polls is ~16 min of sustained degradation, comfortably
longer than the 9-minute transient, and ~$0.12 of arm 2's time to establish.
The asymmetry still holds: a false abort costs only the ~5 h saving and falls
back to the plan that already existed, while a missed regression costs arm 2.
