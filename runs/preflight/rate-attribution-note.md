## The rate question is now measured, not guessed — and it favours `go` (58B)

The gate asks you to read the hourly rate off the vast console because I could
not. That is still the ground truth and still worth ten seconds. But I found a
measurement I had overlooked, and it prices the bandwidth hypothesis rather than
just asserting it. **The answer is much less scary than the table in the post.**

### What I missed until now

`/proc/net/dev` counts bytes cumulatively **from container start**, so even
though nobody thought to snapshot it during the billing window, the container's
whole-life traffic was sitting there the entire time.

| | |
|---|---|
| container up since | 2026-08-08 20:54:06Z (**60.2 h**) |
| received | **358.8 GB** |
| transmitted | **52.2 GB** |
| **total** | **411.0 GB → 6.83 GB/h** |

That is a lot of traffic, and it lands exactly where the bandwidth explanation
predicted: dataprep streaming ten sources plus the 34 GB corpus and the first
checkpoints going up to the Hub.

### Why this settles the part that matters

**`hero` is not a bandwidth-heavy job.** It reads its shards from local disk —
the corpus is already here — and ships ~24 GB of checkpoints over 133.3 h:

> **0.18 GB/h, against the 6.83 GB/h this container has averaged — 37.9× less.**

So even if *all* $6.78 of unexplained spend in your window was metered traffic,
`hero` cannot inherit much of it. Sweeping the one number nobody recorded (how
many of those 411 GB fell inside your 30.92 h window):

| GB in the window | implied $/GB | `hero` traffic | `hero` effective $/hr | **`hero` total** |
|---|---|---|---|---|
| 400 | $0.0169 | $0.41 | $0.452 | **$60.26** |
| 300 | $0.0226 | $0.54 | $0.453 | **$60.39** |
| 200 | $0.0339 | $0.81 | $0.455 | **$60.67** |
| 100 | $0.0678 | $1.63 | $0.461 | **$61.48** |
| 50 | $0.1356 | $3.25 | $0.473 | **$63.11** |
| 25 | $0.2712 | $6.51 | $0.498 | **$66.36** |

Across a **16× range** of assumptions, `hero` at 58B costs **$60–$66** — not the
**$89.07** the flat-$0.668/hr reading implies. The worst case in that table is
~$6.50 over the $59.85 budgeted, not ~$29 over.

Reproduce: `python scripts/rate_attribution.py`. Pinned by
`tests/test_rate_attribution.py` (8 tests), including one asserting the
conclusion holds across the whole sweep rather than at a flattering point.

### Confirming it directly: an idle box moves almost nothing

The box has been idle since 08:27Z waiting on you, so the idle traffic rate is
measurable right now:

> **0.10 GB/h** over a 2.2-minute window — **67× below** the container average,
> and that figure is mostly my own git pushes and the 5-minute heartbeat commits.

Which is the point: the 6.83 GB/h average was **dataprep and the corpus upload**,
both finished and neither recurring. `hero`'s 0.18 GB/h is the same order as an
idle box.

### What this does NOT settle — read this before deciding

**It cannot prove the base rate is $0.449/hr.** If vast is simply charging
$0.668/hr for GPU+storage, then traffic is irrelevant and `hero` really does cost
~$89 and finish $2.56 short. Nothing readable from inside this container
distinguishes those two worlds — the balance is only readable via `vastai`, which
your standing instructions forbid outright.

What changed is the **prior**, and it changed a lot: there is now a measured,
sufficient, non-recurring explanation for the entire $6.78 gap, sitting in
exactly the window where the corpus went to the Hub.

### So the reply options, re-priced

| reply | at $0.449/hr base + measured traffic | if the base really is $0.668/hr |
|---|---|---|
| **`go`** (58B) | **≈$30.1–$30.6 left** | ≈−$2.56, does not finish |
| `go 51B` | ≈$37.4–$37.9 left | ≈$8.19 left |

**My recommendation is now `go` at 58B**, where before I hedged it on the console
reading. The bandwidth explanation is measured rather than assumed, it is
sufficient on its own, and its mechanism is spent. If you would rather not carry
the residual risk at all, `go 51B` still closes under both worlds and costs 7B
tokens of training.

**If you can glance at the console anyway, please still do** — and tell me the
rate *and* whether there is a separate bandwidth or storage line. That converts
this from a well-supported inference into a fact, and I will re-anchor
`COSTS.md` either way.

### Two related fixes, since they touch what you monitor

**1. `HEARTBEAT.md` has been reporting a job that was not running.** Job
detection used `pgrep -f "[t]rain\.py"`, which matches any command line
*containing* the string — and the agent runs as `claude -p '<prompt>'` with the
whole prompt as one argument. That prompt names `train.py` throughout, so every
tick taken while an agent iteration was in flight reported
`Background job: train` **on an idle box with the GPU at 0%**. It said exactly
that at 08:53Z.

That is not cosmetic. The `POSSIBLY STALLED` verdict is guarded by
`[ "$JOB" = "none" ]`, so the false positive **suppressed the stall warning** and
fell through to the reassuring *"Working … Expected during long jobs"*. During
`hero`, a trainer that died at hour 90 with an agent iteration running would have
shown yellow on your phone.

The same class was in the supervisor check: tmux's own argv is
`tmux new-session -d -s agent /workspace/agent_loop.sh`, so it keeps matching
after the loop dies — `Supervisor loop alive` would have read **yes** with no
agent running, hiding the one red verdict that matters.

Fixed by requiring an argv element to *end with* the script name plus a comm
exclusion, verified in both directions against real processes, with regression
tests. `heartbeat.sh` is also now in the repo — `/workspace` is not a volume
here, so the live copy would not survive a recycle.

**2. `HEARTBEAT.md` now publishes RX/TX every 5 minutes.** Nothing was recording
bytes, which is the only reason this question needed a sweep instead of a
subtraction. The heartbeat commits every tick, so from now on git history *is*
the traffic time series and any two heartbeats bound the traffic between them.
Next time a balance statement arrives, the traffic term can be subtracted
directly and the base rate falls out.
