# Steady-state throughput at seq 2048 — the number `hero`'s $43.70 is priced from

The hero gate projects 40B tokens from **115,692 tok/s**, measured by `smoke.py`
on a synthetic 5-step run *while `dataprep` was competing for CPU*. `abl-arch`
is now training the hero architecture, at the hero micro-batch, on a quiet box,
so the projection can be replaced with a production measurement instead of a
benchmark one.

## Why the live number could not simply be read off mid-run

At 06:00Z the run was showing ~125k tok/s, which would take $43.12 down to about
$40. That is not the same quantity. `ramp_frac` is 0.1, so `abl-arch` ramps
seq 1024→2048 and batch 128k→512k tokens over its first 500M tokens
(`train.py:59-81`), and `hero` spends ~90% of its life at the *end* of that ramp,
not in it.

Snapshot during the ramp:

| step | seq | batch tokens | tok/s |
|---|---|---|---|
| 80 | 1024 | 130k | 124,051 |
| 560 | 1152 | 188k | 125,447 |
| 1040 | 1408 | 271k | 125,120 |

Throughput is remarkably flat across seq 1024→1408 — expected, since only 6 of
the 18 blocks carry the quadratic term — but "flat so far" is not "flat to
2048".

`seq_len_schedule` snaps to a 128 grid, so seq reaches 2048 at
`p >= 0.9375` → **468.75M tokens**, while `batch_tokens_schedule` reaches 512k
only at **exactly 500M**. Steady state therefore begins at 500M, and because
`tok_per_sec` is windowed rather than cumulative (`train.py:708`), a clean read
needs several windows past that point — hence measuring at ≥560M.

## Prediction, written down before the measurement

Two effects pull in opposite directions from the ramp numbers above:

- **seq 1408 → 2048 (+45%) should cost a little.** Fitting the three ramp points
  gives no usable slope — they vary by under 1% and not even monotonically, so
  the quadratic term is buried in noise over that range. Bounding it instead: if
  attention were ~5% of step time at 1408 and its per-token cost scales with
  seq, the move to 2048 costs ~2%.
- **batch 271k → 512k tokens should help.** More grad-accum micro-steps per
  optimizer step amortises the Muon/AdamW update — including Muon's
  Newton–Schulz orthogonalisation — over more tokens.

Net: roughly flat, plausibly slightly up.

> **Predicted steady state: 120,000–126,000 tok/s.**
> Above the gate's 115,692 either way, because that figure was taken with
> `dataprep` on the CPU.

If it comes in *below* 115,692 the gate projection stands unchanged and the
reason gets investigated before `hero` launches — a throughput regression at the
exact shape `hero` runs at is worth more than the $3 the re-pricing would save.

## Measurement — 2026-08-10 06:25–06:32Z, quiet box

Six consecutive windows, every one at exactly 524,288 tokens/step (seq 2048,
batch 512k — the shape `hero` spends ~90% of its life in), no eval running, no
checkpoint save inside any of them:

| step | wall | tok/s |
|---|---|---|
| 1820 | 06:25:10 | 122,025 |
| 1840 | 06:26:36 | 121,731 |
| 1860 | 06:28:02 | 122,035 |
| 1880 | 06:29:28 | 122,037 |
| 1900 | 06:30:54 | 122,051 |
| 1920 | 06:32:20 | 122,084 |

> **Steady state: 121,994 tok/s** (sd 130, **0.11%** — six windows spanning 7
> minutes and 63M tokens).

**+5.4% above the 115,692 the gate is priced from**, and inside the predicted
120,000–126,000 band.

### Revised 06:50Z after 25 minutes rather than 7: the card is power-capped

The six windows above were the first six. Extending to **16 uninterrupted
windows over 25 minutes** shows the 0.11% was a real property of that stretch and
**not** the dispersion of the steady state:

| | tok/s |
|---|---|
| first 8 clean windows (06:25–06:36) | 122,016 |
| last 8 clean windows (06:39–06:50) | 123,209 |
| **all 16** | **122,612** (sd 989, **0.81%**, range 121,665–124,275) |

Window durations are not continuous but land in modes — **84.4 s**, **85.9 s**,
**86.2 s** — which is the signature of a clock change, not of load.

**Cause, checked rather than guessed:**

```
clocks.sm 2745 MHz / max 3105 MHz | power.draw 500.05 W / limit 500.00 W
temperature 71 C | throttle reasons active: 0x4  (SW Power Cap)
```

The 5090 is sitting **exactly on its 500 W cap** and boosting to whatever clock
that allows, 88% of its maximum SM clock. So throughput moves with the thermal
and power envelope, and the ±1% band is the hardware, not the measurement.

Two consequences I would rather state than let a reader infer:

- **A 25-minute sample cannot bound 92 hours of thermal behaviour.** It brackets
  the band the card runs in right now; ambient conditions over four days are not
  something this measurement speaks to.
- **The earlier "−2.3% from ramp to steady state" is partly confounded.** The
  ramp windows were taken 06:11–06:25 and the steady ones after, so a clock drift
  and a sequence-length effect are entangled in that number. The drift I can see
  runs *upward* over time, i.e. against the −2.3%, so the seq cost is real but is
  more likely **~1.3%** (124,271 vs the 123,209 of the later windows) than 2.3%.
  Either way it is small, and the projection below takes the slow end.

### What that 0.11% is, and what it is not

It is the dispersion of the **uninterrupted** windows, not the uncertainty on the
projection — and the two must not be conflated. Extending to 12 windows
(06:25–06:41Z) makes the point:

| | n | tok/s | sd |
|---|---|---|---|
| uninterrupted windows only (≤86.5 s) | 10 | **122,169** | 0.11% |
| **every window, interruptions included** | 12 | **121,326** | 1.69% |

The 1.69% is not noise in the compute; it is two identifiable events. Every clean
window takes **85.9 s**, a strikingly stable mode. The two that do not are step
1940 (**88.8 s**) — my own analysis commands for this file — and step 2000
(**90.3 s**), which is a `val_every_steps` boundary. Both are already priced as
explicit lines below, so quoting the all-in rate *and* the overhead lines would
double-count them.

**Which is a free cross-check on the projection**, since the two routes are
independent:

| method | wall clock | cost |
|---|---|---|
| clean rate + overheads priced individually | 91.88 h | **$41.26** |
| flat at the all-in achieved rate (121,326) | 91.58 h | $41.12 |

**0.3% apart.** The number carried into the gate is the higher one.

### The prediction was right on the bound and wrong on the direction

Predicted "roughly flat, plausibly slightly up". Measured **−2.3%**: clean ramp
windows (seq 1024→2048, batch ramping, steps 600–1800, excluding the eval and
checkpoint windows below) average **124,271 tok/s (n=48, sd 1,905)** against
121,994 at steady state.

So of the two effects reasoned about, only one showed up. The seq cost was
bounded at ~2% and came in at ~2.3% — close. The batch-amortisation gain
(271k → 512k tokens/step spreading Muon's Newton–Schulz over more tokens) did
**not** materialise as a visible offset. Not worth chasing at 2%, but recorded
because the reasoning predicted a partial cancellation that did not happen, and
the next projection should not assume it.

### Two overheads the clean windows exclude, now priced from measurement

Neither was in the gate's arithmetic. Both are visible as isolated dips:

- **Checkpoint save**: `ckpt_every_sec` is 1800, and the 1.435 GB
  full-optimizer-state save at **06:13:26** lands in the step-1660 window —
  113,822 tok/s against a ~125,300 neighbour baseline, i.e. **7.4 s lost**.
- **Periodic val**: `val_every_steps` 500, measured at steps 500/1000/1500 as
  **7.3 / 4.3 / 3.4 s** — falling as the batch grows, so 3.4 s is the
  steady-state figure.

And one I caused: **step 1940 dropped to 118,045 tok/s (+2.9 s)** while I ran the
analysis commands for this file. Same mechanism as the eval contention at steps
1140–1240 (~80k tok/s for five minutes, already costed at $0.05), two orders of
magnitude smaller. Worth pricing rather than waving away: at a 10-minute
check-in cadence over a 4-day run that is ~545 interruptions.

## `hero` projection, from measured numbers

| | h |
|---|---|
| ramp, 4B @ 124,271 | 8.94 |
| steady state, 36B @ 121,994 | 81.97 |
| 182 checkpoint saves × 7.4 s | 0.37 |
| 166 val passes × 3.4 s | 0.16 |
| ~545 agent check-ins × 2.9 s | 0.44 |
| **total** | **91.88** |

> **`hero` = 91.9 h = $41.26**, against the gate draft's 96.0 h / $43.12.
> **$1.86 cheaper**, and now measured on the production path at the exact shape
> rather than benchmarked on a synthetic 5-step run with `dataprep` on the CPU.

Swept across the whole observed power-capped band rather than quoted from one
mean, since that band is the hardware and not the measurement:

| basis | tok/s | h | cost |
|---|---|---|---|
| slowest clean window | 121,665 | 92.30 | **$41.44** |
| the 121,994 quoted above | 121,994 | 92.05 | $41.33 |
| 16-window mean | 122,612 | 91.59 | $41.12 |
| last-8 mean (warmed) | 123,209 | 91.15 | $40.93 |
| fastest clean window | 124,275 | 90.38 | $40.58 |

**The entire band is $40.58–$41.44 — a spread of $0.86 on a $41 ask.** The gate
carries **$41.26**, which sits in the upper half. The right way to read this is
that hero's cost is not sensitive to the throughput number at the precision this
argument needs; it is sensitive to whether the run completes without rework, which
is a different risk and a much larger one.

Sensitivity, so the headline is not read as more precise than it is. The compute
rate is tight (0.11% across uninterrupted windows) and an independent all-in
route lands 0.3% away, so the arithmetic is not where the risk is — **the risk is
whether the box stays quiet for four days**, which no seven-minute measurement can
speak to. Every defensible variant sits inside $0.40: the most conservative
(ignore the faster ramp, ignore every overhead, run all 40B at the clean steady
rate) is **91.08 h / $40.89**; the all-in route is $41.12; the quoted figure is
the highest of them. **30B** on the same basis is **68.6 h / $30.80**.

QAT is not a line in this table because it was measured separately at **−0.9%,
free within noise** (`qat-compile-lattice.md`), and Hub uploads are not either
because they run out-of-band — training held 124,493 tok/s through a 321 MB
transfer at 05:18Z.
