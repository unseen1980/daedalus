# Phase 5 paired escalation: what was decided before it ran

Written at launch, 2026-08-25T04:13Z, with no escalation results in existence.
The plan's rule is that thresholds are not tuned after seeing outcomes, and the
cheapest way to hold to that is to write down the rule while it is still
impossible to know which way it cuts.

## What the probe established

Four schedules, 600 steps at hidden 256 and Muon lr 0.15, scored on the coupled
channel-health instrument. `runs/conv-health/verdict-probe.json`:

| arm | dead | in_proj / out_proj norm vs control | held-out loss vs control |
|---|---|---|---|
| shipped-0.1 (control) | 72.4% | 1.00 / 1.00 | — |
| weak-0.0133 | 5.5% | 1.27 / 1.61 | +0.06% |
| weak-then-0.1 | 68.9% | 0.95 / 0.96 | +0.02% |
| warmup-0-to-0.1 | 72.5% | 1.02 / 1.02 | +0.19% |

The positive control reproduces: removing all 1,112 of the control's flagged
channels moves held-out loss by -8.9e-08, so they were carrying nothing.

Neither ramp arm separates from the control. Both end at the shipped 0.1 and
hold it, and that is what the numbers track -- the death follows the *sustained*
decay strength, not the early window the two ramp shapes were designed to
protect. That is a negative result for the hypothesis those arms encode.

## Arms advanced, and why these

Plan step 6 advances the top two schedules. By probe dead fraction those are
`weak-0.0133` and `weak-then-0.1`; `warmup-0-to-0.1` is dropped as the worst of
the three fix arms.

`shipped-0.1` runs as well, and is not optional. Every criterion the arms are
read against is stated relative to it: norms within 2x *its* alive-channel
baseline, held-out loss no worse than *its* by 0.5%, and a matched ablation
sized from *its* flagged set. A subset without the control is not a cheaper
experiment, it is an unreadable one.

`weak-then-0.1` is escalated despite its probe result on purpose. Its ramp is a
fraction of the run, so at 3,815 steps it holds 0.0133 for ~1,145 steps before
reaching the shipped decay, where in the probe it held it for 180. That is a
materially different regime and the probe does not settle it. If it dies again
here, the ramp shape is finished and the report says so.

## The shape, and the one number that decides whether it can answer anything

`daedalus-150m`, 500M tokens, Muon lr 0.04, flat seq 2048, flat 131,072
tokens/step, warmup 300, decay-frac 0.45.

Muon decays as `w *= (1 - lr*wd)` once per optimizer *step*, so what a channel
experiences is `sum(lr_t) * wd`. Tokens do not appear in that. `batch_tokens` is
therefore not a throughput knob but the field that sets the experiment:

| shape | tokens | tokens/step | steps | decay clock | shipped arm sees |
|---|---|---|---|---|---|
| hero (shipped) | 59.9B | ~481k | 124,476 | ~1,890 | `exp(-189)` |
| paired escalation | 500M | 131,072 | 3,815 | 112.3 | `exp(-11.2)` |
| probe | 1.23M | 2,048 | 600 | 77.3 | `exp(-7.7)` |
| 500M at hero's 512k/step | 500M | 524,288 | 954 | 28.1 | `exp(-2.8)` |

The dead threshold needs roughly `exp(-4.6)` of shrink relative to a layer's
p95. The last row is the run that would have been launched by picking a batch
size for throughput: it would have spent three GPU-hours to find nothing dead in
any arm, which the verdict correctly reads as an invalid sweep. The chosen shape
sits above the probe's clock, and the probe killed 72% of channels.

`warmup` and `decay-frac` are `train.py`'s shipped defaults rather than the
probe's, because the regime being escalated to is the one the released model was
trained in.

## The bar, unchanged from the plan

Dead fraction under 1%; projection norms at or under 2x the control's
alive-channel baseline; held-out loss no worse by more than 0.5%; the matched
baseline-sized ablation measurably worse than baseline; training finite
throughout. The sweep is invalid unless the control itself dies by at least 5%.

Two outcomes are worth naming in advance:

- **`weak-0.0133` clears the bar.** Then the V2 recipe is a constant weak decay
  on the conv projections, and the open question its probe result already
  raises -- 1.27x/1.61x norm growth at 600 steps -- has to be answered here on
  the 3,815-step norms, not deferred. Growth that has not equilibrated by the
  end of this run does not become a recommendation.
- **Nothing clears the bar.** Then phase 5 reports a negative result: none of
  the three preregistered shapes prevents ShortConv channel death at the shipped
  decay strength, the death is governed by the sustained decay rather than its
  early schedule, and the next candidate is a decay weak enough to lose the race
  permanently -- which is the zero-decay arm this phase deliberately excluded on
  stability grounds and would have to re-open with a norm-equilibrium result,
  not a death result.

Neither is decided here.

## Provenance

- probe verdict `runs/conv-health/verdict-probe.json`, scored `scored-probe.json`
- shape `PAIRED_SHAPE` in `scripts/conv_health.py`, asserted in
  `tests/test_conv_health_sweep.py::test_the_escalation_batch_gives_the_decay_enough_steps_to_act`
- launched detached as controller phase `phase5-paired-escalation`, log
  `runs/conv-health/paired-escalation.log`
- 12-step GPU smoke at this shape: 14.1 GB peak of 24 GB, in-flight marker
  written beside the checkpoint `train.py` actually writes
