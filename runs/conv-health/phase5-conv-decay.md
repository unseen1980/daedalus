# Phase 5: does a decay schedule stop ShortConv channel death?

Four schedules for the conv projections, one variable between them, read on the coupled `in_proj` x kernel x `out_proj` instrument rather than on the shipped weight proxy. The fix under test *is* a change to weight decay, so a magnitude metric can be satisfied by an arm where nothing shrank as easily as by one where nothing died.

**Scope: V2 only.** Every arm here is trained from initialization at a proxy shape. Nothing in this phase touches the released V1 weights, and no result below says a dead channel in the released model was revived -- a channel that collapsed during a 59.9B-token run is not brought back by choosing a different schedule for a future run.

## The preregistered rule

> An arm is selected only when its dead fraction is under 1%, its alive-channel projection norms stay within 2x the control's, its held-out loss is no worse than the control's by more than 0.5%, and removing its weakest *baseline-sized* channel set measurably costs held-out loss.

A stage is readable only if the control itself dies by at least 5%. These four thresholds are constants in `verdict()`, and `runs/conv-health/verdict-probe.json` records the same values from before the escalation was launched, so none of them moved after the results landed.

## Screen -- hidden 256, Muon lr 0.15, 600 steps

Positive control `shipped-0.1` died at 72.40%, and removing every channel it flagged moved held-out loss by -8.94e-08 nats. The stage is readable, and those channels were carrying nothing.

Norm columns are the arm's alive-channel mean over the control's, held-out loss is relative to the control, and negative is better.

| arm | dead | in_proj | out_proj | kernel | held-out loss | matched ablation | verdict |
|---|---|---|---|---|---|---|---|
| `weak-0.0133` | 5.53% | 1.27x | 1.61x | 0.75x | 0.06% | +0.388 nats over 1106/1112 (uncredited) | FAIL |
| `warmup-0-to-0.1` | 72.53% | 1.02x | 1.02x | 1.00x | 0.19% | +1.767 nats over 422/1112 (uncredited) | FAIL |
| `weak-then-0.1` | 68.88% | 0.95x | 0.96x | 0.99x | 0.02% | +1.631 nats over 478/1112 (uncredited) | FAIL |

## Escalation -- the shipped 150M shape, 500M tokens, Muon lr 0.04

`daedalus-150m` at 500,000,000 tokens, Muon lr 0.04.

Positive control `shipped-0.1` died at 53.86%, and removing every channel it flagged moved held-out loss by +2.98e-08 nats. The stage is readable, and those channels were carrying nothing.

Norm columns are the arm's alive-channel mean over the control's, held-out loss is relative to the control, and negative is better.

| arm | dead | in_proj | out_proj | kernel | held-out loss | matched ablation | verdict |
|---|---|---|---|---|---|---|---|
| `weak-0.0133` | 14.52% | 1.82x | 2.33x | 2.13x | 0.14% | +3.954 nats over 4855/4964 (uncredited) | FAIL |
| `weak-then-0.1` | 42.43% | 0.94x | 0.99x | 1.26x | -0.31% | +4.017 nats over 4021/4964 (uncredited) | FAIL |

## The same arms at both shapes

Left to right: `probe` -> `paired`. Muon decays once per optimizer step, so what a channel experiences is `sum(lr_t) * wd` and not tokens. The escalation runs a ~6x longer decay clock than the screen, and an arm whose cost grows with the clock is an arm the screen would have passed.

| arm | dead | max norm ratio |
|---|---|---|
| `shipped-0.1` (control) | 72.40% -> 53.86% | 1.00x by definition |
| `weak-0.0133` | 5.53% -> 14.52% | 1.61x -> 2.33x |
| `weak-then-0.1` | 68.88% -> 42.43% | 0.99x -> 1.26x |

## Verdict

**Negative result.** No tested schedule cleared the preregistered rule; recording the negative result rather than relaxing a bar after seeing the numbers.

- At the `paired` stage the shipped decay left 53.86% of conv channels dead and removing all of them cost +2.98e-08 nats. The death is real and the channels were not being used, so this is a capacity-allocation result: a schedule that keeps them alive has to show they then *earn* their place, not merely that they are alive.
- The matched-ablation clause could not be met by any arm at the `paired` stage. An arm needs the control's *per-layer* count of weakest-alive channels to spare, and a control that killed most of a layer requests more than any arm has left there. It decided no verdict: every arm it declined to credit also failed on its dead fraction, so the rule's outcome would be unchanged either way.
- The lowest dead fraction any arm reached at the `paired` stage was 14.52% (`weak-0.0133`), against a 1.00% bar. No tested schedule is close, so this is not a threshold that a slightly different ramp would have cleared.
- `weak-0.0133` bought its lower dead fraction with projection growth: out_proj reached 2.33x the control's alive-channel baseline against a 2x limit -- at the screen the same arm read 1.61x, so the cost grows with the decay clock rather than staying put. That is the equilibrium objection that kept a zero-decay arm out of this sweep, now measured on an arm that is in it.

## What this licenses for V2

- The instrument works and the shipped schedule's death is real, so a future from-scratch V2 can be measured on the same rule without re-establishing the baseline.
- No schedule in this sweep is a recipe. A V2 candidate has to clear the dead-fraction bar *and* the norm bar at a decay clock at least as long as the escalation's; every arm here failed at least one.
- The dead channels cost nothing to remove, so the honest framing of the opportunity is parameters that were paid for and not used, not quality that was lost. Any claimed gain from reviving them has to be shown as a held-out improvement, not as a higher alive count.
