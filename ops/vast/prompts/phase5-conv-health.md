# Phase 5 ShortConv channel health turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

Find a from-initialization optimizer schedule that prevents ShortConv channel
death without unbounded weight growth. The deliverable is a V2 recipe and its
evidence. It does not revive the released model's dead channels and must never
be described as doing so.

Phase 3 met these channels already: the released checkpoint could not be
QAT-trained because blocks whose absmax was denormal-small produced a
non-representable reciprocal, and every exactly-zero weight in those blocks then
became NaN. Three FFN tensors carried 3,095 NaNs. Channels on their way to zero
are not a cosmetic problem; they broke a phase outright.

## Measuring health

Define functional channel health from the coupled contribution of `in_proj`, the
depthwise kernel, and `out_proj` together. A channel is not alive because one of
the three has norm; it is alive because the composition carries signal. Keep the
existing relative-weight proxy only as a cheap leading indicator, and say which
one any reported number came from.

Reproduce the established positive control first -- hidden 256, LR 0.15, 600
steps -- and confirm it still shows accelerated death. An experiment that cannot
reproduce the problem cannot demonstrate a fix.

## Arms

Compare the shipped constant 0.1 decay, a constant 0.0133, a zero-to-0.1 warmup
over the first 10%, and 0.0133 early followed by a ramp to 0.1 by 30%.

Every arm that looks clean must then pass a matched functional ablation:
removing its weakest baseline-sized channel set must measurably worsen a
deterministic held-out loss. A schedule that merely keeps norms above a
threshold while those channels carry nothing has moved the metric, not the
problem.

Zero decay is not a production answer unless projection norms reach a stable
equilibrium; the recorded 6.8x to 10.5x growth warning stands.

## Escalation and selection

Advance the top two schedules to a paired 150M-parameter, 500M-token run at LR
0.04, where the shipped baseline previously showed material death.

Select only if dead fraction is under 1%, projection norms stay at or under 2x
the alive-channel baseline, held-out BPB is no worse by more than 0.5%, Q4
damage does not increase materially, and training stays finite throughout.

Record a negative result plainly if no schedule clears that bar.

## Working rules

Modify `daedalus/muon.py` to support a named conv-projection parameter group and
a scheduled weight decay without changing the default optimizer state layout, so
existing runs are unaffected. Long runs must outlive the turn that starts them.
Run focused checks before every commit. All shell, test, Git, PR, hash, phase,
and log actions go through `/usr/local/bin/daedalus-approved`.
