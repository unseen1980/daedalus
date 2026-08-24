# Phase 3 QAT recovery turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

Recover as much of the released model's Q4_0 quality penalty as possible without
repeating pretraining. Work in preregistered order and record every outcome,
including negative ones.

## Experiment design

Start each probe with `--init-from` on the released base checkpoint, never
`--resume`: the completed pretraining schedule and optimizer state must not be
restored. Enable exact-grid QAT from the first recovery step with `qat_frac=1.0`
and keep stock llama.cpp Q4_0/Q6_K export behaviour unchanged.

Run three fully decayed 100M-token learning-rate probes over identical data,
order, and seeds. Initial Muon candidates are 2e-4, 5e-4, and 1e-3; Adam rates
preserve the existing Muon:Adam ratio, and warmup is shortened for the small
budget. Select by paired Q4 perplexity reduction first, then FP16 retention,
full-pass BPB, five-task mean, and retrieval retention.

Run the winner for 300M tokens. Escalate to a 1B-token recovery only if Q4
damage improves at least 10% relative over the best 100M probe without violating
a retention gate. If no probe reaches that bar, stop the escalation and report
the negative result rather than spending 1B tokens.

Export FP16 and Q4_0 from every scored checkpoint. Verify that QAT master
weights survive checkpoint and export, and that the shipping lattice matches the
fake-quant lattice.

## Mandatory gates

All losses, gradients, and weights stay finite with no skipped non-finite
updates. FP16 perplexity regression stays at or below 0.5%. The five-task mean
drops by no more than 0.5 points. Retrieval drops by no more than 1 point
absolute at any depth. The improvement gate is at least a 50% reduction in the
roughly 6% Q4 perplexity penalty; the target is a penalty at or below 3% and the
stretch target is 1%.

Do not tune a threshold after seeing an outcome. Never resume weights after
confirmed numerical divergence: restart that candidate from its immutable input
checkpoint or mark it failed and advance per the preregistered gate.

## Working rules

Expose `--warmup-steps`, `--decay-frac`, loss chunk size, and gradient
checkpointing in `train.py` while preserving current defaults. Log
`qat_rel_rmse`, QAT tensor count, quantized and float validation identifiers,
and skipped-update counts. Extend `tests/test_qat.py`, `tests/test_train.py`,
and `tests/test_export.py`, and add `scripts/qat_recovery.py` with its own
focused tests.

Long training runs go through the existing supervised launcher with atomic
checkpoints and watchdog coverage; never sit inside a training loop. Publish the
best recovery privately with full provenance and leave the original release
untouched.

Run the focused checks before every commit. All shell, test, Git, PR, hash,
phase, and log actions must go through `/usr/local/bin/daedalus-approved`. Push
every tested commit to the active source branch immediately.
