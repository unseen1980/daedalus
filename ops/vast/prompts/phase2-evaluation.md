# Phase 2 evaluation turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

Build the retrieval, paired FP16/Q4_0, and code-execution evaluation
infrastructure that every later gate consumes, with focused tests and provenance
sidecars beside every score. Preserve stock llama.cpp compatibility and keep all
artifact publication private until explicit approval.

## Remaining work

Score the released base and instruct artifacts to establish retrieval,
quantized, and coding baselines before any adaptation touches the weights. Use
the wrapper subcommands `eval-retrieval`, `eval-quant`, `eval-code`, `eval-bpb`,
and `eval-tasks`; they own the artifact and llama.cpp paths this session may not
read directly. Write each result under `runs/eval/` with its scorecard, and
distinguish full-pass BPB from the bounded 100-batch evaluator sample in every
result file.

## Gate before phase 3

Synthetic controls score 100% where expected. Repeated evaluations are
bit-for-bit identical at temperature zero. Per-item counts and hashes match
between the FP16 and Q4_0 paired comparisons. Code execution reaches neither the
network nor files outside its sandbox. The original roughly 6% Q4 damage and
47.31 task mean are reproduced rather than silently replaced by a different
checkpoint, tokenizer, or evaluation limit.

When every gate passes, record the verdict with its evidence and transition the
controller to `phase3-qat-recovery`.

## Working rules

Run the focused checks before every commit. All shell, test, Git, PR, hash,
phase, and log actions must go through `/usr/local/bin/daedalus-approved`. Push
every tested commit to the active source branch immediately.
