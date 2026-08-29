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

## Open gate finding: the code sandbox does not yet hold

`scripts/code_eval.py` blocks Python-level network calls, but the gate claim is
that executed code reaches neither the network nor files outside its sandbox,
and today it can do both:

- The child runs as root with `PATH=/usr/bin:/bin`, so `subprocess.run(["curl",
  ...])` or `os.system` reaches the network without touching the patched
  `socket` module, and any absolute path works even with `PATH` emptied.
- Running as root, the child can read `/root/daedalus` checkpoints and the
  mode-0600 credential files under `/root/.config/daedalus`, and can write
  anywhere in the repository.

`unshare -n` is unavailable here: this container lacks the capability and the
call fails with `unshare failed: Operation not permitted`. Do not build the fix
on network namespaces.

Close it by dropping the child to an unprivileged uid and gid in the existing
`preexec_fn` (the parent is root, so `os.setgid` then `os.setuid` works),
chowning the per-item working directory to that uid so the candidate can still
write its own scratch files, and refusing to run at all if the drop did not take
effect. Harden the preamble alongside it so `subprocess`, `os.system`, and the
`os.exec*` family raise the same blocked-access error as the socket calls, and
categorise that refusal distinctly from a network block.

Add executable regression tests that fail on the current behaviour: a candidate
that shells out to a network client is refused, a candidate that reads a
root-owned mode-0600 file outside the sandbox is refused, and a candidate that
writes outside its working directory is refused. A test that only asserts the
socket patch is not evidence for this gate.

## The gate verdict is a script, not a claim

Every later phase reads this phase's gate. Record the verdict with
`scripts/gate_check.py`, which reads the written scorecards and decides each
criterion mechanically, rather than asserting the outcome in prose:

- synthetic controls scored 100% where the control expects it;
- two runs of the same evaluation at temperature zero produced identical
  per-item outcomes and an identical scorecard fingerprint;
- the FP16 and Q4_0 paired comparison covered the same item ids in the same
  order, with matching item counts and digests;
- the code sandbox refused network access and refused reads and writes outside
  its working directory.

Emit one machine-readable verdict per criterion with the scorecard path and the
observed value that decided it, exit non-zero when any criterion fails, and give
it focused tests including a scorecard that must fail each criterion. A gate
that cannot fail has not been tested.
