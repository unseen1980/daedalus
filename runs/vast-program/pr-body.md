Unattended research program from `docs/superpowers/plans/2026-08-24-daedalus-vast-program.md`, running on a single RTX 3090 Ti under a 144-hour hard deadline with an 8-hour reserved finalization window.

**Draft. Do not merge.** Phases land as atomic, tested commits and this description tracks their status.

## Live progress

- Heartbeat: [`vast/progress-20260824`](https://github.com/unseen1980/daedalus/tree/vast/progress-20260824) — refreshed every five minutes with elapsed hours, hours to the finalization window, and an action banner when a human is needed.
- Durable controller state: `runs/vast-program/state.json` with an append-only `events.jsonl` timeline.
- Operator procedures: `ops/vast/RUNBOOK.md`.

## Phase status

| Phase | State |
| --- | --- |
| 0 — Secure bootstrap and immutable baseline | passed |
| 1 — Unattended control plane | passed |
| 2 — Evaluation infrastructure | **passed**, verdict in `runs/eval/phase2-gate.json` |
| 3 — QAT recovery of released Daedalus | **complete**, verdict in `runs/qat-recovery/verdict.json`, reading in `runs/eval/phase3-evidence.md` — **no arm accepted; two operator decisions below** |
| 4 — Tokenizer lab for V2 | not started |
| 5 — ShortConv channel death prevention | not started |
| 6 — Architecture Pareto proxies | not started |
| 7 — Improved general corpus and mixture | not started |
| 8 — Daedalus-Code | not started |
| 9 — Finalization and reporting | begins no later than T+136h |

## Released-model baselines

Measured on this host with the artifacts named in each scorecard's provenance, before any adaptation.

| Metric | Value |
| --- | --- |
| FP16 perplexity | 6.6135 |
| Q4_0 perplexity | 6.9798 |
| **Q4_0 penalty** | **5.539%**, as a paired per-chunk delta over 292 chunks |
| Five-task mean | 47.374 |

The five-task mean sits 0.061 above the recorded 47.313. `runs/eval/phase2-evidence.md` demonstrates that residual is floating-point arithmetic by running the same checkpoint on CPU and CUDA, rather than asserting it. Phase 3 retention gates therefore measure against **47.374**, this host's figure: comparing across hosts would spend most of the gate's 0.5-point budget on numerics noise before QAT changed anything.

## Phase 2 gate

`scripts/gate_check.py` decides each criterion from the written scorecards and exits non-zero on failure. The command that produced the verdict is recorded in the controller state.

| Criterion | Observed |
| --- | --- |
| Synthetic controls | passkey 1.0, MQAR 1.0, copy-control 1.0 |
| Temperature-zero repeats | Identical fingerprint, item count, and item digest across two independent runs |
| FP16/Q4 pairing | 292 items per side, same ids in the same order, `gguf-f16` against `gguf-q4_0` |
| Code sandbox | Probed at runtime: network client and `os.system` blocked, writes and secret reads outside the sandbox refused, privileges dropped to uid 65534 |

Three defects were found by using the evaluators rather than only unit-testing them, each of which would have silently corrupted a downstream comparison: the pinned `llama-cli` writes a chat UI to stdout, the code sandbox ran candidates as root with a network client on `PATH`, and the code harness was not executing a real test program.

## Phase 3 — QAT recovery: the penalty is gone, no arm was accepted

Full reading in `runs/eval/phase3-evidence.md`; the verdict is mechanical, from `runs/qat-recovery/verdict.json`.

**A defect first: the released checkpoint could not be quantization-aware trained at all.** `qat._safe_reciprocal` guarded `d == 0`, where a *fully* dead channel lands. A channel on its way there passes through a window where the block absmax is denormal-small but not zero, so `d` is representable and `1/d` is not: it overflows fp32 to `inf`, and the block's exactly-zero elements compute `0 * inf = NaN`. Three FFN tensors, 3,095 NaNs, every stored weight finite. The first smoke run produced a NaN loss on step one and then span forever, because `train_step` does not advance `step` or `tokens_seen` when it skips and both of `fit`'s break conditions are thresholds on those. Separately, the four tests certifying our Q4_0 grid *is* llama.cpp's had been skipping on this box since Phase 2 — `_find_libggml` never looked in `/opt/llama.cpp`. All four now run and pass against the real `libggml-base.so`.

**Results**, three preregistered 100M-token arms from the released base with `--init-from`, QAT from step 0, identical data and order:

| | released base | lr 2e-4 | lr 5e-4 | lr 1e-3 |
| --- | --- | --- | --- | --- |
| FP16 perplexity | 6.6135 | 6.7126 | 6.7034 | 6.7305 |
| Q4_0 perplexity | 6.9798 | 6.7057 | **6.6873** | 6.7054 |
| Q4 penalty | +5.539% | −0.103% | −0.240% | −0.373% |
| five-task mean | 47.374 | 47.050 | 47.632 | **48.082** |
| retrieval, worst depth | — | −7.0 | −23.0 | −7.0 |
| skipped updates | — | 0 | 0 | 0 |

The quantization penalty is not halved but **inverted** — 102–107% reduction, clearing both the 3% target and the 1% stretch. Q4_0 perplexity, the format that ships, improves **4.19%** for 1.03 GPU-hours and about $0.46. Two of three arms raised the five-task mean.

**Two gates blocked every arm, and they are not equally serious.** FP16 perplexity regressed 1.36–1.77% against a 0.5% limit — inherent to QAT, since the STE optimizes the quantized model and the FP16 artifact is the float master. Retrieval degraded: MQAR is worse at d512/d1024/d2048 in all three arms by 2–6 points, growing with depth. Three independent arms moving the same way is a pattern; the single −23 at passkey d256 is one cell, unreproduced by neighbouring rates, and carries ~3.8 points of binomial noise at n=100.

Per the preregistered stop rule the 300M follow-up and 1B escalation **did not run** — about 14 GPU-hours deliberately not spent. Nothing is published, because publishing implies endorsement.

**Two decisions for the operator, both of which change a preregistered gate and so are not this branch's to make:**

1. Is a 0.5% FP16 perplexity limit the right gate for a *ship-format* recovery? As written it rejects a model whose Q4_0 artifact is 4.19% better and whose task mean is up. Scoring ship-format against ship-format (Q4_0 6.6873 vs released 6.9798) passes comfortably.
2. Is the 2–6 point MQAR decline acceptable, or does the recipe need a retrieval-safe variant first?

One preregistered deviation, made before any arm ran: the retrieval gate was re-baselined at 100 items per depth, because at the Phase 2 baseline's 10 items one item is 10 points and a 1-point gate could only ever be met by exact equality. Re-measuring showed the old baseline moving up to 8 points per depth on an unchanged model. 100 items makes the gate expressible, not statistically resolvable — noise is still ~3.6 points — and the evidence file says so rather than implying a precision the instrument lacks.

## Control plane

`daedalus/program_state.py` holds an atomic snapshot beside an append-only timeline; `scripts/vast_program.py` owns phases, one-process leases, and deadline gates; `scripts/boot_resume.py` resumes only an approved incomplete marker; `scripts/github_progress.py` publishes a sanitized heartbeat from an isolated worktree; `ops/vast/run-approved` is the only shell, Git, PR, and evaluation surface an engineering session may use, and it refuses default-branch pushes, PR merges, secret paths, and arbitrary shell fragments.

`daedalus/session_keeper.py` closes the gap that stopped the program before this branch: the controller owned phases, but nothing owned the Claude sessions implementing them. The keeper verifies both plan hashes before every launch, concatenates the verified plans into a mode-0600 system prompt, assigns and records the session id before launching so any death stays resumable, resumes the same session for bounded repair continuations, escalates to a fresh independent session, records a hard blocker instead of relaunching forever, yields the box while a supervised job holds it, opens the finalization phase at T+136h, and refuses to start beside an orphaned session.

Both failure drills pass and are recorded in the timeline: a session killed mid-turn is counted as a failure and relaunched, and a keeper killed with SIGKILL is restarted by supervisord without a duplicate session.

## Security fix worth reviewing

The workspace was never marked trusted, so Claude Code ignored `.claude/settings.json` in full — including the `deny` rules keeping an engineering session away from `.env`, the runtime credential directory, and the SSH material. It reports this only in a stderr warning that a non-interactive session shows nobody, so the control plane looked configured while enforcing none of it. `ops/vast/trust_workspace.py` records trust in the user-level config as an installed, tested step.

## Constraints held

- Unmodified stock llama.cpp remains the runtime target.
- No commit, push, merge, or rewrite touches the default branch.
- No model or dataset is published; experiment repositories stay private pending explicit approval.
- The live instance endpoint appears in no commit, log, or progress file.
