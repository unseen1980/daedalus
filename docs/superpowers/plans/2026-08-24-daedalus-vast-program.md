# Daedalus Improvement and Code Program

Run a six-day, evidence-gated research program on a Vast RTX 3090 Ti box. A deterministic controller owns training, deadlines, checkpoints, and pass/fail gates; bounded Claude Code sessions on Opus with xhigh effort implement and review each slice. The program produces a QAT-recovered Daedalus V1, evaluated tokenizer/data/optimizer/architecture recommendations for V2, and a Python-first multilingual Daedalus-Code candidate trained from released base weights for up to 3B continued-pretraining tokens.

## Goals

1. Recover as much of the released model's Q4_0 quality penalty as possible without repeating pretraining.
2. Train and compare 24,576-, 32,768-, and 40,960-token vocabularies for a future V2.
3. Validate a from-step-zero optimizer schedule that prevents ShortConv channel death without unbounded weight growth.
4. Tune depth, attention-layer count, and KV-head count through stock-llama.cpp-compatible proxy experiments.
5. Rebuild the general corpus with complete decontamination, stronger uniqueness controls, and proxy-selected mixture weights.
6. Add retrieval-sensitive, quantized-artifact, and code execution evaluations.
7. Produce Daedalus-Code from `hero-base-f16` by continued pretraining, SFT, optional execution-grounded preference tuning, and QAT.
8. End with a measured report separating released-model gains, code-model gains, proxy evidence, and V2 projections.

## Fixed Decisions

- The Vast endpoint is supplied only through the protected bootstrap environment (`VAST_SSH_HOST`, `VAST_SSH_PORT`, `VAST_SSH_USER`). SSH disconnects must not stop work. Never commit or publish the live endpoint.
- Hardware target: one RTX 3090 Ti 24GB, 24 CPU threads, 64GB RAM, and at least 302GB NVMe.
- Hard controller deadline: 144 hours from accepted program start. The controller stops work, checkpoints, uploads, and reports at the deadline; it does not destroy the instance.
- Claude Code authentication: `CLAUDE_CODE_OAUTH_TOKEN`; model `opus`; effort `xhigh`.
- Git policy: optimization/control-plane/evaluation/V2 research changes go to `vast/daedalus-improvements-20260824`; Daedalus-Code-specific changes go to `vast/daedalus-code-20260824`. Never commit, push, merge, or rewrite the default branch directly.
- PR policy: open a draft optimization PR to `main` after the bootstrap/control-plane gate. Create the code branch from the optimization branch and open a stacked draft PR to the optimization branch. Mark ready only after each PR's final gates; never auto-merge.
- Progress policy: maintain `vast/progress-20260824` in a separate worktree and push `STATUS.md`, `status.json`, `recent-metrics.json`, and an append-only phase timeline every five minutes. Push every atomic tested source commit immediately.
- Autonomy: proceed through preregistered passing gates; diagnose and repair implementation, test, environment, data, export, and orchestration failures without routine user approval. Stop only the affected phase on an unrecoverable hard gate.
- Runtime compatibility: unmodified stock llama.cpp is non-negotiable.
- Daedalus-Code is Python-first with multilingual support; train to 1B and continue to 3B only if quality-retention gates pass.
- All Hugging Face experiment repositories remain private until explicit user approval.

## GitHub Branches, PRs, and Live Progress

1. Validate GitHub push and PR permissions before expensive work. Accept `GH_TOKEN`, `GITHUB_TOKEN`, or a pre-authenticated `gh` credential from the protected runtime environment; never print credentials.
2. Fetch the current default branch and record its commit as `PROGRAM_BASE_SHA`. Every source branch must descend from that SHA.
3. Transfer the exact private execution plan through the bootstrap channel to a mode-0600 path outside the repository. Record its SHA-256 in the private run manifest. This committed document is the sanitized reviewable equivalent; live endpoints, local-machine paths, and operational identifiers stay private.
4. Pass both plans to each Claude Code engineering session through a root-owned temporary prompt file. Check both hashes before every Claude launch. Never copy the private plan into Git, progress files, PR bodies, logs, W&B, or Hugging Face artifacts.
5. Open the optimization draft PR immediately after the unattended-control gate passes. Keep its body updated with phase status, tests, experiment links, artifact hashes, known failures, and the progress-branch URL.
6. Create the code branch only after shared training/evaluation interfaces stabilize. Open it as a stacked draft PR so its diff contains only code-specific preparation, evaluation, training, post-training, and documentation.
7. Use atomic source commits. A failed test or incomplete edit may be committed only as an explicitly labeled `wip:` recovery point; the next autonomous turn must repair or revert it before a PR can be ready.
8. Every five minutes, the progress worktree commits a heartbeat containing UTC timestamp, active phase/job, source branch and SHA, latest test/gate result, GPU utilization and memory, latest loss/BPB/QAT/channel metrics, elapsed/deadline time, ETA, artifact links, last error/retry, and whether user action is required.
9. Progress pushes are best-effort and never block training. Queue failures locally and retry with finite exponential backoff. Never force-push or rewrite the progress timeline.
10. At finalization, update both PR descriptions from immutable reports. Leave failed or partial work as draft with an explicit blocker. Never merge either PR automatically.

## Scope Boundaries

Included:

- Engineering, tests, data processing, proxy training, QAT recovery, code specialization, private artifact publication, and final reports.
- A complete improved general-corpus rebuild and a separate code corpus.
- Final Mac commands for representative Apple Silicon decode measurements.

Excluded:

- A fresh 59.9B-token V2 pretraining run.
- Changing the LFM2 operator or requiring a patched llama.cpp binary.
- Automatically destroying the Vast instance.
- Publishing any new model or dataset publicly.
- Claiming full-model gains from tokenizer, convolution, architecture, or mixture proxy results.

## Execution Architecture

1. `supervisord` or the Vast on-start mechanism keeps control processes alive after SSH disconnect and restarts them after failure. Reboot recovery must be simulated before expensive work.
2. A deterministic Python controller stores atomic state and an append-only event log, launches named phases, enforces gates and the deadline, reserves eight final hours for export/report/upload, and never depends on an active Claude conversation.
3. Existing `run_with_resume`, watchdog, atomic checkpoints, W&B run IDs, and out-of-band Hugging Face uploads supervise long training jobs.
4. Claude Code runs bounded engineering and review turns. Rate limits or a stopped Claude session cannot interrupt already-launched data or training jobs.

Secrets are split before Claude starts:

- `$DAEDALUS_CONFIG_DIR/claude.env`: only `CLAUDE_CODE_OAUTH_TOKEN`.
- `$DAEDALUS_CONFIG_DIR/runtime.env`: Hugging Face, W&B, GitHub, and other runtime credentials, excluding the Claude token.
- Both files are mode `0600` and outside the repository. The staging `.env` is deleted after splitting.
- Claude file permissions deny `.env`, `.env.*`, `$DAEDALUS_CONFIG_DIR/**`, SSH material, and credential stores.
- Claude receives no HF/W&B/GitHub token directly. Approved wrappers and supervisor services own those credentials.

## Autonomous Repair Contract for Claude Code

1. Read the versioned plan, private execution plan, current controller state, latest progress snapshot, failing command output, and nearby owning code before acting.
2. Do not ask the user for routine implementation decisions. Choose the smallest evidence-backed, reversible solution consistent with repository patterns, stock llama.cpp compatibility, branch boundaries, experiment gates, and deadline policy; record the decision.
3. On code/test failure, reproduce narrowly, state a falsifiable hypothesis, add or update a focused regression test, implement the smallest repair, run the focused check, then run the affected suite.
4. On environment/tooling failure, verify disk/RAM/GPU/network/version state, repair through approved wrappers, and retry up to three times with bounded backoff. Never weaken correctness or provenance gates.
5. On training/export/evaluation failure, preserve the last good checkpoint and logs, repair orchestration or implementation defects, then resume. Never resume weights after confirmed numerical divergence; restart that candidate from its immutable input checkpoint or mark it failed.
6. On a valid experiment that misses its target, record a negative result and continue to the next preregistered arm. Do not tune thresholds after seeing outcomes.
7. Before every commit, run focused executable checks. Before each PR-ready transition, run the full suite and artifact/provenance checks. Commit and push repairs without waiting for approval.
8. If Claude exits, is rate-limited, reaches a turn cap, or leaves an incomplete repair, the controller continues the same session with explicit failure context. After three unsuccessful continuations, launch a fresh Opus+xhigh review session. After two independent sessions fail, mark a hard blocker and continue unrelated safe phases where dependencies permit.
9. Hard blockers that must not be guessed through include missing credentials, ambiguous data license, corrupt baseline, spend/deadline overflow, released-artifact or default-branch overwrite risk, repeated numerical divergence, disk exhaustion threatening durable artifacts, or work outside approved scope.
10. Claude runs non-interactively with `--permission-mode dontAsk`, repository Read/Edit permissions, and only the approved wrapper for shell/network/Git operations. Unapproved actions are denied rather than waiting for a prompt.

## Phase 0: Secure Bootstrap and Immutable Baseline

1. Connect with the existing local private key and verify the host fingerprint before accepting it.
2. Record instance identity, UTC start time, deadline, GPU/CPU/RAM/disk/network details, and advertised rate in a private run manifest.
3. Synchronize the local source snapshot without `.git`, `.env`, data, runs, caches, checkpoints, or generated paper files onto a fresh clone. Verify private and sanitized plan hashes, create the optimization branch from `PROGRAM_BASE_SHA`, and make this plan the first branch commit.
4. Transfer the local `.env` to a temporary root-only path, split it without echoing values, validate only required variable presence, and remove the staging file.
5. Configure protected outbound GitHub authentication and validate clone/fetch/push/PR access. The wrapper may push only the optimization, code, and progress branches; it may open/edit but never merge or close the two PRs.
6. Create private experiment namespaces distinct from released repositories. Refuse startup if an experiment target resolves to a released-model path.
7. Install system packages, Python environment, pinned dependencies, llama.cpp at a recorded commit, and evaluation tools. Record versions.
8. Restore the released base FP16 checkpoint, original tokenizer, final hero checkpoint, Q4_0 GGUF, evaluation files, and available tokenized shards. Record SHA-256 hashes and Hub revisions.
9. Run the current test suite and real-model forward/export smoke.
10. Reproduce baseline task scores, full-pass BPB, FP16 perplexity, Q4_0 perplexity, artifact size, and retrieval/code baselines. Distinguish full-pass BPB from bounded sample BPB in every result.

Gate:

- Checkpoint/tokenizer hashes match published artifacts.
- Five-task scores reproduce exactly or within one item per task.
- FP16/Q4 perplexity reproduces within 0.5% relative.
- The original Q4 damage and 47.31 task mean are not silently replaced by a different checkpoint, tokenizer, or evaluation limit.

## Phase 1: Unattended Control Plane and Claude Code

Create:

- `scripts/boot_resume.py`
- `scripts/vast_program.py`
- `daedalus/program_state.py`
- `ops/vast/bootstrap.sh`
- `ops/vast/supervisord.conf`
- `ops/vast/run-approved`
- phase prompts under `ops/vast/prompts/`
- `scripts/github_progress.py`
- `scripts/pr_status.py`
- focused tests for boot resume, controller state, GitHub progress, and PR policy

Steps:

1. Write failing tests for completed/halted/malformed markers, boot-resume caps, leases, deadline reservation, phase retries, and branch/repository guards.
2. Implement safe post-reboot resume from approved incomplete markers, always resuming on attempt one and refusing halt/completion markers.
3. Implement atomic controller state plus append-only events. Record input hashes, commands, timestamps, exits, metrics, and gate verdicts.
4. Enforce one active controller through PID/start-ticks identity.
5. Refuse new work projected beyond hour 136; start finalization at hour 136; checkpoint and stop by hour 144.
6. Implement the approved wrapper for tests, formatting, phase launches, status, hashes, branch-scoped commit/push, draft-PR create/edit, and safe log reads. Reject arbitrary shell fragments, destructive Git operations, PR merge/close, and default-branch pushes.
7. Install current Claude Code, require version 2.1.219 or newer, run `claude doctor`, record the version, and disable mid-program auto-upgrade.
8. Configure Opus+xhigh, `dontAsk`, bypass disabled, credential paths denied, and only the approved wrapper plus repository Read/Edit tools available.
9. Validate authentication and one non-editing structured call without printing the OAuth token.
10. Launch bounded `claude -p --model opus --effort xhigh --permission-mode dontAsk` turns with both plan files and phase-specific repair prompts.
11. Create the progress worktree and verify a five-minute heartbeat reaches GitHub without secrets or large logs.
12. Kill Claude, controller, progress publisher, and a supervised smoke trainer independently. Confirm clean recovery, W&B continuity, progress history continuity, and no duplicate processes.
13. Push tested control-plane commits, open the draft optimization PR to `main`, and publish its URL on the progress branch.

No expensive training begins until every restart, branch, secret, deadline, and upload drill passes.

## Phase 2: Evaluation Infrastructure

Create deterministic retrieval, paired GGUF, and code execution evaluators with focused tests. Modify `eval.py` to record checkpoint/tokenizer hashes, BPB full/sample mode, task revisions, seeds, item counts, and per-item sidecars.

1. Add passkey and multi-query associative recall at context depths 256, 512, 1024, and 2048.
2. Add copy/retrieval controls that independently catch prompt formatting defects.
3. Run retrieval through PyTorch FP16 and stock llama.cpp GGUF with exact prompts and per-item outcomes.
4. Score identical task items for FP16 and Q4_0 and report paired deltas.
5. Add EvalPlus-backed HumanEval+ and MBPP+ execution with no network, timeouts, resource limits, isolated directories, deterministic generation, pass@1, syntax validity, and failure categories.
6. Add code BPB holdouts by language and a general replay holdout.
7. Define one scorecard schema consumed by every later gate.
8. Score released base and instruct artifacts before adaptation.

Gate:

- Synthetic controls score 100% where expected.
- Repeated temperature-zero evaluations are bit-identical.
- Paired FP16/Q4 item counts and hashes match.
- Code execution cannot access network or files outside its sandbox.

## Phase 3: QAT Recovery of Released Daedalus

Expose warmup, decay fraction, loss chunk size, and gradient checkpointing in `train.py` while preserving defaults. Log QAT RMSE, tensor count, validation identifiers, and skipped updates. Add a QAT recovery orchestrator and tests.

1. Use `--init-from` on the released base checkpoint, not `--resume`.
2. Enable exact-grid QAT from the first recovery step with `qat_frac=1.0` and retain stock Q4_0/Q6_K export behavior.
3. Run fully decayed 100M-token probes at Muon LRs `2e-4`, `5e-4`, and `1e-3`, with identical data/order/seeds.
4. Select by paired Q4 perplexity reduction, FP16 retention, full BPB, five-task mean, and retrieval retention.
5. Run the winner for 300M. Continue to 1B only if Q4 damage improves at least 10% relative over the best 100M probe without retention violations.
6. Export FP16 and Q4_0 from each scored checkpoint and verify master-weight and shipping-lattice correctness.
7. Publish the best recovery privately with full provenance; never alter the original release.

Acceptance:

- No non-finite loss, gradients, weights, or skipped non-finite updates.
- FP16 perplexity regression at most 0.5%.
- Five-task drop at most 0.5 points.
- Retrieval drop at most one point at any depth.
- Improvement target: at least 50% reduction in the current Q4 penalty; target Q4 penalty at most 3%, stretch at most 1%.
- If no probe improves Q4 damage by at least 10% relative, stop escalation and report the negative result.

This phase directly improves released V1 weights.

## Phase 4: Tokenizer Lab for V2

Create explicit tokenizer training/evaluation support while preserving SmolLM2 defaults.

1. Build a deterministic source-balanced text sample with immutable row/revision hashes covering general text, math, technical prose, dialogue, and planned code languages.
2. Train byte-level BPE tokenizers at 24,576, 32,768, and 40,960 entries.
3. Pin special-token strings/IDs, `<|endoftext|>` at ID 0, ChatML tokens, byte fallback, and UTF-8 round trips.
4. Measure bytes/token by domain, code fragmentation, indentation/newline behavior, pathologies, throughput, embedding parameters, and projected Q6_K bytes.
5. Train identical tiny models under equal compute/bytes and compare BPB rather than token perplexity.
6. Select by a preregistered Pareto rule: no domain fertility regression beyond 5%, code improves or ties, tiny BPB improves or remains within 0.5%, and embedding bytes materially fall.
7. Produce a migration report. Do not transplant the tokenizer into V1 or Daedalus-Code weights.

## Phase 5: Prevent ShortConv Channel Death for V2

Add named conv-projection optimizer groups, scheduled weight decay, functional channel-health instrumentation, and tests while preserving default optimizer layouts.

1. Define functional health from coupled input projection, depthwise kernel, and output projection contribution.
2. Reproduce the established accelerated baseline death positive control.
3. Compare shipped constant 0.1 decay, constant 0.0133, zero-to-0.1 over 10%, and 0.0133 followed by a ramp to 0.1 by 30%.
4. Require matched functional ablation: removing a baseline-sized weakest-channel set must worsen deterministic held-out loss.
5. Reject zero decay unless projection norms show a stable equilibrium.
6. Advance the top two schedules to a paired 150M, 500M-token LR-0.04 experiment.
7. Select only when dead fraction is below 1%, norms remain within 2x the alive baseline, BPB is no worse by more than 0.5%, Q4 damage does not materially increase, and training remains finite.
8. Record a V2 from-initialization recipe. Never claim existing dead channels are revived.

## Phase 6: Architecture Pareto Proxies

Generate validated stock-compatible experiment presets and use successive halving.

1. Vary depth, attention layers, KV heads, and FFN width while checking divisibility, quantization dimensions, parameters, KV bytes, and exportability.
2. Stage A trains scaled variants. Stage B trains the best parameter-matched 150M candidates for 250M tokens. Stage C trains at most two finalists for 1B only when deadline reserve and prior discrimination justify it.
3. Include the shipped 18x768, six-attention, four-KV configuration as every-stage control.
4. Include a truly parameter-matched corrected 24x640 depth comparison.
5. Evaluate full BPB, appropriately powered task samples, retrieval by depth, artifact size, KV traffic, GGUF load, and Vast CPU decode shape. Leave final Apple Silicon speed pending.
6. Select a Pareto set, not a quality-only winner.

Recommendation gate:

- BPB within 0.5% of control.
- Retrieval within two points at every depth.
- KV bytes/context-token no more than 6,144, preferably 4,096.
- Stock llama.cpp export/load succeeds.
- Artifact size and depth-zero decode do not erase the long-context benefit.

## Phase 7: Improved General Corpus and Mixture

Improve `dataprep`, persistent deduplication, contamination coverage, mixture optimization, and provenance with preserved defaults and focused tests.

1. Freeze complete contamination indexes for every scored item/split.
2. Keep normalized exact hashes across the entire build. Expand cross-source near-duplicate groups while retaining memory bounds and recording coverage/resets.
3. Record source revision, license, filters, document count, unique tokens, duplicate drops, and contamination drops.
4. Remove or effectively zero the tiny conversation source in general pretraining; reserve dialogue for SFT.
5. Top up unique web, code, and technical data so no selected source exceeds four epochs.
6. Build whole-document or whole-repository holdouts before packing.
7. Compare baseline, quality-heavy, and proxy-derived mixture weights under equal compute.
8. Select by aggregate BPB plus domain floors.
9. Rebuild old-tokenizer shards needed for V1/code experiments and selected-tokenizer shards for future V2, subject to disk budget. Upload completed private shards incrementally and delete only verified-safe intermediates.
10. Run full contamination, split integrity, source headroom, epoch-cap, and manifest hash audits.

Acceptance:

- No known exact evaluation contamination.
- Complete scored-split coverage.
- No source above four planned epochs.
- Mixture L1 skew at most five points and no all-capped fallback.
- Every source/transformation is revision-pinned and reproducible.

## Phase 8: Daedalus-Code, Reusing Base Weights

### Branch and PR Boundary

1. Ensure the optimization branch is pushed and its draft PR current. Create `vast/daedalus-code-20260824` from the current tested optimization SHA and record that parent SHA.
2. Open a stacked draft PR from the code branch to the optimization branch before code-specific implementation.
3. Commit only code-specific preparation, evaluation, training/post-training, model-card, and result changes on this branch. Shared fixes go to the optimization branch first and are then merged forward.
4. Push every tested code commit immediately and update the stacked PR body with 250M/1B/3B/SFT/DPO/QAT gates. Never auto-merge.

### Data Design

- Start from `hero-base-f16`, never the SFT/DPO checkpoint.
- Continued-pretraining mixture: 65% permissively licensed code, 15% technical/math prose, and 20% original general replay.
- Code split: Python 55%, JavaScript/TypeScript 12%, C/C++ 10%, Rust 8%, Go 6%, Java 5%, shell/SQL/other 4%.
- Record repository identity, commit/revision, and license. Split train/holdout by repository.
- Decontaminate against HumanEval+, MBPP+, and additional code benchmarks using prompts, reference solutions, tests, and repository metadata.

Training:

1. Score the untouched base on code BPB, syntax validity, HumanEval+/MBPP+ pass@1, retrieval, general BPB, and five general tasks.
2. Run three fully decayed 250M-token probes from the same base at Muon LR `5e-4`, `1e-3`, and `2e-3`, with proportional Adam rates and identical data order.
3. Select by code BPB and execution pass@1 subject to general retention. Stop if no branch improves code BPB by at least 2% or execution/syntax signal.
4. Train a fresh 1B branch with the selected settings. Continue only if general BPB regression is at most 1.5%, five-task mean drop at most one point, retrieval drop at most two points, and code metrics improve.
5. If the 1B gate passes and completion fits before finalization, continue from 1B weights for another 2B with `--init-from`, lower LR, fresh WSD, and the same replay floor. Treat it as staged adaptation.
6. Run code/general SFT on syntax-checked and execution-tested conversations.
7. Run DPO only if held-out preference accuracy and execution pass@1 improve; otherwise keep the SFT winner.
8. Apply the winning QAT recipe, export FP16/Q4_0, and rerun all code/general/retrieval evaluations.
9. Publish private base, instruct, and Q4_0 artifacts under a new experiment repo. Preserve original releases.

Success gates:

- Final code BPB improves by at least 5%.
- HumanEval+/MBPP+ pass@1 and syntax validity improve over untouched base.
- General full-pass BPB regression at most 1.5%.
- Five-task mean drop at most one point and no task drops more than two points without explicit review.
- Retrieval drop at most two points at every depth.
- Q4 penalty meets the selected V1 QAT target or is transparently reported.
- Label this Daedalus-Code V1 because it inherits the 49k tokenizer and dead convolution channels.

## Phase 9: Finalization, Reporting, and Handoff

Begin no later than hour 136.

Create:

- `runs/final/improvement-report.json`
- `runs/final/improvement-report.md`
- `runs/final/v2-recommendation.md`
- `runs/final/daedalus-code-next.md`
- immutable artifact manifests with hashes, Hub locations, configs, tokenizer/data manifests, seeds, and producing commits

Report:

1. Released V1 baseline and provenance.
2. QAT-recovered V1 FP16/Q4 perplexity, damage reduction, BPB, five tasks, retrieval, artifact size, and speed.
3. Tokenizer fertility/BPB and projected parameters/bytes, marked V2-only.
4. Convolution dead fraction, functional ablation, norms, BPB/Q4 effects, marked V2-only.
5. Architecture Pareto quality, retrieval, KV bytes, file size, export, and CPU decode shape, with Mac speed pending.
6. Data unique tokens, duplicate/contamination drops, source shares, epoch headroom, and selected mixture.
7. Daedalus-Code base versus 250M/1B/3B/SFT/DPO/QAT checkpoints on code and general metrics.
8. Negative results and stopped branches.
9. Exact recommendation for future V2 and continuation plan for Daedalus-Code.
10. Remaining Mac work: alternating Apple Silicon llama.cpp benchmarks and fixed-prompt generation review.

Final validation:

1. Run focused tests after each phase and the full suite before finalization.
2. Re-run headline metrics from immutable final artifacts.
3. Fresh-download private Hub artifacts and hash-match them.
4. Verify optimization/code branches are pushed, progress heartbeat current, both draft PRs have correct bases/descriptions, default branch unchanged, no secrets tracked, and `.env` remains ignored.
5. Resolve every `wip:` commit before marking a PR ready. Otherwise retain draft status with an explicit blocker.
6. Record W&B/HF/Git/PR links without credentials.
7. Stop training/controller Claude workers after upload drain. Push a final progress snapshot, then stop the publisher.
8. Emit `PROGRAM_COMPLETE` or `PROGRAM_HALTED` with reason, final SHAs, and PR URLs. The user reviews and destroys the instance.

## Deadline and Degradation Policy

- Hours 0-8: bootstrap, controls, baseline setup.
- Hours 8-28: evaluation and tokenizer tooling, baseline scoring.
- Hours 28-50: corpus rebuild/audit and QAT probes where resources do not contend.
- Hours 50-72: QAT escalation and convolution experiments.
- Hours 72-104: architecture and mixture proxies.
- Hours 104-136: Daedalus-Code probes, 1B gate, conditional extension/SFT/QAT.
- Hours 136-144: immutable rescoring, uploads, reports, shutdown.

After measured throughput is available, prune in this order if necessary:

1. Skip Stage C architecture finalists while retaining Stage A/B.
2. Reduce repeated proxy seeds, never controls or per-item evaluation.
3. Stop Daedalus-Code at 1B rather than launch an unfinishable extension.
4. Never skip final artifact upload, baseline comparison, or reporting.

## Relevant Files

- `train.py`
- `daedalus/model.py`
- `daedalus/config.py`
- `daedalus/muon.py`
- `daedalus/qat.py`
- `daedalus/data.py`
- `daedalus/dataprep.py`
- `daedalus/supervise.py`
- `watchdog.py`
- `eval.py`
- `post.py`
- `daedalus/dpo.py`
- `export.py`
- `daedalus/publisher.py`
- `abl_arch.py`
- `scripts/decode_bench.py`
- `.env` (local secret source only; never read into reports, prompts, Git, or logs)

## Verification Summary

Automated:

- Focused pytest nodes for every new module and modified contract.
- Full test suite at the control-plane gate, before each expensive experiment family, and at finalization.
- Real llama.cpp QAT-grid, GGUF load, tokenizer round-trip, checkpoint-resume, and boot-resume tests.
- Deterministic scorecard schema validation and artifact hash verification.

Manual after the remote run:

- Review final measured report and negative results.
- Run supplied alternating Apple Silicon decode suite on final Q4 artifacts.
- Inspect a fixed prompt pack for general and code quality.
- Decide whether to publish QAT V1 and Daedalus-Code and whether V2 evidence justifies a future fresh run.
