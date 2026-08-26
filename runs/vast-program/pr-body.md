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
| 4 — Tokenizer lab for V2 | **complete**, 32,768 selected, reading in `runs/tokenizer-lab/v2-tokenizer-migration.md` |
| 5 — ShortConv channel death prevention | **complete**, **no schedule selected**, verdict in `runs/conv-health/verdict-paired.json`, reading in `runs/conv-health/phase5-conv-decay.md` |
| 6 — Architecture Pareto proxies | **complete**, **no shape recommended and stage C is a no-go**, verdicts in `runs/architecture/stageb-recommendation.json` and `runs/architecture/stageb-stage-c.json` |
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

Per the preregistered stop rule the 300M follow-up and 1B escalation **did not run** — about 14 GPU-hours deliberately not spent.

All three arms are published **privately** and hash-verified to `Unseen1980/daedalus-qat-recovery-probes` (FP16 and Q4_0 GGUFs per arm, the lr 5e-4 arm in HF format, and the full evidence set). The card leads with the fact that no arm passed its gates — endorsement is a property of what the card says, not of the upload — and the artifacts exist so the two decisions below can be made against real files instead of a summary. The `.pt` checkpoints are deliberately not published: the preregistered 300M follow-up starts from the *released base* with `--init-from`, so no probe checkpoint is an input to anything.

Publishing surfaced a safety gap worth reviewing on its own: `publish_model`'s `repo_id` **defaulted to `Unseen1980/daedalus-150m`**, the shipped model. An experiment published correctly in every other respect and simply missing `--repo-id` would have overwritten the release. It now refuses a released-model target (and the released checkpoint repo) unless `--allow-released` is passed, normalizing case, whitespace and trailing slashes, and checking the target *before* the publishability check so a mistargeted upload reports the wrong target rather than a missing README. `verify_published` re-downloads into a fresh directory and hash-matches, because a successful upload means the request was accepted, not that the bytes match — a truncated GGUF loads far enough to produce numbers, and those numbers are wrong.

**Two decisions for the operator, both of which change a preregistered gate and so are not this branch's to make:**

1. Is a 0.5% FP16 perplexity limit the right gate for a *ship-format* recovery? As written it rejects a model whose Q4_0 artifact is 4.19% better and whose task mean is up. Scoring ship-format against ship-format (Q4_0 6.6873 vs released 6.9798) passes comfortably.
2. Is the 2–6 point MQAR decline acceptable, or does the recipe need a retrieval-safe variant first?

One preregistered deviation, made before any arm ran: the retrieval gate was re-baselined at 100 items per depth, because at the Phase 2 baseline's 10 items one item is 10 points and a 1-point gate could only ever be met by exact equality. Re-measuring showed the old baseline moving up to 8 points per depth on an unchanged model. 100 items makes the gate expressible, not statistically resolvable — noise is still ~3.6 points — and the evidence file says so rather than implying a precision the instrument lacks.

## Phase 4 — Tokenizer lab: 32,768 selected for V2

Rule digest `4557ac1d70b0f27be7f86395370cd41d`, written before any measurement. **A tokenizer cannot be transplanted into a trained model**, so nothing here changes released V1 weights or Daedalus-Code; this is a recommendation for a future from-scratch run.

24,576 fails the code clause (+0.09% code fertility against a 0% limit). 32,768 and 40,960 both clear all five clauses, and **32,768 is selected**: worst domain −0.72% (dialogue), code −3.04%, tiny-model BPB +0.038% against a 0.5% limit, embedding bytes −33.3%, full byte round-trip.

The honest caveat is the matched control. Against a 49,152-token vocabulary *retrained on the same sample*, 32,768 is 2.8–3.1% worse on fertility in every domain — so most of the gain against SmolLM2 comes from training the tokenizer on this corpus, not from shrinking the vocabulary. Both columns are reported; the rule is evaluated against SmolLM2 exactly as written.

## Phase 5 — ShortConv decay: no schedule selected

Read on the coupled `in_proj` × kernel × `out_proj` instrument rather than the shipped weight proxy, because the fix under test *is* a change to weight decay and a magnitude metric is satisfied as easily by an arm where nothing shrank as by one where nothing died. **Scope: V2 only** — no result here says a channel that collapsed during the 59.9B-token run was revived.

The stage is readable: the positive control (the shipped constant 0.1) dies at 53.9% in the paired 150M/500M-token escalation. Neither candidate clears the preregistered bar of dead fraction under 1%, alive-channel norms within 2× and a matched ablation that bites:

| arm | dead fraction | norms vs control | held-out loss | verdict |
| --- | ---: | ---: | ---: | :--- |
| `shipped-0.1` (positive control) | 53.9% | — | — | dies, as intended |
| `weak-0.0133` | 14.5% | 1.82–2.33× | +0.14% | fails norms, dead fraction, ablation |
| `weak-then-0.1` | 42.4% | 0.94–1.26× | −0.31% | fails dead fraction, ablation |

Weakening decay cuts the death substantially (53.9% → 14.5%) and pays for it in projection norms; ramping decay back up holds the norms and loses most of the benefit. Neither is a recipe, and the phase says so rather than promoting the better of two failures.

## Phase 6 — Architecture: no shape recommended, no stage C

Stage A screened fifteen shapes on BPB; stage B trained the four it advanced at 159M over 252M tokens and gated them on all five preregistered columns. **The gate recommends none**, and not on decode — decode passes for every arm (241–250 tok/s at depth 0, 208–226 at 2,048, artifacts 101–103 MB, measured in one alternating invocation on an idle box).

| arm | attn | KV B/tok | BPB vs control | retrieval | verdict |
| --- | ---: | ---: | ---: | :--- | :--- |
| `a8-kv4` (shipped control) | 8 | 8,192 | +0.00% | no power: 2 of 8 cells are MQAR at the floor | blocked on KV, over the 6,144 ceiling |
| `a6-kv4` | 6 | 6,144 | +0.07% | passkey d1024 **−20.0 pts** | blocked |
| `a4-kv4` | 4 | 4,096 | +0.26% | passkey d256 **−42.0 pts** | blocked |
| `a3-kv4` | 3 | 3,072 | +0.18% | passkey d256 **−38.0 pts** | blocked |

So at this scale cutting attention layers buys 25–62% of the KV cache and costs passkey retrieval by margins nothing in the gate can absorb. BPB, KV, export and decode all pass for the three candidates; retrieval is the column that decides, and it decides on passkey — MQAR is at the floor for every arm, which is carried into the V2 recommendation as a named gap rather than resolved by loosening the gate.

`stage-c` then answers **no-go** on two of three conditions: discrimination (0.26 BPB points across the arms, inside the 0.84 their 2.48% parameter mismatch alone could explain) and finalists (every non-control arm blocked). The deadline condition would have allowed it — 7.8 hours needed against 92.0 — so roughly 7 GPU-hours were deliberately not spent. The rule was committed before this table existed and no threshold moved after it.

One defect worth reviewing: the evidence chain wrote every stage's decode numbers to one shared report, and `run_decode` refuses to overwrite a report measuring models it does not — so the guard that protects a full sweep from a narrow rerun was certain to fire on the *second* stage to reach decode. Stage B's decode column was unmeasured and its phase marked failed for that reason alone. The report is now scoped by tag, beside the per-tag export and retrieval manifests, and the recommendation records which decode file it read.

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
