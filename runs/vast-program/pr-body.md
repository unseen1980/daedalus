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
| 7 — Improved general corpus and mixture | **in progress**, steps 1–8 and 10 complete — mixture verdict **keep-baseline** in `runs/corpus/mixture-verdict-probe.json`, acceptance gate in `runs/corpus/phase7-gate.md`, headroom in `runs/corpus/headroom-curve.md`; step 9 demonstrated on a rebuilt source, open at full scale |
| 8 — Daedalus-Code | not started — step 1 is a branch switch, which the wrapper can now do (`branch`) but which needs the operator install below |
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

## Phase 7 — Corpus headroom: the ceiling is 53.8B, and it is one exhausted source

The phase's headline deliverable is a curve rather than a verdict on one budget, because the operator has not fixed a successor size and a month on one RTX 5090 reaches somewhere between 60B and 200B tokens. `scripts/source_headroom.py epochs` reports per-source epochs and the four-epoch shortfall across 30B/60B/100B/200B/500B/1T, plus the inverse every reader would otherwise compute by hand: the largest total budget each source can feed at the cap.

The corpus as built holds **17.15B unique tokens** — which is exactly where the released run's 59.9B budget and its ~3.5 epochs came from. Counting the files the stream never opened, the same ten sources reach **6,582B**. But only three ever bind:

| source | share | unique | supports a total budget of |
| --- | ---: | ---: | ---: |
| `everyday-conversations` | 0.020 | 0.0004B | **0.08B** — remove it, as the plan already directs |
| `stack-edu-python` | 0.090 | 1.21B | **53.8B** — out of documents |
| `finewiki-en` | 0.030 | 6.76B | 902B |
| everything else | — | — | past 1T |

So **the mixture supports ~53.8B tokens at four epochs**, and above that code is the only constraint needing a new source: a 9% share wants 4.5B unique code tokens at 200B and 22.5B at 1T against 1.21B today. Web, PDF and math need *re-streaming*, not new datasets. That makes phase 8's code corpus the growth path for a general successor and not only for Daedalus-Code.

Two things the numbers do not say. The **aggregate lies**: at 1T the corpus-level ratio is 0.2 epochs while three sources are over the cap, because a mixture cannot spend `dclm-baseline`'s 3,746B of headroom on `stack-edu-python`'s shortfall — so `corpus_epochs` is only ever reported beside `binding_source`. And **this phase's own mandate lowers these figures**: supply is counted after the released build's memory-bounded dedup, and persistent exact hashes with shared near-duplicate groups keep fewer tokens, not more.

Supply is measured, not assumed — realized tokens plus a lower bound on what is reachable in untouched files at the density the source itself achieved. Two silent traps are pinned by regression tests with the real numbers: a shard manifest's `total_tokens` describes the *fetched subset* while `subset_of` describes the whole build (479M against 5.19B for `fineweb-edu`), and a manifest with no stream position must never reach the density arithmetic, which would place it at file 0 and read `stack-edu-python`'s 1.2B as ~13B. Cross-checks: `fineweb-edu` measures 1,424B against a published ~1.3T, `dclm-baseline` 3,746B against a published multi-trillion corpus.

The curve exits 0 even when short. "This corpus does not support 1T at four epochs" is a result; a non-zero exit would mark the phase failed and make moving the bar the cheapest way to pass it.

## Phase 7 — Decontamination: 47.5% of scored items covered, now 100%

`daedalus/eval_index.py` builds the index once, sorted and content-addressed, beside the item counts, splits and revisions it came from; `run_dataprep --eval-index PATH [--eval-index-digest SHA]` loads it and records `decontam_index` in the corpus manifest. Reading in `runs/corpus/decontam-index.md`.

| | items indexed | scored split | covered |
| --- | ---: | ---: | ---: |
| `hellaswag` | 2,000 → **10,042** | 10,042 | 19.9% → **100%** |
| `arc_easy` | 2,000 → **2,376** | 2,376 | 84.2% → **100%** |
| `piqa` / `openbookqa` / `winogrande` | 3,605 | 3,605 | 100% |
| **total** | 7,605 → **16,023** | 16,023 | 47.5% → **100%** |

**1,371,773 13-grams against 214,682** — 6.39x from 2.11x the items, because the missing coverage was HellaSwag's and HellaSwag items are the long ones. Against what the released corpus was actually filtered with (183,359 grams, ARC-Easy and OpenBookQA on `validation`) it is **7.5x** larger and the first index built on the splits the model is scored on. The 214,682 reproduces `scripts/contam_scan.py`'s recorded figure for the same limit from independent code, which is why the larger number reads as a measurement rather than as the tool's opinion of itself.

The 2,000-item limit was the visible hole. The one underneath it is that **nothing recorded which index a source was filtered against**: `334c86c` moved two tasks onto their scored splits mid-build, and establishing which sources predate it meant rebuilding the index at the old splits and matching a gram count that happened to be in a log. That worked once; it is not a procedure. An index derived at run start is a function of what `datasets` returned that day and what `TASK_SPLITS` said that week, neither of which was written down — so it is now an input the build is *given*.

The refusals are the deliverable. `eval.load_all_tasks` skips an unavailable benchmark with a warning, which is right for scoring and wrong here: a HellaSwag outage yields an index that looks fine, filters nothing against HellaSwag, and leaves no trace in the corpus. `build_index` refuses a missing task, a task at zero items, a split other than the scored one, a limit — and a task that merely came back **short**, which none of the others can see. The Hub is read unauthenticated on this box, so a rate-limited split returns fewer items rather than failing; `EXPECTED_ITEMS` pins the five sizes so a truncated index cannot be built, marked complete, digested and used while its own provenance asserts the opposite. A partial index stops `run_dataprep` rather than being recorded. `coverage_problems` re-asks the same questions of a frozen file later, when a task may have been added or a split grown.

Two properties make the digest an identity rather than a timestamp: the file is sorted, and gzip's header carries neither mtime nor source filename — without the second, the same index written to two paths produced two different files, which `test_the_file_on_disk_is_a_function_of_the_set_alone` caught on its first run.

Cost to the rebuild, measured rather than assumed: **241 MB resident per dataprep worker**, against a 4.0 GB cap on a ~2.6–2.9 GB baseline, leaving ~0.9–1.2 GB of margin at `--max-workers 4`.

This says nothing about what the released corpus let through — that remains `contam_scan.py`'s measurement and is unchanged.

## Phase 7 — Mixture optimisation: keep-baseline, and what the arms did separate

**Verdict: `keep-baseline`** (`runs/corpus/mixture-verdict-probe.json`, page beside it). Both candidates are admissible — no floor violation, no source past the 5% regression bound — and neither clears the preregistered 0.5% minimum aggregate gain: quality-heavy 0.081%, derived 0.057%, against a 1.2486 baseline. That is the plan's instruction for a proxy that cannot separate the arms, recorded rather than resolved by advancing the best of a tie.

| arm | aggregate BPB | vs baseline | dclm-baseline | fineweb-edu | stack-edu-python |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 1.2486 | — | 1.3828 | 1.2441 | 0.9319 |
| `quality-heavy` | 1.2476 | +0.081% | 1.4035 | 1.2373 | 0.9005 |
| `derived` | 1.2479 | +0.057% | 1.3899 | 1.2490 | **0.8879** |

Per source they separate clearly. Both candidates buy code BPB — 0.9319 → 0.9005 → 0.8879, the best of the three going to the arm with the largest code share — and both pay for it on raw web, 1.3828 → 1.4035 and 1.3899. Under the blueprint weighting, which gives raw web 32.6% and code 13.0%, that trade is worth 0.06–0.08%. The tie is a measurement, not an accident of noise: the aggregate is dominated by the two web sources, and a mixture that moves mass toward code is buying the smallest of the three weights.



`daedalus/mixture_opt.py` is the decision half — arms, derivation, floors, selection — and it costs nothing to run, so all of it landed **before a single arm had been scored**. A rule that arrives in the same commit as the numbers it judges is indistinguishable from one fitted to them.

The mixture is now an argument of the run: `train.py --mixture-weight NAME=FRACTION` threads explicit shares into `MixtureBatchSource`, defaulting to `None` so every earlier run resolves its mixture exactly where it did. Every other way of varying a mixture — a second data root per arm, an edited `MIXTURE`, a patched loader — also varies something that is supposed to be held.

**Excess loss is measured against a specialist**: `bpb_baseline(s) − bpb_specialist_s(s)`, where the specialist is the same tiny model on the same budget trained on `s` alone. That is what this architecture at this scale can do with that source, so the gap is the part the mixture leaves unclaimed; every other reading of "excess" needs a quantity nothing on this box measures. A negative excess is kept as measured rather than clipped — a specialist re-reading a short source really can lose on that source's own holdout — and a 2× ratio cap bounds how far one such number can move the mixture.

| preregistered | value |
| --- | --- |
| derivation | blueprint share × `exp(excess / 0.10)`, clipped to 2×, then floored |
| domain floors | 0.4 of each floored domain's blueprint share, on `web-raw` / `math` / `code` |
| quality-heavy arm | raw web scaled to 0.45, freed mass over the filtered sources by blueprint share |
| selection | floors, then ≤5% per-source BPB regression, then lowest aggregate BPB — adopt only above 0.5% relative gain |

Floors are fractions of the blueprint rather than invented absolutes, so the same rule means the same thing over the three sources on this box and over all ten. `math` has no source here, so `unrepresented_floored_domains` names it in the artifact instead of letting a report claim three floors held over a corpus one of them never touched.

**Preflight refuses three failures the epoch-cap flag cannot tell apart**, all before the GPU is touched: a source the arm names that has no shards — which `resolve_mixture` renormalizes away and reports as *zero* skew, because `target_probs` is taken after that renormalization; a sampled mixture more than a thousandth of a point of L1 from the arm's own; and an arm that would re-read a source at or past the four-epoch cap, where the mixture is preserved and the repetition is what makes the arm unusable.

One trap worth stating because it would have quietly decided the phase: `train.py`'s in-run `val_bpb` weights the holdout by the mixture each run samples, which is right for one run and useless across arms — six models scored on six different corpora. The comparison is `scripts/bpb_eval.py` under one fixed weighting for every arm, written into the sweep artifact so it can be checked rather than assumed.

### How the arms were trained and read

The four reference arms (baseline plus one specialist per source) trained at phase 4's LM probe recipe re-used unchanged — `tok-probe-49152`, 200,015,872 tokens, 1,526 whole steps of 131,072 — so throughput, memory headroom and schedule shape are measured facts on this box rather than estimates. All four completed; `runs/corpus/mixture-sweep-reference.json` records each arm's sampled mixture beside the one it names, at 0.0 points of L1 skew and 0.26 epochs per source at the baseline.

`score` then re-measures every arm from its **final checkpoint, over every held-out window**, under the one fixed weighting — never the in-run `val_bpb` above. A specialist is scored on its own source alone, because excess loss reads it through exactly one number and the other two sources would answer a transfer question this phase never asks at 3× the GPU hours per specialist. `derive` applies the committed rule to those scorecards and refuses the inputs it cannot honestly subtract: a source whose specialist has not been scored (scoring it at 1.0 would leave that source at the blueprint while the artifact called itself derived), a sampled scorecard, cards measured against different holdout roots or context lengths, and a baseline scored on fewer sources than the mixture names. Every BPB in the derived record travels with the scorecard and the checkpoint digest it came from, so the weights can be recomputed from the artifact rather than trusted — and a scorecard is reused only while its checkpoint digest still matches, so a `--refresh` retrain cannot keep the old arm's number.

**A share the cap set is not a measured optimum, and the number alone cannot say which it is.** `stack-edu-python`'s 0.213 bits/byte of headroom asks for 8.4× its blueprint share; `EXCESS_RATIO_CAP` grants 2×. Read back from the derived arm's 0.178 code share, an ask of 2.1× and an ask of 84× are the same number. `cap_saturation` names the sources a bound decided, with the ask beside what was granted; `derive` records it and the verdict carries it onto the arm it qualifies. Disclosure only — every threshold here was committed before the first arm trained, so a saturated source does not become inadmissible and no cap widens. For the derivation the candidate arms were actually launched from, written before this field existed, the caveat is recomputed at read time from the excess loss it already carries: that artifact is the launch record the trained checkpoints are tied to, and rewriting evidence to add a field to it is worse than deriving the field from it. With no artifact at all the caveat is `null` rather than empty — "nothing here knows" is not "nothing saturated".

`report` is the last step of the chain and the one that states the answer: it reads the candidate stage's scorecards, hands them to the committed `select_mixture`, and writes the verdict beside a rendered page. Every aggregate is **recomputed here** under the fixed evaluation weighting rather than read off each card, so "one weighting for every arm" is true by construction and each card's own `bpb` becomes a free check on it. Four refusals guard the inputs, because each is a way the decision would be made from numbers that are not what they are labelled: a card that measured only part of the corpus (the aggregate is an average over the corpus and the regression check compares source by source — neither is defined on a subset); a card whose checkpoint trained on a *different mixture* than the arm now bearing its name, which `derive` being legitimately re-runnable makes reachable while the run directory, scorecard name and tag all still match; a card whose recorded aggregate disagrees with the recomputed one, which is what an equal-weighted card looks like; and a verdict built with no derived arm at all, which `candidate_arms` would otherwise render as a two-arm comparison describing itself as the comparison of three. Refused arms are rendered in the page rather than dropped — whether an arm was excluded by a floor, by a per-source cliff, or simply by losing is the most informative line in the artifact.

## Phase 7 — The acceptance list, decided from the corpus rather than asserted

`scripts/corpus_gate.py` reads phase 7's five acceptance claims off the files that decide them — the frozen n-gram index, the scan artifact, ten `manifest.json`s — and exits non-zero on a refusal, so a controller gates on it without parsing prose. Same discipline as the phase 2 gate, for the same reason: the phases downstream spend real budget on "the corpus is clean".

**The trap it is built around.** `l1_skew_pts` sees one of `cap_weights_by_epochs`'s two failure modes. When the epoch cap binds it reweights and the skew rises — visible. When *no* allocation satisfies the cap, the target shares come back unchanged and the skew is **0.00 by construction**, its best possible value, at the one budget where repetition is bounded by nothing at all. So the skew criterion carries the fallback guard with it, using `train.py`'s own discriminator: after a successful cap no source exceeds the limit, so `max_epochs_seen > max_epochs` is true precisely in the unbounded case. It is tested against a corpus whose skew is a perfect 0.0, and `runs/corpus/phase7-gate-as-built.json` is the guard firing on real data at 2,968 epochs.

**Supply is the built source, not the shards fetched here.** A shard directory on this box is a fetch and its manifest records what from in `subset_of`; counting the local files understates supply by 10×–30×, which made the first run of this gate report `fineweb-edu` at 46.9 epochs where the corpus gives 4.0. `--local-supply` asks the other question deliberately. The cap and the summary are `train.py`'s own, not a second implementation, so the gate cannot pass a corpus the trainer would refuse.

| criterion | at 59.9B | what decided it |
| --- | --- | --- |
| `decontam-index-complete` | **PASS** | 1,371,773 n-grams, five scored tasks at their scored splits, no per-task limit |
| `corpus-contamination` | **FAIL** | 1 `split_gap` doc, 1 `limit_gap` doc, and 157,921,561 `fineweb-edu` tokens the scan never read |
| `epoch-cap` | **PASS** | worst source 4.000 epochs, 7 of 10 pinned |
| `mixture-skew` | **FAIL** | 11.4593 pts against a 5-pt bound |
| `manifest-provenance` | **FAIL** | 10 of 10 manifests pin nothing |

**The criterion above could not have passed, and a rebuild is what proved it.** `manifest_provenance_verdict` read `source_release.resolved_commit`; `resolve_source_release` writes the served commit as `sha`. No writer produces that key, so the criterion was unpassable by construction — and it looked correct because the corpus it runs against carries no `source_release` at all, so it failed for the reason it was designed to and this sat underneath. The test agreed with the code and both disagreed with reality: the fixture manifest had been written to the criterion's expectation rather than to the writer's output. A 200,000-token rebuild of one source under the 32,768 tokenizer — full provenance on disk, `no source_revision and no resolved commit` in the verdict — surfaced it in the first minute it was pointed at a real rebuilt tree. `runs/corpus/phase7-gate-rebuild-smoke.json` is that tree after the fix, re-scanned with its own contamination artifact and handed back to the same gate: `manifest-provenance` **PASS** and `corpus-contamination` **PASS** — zero hits over 93.34% of the tree by tokens, which the criterion states as a 1.01e-2 upper bound on the document rate rather than as proof of zero. That is phase 7's claim that the two closable failures clear on a rebuild, measured rather than asserted. The other three criteria pass trivially on a one-source tree and are **not** evidence about a full rebuild: a single source at a budget it can fund cannot skew from its own target or exceed an epoch cap. The as-built verdict is re-run and unchanged — 11.4593 pts, 157,921,561 unscanned tokens, 10 of 10 pinning nothing — so no recorded outcome moves.

`docs_filtered` is **0** — the negative control — so `dataprep` removed everything it indexed; the two hits are the split gap and the limit gap this phase's frozen index already closes, and they clear on the rebuild rather than through more indexing. The provenance failure is the hole `source_provenance` was added to close and cannot be fixed retroactively on manifests that predate it.

Two findings are new. A scan artifact names no shard tree, so the contamination criterion now checks `per_source[].source_tokens` against what each source holds: nine of ten match exactly and `fineweb-edu` does not, leaving **3.0% of the largest source covered by a clean verdict that never read it**.

The skew is a measurement rather than a known gap. Swept across budgets, **the corpus as built delivers the blueprint within the 5-pt bound to ~55.4B, and to ~56.9B with the dialogue source dropped** — the released run's 59.9B is past both. Below the knee the entire 3.99 pts is `everyday-conversations`: 403,573 tokens cannot fund a 2% share, and dropping it gives a skew of **exactly 0.0000** at 30B and at 50B, which is the plan's step 4 measured rather than argued. Above the knee the same removal buys about 1.5B of budget, and 0.84 of the 6.46 pts it would need at 59.9B, because the freed 2% renormalizes onto sources already at the cap. That knee sits just above the 53.8B ceiling the headroom curve derives from `stack-edu-python` having no more documents, and the two are the same fact: past it the constraint is supply — step 5's top-up and phase 8's code corpus — not a source that can be removed.

This is an independent measurement of the phenomenon `daedalus/data.py::select_holdout_shards` records at 10.21 pts post-carve at 60B. The 11.4593 here is pre-carve; they are not the same number and should not be quoted as one.

**Step 9's prerequisite, and a provenance hole it exposed.** The step asks for shards rebuilt under the selected V2 vocabulary, and `dataprep` could only pack under SmolLM2 — while the manifests it wrote said nothing about which vocabulary produced their ids. Those are the same hole from two sides: a shard file is uint16 under every vocabulary, so reading a tree under the wrong one raises nothing, it trains and logs a loss and exports. `assert_manifest_tokenizer` is the guard and the corpus manifests gave it nothing to check. `--tokenizer` now picks the vocabulary and reaches the workers the way the mixture does — a module global inherited across the fork, rather than an argument threaded through the pool queue, the respawn path and the crash-recovery path, because a rebuild that silently fell back to the default would look exactly like one that worked. The fingerprint lands in `source_provenance` beside the filters and the served commit, for the same reason those are there: these directories are uploaded, hardlinked and read by four later phases without the corpus manifest travelling with them. The default stays `None` all the way into the worker, so every tree built so far — and the 49,152 one phase 8 continues from — is byte-identical. The acceptance gate is deliberately untouched: phase 7's criteria were preregistered, and this records rather than judges.

**A fingerprint nothing reads is half a guard**, so both readers now check it, each with the comparison it can actually make. `train.py` has no tokenizer — training reads ids and an embedding table — so it checks the row count, per source, before the sampler exists: a 49,152-row model reading 32,768-vocabulary shards indexes every id successfully and trains on text that means something else. `scripts/bpb_eval.py` does hold one, and it is part of the measurement rather than a label on it, so it checks the full fingerprint: BPB is nats per token converted through the bytes those tokens stand for, and the byte count comes from decoding them. Pointed at the rebuilt 32,768 tree with the default tokenizer, it now refuses before the model is built — `shard tokenizer mismatch on 'vocab_size' … packed by 'data/tokenizer-lab/tokenizers/v32768' (32768) but 'HuggingFaceTB/SmolLM2-135M' gives 49152` — where it would previously have returned a finite, plausible, wrong number. Both are inert for a manifest with no fingerprint, which is every tree on this box.

## Control plane

`daedalus/program_state.py` holds an atomic snapshot beside an append-only timeline; `scripts/vast_program.py` owns phases, one-process leases, and deadline gates; `scripts/boot_resume.py` resumes only an approved incomplete marker; `scripts/github_progress.py` publishes a sanitized heartbeat from an isolated worktree; `ops/vast/run-approved` is the only shell, Git, PR, and evaluation surface an engineering session may use, and it refuses default-branch pushes, PR merges, secret paths, and arbitrary shell fragments.

`daedalus/session_keeper.py` closes the gap that stopped the program before this branch: the controller owned phases, but nothing owned the Claude sessions implementing them. The keeper verifies both plan hashes before every launch, concatenates the verified plans into a mode-0600 system prompt, assigns and records the session id before launching so any death stays resumable, resumes the same session for bounded repair continuations, escalates to a fresh independent session, records a hard blocker instead of relaunching forever, yields the box while a supervised job holds it, opens the finalization phase at T+136h, and refuses to start beside an orphaned session.

Both failure drills pass and are recorded in the timeline: a session killed mid-turn is counted as a failure and relaunched, and a keeper killed with SIGKILL is restarted by supervisord without a duplicate session.

**One operator step is outstanding and now blocks three things**, including the start of phase 8: `bash ops/vast/install_supervisor.sh`. Sessions call `/usr/local/bin/daedalus-approved`, not the repository copy — deliberately, because a wrapper that changed whenever a branch edited a file would let a session widen its own permissions — so `pr-find` (how a session learns the PR number to apply this body), `reload-service` (why the heartbeat still omits the lanes field), and now `branch` (the only way a session can reach `vast/daedalus-code-20260824`, since it cannot run `git checkout` itself) all wait on it.

## Security fix worth reviewing

The workspace was never marked trusted, so Claude Code ignored `.claude/settings.json` in full — including the `deny` rules keeping an engineering session away from `.env`, the runtime credential directory, and the SSH material. It reports this only in a stderr warning that a non-interactive session shows nobody, so the control plane looked configured while enforcing none of it. `ops/vast/trust_workspace.py` records trust in the user-level config as an installed, tested step.

## Constraints held

- Unmodified stock llama.cpp remains the runtime target.
- No commit, push, merge, or rewrite touches the default branch.
- No model or dataset is published; experiment repositories stay private pending explicit approval.
- The live instance endpoint appears in no commit, log, or progress file.
