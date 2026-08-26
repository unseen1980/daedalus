# Daedalus-Code V1 — continued pretraining from the released base

Stacked on #14 (`vast/daedalus-improvements-20260824`). This branch carries **only** Daedalus-Code work: code data preparation, code evaluation results, code training and post-training, and this model's documentation. Shared fixes go to #14 first and are merged forward, which is where the trainer's per-source BPB, the held-out preference gate, and the MBPP+ harness already landed.

Nothing here is merged automatically, and no artifact is published outside a private experiment repository.

## What phase 8 starts from

`hero-base-f16` — the released **base**, never the SFT/DPO checkpoint. The plan's mixture is 65% permissively licensed code (Python 55%, JS/TS 12%, C/C++ 10%, Rust 8%, Go 6%, Java 5%, shell/SQL/other 4%), 15% technical and mathematical prose, and 20% original general replay. Train and holdout split **by repository**, not by file or by packed window, and decontaminated against HumanEval+ and MBPP+ prompts, reference solutions, tests, and repository metadata.

It also inherits the released model's 49,152-entry tokenizer and its dead ShortConv channels. Phase 4 selected a 32,768 vocabulary and phase 5 studied decay schedules, but both are from-initialization recommendations for a future V2 and neither can be transplanted into these weights. That is why this is labelled **V1**.

## Base measurements, taken before any adaptation

| Measurement | Released base | Oracle on the same items |
| --- | --- | --- |
| HumanEval+ pass@1 | **0.0000** (0/164) | 1.0000 |
| HumanEval+ pass@1 (extended) | 0.0000 | 1.0000 |
| HumanEval+ syntax valid | 1.0000 | — |
| MBPP+ pass@1 | **0.0079** (3/378) | 1.0000 |
| MBPP+ pass@1 (extended) | 0.0053 (2/378; 1 problem ships no extended inputs) | 1.0000 |
| MBPP+ syntax valid | 0.2381 (289 of 378 generations do not parse) | — |

Checkpoint `cfbf27dc…e153` at `/root/daedalus/final/hero/checkpoint.pt`, seed 20260824, `max_new_tokens` 384, EvalPlus 0.3.1, 30s and 1GB per sandboxed test. Scorecards in `runs/eval/code-base/`, oracles in `runs/eval/code-base-oracle/`.

A 150M base scoring ~zero on execution is the expected result, not a broken harness — which is why both oracles were run first. The oracle feeds EvalPlus's own canonical solutions through the identical sandbox, timeout, memory cap and scorer and returns 1.000, so the base's 0.000 is the model's answer rather than the harness failing to run. Per-item outcomes are kept for every item, so post-training comparisons are paired rather than two aggregates.

The number to watch first is **MBPP+ syntax validity at 0.238** against HumanEval+'s 1.000. The base can continue a function body it is given; asked for a program from a prose description it mostly emits something that does not parse. That is the headroom continued pretraining is supposed to take, and it moves long before pass@1 does — which matters for a 250M-token probe, where a gate resting on pass@1 alone would read every arm as identical zero.

## Decontamination, before any corpus is built

`daedalus/eval_index.py` freezes the n-grams of the five multiple-choice tasks and contains nothing from HumanEval+ or MBPP+ — nothing was scored on them when it was built. Filtering a 65%-code corpus against it alone would decontaminate against the benchmarks this model is *not* judged by and leave the two it is. HumanEval and MBPP are public repositories, and their prompts, references and assertions are copied verbatim into tutorials, harness forks and solution sets: ordinary permissive Python that a code corpus ingests happily. The gate below reads "pass@1 improves over untouched base", and an unfiltered corpus can deliver that by memorisation.

So `daedalus/codeprep.py` freezes a second index, content-addressed the same way, built with `python scripts/codeprep.py decontam build`:

| | HumanEval+ | MBPP+ |
| --- | --- | --- |
| items | 164 | 378 |
| prompt | 9,133 13-grams | 5,341 (3 too short) |
| canonical solution | 2,587 (36 too short) | 2,686 (**197** too short) |
| reference (prompt + solution) | 13,458 | 11,790 |
| test | 9,955 | ships none; its assertion is inside the prompt |

34,286 13-grams, `sha256:67d21afc…5cdf78`, EvalPlus 0.3.1. Sidecar in `data/decontam/code-index-13gram.txt.gz.json`, verification record in `runs/codeprep/decontam-index.json`.

Two things are pinned rather than defaulted. **`n` is 13** because `DedupState.keep` calls `is_contaminated` at 13 over one set, so a code build filters against `general | code`; an index at any other length loads without complaint, unions without complaint, and matches nothing, so `code_coverage_problems` refuses it — as it refuses a general sidecar handed in as a code one. And **52% of MBPP+ canonical solutions are under 13 whitespace tokens** and cannot be filtered on alone. That is recorded, not fixed: `return min(x)` is three tokens that occur in every Python corpus ever built, and an index that matched them would empty this one rather than clean it. Every item is still covered through its joined reference, which is what a solutions repo actually contains, and no item is unfilterable outright — that case is a build refusal.

## The admission gate

Phase 8's corpus has two properties the general corpus never needed, both decided per row before anything is tokenized: every document is permissively licensed, and no repository appears on both sides of the train/holdout split. `daedalus.codeprep.RepositoryGate` is both, as the single `SourceSpec.filter_fn` the build already calls.

**The licence check is an allow-list, not a deny-list.** A deny-list fails open — it admits every string nobody thought of, including a `None` from an unpopulated column — and what results is a model whose training data cannot be described. `license_verdict` returns three answers rather than two: `permissive`, `non-permissive`, and `unknown`. Only the third is news, so it is counted separately and the offending strings are kept verbatim, because widening the list on a guess is the alternative.

**The split is a pure function of the repository name.** Not `hash()`, which is salted per interpreter: a `hash()`-based split re-partitions every repository the moment a long build restarts, and the resulting train/holdout overlap shows up only as a holdout that reads better than it should. It is `blake2b-64(salt\0repository) / 2**64`, pinned by a test that runs it in subprocesses under three `PYTHONHASHSEED` values. Names are lowercased because GitHub treats `Owner/Repo` and `owner/repo` as one project and blake2b does not. A row whose repository cannot be identified is refused rather than defaulted — there is no side to put it on, and if both gates guessed alike the same document would enter both splits.

Two gates over one source with `want="train"` and `want="holdout"` therefore partition it, which is what makes the holdout a second independent stream rather than a second reading of the first. The manifest records the salt and the fraction as well as the outcome, since the outcome alone is not re-derivable. The repository list is bounded at 200,000 names and then keeps counts only; an unbounded name set inside a `dataprep` worker is the growth its RSS caps exist to catch.

## What the sources actually contain

Every field name and licence string above was a guess about a dataset until a row of it was read, and each guess fails in the same silent direction: the gate refuses everything and the build writes an empty shard directory with a zero exit. So `scripts/codeprep.py corpus probe` reads real rows first. Across ten configs and 32,000 rows it confirmed `repo_name` is the repository field everywhere, and that **every licence string met is one the gate classifies** — no unknown values. Permissive yield: TypeScript 87%, JavaScript 83%, Python 62%, Java 56%, C++ 48%, C 42%. Realised holdout 1.3–2.1% of admitted rows against the 2.0% target.

It also found what the plan's mixture cannot have. `codeparrot/github-code`'s auto-converted parquet branch carries **19 directories, and no `Go`, `Rust`, `Shell` or `SQL` among them** — no near miss on case or spacing, so they were never converted rather than misspelled. `codeparrot/github-code-clean` is the same subset and additionally lacks TypeScript. The interleaved `all-all` directory does carry all 30 languages, but at rates that do not reach the shares:

| bucket | plan share | rows in `all-all` (of 20,000) |
| --- | --- | --- |
| Rust | 8% | 54 (**0.27%**) |
| Go | 6% | 331 (1.66%) |
| shell/SQL/other | 4% | Shell 248 (1.24%), SQL 122 (0.61%) |

Those four are 18% of the code portion. Reaching them from `all-all` would mean streaming roughly seven times the entire code budget, and `all-all` is a `partial-train` conversion of ten files that very likely does not hold that much. **Rust is the binding one** — Go, Shell and SQL are merely expensive; Rust is not reachable at share from this dataset at all.

This is a decision, not an implementation detail, and it is open: either substitute a per-language permissive source for the four buckets — the candidates (`bigcode/the-stack-dedup`, `starcoderdata`) are gated, and an ambiguous licence is on the plan's hard-blocker list — or redistribute their 18% proportionally onto the six buckets that resolved, the way phase 7's `GATED_SUBSTITUTION_NOTES` redistributed Nemotron-CC. Evidence in `runs/codeprep/github-code-configs.json`, `github-code-clean-configs.json`, `source-probe.json` and `all-all-probe.json`.

One trap worth carrying forward: a probe that finished, printed its verdict and wrote its JSON still exits `-6`. pyarrow aborts in `PyGILState_Release` during interpreter finalization, *after* everything is written, so the controller records the phase as failed and the return code says nothing about whether the work succeeded. Read the JSON, not the exit code. A long build will hit this too.

## Preregistered gates

Set before any arm runs, and not adjusted after seeing a number.

1. Three fully decayed **250M-token** probes from the same base checkpoint at Muon LR `5e-4`, `1e-3`, `2e-3`, proportionally matched Adam rates, identical data order and seed. Selected on code BPB and execution pass@1 subject to general retention. **Stop** if no arm improves code BPB by ≥2% or moves execution/syntax signal.
2. A fresh **1B-token** branch with the selected settings. Continue only if general BPB regression ≤1.5%, five-task mean drop ≤1 point, retrieval drop ≤2 points at every depth, and code metrics improve.
3. A staged **+2B** continuation from the 1B weights via `--init-from`, a lower LR and a fresh WSD schedule, only if the 1B gate passes and the controller projects completion before T+136h. Staged adaptation, reported as such — not one uninterrupted 3B schedule.
4. Code and general **SFT** on syntax-checked and execution-tested conversations.
5. **DPO** only if held-out preference accuracy *and* execution pass@1 improve; otherwise the SFT model is the instruct winner and DPO is recorded as rejected.
6. The winning **QAT** recipe applied to the final checkpoint, FP16 and Q4_0 exported, and every code/general/retrieval evaluation re-run from the immutable artifacts.

Final acceptance: code BPB improves ≥5%, HumanEval+/MBPP+ pass@1 and syntax validity improve over the untouched base, general full-pass BPB regression ≤1.5%, five-task mean drop ≤1 point with no single task dropping >2 points unreviewed, retrieval drop ≤2 points at every depth, and the Q4 penalty either meets the selected V1 QAT target or is reported transparently.

## Status

Branch created from #14's tested SHA; parent recorded in `runs/vast-program/code-run-manifest.json`. Base baselines and both harness oracles are in, the code decontamination index is frozen and verified, and the admission gate is written and measured against the real sources. No corpus has been built and no training has started — the corpus build is blocked on the mixture decision above, not on code.
