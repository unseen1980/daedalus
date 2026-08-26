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

This is a decision, not an implementation detail. The substitution arm is closed: the per-language candidates (`bigcode/the-stack-dedup`, `starcoderdata`) are gated, and an ambiguous licence is on the plan's hard-blocker list. So it is the redistribution arm, the way phase 7's `GATED_SUBSTITUTION_NOTES` redistributed Nemotron-CC — but taken per bucket rather than for all four at once, because the four are not equally out of reach and a blanket drop would throw away Go and Shell to save Rust.

## The mixture this revision can actually serve

`scripts/codeprep.py corpus plan` reads the directory listing and a 200,000-row probe of `all-all` and decides each bucket in one pass. The rate it divides by is measured **after** the licence gate — `all-all` admitted 897.7 MB from 200,000 rows, and the gate refuses about a third of the dataset at rates that differ by language, so the offered-rows histogram in the table above overstates every fallback bucket.

| bucket | plan | source | share | passes to reach plan |
| --- | --- | --- | --- | --- |
| python | 55% | `Python-all` | **64.4%** | — |
| javascript-typescript | 12% | `JavaScript-all`, `TypeScript-all` | **14.1%** | — |
| c-cpp | 10% | `C-all`, `C++-all` | **11.7%** | — |
| rust | 8% | — | **dropped** | 23.8× |
| go | 6% | `all-all`, `language=GO` | **2.8%** | 2.1× |
| java | 5% | `Java-all` | **5.9%** | — |
| shell-sql-other | 4% | `all-all`, `language∈{Shell,SQL}` | **1.1%** | 3.7× |

The budget is one pass: the fallback stream may admit as many bytes as the entire rest of the code corpus, which is already a large concession for 18% of the mixture. Under it Rust reaches 0.336% and is **dropped by name** rather than carried — a bucket at a third of a percent is a few million tokens, too little to teach the language and enough for a model card to claim it. Go and shell/SQL clear the 0.5% floor and are carried at what the yield reaches, capped. The 14.07 points the three cannot serve go proportionally to the four buckets with directories of their own; they cannot go to the capped buckets, which are capped *because* the rows are not there, and raising their target would only move the shortfall from a function that announces it to a build that does not.

Nothing here is unreachable in principle, only at a price, so every bucket's `required_passes` is recorded and the same measurement re-derives the mixture at any other budget without re-reading a row. Plan in `runs/codeprep/source-plan.json`; evidence in `runs/codeprep/github-code-configs.json`, `github-code-clean-configs.json`, `source-probe.json`, `directory-yield.json` and `allall-yield.json`.

### Whether the raised shares are there to be read

Redistributing moved 14 points onto Python. That fixes the shortfall only if `Python-all` holds 64% of the budget; otherwise it moves the shortfall from a bucket that announces it to one that does not — and phase 7 has already paid for that mistake once, when `stack-edu-python` came up 139M tokens short of its share and one metadata call would have said so before a document was streamed.

So `corpus headroom` asks phase 7's question with phase 8's evidence, reusing `source_headroom.epoch_curve`, its four-epoch cap and its verdicts. The supply is two measured factors and no assumed ones: the rate a directory admits bytes at, from a probe of its real rows, times the rows it holds, from its parquet footers — ten range requests per directory rather than a pass over gigabytes — over the tokenizer's measured **2.862 bytes/token of code** (phase 4's `49152-smollm2` fertility reading, not the ~4.0 that general text gives, which would understate every directory by 40%).

| bucket | share | rows | unique tokens | epochs at 3B |
| --- | --- | --- | --- | --- |
| python | 64.4% | 641,000 (`Python-all`) | 921M | **1.4** |
| javascript-typescript | 14.1% | 605,000 + 369,000 | 2,531M | 0.1 |
| c-cpp | 11.7% | 354,000 + 389,000 | 1,317M | 0.2 |
| java | 5.9% | 828,000 (`Java-all`) | 822M | 0.1 |
| go | 2.8% | 547,000 (`all-all`) | 19M | 2.3 |
| shell-sql-other | 1.1% | 547,000 (`all-all`) | 8M | 2.3 |

**SUPPORTED at all three gates** — 250M, 1B and 3B total tokens. Python at its raised share is read 1.4 times at the largest budget, well inside the cap, so the redistribution is sound rather than merely arithmetic.

One caveat is carried rather than hidden: `all-all`'s tenth file lost its footer to a 429 that outlasted its retries, so the two fallback buckets are counted from nine files of ten. Their supply is a **floor**, flagged as `lower_bound` in the record and on the report. It matters in one direction only — a floor that clears the cap has cleared it, but a floor that failed would have to be re-measured before being believed, so the two are distinguishable at the point the verdict is read. Record in `runs/codeprep/headroom.json`, footers in `config-rows.json`.

One trap worth carrying forward: a probe that finished, printed its verdict and wrote its JSON still exits `-6`. pyarrow aborts in `PyGILState_Release` during interpreter finalization, *after* everything is written, so the controller records the phase as failed and the return code says nothing about whether the work succeeded. Read the JSON, not the exit code. A long build will hit this too.

## The build

`scripts/codeprep.py corpus build` runs the plan through `dataprep.run_source` unchanged. The shard format, the O(1) stream-position resume, the per-source manifest and the memory discipline are the general corpus's; a code-specific shard writer would have been a second implementation of all four, and the RSS caps and resume semantics in that one were paid for over ten dataprep attempts.

Shards go to `<root>/train/<key>` and `<root>/holdout/<key>`, so each side is an ordinary mixture root — the shape `--data-dir` and `--holdout-root` already take, with one key naming both sides of a source the way `make_mixture_holdout_split` does. `C-all` and `C++-all` get distinct keys: they differ only in characters a naive slug throws away, and under one key the second source's shards would land in the first's directory and be read back as an interrupted run of it.

Three decisions are worth naming.

**A bucket's budget is divided across its directories on the supply already measured, not evenly.** `bucket_supply` summed per-directory unique tokens and kept only the sum, so an even split was the only division available — and the measurement says `TypeScript-all` holds **1,443M** unique tokens against `JavaScript-all`'s **1,088M**, and `C++-all` 830M against `C-all`'s 487M. An even split would ask the smaller directory for more than it has, and that shortfall would be reported per directory and invisible per bucket, which is the miscount the whole headroom pass exists to prevent. Where no supply was measured it still splits evenly and the basis says so, because a budget that was guessed and one that was measured must not read alike in a manifest.

**The holdout is capped, not sized as a share.** The holdout pass reads the whole directory and keeps the 2% of repositories on its side, so every holdout token costs roughly fifty tokens of streaming. At 3B total an uncapped holdout is ~39M tokens and ~2B tokens of streaming to collect them, to measure a BPB that a few million measure as well. Capped at 2M per bucket, and it scales down below the cap so a small build is not all holdout.

**A resumed source's gate manifest is merged only when the resume restored a stream *position*.** The replay fallback re-reads the prefix from row zero and `_DocumentStream` consults the gate on every replayed row before it skips it, so that attempt's manifest already covers the whole source; folding the previous one onto it would report a licence histogram twice the size of the corpus it describes. Merging is refused outright across a difference in split side, holdout fraction, salt or language filter — those counters are not addable.

Resume is the **default**, not a flag: a relaunch after a crash costs the remainder rather than the corpus, which is what makes `--max-attempts 3` on the controller mean anything. A source that stops short of the budget headroom said it could fill exits non-zero rather than leaving a log line under a corpus that will be read as whole. A source that trips the resident-memory cap stops the process rather than starting the next one in it — `_run_group_worker`'s rule, after the eighth dataprep incident, where a graceful cap trip was followed moments later by a raw C-level malloc failure that took every other source's in-flight progress with it.

The corpus is filtered against **`general | code`**: 1,406,059 13-grams, 34,286 from HumanEval+/MBPP+ and 1,371,773 from the five general tasks. Dropping the general half to add the code half would decontaminate against the benchmarks phase 8 added and re-contaminate against the ones it inherited; both records, with their digests, are in the corpus manifest.

A 25-document live smoke over all sixteen sources resolved every directory, admitted rows on both sides of every split, and wrote shards and manifests — and found one false record on the way out. `run_source` reads exhaustion off the stream ending, and under a document cap the stream ends at the cap, so the smoke manifested `Java-all` — a directory holding 0.82B unique tokens — as having no more documents. That is recorded as `stopped_by_doc_cap` now; an uncapped stream that really ends still says so, because that one is the finding.

**Running:** the corpus for the 250M and 1B gates — 1B total, so **650M code tokens** — is building detached under the controller (`phase8-code-corpus-build`, three attempts, log in `runs/codeprep/corpus-build.log`). The 3B continuation needs the same command at a larger `--total-tokens`, which resumes each source from where this one leaves it rather than rebuilding.

**One operator ask.** Every Hub read in this program is unauthenticated: the runtime environment carries `HF_TOKEN_WRITE` but not `HF_TOKEN`, which is the name `huggingface_hub` reads. It has already cost one measurement — the 429 that took `all-all`'s tenth footer and made two buckets a flagged lower bound — and it is a worse risk over a multi-GB streaming build. Not repaired here: the fix belongs in the approved wrapper, which is control-plane and lives on #14, and the installed copy is only refreshed by an operator run of `ops/vast/install_supervisor.sh`. Promoting a write-scoped credential into `HF_TOKEN` from inside a build script would put a secret in code. Adding a read-scoped `HF_TOKEN` to the runtime environment closes it.

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

Branch created from #14's tested SHA; parent recorded in `runs/vast-program/code-run-manifest.json`. Base baselines and both harness oracles are in, the code decontamination index is frozen and verified, and the admission gate is written and measured against the real sources.

The mixture decision is closed: the plan is measured, Rust is dropped by name, the remainder is redistributed, and every bucket's directories have been shown to hold the share the plan gives them inside the four-epoch cap.

The build is written, smoked against the real sources, and **running** for the 250M and 1B gates — 650M code tokens across sixteen sources, detached under the controller so it outlives the session that started it. No training has started. The next slice is the three 250M probes, which now have a corpus to read.
