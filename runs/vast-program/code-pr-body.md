# Daedalus-Code V1 — continued pretraining from the released base

Stacked on #14 (`vast/daedalus-improvements-20260824`). This branch carries **only** Daedalus-Code work: code data preparation, code evaluation results, code training and post-training, and this model's documentation. Shared fixes go to #14 first and are merged forward, which is where the trainer's per-source BPB, the held-out preference gate, and the MBPP+ harness already landed.

Nothing here is merged automatically, and no artifact is published outside a private experiment repository.

## What phase 8 starts from

`hero-base-f16` — the released **base**, never the SFT/DPO checkpoint. The mixture is 65% permissively licensed code, 15% technical and mathematical prose, and 20% original general replay. Train and holdout split **by repository**, not by file or by packed window, and decontaminated against HumanEval+ and MBPP+ prompts, reference solutions, tests, and repository metadata.

The code portion is **Python 55% and JavaScript/TypeScript 45%**. The plan preregistered seven buckets; two separate decisions reduced them to two, and the record keeps them apart because they are not the same kind of fact — one is a measurement, the other is a choice. Both are in `runs/vast-program/code-run-manifest.json` as amendments beside the preregistered split, which is retained rather than overwritten.

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

## The mixture the user asked for

Everything above is the mixture this dataset *can* serve. What it builds is narrower, by direction: **the code portion is retargeted to Python and JavaScript/TypeScript only**, and `c-cpp`, `java`, `go` and `shell-sql-other` are dropped.

That drop is **not** the Rust drop and the record must not read as though it were. Rust could not be served — no converted directory, 0.336% of the interleaved one, 23.8 passes to reach its share. These four *were* reachable: C/C++ and Java have their own directories, Go and shell/SQL cleared the 0.5% floor from the interleaved directory, and the headroom pass above verified every one of them SUPPORTED at all three gates. They were not asked for. Conflating a scope choice with an availability constraint would misdescribe both, so `code-run-manifest.json` carries two amendments with two authorities, and `PREREGISTERED_CODE_LANGUAGE_SHARES` stays in the module beside the live table as the thing the drops are drops from.

The freed 20 points go to **JavaScript/TypeScript, not to Python**. The measured plan had drifted Python to 64.4% precisely because every unreachable bucket was non-Python and the shortfall redistributed proportionally — and both gate benchmarks are Python-only, so that drift was a reporting hazard rather than a bonus. Directing the freed share elsewhere returns Python to exactly its preregistered 55% and removes the eval-language concentration rather than deepening it.

Re-measured, not assumed:

| bucket | share | unique tokens | epochs at 3B |
| --- | --- | --- | --- |
| python | 55% | 921M (`Python-all`) | **1.2** |
| javascript-typescript | 45% | 2,531M (`JavaScript-all` + `TypeScript-all`) | **0.3** |

**SUPPORTED at 250M, 1B and 3B**, both cheaper than the plan they replace. The `all-all` lower-bound caveat leaves with the fallback buckets: neither remaining bucket is drawn from the directory whose tenth footer was lost.

The fallback and language-filter code stays and stays tested — the plan tests pin themselves to the preregistered seven-bucket table, the only mixture where that logic has anything to decide. Deleting it along with the buckets would discard the measurement that proved Rust unreachable.

One trap worth carrying forward: a probe that finished, printed its verdict and wrote its JSON still exits `-6`. pyarrow aborts in `PyGILState_Release` during interpreter finalization, *after* everything is written, so the controller records the phase as failed and the return code says nothing about whether the work succeeded. Read the JSON, not the exit code. A long build will hit this too.

## The build

`scripts/codeprep.py corpus build` runs the plan through `dataprep.run_source` unchanged. The shard format, the O(1) stream-position resume, the per-source manifest and the memory discipline are the general corpus's; a code-specific shard writer would have been a second implementation of all four, and the RSS caps and resume semantics in that one were paid for over ten dataprep attempts.

Shards go to `<root>/train/<key>` and `<root>/holdout/<key>`, so each side is an ordinary mixture root — the shape `--data-dir` and `--holdout-root` already take, with one key naming both sides of a source the way `make_mixture_holdout_split` does. `C-all` and `C++-all` get distinct keys: they differ only in characters a naive slug throws away, and under one key the second source's shards would land in the first's directory and be read back as an interrupted run of it.

Three decisions are worth naming.

**A bucket's budget is divided across its directories on the supply already measured, not evenly.** `bucket_supply` summed per-directory unique tokens and kept only the sum, so an even split was the only division available — and the measurement says `TypeScript-all` holds **1,443M** unique tokens against `JavaScript-all`'s **1,088M**. That matters more after the retarget than before it: JS/TS is now 45% of the code portion, so the split inside it decides 293M tokens at the 1B gate, and an even one would ask the smaller directory for more than it has. The shortfall would then be reported per directory and invisible per bucket, which is the miscount the whole headroom pass exists to prevent. Where no supply was measured it still splits evenly and the basis says so, because a budget that was guessed and one that was measured must not read alike in a manifest.

**The holdout is capped, not sized as a share.** The holdout pass reads the whole directory and keeps the 2% of repositories on its side, so every holdout token costs roughly fifty tokens of streaming. At 3B total an uncapped holdout is ~39M tokens and ~2B tokens of streaming to collect them, to measure a BPB that a few million measure as well. Capped at 2M per bucket, and it scales down below the cap so a small build is not all holdout.

**A resumed source's gate manifest is merged only when the resume restored a stream *position*.** The replay fallback re-reads the prefix from row zero and `_DocumentStream` consults the gate on every replayed row before it skips it, so that attempt's manifest already covers the whole source; folding the previous one onto it would report a licence histogram twice the size of the corpus it describes. Merging is refused outright across a difference in split side, holdout fraction, salt or language filter — those counters are not addable.

Resume is the **default**, not a flag: a relaunch after a crash costs the remainder rather than the corpus, which is what makes `--max-attempts 3` on the controller mean anything. A source that stops short of the budget headroom said it could fill exits non-zero rather than leaving a log line under a corpus that will be read as whole. A source that trips the resident-memory cap stops the process rather than starting the next one in it — `_run_group_worker`'s rule, after the eighth dataprep incident, where a graceful cap trip was followed moments later by a raw C-level malloc failure that took every other source's in-flight progress with it.

The corpus is filtered against **`general | code`**: 1,406,059 13-grams, 34,286 from HumanEval+/MBPP+ and 1,371,773 from the five general tasks. Dropping the general half to add the code half would decontaminate against the benchmarks phase 8 added and re-contaminate against the ones it inherited; both records, with their digests, are in the corpus manifest.

A 25-document live smoke over all sixteen sources of the pre-retarget plan resolved every directory, admitted rows on both sides of every split, and wrote shards and manifests — and found one false record on the way out. `run_source` reads exhaustion off the stream ending, and under a document cap the stream ends at the cap, so the smoke manifested `Java-all` — a directory holding 0.82B unique tokens — as having no more documents. That is recorded as `stopped_by_doc_cap` now; an uncapped stream that really ends still says so, because that one is the finding.

One more cadence defect the smoke exposed. `run_source` writes its durable checkpoints every N *yielded* documents, and a holdout pass yields about 2% of what it streams — so at the shared 50,000-document default the whole 2M-token Python holdout yields ~800 documents and never checkpoints once. The pass with the most streaming behind each token it keeps was the one no crash could be resumed from. It cannot simply be made small either: every checkpoint closes the buffer as a *new shard file*, so a 500-document cadence over a 357M-token source would leave hundreds of shards behind. It is derived from each source's own budget now — about twenty checkpoints per source, never coarser than 200 documents — and `--progress-every` follows the same rule, because a detached multi-hour source that prints nothing between its start and its finish cannot be told from one that has hung.

**Built.** The corpus for the 250M and 1B gates — 1B total, so **650M code tokens** — finished in 40 minutes under the controller (`phase8-code-corpus-build`, one attempt, exit 0, log in `runs/codeprep/corpus-build.log`), detached so it outlived the session that started it:

| side | source | tokens | of budget |
| --- | --- | --- | --- |
| train | `code-python` | 357.5M | 100.0% |
| train | `code-javascript-typescript-javascript-all` | 125.7M | 100.0% |
| train | `code-javascript-typescript-typescript-all` | 166.8M | 100.0% |
| holdout | the same three, split by repository | 4.1M | — |

**650.0M train tokens of a 650.0M code budget.** The 3B continuation is the same command at a larger `--total-tokens`, which resumes each source from where this one left it rather than rebuilding.

An earlier launch of this build was stopped by the operator two minutes in, and it is worth recording why rather than quietly relaunching: it was streaming the **old seven-bucket mixture**. The retarget landed at 20:13:31Z and the build went out at 20:18:28Z, from a session that had read the controller state at the top of its turn and not again before starting a six-hour job. The build code itself needed no change — `corpus build` consumes whatever `corpus plan` produced — so the fault was ordering, and the cost was the 4.0 MB that attempt had written, removed rather than resumed.

## The mixture a probe actually reads

The corpus above is 65% of an arm. The other 35% is the original pretraining data, already on this box, and replaying it is the entire mechanism by which the general retention gates below are meant to be passable. `train.py`'s `resolve_mixture` reads **one** root and looks for `<root>/<source>/manifest.json` per weight, so the two corpora have to be reachable under a single directory before `--data-dir` can name them both. `scripts/codeprep.py corpus mixture` is that directory — a farm of symlinks, so no token is copied and the composed root stays a *view* of corpora that are still being written.

| bucket | share | sources | read from |
| --- | --- | --- | --- |
| code | 65% | `code-python` 0.3575, `…javascript-all` 0.1257, `…typescript-all` 0.1668 | `data/code-shards/train` |
| technical | 15% | `finepdfs-edu` 0.0857, `finemath-3plus` 0.0321, `infiwebmath-3plus` 0.0321 | `data/shards` |
| general replay | 20% | `fineweb-edu` 0.0974, `dclm-baseline` 0.0584, `finephrase` 0.0182, `cosmopedia-v2` 0.0130, `finewiki-en` 0.0078, `everyday-conversations` 0.0052 | `data/shards-train`, else `data/shards` |

Four decisions, none of them free.

**The replay side is read off `dataprep.MIXTURE`, not retyped.** "Original general replay" means the distribution the released model was pretrained on and there is exactly one record of that. A second copy would drift invisibly — an arm training on a replay mixture that no longer matched the model it was replaying for, with every number it produced still looking reasonable.

**`stack-edu-python` leaves the replay bucket.** It is 9% of the original mixture and it is GitHub code from `codeparrot/github-code` at the revision the code bucket streams. Counted as general replay it would put its share on top of the code share — 66.8% code in a corpus whose manifest, model card and gate all say 65%.

**The three carved sources are read from `data/shards-train`, not `data/shards`.** `data/shards` still holds the windows `data/holdout` scores, and the retention gate reading them decides whether 1B tokens get spent. The other seven were never carved — one shard each, nothing to hold out — so they resolve to `data/shards` and are scored by nothing. Five of the twelve have a holdout at all: the three code buckets, split by repository by the build, plus `fineweb-edu` and `dclm-baseline`. The seven that do not are recorded rather than quietly absent.

**A bucket's missing source moves share within its own bucket.** 65/15/20 is the preregistered quantity, so a directory that was never built must not quietly reweight prose against code.

Measured at both budgets it will be read at, with `train.py`'s own resolver rather than a second implementation of it:

| budget | l1 skew | most repeated | code sources |
| --- | --- | --- | --- |
| 250M (one probe) | **0.00 pts** | `everyday-conversations`, 3.22 epochs | 0.25 epochs each |
| 1B (the branch) | **0.72 pts** | `everyday-conversations`, capped at 4.00 | **1.00 epoch each** |

Only `everyday-conversations` ever caps, and it is 403,573 tokens — the whole dataset, which is why phase 7 recommends dropping it from a future general corpus. At 1B the code sources are read exactly once, which is what the corpus was sized for. Record in `runs/codeprep/train-mixture.json`, including the twelve `--mixture-weight` flags to train with.

**One caveat carried to the scoring slice, not fixed here.** `data/holdout/stack-edu-python` is GitHub Python from the same dataset and revision the code bucket streams, and the code corpus is repository-split against *its own* holdout only. That source therefore cannot serve as a general-retention measurement for a model being trained on code — it can be improved by the very training the gate is meant to constrain. The ≤1.5% general BPB bound will be read over the general-text sources separately from it. It is in the mixture record's `caveats` so the scoring cannot quietly inherit it.

**An unfinished build is refused rather than composed**, and it took two checks to mean it. A source that is present and short is the easy half. The half the first live dry run found is a source that is *not there at all*: `corpus build` appends to its manifest as each source finishes, so the directory it has not started is absent rather than short — and the bucket check passes anyway, because the bucket's other directory is there. It composed a 74/26 python-to-javascript mixture at weights that said 55/45. The manifest's own `code_tokens` closes it: the train budgets are a partition of it, so a short sum names the gap in tokens even though nothing in the file is missing a value.

**One operator ask.** Every Hub read in this program is unauthenticated: the runtime environment carries `HF_TOKEN_WRITE` but not `HF_TOKEN`, which is the name `huggingface_hub` reads. It has already cost one measurement — the 429 that took `all-all`'s tenth footer and made two buckets a flagged lower bound — and it is a worse risk over a multi-GB streaming build. Not repaired here: the fix belongs in the approved wrapper, which is control-plane and lives on #14, and the installed copy is only refreshed by an operator run of `ops/vast/install_supervisor.sh`. Promoting a write-scoped credential into `HF_TOKEN` from inside a build script would put a secret in code. Adding a read-scoped `HF_TOKEN` to the runtime environment closes it.

## Preregistered gates

Set before any arm runs, and not adjusted after seeing a number.

1. Three fully decayed **250M-token** probes from the same base checkpoint at Muon LR `5e-4`, `1e-3`, `2e-3`, proportionally matched Adam rates, identical data order and seed. Selected on code BPB and execution pass@1 subject to general retention. **Stop** if no arm improves code BPB by ≥2% or moves execution/syntax signal — read as set out below.
2. A fresh **1B-token** branch with the selected settings. Continue only if general BPB regression ≤1.5%, five-task mean drop ≤1 point, retrieval drop ≤2 points at every depth, and code metrics improve.
3. A staged **+2B** continuation from the 1B weights via `--init-from`, a lower LR and a fresh WSD schedule, only if the 1B gate passes and the controller projects completion before T+136h. Staged adaptation, reported as such — not one uninterrupted 3B schedule.
4. Code and general **SFT** on syntax-checked and execution-tested conversations.
5. **DPO** only if held-out preference accuracy *and* execution pass@1 improve; otherwise the SFT model is the instruct winner and DPO is recorded as rejected.
6. The winning **QAT** recipe applied to the final checkpoint, FP16 and Q4_0 exported, and every code/general/retrieval evaluation re-run from the immutable artifacts.

### What gate 1 actually says

"No arm improves code BPB by ≥2% **or** moves execution/syntax signal" has two readings, and they spend a 1B-token budget differently. **Loose:** stop only when every arm fails both criteria, so one arm moving the execution signal alone continues the run. **Strict:** stop when no arm clears the BPB bar *or* when no arm moves the execution signal, so continuing needs both. The wording is settled now, before any arm has produced a number, which is the only time it can be settled without being an adjustment to a result.

**The reading is loose**, for the reason the execution signal is in the gate at all. This model's base scores 0.000 pass@1 on HumanEval+ and 0.008 on MBPP+, so a probe gated on pass@1 alone reads every arm as identical zero, and MBPP+ syntax validity at 0.238 is the headroom that moves first. The signal was added as the *more sensitive alternative* to a BPB bar a 250M-token probe may not clear. Under the strict reading, adding a more sensitive signal makes the gate harder — and would stop a run whose code BPB improved 5% without moving Python syntax validity. That inverts the purpose of the addition.

The cost is stated rather than left implicit: the loose reading is easier to pass than a BPB-only gate would have been. What keeps it from being passed by noise is the second half, which the wording never had — **"moves" had no bar**, and an unbounded "moves" is satisfied by one item of 378, or 0.26 points. A gate that cannot return no is not a gate; this is the defect the DPO preference gate was repaired for two days earlier in the same phase.

| criterion | bar | what it is in items |
| --- | --- | --- |
| A — code BPB | ≥2.0% relative improvement on the held-out code split | — |
| B — syntax validity | ≥2.0 points absolute | ~8 of MBPP+'s 378; HumanEval+ is 1.000 at base and can only fall, so this half is MBPP+'s in practice |
| B — pass@1 or pass@1 (extended) | ≥1.0 point absolute | ~2 of HumanEval+'s 164, ~4 of MBPP+'s 378 — above one item, so a single lucky solve cannot authorise a 1B-token run |

`daedalus/code_gates.py::probes_250m_verdict` is the rule, and it refuses rather than scores where a comparison would not be evidence: a non-finite BPB, a zero base BPB, an arm scored on different benchmarks than the base, a differing item count (the `--task-limit` failure `scorecard.paired_outcomes` already refuses), no arms, and two arms under one name. Regressions are recorded with their size and never count as movement — a falling syntax validity is how an arm unlearning Python announces itself, but it is not evidence for spending 1B tokens.

It does **not** pick the winner. `select_on` reads "subject to general retention" and no retention bound is preregistered for the *probe* stage — only for the 1B branch — so the verdict ranks the qualifying arms by code BPB, breaks ties on pass@1 movement, and names `best_before_retention`. Applying retention stays with the caller holding the general-side scorecards. Inventing a probe-stage retention bound would have been a second preregistration decision nobody asked for.

### How the three arms are run

`scripts/code_probes.py sweep` is gate 1's launcher. Everything that is not the learning rate is built once — a test asserts the three argvs differ in exactly **three** arguments — because "identical data, order and seed" is a claim three shell lines cannot support and a diff of three argvs can. The seed is `TrainArgs`' default 0 for all three; `train.py` has no `--seed` flag to disagree about.

**One sweep, not three phases.** The box has one GPU and the controller holds one lease per lane, so three separately detached phases would not queue behind each other — the second would be *refused* while the first ran, and the box would sit idle between turns waiting to be asked again. This is the shape phases 5 and 6 used.

**Restart-safe in both directions**, which is the property phase 4 lost an arm to. Each arm goes through the existing watchdog, halt marker and `run_with_resume`, so a crash *inside* an arm resumes it rather than restarting it; a relaunch of the *sweep* skips the arms that reached their budget and resumes the one that was in flight. Completion is read off `metrics.jsonl` rather than off the checkpoint — the checkpoint is written throughout a run, so skipping on it would skip the very arm the relaunch existed to resume.

Three refusals happen before any GPU time is spent: a base checkpoint that does not hash to the pinned released one (every arm's result is a difference against that file); a composed root missing a source (`resolve_mixture` renormalizes over what it finds, so a root that lost half its sources trains a healthy-looking arm on a mixture no artifact describes); and a mixture whose epoch cap makes it a different experiment at 250M. A shortened smoke must carry its own `--tag`, so it cannot land in the gate's run directory and be resumed as the real arm or read as one that finished at a budget nobody chose.

Final acceptance: code BPB improves ≥5%, HumanEval+/MBPP+ pass@1 and syntax validity improve over the untouched base, general full-pass BPB regression ≤1.5%, five-task mean drop ≤1 point with no single task dropping >2 points unreviewed, retrieval drop ≤2 points at every depth, and the Q4 penalty either meets the selected V1 QAT target or is reported transparently.

### What gate 2 actually says

Three of its four clauses came with their own numbers — general BPB regression ≤1.5%, five-task mean drop ≤1 point, retrieval drop ≤2 points at every depth — and are implemented as written. The fourth, **"code metrics improve"**, has no threshold, no metric list and no *and*/*or*. It is settled here for the same reason gate 1's wording was: the branch has produced no number yet, and that is the only time a threshold can be set without being an adjustment to a result.

**Code BPB improvement is required, at the 250M stage's own 2% bar; execution is checked for regression only.** Gate 1 is a `stop_if` with an explicit "or"; this is a `continue_if` with none, so a clause listed beside three hard retention bounds is a requirement rather than one of two ways to pass. Under the disjunction, a 1B run with no BPB movement that shifted MBPP+ syntax validity by two points would authorise a further **2B tokens**.

**The bar is borrowed, not invented.** The only two preregistered figures adjacent to this gate are 2.0% (the 250M stage) and 5.0% (final acceptance). Borrowing the final bar would require the 1B midpoint to already meet the end-of-program bar, which inverts the plan's staged 1B → +2B → SFT design. Borrowing the previous, smaller stage's bar is the conservative direction: a 1B run that cannot clear what 250M had to clear has not moved forward. This is the same rule applied when the probe stage had no retention bound of its own — reuse an adjacent preregistered number rather than write a new one.

**"Metrics" is plural, so BPB alone under-reads it** — but demanding strict improvement in HumanEval+ pass@1, which this base measures at 0.000, would stop the gate on an artifact of the base rather than on the branch. So every execution metric is checked for a *fall* past its own preregistered movement bar: a two-point rise in MBPP+ syntax validity is preregistered as meaningful, so a two-point fall is too. Movement upward is reported without being sufficient on its own.

The cost is stated rather than left implicit: **this is stricter than gate 1.** A branch that improved execution strongly while leaving code BPB flat is stopped by it, and that is a real outcome this reading would call wrong. It is accepted because the plan's degradation policy names stopping Daedalus-Code at 1B as an acceptable outcome and an unfinishable extension as the expensive mistake.

`daedalus/code_gates.py::branch_1b_verdict` is the rule. `BranchScore` subclasses `ProbeScore`, so the code half of gate 2 is literally the code gate 1 runs; its three general-side fields default to *not measured* rather than to a number, because a retention gate that passes when the evaluation did not run is the one failure it exists to catch. An absent measurement is a **failed clause with its reason**; non-comparable evidence — different benchmarks, a differing item count, a non-finite BPB — **raises**, as it does at 250M. Retrieval is gated per depth and never on the aggregate (a model that loses one depth and gains another nets out flat while being worse at the thing the clause protects), and a depth present in the baseline but missing from the branch **fails** rather than being skipped: otherwise "at every depth" quietly becomes "at every depth we happened to measure", which is exactly how a long-context regression goes unseen. Retrieval is keyed `<task>:d<depth>` as `qat_recovery.collect_observation` already writes it, so one collector feeds phase 3's retention gate and this one.

## Status

Branch created from #14's tested SHA; parent recorded in `runs/vast-program/code-run-manifest.json`. Base baselines and both harness oracles are in, the code decontamination index is frozen and verified, and the admission gate is written and measured against the real sources.

The mixture decision is closed twice over: the plan is measured and Rust is dropped by name for what the revision could serve, and the code portion is then retargeted by the user to Python and JavaScript/TypeScript, with the two kinds of drop recorded apart. Both remaining buckets are verified SUPPORTED at all three gates — 1.2 and 0.3 epochs at 3B against a cap of 4.

The corpus is **built**: 650.0M train tokens of a 650.0M budget across six sources, every one at 100%, with 4.1M of by-repository holdout beside it. The training mixture is **composed and measured** — twelve sources under one root at 65/15/20, 0.00 points of skew at 250M and 0.72 at 1B, where the code sources are read exactly once.

Gate 1's wording was settled while the corpus was still streaming, with no arm in existence to settle it in favour of. Its launcher is written, tested, and smoked end-to-end: three four-step arms under their own `code-smoke-*` names trained from the released base on the composed mixture, wrote per-source BPB over all five holdout sources, and a relaunch skipped all three as already complete rather than retraining them.

**The three 250M probes are done.** All three reached 250,085,376 tokens at step 477 in ~1.41h each, one attempt apiece, zero skipped non-finite updates, final training loss 1.6159 / 1.6003 / 1.5966 at Muon `5e-4` / `1e-3` / `2e-3`. `runs/code-probes/probes.json` carries each arm's argv, supervisor history and mixture preflight.

**Scoring them is the pass now running** — `phase8-probe-scoring`, detached, log in `runs/code-probes/scoring.log`. `scripts/code_probe_report.py` measures full-pass code BPB and general-replay BPB for the untouched base and each arm, runs HumanEval+ and MBPP+ where a card does not already exist, and writes `probes_250m_verdict` to `runs/code-probes/verdict.json`. The base's own code BPB is in: **0.5871** mixture-weighted over the three code languages (Python 0.5212, TypeScript 0.5963, JavaScript 0.7624), a full pass, not a sample.

Two BPB numbers per checkpoint, never one. The gate is a trade — code down, general held — and a blended figure reports their sum, so a code gain and a replay regression cancel in the one number that would decide the phase. The buckets come from the mixture record's own `buckets` block rather than a name prefix, so there is exactly one definition of which source is code. `code-bpb` covers 100% of the code bucket; `general-bpb` covers **77.9%** of the general-replay bucket, because only `fineweb-edu` and `dclm-baseline` of its six sources are in this holdout — and that fraction is written into every card as `details.bucket_share_covered` so the number cannot later be read as the whole replay distribution. `stack-edu-python` is refused by name as general retention, per the mixture record's own caveat: it is GitHub Python from the dataset the code bucket streams, so scoring it as replay would credit code training as retention and the number it produced would be finite and plausible.

**Gate 2's launcher is written and waiting on that verdict.** `scripts/code_branch.py` turns the selected arm into the 1B run as one command, and it is mostly refusals, because every way that command goes wrong produces a run that trains, exits 0 and reports numbers nobody can interpret. A gate that said stop cannot be launched anyway, and one that continued without naming an arm has no rate to launch at. `--init-from` is the released base, hash-pinned, and a path inside a probe arm's run directory is refused *by name* — continuing an arm would be a staged 1.25B run reported as a fresh 1B one, against a base its numbers are differences from. The rate comes back from `probe_arms()` by name, so a hand-edited verdict cannot smuggle in a rate no arm ran. Warmup is recomputed for the larger budget — 95 steps, not the probes' 23 — and the mixture is re-preflighted at 1B, because the epoch cap moves shares with the budget and a clean 250M preflight does not imply a clean 1B one. `--estimated-hours` is measured off the probes' windowed `tok_per_sec`, slowest arm, rather than typed from memory into a deadline reserve: at the ~49.7k tok/s the three arms ran at, 1B projects to about 7h.

The supervised checkpoint is asked of `train.py` rather than composed from a run root. `train.py` has no `--run-dir` flag, so it always writes `runs/<run-name>/checkpoint.pt` relative to its own working directory; a composed path would have handed the supervisor an in-flight marker beside a file that never appears, and every relaunch would then have restarted from step zero without saying so. Supervision itself is the launcher's `--supervise-checkpoint` and `--watchdog-tokens` — this is the single long run with no orchestrator that capability was added for — and the launch is detached, so it outlives the session that starts it. A guarded test runs the whole plan against the real corpus and the real base checkpoint: the 1B draw sits inside the 5-point skew limit and the base still hashes to the pinned digest, so the branch can launch as preregistered the moment the verdict lands.

**The probe-stage retention bound is 1.5%**, named in the commit that precedes the first score. `code_gates` deliberately stops short of retention because no probe-stage bound is preregistered, and inventing one after three arms are measured is the one thing that must not happen — so this reuses the **1B branch gate's own** bound instead of writing a new one. An arm already past the 1B gate's retention bound at 250M tokens is not a candidate on which to spend 1B tokens. It can only ever remove an arm the loose gate already qualified, never add one.

**Gate 2's rule is now written too**, in the same window and for the same reason its launcher was: the branch has produced no number, so the one clause that arrived without a threshold could still be settled honestly. `branch_1b_verdict` reports every clause with the number that decided it — 51 focused tests, 158 across the code-gate, probe-report, probe-launcher, branch-launcher and scorecard suites. What is deliberately *not* written yet is the collector that assembles the general-BPB, five-task and retrieval scorecards into it; the 1B run gives about seven hours in which to write that against real card paths rather than guessed ones.

One defect the probes exposed and did not depend on: `train.py`'s in-run validation was gated on `step % val_every_steps` but only ever reached on the metrics cadence, so both moduli had to agree and `--val-every-steps 250` beside the default 20-step metrics interval validated every 500 steps — never, in a 477-step arm. All three arms reported `val_bpb: null` for their whole length. No preregistered number depended on it (every gate figure comes from the out-of-band pass above), but the live early warning it exists for could not fire. Validation is now due on elapsed steps, so no cadence is silently unreachable; the regression test is nine steps at metrics 3 / val 4 and fails on the old condition.
