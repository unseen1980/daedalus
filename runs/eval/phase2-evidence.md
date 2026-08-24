# Phase 2 gate: what was measured, and what it cost to measure it

Written 2026-08-24 from the released artifacts on the Vast box. The machine-
readable verdict is `runs/eval/phase2-gate.json`, produced by
`scripts/gate_check.py` and recorded by the controller as phase `phase2-gate`,
status `passed`. This file is the reading of it, including the parts that did
not go to plan.

## The verdict

| criterion | verdict | what decided it |
|---|---|---|
| synthetic controls | pass | oracle-backed passkey, MQAR and copy-control all `exact_match == 1.0` |
| determinism | pass | three retrieval scorecards, two runs each, identical item digests and fingerprints |
| paired-quant identity | pass | 292 chunks each side, same ids in the same order, `paired_outcomes` accepted the pairing |
| code sandbox | pass | containment probes *run*, not read: see below |

The sandbox criterion executes its probes rather than reading a claim, because
the version of the sandbox shipped this morning would have passed any
read-the-scorecard check while a candidate could still shell out to `curl`.

## Q4_0 damage on the released model: +5.539%

| | perplexity (finewiki 150k, `-c 512`, 292 chunks) |
|---|---|
| `hero-base-f16.gguf` | 6.6135 |
| `hero-base-q4_0.gguf` | 6.9798 |
| **penalty** | **+5.539%** |

This reproduces the roughly 6% the plan expects, and the paired view is what
makes it usable: **285 of 292 chunks are worse, 7 better, none tied**. A penalty
carried by 98% of chunks is a different finding from one carried by a handful of
outliers, and only the paired per-chunk delta can tell them apart. Phase 3's
improvement gate — halve this — is measurable against it.

Worth stating plainly, because the number has been misread before: the "~6%" in
`runs/preflight/gguf-vs-pytorch-fidelity.md` is a statement that a *dirty eval
file* made Q4 damage read ~6% **low**, not a measurement of 6% damage. The two
happen to land near each other. This measurement is the direct one.

It is also much larger than anything measured during preflight, and that gap is
the interesting part. Every earlier figure came from 0.5B-token proxies:
+1.549% annealed (`q4-task-damage.md`), +1.558% and +2.576% non-annealed. The
released model, at 59.9B tokens, takes **+5.539%** — between two and three and a
half times as much. A better-trained model has sharper weights and more to lose
to a 4-bit lattice, so this is the expected direction, but the size of it means
the proxy numbers should not be used to size Phase 3's budget. QAT is
load-bearing here by a wider margin than preflight suggested.

## Retrieval baseline

Through PyTorch, on the released base checkpoint:

| task | exact match | d256 | d512 | d1024 | d2048 |
|---|---|---|---|---|---|
| passkey | 0.825 | 0.90 | 0.80 | 0.80 | 0.80 |
| MQAR | 0.950 | 1.00 | 1.00 | 1.00 | 0.80 |
| copy-control | 1.000 | — | — | — | — |

Copy-control at 100% says the prompt formatter is intact, independently of
model ability. The depth curve is flat to 2048 tokens, which is the property
Phase 6's KV-head candidates have to preserve.

**The llama.cpp retrieval path is not usable as an absolute measure on this
build, and the reason is not the model.** See below.

## Five tasks: 47.374 here against 47.313 recorded

| task | recorded | this host | Δ | Δ in items |
|---|---|---|---|---|
| HellaSwag | 37.9307 | 37.9108 | −0.020 | −2 of 10,042 |
| ARC-Easy | 50.4209 | 50.2104 | −0.211 | −5 of 2,376 |
| PIQA | 65.7780 | 65.9956 | +0.218 | +4 of 1,838 |
| OpenBookQA | 32.4000 | 32.4000 | 0.000 | 0 of 500 |
| WinoGrande | 50.0395 | 50.3552 | +0.316 | +4 of 1,267 |
| **mean** | **47.313** | **47.374** | **+0.061** | |

Nothing was substituted, and this is checkable rather than asserted:

- the checkpoint is the released one by digest (`cfbf27dc…`), the tokenizer by
  digest (`c191819e…`);
- item counts match exactly on all five tasks — no `--task-limit` in play;
- **all five per-task item digests match the recorded run**, so the same items
  were scored in the same order;
- two runs on this box produced **bit-identical per-item outcomes** across all
  16,023 items, so it is not run-to-run noise here either.

Same weights, same items, deterministic — and 15 items answered differently.

That residual is floating-point arithmetic, and it is demonstrated rather than
assumed. Scoring the **same 200 items per task on this one box**, changing only
the compute backend:

| task | CPU | CUDA | flipped |
|---|---|---|---|
| WinoGrande (`acc`, the headline) | 0.4600 | 0.4650 | 1 of 200 |
| PIQA (`acc`) | 0.6850 | 0.6800 | 1 of 200 |
| HellaSwag, ARC-Easy, OpenBookQA | — | — | 0 |

Two flips in 1,000 scored examples from a backend change alone. Extrapolated to
the 16,023-item suite that is ~32 expected flips, against the 15 observed
between hosts — the same order. Cloze scoring picks the argmax of a summed
log-probability, and candidates whose margins sit near zero change side under
any different reduction order. Nothing about the model moved.

**This is not a footnote, it is a constraint on Phase 3.** That gate is a
0.5-point five-task drop, and single tasks move up to 0.32 points between hosts
with the weights held fixed. Phase 3 must compare QAT checkpoints against
**this host's baseline (47.374)**, measured with this stack, and not against the
historical 47.313. Comparing across hosts spends two thirds of the gate's
margin on arithmetic.

## Four defects found by trying to use the evaluators

None of these were visible from the tests; all four needed the real artifacts.
Three of them scored rather than failed, which is the property that makes them
worth writing down: each one produced a confident number.

**The pinned `llama-cli` runs a chat UI.** Build `b1-7584430` has dropped
`-no-cnv`/`--no-conversation` in favour of `-st`/`--single-turn`, so the flag
probe found no way out of conversation mode and every item sat at an
interactive `> ` prompt until the 60 s timeout killed it. Closing stdin did not
help — the binary loops on EOF rather than exiting. Fixed by accepting `-st`,
refusing outright a binary that offers neither (with the binary's own chat-mode
help lines quoted in the refusal, so the next rename is one run to diagnose),
and recording the resolved flag set in provenance.

**The same binary prints its banner to stdout and ignores
`--no-display-prompt`.** Filtering known noise markers left `"Loading model..."`
as the recorded completion for *every* item — a confident zero that reads
exactly like a model which cannot retrieve. Fixed by locating the completion
after the echoed prompt instead of filtering around it, handling the elided echo
long prompts get, and **raising if banner text survives**: a changed UI must
fail the run rather than become the score.

**`pass@1` measured nothing at all.** `evaluate_problems` read
`problem.get("test", "")` and `problem.get("plus_test")`. EvalPlus ships neither
in that form: `test` holds HumanEval's original suite, which only *defines*
`check(candidate)` and never calls it, and there is no `plus_test` key — the
extended suite is inputs whose expectations come from the canonical solution. So
every candidate ran a program containing no assertions, exited zero, and scored
as a pass. **Measured: the released 150M base model scored `pass@1 = 1.0` on
HumanEval+**, while emitting function bodies that were nothing but a repeated
signature and docstring. Phase 8's entire code gate would have rested on a
metric that returns 1.0 for anything that parses.

The fixture was the reason it survived review: it invented `test` as a bare
assertion and a `plus_test` key that does not exist, so every test agreed with
the code and none touched the real schema. Fixed by calling `check(entry_point)`
for the base suite, building the plus suite as a differential test against the
executed reference (deep-copying arguments, so a candidate that mutates its
input cannot feed the reference something else), and removing the default
entirely — a missing suite now raises, because the empty string *was* the bug.
The same 20 problems now score `pass@1 = 0.0` with 19 assertion failures and one
exception.

**The code sandbox did not contain what the gate claimed.** It patched
`socket` and `urllib`, which stops code that asks Python politely, but the child
ran as **root**: `subprocess.run(["curl", …])` reached the network with every
block in place, and the mode-0600 credential files and released checkpoints were
readable. Fixed by dropping to an unprivileged uid — verified inside the child,
with the sandbox refusing to start if the drop did not take — handing the
per-item directory to that uid, and removing process creation outright
(`subprocess`, `os.system`, `os.popen`, the `exec`/`spawn`/`fork` families) under
its own failure category. Network namespaces are unavailable in this container
(`unshare` fails with `Operation not permitted`), so file permissions carry the
containment. Five regression tests fail on the previous behaviour.

## Known limits, carried forward

**llama.cpp retrieval is confounded by the chat template.** With conversation
mode unavoidable on this build, `-st` still applies a chat template to the
prompt; the same weights that score 82.5% on passkey through PyTorch score 0%
through `llama-cli`, and the completions contain a leaked `assistant` role
marker. The paired FP16-vs-Q4_0 *perplexity* comparison is unaffected — it runs
through `llama-perplexity`, which does no templating — and that is where the
quantization evidence comes from. GGUF retrieval numbers should not be quoted
as this model's retrieval ability, and the same caution applies to *any*
`llama-cli` generation on this build, including code: the baselines below were
taken through PyTorch for that reason. Whether the base export should carry a
chat template at all is a question for Phase 3's export work — a base model that
triggers conversation mode in stock `llama-cli` is arguably an export defect, and
it is the released artifact that does it.

**No held-out shards on this box, so there is no BPB baseline.** The released
tree holds `gguf/` and `final/` and no tokenized shards: `/root/daedalus/data`,
`/root/daedalus/shards` and `/root/daedalus/holdout` do not exist, and
`/root/daedalus/final` has no manifest-backed source directories. `eval-bpb` is
tested but unexercised on real data, and every scorecard written here correctly
records `bpb_mode: not-applicable` rather than implying a measurement that was
never made. **Phase 3 needs these**: its acceptance criteria include full-pass
BPB, and its FP16-retention gate is stated in perplexity terms that the current
evidence supports but its BPB terms are not yet measurable. Restoring or
rebuilding a holdout set is a Phase 3 precondition, not a Phase 7 nicety.

## What produced these files

| artifact | evidence |
|---|---|
| `phase2-gate.json` | the four criteria, each with its observed value |
| `quant-base/` | paired FP16/Q4_0 perplexity, per-chunk NLL sidecars |
| `retrieval-base-torch/`, `-repeat/` | retrieval baseline and its determinism repeat |
| `retrieval-control/` | oracle-backed control run |
| `baseline-hero-tasks.json`, `-repeat` | five tasks and their determinism repeat |
