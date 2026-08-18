# How the 5B eval gets read, decided before it exists

Written 2026-08-10 ~09:15Z. Arm 1 is at step ~3,700 of 10,391; its 5B
checkpoint does not exist yet and will not be scored until ~16:35Z.

This exists for the same reason `runs/preflight/abl-arch-decision-rule.md`
does. That number goes into a $41.26 ask as the project's evidence that 40B
tokens will clear the quality bar, and it is the single number most likely to
be read into whatever shape the argument needs. So the reading is fixed first.

## First, the finding that changes what can honestly be claimed

`scripts/eval_noise.py`, new today, computes the part of the uncertainty that
*is* computable: each task is n independent Bernoulli trials, so accuracy has a
standard error of `sqrt(p(1-p)/n)`, and the 5-task mean's is `sqrt(sum)/5`.

| model | 5-task mean |
|---|---|
| us @0.5B | **40.22 ± 0.59** |
| GPT-2 124M | **42.17 ± 0.58** |

**The 2.0-point gap to the bar is 2.4σ of pure benchmark sampling noise, and a
difference has to reach 1.65 points to be two sigmas.** That is uncomfortably
close to the margins this project's success definition is written in — the
three named peers sit inside a **1.1-point band**:

| peer | 5-task mean | we must reach, for a 2σ claim |
|---|---|---|
| Pythia-160M | 41.0 | **42.65** |
| GPT-neo-125M | 41.9 | **43.55** |
| OPT-125M | 42.1 | **43.75** |

So "beat Pythia-160M, OPT-125M and GPT-neo-125M" is not a bar that a 42.5
finish would statistically support. It would support "matches them"; the
honest claim at 42.5 is a tie, not a win. **This does not change the plan** —
the same run produces whatever score it produces — but it changes what may be
said afterwards, and the operator should know it before spending, not after.

Two caveats, both stated in the tool's own output rather than only here:

- This is **benchmark sampling only**. It says nothing about seed variance,
  which at this scale is reported at 2–3 points and which nothing in this
  project has measured (`train.py` has no `--seed` flag; every run has used
  seed 0). The real uncertainty is larger than ±0.59, not smaller.
- Comparing two models on the **same items** is paired, so the unpaired σ above
  **overstates** the error of a difference. How much is unknowable without
  per-item outputs, which this harness does not keep. It is therefore an upper
  bound: a difference that clears it is real; one that does not is unresolved.

That second caveat is worth ~$1 to remove, and there is a free window to do it
in — see the bottom of this file.

## The pre-registered reading

Let `X` be the 5-task mean of the 5B checkpoint. Log-linear in tokens is the
convention, not a measured law (see the honesty note below), but it is what the
gate's slope argument implicitly assumes, so it is what gets pre-registered:
0.5B→5B is one decade, 5B→40B is 0.90 more, giving
`final ≈ X + 0.903 × (X − 40.22)`.

| `X` at 5B | what the gate will say |
|---|---|
| **≥ 42.1** | extrapolates past the 2σ bar for the OPT-class peers. Report as the strongest available evidence, still one noisy dot |
| **41.3 – 42.1** | on track for the 42.2 bar, **not** for a 2σ win over any named peer. This is the expected band |
| **40.8 – 41.3** | ambiguous: below what 42.2 needs, but inside ±1σ of it. Report as not-discriminating and let the decision rest on the rest of the case |
| **< 40.8** | discouraging — no measurable gain across a 10× token increase. Say so in the gate's decision box, not in a footnote |
| any task at chance (≈35.0 mean) | **stop**: that is the tokenizer/packing/eval bug this eval exists to catch, and it is worth far more than the $0.30 it costs |

**In every band the number is reported as `X ± 0.59` with the statement that
one 5B dot cannot resolve a 1-point trend.** Its 1σ interval spans 1.2 points,
which extrapolates to a ~2.2-point spread at 40B — wider than the entire gap it
is being used to argue about. That is a property of the measurement and no
amount of presentation fixes it.

### Where log-linear could be wrong, in both directions

Stated now so neither direction is available as an excuse later:

- **It could understate.** HellaSwag sits near its 25% chance floor at 0.5B
  (27.3) and floors compress early gains; models typically break away from it
  only after enough tokens. A 0.5B→5B slope measured mostly below the floor
  will underpredict what happens above it.
- **It could overstate.** Cross-model evidence on this harness does not support
  a clean token law at all: OPT at 180B scores **42.1** and Pythia at 300B
  scores **41.0** — 1.7× more tokens, 1.1 points *worse*. Recipe and data
  dominate budget at this scale, which is the whole reason this project bets on
  modern curated data, and it equally means a token-count extrapolation is a
  weak instrument.

Both of those were true before the number arrived, and they stay in the gate
whichever way it lands.

## The cheap upgrade — done the same hour, not queued

I first wrote this section as "queued for the idle window during the gate
wait". That was wrong sequencing and it is now landed instead: the **code**
costs nothing, and writing it before tonight's eval means the 5B run emits the
paired data as a by-product rather than needing a re-run afterwards.

`eval.py` now writes a per-item correctness sidecar beside `--out` — derived
from the output path rather than gated behind a flag, because tonight's eval
fires from a waiter script with nobody watching and a sidecar that appears only
when someone remembers a flag will not exist when the claim is made.
`scripts/mcnemar.py` compares two of them, refusing to pair tasks whose item
fingerprints differ (a `--task-limit` on one side would otherwise pair item 7
against a different question and return a confident, meaningless p-value).

What remains for the idle window is only the **GPU time**: re-scoring the 0.5B
checkpoint and the five peers under the new code, so every comparison in the
writeup can be paired. That is ~$1 of a box that would otherwise sit at 0% GPU
waiting for an answer to the gate.
