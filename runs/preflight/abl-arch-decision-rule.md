# `abl-arch`'s decision rule, pre-registered before the numbers land

Written 2026-08-10 ~08:30Z, with arm 1 at step ~3,340 of 10,391 and arm 2 not
yet started. **Neither arm's val_bpb or decode speed is known.** The ablation
result lands ~06:45Z on the 11th and feeds the `[ASK HUMAN] ready for hero`
gate within minutes, so the rule for reading it has to exist first.

The precedent is this project's own: `sweep` had a tie rule written down before
its probes scored, and it fired — the winner beat the runner-up by 0.05%, the
rule said that is noise, and the blueprint's 0.02 was taken instead of a
noise winner. Without it the choice would have been made by looking at three
numbers and picking the smallest, which is how a 0.05% difference becomes a
$41 decision.

`abl-arch` has had no such rule. `hero.py --config` defaults to `daedalus-150m`
and its help text says "the abl-arch winner; **not decided by this script**"
(`hero.py:82`) — the decision is explicitly left to me, and until now it was
left to me *at the moment I would first see the numbers*.

## What is actually being decided

Two separate things, and conflating them is the trap:

1. **Which config `hero` trains** — a $41.26, 92-hour, unrecoverable choice.
2. **What the ablation reports** — the project's novel contribution, published
   as a Pareto claim on *both* quality and measured CPU decode.

(2) is a measurement and gets reported however it comes out. (1) is a decision
under uncertainty and needs a rule.

## The uncertainty is asymmetric, and only one half of it is measured

**Decode speed has a measured sigma.** `export.py`'s bench reports
`tok_per_sec_stddev` over repeated llama-bench runs, so "is this gap real"
is answerable from the data itself.

**Held-out BPB has none.** `train.py` has no `--seed` flag; `TrainArgs.seed`
is 0 for every run this project has ever launched (`train.py:662`), so all
three sweep probes and both ablation arms share one seed. We therefore have
**zero measurement of seed variance in bits-per-byte at any scale**, and the
blueprint's second seed was deferred (`README.md:180`).

Two consequences worth stating precisely, because they point in opposite
directions:

- `eval_val_bpb` is a deterministic function of a checkpoint over a fixed
  holdout, so a 0.1% gap is a **real** difference between these two particular
  training runs. There is no measurement error to speak of.
- Whether that gap is a property of the **architecture** rather than of one
  initialization draw is exactly what is unmeasured.

So the floor below cannot be "is the difference real". It is "is the difference
large enough to act on given that its sigma is unknown".

## The floor, and where it comes from — inherited, not measured

`QUALITY_NOISE_FRAC = 0.005` (0.5% relative BPB).

Its provenance is `check_sweep.NOISE_FRAC`, whose own comment says half a
percent "is comfortably inside seed variation at this scale" — an **assertion,
not a measurement**, and it was calibrated for 0.5B-token lr probes, where the
arms differ by a hyperparameter. `abl-arch` arms are 5B tokens (10× more, so
plausibly quieter) and differ by architecture. Reusing the constant without
saying so would be exactly the transcription this project has been bitten by
twice.

It is therefore **deliberately not imported** from `abl_arch.NOISE_FRAC`, and no
test pins the two equal, because pinning them would pin a coincidence rather
than a shared quantity.

The only calibration I have for scale: across the sweep's 4× lr range at 0.5B
tokens, val_bpb moved **0.434%** (1.091783 → 1.087067). So 0.5% is roughly
"larger than a 4× learning-rate change" — a demanding bar, chosen to be
conservative in the direction of not acting on noise.

**The rule is built so that a wrong floor is cheap.** The threshold decides
whether the operator is *asked*, never whether the config silently switches.
If 0.5% is too large the cost is one unnecessary paragraph in the gate issue;
if too small, the same. Nothing silently changes on a knife edge.

For decode, `DECODE_SIGMAS = 2.0`: the gap must exceed two combined standard
deviations, `sqrt(sigma_h^2 + sigma_d^2)`. That one is measured.

## The rule

Let `dq = (dense_bpb - hybrid_bpb) / hybrid_bpb` — positive means the hybrid is
better. Let `R = hybrid_decode / dense_decode` — above 1 means the hybrid is
faster.

| case | `hero` trains | why |
|---|---|---|
| either arm missing `val_bpb` | **hybrid** | the ablation did not decide it; blueprint default stands, and the gate says so |
| `\|dq\| < 0.5%` (tie) | **hybrid** | same discipline as the sweep tie rule — a tie goes to the blueprint. It is also ~$5.0 cheaper and 11 h faster to train, and it is the arm the CPU-decode claim needs |
| hybrid wins by ≥ 0.5% | **hybrid** | unambiguous; the Pareto claim is clean |
| **dense wins by ≥ 0.5%** | **escalate — operator decides** | this is the case that must not be decided quietly, in either direction |

### Why a dense quality win escalates rather than auto-switching

Because the two halves of the mission then disagree, and only the operator owns
that trade:

- The confirmed success definition is *"beat Pythia-160M, OPT-125M and
  GPT-neo-125M on quality; concede SmolLM2-135M on quality while beating it
  **decisively on CPU decode**"*. CPU decode is not a nice-to-have; it is the
  only axis on which the strongest peer is beaten at all.
- Switching costs real money, priced now rather than at the gate: dense trains
  at **0.88848×** the hybrid's rate (preflight, 100,561.4 vs 113,183.6 tok/s at
  micro-batch 16 — `runs/preflight/{dense,daedalus}-150m-b16.json`). Against
  arm 1's measured steady state of 122,612 tok/s, `hero` on dense is
  **~103.0 h / $46.25** versus **91.9 h / $41.26** — **+$4.99 and +11.1 h**,
  from a balance with no slack.
- It also puts the four-day run on the thinnest memory margin in the plan:
  dense peaked at 29.55 GB of 32.6 in the preflight, ~28.4 GB once corrected by
  the 4.0% conservatism that arm 1 measured — ~13% spare for 92 unattended
  hours, against the hybrid's live 24.29 GB.

So a dense quality win is a genuine finding that costs $5, 11 hours, headroom,
and the headline claim. That is an operator decision, presented with numbers,
not an agent one.

### And if dense decodes *faster* on CPU

Then the project's central premise is falsified, and that becomes the headline
of the writeup rather than a footnote. Pre-registering this so it cannot be
softened later: the conv-hybrid exists in this design *because* it is the
CPU-fast option (`DAEDALUS-BLUEPRINT-v6.md:31`). If the measurement says
otherwise, the ablation has done its job and the result is the negative one.

## Prediction, written down first

So that reading the real numbers tomorrow is a comparison and not a
rationalisation:

| | prediction | basis |
|---|---|---|
| quality | **tie** (`\|dq\| < 0.5%`), hybrid very slightly ahead | LFM2's finding is that the hybrid is *not worse*; at 5B tokens on identical data and one seed I expect a gap in the 0.1–0.4% band |
| decode | **hybrid faster, R in 1.3–2.0×** | short conv is cheaper than GQA attention at 2048 ctx on CPU; the earlier decode work measured **2.08×** on this box, which is the number to beat |
| net | `hero` trains **`daedalus-150m`**, no escalation | |

If quality comes out a tie and decode confirms a large hybrid win, the ablation
is the clean publishable result the project was designed around. **The rule
above is what makes that claim worth anything — it was fixed before the data.**

## What this changed in the code

`scripts/abl_table.py` `verdict()` declared a winner on *any* gap, however
small: `better = "hybrid" if hb < db else "dense"`. At a 0.02% gap it would
have printed "**dense wins on held-out BPB**" into the operator-facing gate
issue and the writeup, with no hint that the number is inside its own unmeasured
noise — the identical defect the neighbouring `check_sweep.py` already fixed for
the lr grid. Now it calls a tie a tie, applies the rule above, and prints the
`hero` config it implies.
