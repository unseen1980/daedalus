# paper/

IEEE-format write-up of Daedalus-150M.

| file | what it is |
|---|---|
| `daedalus.tex` | the paper, IEEEtran conference format — the deliverable |
| `daedalus.pdf` | built output, if present |

## Building

```bash
pdflatex daedalus.tex && pdflatex daedalus.tex   # twice, for cross-references
```

Requires `IEEEtran.cls` (Debian/Ubuntu: `texlive-publishers`). On macOS,
MacTeX includes it. Or paste `daedalus.tex` into Overleaf and select
"IEEE Conference" — no local install needed.

## What is deliberately empty

Section "Headline Results" is **PENDING**. The 59.9B-token `hero` run had not
finished when this was written, and a projected number does not belong in a
results table. Everything else is measured and traceable to an artefact listed
below.

## Where each claim comes from

| Claim | Artefact |
|---|---|
| Ablation quality, decode, quantised sizes | `runs/abl-arch/results.json` |
| Decision rule, fixed before scoring | `runs/preflight/abl-arch-decision-rule.md` |
| Decode methodology and confounds | `runs/eval/decode-hybrid-vs-dense.md` |
| Peer scores and the 42.2 bar | `README.md` |
| Corpus shares | `daedalus/dataprep.py` (`MIXTURE`) |
| Dead-channel plateau | `runs/preflight/conv-channel-death.md` |
| Pruning infeasibility | `runs/preflight/conv-prune-feasibility.json` |
| Deviations, resumption, QAT failure | the paper's own "Deviations and Threats to Validity" section |

## Numbers that must not be quoted

Two superseded decode figures appear in this project's history. **1.29×** came
from a non-alternating measurement that does not reproduce. **1.15–1.17×** is
real but is the depth-0 row — a floor, not the result. The supported figures are
in the paper's decode tables.
