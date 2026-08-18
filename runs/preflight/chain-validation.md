# The overnight chain, validated step by step before it runs unattended

The chain fires at ~01:20Z and runs ~26 h of jobs with nobody watching:
`dataprep` exits → `sweep` (~3.5 h) → batch preflight → `abl-arch` (~25 h,
~$11.4). Every step below was exercised for real on 2026-08-09 ~20:20-21:00Z,
alongside the live `dataprep`, at flat memory (20 GB available throughout).

| step | how it was checked | result |
|---|---|---|
| dataprep → sweep handoff | chain waits on the *process pattern* over 3 consecutive checks | armed, `/tmp/chain-sweep.log` |
| sweep source selection | read `data/shards/fineweb-edu/manifest.json` for `total_tokens` | 2.21B ≥ the 1.5B threshold, so it probes the real corpus, not the 300M stand-in |
| sweep's holdout split | `make_holdout_split` on the real `fineweb-edu` | 40 shards / 2,132.5M train, 2 / 82.0M holdout; `eos_id` and `dtype` carried into both manifests |
| **`sweep.py` end to end** | `run_sweep` with 3 lrs, tiny budget, `--micro-batch 2 --max-steps 2` | **exit 0**; 3 probes trained, scored and written |
| the chain's own gate on `best.json` | applied verbatim to the produced file | `best_lr=0.04` accepted; `winner_at_grid_edge` and `git_commit` present |
| batch preflight | run against the tiny preset (`a14c988`) | clean value on stdout, exit 0 |
| `dense-150m` trains | optimizer split + one real step, CPU | 100% param coverage, all finite |
| abl-arch's mixture split | `make_mixture_holdout_split` on the real `data/shards` | 9 sources, `everyday-conversations` skipped (1 shard), `uploaded.json` ignored, hardlinked (df unchanged) |
| abl-arch's val step | real checkpoint over the real 609M holdout | works; **priced at ~38 min/arm** |
| export → GGUF → llama-bench | `dense-150m` dry run (`1e6022c`) | Q4_0 verified end to end |
| stale state | checked for pre-existing split roots and run dirs | none — nothing for the chain to silently reuse |

## What this did *not* prove

- The sweep validation used **2 optimizer steps on random tokens**, so its
  three val_bpb values are identical to 5 decimal places. It validates the
  *orchestration* — split, three subprocesses, scoring, `best.json` shape, the
  chain's gate — and says nothing about which lr is actually best. That is what
  the real 3 × 0.5B run at ~01:20Z is for.
- Both smoke runs used `--micro-batch 2`. The real jobs run at the micro-batch
  the preflight chooses (16/12/8), where memory behaves differently — which is
  exactly why the preflight exists and measures rather than assumes.
- `post.py` **is** now covered, separately: run end to end on 2026-08-09
  ~22:00Z with production flags. See `post-smoke.log` and the commit; five
  silent bugs, two of which would have shipped the wrong model.
- `abl_arch.py` **as a whole script** is now covered too (`abl-arch-smoke.log`,
  `abl-arch-smoke-results.json`), against an isolated synthetic mixture under
  `--run-prefix abl-smoke`. Both arms trained, evaluated, exported to Q4_0 and
  benched; exit 0. What that does *not* prove is the real budget: 60k tokens
  per arm at micro-batch 2, so it exercises orchestration, not throughput or
  memory at the preflight-chosen micro-batch.

## Artifacts removed after checking

`runs/sweep-validate-lr*` (1.4 GB each), `runs/hero-smoke`, `runs/valfix-smoke`,
`/tmp/ablsplit-probe`, `/tmp/sweepsplit-probe`, `/tmp/hero-smoke`,
`/tmp/sweep-val`, and the two smoke lines `hero.py` prepended to `STATUS.md`.
Deliberate: a leftover `runs/sweep-wsdfix-*` or split root is precisely what
would make the real chain skip training and score stale weights.
