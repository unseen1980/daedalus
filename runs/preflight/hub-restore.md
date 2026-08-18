# Checkpoint durability: verified live against the real Hub

**2026-08-09.** Closes the hard precondition on `hero` — that a lost instance
must not cost four days and ~$43.70 with nothing recoverable.

Repo: **`Unseen1980/daedalus-checkpoints`** (private, model repo).

## What was missing

`train.py` wrote one rolling `runs/<RUN_NAME>/checkpoint.pt` every 30 min and
nothing else. Hub upload machinery existed only for *dataset shards*
(`daedalus/data.py`, `daedalus/shard_uploader.py`); no model checkpoint had ever
been uploaded, which contradicts AGENT.md §0.2 ("Never store state only on this
box... If it isn't pushed, it doesn't exist") and §0.4.

## What exists now

| | rolling | milestone |
|---|---|---|
| when | every ~2 h, and immediately on the first step | once, at the WSD decay-start step |
| contents | weights only, bf16 | weights **and** Muon + AdamW state, fp32 |
| measured size | **321.0 MB** | **1435.3 MB** |
| destination | `rolling/<run>/weights.pt` on branch `rolling` | `milestone/<run>/checkpoint.pt` on branch `<run>-stable-end-step<N>` |
| purpose | disaster recovery | reusable branch point |

Transfers happen in a separate torch-free process (`daedalus/ckpt_uploader.py`),
so a slow or hung link costs no training time — the same reason
`shard_uploader.py` exists. The trainer only ever does a local write plus a
sidecar; the sidecar is what tells the uploader a payload is complete, so a
`torch.save` still in flight can never be pushed as a truncated checkpoint.

`--resume` now also accepts `hub://owner/repo/path?rev=branch`, resolved to a
local file before loading, so restoring from the Hub takes the *same* code path
as restoring from disk rather than a separate branch that only ever runs on the
worst day.

## The live check

`runs/preflight/hub_restore_live.py`, run against the real Hub with real
network. The offline suite fakes `HfApi`, so it cannot exercise LFS (files over
10 MB take a different upload path), real branch creation, or `hf_hub_download`.
Results in `hub-restore-live.json`:

| phase | what | result |
|---|---|---|
| A | train `daedalus-150m`, stage + upload rolling bf16 and a milestone | 1.76 GB uploaded in **58.8 s**, outbox drained to 0 |
| B | restore from `hub://...?rev=rolling` into a **clean directory**, resume, keep training | step and `tokens_seen` restored exactly; worst relative weight delta **0.0039** (= bf16 precision, 2⁻⁸); training continued |
| C | milestone on its own revision, restored | `lr_mult_at_branch` **1.0**, restored at the exact decay-start step, **34** Muon entries came back with momentum state |
| D | branch isolation | `rolling` and the milestone revision both exist and are distinct |

Phase A ran while `dataprep` was streaming, so ~240 Mbit/s effective is a
pessimistic figure, and a 321 MB copy every 2 h is far inside the link.

**Correction, 2026-08-10.** This section previously said the smoke branches and
files "were deleted afterwards" and that the repo "now holds only `main` and
`rolling`". Listed against the real repo, that is false on both counts:

```
branches: main, rolling, mixsmoke-stable-end-step1, e2ewb-stable-end-step2
rolling:  .gitattributes, latest-milestone-hubsmoke.json,
          rolling/abl-arch-daedalus-150m/weights.pt,
          rolling/e2ewb/weights.pt, rolling/mixsmoke/weights.pt
```

Operationally harmless — every real revision is named `<run>-stable-end-step<N>`,
so `hero`'s branch point is unambiguous next to debris called `mixsmoke` — but
this document is evidence for a $41.26 gate, so it should say what is there
rather than what was intended. The debris is left in place deliberately: it costs
nothing, and deleting Hub branches is not worth doing unattended. Cleanup belongs
with `publisher.py` at ship time.

What the same listing does establish, against the repo rather than local
bookkeeping: **`rolling/abl-arch-daedalus-150m/weights.pt` is genuinely on the
Hub**, which is precondition #1 confirmed from outside this box.

## Tests

Offline, in the default suite:

- `tests/test_ckpt_uploader.py` — 25 tests: sealing, truncated-payload refusal,
  supersede, branch creation, per-failure retry semantics, pointer failure not
  losing a checkpoint, and `watch` surviving an exploding pass.
- `tests/test_train.py` — the Hub block, notably
  **`test_restore_from_hub_end_to_end`** (stage → upload → download into a clean
  directory → resume → keep training),
  `test_a_bf16_weights_only_checkpoint_still_resumes`,
  `test_milestone_written_at_decay_start_with_optimizer_state`,
  `test_milestone_is_written_once_across_a_restart`, and
  `test_milestone_step_is_exactly_the_wsd_decay_start`.

Two bugs the tests caught before any run depended on them: the exit drain made a
live network call with no token guard, and the suite would have hit the real Hub
on the operator's account because this box exports `HF_TOKEN_WRITE` in
production shells (now an autouse fixture that deletes it).

## Which jobs use it

`abl-arch` and `hero` — on by default, no flag needed, because the default repo
lives in code (`ckpt_uploader.DEFAULT_MODEL_REPO`) rather than only in `.env`.
The overnight chain sources `.env` once at launch, so a variable added later
would never have reached the jobs it starts.

`sweep` opts out (`sweep.py` passes `--hub-repo ""`): an lr probe is ~1 h and
reproducible, so durability buys nothing, and each probe would otherwise push a
1.4 GB milestone that nothing will ever branch from.

## Branching from the milestone later

Once `hero` reaches 55% of its steps (`decay_frac=0.45`), the branch point is at
`Unseen1980/daedalus-checkpoints`, revision `hero-stable-end-step<N>`. Continue
stable-phase training on more or different data with:

```bash
python train.py --run-name hero-ext --config daedalus-150m \
  --data-dir data/shards --total-tokens <new budget> \
  --resume 'hub://Unseen1980/daedalus-checkpoints/milestone/hero/checkpoint.pt?rev=hero-stable-end-step<N>'
```

This is the practical advantage of WSD over cosine: the pre-decay checkpoint can
be continued and then re-decayed, whereas continuing from a model already
annealed to lr≈0 needs an lr re-warmup from a converged state and is measurably
worse. `runs/hero/milestone.json` records the exact step, tokens seen, lr
multiplier and revision.
