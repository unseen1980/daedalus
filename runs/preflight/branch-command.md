# The branch-from-milestone command trained nothing

`hero` hard precondition 4 requires the model card to carry *"the exact command
to branch stable-phase training from the milestone"*. That command is the
operator's stated reason for the whole milestone mechanism:

> It converts the $43.70 `hero` run from a one-shot artifact into a resumable
> asset: if the operator later funds more compute, they can extend training from
> the stable checkpoint instead of paying for the whole run again.

It had never been run. `runs/preflight/milestone-fires-on-abl-arch.md` records
it as *"confirmed runnable as written"*, which on inspection meant it was read,
not executed.

## What the card published

Rendered from the **real** `runs/abl-arch-daedalus-150m/milestone.json`:

```
python train.py --run-name abl-arch-daedalus-150m-ext \
  --resume 'hub://Unseen1980/daedalus-checkpoints/milestone/abl-arch-daedalus-150m/checkpoint.pt?rev=abl-arch-daedalus-150m-stable-end-step5715'
```

Two flags are missing and **neither omission produces an error.**

## Measured, on a real 150M milestone checkpoint

The artifact is `runs/sweep-wsdfix-lr0.02/milestone-checkpoint.pt` — 1.4 GB, real
weights, real Muon and AdamW state, `config: daedalus-150m`. Exactly two fields
were changed, `step` and `tokens_seen`, to the values `hero`'s own milestone will
carry. Those were not guessed: they are replayed from `train.py`'s own
`estimate_total_steps` / `seq_len_schedule` / `batch_tokens_schedule` against
hero's live launch args, read back from `runs/hero/inflight.json`.

| | |
|---|---|
| hero total steps | **124,476** |
| decay / milestone step | **68,461** (55.0%) |
| tokens_seen at the milestone | **30,532,341,760** (30.53B) |

Both agree with the live run's own W&B config, so the replay is right.

Every run below is `train.py` itself, launched from an isolated cwd so nothing
touched the repo's `runs/`, on CPU so nothing touched `hero`'s GPU.

| command | result |
|---|---|
| **as the card published it** | **0 steps, rc 0, no `metrics.jsonl` written** |
| + `--total-tokens 90e9` | trains — on **random tokens**, loss **13.04** |
| + `--data-dir …/shards-hero-split/train` | loss **4.74 → 3.94**, real text |

### Why each fails silently

- **`--total-tokens` defaults to 5,000,000,000** (`train.py:1632`). The
  milestone restores `tokens_seen = 30.5e9`, so `fit()` breaks at the top of its
  first iteration (`train.py:1579`) before a single step. The process prints
  `resumed from …: step=68461 tokens_seen=30532341760` and exits **0**. From a
  phone that is indistinguishable from a successful launch.
- **`--data-dir` defaults to `None`**, which selects `SyntheticBatchSource`
  (`train.py:926`) — `torch.randint` over the vocab. Branching would train the
  finished model on uniform noise. Loss 13.04 against ln(49152)=10.8 is that,
  measured.
- **`--config` defaults to `daedalus-150m`**, so `abl-arch`'s *dense* arm
  published a command that loads dense weights into the hybrid class. That one
  at least fails loudly.

This is the same defect that already cost this project a silent no-op once:
`post.py` passed hero's checkpoint as `resume` and its entire SFT stage did
nothing, which `TrainArgs.init_from`'s docstring records. The fix then was
applied to `post.py` only, so the next caller walked into it.

## Fixed in two places

**1. The command, `export.py:_branch_command`.** Now carries `--config` from the
milestone record, `--data-dir`, and a `--total-tokens` placeholder that names the
number it has to exceed:

```
python train.py --run-name abl-arch-daedalus-150m-ext --config daedalus-150m \
  --data-dir <YOUR_SHARD_DIR> \
  --total-tokens <NEW_BUDGET_GREATER_THAN_2548512768> \
  --resume 'hub://…?rev=abl-arch-daedalus-150m-stable-end-step5715'
```

The two placeholders cannot be filled from the record — where the reader's
shards live and how much further they want to train are theirs to choose.
Pasted verbatim they fail **loudly**: a bad `--data-dir` raises `no source under
… has a manifest.json` (verified), and argparse rejects a non-integer budget.
Omitting the flag is the silent case, so a placeholder is strictly safer than no
flag at all. The card's prose now states both hazards outright.

**2. `train.py` refuses the no-op**, because the model card is not the only way
in — `post.py` reached the same state by a different route. A resume that
restores `tokens_seen >= total_tokens` now raises `NoOpResume` naming the budget
it needs, instead of exiting 0.

### The discriminator, and why it cannot touch `hero`

A resume that trains nothing is *correct* for the recovery paths: `supervise`
and `abl_arch` relaunch a finished run and must exit 0 quietly, or a recoverable
stall becomes a chain abort. Two signals separate that from a mistaken branch,
and both come from how production actually calls it:

- the run directory has a `metrics.jsonl` of its own — a crash-resume continues
  a run that has been writing metrics for hours;
- the resume path is inside this run's own directory — both recovery paths build
  `--resume <run_dir>/checkpoint.pt` (`supervise.py:311`, `abl_arch.py:110`).

A fresh directory pointed at a foreign checkpoint is neither. `runs/hero` holds
440+ metrics rows and resumes from its own `checkpoint.pt`, so the guard is
unreachable for the live run — verified by running all three cases rather than
arguing them:

| case | required | observed |
|---|---|---|
| fresh dir, foreign ckpt, over budget | refuse | **`NoOpResume`, rc 1**, message names 30,532,341,760 |
| dir **with** metrics history, over budget | exit 0 | **rc 0**, no exception — `hero`'s crash-resume shape |
| budget above tokens_seen, real shards | train | **rc 0**, loss 4.742 — unchanged by the fix |

## Tests

Seven, each verified failing against the pre-fix file:

- three in `tests/test_export.py` — the flags are present; `--config` comes from
  the record (checked on a `dense-150m` milestone); and the whole command
  **parses against `train.parse_args`** with the placeholders substituted, so a
  renamed flag fails here rather than after a six-day run.
- four in `tests/test_train.py` — the refusal fires with an actionable message;
  a real branch onto a larger budget still trains; `--max-steps` defers; and a
  relaunch that left no metrics is still a relaunch.

`test_resume_from_an_already_complete_checkpoint_exits_cleanly` was pointing a
*fresh* run directory at a finished checkpoint — not the relaunch it documents,
but the branch mistake. Corrected to the shape `supervise.py` and `abl_arch.py`
actually build, and it now asserts no metrics row is appended rather than that
the file is absent.

**488 tests pass** across `test_train`, `test_supervise`, `test_boot_resume`,
`test_export`, `test_hero`, `test_abl_arch`, `test_post`, `test_early_finish`,
`test_guard_exit_drain`, `test_publisher` and `test_after_hero`.

## Cost of not finding it

The card ships attached to the released model. The operator pastes the command,
it prints a resume line and exits 0, and nothing has trained — on the one
mechanism that exists so a future top-up does not have to pay for `hero` twice.
If they then supplied only the budget, the branch would have trained the model
on random tokens.
