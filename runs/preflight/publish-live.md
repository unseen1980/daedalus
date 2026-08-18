# Publishing the finished model: verified live against the real Hub

**2026-08-10.** Closes the last gap in AGENT.md §0.2 for the *deliverable* —
as opposed to the corpus and the resume checkpoints, which already had one.

## What was missing

Three upload paths existed, and none covered the thing the project is for:

| path | what it moves | where |
|---|---|---|
| `daedalus/shard_uploader.py`, `data.py` | the corpus | a **dataset** repo |
| `daedalus/ckpt_uploader.py` | rolling + milestone `.pt` | a **model** repo |
| — | **HF weights, tokenizer, card, GGUF** | **nowhere** |

The checkpoints are *resume* artifacts: a `torch.save` of this repo's module
tree, unusable without the repo checked out. The Q4_0 GGUF the operator
downloads and runs would have ended its life in `runs/<run>/export/`, one lost
instance away from never having existed.

## The live check

`runs/preflight/publish_live.py`, against the real Hub with real network. The
offline suite fakes `HfApi`, so it cannot exercise LFS (files over 10 MB take a
different upload path), `upload_folder` over a 321 MB safetensors, real repo
creation, or the privacy default. That is exactly the gap that made
`hub-restore.md` necessary for the checkpoint path.

Random-init `daedalus-150m`, so the weights are meaningless — the point is the
transfer, and every byte size is the real one.

| phase | what | result |
|---|---|---|
| A | export `daedalus-150m` → HF dir + card, collect a real llama.cpp Q4_0 | 6 files, `check_publishable` → yes |
| B | publish folder then GGUF | **426.5 MB in 29.6 s** (115.4 Mbit/s) |
| C | verify by **listing the repo back**, not by trusting the return value | 8 files, nothing missing, `private=True` |
| D | delete the smoke repo | done — the release name is not squatted with junk weights |

Results in `publish-live.json`.

### What C actually proves

```
model.safetensors        321.00 MB   lfs=True
gguf/model-Q4_0.gguf     101.98 MB   lfs=True
tokenizer.json             3.52 MB   lfs=False
README.md                             (the model card)
config.json, generation_config.json, tokenizer_config.json
```

Both large files went over **LFS**, which is the one behaviour the fake `HfApi`
could never have caught and the one that silently differs at the 10 MB
boundary. The card shipped with the weights rather than being a separate step
someone has to remember. GGUFs land under `gguf/` so they cannot collide with
the HF weights the same repo serves to `transformers`.

Phase B ran while `abl-arch` was training and pushing its own rolling
checkpoints, so 115 Mbit/s is a loaded-link figure, not a best case. A ~430 MB
release is half a minute.

## Tests

Offline, in the default suite: `tests/test_publisher.py`, 16 tests. The ones
that matter are the refusals — a directory missing weights, config, tokenizer
**or `README.md`** is rejected before any network call, so "we'll add the card
later" cannot come back; repos are created private unless `--public` is passed
explicitly; and a folder-upload failure propagates rather than being swallowed,
because unlike `ckpt_uploader` (which runs inside a training loop and must
never raise) this is the last thing standing between a trained model and nobody
being able to use it.

## Not automatic

`publisher` is a CLI step run after `post`, never a side effect of a job
finishing. Publishing is outward-facing and hard to walk back.

```bash
python -m daedalus.publisher --model-dir runs/<run>/export/hf \
  --gguf-dir runs/<run>/export --check-only
python -m daedalus.publisher --model-dir runs/<run>/export/hf \
  --gguf-dir runs/<run>/export
```
