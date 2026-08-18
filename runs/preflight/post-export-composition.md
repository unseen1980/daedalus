# `post.py`'s `final.pt` had never been exported. It works.

2026-08-11 ~14:50-15:05 UTC. CPU only, isolated cwd and outputs under `/tmp`;
`hero` untouched. ~15 min of CPU on a box already rented for the GPU.

## Why look

`post.py:313` describes `runs/<run>/final.pt` as *"the weights export.py should
ship"*. `grep` for `final.pt` across `runs/preflight/*.md` and `scripts/*.sh`
returned **nothing**: no note, no chain step, no procedure ever fed one to
`export.py`. `scripts/after_hero.sh` exports `hero`'s `checkpoint.pt` and
deliberately stops before `post`, so the composition `post -> export -> GGUF`
existed only as an assumption — the same shape as the six after-run-chain bugs
found earlier today and the `_chunked_loss` / branch-command no-ops before them.

`post` is ~$4.50 of the approved plan and produces the chat model a user
actually runs, so "does its output ship" is worth fifteen minutes now rather
than at the end of the sequence.

## Executed, not reasoned about

A real `post` run on CPU against `abl-arch` arm 1's checkpoint (3 SFT steps,
1 DPO step, `--limit 32`), then the real `export.py` CLI against the `final.pt`
it wrote:

| | |
|---|---|
| `post.py` | **rc 0** — SFT 6 examples, 857 supervised of 2,564 tokens (33.4%); DPO loss 0.6931 = ln 2, the correct initial value; `final.pt` 642 MB |
| `export.py` | **rc 0** — `hf/`, `model-f16.gguf`, `model-q4_0.gguf` (102.0 MB), `quantization_check.json` |
| GGUF architecture | `lfm2` |
| **`tokenizer.chat_template` in the GGUF** | **present** — `{% for message in messages %}{{ '<\|im_start\|>' + ...` |
| decode by depth | 1039.2 / 914.0 / 675.8 tok/s at depth 0 / 512 / 2048 |
| Q4_0 delta | 0.268% (meaningless as quality — the ppl corpus is repeated filler; what it shows is that the measurement path runs) |

**Verdict: the composition works.** `save_final` reuses `save_checkpoint` with
`save_optimizer=False`, and `load_checkpoint` only touches optimizer keys when
handed an optimizer, so the weights-only shape loads cleanly. The chat template
reaches the GGUF, which is the part that would have made a chat model
unusable while looking fine.

A negative result, recorded so it is not re-derived: this is now executed rather
than assumed.

## One real thing it did turn up

The first export written this way carried `torch_dtype: null` in `config.json`,
which looked like a defect — the lfm2 path never declares a dtype. It is not:
transformers 5.14.1's `save_pretrained` writes the model's **actual** dtype into
the modern `dtype` key, and the real file says `dtype: float16` beside fp16
safetensors. My first fix and its test asserted a defect that did not exist and
were withdrawn.

What replaced them is a test against the **written files** rather than either
assumption, because both mislead in opposite directions: `to_qwen3_config`
carries a `torch_dtype` literal that `save_pretrained` overwrites, and
`to_hf_config` sets none while the correct value is written anyway.
`test_the_written_config_declares_the_dtype_the_weights_are_in` loads
`config.json` and `model.safetensors` from a real `tiny` export and asserts they
agree **and** that the dtype is fp16 — which also pins the QAT-grid property
from `qat-survives-export.md` at the artifact level. Verified failing against a
bf16 `export.py`.

141 pass across `test_export.py`, `test_qat.py`, `test_post.py`,
`test_publisher.py`.

## Still not covered, stated so it is not over-read

- The Q4_0 chat model has not been **generated from** — no `llama-cli` round
  trip. A 3-step SFT on a 5B-token base would produce noise, so it would prove
  nothing about quality; what it would prove is that the template renders and
  the model stops at `<|im_end|>`. Worth doing once `post` runs for real.
- There is still **no written procedure** for the post-`post` steps (export,
  publish, model card). The after-run chain deliberately stops before `post`
  and hands the decision to the operator; that hand-off currently has no
  documented next command. Cheap to write, and it should be, before `post`
  runs.
