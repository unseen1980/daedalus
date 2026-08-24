"""Daedalus export (AGENT.md SS3, item 4): checkpoint -> HF Lfm2ForCausalLM ->
GGUF (llama.cpp's convert_hf_to_gguf.py) -> Q4_0 quantization (no imatrix) ->
verify fp16-vs-Q4_0 perplexity delta < 1%.

The Daedalus <-> Lfm2ForCausalLM weight mapping was verified to produce
bit-identical forward-pass logits (max abs diff 0.0 on a random init) --
model.py's tensor names were deliberately chosen to make this close to a
pure rename. One real gotcha found while wiring this up: HF's Lfm2Config
auto-adjusts the SwiGLU inner dim (`block_multiple_of` /
`block_ffn_dim_multiplier`) unless `block_auto_adjust_ff_dim=False` --
left on, it silently produces a different `intermediate_size` than our
`block_ff_dim` (2048 -> 1536 at daedalus-150m's settings) and every FFN
weight would fail to load.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from typing import Optional, Sequence

import torch

from daedalus import ckpt_uploader
from daedalus.config import PRESETS, DaedalusConfig
from daedalus.data import SMOLLM2_TOKENIZER
from daedalus.model import Daedalus

MAX_PPL_DELTA_PCT = 1.0  # AGENT.md SS3: fp16-vs-Q4_0 perplexity delta must be < 1%
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


# --------------------------------------------------------------- HF export ---

def to_hf_config(cfg: DaedalusConfig):
    from transformers import Lfm2Config
    d = cfg.to_hf_dict()
    d["block_auto_adjust_ff_dim"] = False
    # Deliberately does *not* declare a dtype: `save_pretrained` writes the
    # model's real one into config.json's `dtype` (checked on a real export
    # under transformers 5.14.1), so a literal here can only go stale.
    # `test_the_written_config_declares_the_dtype_the_weights_are_in` pins the
    # property against the file rather than against either assumption.
    return Lfm2Config(**d)


def is_all_attention(cfg: DaedalusConfig) -> bool:
    """True for the `dense-150m` twin -- no conv blocks at all."""
    return all(t == "full_attention" for t in cfg.layer_types)


# ------------------------------------------------------- dense twin (qwen3) ---
# The dense twin cannot ship as `lfm2`. llama.cpp models lfm2 as a *hybrid*
# memory architecture (recurrent conv state alongside the KV cache); with zero
# conv layers the recurrent half has no buffer and `llama_decode` aborts in
# `llm_graph_input_mem_hybrid::set_input` -> `ggml_backend_buffer_is_host(NULL)`.
# Verified live: convert_hf_to_gguf.py and llama-quantize both succeed, then
# llama-bench crashes with GGML_ASSERT(buffer) the moment it decodes a token.
#
# `qwen3` is the right target instead: it is exactly this architecture -- GQA
# with per-head RMSNorm QK-norm, SwiGLU, RoPE, optional tied embeddings -- and
# it is llama.cpp's best-optimised dense CPU path, which also makes the
# abl-arch comparison fair (each architecture on its own tuned kernels rather
# than one of them on a graph it doesn't fit).
_QWEN3_LAYER_MAP = {
    "input_layernorm": "operator_norm",
    "post_attention_layernorm": "ffn_norm",
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj",
    "self_attn.o_proj": "self_attn.out_proj",
    "self_attn.q_norm": "self_attn.q_layernorm",
    "self_attn.k_norm": "self_attn.k_layernorm",
    "mlp.gate_proj": "feed_forward.w1",
    "mlp.up_proj": "feed_forward.w3",
    "mlp.down_proj": "feed_forward.w2",
}


def to_qwen3_config(cfg: DaedalusConfig):
    from transformers import Qwen3Config
    return Qwen3Config(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.block_ff_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.norm_eps, rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings, attention_bias=False,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
        # Declarative only -- `save_pretrained` overwrites this from the
        # model's actual dtype. Kept in step with `export_hf_model`'s fp16
        # default so the source does not carry a literal that contradicts it.
        torch_dtype="float16",
    )


def to_qwen3_state_dict(our_state_dict: dict, num_hidden_layers: int,
                        tie_word_embeddings: bool) -> dict:
    """Maps the all-attention Daedalus state_dict onto Qwen3ForCausalLM's."""
    sd = our_state_dict
    out = {
        "model.embed_tokens.weight": sd["embed_tokens.weight"],
        "model.norm.weight": sd["norm.weight"],
    }
    for i in range(num_hidden_layers):
        for qwen_suffix, ours_suffix in _QWEN3_LAYER_MAP.items():
            key = f"layers.{i}.{ours_suffix}.weight"
            if key not in sd:
                raise ValueError(f"unmapped Daedalus state_dict key: {key!r} "
                                 "(is this config really all-attention?)")
            out[f"model.layers.{i}.{qwen_suffix}.weight"] = sd[key]
    out["lm_head.weight"] = (sd["embed_tokens.weight"] if tie_word_embeddings
                             else sd["lm_head.weight"])
    return out


def to_hf_state_dict(our_state_dict: dict, tie_word_embeddings: bool) -> dict:
    """Maps Daedalus's state_dict keys to Lfm2ForCausalLM's."""
    hf_sd = {}
    for k, v in our_state_dict.items():
        if k == "norm.weight":
            hf_k = "model.embedding_norm.weight"
        elif k == "embed_tokens.weight":
            hf_k = "model.embed_tokens.weight"
        elif k.startswith("layers."):
            hf_k = "model." + k
        elif k == "lm_head.weight":
            hf_k = "lm_head.weight"
        else:
            raise ValueError(f"unmapped Daedalus state_dict key: {k!r}")
        hf_sd[hf_k] = v
    if tie_word_embeddings and "lm_head.weight" not in hf_sd:
        hf_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]
    return hf_sd


def export_hf_model(ckpt_path: str, cfg_name: str, out_dir: str,
                    dtype: torch.dtype = torch.float16) -> str:
    """checkpoint.pt -> a HF-format model directory (config.json + weights),
    ready for llama.cpp's convert_hf_to_gguf.py once export_tokenizer() has
    also written the tokenizer files there.

    **fp16, not bf16 -- this is load-bearing for QAT and was measured.** QAT
    trains the weights onto llama.cpp's exact Q4_0 lattice, whose values are
    `(q - 8) * fp16(d)`. bf16 keeps 8 mantissa bits where fp16 keeps 11, so
    writing the HF tensors as bf16 perturbs the block scale, `llama-quantize`
    then recovers `bf16(d)` instead of `fp16(d)`, and the shipped weights are
    no longer the weights QAT converged to. Measured on a real 150M checkpoint
    projected onto the grid, through the real export -> convert -> quantize
    path, comparing the Q4_0 blocks read back out of the GGUF:

        HF dtype   ||shipped - QAT grid|| / ||QAT grid||
        bfloat16   0.1712 %      <- 2.7% of the RTN damage QAT is bought to remove
        float16    0.0000 %      <- bit-exact, 0 of 122,683,392 weights moved

    fp32 is also bit-exact but doubles the published artifact for nothing.
    fp16 range is not a concern here: max |w| is 2.19 against fp16's 65504
    (~30,000x headroom), and every weight that underflows fp16 is >=5 orders of
    magnitude below one Q4_0 level, so it quantizes to zero either way -- which
    is why the measured result is exactly 0.0000%, not merely small.
    See `runs/preflight/qat-survives-export.md`.

    Hybrid configs export as `Lfm2ForCausalLM`; the all-attention dense twin
    exports as `Qwen3ForCausalLM` instead, because llama.cpp's lfm2 graph
    crashes on a conv-free model (see the qwen3 section above)."""
    from train import load_checkpoint

    cfg = PRESETS[cfg_name]
    ours = Daedalus(cfg)
    info = load_checkpoint(ckpt_path, ours, map_location="cpu")
    ours.eval()

    if is_all_attention(cfg):
        from transformers import Qwen3ForCausalLM
        hf_model = Qwen3ForCausalLM(to_qwen3_config(cfg))
        hf_sd = to_qwen3_state_dict(ours.state_dict(), cfg.num_hidden_layers,
                                    cfg.tie_word_embeddings)
    else:
        from transformers import Lfm2ForCausalLM
        hf_model = Lfm2ForCausalLM(to_hf_config(cfg))
        hf_sd = to_hf_state_dict(ours.state_dict(), cfg.tie_word_embeddings)
    missing, unexpected = hf_model.load_state_dict(hf_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    hf_model.to(dtype)
    os.makedirs(out_dir, exist_ok=True)
    hf_model.save_pretrained(out_dir)
    # Written here, at the one place every export goes through, so abl-arch's
    # two arms, hero and post all get a card without each caller remembering to
    # ask for one. Callers with more to say (eval metrics, the Q4_0 delta)
    # overwrite it by calling write_model_card again -- see _cli below.
    #
    # `tokens_seen` comes from the checkpoint we just loaded, never from the
    # milestone: the milestone is written at 55% of a run, so quoting its count
    # as the training total would understate every finished run by 45%.
    write_model_card(out_dir, cfg_name,
                     milestone=_load_milestone(ckpt_path),
                     run_name=_run_name_of(ckpt_path),
                     tokens_seen=info.get("tokens_seen"))
    return out_dir


def _run_name_of(ckpt_path: str) -> Optional[str]:
    """A checkpoint lives at `runs/<run_name>/checkpoint.pt`; the payload
    itself does not carry the run name (train.py's save_checkpoint writes
    model/step/tokens_seen/config), so the directory is the only source."""
    if not ckpt_path:
        return None
    return os.path.basename(os.path.dirname(os.path.abspath(ckpt_path))) or None


# The prompt format `post.py` trains on, as a Jinja template, so the shipped
# artifact declares it instead of relying on a consumer's fallback.
#
# `daedalus/chatml.py`'s own docstring says a train/inference formatting
# mismatch "is invisible at training time and shows up only as a model that
# ignores its prompt", and requires that "post.py and export/inference must
# both go through this function" -- and the export did not. SmolLM2's tokenizer
# ships no `chat_template`, so nothing reached the GGUF either
# (`tokenizer.chat_template`: absent, checked on a real file).
#
# It happened to work: llama.cpp's `common_chat_template` "always set (defaults
# to chatml)" (common/chat.cpp:340) and its chatml is ours. That is a
# coincidence between two independent defaults, not a contract, and it is the
# kind that holds until the day a consumer picks a different fallback. The
# template below is asserted equal to `render_prompt` on real conversations by
# `test_the_exported_chat_template_matches_what_post_trains_on`, so the two
# cannot drift.
#
# The *base* model export carries it too, which is deliberate: one code path is
# worth more than the cosmetic oddity of a pre-`post` checkpoint advertising a
# chat format, and the failure this prevents lands on the instruct artifact --
# the one an operator actually talks to. The model card is where "this
# checkpoint is not instruction-tuned" belongs, not the tokenizer.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] "
    "+ '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def export_tokenizer(out_dir: str, tokenizer: Optional[str] = None,
                     expected_vocab_size: Optional[int] = None) -> str:
    """SmolLM2's tokenizer, reused byte-identical (AGENT.md SS2) -- its
    pre-tokenizer hash is registered with llama.cpp's converter, so any
    modification breaks GGUF conversion.

    Byte-identical refers to the *vocabulary and merges*, which are untouched:
    all 49,152 id->token mappings are verified against SmolLM2 upstream
    (`runs/preflight/gguf-tokenizer-and-q4-damage.md`). `chat_template` is
    metadata beside them, read by `apply_chat_template` and copied into the
    GGUF -- it changes no id, so the converter's pre-tokenizer hash is
    unaffected.

    `tokenizer` overrides the default with a path or Hub name, for Phase 4's
    candidate vocabularies. `None` keeps SmolLM2, so every shipped export is
    unchanged. **A non-default vocabulary does not convert under stock
    llama.cpp** -- `conversion/base.py::get_vocab_base_pre` hashes the token
    ids of a fixed probe string and raises `NotImplementedError` for any hash
    it does not already carry, so a newly trained vocabulary has to be
    registered upstream first. That is a finding for the V2 migration report,
    not something this function can work around, and it is why the override
    exists for measurement rather than for shipping.

    `expected_vocab_size` is the guard that makes the override safe to use.
    Writing a 49,152-token tokenizer beside a 32,768-row embedding produces a
    directory that converts, quantizes and loads, and answers with the wrong
    token for every id above the model's range.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer or SMOLLM2_TOKENIZER)
    if expected_vocab_size is not None and tok.vocab_size != expected_vocab_size:
        raise ValueError(
            f"tokenizer {tokenizer or SMOLLM2_TOKENIZER!r} has "
            f"{tok.vocab_size} tokens but the model's embedding has "
            f"{expected_vocab_size} rows; exporting them together produces an "
            f"artifact that loads and decodes the wrong token for every id")
    tok.chat_template = CHATML_TEMPLATE
    tok.save_pretrained(out_dir)
    return out_dir


# -------------------------------------------------------------- model card ---
# The `hero` hard precondition asks for the Hub checkpoint repo and the exact
# branch-from-milestone command to live "in STATUS.md and the model card".
# STATUS.md had both; there was no model card -- `export_hf_model` wrote
# config.json + weights and nothing else, so every artifact that ever leaves
# this box would have arrived with no provenance, no stated success bar, and no
# way for a reader to tell a 13B-token run from a 40B one.
#
# Everything here is rendered from data that was actually measured. A section
# whose input is missing says so rather than being dropped silently or filled
# with a plausible number -- an unlabelled gap in a model card reads as "not
# applicable", which is the one thing it must never mean for an eval table.

# The operator-confirmed definition of success (issue #4 SS4.3). Written into
# every card so that later results are read against the bar that was set before
# they landed, not one adjusted to fit them.
SUCCESS_BAR = (
    "Beat Pythia-160M, OPT-125M and GPT-neo-125M on quality; target "
    "MobileLLM-125M as a stretch; concede SmolLM2-135M on quality while "
    "beating it decisively on CPU decode."
)

# Costed, operator-approved departures from DAEDALUS-BLUEPRINT-v6.md. Kept here
# rather than only in the repo so the deviation travels with the weights.
BLUEPRINT_DEVIATIONS = [
    "**No distillation** from SmolLM2-1.7B during decay. 288 GB of top-16 "
    "logits does not fit the disk and the online-teacher variant cost ~$29 of "
    "a $94.66 budget; its own evidence was only \"+1-3 points plausible\".",
    "**Corpus stops at ~14.2B tokens, not 45B.** Training repeats a balanced "
    "corpus rather than seeing 45B unique tokens; at this scale repetition up "
    "to ~4 epochs costs little against fresh tokens, and mixture balance "
    "mattered more than raw size.",
    "**Document-aligned packing not implemented** -- sequences may cross "
    "document boundaries.",
    "**NoPE skipped** -- it breaks GGUF export.",
    "**Single seed** for the hero run, so no seed-sigma is reported.",
    "**`everyday-conversations` contributes ~0.00%** of pretraining instead of "
    "its 2% share (the whole dataset is 0.4M tokens, which the 4-epoch cap "
    "reduces to nothing); dialogue enters at the `post` SFT stage instead.",
]


def _branch_command(milestone: dict) -> Optional[str]:
    """The exact command that resumes stable-phase training from a milestone.

    WSD's practical advantage is that the pre-decay checkpoint is a reusable
    branch point: more data can be trained on top of it and re-decayed, which
    is much cheaper and measurably better than re-warming an already-annealed
    model. That is only true if someone can find the revision, so the command
    is rendered from the milestone record rather than described in prose.

    **Every flag below is load-bearing, and each was measured missing.** The
    first version of this function emitted only `--run-name` and `--resume`,
    and both omissions fail without an error (`runs/preflight/branch-command.md`):

    - no `--total-tokens`: the default is 5e9, and `hero`'s milestone carries
      30.5e9 tokens_seen, so `fit()` breaks at the top of its first iteration
      (`train.py:1579`) -- **0 steps, rc 0, no metrics.jsonl**, after printing
      a `resumed from ...` line that reads like success.
    - no `--data-dir`: `TrainArgs.data_dir=None` selects `SyntheticBatchSource`,
      so branching trains the model on **random tokens** -- measured loss 13.04
      against 3.94 on the same checkpoint with the flag.
    - no `--config`: defaults to `daedalus-150m`, which silently mislabels any
      other arm (`dense-150m` would load into the wrong class).

    The two placeholders cannot be filled from the record -- where the reader's
    shards live and how many more tokens they want are theirs to choose -- so
    they are named in caps. Pasted verbatim they fail loudly: a bad `--data-dir`
    raises `no source under ... has a manifest.json`, and argparse rejects a
    non-integer `--total-tokens`. Omitting the flag is the silent case, which
    is why placeholders beat leaving it out.
    """
    repo = milestone.get("repo")
    path = milestone.get("path_in_repo")
    revision = milestone.get("revision")
    if not (repo and path and revision):
        return None
    # The run name is not a field of the milestone record -- it is the middle
    # segment of `milestone/<run_name>/checkpoint.pt`, which train.py builds
    # from `run_name` and which is what the revision is named after too.
    parts = path.split("/")
    run = parts[1] if len(parts) > 2 else "run"
    config = milestone.get("config") or "daedalus-150m"
    seen = int(milestone.get("tokens_seen") or 0)
    return (f"python train.py --run-name {run}-ext --config {config} \\\n"
            f"  --data-dir <YOUR_SHARD_DIR> \\\n"
            f"  --total-tokens <NEW_BUDGET_GREATER_THAN_{seen}> \\\n"
            f"  --resume 'hub://{repo}/{path}?rev={revision}'")


def _load_milestone(ckpt_path: Optional[str]) -> dict:
    """The milestone record written beside a checkpoint, or {}.

    Never raises: a missing or malformed milestone must cost the card a
    section, not cost the export.
    """
    if not ckpt_path:
        return {}
    path = os.path.join(os.path.dirname(ckpt_path) or ".", "milestone.json")
    try:
        with open(path) as f:
            record = json.load(f)
        return record if isinstance(record, dict) else {}
    except Exception:
        return {}


def render_model_card(cfg_name: str, *, run_name: Optional[str] = None,
                      milestone: Optional[dict] = None,
                      hub_repo: Optional[str] = None,
                      metrics: Optional[dict] = None,
                      quantization: Optional[dict] = None,
                      tokens_seen: Optional[int] = None) -> str:
    """The card's markdown. Pure -- takes no filesystem or network, so tests
    can assert on the text without an export having run."""
    cfg = PRESETS[cfg_name]
    counts = cfg.param_count()
    milestone = milestone or {}
    interleave = "".join("A" if t == "full_attention" else "c"
                         for t in cfg.layer_types)
    arch = "Qwen3ForCausalLM" if is_all_attention(cfg) else "Lfm2ForCausalLM"
    # A run's own milestone records the repo it actually pushed to; fall back to
    # the configured default only when it does not (e.g. `sweep`, which opts out
    # of Hub durability on purpose).
    repo = milestone.get("repo") or hub_repo or os.environ.get(
        "DAEDALUS_HF_MODEL_REPO", ckpt_uploader.DEFAULT_MODEL_REPO)

    lines = [
        "---",
        "license: apache-2.0",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- daedalus",
        "- cpu-inference",
        "- gguf",
        "- q4_0",
        "---",
        "",
        f"# {cfg_name}",
        "",
        f"A {counts['total'] / 1e6:.1f}M-parameter causal LM built for the best "
        "quality-per-token-per-second on **CPU** inference, exported to GGUF "
        "Q4_0 for llama.cpp.",
        "",
        "## What this model is trying to beat",
        "",
        f"> {SUCCESS_BAR}",
        "",
        "This bar was fixed before any result landed. Numbers below are "
        "reported against it whether or not they clear it.",
        "",
        "## Architecture",
        "",
        "| | |",
        "|---|---|",
        f"| exported as | `{arch}` |",
        f"| parameters | {counts['total']:,} "
        f"({counts['non_embedding']:,} non-embedding) |",
        f"| blocks | {cfg.num_hidden_layers} (`{interleave}` -- "
        f"`c` = gated short conv, `A` = GQA attention) |",
        f"| hidden size | {cfg.hidden_size} |",
        f"| SwiGLU inner dim | {cfg.block_ff_dim} |",
        f"| heads | {cfg.num_attention_heads} query / "
        f"{cfg.num_key_value_heads} KV, head_dim {cfg.head_dim}, QK-norm |",
        f"| RoPE theta | {cfg.rope_theta:,.0f} |",
        f"| context | {cfg.max_position_embeddings} |",
        f"| tied embeddings | {cfg.tie_word_embeddings} |",
        # `cfg.tokenizer` is None for every shipped preset, so this renders the
        # SmolLM2 line unchanged. A Phase 4 probe preset names its own
        # vocabulary instead, because a card claiming a byte-identical SmolLM2
        # tokenizer beside a 32,768-row embedding would be false.
        (f"| tokenizer | [`{SMOLLM2_TOKENIZER}`](https://huggingface.co/"
         f"{SMOLLM2_TOKENIZER}), reused byte-identical, vocab "
         f"{cfg.vocab_size:,} |")
        if cfg.tokenizer is None else
        (f"| tokenizer | `{cfg.tokenizer}`, vocab {cfg.vocab_size:,} "
         f"(not SmolLM2; see the V2 tokenizer migration report) |"),
        "",
    ]

    lines += ["## Training", ""]
    if run_name:
        lines.append(f"- run: `{run_name}`")
    if tokens_seen:
        lines.append(f"- tokens seen: {tokens_seen:,} "
                     f"({tokens_seen / counts['total']:.0f} tokens/parameter)")
    if milestone.get("muon_lr") is not None:
        lines.append(f"- Muon lr {milestone['muon_lr']} on 2D hidden matrices; "
                     f"AdamW lr {milestone.get('adam_lr')} on "
                     f"embeddings/head/norms")
    if milestone.get("decay_frac") is not None:
        lines.append(f"- WSD schedule, linear decay to zero over the final "
                     f"{milestone['decay_frac']:.0%} of the run")
    lines += ["", "## Evaluation", ""]
    if metrics:
        lines += ["| metric | value |", "|---|---|"]
        for key in sorted(metrics):
            value = metrics[key]
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {key} | {rendered} |")
    else:
        lines.append("_Not yet measured for this export._")
    lines += ["", "## Q4_0 quantization", ""]
    if quantization:
        delta = quantization.get("delta_pct")
        lines += [
            f"- fp16 perplexity **{quantization.get('fp16_ppl')}** vs Q4_0 "
            f"**{quantization.get('q4_0_ppl')}**",
            f"- delta **{delta:.3f}%**" if isinstance(delta, float)
            else f"- delta {delta}",
            f"- passes the <{MAX_PPL_DELTA_PCT}% threshold: "
            f"**{quantization.get('passes_threshold')}**",
        ]
        speed = quantization.get("decode_speed") or {}
        if speed.get("tok_per_sec"):
            # Always say which context depth. Only 6 of 18 blocks keep a KV
            # cache, so this model's decode speed falls off far more slowly
            # with context than a dense one's -- a single unlabelled tok/s
            # figure is both unreproducible and, at depth 0, the least
            # interesting number the model produces.
            by_depth = speed.get("by_depth") or {}
            rows = sorted((int(k), v) for k, v in by_depth.items()
                          if (v or {}).get("tok_per_sec"))
            threads = speed.get("n_threads")
            if rows:
                lines.append(f"- llama.cpp CPU decode at {threads} threads, by "
                             f"context depth:")
                for depth, item in rows:
                    tag = " (the trained context)" if depth == 2048 else ""
                    lines.append(
                        f"  - depth **{depth}**{tag}: "
                        f"**{item['tok_per_sec']:.1f} tok/s** "
                        f"(+/- {item.get('tok_per_sec_stddev') or 0:.1f})")
            else:
                lines.append(
                    f"- llama.cpp CPU decode **{speed['tok_per_sec']:.1f} "
                    f"tok/s** (+/- {speed.get('tok_per_sec_stddev', 0):.1f}) at "
                    f"{threads} threads, context depth 0")
    else:
        lines.append("_Not yet measured for this export._")

    lines += ["", "## Checkpoints and how to continue training", ""]
    lines.append(f"Checkpoints are pushed to the private Hub model repo "
                 f"**`{repo}`**: weights-only bf16 rolling copies every ~2 h "
                 f"under `rolling/<run>/weights.pt`, plus a milestone with "
                 f"full Muon + AdamW optimizer state at the WSD decay-start "
                 f"step on its own revision.")
    command = _branch_command(milestone)
    if command:
        lines += [
            "",
            f"The stable-phase branch point for this model is revision "
            f"**`{milestone['revision']}`** (step {milestone.get('step'):,}, "
            f"{milestone.get('tokens_seen', 0):,} tokens seen, lr multiplier "
            f"{milestone.get('lr_mult_at_branch')}). To continue stable-phase "
            f"training from it on more or different data and then re-decay:",
            "",
            "```bash",
            command,
            "```",
            "",
            f"Fill both placeholders. `--total-tokens` **must exceed the "
            f"{milestone.get('tokens_seen', 0):,} tokens already seen** — a "
            f"smaller budget makes the run stop at the top of its first "
            f"iteration, printing a `resumed from ...` line and exiting 0 "
            f"having trained nothing. And `--data-dir` is not optional: "
            f"without it training falls back to randomly generated tokens, "
            f"which silently destroys the checkpoint you branched from.",
            "",
            "Branching from the pre-decay checkpoint is the point of WSD: "
            "resuming an already-annealed model needs an lr re-warmup from a "
            "converged state, which is measurably worse.",
        ]
    else:
        lines += ["", "_No milestone record was found beside this "
                  "checkpoint, so no branch point is published for it._"]

    lines += ["", "## Deviations from the blueprint", "",
              "Each was costed and approved rather than silently dropped; "
              "see `DAEDALUS-BLUEPRINT-v6.md` and issue #4.", ""]
    lines += [f"- {d}" for d in BLUEPRINT_DEVIATIONS]
    lines.append("")
    return "\n".join(lines)


def write_model_card(out_dir: str, cfg_name: str, **kwargs) -> Optional[str]:
    """Write `README.md` (the Hub model card) into an exported model dir.

    Never raises. A card is documentation; an export that has already produced
    correct weights must not fail because a string could not be formatted.
    llama.cpp's converter reads config.json/tokenizer/safetensors by name, so
    an extra README.md is inert to it.
    """
    path = os.path.join(out_dir, "README.md")
    try:
        # Render before opening, and replace atomically. `open(path, "w")`
        # truncates first, so rendering inside the `with` would leave a
        # zero-byte card behind on any failure -- and _cli calls this a second
        # time to add the Q4_0 numbers, so that failure mode would destroy a
        # good card rather than merely fail to improve it.
        text = render_model_card(cfg_name, **kwargs)
        os.makedirs(out_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        return path
    except Exception as e:
        print(f"WARNING: could not write the model card to {path} ({e}); "
              f"the export itself is unaffected")
        return None


# --------------------------------------------------------------- llama.cpp ---

# Every `subprocess.run` below is time-bounded, for the reason spelled out at
# BENCH_TIMEOUT_S: this code runs unattended at the end of a multi-day job, and
# a hang is not an exception. No `except` can see one, the export step never
# returns, whatever waits on it waits forever, and the box bills $10.78/day with
# the GPU idle. `runs/preflight/milestone-fires-on-abl-arch.md` recorded these
# four call sites as the remaining unbounded ones and deferred the fix until
# their real durations were known, so as not to guess a bound hours before a
# 25 h job. They are known now.
#
# Measured off the mtimes of the two real `abl-arch` exports on this box -- same
# 150M-class model and the same `data/eval/ppl-finewiki-150k.txt` (529 KB) the
# `hero` chain passes, so the numbers transfer directly:
#
#   step                 arm 1    arm 2   bound      headroom
#   convert_hf_to_gguf   3.29 s   2.84 s  1800 s     ~550x
#   llama-quantize       0.72 s   0.66 s   900 s    ~1250x
#   llama-perplexity     ~50 s each        3600 s     ~72x
#
# The perplexity figure is an upper bound: both runs *plus* the 14.8 s decode
# sweep fit inside the 114.5 s between arm 2's `model-q4_0.gguf` and
# `runs/abl-arch/results.json`. Every bound is a wide multiple of a measurement,
# so none can fail a healthy run even on a box loaded by something else -- they
# exist only to turn a wedge into a traceback.
CONVERT_TIMEOUT_S = 1800.0
QUANTIZE_TIMEOUT_S = 900.0
PPL_TIMEOUT_S = 3600.0

# The build path is skipped entirely whenever the four artifacts already exist
# (they do, on this box), so these bite only on a fresh clone: a shallow clone
# over this ~660 Mbit link and a CPU-only three-target cmake build.
CLONE_TIMEOUT_S = 1800.0
CMAKE_CONFIGURE_TIMEOUT_S = 900.0
CMAKE_BUILD_TIMEOUT_S = 3600.0


def setup_llama_cpp(dest_dir: str, jobs: Optional[int] = None) -> str:
    """Shallow-clone and build just the targets export.py needs
    (llama-quantize, llama-perplexity, llama-bench) plus the pure-python
    converter script. Idempotent: skips the clone/build if all are present."""
    convert_script = os.path.join(dest_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(dest_dir, "build", "bin", "llama-quantize")
    perplexity_bin = os.path.join(dest_dir, "build", "bin", "llama-perplexity")
    bench_bin = os.path.join(dest_dir, "build", "bin", "llama-bench")
    if os.path.exists(convert_script) and os.path.exists(quantize_bin) \
            and os.path.exists(perplexity_bin) and os.path.exists(bench_bin):
        return dest_dir

    if not os.path.exists(os.path.join(dest_dir, ".git")):
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, dest_dir],
                       check=True, timeout=CLONE_TIMEOUT_S)

    build_dir = os.path.join(dest_dir, "build")
    subprocess.run(["cmake", "-B", build_dir, "-S", dest_dir,
                    "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF",
                    "-DGGML_CUDA=OFF"],  # CPU-only: this is for the CPU-decode target
                   check=True, cwd=dest_dir, timeout=CMAKE_CONFIGURE_TIMEOUT_S)
    jobs = jobs or os.cpu_count() or 4
    subprocess.run(["cmake", "--build", build_dir, "--config", "Release",
                    "-j", str(jobs), "--target",
                    "llama-quantize", "llama-perplexity", "llama-bench"],
                   check=True, cwd=dest_dir, timeout=CMAKE_BUILD_TIMEOUT_S)
    return dest_dir


def convert_to_gguf(hf_dir: str, out_gguf: str, llama_cpp_dir: str,
                    outtype: str = "f16",
                    timeout_s: float = CONVERT_TIMEOUT_S) -> str:
    script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    os.makedirs(os.path.dirname(out_gguf) or ".", exist_ok=True)
    # sys.executable, not a bare "python3" -- the converter needs torch, which
    # is only installed in this venv, not the system interpreter (found live:
    # a bare "python3" silently resolved to a torch-less system Python).
    subprocess.run([sys.executable, script, hf_dir, "--outfile", out_gguf,
                    "--outtype", outtype], check=True, timeout=timeout_s)
    return out_gguf


def quantize_gguf(in_gguf: str, out_gguf: str, llama_cpp_dir: str,
                  qtype: str = "Q4_0",
                  timeout_s: float = QUANTIZE_TIMEOUT_S) -> str:
    """AGENT.md: Q4_0 *without* imatrix -- so no --imatrix flag, ever.

    Deliberately does **not** pass `--token-embedding-type q8_0`, so llama.cpp's
    own default applies and `token_embd.weight` ships as **Q6_K**. This is a
    measured, intentional deviation from `DAEDALUS-BLUEPRINT-v6.md:59` ("Keep
    `token_embd`/`output` at Q8_0") -- do not "restore compliance" here without
    re-reading `runs/preflight/token-embd-quant-grid.md`. Measured on a real
    500M-token checkpoint, alternating back-to-back:

        token_embd  PPL(-c 512)  vs fp16   decode t/s (-t 8)   size
        Q6_K        35.9114      +1.558%   1030.5              95.56 MiB
        Q8_0        35.8954      +1.513%    924.9             104.28 MiB

    Q8_0 buys +0.045% perplexity -- inside its own +/-0.54 error bars -- and
    costs 10.2% of CPU decode and 9.1% of file size. The mission is the best
    ~150M model that is also the fastest on CPU, so the blueprint's preference
    loses here on its own terms.

    Note the mismatch this leaves: `daedalus/qat.py::plan_qat` still fake-
    quantizes the embedding to Q8_0. Bounded by the same measurement at
    <=0.045% of perplexity, and left alone on purpose (see the writeup).
    """
    bin_path = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    os.makedirs(os.path.dirname(out_gguf) or ".", exist_ok=True)
    subprocess.run([bin_path, in_gguf, out_gguf, qtype], check=True,
                   timeout=timeout_s)
    return out_gguf


_PPL_RE = re.compile(r"Final estimate: PPL = ([\d.]+)")


def parse_perplexity(output: str) -> float:
    m = _PPL_RE.search(output)
    if not m:
        raise ValueError("could not find 'Final estimate: PPL = ...' in "
                         f"llama-perplexity output:\n{output}")
    return float(m.group(1))


def measure_perplexity(gguf_path: str, llama_cpp_dir: str, text_file: str,
                       n_ctx: int = 512,
                       timeout_s: float = PPL_TIMEOUT_S) -> float:
    bin_path = os.path.join(llama_cpp_dir, "build", "bin", "llama-perplexity")
    result = subprocess.run([bin_path, "-m", gguf_path, "-f", text_file,
                             "-c", str(n_ctx), "--no-warmup"],
                            check=True, capture_output=True, text=True,
                            timeout=timeout_s)
    return parse_perplexity(result.stdout + result.stderr)


def perplexity_delta_pct(fp16_ppl: float, q4_0_ppl: float) -> float:
    return abs(q4_0_ppl - fp16_ppl) / fp16_ppl * 100


# Context depths the decode benchmark reports, in the order it runs them.
#
# Depth 0 is what a bare `llama-bench` measures and what every decode number
# recorded on this project so far used, so it stays first and stays the value
# of the top-level `tok_per_sec` keys -- consumers and the historical record
# keep meaning what they meant.
#
# The other two exist because depth is the entire argument for this
# architecture, and measuring only at 0 measures it exactly where it has least
# to gain. A gated short conv's decode cost is flat in context length;
# attention's KV reads grow with it. Only 6 of Daedalus's 18 blocks keep a KV
# cache, so the advantage over an all-attention model of the same size is
# near-invisible into an empty context and large at the context we train for.
# Measured against SmolLM2-135M on this box, the same model reads 1.06x at
# depth 0 and 2.08x at depth 2048 (`runs/eval/decode-vs-smollm2.json`).
#
# 2048 is the trained context and the number worth quoting; 512 is kept so the
# hybrid-vs-dense table has the same three rows as the SmolLM2 one and the two
# can be read side by side.
DECODE_DEPTHS = (0, 512, 2048)

# A whole three-depth sweep on the real 150M Q4_0 GGUF measures 14.8 s
# (runs/preflight/export-depth-sweep-live.md), so 900 s cannot fail a healthy
# bench -- it exists only to bound a wedged one. The "best effort, record the
# error, never raise" contract below is written against exceptions, and a hang
# is not an exception: `subprocess.run` with no timeout waits forever, the
# export step never returns, and `abl_arch.py` waits on it with the GPU idle at
# $10.78/day. This project has already lost time to exactly that shape twice
# (the Hub uploader wedged in CLOSE-WAIT and no retry path could see it), and
# this code runs unattended at the end of both ablation arms and of `hero`.
BENCH_TIMEOUT_S = 900.0


def _bench_one_depth(bin_path: str, gguf_path: str, n_gen: int, depth: int,
                     threads: Optional[int],
                     timeout_s: float = BENCH_TIMEOUT_S) -> dict:
    """One `llama-bench` decode measurement at a single context depth."""
    cmd = [bin_path, "-m", gguf_path, "-p", "0", "-n", str(n_gen), "-o", "json"]
    # Depth 0 is llama-bench's own default, so it runs the byte-identical
    # command this function used before depths existed.
    if depth:
        cmd += ["-d", str(depth)]
    if threads:
        cmd += ["-t", str(threads)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                            timeout=timeout_s)
    rows = json.loads(result.stdout)
    # `-p 0` leaves exactly one row, but select on n_gen rather than trusting
    # the position -- a future llama-bench that emits a prefill row too would
    # otherwise silently report prompt-processing speed as decode speed.
    row = next((r for r in rows if int(r.get("n_gen", n_gen)) == n_gen), rows[0])
    return {
        "depth": depth,
        "tok_per_sec": row["avg_ts"],
        "tok_per_sec_stddev": row["stddev_ts"],
        "n_threads": row["n_threads"],
    }


def measure_decode_speed(gguf_path: str, llama_cpp_dir: str, n_gen: int = 128,
                         threads: Optional[int] = None,
                         depths: Sequence[int] = DECODE_DEPTHS) -> dict:
    """Pure CPU decode (text-generation) tok/s via `llama-bench`, by depth.

    `-p 0` disables the prompt-processing test so only the `n_gen`-token
    decode phase is timed -- this is AGENT.md's "measured llama.cpp CPU
    decode tok/s", not prefill throughput.

    Returns the depth-0 measurement at the top level (unchanged from before
    this function grew depths) plus a `by_depth` map. Only depth 0 is allowed
    to fail the caller: the deeper measurements are best-effort and record
    their error instead of raising, because they are an addition to a step
    that already gates an expensive training arm's export -- a new benchmark
    must not be able to cost an arm its GGUF.
    """
    bin_path = os.path.join(llama_cpp_dir, "build", "bin", "llama-bench")
    ordered = [int(d) for d in (depths or ())]
    if 0 not in ordered:
        ordered.insert(0, 0)

    by_depth: dict = {}
    base: dict = {}
    for depth in ordered:
        if depth == 0:
            base = _bench_one_depth(bin_path, gguf_path, n_gen, 0, threads)
            by_depth["0"] = base
            continue
        try:
            by_depth[str(depth)] = _bench_one_depth(
                bin_path, gguf_path, n_gen, depth, threads)
        except Exception as e:  # noqa: BLE001 - best effort by design
            print(f"=== decode bench at depth {depth} failed, continuing: "
                  f"{type(e).__name__}: {e} ===", flush=True)
            by_depth[str(depth)] = {"depth": depth, "tok_per_sec": None,
                                    "tok_per_sec_stddev": None,
                                    "error": f"{type(e).__name__}: {e}"}

    return {
        "tok_per_sec": base["tok_per_sec"],
        "tok_per_sec_stddev": base["tok_per_sec_stddev"],
        "n_gen": n_gen,
        "n_threads": base["n_threads"],
        "by_depth": by_depth,
    }


def verify_quantization(hf_dir: str, llama_cpp_dir: str, text_file: str,
                        work_dir: str, n_ctx: int = 512) -> dict:
    fp16_path = os.path.join(work_dir, "model-f16.gguf")
    q4_0_path = os.path.join(work_dir, "model-q4_0.gguf")
    convert_to_gguf(hf_dir, fp16_path, llama_cpp_dir, outtype="f16")
    quantize_gguf(fp16_path, q4_0_path, llama_cpp_dir, qtype="Q4_0")

    fp16_ppl = measure_perplexity(fp16_path, llama_cpp_dir, text_file, n_ctx)
    q4_0_ppl = measure_perplexity(q4_0_path, llama_cpp_dir, text_file, n_ctx)
    delta_pct = perplexity_delta_pct(fp16_ppl, q4_0_ppl)
    return {
        "fp16_ppl": fp16_ppl, "q4_0_ppl": q4_0_ppl, "delta_pct": delta_pct,
        "passes_threshold": delta_pct < MAX_PPL_DELTA_PCT,
        "fp16_gguf": fp16_path, "q4_0_gguf": q4_0_path,
    }


# --------------------------------------------------------------------- cli ---

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="daedalus-150m")
    p.add_argument("--out-dir", default="runs/export")
    p.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", "vendor/llama.cpp"))
    p.add_argument("--setup-llama-cpp", action="store_true")
    p.add_argument("--ppl-text-file", default=None,
                   help="text file for the fp16-vs-Q4_0 perplexity check; "
                        "GGUF conversion/quantization is skipped without it")
    p.add_argument("--n-ctx", type=int, default=512)
    p.add_argument("--bench-n-gen", type=int, default=128,
                   help="tokens to decode for the llama-bench CPU tok/s "
                        "measurement; 0 skips it")
    args = p.parse_args()

    if args.setup_llama_cpp:
        setup_llama_cpp(args.llama_cpp_dir)

    hf_dir = os.path.join(args.out_dir, "hf")
    export_hf_model(args.checkpoint, args.config, hf_dir)
    export_tokenizer(hf_dir)
    print(f"exported HF model to {hf_dir}")

    if not args.ppl_text_file:
        print("no --ppl-text-file given; skipping GGUF conversion/quantization check")
        return

    result = verify_quantization(hf_dir, args.llama_cpp_dir, args.ppl_text_file,
                                 args.out_dir, args.n_ctx)
    if args.bench_n_gen > 0:
        result["decode_speed"] = measure_decode_speed(
            result["q4_0_gguf"], args.llama_cpp_dir, n_gen=args.bench_n_gen)
    print(json.dumps(result, indent=2))
    with open(os.path.join(args.out_dir, "quantization_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    # Rewrite the card now that the Q4_0 delta and the CPU decode speed exist.
    # The decode number is the project's headline claim, so a card that shipped
    # without it would omit the one figure a reader most wants to check.
    write_model_card(hf_dir, args.config,
                     milestone=_load_milestone(args.checkpoint),
                     run_name=_run_name_of(args.checkpoint),
                     quantization=result)
    if not result["passes_threshold"]:
        print(f"WARNING: fp16-vs-Q4_0 perplexity delta {result['delta_pct']:.2f}% "
             f"exceeds the {MAX_PPL_DELTA_PCT}% threshold")


if __name__ == "__main__":
    _cli()
