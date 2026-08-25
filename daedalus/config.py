"""Daedalus model configurations.

Every config maps 1:1 onto HuggingFace `Lfm2Config` fields so that checkpoints
can be saved as `Lfm2ForCausalLM` and converted to GGUF with the stock
llama.cpp converter. Do not rename fields.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


def _interleave(n_blocks: int, n_attn: int) -> List[str]:
    """Spread `n_attn` attention blocks as evenly as possible through the stack.

    LFM2-350M uses 10 conv + 6 attention over 16 blocks with attention placed
    toward the back of the network. We keep that spirit: no attention in the
    first two blocks (cheap local mixing first), then even spacing.
    """
    assert 0 < n_attn <= n_blocks
    if n_attn == n_blocks:                      # the dense-transformer twin
        return ["full_attention"] * n_blocks
    types = ["conv"] * n_blocks
    # candidate positions, excluding the first two blocks
    start = 2 if n_blocks > 6 else 0
    span = n_blocks - start
    for i in range(n_attn):
        pos = start + round((i + 1) * span / (n_attn + 1))
        pos = min(max(pos, start), n_blocks - 1)
        while types[pos] == "full_attention":  # avoid collisions
            pos = (pos + 1) % n_blocks
        types[pos] = "full_attention"
    return types


@dataclass
class DaedalusConfig:
    # --- identity (read by the llama.cpp converter) ---
    architectures: List[str] = field(default_factory=lambda: ["Lfm2ForCausalLM"])
    model_type: str = "lfm2"

    # --- shape ---
    hidden_size: int = 768
    num_hidden_layers: int = 18
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 64
    block_ff_dim: int = 2048          # SwiGLU inner dim (must be %256 for k-quants)
    vocab_size: int = 49152           # SmolLM2 tokenizer; 192 * 256
    max_position_embeddings: int = 2048

    # --- layer layout ---
    layer_types: Optional[List[str]] = None   # filled in __post_init__
    num_attention_blocks: int = 6
    conv_L_cache: int = 3                     # depthwise conv kernel width
    conv_bias: bool = False

    # --- numerics ---
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    # --- tokenizer (stripped before HF save) ---
    # `None` means the SmolLM2 default every existing run uses, so no shipped
    # config changes behaviour. A path or Hub name here is what lets Phase 4's
    # candidate vocabularies be trained and evaluated without a second copy of
    # the model definition -- `vocab_size` alone says how many rows the
    # embedding has, not which tokenizer produced the ids that index them, and
    # a shard packed under one vocabulary read under another is a silent
    # mis-embedding rather than an error.
    tokenizer: Optional[str] = None

    # --- training-only knobs (stripped before HF save) ---
    z_loss: float = 1e-4
    logit_softcap: float = 0.0        # 0 disables; use either this or z_loss
    # Tokens per chunk in the fused loss head. The full [N, vocab] logit tensor
    # is never materialised during training; peak loss-head memory is
    # loss_chunk_size * vocab_size instead of batch * seq * vocab_size.
    # 0 disables chunking and restores the single-shot path.
    loss_chunk_size: int = 1024
    # Recompute block activations in backward instead of storing them. Trades
    # roughly one extra forward pass (~30% step time) for a large drop in
    # activation memory, which is what gates batch size on 16 GB cards.
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = _interleave(self.num_hidden_layers,
                                           self.num_attention_blocks)
        assert len(self.layer_types) == self.num_hidden_layers
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.block_ff_dim % 256 == 0, "keep FFN dim %256 for clean quantization"
        assert self.vocab_size % 256 == 0

    # ---- derived ----
    @property
    def n_attn_layers(self) -> int:
        return sum(t == "full_attention" for t in self.layer_types)

    def param_count(self) -> dict:
        h, V = self.hidden_size, self.vocab_size
        emb = V * h
        per_conv = 3 * h * h + self.conv_L_cache * h + h * h      # in_proj, dwconv, out_proj
        qo = self.num_attention_heads * self.head_dim
        kv = self.num_key_value_heads * self.head_dim
        per_attn = h * qo + 2 * (h * kv) + qo * h + 2 * self.head_dim  # +qk norms
        per_ffn = 3 * h * self.block_ff_dim
        norms = 2 * h
        n_attn = self.n_attn_layers
        n_conv = self.num_hidden_layers - n_attn
        blocks = (n_conv * per_conv + n_attn * per_attn
                  + self.num_hidden_layers * (per_ffn + norms))
        total = emb + blocks + h + (0 if self.tie_word_embeddings else V * h)
        return {
            "embedding": emb,
            "blocks": blocks,
            "total": total,
            "non_embedding": total - emb,
            "embedding_frac": emb / total,
            "q4_0_MB": total * 4.5 / 8 / 1e6,   # ~4.5 bits/weight incl. scales
        }

    def to_hf_dict(self) -> dict:
        d = asdict(self)
        # `tokenizer` is stripped with the other training-only knobs: config.json
        # is read by llama.cpp's converter, which fingerprints the tokenizer from
        # the tokenizer files beside it, and an unexpected key there is a
        # gratuitous difference from every `Lfm2Config` the converter has seen.
        for k in ("z_loss", "logit_softcap", "num_attention_blocks",
                  "loss_chunk_size", "gradient_checkpointing", "tokenizer"):
            d.pop(k, None)
        # SmolLM2's tokenizer maps bos/eos/unk/pad all to `<|endoftext|>` (id 0);
        # it has no distinct pad token. Verified against the real tokenizer_config
        # (HuggingFaceTB/SmolLM2-135M) -- do not "correct" these to Llama-style
        # 1/2, that was a bug (id 2 is `<|im_end|>`, a chat special token).
        d["bos_token_id"] = 0
        d["eos_token_id"] = 0
        d["pad_token_id"] = 0
        d["torch_dtype"] = "bfloat16"
        return d


# ---------------------------------------------------------------- presets ----

PRESETS = {
    # the shipped model
    "daedalus-150m": DaedalusConfig(),

    # ablation: same params, all attention, deeper+narrower (the dense twin)
    "dense-150m": DaedalusConfig(
        hidden_size=640, num_hidden_layers=24, num_attention_heads=10,
        num_key_value_heads=2, head_dim=64, block_ff_dim=2304,
        num_attention_blocks=24,
    ),

    # ablation: depth
    "daedalus-150m-deep": DaedalusConfig(
        hidden_size=640, num_hidden_layers=24, num_attention_heads=10,
        num_key_value_heads=2, block_ff_dim=1792, num_attention_blocks=8,
    ),

    # CPU dev / smoke tests (trains on a laptop in minutes)
    "tiny": DaedalusConfig(
        hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, block_ff_dim=256,
        vocab_size=512, num_attention_blocks=2, max_position_embeddings=256,
    ),
}


# ------------------------------------------------- Phase 4 tokenizer probes ----
# One preset per candidate vocabulary, identical in every other field, so a
# tiny-model comparison between vocabularies varies the vocabulary and nothing
# else. Generated rather than written out four times: four hand-copied configs
# are four chances for one of them to differ in a field nobody re-reads, and
# the whole comparison rests on them being otherwise the same.
#
# The shape is the shipped model's layout narrowed rather than an unrelated
# small model: 18 blocks with 6 attention, hidden 512 instead of 768. Two
# properties matter and the obvious cheaper shapes have neither.
#
# **The embedding fraction has to be realistic**, because it is the thing under
# test. Shipped, `token_embd` is 25.1% of parameters. These probes run 17.6%
# (24,576) to 29.9% (49,152), bracketing it -- so a vocabulary's cost lands
# roughly where it lands in the real artifact. A shallower probe would push the
# embedding past 40% and turn a vocabulary comparison into a comparison of how
# much of the model is a lookup table.
#
# **The layout is the shipped one**, so Phase 4 is not quietly answering a
# Phase 6 question. Depth, attention count and KV heads are what Phase 6
# varies; holding them at the shipped ratio here keeps the two separable.

TOKENIZER_PROBE_VOCAB_SIZES = (24576, 32768, 40960, 49152)


def tokenizer_probe_config(vocab_size: int, tokenizer: Optional[str] = None
                           ) -> DaedalusConfig:
    """The shared Phase 4 probe shape at one vocabulary size."""
    return DaedalusConfig(
        hidden_size=512, num_hidden_layers=18, num_attention_heads=8,
        num_key_value_heads=2, head_dim=64, block_ff_dim=1536,
        num_attention_blocks=6, vocab_size=vocab_size,
        max_position_embeddings=1024, tokenizer=tokenizer,
    )


def tokenizer_probe_preset_name(vocab_size: int) -> str:
    return f"tok-probe-{vocab_size}"


for _vocab in TOKENIZER_PROBE_VOCAB_SIZES:
    PRESETS[tokenizer_probe_preset_name(_vocab)] = tokenizer_probe_config(_vocab)
del _vocab


# ------------------------------------------- Phase 5 conv-death probe shape ----
# The positive control for channel death, at the shape the 2026-08-11 mechanism
# experiment established (`runs/preflight/conv-death-fix-validated.md`): hidden
# 256, 9 layers at the shipped 2:1 conv:attention ratio, run at Muon lr 0.15.
#
# The lr is the accelerant, not a different mechanism. Muon's decay is
# `w *= (1 - lr*wd)`, so lr sets the clock while `wd` alone decides the race --
# 0.15 makes in ~600 steps what `hero` took ~10,000 steps at 0.02 to show. The
# arms differ in `wd`, which is the variable, so the accelerant is shared.
#
# **The vocabulary is the shipped one, which the CPU-era control could not
# afford.** That probe remapped to 8,192 tokens by frequency rank to stay
# runnable on a laptop; on the GPU that remap only adds a step to get wrong, and
# the death is a property of decay on the conv projections, which no vocabulary
# touches. The cost is an embedding-heavy probe (~80% of parameters), which
# matters for a loss comparison and not for a within-model ablation delta.

PRESETS["conv-probe"] = DaedalusConfig(
    hidden_size=256, num_hidden_layers=9, num_attention_heads=4,
    num_key_value_heads=2, head_dim=64, block_ff_dim=768,
    num_attention_blocks=3, max_position_embeddings=256,
)


# ------------------------------------------ Phase 6 architecture Stage A ------
# The stage-A screen: attention fraction crossed with KV heads, everything else
# held. `daedalus/arch_space.py` decides which *shapes* are comparable at all;
# this is the fifteen-point grid those rules leave once depth, width and FFN are
# fixed, generated so the two knobs under test are the only fields that differ.
#
# **`block_ff_dim` is held fixed here, and that is the opposite of what
# `arch_space.matched_candidate` does on purpose.** Parameter-matching by
# solving the FFN width is right at the scale that module screens and wrong at
# this one: one `block_ff_dim` step is `3 * hidden * layers * 256` parameters,
# which here is 9.44M against a ~105M model -- 9.0%. Solving per arm would
# therefore snap arms as much as 4.5% away from each other, while simply
# holding the FFN fixed leaves the whole grid within 1.5% of its own midpoint,
# because the only parameter difference left is a conv block costing 1.05M
# where an attention block costs 0.59M-0.79M. The cheap knob is the accurate
# one at proxy scale (`tests/test_architecture_sweep.py` asserts both halves).
#
# **Depth 24, not the shipped 18.** The grid spans attention fractions from the
# shipped 1/3 down to 1/12, and 18 layers cannot express those as five distinct
# counts -- 1/9 and 1/12 both round to 2. At 24 they are 8/6/4/3/2, so every
# fraction in the plan's range is a different model.
#
# **Width 512 and the shipped 49,152 vocabulary and 2,048 context.** The
# embedding lands at 24% of parameters against the shipped model's 25.1%, so
# attention's share of the model is roughly what it is in the real artifact;
# and KV cost is only meaningful against a context length the shipped model
# also has, so the screen runs at the same 2,048.
ARCH_PROBE_DEPTH = 24
ARCH_PROBE_HIDDEN = 512
ARCH_PROBE_HEAD_DIM = 64
ARCH_PROBE_FF_DIM = 1536

#: Attention layers out of 24: the plan's 1/3, 1/4, 1/6, 1/9 and 1/12.
ARCH_PROBE_ATTENTION_BLOCKS = (8, 6, 4, 3, 2)

#: GQA groups. Powers of two that divide the eight query heads; 8 (i.e. plain
#: MHA) is left out because it only clears the KV ceiling at the sparsest
#: attention counts, where it buys the same cache as a denser stack with fewer
#: KV heads and is dominated by it on every other axis.
ARCH_PROBE_KV_HEADS = (1, 2, 4)

#: The shipped model's own point in this grid -- attention every third layer,
#: four KV heads -- and therefore the control every other arm is read against.
ARCH_PROBE_CONTROL = (8, 4)


def arch_probe_config(num_attention_blocks: int,
                      num_key_value_heads: int) -> DaedalusConfig:
    """One stage-A arm. The two arguments are the only fields that vary."""
    return DaedalusConfig(
        hidden_size=ARCH_PROBE_HIDDEN,
        num_hidden_layers=ARCH_PROBE_DEPTH,
        num_attention_heads=ARCH_PROBE_HIDDEN // ARCH_PROBE_HEAD_DIM,
        num_key_value_heads=num_key_value_heads,
        head_dim=ARCH_PROBE_HEAD_DIM,
        block_ff_dim=ARCH_PROBE_FF_DIM,
        num_attention_blocks=num_attention_blocks,
        max_position_embeddings=2048,
    )


def arch_probe_preset_name(num_attention_blocks: int,
                           num_key_value_heads: int) -> str:
    return f"arch-a{num_attention_blocks}-kv{num_key_value_heads}"


for _blocks in ARCH_PROBE_ATTENTION_BLOCKS:
    for _kv in ARCH_PROBE_KV_HEADS:
        PRESETS[arch_probe_preset_name(_blocks, _kv)] = arch_probe_config(
            _blocks, _kv)
del _blocks, _kv


# ------------------------------------------ Phase 6 architecture Stage B ------
# The same fifteen grid points at the scale a recommendation is allowed to rest
# on: ~150M parameters rather than ~105M. Stage A ranks shapes cheaply; stage B
# re-runs the survivors where the plan asks for them.
#
# **Depth 24 and head_dim 64 are carried over, not re-chosen.** KV cache is
# `2 * kv_heads * head_dim * attention_layers * 2` bytes per context token -- it
# depends on exactly those three fields and on nothing else this preset varies.
# Holding them means every stage-B arm costs the identical cache to the stage-A
# arm it re-runs, so the KV column the stage-A screen selected on is still the
# KV column stage B is measured on. A stage B at a different depth would re-rank
# candidates whose cache cost had moved underneath the screen that chose them,
# and the saving is the entire reason to run a hybrid.
#
# **Width 768 is forced rather than picked.** `num_attention_heads` is
# `hidden_size / head_dim` and every KV-head count in the grid has to divide it:
# 640 gives ten heads and 4 does not divide 10, and 1024 overshoots badly (its
# conv block alone is 5.25M against 768's 2.36M). 768 is the only 256-aligned
# width between stage A's 512 and that overshoot, and it is the shipped model's
# own width.
#
# **`block_ff_dim` 1280 puts the control at 158.9M against the shipped model's
# 160.5M** -- 0.97% low, inside `arch_space.PARAM_MATCH_TOLERANCE` -- at a
# 1.67x FFN ratio, inside the validity band. The next step up (1536) lands at
# 173.1M, 7.9% high.
#
# **The FFN is held fixed across arms, and the residual is worse here than at
# stage A rather than better.** One `block_ff_dim` step is
# `3 * 768 * 24 * 256` = 14.16M parameters, 8.8% of the model, so solving per
# arm would snap arms up to 4.4% apart to correct a spread of 4.5% -- the same
# trade stage A refused, and no better at this width. Holding it leaves the grid
# 2.2% either side of its midpoint against stage A's 1.5%, because widening
# raises the conv-block premium (2.36M against an attention block's 1.28M-1.57M)
# faster than it raises the model. So the parameter discount the scoring applies
# is *more* load-bearing at stage B, not less; `parameter_spread` is written
# into every stage-B artifact for that reason.
ARCH_STAGEB_HIDDEN = 768
ARCH_STAGEB_FF_DIM = 1280


def arch_stageb_config(num_attention_blocks: int,
                       num_key_value_heads: int) -> DaedalusConfig:
    """One stage-B arm: a stage-A grid point at ~150M parameters."""
    return DaedalusConfig(
        hidden_size=ARCH_STAGEB_HIDDEN,
        num_hidden_layers=ARCH_PROBE_DEPTH,
        num_attention_heads=ARCH_STAGEB_HIDDEN // ARCH_PROBE_HEAD_DIM,
        num_key_value_heads=num_key_value_heads,
        head_dim=ARCH_PROBE_HEAD_DIM,
        block_ff_dim=ARCH_STAGEB_FF_DIM,
        num_attention_blocks=num_attention_blocks,
        max_position_embeddings=2048,
    )


def arch_stageb_preset_name(num_attention_blocks: int,
                            num_key_value_heads: int) -> str:
    return f"arch-b-a{num_attention_blocks}-kv{num_key_value_heads}"


for _blocks in ARCH_PROBE_ATTENTION_BLOCKS:
    for _kv in ARCH_PROBE_KV_HEADS:
        PRESETS[arch_stageb_preset_name(_blocks, _kv)] = arch_stageb_config(
            _blocks, _kv)
del _blocks, _kv


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        p = cfg.param_count()
        print(f"{name:22s} {p['total']/1e6:7.1f}M total  "
              f"{p['non_embedding']/1e6:6.1f}M non-emb  "
              f"emb {p['embedding_frac']*100:4.1f}%  "
              f"~{p['q4_0_MB']:5.1f}MB @q4_0  "
              f"layers={cfg.layer_types.count('conv')}c/{cfg.n_attn_layers}a")
