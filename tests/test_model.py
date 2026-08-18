"""Tests for daedalus/model.py and daedalus/config.py.

These did not exist before the issue #4 audit. AGENT.md SS3 asserts the model is
"CPU-tested" (parameter counts match analytically, incremental decoding matches a
full forward to 1e-6) but no such test lived in this repo -- only
tests/test_loss_chunking.py, which covers the loss head and nothing else.

Run: python -m pytest tests/test_model.py -v
"""
import dataclasses
import math

import pytest
import torch

from daedalus.config import PRESETS, DaedalusConfig, _interleave
from daedalus.model import Daedalus, ShortConv, apply_rope, build_rope_cache


# ------------------------------------------------------------------ config ---

def test_layer_pattern_matches_the_locked_blueprint_string():
    """AGENT.md SS2 locks the interleave as `ccccAccAcAcAcAccAc`. If this
    changes, every trained checkpoint and the GGUF layer_types change with it."""
    cfg = PRESETS["daedalus-150m"]
    pattern = "".join("A" if t == "full_attention" else "c" for t in cfg.layer_types)
    assert pattern == "ccccAccAcAcAcAccAc"
    assert len(cfg.layer_types) == 18
    assert cfg.n_attn_layers == 6


def test_interleave_never_puts_attention_in_the_first_two_blocks():
    for n_blocks, n_attn in [(18, 6), (24, 8), (16, 6), (20, 5)]:
        types = _interleave(n_blocks, n_attn)
        assert types[:2] == ["conv", "conv"]
        assert sum(t == "full_attention" for t in types) == n_attn


def test_interleave_all_attention_is_the_dense_twin():
    assert _interleave(24, 24) == ["full_attention"] * 24


def test_shipped_config_matches_the_locked_dims():
    cfg = PRESETS["daedalus-150m"]
    assert (cfg.hidden_size, cfg.block_ff_dim) == (768, 2048)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (12, 4)
    assert cfg.head_dim == 64
    assert cfg.vocab_size == 49152          # SmolLM2 tokenizer
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.tie_word_embeddings is True
    assert cfg.max_position_embeddings == 2048


def test_param_count_matches_the_real_module():
    """The analytic calculator is what the blueprint's param budget is based
    on; if it drifts from the actual model, the budget is fiction."""
    for name in ("tiny", "daedalus-150m", "dense-150m"):
        cfg = PRESETS[name]
        model = Daedalus(cfg)
        assert model.num_params() == cfg.param_count()["total"], name


def test_dense_twin_is_param_matched_to_the_hybrid():
    """abl-arch compares them at matched params; >2% apart would confound it."""
    hybrid = PRESETS["daedalus-150m"].param_count()["total"]
    dense = PRESETS["dense-150m"].param_count()["total"]
    assert abs(dense - hybrid) / hybrid < 0.02


def test_hf_dict_strips_training_only_keys_and_sets_smollm2_token_ids():
    d = PRESETS["daedalus-150m"].to_hf_dict()
    for k in ("z_loss", "logit_softcap", "num_attention_blocks",
              "loss_chunk_size", "gradient_checkpointing"):
        assert k not in d
    # SmolLM2 maps bos/eos/unk/pad all to `<|endoftext|>` (id 0). Llama-style
    # 1/2 would be wrong -- id 2 is `<|im_end|>`, a chat special token.
    assert d["bos_token_id"] == d["eos_token_id"] == d["pad_token_id"] == 0
    assert d["architectures"] == ["Lfm2ForCausalLM"]
    assert d["model_type"] == "lfm2"


def test_config_rejects_quantization_hostile_shapes():
    with pytest.raises(AssertionError):
        DaedalusConfig(block_ff_dim=2000)       # not %256
    with pytest.raises(AssertionError):
        DaedalusConfig(vocab_size=50000)        # not %256
    with pytest.raises(AssertionError):
        DaedalusConfig(num_attention_heads=12, num_key_value_heads=5)  # not divisible


# -------------------------------------------------------------------- rope ---

def test_rope_cache_is_built_beyond_the_training_context():
    """4x max_position_embeddings, so the blueprint's decay-phase 8K context
    extension needs no cache rebuild."""
    cfg = PRESETS["daedalus-150m"]
    model = Daedalus(cfg)
    assert model.rope_cos.shape[0] >= 4 * cfg.max_position_embeddings
    assert model.rope_cos.shape[1] == cfg.head_dim


def test_rope_preserves_norm_and_relative_position():
    cos, sin = build_rope_cache(head_dim=16, max_pos=32, theta=1e6)
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    qr, kr = apply_rope(q, k, cos[:8], sin[:8])
    # rotation is norm-preserving
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    # and the dot product depends only on the offset between positions
    q1 = torch.randn(1, 1, 1, 16)
    k1 = torch.randn(1, 1, 1, 16)
    def dot(i, j):
        a, _ = apply_rope(q1, q1, cos[i:i + 1], sin[i:i + 1])
        _, b = apply_rope(k1, k1, cos[j:j + 1], sin[j:j + 1])
        return (a * b).sum().item()
    assert dot(0, 3) == pytest.approx(dot(5, 8), abs=1e-4)


# --------------------------------------------------------------- short conv ---

def test_short_conv_is_causal():
    """No token may see the future: changing a later input must not move an
    earlier output. This is the property dense packing relies on."""
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    conv = ShortConv(cfg).eval()
    x = torch.randn(1, 12, cfg.hidden_size)
    y1, _ = conv(x)
    x2 = x.clone()
    x2[:, 8:] += 5.0
    y2, _ = conv(x2)
    assert torch.allclose(y1[:, :8], y2[:, :8], atol=1e-5)
    assert not torch.allclose(y1[:, 8:], y2[:, 8:], atol=1e-3)


def test_short_conv_state_carries_across_a_split():
    """Streaming the sequence in two chunks with the state handed over must
    equal processing it whole -- the contract incremental decoding relies on."""
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    conv = ShortConv(cfg).eval()
    x = torch.randn(1, 10, cfg.hidden_size)
    with torch.no_grad():
        full, _ = conv(x)
        a, state = conv(x[:, :6])
        b, _ = conv(x[:, 6:], state)
    assert torch.allclose(full, torch.cat([a, b], dim=1), atol=1e-5)


# ------------------------------------------------------------------- model ---

def test_incremental_decoding_matches_a_full_forward():
    """The claim AGENT.md SS3 makes about model.py, actually asserted: feeding
    tokens one at a time through the caches must reproduce the full forward
    pass. Covers both the conv state and the attention KV cache."""
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full, _, _ = model(ids, targets=None, return_logits=True)
        caches = None
        stepwise = []
        for t in range(ids.size(1)):
            logits, _, caches = model(ids[:, t:t + 1], caches=caches)
            stepwise.append(logits[:, -1])
    incremental = torch.stack(stepwise, dim=1)
    assert torch.allclose(full, incremental, atol=1e-5)


def test_prefill_then_decode_matches_a_full_forward():
    """The real serving path: prefill a prompt, then decode one token."""
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 9))

    with torch.no_grad():
        full, _, _ = model(ids, targets=None, return_logits=True)
        _, _, caches = model(ids[:, :8], caches=None)
        last, _, _ = model(ids[:, 8:9], caches=caches)
    assert torch.allclose(full[:, 8], last[:, 0], atol=1e-5)


def test_residual_output_projections_are_zero_initialized():
    """Speedrun-proven stability trick: every block is an identity at init, so
    the residual stream starts clean regardless of depth."""
    model = Daedalus(PRESETS["daedalus-150m"])
    for blk in model.layers:
        assert torch.all(blk.feed_forward.w2.weight == 0)
        if blk.layer_type == "conv":
            assert torch.all(blk.conv.out_proj.weight == 0)
        else:
            assert torch.all(blk.self_attn.out_proj.weight == 0)


def test_model_is_an_identity_over_the_residual_stream_at_init():
    """Follows from the zero-init above: the hidden state entering the final
    norm equals the raw embedding."""
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        x = model.embed_tokens(ids)
        h = x
        for blk in model.layers:
            h, _ = blk(h, model.rope_cos, model.rope_sin, None)
    assert torch.allclose(h, x, atol=1e-6)


def test_tied_embeddings_share_storage_with_the_head():
    cfg = PRESETS["daedalus-150m"]
    model = Daedalus(cfg)
    assert model.lm_head is None                 # head *is* the embedding
    assert "lm_head.weight" not in model.state_dict()


def test_untied_config_materializes_a_separate_head():
    cfg = dataclasses.replace(PRESETS["tiny"], tie_word_embeddings=False,
                              layer_types=None)
    model = Daedalus(cfg)
    assert model.lm_head is not None
    assert "lm_head.weight" in model.state_dict()


def test_generate_extends_the_sequence_and_stays_in_vocab():
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    out = model.generate(ids, max_new_tokens=5, temperature=0.8, top_k=10)
    assert out.shape == (2, 9)
    assert torch.equal(out[:, :4], ids)          # prompt preserved
    assert int(out.max()) < cfg.vocab_size and int(out.min()) >= 0


def test_num_params_non_embedding_excludes_the_embedding_table():
    cfg = PRESETS["daedalus-150m"]
    model = Daedalus(cfg)
    assert (model.num_params() - model.num_params(non_embedding=True)
            == cfg.vocab_size * cfg.hidden_size)


def test_attention_is_causal_in_the_full_forward():
    """Changing a later token must not move an earlier position's logits."""
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    ids2 = ids.clone()
    ids2[0, 7] = (ids2[0, 7] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _, _ = model(ids, targets=None, return_logits=True)
        b, _, _ = model(ids2, targets=None, return_logits=True)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)
    assert not torch.allclose(a[:, 7:], b[:, 7:], atol=1e-4)


# ------------------------------------------------ initial-loss behaviour ---

@pytest.mark.parametrize("preset", ["daedalus-150m", "dense-150m"])
def test_initial_loss_is_hidden_times_init_std_not_ln_vocab(preset):
    """Pins a real, deliberate consequence of the init that is easy to mistake
    for a bug (and easy to break while "fixing" it).

    Every residual-output projection is zero-initialized, so at step 0 each
    block is the identity and the final hidden state *is* the token embedding.
    Embeddings are tied, so the largest logit is the input token's own
    self-similarity: RMSNorm scales `E[t]` by `1/rms(E[t]) = 1/init_std`, and
    the self logit becomes `||E[t]||^2 / init_std = hidden_size * init_std`.

    The model therefore starts by confidently predicting *the current token*
    rather than a uniform distribution, so initial CE is `hidden * init_std`
    (15.36 / 12.80) rather than `ln(vocab)` = 10.80. It unlearns this within
    the first few hundred steps of a ~1e5-step run.

    It also means the two `abl-arch` arms start at systematically different
    losses purely because they are different widths (768 vs 640) -- not seed
    noise, and not an architecture effect. Worth knowing before reading the
    first few hundred steps of either arm's loss curve as a difference between
    the architectures.
    """
    cfg = PRESETS[preset]
    torch.manual_seed(0)
    model = Daedalus(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 256))

    saved = cfg.z_loss
    try:
        cfg.z_loss = 0.0                       # measure CE alone
        with torch.no_grad():
            _, ce, _ = model(ids, targets=ids)
    finally:
        cfg.z_loss = saved

    predicted = cfg.hidden_size * cfg.initializer_range
    assert ce.item() == pytest.approx(predicted, rel=0.02), (
        f"initial CE {ce.item():.3f} != hidden*init_std {predicted:.3f}; the "
        "zero-init/tied-embedding interaction changed")
    assert ce.item() > math.log(cfg.vocab_size)


def test_rope_cache_overrun_names_the_problem():
    """A sequence longer than the RoPE cache used to surface as a broadcast
    error from inside torch.compile's fake-tensor tracing, naming neither the
    config nor the sequence length -- `cos[off:off + T]` past the end silently
    yields a shorter tensor instead of raising. Found by smoke-testing hero.py
    with the `tiny` preset, whose 256-position config gives a 1024-row cache
    against train.py's default seq_end of 2048."""
    cfg = dataclasses.replace(PRESETS["tiny"], max_position_embeddings=8)
    model = Daedalus(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4 * 8 + 1))
    with pytest.raises(ValueError, match="RoPE cache covers"):
        model(ids, targets=None, return_logits=True)

    # And the boundary still works.
    ok = torch.randint(0, cfg.vocab_size, (1, 4 * 8))
    model(ok, targets=None, return_logits=True)
