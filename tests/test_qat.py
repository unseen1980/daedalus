"""Tests for `daedalus.qat`.

The load-bearing claim of QAT is that the grid we train against is *exactly*
the grid `llama-quantize` lands on. If it isn't, QAT optimizes for one lattice
and shipping snaps to another -- strictly worse than not doing QAT at all, and
silently so. So this file checks the implementation two independent ways:

1. against a literal, scalar transcription of `quantize_row_q4_0_ref`, and
2. against the **real shipped C**, by calling `ggml_quantize_chunk` in
   llama.cpp's own `libggml-base.so` through ctypes (skipped if absent).

(2) is the one that actually settles it; (1) keeps the test suite meaningful
on a box without llama.cpp built.
"""
import ctypes
import glob
import json
import math
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from daedalus import qat
from daedalus import qat as qat_mod
from daedalus.config import DaedalusConfig, PRESETS
from daedalus.model import Daedalus


# --------------------------------------------------------- C reference ---

def _q4_0_reference(block):
    """Literal transcription of ggml-quants.c::quantize_row_q4_0_ref for one
    32-element block, returning the dequantized values."""
    amax, vmax = 0.0, 0.0
    for v in block:
        if amax < abs(v):
            amax, vmax = abs(v), v
    d = vmax / -8.0
    inv_d = (1.0 / d) if d else 0.0
    d_fp16 = float(torch.tensor(d, dtype=torch.float32).half())
    out = []
    for v in block:
        # (int8_t) cast truncates toward zero; the argument is always > 0 here.
        q = min(15, int(v * inv_d + 8.5))
        out.append((q - 8) * d_fp16)
    return out


def test_q4_0_matches_scalar_c_reference():
    torch.manual_seed(0)
    w = torch.randn(7, 32 * 5) * 0.02
    got = qat.q4_0_qdq(w)
    for i in range(w.shape[0]):
        for b in range(5):
            block = w[i, b * 32:(b + 1) * 32].tolist()
            expected = _q4_0_reference(block)
            actual = got[i, b * 32:(b + 1) * 32].tolist()
            assert actual == pytest.approx(expected, abs=1e-7), (i, b)


# ------------------------------------------------- real llama.cpp check ---

def _find_libggml():
    """The real `libggml-base.so`, wherever this box built llama.cpp.

    These four tests are the ones that certify our Q4_0 grid *is*
    llama.cpp's -- the single claim QAT rests on. They skip when the library
    is absent, which is correct on a laptop and dangerous on a box that has
    llama.cpp and simply keeps it somewhere the list did not name: the suite
    then reports green having never checked the grid at all. That is what
    happened here. llama.cpp lives at `/opt/llama.cpp` on the Vast image
    (`gguf_eval.py` and the approved wrapper both point there), which no
    pattern matched, so every grid test silently skipped through Phase 2 and
    into the Phase 3 fix that changed the quantizer.
    """
    override = os.environ.get("DAEDALUS_LIBGGML")
    if override and os.path.exists(override):
        return override
    for pat in ("/opt/llama.cpp/build/bin/libggml-base.so",
                "/tmp/llama.cpp/build/bin/libggml-base.so",
                "vendor/llama.cpp/build/bin/libggml-base.so",
                "/opt/llama.cpp/build/**/libggml-base.so",
                "/tmp/llama.cpp/build/**/libggml-base.so"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


def _ggml_quantize(lib, ggml_type, w, block_bytes):
    """Quantize `w` (2D fp32 tensor) with the real ggml routine, returning raw
    bytes."""
    rows, cols = w.shape
    src = (ctypes.c_float * w.numel())(*w.flatten().tolist())
    nblocks = rows * cols // 32
    dst = (ctypes.c_ubyte * (nblocks * block_bytes))()
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_quantize_chunk(ggml_type, src, ctypes.cast(dst, ctypes.c_void_p),
                            0, rows, cols, None)
    return bytes(dst)


def _dequant_q4_0(raw, rows, cols):
    """Unpack ggml's Q4_0 block layout: fp16 scale, then 16 bytes of nibbles
    where byte j holds element j in the low nibble and element j+16 in the
    high nibble."""
    out = torch.empty(rows * cols, dtype=torch.float32)
    nblocks = rows * cols // 32
    for b in range(nblocks):
        off = b * 18
        d = float(torch.frombuffer(raw[off:off + 2], dtype=torch.float16)[0])
        qs = raw[off + 2:off + 18]
        for j in range(16):
            out[b * 32 + j] = ((qs[j] & 0x0F) - 8) * d
            out[b * 32 + j + 16] = ((qs[j] >> 4) - 8) * d
    return out.reshape(rows, cols)


def _dequant_q8_0(raw, rows, cols):
    out = torch.empty(rows * cols, dtype=torch.float32)
    nblocks = rows * cols // 32
    for b in range(nblocks):
        off = b * 34
        d = float(torch.frombuffer(raw[off:off + 2], dtype=torch.float16)[0])
        qs = torch.frombuffer(raw[off + 2:off + 34], dtype=torch.int8)
        for j in range(32):
            out[b * 32 + j] = float(qs[j]) * d
    return out.reshape(rows, cols)


@pytest.mark.skipif(_find_libggml() is None, reason="llama.cpp not built here")
def test_q4_0_matches_real_llama_cpp_bit_for_bit():
    """The claim that matters: our grid IS llama.cpp's grid. Checked against
    the shipped `libggml-base.so`, not against my reading of the source."""
    lib = ctypes.CDLL(_find_libggml())
    torch.manual_seed(1)
    # Deliberately mixed scales, plus a row of near-zeros to exercise the
    # fp16-underflow path and a row with a positive extreme (negative `d`).
    w = torch.randn(6, 256) * 0.02
    w[0] *= 1e-4
    w[1] = w[1].abs()
    w[2] = -w[2].abs()
    w = w.float().contiguous()

    theirs = _dequant_q4_0(_ggml_quantize(lib, GGML_TYPE_Q4_0, w, 18), 6, 256)
    ours = qat.q4_0_qdq(w)
    assert torch.equal(ours, theirs), (ours - theirs).abs().max()


@pytest.mark.skipif(_find_libggml() is None, reason="llama.cpp not built here")
def test_q8_0_matches_real_llama_cpp_bit_for_bit():
    lib = ctypes.CDLL(_find_libggml())
    torch.manual_seed(2)
    w = (torch.randn(4, 128) * 0.05).float().contiguous()
    theirs = _dequant_q8_0(_ggml_quantize(lib, GGML_TYPE_Q8_0, w, 34), 4, 128)
    ours = qat.q8_0_qdq(w)
    assert torch.equal(ours, theirs), (ours - theirs).abs().max()


# ------------------------------------------------------ grid properties ---

def test_q4_0_is_idempotent_so_shipping_is_lossless():
    """After QAT the weights sit ON the grid, so the real quantization step
    should be a no-op. This is the property that makes QAT worth doing."""
    torch.manual_seed(3)
    w = torch.randn(5, 64) * 0.02
    once = qat.q4_0_qdq(w)
    twice = qat.q4_0_qdq(once)
    assert torch.equal(once, twice)


def test_q4_0_scale_uses_signed_absmax_not_absolute():
    """`d = max/-8` where `max` keeps its sign. A block whose extreme value is
    positive gets a NEGATIVE scale. Getting this wrong mirrors the lattice and
    is invisible in aggregate error metrics."""
    pos = torch.zeros(1, 32)
    pos[0, 0] = 1.0                      # extreme is +1 -> d = -0.125
    neg = -pos                           # extreme is -1 -> d = +0.125
    # The extreme element must round-trip exactly in both cases.
    assert qat.q4_0_qdq(pos)[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert qat.q4_0_qdq(neg)[0, 0] == pytest.approx(-1.0, abs=1e-3)
    # Zeros land on q=8, i.e. exactly 0, not on a biased level.
    assert qat.q4_0_qdq(pos)[0, 1:].abs().max() == 0.0


def test_q4_0_ties_pick_the_first_extreme_like_the_c_loop():
    """`if (amax < fabsf(v))` is a strict comparison, so the FIRST element of
    maximal magnitude wins and sets the sign of `d`."""
    a = torch.zeros(1, 32)
    a[0, 0], a[0, 1] = 3.0, -3.0         # tie in magnitude, first is positive
    b = torch.zeros(1, 32)
    b[0, 0], b[0, 1] = -3.0, 3.0
    assert torch.equal(qat.q4_0_qdq(a), _as_row(_q4_0_reference(a[0].tolist())))
    assert torch.equal(qat.q4_0_qdq(b), _as_row(_q4_0_reference(b[0].tolist())))


def _as_row(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1)


def test_q4_0_preserves_dtype_but_computes_in_fp32():
    w = (torch.randn(2, 32) * 0.02).to(torch.bfloat16)
    out = qat.q4_0_qdq(w)
    assert out.dtype == torch.bfloat16
    # Same lattice as the fp32 path, up to the bf16 store.
    ref = qat.q4_0_qdq(w.float()).to(torch.bfloat16)
    assert torch.equal(out, ref)


def test_block_size_mismatch_raises_rather_than_lying():
    with pytest.raises(ValueError, match="not divisible by the block size"):
        qat.q4_0_qdq(torch.randn(2, 33))


# --------------------------------------------------------------- STE ------

def test_straight_through_forward_is_quantized_backward_is_identity():
    torch.manual_seed(4)
    w = (torch.randn(4, 32) * 0.02).requires_grad_(True)
    fq = qat.FakeQuant("q4_0")
    out = fq(w)
    assert torch.equal(out.detach(), qat.q4_0_qdq(w.detach()))
    g = torch.randn_like(out)
    out.backward(g)
    assert torch.equal(w.grad, g)        # identity, not zero and not scaled


def test_fake_quant_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown quantization kind"):
        qat.FakeQuant("q3_k")


# ------------------------------------------------------ model plumbing ---

def _tiny_model():
    cfg = DaedalusConfig(hidden_size=128, num_hidden_layers=4, block_ff_dim=256,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         vocab_size=512, num_attention_blocks=2)
    return Daedalus(cfg)


def test_plan_targets_linears_and_embeddings_only():
    m = _tiny_model()
    plan = qat.plan_qat(m)
    kinds = {name: kind for name, _, kind in plan}
    assert kinds, "plan must not be empty"
    for name, mod, kind in plan:
        assert isinstance(mod, (nn.Linear, nn.Embedding)), name
        expected = "q8_0" if name.endswith(("embed_tokens", "lm_head")) else "q4_0"
        assert kind == expected, (name, kind)
    # Conv1d depthwise kernels ship as F32 -- training against a grid inference
    # never uses would be wrong.
    assert not any(isinstance(mod, nn.Conv1d) for _, mod, _ in plan)


def test_tied_embeddings_are_registered_exactly_once():
    m = _tiny_model()
    plan = qat.plan_qat(m)
    ids = [id(qat.master_weight(mod)) for _, mod, _ in plan]
    assert len(ids) == len(set(ids)), "a tied tensor was planned twice"


def test_enable_then_disable_restores_weights_bit_exactly():
    m = _tiny_model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    qat.enable_qat(m)
    assert qat.is_qat_active(m)
    qat.disable_qat(m)
    assert not qat.is_qat_active(m)
    after = m.state_dict()
    assert set(after) == set(before)
    for k in before:
        assert torch.equal(after[k], before[k]), k


def test_checkpoint_keys_are_unchanged_after_disable():
    """A checkpoint written after `disable_qat` must be loadable by a model
    that has never heard of QAT, or resume breaks in the last 5% of hero."""
    m, fresh = _tiny_model(), _tiny_model()
    qat.enable_qat(m)
    qat.disable_qat(m)
    assert not any("parametrizations" in k for k in m.state_dict())
    fresh.load_state_dict(m.state_dict(), strict=True)


def test_enable_is_idempotent():
    m = _tiny_model()
    first = qat.enable_qat(m)
    second = qat.enable_qat(m)
    assert first and second == {}
    qat.disable_qat(m)
    # One removal per module; no stacked transforms left behind.
    assert not qat.is_qat_active(m)


def test_optimizer_keeps_the_same_parameter_objects_across_enable():
    """Muon/AdamW state is keyed on Parameter identity. If `parametrize`
    replaced the objects, switching QAT on at 95% of hero would silently reset
    every optimizer moment."""
    from daedalus.muon import build_optimizers
    m = _tiny_model()
    muon, adamw, _ = build_optimizers(m)
    before = {id(p) for g in list(muon.param_groups) + list(adamw.param_groups)
              for p in g["params"]}
    qat.enable_qat(m)
    after = {id(p) for _, mod, _ in qat.plan_qat(m)
             for p in [qat.master_weight(mod)]}
    assert after <= before, "QAT replaced parameters the optimizer holds"


def test_forward_changes_under_qat_and_is_differentiable():
    torch.manual_seed(5)
    m = _tiny_model().eval()
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        clean = m(ids)[0].clone()
    qat.enable_qat(m)
    out = m(ids)[0]
    assert not torch.allclose(out, clean), "QAT forward should differ from fp32"
    out.sum().backward()
    grads = [qat.master_weight(mod).grad for _, mod, _ in qat.plan_qat(m)]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_quantization_error_reads_the_master_weight_not_the_quantized_view():
    """Under active QAT `mod.weight` is already on the grid, so measuring
    error against it would report ~0 by construction and the metric would look
    perfect while telling us nothing."""
    m = _tiny_model()
    off = qat.quantization_error(m)["qat_rel_rmse"]
    qat.enable_qat(m)
    on = qat.quantization_error(m)["qat_rel_rmse"]
    assert off > 0
    assert on == pytest.approx(off, rel=1e-6)


def test_a_near_dead_block_quantizes_finitely():
    """The released checkpoint's own failure, reduced to one block.

    `_safe_reciprocal` guards `d == 0`, which is where a *fully* dead channel
    lands. A channel on its way there passes through a window where the block
    absmax is denormal-small but not zero: `d = absmax / -8` is representable,
    `1/d` is not, and it overflows fp32 to inf. Every element of the block that
    is exactly zero then computes `0 * inf = NaN`, `floor(NaN + 8.5)` stays
    NaN, and the whole tensor is poisoned.

    Measured on `/root/daedalus/final/hero/checkpoint.pt`: 1,418 NaNs in
    `layers.1.feed_forward.w1`, 1,674 in `layers.1.feed_forward.w3`, 3 in
    `layers.13.feed_forward.w1` -- with every stored weight finite. A recovery
    run on that checkpoint produced a NaN loss on its first step.
    """
    block = torch.zeros(1, qat.QK4_0)
    block[0, 0] = 1e-40          # denormal: 1/(1e-40/8) overflows fp32
    out = qat.q4_0_qdq(block)
    assert torch.isfinite(out).all(), "a near-dead block poisoned the tensor"
    # It is dead, so it quantizes to zero -- the same answer llama.cpp reaches
    # under flush-to-zero, where `d` denormal makes `d ? 1/d : 0` take the
    # zero branch.
    assert torch.equal(out, torch.zeros_like(out))


def test_the_near_dead_guard_does_not_disturb_ordinary_blocks():
    """The bit-exactness certified against real llama-quantize output is the
    reason this fix has to be surgical: it may only change blocks whose
    reciprocal was not representable in the first place."""
    torch.manual_seed(3)
    w = (torch.randn(16, qat.QK4_0 * 4) * 0.02).float()
    reference = w.clone()
    out = qat.q4_0_qdq(w)
    assert torch.isfinite(out).all()
    # Recomputing by hand with the unguarded reciprocal must agree exactly.
    x = reference.reshape(16, -1, qat.QK4_0)
    signed_absmax = torch.gather(x, -1, x.abs().argmax(dim=-1, keepdim=True))
    d = signed_absmax / -8.0
    expected = (torch.clamp(torch.floor(x * (1.0 / d) + 8.5), 0, 15) - 8.0) \
        * d.half().float()
    assert torch.equal(out, expected.reshape(reference.shape))


def test_a_fully_dead_block_still_quantizes_to_zero():
    """The case the original guard was written for, kept under test."""
    out = qat.q4_0_qdq(torch.zeros(1, qat.QK4_0))
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_a_near_dead_block_does_not_poison_the_gradient():
    """The reason the guard masks before dividing rather than after: a
    `where` whose unselected branch is inf multiplies inf by a zero mask in
    backward, and `clip_grad_norm_` then scales every parameter in the model
    by a NaN total norm."""
    w = torch.zeros(2, qat.QK4_0, requires_grad=True)
    with torch.no_grad():
        w[0, 0] = 1e-40
        w[1, 0] = 0.02
    fake = qat.FakeQuant("q4_0")
    fake(w).sum().backward()
    assert torch.isfinite(w.grad).all()


def test_quantization_error_counts_tensors_and_elements_separately():
    """`qat_tensors` used to hold the *element* count under a name that says
    tensors. The two answer different questions: a coverage regression -- a
    module class the plan stopped matching, or a tie that collapsed two
    entries into one -- moves the tensor count while barely touching the
    element count."""
    m = _tiny_model()
    err = qat.quantization_error(m)
    planned = qat.plan_qat(m)
    assert err["qat_tensors"] == float(len(planned))
    assert err["qat_elements"] == float(
        sum(qat.master_weight(mod).numel() for _, mod, _ in planned))
    # The distinction is only worth logging if the numbers actually differ.
    assert err["qat_tensors"] < err["qat_elements"]


def test_the_tensor_count_matches_what_enabling_qat_registers():
    """The logged count is evidence of coverage, so it must equal the number
    of modules that really ended up on the grid."""
    m = _tiny_model()
    applied = qat.enable_qat(m)
    assert qat.quantization_error(m)["qat_tensors"] == float(len(applied))


def test_grid_id_names_both_lattices():
    """Logged beside every quantized number so a Q4_0-forward figure can never
    be read as an fp32 one."""
    assert qat.grid_id() == "q4_0/q8_0"
    assert qat.grid_id("q4_0", None) == "q4_0"


# ------------------------------------------------------------ schedule ---

@pytest.mark.parametrize("progress,frac,expected", [
    (0.0, 0.05, False), (0.94, 0.05, False), (0.95, 0.05, True),
    (1.0, 0.05, True), (0.99, 0.0, False), (0.0, 1.0, True),
])
def test_qat_active_at(progress, frac, expected):
    assert qat.qat_active_at(progress, frac) is expected


def test_qat_disabled_by_default_fraction():
    """Every job except hero runs with qat_frac=0; a stray default would put
    the sweep and abl-arch on a quantized forward and invalidate them."""
    assert qat.qat_active_at(1.0, 0.0) is False


# --------------------------------------------- checkpointing under QAT ---

def test_strip_qat_state_dict_restores_normal_keys_and_values():
    """QAT turns on in the last 5% of hero -- exactly where an interruption is
    most expensive. A checkpoint written then must still load into a plain
    model."""
    m, fresh = _tiny_model(), _tiny_model()
    plain = {k: v.clone() for k, v in m.state_dict().items()}
    qat.enable_qat(m)
    live = m.state_dict()
    assert any(k.endswith(".parametrizations.weight.original") for k in live)

    stripped = qat.strip_qat_state_dict(live)
    assert set(stripped) == set(plain)
    for k in plain:
        assert torch.equal(stripped[k], plain[k]), k
    fresh.load_state_dict(stripped, strict=True)


def test_strip_qat_state_dict_is_a_noop_when_qat_is_off():
    m = _tiny_model()
    sd = m.state_dict()
    assert qat.strip_qat_state_dict(sd).keys() == sd.keys()


def test_stripped_checkpoint_holds_the_float_master_not_the_quantized_view():
    """The master weight is what training continues from. Saving the quantized
    view instead would silently freeze the model onto the grid at the moment of
    the first QAT checkpoint."""
    m = _tiny_model()
    qat.enable_qat(m)
    stripped = qat.strip_qat_state_dict(m.state_dict())
    differs = 0
    for name, mod, _ in qat.plan_qat(m):
        key = f"{name}.weight"
        if key not in stripped:
            continue
        assert torch.equal(stripped[key], qat.master_weight(mod))
        # Zero-initialized residual projections quantize to themselves, so the
        # distinction is only observable on the tensors that actually move.
        if not torch.equal(stripped[key], mod.weight):
            differs += 1
    assert differs > 0, "saved weights were indistinguishable from the quantized view"


# ------------------------------------------ integration with train.py ---

def test_trainer_turns_qat_on_in_the_tail_and_keeps_training(tmp_path):
    """End to end on the real Trainer: QAT must engage at the right step, the
    loss must stay finite through the switch, and the checkpoint written
    afterwards must load into a plain model."""
    import train as train_mod

    args = train_mod.TrainArgs(
        run_name="qat-smoke", config="tiny", max_steps=10,
        micro_batch=2, seq_start=16, seq_end=16, tok_start=32, tok_end=64,
        compile=False,
        device="cpu", wandb_enabled=False, qat_frac=0.5,
        run_dir=str(tmp_path / "run"), ckpt_every_sec=1e9, push_every_sec=1e9,
    )
    t = train_mod.Trainer(args)

    assert not t._qat_on
    losses = []
    for _ in range(10):
        t.maybe_enable_qat()
        losses.append(t.train_step()["loss"])
    assert t._qat_on, "QAT never engaged despite qat_frac=0.5"
    assert all(math.isfinite(x) for x in losses), losses

    ckpt = tmp_path / "ck.pt"
    train_mod.save_checkpoint(str(ckpt), t.model, t.muon, t.adamw,
                              t.step, t.tokens_seen, t.cfg)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert not any("parametrizations" in k for k in payload["model"])


def test_trainer_leaves_qat_off_by_default(tmp_path):
    """sweep and abl-arch must run a plain fp32/bf16 forward. A stray default
    here would quietly invalidate the lr sweep and the headline ablation."""
    import train as train_mod
    args = train_mod.TrainArgs(
        run_name="no-qat", config="tiny", max_steps=4, micro_batch=2,
        seq_start=16, seq_end=16, tok_start=32, tok_end=64,
        compile=False, device="cpu", wandb_enabled=False,
        run_dir=str(tmp_path / "run"), ckpt_every_sec=1e9, push_every_sec=1e9,
    )
    assert args.qat_frac == 0.0
    t = train_mod.Trainer(args)
    for _ in range(4):
        t.maybe_enable_qat()
        t.train_step()
    assert not t._qat_on
    assert not qat.is_qat_active(t.model)


def _recovery_args(tmp_path, **overrides):
    """Phase 3's shape: QAT on from the very first step, not in a tail."""
    import train as train_mod
    kwargs = dict(
        run_name="recovery", config="tiny", max_steps=4, micro_batch=2,
        seq_start=16, seq_end=16, tok_start=32, tok_end=64, compile=False,
        device="cpu", wandb_enabled=False, qat_frac=1.0,
        metrics_every_steps=1, log_every_steps=10 ** 9,
        run_dir=str(tmp_path / "run"), ckpt_every_sec=1e9, push_every_sec=1e9,
    )
    kwargs.update(overrides)
    return train_mod.TrainArgs(**kwargs)


def test_qat_frac_one_puts_the_first_step_on_the_grid(tmp_path):
    """Phase 3 recovers a *finished* model, so there is no tail to wait for:
    `qat_frac=1.0` has to mean "quantized from step 0". Training the opening
    steps in fp32 would spend part of a small budget moving weights the grid
    then has to move back."""
    import train as train_mod
    t = train_mod.Trainer(_recovery_args(tmp_path))
    assert not t._qat_on
    assert t.maybe_enable_qat() is True
    assert t._qat_on and qat.is_qat_active(t.model)
    assert t.step == 0, "QAT engaged only after the first update"
    assert math.isfinite(t.train_step()["loss"])


def test_metrics_say_which_forward_produced_val_bpb(tmp_path):
    """A recovery run's val_bpb is measured *through* the Q4_0 lattice, and
    nothing in the number says so. Without the identifier a reader compares it
    against the pretraining run's fp32 figure and reads the lattice cost as a
    regression."""
    import train as train_mod
    t = train_mod.Trainer(_recovery_args(tmp_path))
    t._val_bpb = lambda: 1.234
    t.maybe_enable_qat()
    t.log_step(t.train_step(), force=True)

    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text()
                     .strip().splitlines()[-1])
    assert row["val_bpb"] == 1.234
    assert row["val_forward"] == "quantized"
    assert row["val_grid"] == qat.grid_id()
    assert row["qat_active"] == 1
    assert row["qat_tensors"] == float(len(qat.plan_qat(t.model)))
    assert row["qat_rel_rmse"] > 0


def test_a_float_run_labels_its_validation_float(tmp_path):
    import train as train_mod
    t = train_mod.Trainer(_recovery_args(tmp_path, qat_frac=0.0))
    t._val_bpb = lambda: 1.234
    t.maybe_enable_qat()
    t.log_step(t.train_step(), force=True)

    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text()
                     .strip().splitlines()[-1])
    assert row["val_forward"] == "float"
    assert row["val_grid"] is None
    assert "qat_rel_rmse" not in row


def test_no_validation_means_no_validation_identifier(tmp_path):
    """An absent val_bpb must not carry a forward label, or a reader would
    take `val_forward: quantized` with `val_bpb: null` for a failed
    measurement on the grid rather than for validation being switched off."""
    import train as train_mod
    t = train_mod.Trainer(_recovery_args(tmp_path))
    t.maybe_enable_qat()
    t.log_step(t.train_step(), force=True)

    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text()
                     .strip().splitlines()[-1])
    assert row["val_bpb"] is None
    assert "val_forward" not in row and "val_grid" not in row


def test_a_checkpoint_written_under_qat_holds_masters_that_reproduce_the_grid(
        tmp_path):
    """The end-to-end claim Phase 3 rests on: what a recovery run *ships* is
    the lattice it *trained* against.

    The checkpoint stores float master weights, not the quantized view -- so
    re-quantizing those masters must land exactly where the fake-quant forward
    was already sitting. If it did not, QAT would be optimizing one lattice
    and `llama-quantize` would snap the export to another.
    """
    import train as train_mod
    t = train_mod.Trainer(_recovery_args(tmp_path, max_steps=2))
    t.maybe_enable_qat()
    t.train_step()
    t.train_step()

    # What the forward pass is currently using, per parametrized module, and
    # the grid each one is on. The kind comes from `plan_qat` rather than from
    # a name test rewritten here: a second copy of that rule could disagree
    # with the real one and the test would still pass.
    kinds = {name: kind for name, _, kind in qat.plan_qat(t.model)}
    quantized_view = {
        name: mod.weight.detach().clone()
        for name, mod in t.model.named_modules()
        if parametrize.is_parametrized(mod, "weight")
    }
    assert quantized_view, "no module ended up parametrized"
    assert set(quantized_view) == set(kinds)

    ckpt = tmp_path / "recovery.pt"
    train_mod.save_checkpoint(str(ckpt), t.model, t.muon, t.adamw, t.step,
                              t.tokens_seen, t.cfg)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert not any("parametrizations" in k for k in payload["model"])

    # Reload into a plain model, exactly as export.py does.
    plain = train_mod.Daedalus(t.cfg)
    plain.load_state_dict(payload["model"])
    reloaded = dict(plain.named_modules())

    for name, on_the_grid in quantized_view.items():
        master = reloaded[name].weight.detach()
        # The master is *not* the quantized view: it kept full precision.
        assert not torch.equal(master, on_the_grid)
        # ... but it re-quantizes onto exactly the same lattice, bit for bit.
        requantized = qat._QDQ[kinds[name]](master)
        assert torch.equal(requantized, on_the_grid), name


def test_qat_frac_reaches_train_args_from_the_cli():
    import train as train_mod
    assert train_mod.parse_args(["--run-name", "x"]).qat_frac == 0.0
    assert train_mod.parse_args(["--run-name", "x", "--qat-frac", "0.05"]).qat_frac == 0.05


def _parametrized_muon_matrix(trainer):
    """A weight that is both fake-quantized and owned by Muon, with its module.

    Muon's momentum buffers are the state a resume can silently lose, and the
    2D hidden matrices are exactly the tensors QAT renames, so this is where
    the two features interact.
    """
    muon_params = {id(p) for g in trainer.muon.param_groups for p in g["params"]}
    for name, mod in trainer.model.named_modules():
        if not parametrize.is_parametrized(mod, "weight"):
            continue
        w = qat.master_weight(mod)
        if w.dim() == 2 and id(w) in muon_params:
            return name, mod, w
    raise AssertionError("no parametrized weight is owned by Muon")


def test_resume_from_a_mid_qat_checkpoint_reenters_qat_with_its_momentum(tmp_path):
    """Restart during the QAT tail: the single most expensive moment to lose.

    `hero.py` relaunches `train.py --resume` after a crash, and the QAT phase
    is the last 5% of a ~96 h run -- so this path first executes ~91 hours in,
    with ~$41 already spent. Three separate things have to survive it, and only
    the first had a test:

    1. The checkpoint loads at all. It is written with the QAT key layout
       stripped (`train.py:130`), so a fresh plain model must accept it
       `strict=True`, which is what `load_checkpoint` does.
    2. QAT re-engages. `_qat_on` is read off the *loaded* model
       (`train.py:752`) and a stripped checkpoint is by definition not
       parametrized, so the flag comes back False. If `maybe_enable_qat` did
       not fire again immediately, the tail would quietly finish training in
       full precision and the shipped Q4_0 model would be the un-QAT'd one --
       no error, no log line, just a worse model.
    3. Muon's momentum buffers come back attached to the *same* tensors QAT
       then wraps. Optimizer state is keyed positionally by `param_groups`, so
       a mismatch between the ordering at save time and at load time would
       hand each matrix somebody else's momentum: not a crash, just corrupted
       updates for the final 2B tokens.
    """
    import train as train_mod

    def mk(**kw):
        return train_mod.TrainArgs(
            config="tiny", max_steps=10, micro_batch=2,
            seq_start=16, seq_end=16, tok_start=32, tok_end=64,
            compile=False, device="cpu", wandb_enabled=False, qat_frac=0.5,
            ckpt_every_sec=1e9, push_every_sec=1e9, hub_repo=None, **kw)

    t = train_mod.Trainer(mk(run_name="qat-resume-a", run_dir=str(tmp_path / "a")))
    for _ in range(8):
        t.maybe_enable_qat()
        t.train_step()
    assert t._qat_on, "QAT never engaged; the rest of this test would be vacuous"

    name, mod, w_before = _parametrized_muon_matrix(t)
    master_before = w_before.detach().clone()
    momentum_before = t.muon.state[w_before]["momentum_buffer"].detach().clone()
    assert momentum_before.abs().sum() > 0, "momentum is zero; nothing to lose"

    ckpt = str(tmp_path / "mid-qat.pt")
    train_mod.save_checkpoint(ckpt, t.model, t.muon, t.adamw,
                              t.step, t.tokens_seen, t.cfg)

    t2 = train_mod.Trainer(mk(run_name="qat-resume-b",
                              run_dir=str(tmp_path / "b"), resume=ckpt))
    assert (t2.step, t2.tokens_seen) == (t.step, t.tokens_seen)
    assert not t2._qat_on, "a stripped checkpoint should restore an unparametrized model"

    # (1) the float master survived, not the quantized view of it
    plain = dict(t2.model.named_parameters())[f"{name}.weight"]
    assert torch.equal(plain, master_before)

    # (2) the very first call re-engages, so no step trains outside the grid
    assert t2.maybe_enable_qat() is True
    assert qat.is_qat_active(t2.model)
    mod2 = dict(t2.model.named_modules())[name]
    w_after = qat.master_weight(mod2)
    assert torch.equal(w_after, master_before)

    # (3) the momentum landed on that same tensor, not on a neighbour
    assert torch.equal(t2.muon.state[w_after]["momentum_buffer"], momentum_before)

    losses = [t2.train_step()["loss"] for _ in range(2)]
    assert all(math.isfinite(x) for x in losses), losses


# ------------------------------------------------- QAT under compile ---
#
# Everything above runs eager on CPU. `hero` does not: `train.py` wraps the
# model in `torch.compile` at construction (train.py:657) and then registers
# the parametrizations ~90 hours later, into a graph that was traced when
# `mod.weight` was still a plain Parameter. If Dynamo's guards do not notice
# that `weight` has become a property backed by a new submodule, the compiled
# forward keeps using the *unquantized* weight: the log still prints "QAT ON",
# no exception is raised, and the last 5% of a four-day run silently buys
# nothing. The failure is invisible in the loss curve, so it has to be a test.

def _free_vram_gb() -> float:
    """Free VRAM, or 0.0 if CUDA is unusable for any reason.

    Wrapped because `mem_get_info` has to create a CUDA context to answer, and
    creating one on a GPU that a training job has already filled raises rather
    than returning 0 -- which would turn a collection-time probe into a
    collection-time error.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return 0.0


# Skip on a *busy* GPU, not just an absent one.
#
# `skipif(not torch.cuda.is_available())` is the obvious gate and it is wrong
# here. While `sweep`/`abl-arch`/`hero` are running, CUDA is perfectly
# available and the GPU is ~31.9 of 32.6 GB full, so these tests do not skip --
# they run and die with `CUDA error: out of memory`. That turns "run the full
# suite before every push" (AGENT.md) into a guaranteed 7 failures for the ~28 h
# the chain owns the box, which trains the reader to ignore red tests at exactly
# the moment a real regression would matter.
#
# 4 GB is comfortably above what these probes need (a small model plus inductor
# workspace) and comfortably below what an idle box offers.
_MIN_FREE_VRAM_GB = 4.0


def _training_is_live() -> bool:
    """Is a `train.py` running right now (not counting this pytest process)?

    The VRAM check above protects *the tests* from a busy GPU. It does not
    protect *the job* from the tests, and that is the more expensive direction:
    "run the full suite before every push" means these probes fire every ~10
    minutes for the four days `hero` owns the box. A free-VRAM reading is a
    snapshot of the trainer's current reservation, and its peak comes later --
    a sequence-length ramp step allocates more activation memory than the
    reading that let these tests in. Losing `hero` to an OOM caused by its own
    test suite costs up to 30 minutes of a ~$43.70 run; skipping a GPU test
    costs nothing, because the box is idle often enough to run them.

    /proc rather than psutil: this decides collection, so it must not depend on
    an import that might not be there.
    """
    me = str(os.getpid())
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit() and p != me]
    except OSError:
        return False
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue          # exited between listdir and open, or not ours
        if any(a.endswith(b"train.py") for a in argv):
            return True
    return False


_TRAINING_LIVE = _training_is_live()
_CUDA = pytest.mark.skipif(
    _free_vram_gb() < _MIN_FREE_VRAM_GB or _TRAINING_LIVE,
    reason=("a train.py process owns the GPU" if _TRAINING_LIVE else
            f"needs CUDA with >={_MIN_FREE_VRAM_GB} GB free "
            f"(have {_free_vram_gb():.1f} GB; a training job likely owns the GPU)"))


def _qat_probe_model(device, dtype=torch.float32):
    torch.manual_seed(0)
    cfg = DaedalusConfig(
        hidden_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, block_ff_dim=256,
        vocab_size=512, num_attention_blocks=2, max_position_embeddings=256,
    )
    return Daedalus(cfg).to(device=device, dtype=dtype).eval(), cfg


def _logits(m, ids):
    with torch.no_grad():
        logits, _, _ = m(ids)
    return logits.float()


@_CUDA
def test_enabling_qat_mid_run_reaches_an_already_compiled_graph():
    """The one that matters for `hero`: enable QAT *after* compiling.

    Asserts the compiled forward tracks eager both before and after the
    switch. The `after` half is the real check -- if Dynamo served a stale
    graph it would still equal the *pre-QAT* output, which is precisely the
    silent no-op this test exists to catch.
    """
    torch._dynamo.reset()
    model, _ = _qat_probe_model("cuda")
    net = torch.compile(model)
    ids = torch.randint(0, 512, (2, 64), device="cuda")

    compiled_before = _logits(net, ids)
    eager_before = _logits(model, ids)
    assert torch.allclose(compiled_before, eager_before, atol=1e-4), \
        "compiled and eager disagree before QAT; the probe itself is broken"

    applied = qat.enable_qat(model)
    assert applied, "enable_qat registered nothing"

    compiled_after = _logits(net, ids)
    eager_after = _logits(model, ids)

    # Q4_0 on every Linear is a large perturbation; if this is ~0 the
    # parametrization never took effect even in eager.
    assert not torch.allclose(eager_after, eager_before, atol=1e-3), \
        "eager forward did not change under QAT"
    assert not torch.allclose(compiled_after, compiled_before, atol=1e-3), (
        "the compiled graph returned its pre-QAT output: Dynamo did not "
        "retrace after register_parametrization, so QAT is a silent no-op "
        "for the whole tail of the run"
    )
    assert torch.allclose(compiled_after, eager_after, atol=1e-4), \
        "compiled forward disagrees with eager once QAT is on"


@_CUDA
def test_compiled_qat_lands_on_the_same_lattice_as_eager():
    """The bit-exactness certified against real `libggml` is measured in eager.
    `hero` runs the quantizer *inside* an inductor kernel, where a plainly
    written `d.half().float()` gets folded away -- leaving the fp32 scale and a
    lattice `llama-quantize` never produces. Nothing raises when that happens;
    the loss curve looks identical. So compare the grids directly."""
    torch.manual_seed(0)
    for fn in (qat.q4_0_qdq, qat.q8_0_qdq):
        torch._dynamo.reset()
        w = (torch.randn(128, 256, device="cuda") * 0.02)
        eager = fn(w)
        compiled = torch.compile(fn)(w)
        n_diff = int((eager != compiled).sum())
        assert n_diff == 0, (
            f"{fn.__name__}: {n_diff}/{w.numel()} elements land on a different "
            f"grid under torch.compile (max |delta| "
            f"{(eager - compiled).abs().max():.3e}); QAT would be training "
            f"against a lattice llama.cpp does not use"
        )


@_CUDA
def test_compiled_qat_scale_is_the_fp16_one_llama_cpp_stores():
    """Pin the specific failure mode, not just the symptom: recover the scale
    the compiled kernel actually used and check it is the fp16-rounded `d`."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    w = (torch.randn(8, 32, device="cuda") * 0.02)
    compiled = torch.compile(qat.q4_0_qdq)(w)
    for r in range(w.shape[0]):
        blk = w[r].float()
        d_fp32 = blk[blk.abs().argmax()] / -8.0
        d_fp16 = d_fp32.half().float()
        q = torch.clamp(torch.floor(blk * (1.0 / d_fp32) + 8.5), 0, 15)
        k = int((q - 8).abs().argmax())
        steps = (q[k] - 8).item()
        if abs(steps) < 1:
            continue
        used = (compiled[r, k] / steps).item()
        assert used == pytest.approx(d_fp16.item(), rel=1e-9), (
            f"row {r}: compiled kernel used scale {used:.10f}, but llama.cpp "
            f"stores fp16 {d_fp16.item():.10f} (fp32 would be "
            f"{d_fp32.item():.10f})"
        )


@_CUDA
def test_compiled_qat_trains_the_master_weight_through_the_ste():
    """Gradients must land on the float master, not on the quantized view.

    Under `parametrize` the leaf is `parametrizations.weight.original`; if the
    STE were broken the master would receive no gradient and the QAT phase
    would be ~2 hours of the model not moving.
    """
    torch._dynamo.reset()
    model, _ = _qat_probe_model("cuda")
    net = torch.compile(model)
    ids = torch.randint(0, 512, (2, 64), device="cuda")

    qat.enable_qat(model)
    model.train()
    logits, _, _ = net(ids)
    logits.float().pow(2).mean().backward()

    linears = [m for m in model.modules()
               if isinstance(m, nn.Linear) and parametrize.is_parametrized(m, "weight")]
    assert linears, "no parametrized Linear to check"
    for mod in linears:
        master = qat.master_weight(mod)
        assert master.grad is not None, "master weight got no gradient under compile"
        assert torch.isfinite(master.grad).all()
    assert any(qat.master_weight(m).grad.abs().sum() > 0 for m in linears)


@_CUDA
def test_qat_phase_pulls_weights_toward_the_grid_under_compile():
    """The evidence QAT is working is `qat_rel_rmse` falling. Prove it does on
    the compiled path, since that is the only path `hero` runs and the metric
    is read off the master weight -- i.e. it would keep reporting a plausible
    number even if the compiled forward had ignored QAT entirely."""
    torch._dynamo.reset()
    model, _ = _qat_probe_model("cuda")
    model.train()
    net = torch.compile(model)
    ids = torch.randint(0, 512, (2, 64), device="cuda")

    qat.enable_qat(model)
    before = qat.quantization_error(model)["qat_rel_rmse"]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        logits, _, _ = net(ids)
        logits.float().mean().backward()
        opt.step()
    after = qat.quantization_error(model)["qat_rel_rmse"]
    assert math.isfinite(after)
    assert after < before, f"quantization error did not fall: {before} -> {after}"


@_CUDA
def test_checkpoint_written_mid_qat_under_compile_loads_into_a_plain_model():
    """Resume during the QAT tail is the most expensive moment to lose. The
    checkpoint must carry the model's normal key layout even though the live
    module is parametrized and wrapped in `torch.compile`."""
    torch._dynamo.reset()
    model, cfg = _qat_probe_model("cuda")
    net = torch.compile(model)
    ids = torch.randint(0, 512, (2, 64), device="cuda")
    qat.enable_qat(model)
    _logits(net, ids)

    sd = qat.strip_qat_state_dict(model.state_dict())
    assert not any("parametrizations" in k for k in sd)
    fresh = Daedalus(cfg).to("cuda")
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing


@_CUDA
def test_trainer_qat_phase_on_the_real_compiled_cuda_path(tmp_path):
    """The closest reachable stand-in for hero's last 5%: the real `Trainer`,
    on CUDA, with `torch.compile` and bf16 autocast live, crossing into the QAT
    phase mid-loop and then checkpointing.

    Every other Trainer-level QAT test runs `compile=False, device="cpu"`
    (above), which is the one configuration `hero` never uses.
    """
    import train as train_mod

    torch._dynamo.reset()
    args = train_mod.TrainArgs(
        run_name="qat-cuda-smoke", config="tiny", max_steps=8,
        micro_batch=2, seq_start=64, seq_end=64, tok_start=128, tok_end=128,
        compile=True, device="cuda", wandb_enabled=False, qat_frac=0.5,
        run_dir=str(tmp_path / "run"), ckpt_every_sec=1e9, push_every_sec=1e9,
    )
    t = train_mod.Trainer(args)

    losses, switched_at = [], None
    for _ in range(8):
        if t.maybe_enable_qat() and switched_at is None:
            switched_at = t.step
        losses.append(t.train_step()["loss"])

    assert t._qat_on, "QAT never engaged on the compiled CUDA path"
    assert switched_at is not None
    assert all(math.isfinite(x) for x in losses), losses
    assert qat.is_qat_active(t.model)

    # The forward must actually be on the grid: the compiled graph has to have
    # retraced, and the reported error must be the master weight's, not ~0.
    err = qat.quantization_error(t.model)["qat_rel_rmse"]
    assert 0.0 < err < 1.0, err

    # Muon/AdamW must still own the same tensors they were built around --
    # otherwise the switch silently orphans the optimizer state at 95% of a
    # four-day run.
    live = {id(p) for p in t.model.parameters()}
    for opt in (t.muon, t.adamw):
        for g in opt.param_groups:
            for p in g["params"]:
                assert id(p) in live, "optimizer holds a parameter the model dropped"

    ckpt = tmp_path / "ck.pt"
    train_mod.save_checkpoint(str(ckpt), t.model, t.muon, t.adamw,
                              t.step, t.tokens_seen, t.cfg)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert not any("parametrizations" in k for k in payload["model"])


# ------------------------------------------------- QAT's memory cost at hero ---
# QAT is on for `hero` (--qat-frac 0.05) and OFF for both `abl-arch` arms, so
# the only measured peak this project has -- arm 1's 24.29 GB at hero's exact
# config and micro-batch -- does not cover the final 5%. An OOM there arrives
# ~5.5 days and ~$57 into the run.
#
# The cost is not in parameters: `torch.nn.utils.parametrize` *renames*
# `weight` to `parametrizations.weight.original` rather than duplicating it, so
# the parameter and buffer footprint is unchanged and the optimizer sees the
# same tensors. What QAT adds is autograd: each forward recomputes
# `weight = FakeQuant(original)`, and the dequantised weight plus the
# quantise/dequantise intermediates are held for backward. Those are weights,
# so the cost is *batch-independent* -- which is what makes it measurable at
# batch 1 on CPU and valid at micro-batch 16 on the GPU.

def _saved_tensor_bytes(model, cfg, b=1, t=128):
    total, seen = {"n": 0}, set()

    def pack(x):
        if x.data_ptr() not in seen:
            seen.add(x.data_ptr())
            total["n"] += x.numel() * x.element_size()
        return x

    ids = torch.randint(0, cfg.vocab_size, (b, t))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        model(ids, targets=ids)
    return total["n"]


def test_qat_adds_no_parameters_or_buffers():
    """If this ever regresses to a real copy, both optimizer states grow with
    it and the headroom arithmetic below stops holding."""
    cfg = PRESETS["daedalus-150m"]
    m = Daedalus(cfg)

    def footprint(mod):
        return (sum(t.numel() * t.element_size() for t in mod.parameters())
                + sum(t.numel() * t.element_size() for t in mod.buffers()))

    before = footprint(m)
    qat_mod.enable_qat(m)
    assert footprint(m) == before


def test_qat_activation_cost_leaves_hero_headroom():
    """Measured 1.65 GB. Arm 1 peaked at 24.29 GB allocated
    (`torch.cuda.max_memory_allocated()/1e9`, train.py:1047) at hero's config
    and micro-batch 16, so hero's QAT tail projects to ~25.9 GB allocated
    against 32.6 GB of card -- comfortable.

    The bound is deliberately loose: this guards against a change that makes
    fake-quant hold several more full-size copies per module, not against
    normal variation.
    """
    cfg = PRESETS["daedalus-150m"]
    torch.manual_seed(0)
    m = Daedalus(cfg).train()
    off = _saved_tensor_bytes(m, cfg)
    qat_mod.enable_qat(m)
    on = _saved_tensor_bytes(m, cfg)
    delta_gb = (on - off) / 1e9
    assert delta_gb > 0.1, "fake-quant appears not to be in the graph at all"
    assert delta_gb < 4.0, (
        f"QAT now adds {delta_gb:.2f} GB of autograd state; arm 1's measured "
        f"24.29 GB peak leaves ~7.5 GB, so this would put hero's final 5% at "
        f"risk of OOM ~5.5 days into the run")


# ------------------------------------- does QAT survive export? (fp16) ---
#
# The grid tests above prove `q4_0_qdq` IS `llama-quantize`'s grid. That is
# necessary but not sufficient: between the QAT master weight and
# `llama-quantize` sits `export_hf_model`, which writes the HF tensors in a
# reduced dtype. Nothing tested that composition, and it was wrong -- the
# default was bf16, which keeps 8 mantissa bits against fp16's 11, so the
# block scale `d` moved and the shipped weights were no longer the weights QAT
# converged to. Measured end to end in `runs/preflight/qat-survives-export.md`.

def _grid_survives(dtype):
    """Round-trip on-grid weights through `dtype` storage, then through the
    real ggml quantizer, and report the relative RMS drift."""
    lib = ctypes.CDLL(_find_libggml())
    torch.manual_seed(11)
    w = (torch.randn(8, 512) * 0.02).float().contiguous()
    ref = qat.q4_0_qdq(w)                                 # what QAT converges to
    stored = ref.to(dtype).to(torch.float16).float().contiguous()   # HF dtype -> GGUF f16
    ship = _dequant_q4_0(_ggml_quantize(lib, GGML_TYPE_Q4_0, stored, 18), 8, 512)
    return ref, ship, float((ship - ref).norm() / ref.norm())


@pytest.mark.skipif(_find_libggml() is None, reason="llama.cpp not built here")
def test_fp16_export_keeps_qat_weights_exactly_on_the_grid():
    """The property `export_hf_model`'s fp16 default exists to guarantee:
    weights QAT put on the grid are the weights that ship, bit for bit."""
    ref, ship, drift = _grid_survives(torch.float16)
    assert torch.equal(ship, ref), f"fp16 export moved the weights: rel {drift:.2e}"


@pytest.mark.skipif(_find_libggml() is None, reason="llama.cpp not built here")
def test_bf16_export_would_move_the_weights_off_the_qat_grid():
    """Pins *why* the default is fp16, so it cannot be "tidied" back to bf16.

    bf16 is the natural choice (it is the training dtype) and is what this
    project shipped until it was measured. It is wrong here for one reason:
    fewer mantissa bits move the fp16 block scale llama.cpp stores.
    """
    ref, ship, drift = _grid_survives(torch.bfloat16)
    assert not torch.equal(ship, ref)
    assert drift > 1e-4, (
        "bf16 no longer perturbs the grid -- if that is genuinely true the "
        "fp16 default can be revisited, but check the real export path first")


def test_export_defaults_to_fp16_because_bf16_erodes_qat():
    """`hero` spends its final 5% on QAT; the export dtype decides whether any
    of that reaches the shipped Q4_0 file."""
    import inspect

    import export as export_module

    default = inspect.signature(export_module.export_hf_model).parameters["dtype"].default
    assert default is torch.float16, (
        f"export_hf_model writes {default}; bf16 costs 0.17% of the QAT grid "
        f"(measured, runs/preflight/qat-survives-export.md) and fp32 doubles "
        f"the published artifact for no gain")
