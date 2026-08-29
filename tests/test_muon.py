"""Tests for daedalus/muon.py -- the optimizer split, the Newton-Schulz
orthogonalization, and the WSD / momentum schedules.

These did not exist before the issue #4 audit. muon.py is the single largest
training lever in the project (arXiv 2509.02046 puts matrix-preconditioned
methods at ~1.4x over tuned AdamW, peaking at ~130M params -- essentially this
model's size), and a silent regression here would show up only as "the model
trains slightly worse", which is exactly the failure mode nothing else in the
suite can catch.

Run: python -m pytest tests/test_muon.py -v
"""
import math

import pytest
import torch

from daedalus.config import PRESETS
from daedalus.model import Daedalus
from daedalus.muon import (
    Muon,
    build_optimizers,
    conv_proj_wd_schedule,
    momentum_warmup,
    wsd_lr,
    zeropower_via_newtonschulz5,
)


# ------------------------------------------------------------ newton-schulz ---

def test_newtonschulz_output_is_approximately_orthogonal():
    """The point of the iteration: singular values pulled toward 1, so the
    update carries direction only and not the gradient's scale."""
    torch.manual_seed(0)
    g = torch.randn(64, 32)
    x = zeropower_via_newtonschulz5(g, steps=5)
    s = torch.linalg.svdvals(x.float())
    assert s.min() > 0.6
    assert s.max() < 1.4


def test_newtonschulz_is_scale_invariant():
    """It normalizes by the input norm first, which is why Muon's momentum
    buffer being a running sum rather than an EMA is harmless -- the constant
    1/(1-momentum) factor between the two formulations cancels here."""
    torch.manual_seed(0)
    g = torch.randn(32, 16)
    base = zeropower_via_newtonschulz5(g, steps=5).float().flatten()
    # Direction, not elementwise equality: the iteration runs in bfloat16
    # (~3 significant digits), so 5 rounds of matmuls leave elementwise noise
    # of ~0.025 on unit-scale entries. What has to be invariant is the update
    # *direction*, which is all the optimizer consumes.
    for factor in (1e-4, 1e3):
        other = zeropower_via_newtonschulz5(g * factor, steps=5).float().flatten()
        cos = torch.nn.functional.cosine_similarity(base, other, dim=0).item()
        assert cos > 0.999, f"scale {factor} changed the update direction ({cos})"


def test_newtonschulz_handles_wide_and_tall_identically():
    """Wide matrices are transposed internally; the result must be the
    transpose of the tall case, not something shape-dependent."""
    torch.manual_seed(0)
    g = torch.randn(16, 48)
    wide = zeropower_via_newtonschulz5(g, steps=5).float()
    tall = zeropower_via_newtonschulz5(g.T.contiguous(), steps=5).float()
    assert torch.allclose(wide, tall.T, atol=2e-2)


def test_newtonschulz_preserves_shape_and_dtype():
    g = torch.randn(8, 4)
    out = zeropower_via_newtonschulz5(g, steps=5)
    assert out.shape == g.shape
    assert out.dtype == g.dtype


def test_newtonschulz_rejects_non_2d():
    with pytest.raises(AssertionError):
        zeropower_via_newtonschulz5(torch.randn(4, 4, 4), steps=5)


# -------------------------------------------------------------------- muon ---

def test_muon_step_moves_against_the_gradient():
    p = torch.nn.Parameter(torch.eye(8) * 0.0 + torch.randn(8, 8) * 0.01)
    before = p.detach().clone()
    opt = Muon([p], lr=0.02, weight_decay=0.0)
    p.grad = torch.randn(8, 8)
    opt.step()
    delta = (p.detach() - before).flatten()
    assert torch.dot(delta, p.grad.flatten()) < 0     # downhill


def test_muon_applies_decoupled_weight_decay_without_a_gradient_step():
    """wd multiplies the weight by (1 - lr*wd) independently of the gradient --
    the decoupled form. With a zero grad the update direction is zero, so the
    shrink is all that is left and its size is exactly predictable."""
    p = torch.nn.Parameter(torch.ones(4, 4))
    opt = Muon([p], lr=0.1, weight_decay=0.5, momentum=0.0, nesterov=False)
    p.grad = torch.zeros(4, 4)
    opt.step()
    assert torch.allclose(p.detach(), torch.full((4, 4), 0.95), atol=1e-6)


def test_muon_skips_params_without_grads():
    a = torch.nn.Parameter(torch.randn(4, 4))
    b = torch.nn.Parameter(torch.randn(4, 4))
    b_before = b.detach().clone()
    opt = Muon([a, b], lr=0.02, weight_decay=0.0)
    a.grad = torch.randn(4, 4)
    opt.step()                                        # b.grad is None
    assert torch.equal(b.detach(), b_before)


def test_muon_rejects_non_2d_params():
    p = torch.nn.Parameter(torch.randn(8))
    opt = Muon([p], lr=0.02)
    p.grad = torch.randn(8)
    with pytest.raises(AssertionError):
        opt.step()


def test_muon_lr_adjustment_scales_with_aspect_ratio():
    """adjust_lr multiplies by sqrt(max(1, fan_out/fan_in)) so one lr transfers
    across layer shapes. A square matrix must be unaffected (factor 1)."""
    torch.manual_seed(0)
    g = torch.randn(16, 4)

    def step_once(adjust):
        torch.manual_seed(1)
        p = torch.nn.Parameter(torch.zeros(16, 4))
        opt = Muon([p], lr=0.1, weight_decay=0.0, momentum=0.0, nesterov=False,
                   adjust_lr=adjust)
        p.grad = g.clone()
        opt.step()
        return p.detach().norm().item()

    assert step_once(True) == pytest.approx(step_once(False) * math.sqrt(4.0), rel=1e-4)


def test_muon_momentum_buffer_persists_across_steps():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Muon([p], lr=0.02, weight_decay=0.0)
    p.grad = torch.randn(4, 4)
    opt.step()
    assert "momentum_buffer" in opt.state[p]
    assert opt.state[p]["momentum_buffer"].shape == (4, 4)


# ------------------------------------------------------------ param split ---

def test_build_optimizers_routes_every_parameter_exactly_once():
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, adamw, stats = build_optimizers(model)
    routed = sum(p.numel() for g in muon.param_groups for p in g["params"])
    routed += sum(p.numel() for g in adamw.param_groups for p in g["params"])
    assert routed == sum(p.numel() for p in model.parameters())
    assert stats["muon_params"] + stats["adam_params"] == routed


def test_build_optimizers_keeps_embeddings_out_of_muon():
    """Orthogonalizing a vocab-sized matrix is both expensive and wrong -- its
    rows are lookups, not a linear map. This is the universal Muon split rule."""
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, adamw, _ = build_optimizers(model)
    emb = model.embed_tokens.weight
    assert not any(p is emb for g in muon.param_groups for p in g["params"])
    assert any(p is emb for g in adamw.param_groups for p in g["params"])


def test_build_optimizers_sends_only_2d_tensors_to_muon():
    """Muon asserts 2D at step time; anything else routed there would blow up
    mid-run rather than at construction."""
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, _, _ = build_optimizers(model)
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])


def test_build_optimizers_disables_weight_decay_on_norms():
    """Decaying norm gains destabilizes training (SmolLM3 finding). Every 1D
    parameter -- all the RMSNorm weights -- must land in the wd=0 group."""
    model = Daedalus(PRESETS["daedalus-150m"])
    _, adamw, _ = build_optimizers(model)
    nodecay = [g for g in adamw.param_groups if g["weight_decay"] == 0.0]
    assert len(nodecay) == 1
    assert all(p.ndim == 1 for p in nodecay[0]["params"])
    n_1d = sum(1 for p in model.parameters() if p.ndim == 1)
    assert len(nodecay[0]["params"]) == n_1d


def test_build_optimizers_honours_the_requested_learning_rates():
    model = Daedalus(PRESETS["tiny"])
    muon, adamw, _ = build_optimizers(model, muon_lr=0.04, adam_lr=1e-3)
    assert all(g["lr"] == 0.04 for g in muon.param_groups)
    assert all(g["lr"] == 1e-3 for g in adamw.param_groups)


def test_build_optimizers_skips_frozen_params():
    model = Daedalus(PRESETS["tiny"])
    model.embed_tokens.weight.requires_grad_(False)
    _, _, stats = build_optimizers(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert stats["muon_params"] + stats["adam_params"] == trainable


def test_build_optimizers_untied_head_goes_to_adamw():
    import dataclasses
    cfg = dataclasses.replace(PRESETS["tiny"], tie_word_embeddings=False,
                              layer_types=None)
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model)
    head = model.lm_head.weight
    assert not any(p is head for g in muon.param_groups for p in g["params"])
    assert any(p is head for g in adamw.param_groups for p in g["params"])


# ------------------------------------------- the conv-death fix (issue #7) ---
#
# 47.96% of `hero`'s ShortConv channels contribute nothing, and the mechanism
# experiment isolated Muon's weight decay on the conv projections as the cause.
# `conv_proj_wd` is the fix. It is off by default because it changes how a
# multi-day run trains, so the first thing to protect is that *not* passing it
# leaves everything exactly as it was.

def test_conv_proj_wd_defaults_to_the_shipped_single_group_behaviour():
    """The default must be indistinguishable from before the fix existed.

    `hero` is mid-run against this file. If the default grew a second param
    group, every crash-resume would fail to load its optimizer state -- so this
    asserts the group *layout*, not merely the numbers.
    """
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, _, stats = build_optimizers(model)

    assert len(muon.param_groups) == 1, (
        f"default produced {len(muon.param_groups)} Muon groups; the shipped "
        "layout is one, and a resume matches state_dict by group")
    assert all(g["weight_decay"] == 0.1 for g in muon.param_groups)
    assert stats["conv_proj_wd"] is None
    assert stats["conv_proj_params"] == 0


def test_conv_proj_wd_moves_exactly_the_conv_projections_and_nothing_else():
    """The fix has to be one variable. Anything else swept into the no-decay
    group would confound the comparison against the run it replaces."""
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, _, stats = build_optimizers(model, conv_proj_wd=0.0)

    assert len(muon.param_groups) == 2
    decayed, undecayed = muon.param_groups[0], muon.param_groups[1]
    assert undecayed["weight_decay"] == 0.0
    assert decayed["weight_decay"] == 0.1

    expected = {id(p) for n, p in model.named_parameters()
                if p.ndim == 2 and ".conv." in n}
    got = {id(p) for p in undecayed["params"]}
    assert got == expected, (
        f"no-decay group holds {len(got)} tensors, expected {len(expected)} "
        "conv projection matrices")

    # 12 conv blocks x {in_proj, out_proj}
    assert stats["conv_proj_tensors"] == 24, stats["conv_proj_tensors"]
    assert stats["conv_proj_params"] == 28_311_552, stats["conv_proj_params"]


def test_conv_proj_wd_still_routes_every_parameter_exactly_once():
    """A split is the easiest place to drop or double-count a tensor."""
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, adamw, stats = build_optimizers(model, conv_proj_wd=0.0)

    ids = [id(p) for g in muon.param_groups for p in g["params"]]
    ids += [id(p) for g in adamw.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter was routed twice"
    routed = sum(p.numel() for g in muon.param_groups for p in g["params"])
    routed += sum(p.numel() for g in adamw.param_groups for p in g["params"])
    assert routed == sum(p.numel() for p in model.parameters())
    assert stats["muon_params"] + stats["adam_params"] == routed


def test_conv_proj_wd_leaves_the_depthwise_kernel_in_adamw():
    """`conv.conv.weight` is 3D, so it is not a projection and Muon cannot take
    it -- Muon asserts 2D at step time. A name-only match would route it there
    and blow up mid-run."""
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, adamw, _ = build_optimizers(model, conv_proj_wd=0.0)

    kernels = [p for n, p in model.named_parameters()
               if n.endswith("conv.conv.weight")]
    assert kernels, "preset has no depthwise kernels; test is vacuous"
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in adamw.param_groups for p in g["params"]}
    for k in kernels:
        assert id(k) not in muon_ids, "3D kernel routed to Muon"
        assert id(k) in adam_ids
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])


def test_conv_proj_wd_actually_stops_the_decay_it_targets():
    """The behavioural test, not the plumbing one.

    Grads are set to *zero* rather than left as None: Muon skips a param with no
    grad entirely, so a None-grad version of this test would pass even if the
    fix did nothing. With a zero grad the update direction is zero and decay is
    the only remaining effect -- which is precisely the term under test.
    """
    model = Daedalus(PRESETS["daedalus-150m"])
    muon, _, _ = build_optimizers(model, conv_proj_wd=0.0, muon_lr=0.02)

    conv_w = model.layers[0].conv.out_proj.weight
    ffn_w = model.layers[0].feed_forward.w1.weight
    with torch.no_grad():
        conv_w.fill_(0.1)
        ffn_w.fill_(0.1)
    conv_w.grad = torch.zeros_like(conv_w)
    ffn_w.grad = torch.zeros_like(ffn_w)
    before_conv, before_ffn = conv_w.detach().clone(), ffn_w.detach().clone()
    muon.step()

    # Bit-identical, not approximately equal: with no gradient and no decay
    # there is nothing left that could legitimately move this tensor.
    assert torch.equal(conv_w.detach(), before_conv), (
        "conv projection moved despite conv_proj_wd=0.0 and no gradient")
    assert float(ffn_w.detach().abs().mean()) < float(before_ffn.abs().mean()), (
        "FFN matrix was not decayed; the control arm of this test is broken")


def test_conv_proj_wd_state_dict_is_incompatible_with_the_default_and_says_so():
    """The fix cannot be half-applied to a run in progress.

    A restart is the *only* way to adopt it, and that is the whole reason the
    decision is time-sensitive. If this ever loaded silently, the conv
    projections would resume under the wrong decay with no error at all.
    """
    model = Daedalus(PRESETS["daedalus-150m"])
    plain, _, _ = build_optimizers(model)
    fixed, _, _ = build_optimizers(model, conv_proj_wd=0.0)

    with pytest.raises(ValueError, match="parameter group"):
        fixed.load_state_dict(plain.state_dict())


def test_conv_proj_wd_refuses_to_be_a_silent_no_op():
    """A model with no conv blocks must raise rather than report the fix
    applied while changing nothing -- `dense-150m` is exactly such a model."""
    dense = Daedalus(PRESETS["dense-150m"])
    assert not any(p.ndim == 2 and ".conv." in n
                   for n, p in dense.named_parameters()), \
        "dense preset unexpectedly has conv projections; test is vacuous"

    with pytest.raises(ValueError, match="silently do nothing"):
        build_optimizers(dense, conv_proj_wd=0.0)


def test_train_py_exposes_conv_proj_wd_and_defaults_it_off():
    """The flag has to reach the trainer, and its default must not change how
    `hero` trains on a crash-resume."""
    import train as train_mod

    args = train_mod.parse_args(["--run-name", "x"])
    assert args.conv_proj_wd is None, (
        f"--conv-proj-wd defaults to {args.conv_proj_wd!r}, not None")
    args = train_mod.parse_args(["--run-name", "x", "--conv-proj-wd", "0.0"])
    assert args.conv_proj_wd == 0.0


# ------------------------------------------- scheduled conv-projection decay --
# Phase 5 compares four decay schedules for these projections, two of them
# varying. They have to vary because the two constants fail in opposite
# directions (`runs/preflight/conv-death-fix-validated.md`): decay 0 stops the
# death but has no equilibrium, and a decay weak enough to lose the early race
# still wins it later, postponing the death rather than preventing it.

def test_a_constant_schedule_is_what_the_two_constant_arms_get():
    """`end=None` must be the shipped behaviour exactly, at every step, so the
    two constant arms need no code path of their own."""
    for step in (0, 1, 500, 4_999, 5_000, 10_000):
        assert conv_proj_wd_schedule(step, 10_000, 0.1) == 0.1
        assert conv_proj_wd_schedule(step, 10_000, 0.0133) == 0.0133
        # An explicit end equal to start is the same thing said twice.
        assert conv_proj_wd_schedule(step, 10_000, 0.1, end=0.1,
                                     ramp_frac=0.3) == 0.1


def test_the_zero_to_shipped_arm_ramps_over_the_first_tenth_then_holds():
    """Arm 3: nothing while the early race is being decided, the shipped 0.1
    once it is over."""
    total = 10_000
    schedule = lambda step: conv_proj_wd_schedule(  # noqa: E731
        step, total, 0.0, end=0.1, ramp_frac=0.1)

    assert schedule(0) == 0.0
    assert schedule(500) == pytest.approx(0.05)
    assert schedule(1_000) == pytest.approx(0.1)
    assert schedule(9_999) == pytest.approx(0.1)


def test_the_weak_then_shipped_arm_reaches_the_shipped_value_at_thirty_percent():
    """Arm 4: 0.0133 early, ramping to 0.1 by 30% of the run."""
    total = 10_000
    schedule = lambda step: conv_proj_wd_schedule(  # noqa: E731
        step, total, 0.0133, end=0.1, ramp_frac=0.3)

    assert schedule(0) == pytest.approx(0.0133)
    assert schedule(1_500) == pytest.approx(0.0133 + (0.1 - 0.0133) * 0.5)
    assert schedule(3_000) == pytest.approx(0.1)
    assert schedule(10_000) == pytest.approx(0.1)


def test_a_hold_delays_the_ramp_without_moving_where_it_lands():
    """The `hold` shape exists so "weak early, then ramp" can be expressed with
    an explicit early window rather than by reading one into the ramp."""
    total = 1_000
    held = lambda step: conv_proj_wd_schedule(  # noqa: E731
        step, total, 0.0, end=0.1, ramp_frac=0.3, hold_frac=0.1)

    assert held(100) == 0.0                       # still holding at the boundary
    assert held(200) == pytest.approx(0.05)       # halfway through 100..300
    assert held(300) == pytest.approx(0.1)


def test_the_schedule_is_monotone_and_bounded_between_its_endpoints():
    """A ramp that overshoots would put a decay on these projections that no
    arm preregistered, and the arm would still look like the one that was."""
    total = 997                                   # not a round number on purpose
    values = [conv_proj_wd_schedule(s, total, 0.0133, end=0.1, ramp_frac=0.3)
              for s in range(total + 1)]

    assert values == sorted(values)
    assert min(values) == pytest.approx(0.0133)
    assert max(values) == pytest.approx(0.1)


def test_a_ramp_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError, match="hold_frac"):
        conv_proj_wd_schedule(0, 1_000, 0.0, end=0.1, ramp_frac=0.1,
                              hold_frac=0.3)


def test_build_optimizers_reports_which_group_a_schedule_must_write_to():
    """The only other way to find it is to hard-code index 1, and a schedule
    that retuned the other 76.9% of Muon's parameters would show up as nothing
    but a puzzling arm result."""
    model = Daedalus(PRESETS["daedalus-150m"])

    _, _, default_stats = build_optimizers(model)
    assert default_stats["conv_proj_group_index"] is None

    muon, _, stats = build_optimizers(model, conv_proj_wd=0.0133)
    group = muon.param_groups[stats["conv_proj_group_index"]]
    expected = {id(p) for n, p in model.named_parameters()
                if p.ndim == 2 and ".conv." in n}
    assert {id(p) for p in group["params"]} == expected
    assert group["weight_decay"] == 0.0133


def test_train_py_exposes_the_ramp_and_leaves_it_off_by_default():
    import train as train_mod

    args = train_mod.parse_args(["--run-name", "x"])
    assert args.conv_proj_wd_end is None
    assert args.conv_proj_wd_ramp_frac == 0.0
    assert args.conv_proj_wd_hold_frac == 0.0

    args = train_mod.parse_args([
        "--run-name", "x", "--conv-proj-wd", "0.0",
        "--conv-proj-wd-end", "0.1", "--conv-proj-wd-ramp-frac", "0.1"])
    assert (args.conv_proj_wd, args.conv_proj_wd_end) == (0.0, 0.1)
    assert args.conv_proj_wd_ramp_frac == 0.1


def test_a_ramp_without_a_group_to_ramp_is_refused():
    """Same silent no-op `build_optimizers` refuses for the flag itself: the
    run looks configured, trains the shipped schedule, and only the arm's
    result says otherwise -- after the GPU hours are spent."""
    import train as train_mod

    with pytest.raises(ValueError, match="need conv_proj_wd set"):
        train_mod.parse_args(["--run-name", "x", "--conv-proj-wd-end", "0.1",
                              "--conv-proj-wd-ramp-frac", "0.1"])


def test_an_end_value_with_no_ramp_is_refused_rather_than_applied_at_step_zero():
    """`ramp_frac=0` would make the "ramp" a step change at step 0, i.e. the
    end value constant -- an arm silently replaced by a different arm."""
    import train as train_mod

    with pytest.raises(ValueError, match="ramp fraction"):
        train_mod.parse_args(["--run-name", "x", "--conv-proj-wd", "0.0",
                              "--conv-proj-wd-end", "0.1"])


def test_the_trainer_writes_the_schedule_to_the_conv_group_only():
    """The wiring, without training: the scheduled value must land on the conv
    group and the other Muon group's decay must not move."""
    import train as train_mod

    args = train_mod.TrainArgs(
        run_name="conv-wd-wiring", config="tiny", data_dir="unused",
        conv_proj_wd=0.0, conv_proj_wd_end=0.1, conv_proj_wd_ramp_frac=0.1,
        device="cpu")
    trainer = train_mod.Trainer.__new__(train_mod.Trainer)
    trainer.args = args
    model = Daedalus(PRESETS["tiny"])
    trainer.muon, trainer.adamw, trainer.opt_stats = build_optimizers(
        model, conv_proj_wd=args.conv_proj_wd)
    conv_index = trainer.opt_stats["conv_proj_group_index"]
    other_index = 1 - conv_index

    trainer.step = 0
    assert trainer._conv_proj_wd(1_000) == 0.0
    trainer.step = 50
    assert trainer._conv_proj_wd(1_000) == pytest.approx(0.05)
    trainer.step = 100
    assert trainer._conv_proj_wd(1_000) == pytest.approx(0.1)

    assert trainer.muon.param_groups[other_index]["weight_decay"] == 0.1


def test_the_trainer_reports_no_schedule_when_the_group_does_not_exist():
    """With `--conv-proj-wd` unset there is no second group, and writing into
    `param_groups[1]` would either raise or retune the wrong 76.9%."""
    import train as train_mod

    args = train_mod.TrainArgs(run_name="x", config="tiny", data_dir="unused",
                               device="cpu")
    trainer = train_mod.Trainer.__new__(train_mod.Trainer)
    trainer.args = args
    trainer.step = 10

    assert trainer._conv_proj_wd(1_000) is None


# --------------------------------------------------------------- schedules ---

def test_wsd_lr_warms_up_from_zero_to_one():
    assert wsd_lr(0, 1000, warmup=100) == 0.0
    assert wsd_lr(50, 1000, warmup=100) == pytest.approx(0.5)
    assert wsd_lr(100, 1000, warmup=100) == pytest.approx(1.0)


def test_wsd_lr_is_flat_through_the_stable_phase():
    for step in (100, 300, 500, 549):
        assert wsd_lr(step, 1000, warmup=100, decay_frac=0.45) == 1.0


def test_wsd_lr_decays_linearly_to_zero_over_the_final_fraction():
    total, decay_frac = 1000, 0.45
    start = int(total * (1 - decay_frac))     # 550
    assert wsd_lr(start, total, warmup=100, decay_frac=decay_frac) == pytest.approx(1.0)
    mid = start + (total - start) // 2
    assert wsd_lr(mid, total, warmup=100, decay_frac=decay_frac) == pytest.approx(0.5, abs=1e-3)
    assert wsd_lr(total, total, warmup=100, decay_frac=decay_frac) == pytest.approx(0.0)


def test_wsd_lr_is_monotonically_non_increasing_after_warmup():
    prev = 2.0
    for step in range(300, 1001):
        cur = wsd_lr(step, 1000, warmup=300, decay_frac=0.45)
        assert cur <= prev + 1e-12
        prev = cur


def test_wsd_lr_never_goes_below_the_floor():
    assert wsd_lr(2000, 1000, warmup=100, decay_frac=0.45, floor=0.1) == 0.1
    assert wsd_lr(2000, 1000, warmup=100, decay_frac=0.45) == 0.0


def test_momentum_warmup_interpolates_then_holds():
    assert momentum_warmup(0, warmup=300) == pytest.approx(0.85)
    assert momentum_warmup(150, warmup=300) == pytest.approx(0.90)
    assert momentum_warmup(300, warmup=300) == pytest.approx(0.95)
    assert momentum_warmup(10_000, warmup=300) == pytest.approx(0.95)


def test_momentum_warmup_handles_zero_warmup():
    assert momentum_warmup(0, warmup=0) == pytest.approx(0.95)


# ------------------------------------------------------------- integration ---

def test_muon_and_adamw_together_reduce_loss_on_a_fixed_batch():
    """End-to-end sanity: the split optimizers actually train the real model."""
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model, muon_lr=0.02, adam_lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 16))

    _, first, _ = model(x, targets=x)
    for _ in range(15):
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=x)
        loss.backward()
        muon.step()
        adamw.step()
    _, last, _ = model(x, targets=x)
    assert last.item() < first.item()
