"""Tests for daedalus/conv_health.py -- the phase 5 channel-death instrument.

`runs/preflight/conv-death-fix-validated.md` records two versions of this
instrument's negative control that were wrong before it was right, and both
failures looked like success. The lesson it draws is that a metric which decides
between weight-decay schedules cannot itself be a weight-magnitude metric, and
the tests below are written to that: most of them are about the *instrument*
rather than about a model.

Run: python -m pytest tests/test_conv_health.py -v
"""
import pytest
import torch

from daedalus.config import PRESETS
from daedalus.conv_health import (
    DEAD_THRESHOLD,
    FACTORS,
    ChannelHealthUndefined,
    ablated,
    ablated_model,
    channel_factor_scales,
    conv_layers,
    dead_channels,
    layer_health,
    model_health,
    projection_norms,
    strongest_alive,
    weakest_alive,
)
from daedalus.model import Daedalus, ShortConv


def _trained_looking_layer(hidden=32, seed=0):
    """A `ShortConv` whose weights look like a run rather than like step 0.

    `out_proj` is zero-initialised in the real model, so a freshly constructed
    layer has no defined scale at all -- which is itself a test below, but makes
    it useless as the fixture for every other one.
    """
    cfg = PRESETS["tiny"]
    layer = ShortConv(cfg)
    torch.manual_seed(seed)
    with torch.no_grad():
        layer.in_proj.weight.normal_(0.0, 0.02)
        layer.conv.weight.normal_(0.0, 0.05)
        layer.out_proj.weight.normal_(0.0, 0.02)
    return layer


def _collapse(tensor, index, scale=1e-11):
    with torch.no_grad():
        tensor[index] = tensor[index] * scale


# ------------------------------------------------------- refusing to guess ----

def test_an_untrained_layer_is_refused_rather_than_scored_as_all_dead():
    """`model.py:204` zero-initialises `conv.out_proj`, so at step 0 every
    relative scale is 0/0. Reporting 100% dead -- or a NaN that an aggregate
    then absorbs -- would put a number nobody computed into a phase 5 gate."""
    layer = ShortConv(PRESETS["tiny"])
    with torch.no_grad():
        layer.out_proj.weight.zero_()

    health = layer_health(layer)

    assert health.defined is False
    assert "zero-initialised" in health.undefined_reason
    with pytest.raises(ChannelHealthUndefined):
        _ = health.dead_count
    assert health.to_json()["defined"] is False
    assert "dead_fraction" not in health.to_json()


def test_a_model_with_an_undefined_layer_raises_instead_of_aggregating():
    """The aggregate is the number a gate reads. One unscoreable layer has to
    stop it, not be averaged into it."""
    model = Daedalus(PRESETS["tiny"])          # out_proj still zero-init

    with pytest.raises(ChannelHealthUndefined, match="no defined channel scale"):
        model_health(model)

    lenient = model_health(model, strict=False)
    assert lenient.undefined_layers
    assert lenient.to_json()["defined"] is False
    assert "dead_fraction" not in lenient.to_json()


# ------------------------------------------------ the multiplicative reading --

def test_a_collapsed_out_proj_column_is_flagged():
    """The case the shipped proxy was built for, which must still hold."""
    layer = _trained_looking_layer()
    _collapse(layer.out_proj.weight, (slice(None), 7))

    health = layer_health(layer)

    assert dead_channels(health) == [7]
    assert health.proxy_dead[7]


@pytest.mark.parametrize("factor_row,name", [(0, "gate_B"), (1, "gate_C"), (2, "value_x")])
def test_a_collapsed_in_proj_row_is_flagged_where_the_out_proj_proxy_misses_it(
        factor_row, name):
    """The whole reason this module exists.

    `ShortConv` is `out_proj(C * conv(B * x))`, so a channel whose `B` row has
    collapsed contributes exactly nothing while its `out_proj` column is still
    perfectly healthy. The shipped ruler reads only that column and scores it
    alive; under phase 5's arms, where decay on these projections is the
    variable, that is the gap that could certify a fix which does not work.
    """
    layer = _trained_looking_layer()
    hidden = layer.out_proj.weight.shape[0]
    _collapse(layer.in_proj.weight, (factor_row * hidden + 5,))

    health = layer_health(layer)

    assert dead_channels(health) == [5], f"{name} collapse not caught"
    assert not health.proxy_dead[5], (
        "the out_proj proxy is expected to miss this -- if it now catches it, "
        "the fixture no longer isolates the coupled factor")


def test_a_collapsed_kernel_row_is_flagged():
    """The depthwise kernel is the third coupled tensor, and it is in AdamW
    rather than Muon -- so it is the factor a conv-decay schedule does not
    touch, and the one an in_proj/out_proj-only ruler cannot see at all."""
    layer = _trained_looking_layer()
    _collapse(layer.conv.weight, (11,))

    health = layer_health(layer)

    assert dead_channels(health) == [11]
    assert not health.proxy_dead[11]


def test_the_flagged_set_is_a_superset_of_the_shipped_proxy():
    """Strengthening the ruler must not lose anything the old one caught: the
    47.9% reading in `conv-channel-death.md` has to stay readable on this
    instrument, or the phase 5 gate is not comparable to the baseline."""
    layer = _trained_looking_layer()
    hidden = layer.out_proj.weight.shape[0]
    _collapse(layer.out_proj.weight, (slice(None), 1))
    _collapse(layer.out_proj.weight, (slice(None), 2))
    _collapse(layer.in_proj.weight, (hidden + 3,))
    _collapse(layer.conv.weight, (4,))

    health = layer_health(layer)
    functional = set(dead_channels(health))
    proxy = {int(i) for i in torch.nonzero(health.proxy_dead).flatten()}

    assert proxy == {1, 2}
    assert functional == {1, 2, 3, 4}
    assert proxy < functional


def test_the_out_factor_reproduces_the_shipped_proxy_exactly():
    """Not "close to": the `out` factor is mean |w| down each column against
    the layer's p95 of those means, which is the shipped ruler verbatim. If
    these ever diverge, the 47.9% and the phase 5 dead fraction stop being
    measured on one scale."""
    layer = _trained_looking_layer(seed=3)
    _collapse(layer.out_proj.weight, (slice(None), 9))

    health = layer_health(layer)
    columns = layer.out_proj.weight.detach().to(torch.float64).abs().mean(dim=0)
    shipped = columns < DEAD_THRESHOLD * float(torch.quantile(columns, 0.95))

    assert torch.equal(health.proxy_dead, shipped)


def test_every_declared_factor_is_actually_read():
    """`FACTORS` is the contract the report, the ablation and these tests share.
    A factor listed but never scored would silently stop being checked."""
    layer = _trained_looking_layer()
    scales = channel_factor_scales(layer)
    health = layer_health(layer)

    assert set(scales) == set(FACTORS)
    assert set(health.factor_p95) == set(FACTORS)
    assert all(scale.shape == (layer.out_proj.weight.shape[0],)
               for scale in scales.values())


def test_the_dead_count_is_insensitive_to_the_threshold():
    """`conv-channel-death.md` found death is binary here -- the largest dead
    channel sat at 2.195e-11 against a smallest alive one orders above. So the
    count must not move over two decades of threshold, and a count that does is
    reporting where the bar sits rather than how many channels died."""
    layer = _trained_looking_layer()
    _collapse(layer.out_proj.weight, (slice(None), 6))
    _collapse(layer.in_proj.weight, (13,))

    counts = {t: layer_health(layer, threshold=t).dead_count
              for t in (0.001, 0.01, 0.1)}

    assert set(counts.values()) == {2}, counts


# ---------------------------------------------------------------- ablation ----

def test_ablating_a_flagged_channel_leaves_the_output_unchanged():
    """What "dead" has to mean. This is the check that made the 47.9% credible
    (zeroing 4,417 channels moved held-out loss by exactly 0.0), reproduced at
    the level of one layer's forward pass where the delta can be *exactly* zero
    rather than merely small."""
    layer = _trained_looking_layer()
    _collapse(layer.out_proj.weight, (slice(None), 7), scale=0.0)
    torch.manual_seed(1)
    u = torch.randn(2, 16, layer.out_proj.weight.shape[0])

    health = layer_health(layer)
    with torch.no_grad():
        before, _ = layer(u)
        with ablated(layer, dead_channels(health)):
            after, _ = layer(u)

    assert torch.equal(before, after)


def test_ablating_the_strongest_channels_costs_more_than_the_weakest():
    """The ladder from `conv-death-fix-validated.md`.

    An instrument that flags nothing on a clean arm proves nothing by ablating
    its empty flagged set, so the ranking has to be load-bearing on its own.
    The earlier version of this control certified an instrument that could not
    detect anything, because on an *untrained* model every ablation is free --
    hence a layer with real, differentiated weights and a comparison between
    the two ends of the ranking rather than against zero.
    """
    layer = _trained_looking_layer(seed=5)
    hidden = layer.out_proj.weight.shape[0]
    # Give the layer a real spread of channel strengths to rank.
    with torch.no_grad():
        gains = torch.linspace(0.05, 1.0, hidden)
        layer.out_proj.weight.mul_(gains.unsqueeze(0))
    torch.manual_seed(2)
    u = torch.randn(2, 16, hidden)

    health = layer_health(layer)
    k = 4
    with torch.no_grad():
        before, _ = layer(u)
        with ablated(layer, weakest_alive(health, k)):
            weak, _ = layer(u)
        with ablated(layer, strongest_alive(health, k)):
            strong, _ = layer(u)

    weak_cost = float((weak - before).abs().mean())
    strong_cost = float((strong - before).abs().mean())
    assert weak_cost > 0.0, "the weakest live channels must still do something"
    assert strong_cost > 5 * weak_cost, (
        f"ranking is not load-bearing: strongest {strong_cost:.3e} vs weakest "
        f"{weak_cost:.3e}")


def test_ablation_restores_every_slice_it_touched():
    """A scoring pass must not leave the model mutilated for whatever runs
    next, including when the block raises."""
    layer = _trained_looking_layer()
    saved = {name: tensor.detach().clone()
             for name, tensor in layer.state_dict().items()}

    with pytest.raises(RuntimeError):
        with ablated(layer, [0, 1, 2]):
            raise RuntimeError("boom")

    for name, tensor in layer.state_dict().items():
        assert torch.equal(tensor, saved[name]), f"{name} not restored"


def test_ablation_zeroes_all_five_coupled_slices():
    """Zeroing the `out_proj` column alone would remove the contribution, so a
    partial ablation is invisible in a forward pass and only shows up later as
    a channel the norm report still counts."""
    layer = _trained_looking_layer()
    hidden = layer.out_proj.weight.shape[0]

    with ablated(layer, [3]):
        scales = channel_factor_scales(layer)
        assert all(float(scales[name][3]) == 0.0 for name in FACTORS), scales
        # Neighbours untouched: an ablation that spills is not an ablation.
        assert all(float(scales[name][4]) > 0.0 for name in FACTORS)
        assert float(layer.in_proj.weight.detach()[hidden + 3].abs().sum()) == 0.0


def test_ablated_model_reaches_every_named_layer_and_rejects_unknown_ones():
    model = Daedalus(PRESETS["tiny"])
    indices = [index for index, _ in conv_layers(model)]
    assert indices, "the tiny preset is expected to have conv layers"

    with ablated_model(model, {indices[0]: [0, 1]}):
        layer = dict(conv_layers(model))[indices[0]]
        assert float(layer.in_proj.weight.detach()[0].abs().sum()) == 0.0
    restored = dict(conv_layers(model))[indices[0]].in_proj.weight.detach()
    assert float(restored[0].abs().sum()) > 0.0

    with pytest.raises(KeyError):
        with ablated_model(model, {999: [0]}):
            pass


# -------------------------------------------------------------- layer scope ---

def test_conv_layers_finds_only_conv_blocks_and_keeps_block_indices():
    """Indexed by block so a per-layer reading lines up with `cfg.layer_types`
    and with the per-layer table in `conv-channel-death.md`."""
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)

    found = conv_layers(model)
    expected = [i for i, t in enumerate(cfg.layer_types) if t == "conv"]

    assert [index for index, _ in found] == expected
    assert all(isinstance(layer, ShortConv) for _, layer in found)


# --------------------------------------------------------- projection norms ---

def test_projection_norms_are_reported_over_alive_channels_only():
    """`conv-death-fix-validated.md` flags this as a correction to how the
    6.8x/10.5x growth ratios should be read: a baseline arm's mean is dragged
    down by its own near-zero columns, which flatters the arm that is failing.
    Phase 5 gates on norms staying within 2x the alive-channel baseline, so the
    alive-only reading has to be the default rather than a footnote.
    """
    layer = _trained_looking_layer()
    hidden = layer.out_proj.weight.shape[0]
    with_all_alive = projection_norms(layer)

    for channel in range(hidden // 2):
        _collapse(layer.out_proj.weight, (slice(None), channel), scale=0.0)
    half_dead = projection_norms(layer)

    assert half_dead["alive_channels"] == hidden - hidden // 2
    # The surviving channels did not change, so their mean must not either.
    assert half_dead["out_proj"] == pytest.approx(with_all_alive["out_proj"],
                                                  rel=0.35)
    naive = float(layer.out_proj.weight.detach().abs().mean())
    assert naive < 0.6 * half_dead["out_proj"], (
        "a naive all-channel mean is expected to be dragged down here; if it "
        "is not, this test no longer demonstrates the correction")


def test_projection_norms_refuse_a_layer_with_nothing_alive():
    layer = _trained_looking_layer()
    hidden = layer.out_proj.weight.shape[0]
    for channel in range(hidden):
        _collapse(layer.out_proj.weight, (slice(None), channel), scale=0.0)

    with pytest.raises(ChannelHealthUndefined):
        projection_norms(layer)
