"""Tests for daedalus/arch_space.py -- the phase 6 candidate space.

The expensive failure this module prevents is a sweep that trains a candidate
which was never a comparison: not parameter-matched, or over the KV budget, or
a shape stock llama.cpp refuses after the GPU time is already spent. So most of
these tests are about what the space *excludes*, and the two named candidates
the plan calls out by name.

Run: python -m pytest tests/test_arch_space.py -v
"""
import pytest

from daedalus.arch_space import (
    ATTENTION_FRACTIONS,
    CONTROL_PRESET,
    MAX_FF_RATIO,
    MAX_KV_BYTES_PER_CONTEXT_TOKEN,
    MIN_FF_RATIO,
    PARAM_MATCH_TOLERANCE,
    PREFERRED_KV_BYTES_PER_CONTEXT_TOKEN,
    SUCCESSOR_PARAMS,
    ArchCandidate,
    analytic_screen,
    candidate_from_config,
    control_candidate,
    depth_matched_candidate,
    describe,
    generate,
    is_comparable,
    kv_bytes_at_context,
    kv_bytes_per_context_token,
    matched_candidate,
    solve_ff_dim,
    validation_failures,
)
from daedalus.config import PRESETS


CONTROL_PARAMS = PRESETS[CONTROL_PRESET].param_count()["total"]


# ------------------------------------------------------------- the KV budget --

def test_the_shipped_model_sits_exactly_on_the_plans_kv_ceiling():
    """4 KV heads x 64 head_dim x 6 attention layers x 2 tensors x 2 bytes =
    6,144, which is the ceiling the plan names. That it is *exactly* the shipped
    number is why it is a ceiling and not a target: no candidate improves on
    long-context cost without cutting attention layers or KV heads."""
    cfg = PRESETS[CONTROL_PRESET]

    assert kv_bytes_per_context_token(cfg) == MAX_KV_BYTES_PER_CONTEXT_TOKEN
    assert kv_bytes_per_context_token(cfg) == 6144


def test_conv_layers_contribute_no_kv_cache():
    """The whole reason to run a hybrid. Counting every block would erase the
    advantage on paper and rank the shipped model as a dense transformer."""
    cfg = PRESETS[CONTROL_PRESET]
    dense = PRESETS["dense-150m"]

    assert cfg.n_attn_layers == 6 and cfg.num_hidden_layers == 18
    assert kv_bytes_per_context_token(cfg) == 6 * 2 * 4 * 64 * 2
    # 24 attention layers at 2 KV heads still costs more than 6 at 4.
    assert kv_bytes_per_context_token(dense) > kv_bytes_per_context_token(cfg)


def test_kv_cost_at_the_trained_context_is_reported_in_bytes_not_per_token():
    cfg = PRESETS[CONTROL_PRESET]

    assert kv_bytes_at_context(cfg, 2048) == 6144 * 2048


def test_a_candidate_over_the_ceiling_is_refused_before_it_is_trained():
    over = ArchCandidate(
        name="greedy", hidden_size=768, num_hidden_layers=18,
        num_attention_blocks=12, num_attention_heads=12,
        num_key_value_heads=12, head_dim=64, block_ff_dim=2048)

    failures = validation_failures(over)

    assert any("KV cache" in failure for failure in failures)
    assert not is_comparable(over)


# ------------------------------------------------------- parameter matching ---

def test_the_shipped_deep_preset_is_the_under_parameterized_one_the_plan_names():
    """The reason `depth_matched_candidate` exists. `daedalus-150m-deep` is the
    same 24x640 shape at `block_ff_dim=1792`, and it is 7.7% smaller than the
    control -- so a depth result read off it is depth plus 12M missing
    parameters."""
    shipped_deep = PRESETS["daedalus-150m-deep"].param_count()["total"]

    drift = (CONTROL_PARAMS - shipped_deep) / CONTROL_PARAMS

    assert drift > PARAM_MATCH_TOLERANCE
    assert 0.07 < drift < 0.08


def test_the_corrected_depth_candidate_is_parameter_matched_to_the_control():
    candidate = depth_matched_candidate()

    total = candidate.config().param_count()["total"]
    drift = abs(total - CONTROL_PARAMS) / CONTROL_PARAMS

    assert candidate.hidden_size == 640
    assert candidate.num_hidden_layers == 24
    assert candidate.block_ff_dim == 2048
    assert drift < 0.005, f"{total:,} against {CONTROL_PARAMS:,}"
    assert is_comparable(candidate, target_params=CONTROL_PARAMS)


def test_the_corrected_depth_candidate_also_halves_the_kv_cost():
    """Not the point of the depth comparison, but it is the kind of thing that
    has to be visible when the Pareto set is chosen rather than discovered
    afterwards: 8 attention layers at 2 KV heads is 4,096 bytes against the
    control's 6,144, which is the plan's preferred value."""
    candidate = depth_matched_candidate()

    kv = kv_bytes_per_context_token(candidate.config())

    assert kv == PREFERRED_KV_BYTES_PER_CONTEXT_TOKEN
    assert describe(candidate)["kv_at_or_under_preferred"] is True


def test_the_solved_ffn_lands_within_one_rounding_step_of_the_target():
    """Parameter count is affine in `block_ff_dim`, so the solve is exact up to
    the 256 rounding. A search that got this wrong would silently hand every
    candidate a different budget."""
    for hidden, depth, blocks in ((640, 24, 8), (768, 18, 6), (512, 30, 5)):
        ff = solve_ff_dim(
            hidden_size=hidden, num_hidden_layers=depth,
            num_attention_blocks=blocks, num_attention_heads=hidden // 64,
            num_key_value_heads=2, head_dim=64, target_params=CONTROL_PARAMS)
        candidate = ArchCandidate(
            name="x", hidden_size=hidden, num_hidden_layers=depth,
            num_attention_blocks=blocks, num_attention_heads=hidden // 64,
            num_key_value_heads=2, head_dim=64, block_ff_dim=ff)

        step = 3 * hidden * depth * 256
        total = candidate.config().param_count()["total"]

        assert ff % 256 == 0
        assert abs(total - CONTROL_PARAMS) <= step, (hidden, depth, total)


def test_a_shape_too_large_to_reach_the_target_gets_the_smallest_legal_ffn():
    """A 1024-wide 30-layer stack is already past 160M before its FFN exists.
    Clamping to 256 keeps `matched_candidate` total rather than raising, and the
    parameter-match rule then rejects it -- which is the correct outcome and the
    one the caller can report."""
    candidate = matched_candidate(
        "too-big", hidden_size=1024, num_hidden_layers=30,
        num_attention_blocks=10, num_key_value_heads=2,
        target_params=CONTROL_PARAMS)

    assert candidate.block_ff_dim == 256
    failures = validation_failures(candidate, target_params=CONTROL_PARAMS)
    assert any("match tolerance" in failure for failure in failures)


# ---------------------------------------------------------- export validity ---

def test_a_head_layout_that_does_not_fill_hidden_size_is_refused():
    """Representable here and an export risk with no upside. Caught by
    arithmetic rather than by a failed conversion after the GPU time is spent."""
    candidate = ArchCandidate(
        name="ragged", hidden_size=768, num_hidden_layers=18,
        num_attention_blocks=6, num_attention_heads=10,
        num_key_value_heads=2, head_dim=64, block_ff_dim=2048)

    failures = validation_failures(candidate)

    assert any("does not equal hidden_size" in failure for failure in failures)


def test_a_kv_head_count_that_does_not_divide_the_query_heads_is_refused():
    candidate = ArchCandidate(
        name="indivisible", hidden_size=768, num_hidden_layers=18,
        num_attention_blocks=6, num_attention_heads=12,
        num_key_value_heads=5, head_dim=64, block_ff_dim=2048)

    assert any("not divisible" in failure
               for failure in validation_failures(candidate))


@pytest.mark.parametrize("field,value", [
    ("block_ff_dim", 2000), ("vocab_size", 49150)])
def test_a_dimension_the_config_itself_asserts_on_is_refused(field, value):
    """`DaedalusConfig.__post_init__` asserts both of these, so a candidate
    breaking either cannot be instantiated at all -- finding that out at export
    is finding it out after the run."""
    kwargs = dict(
        name="unquantizable", hidden_size=768, num_hidden_layers=18,
        num_attention_blocks=6, num_attention_heads=12,
        num_key_value_heads=4, head_dim=64, block_ff_dim=2048)
    kwargs[field] = value

    failures = validation_failures(ArchCandidate(**kwargs))

    assert any("multiple of 256" in failure for failure in failures)


def test_an_unaligned_hidden_size_is_a_pareto_column_not_a_rejection():
    """640 is not a multiple of 256, and both `daedalus-150m-deep` and
    `dense-150m` ship at that width while the plan asks for the 24x640 depth
    comparison by name. Refusing it would delete the comparison; the k-quant
    fallback it costs belongs beside the artifact size it changes instead."""
    deep = depth_matched_candidate()
    control = control_candidate()

    assert deep.hidden_size % 256 != 0
    assert is_comparable(deep, target_params=CONTROL_PARAMS)
    assert describe(deep)["kquant_aligned_hidden"] is False
    assert describe(control)["kquant_aligned_hidden"] is True


def test_every_failed_rule_is_reported_at_once():
    """One failure per round trip is a round trip per rule."""
    candidate = ArchCandidate(
        name="everything-wrong", hidden_size=700, num_hidden_layers=18,
        num_attention_blocks=6, num_attention_heads=10,
        num_key_value_heads=3, head_dim=64, block_ff_dim=2000)

    assert len(validation_failures(candidate)) >= 3


# ----------------------------------------------------------------- the grid ---

def test_every_generated_candidate_is_comparable_by_construction():
    """An incomparable shape is not a candidate that lost. Carrying one forward
    is how a sweep reports exploring a space it never could."""
    candidates = generate(target_params=CONTROL_PARAMS)

    assert candidates
    for candidate in candidates:
        assert is_comparable(candidate, target_params=CONTROL_PARAMS), (
            candidate.name, validation_failures(candidate,
                                                target_params=CONTROL_PARAMS))


def test_the_grid_produces_no_duplicate_shapes():
    candidates = generate(target_params=CONTROL_PARAMS)
    shapes = {(c.hidden_size, c.num_hidden_layers, c.num_attention_blocks,
               c.num_key_value_heads, c.block_ff_dim) for c in candidates}

    assert len(shapes) == len(candidates)


def test_the_grid_reaches_below_the_controls_kv_cost():
    """If nothing in the space improves on 6,144 the sweep cannot answer the
    question it was run for, and that is a property of the grid rather than of
    any result."""
    screen = analytic_screen(target_params=CONTROL_PARAMS)

    assert screen["counts"]["kv_under_control"] > 0
    assert screen["counts"]["kv_at_or_under_preferred"] > 0


def test_the_control_is_described_alongside_the_candidates():
    """Every criterion is relative to it, so a screen without it is unreadable
    -- the same reason the phase 5 sweep adds its control to any subset."""
    screen = analytic_screen(target_params=CONTROL_PARAMS)

    assert screen["control"]["name"] == CONTROL_PRESET
    assert screen["control"]["kv_bytes_per_context_token"] == 6144
    assert screen["control"]["attention_layers"] == 6


def test_the_control_reads_back_as_the_shape_it_actually_has():
    """`num_attention_blocks` is the requested count and `layer_types` is what
    the interleaver produced. Every KV number depends on the realised one."""
    control = control_candidate()

    assert control.num_attention_blocks == PRESETS[CONTROL_PRESET].n_attn_layers
    assert (control.config().param_count()["total"]
            == PRESETS[CONTROL_PRESET].param_count()["total"])


def test_a_preset_survives_a_round_trip_through_the_candidate_form():
    for name in ("daedalus-150m", "daedalus-150m-deep", "dense-150m"):
        candidate = candidate_from_config(name, PRESETS[name])

        assert (candidate.config().param_count()["total"]
                == PRESETS[name].param_count()["total"])
        assert (kv_bytes_per_context_token(candidate.config())
                == kv_bytes_per_context_token(PRESETS[name]))


def test_a_shape_whose_ffn_absorbs_the_whole_budget_is_not_a_candidate():
    """`block_ff_dim` is solved for the parameter target, so a shape with
    almost no attention puts everything into the FFN: hidden 512 with one
    attention layer solves to 25,088 at the 500M target, a 49x aspect ratio.
    That is the solver saying the shape cannot hold 500M any other way, not an
    architecture that lost, and it would otherwise dominate the KV ranking
    because one attention layer is also the cheapest possible cache."""
    degenerate = matched_candidate(
        "d12x512-a1-kv1", hidden_size=512, num_hidden_layers=12,
        num_attention_blocks=1, num_key_value_heads=1,
        target_params=SUCCESSOR_PARAMS)

    assert degenerate.block_ff_dim / degenerate.hidden_size > 10
    assert any("outside the" in failure
               for failure in validation_failures(degenerate))
    assert degenerate.name not in {
        record["name"] for record
        in analytic_screen(target_params=SUCCESSOR_PARAMS)["candidates"]}


def test_the_shipped_control_sits_inside_the_ffn_band_it_anchors():
    """The band is anchored on the shipped 2.67x. A rule that excluded the
    control would be a rule about nothing."""
    control = control_candidate()

    assert MIN_FF_RATIO <= 2048 / 768 <= MAX_FF_RATIO
    assert is_comparable(control, target_params=CONTROL_PARAMS)
    assert describe(control)["ff_ratio"] == pytest.approx(2048 / 768)


def test_every_surviving_candidate_has_a_trainable_aspect_ratio():
    for target in (CONTROL_PARAMS, SUCCESSOR_PARAMS):
        for record in analytic_screen(target_params=target)["candidates"]:
            assert MIN_FF_RATIO <= record["ff_ratio"] <= MAX_FF_RATIO, record


def test_the_attention_fractions_span_the_shipped_ratio_downwards():
    """The cache is the reason to run a hybrid, so the grid has to be able to
    cut attention rather than only re-arrange it. The shipped 1/3 is the top of
    the range, not the middle of it."""
    assert max(ATTENTION_FRACTIONS) == pytest.approx(1 / 3)
    assert PRESETS[CONTROL_PRESET].n_attn_layers / \
        PRESETS[CONTROL_PRESET].num_hidden_layers == pytest.approx(1 / 3)


# -------------------------------------------------------------- successor ----

def test_the_successor_scale_is_screened_without_being_trained_here():
    """The operator fixed a ~500M successor after this phase was written for
    150M. KV cost scales with attention layers x KV heads rather than with
    parameters, so the analytic screen has to run at the real target -- while a
    quality ranking measured on 150M proxies stays a ranking at 150M."""
    screen = analytic_screen(target_params=SUCCESSOR_PARAMS)

    assert SUCCESSOR_PARAMS == 500_000_000
    assert screen["candidates"], "no comparable shape at the successor scale"
    for record in screen["candidates"]:
        assert abs(record["param_drift_pct"]) <= 100 * PARAM_MATCH_TOLERANCE
        assert record["kv_bytes_per_context_token"] <= \
            MAX_KV_BYTES_PER_CONTEXT_TOKEN


def test_a_screen_above_the_controls_scale_says_the_control_is_not_matched():
    """The control is the shipped 160M model at every target, because the KV
    ceiling the plan gates on is its 6,144 bytes and that does not move with
    parameter count. Its parameter column does, so at the 500M target the row
    would otherwise read as a candidate that came in 68% under budget."""
    at_control = analytic_screen(target_params=CONTROL_PARAMS)
    at_successor = analytic_screen(target_params=SUCCESSOR_PARAMS)

    assert at_control["control_is_parameter_matched"] is True
    assert at_control["control_note"] is None
    assert at_successor["control_is_parameter_matched"] is False
    assert "not comparable here" in at_successor["control_note"]
    # The KV reference is the part that stays valid at both scales.
    assert (at_successor["control"]["kv_bytes_per_context_token"]
            == at_control["control"]["kv_bytes_per_context_token"] == 6144)


def test_the_successor_scale_needs_more_than_a_wider_control():
    """A 500M model at the shipped 18x768 shape would be almost all FFN. If the
    space at that target were a single shape the screen would be describing an
    arithmetic identity rather than a choice."""
    screen = analytic_screen(target_params=SUCCESSOR_PARAMS)

    depths = {record["num_hidden_layers"] for record in screen["candidates"]}
    widths = {record["hidden_size"] for record in screen["candidates"]}

    assert len(depths) > 1 and len(widths) > 1
