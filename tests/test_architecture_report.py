"""Tests for scripts/architecture_report.py -- the stage-A rule and its guards.

What can go wrong here is not arithmetic. It is a screen that reads a bounded
mid-decay sample as if it were the arm's result, an aggregate that quietly
scored the wrong corpus, a table without the control every column is a delta
against, or a parameter surplus reported as a quality win. Each of those
produces a confident number that recommends the wrong successor, and none of
them looks like a failure.

The rule is exercised on synthetic rows so that the thresholds are pinned
independently of what the fifteen real arms happen to have scored.

Run: python -m pytest tests/test_architecture_report.py -v
"""
import json

import pytest

from daedalus.scorecard import (ArtifactRef, Provenance, Scorecard,
                                ScorecardError, write_scorecard)
from scripts.architecture_report import (MATCHED_HOLDOUT_SOURCE,
                                         PARAM_SCALING_EXPONENT,
                                         STAGE_B_FLOOR_PCT, STAGE_B_MAX_ARMS,
                                         arm_bpb, build_report,
                                         credited_bpb_delta_pct,
                                         pareto_frontier, read_rows,
                                         render_markdown, score_arm,
                                         scorecard_path, scored_from,
                                         select_stage_b)
from scripts.architecture_sweep import ARMS_BY_NAME, CONTROL, arm_checkpoint_path


# --------------------------------------------------------------- fixtures ----

def _row(arm, *, bpb, kv, parameters=105_000_000, is_control=False):
    """One pre-join row, in the shape `read_rows` produces before deltas."""
    return {"arm": arm, "preset": f"arch-{arm}", "is_control": is_control,
            "attention_layers": 8, "num_key_value_heads": 4,
            "kv_bytes_per_context_token": kv, "parameters": parameters,
            "q4_0_MB": 60.0, "bpb": bpb, "checkpoint_sha256": "0" * 64,
            "scorecard": f"{arm}.json"}


def _with_deltas(rows):
    """Apply the same delta columns `read_rows` does, without needing files."""
    control = next(row for row in rows if row["is_control"])
    for row in rows:
        row["bpb_delta_pct"] = 100.0 * (row["bpb"] - control["bpb"]) / control["bpb"]
        row["param_surplus_pct"] = 100.0 * (
            row["parameters"] - control["parameters"]) / control["parameters"]
        row["param_explained_bpb_pct"] = -PARAM_SCALING_EXPONENT * row["param_surplus_pct"]
        row["credited_bpb_delta_pct"] = credited_bpb_delta_pct(
            row["bpb_delta_pct"], row["param_surplus_pct"])
        row["kv_saving_pct"] = 100.0 * (
            control["kv_bytes_per_context_token"]
            - row["kv_bytes_per_context_token"]) \
            / control["kv_bytes_per_context_token"]
        row["passes_floor"] = row["bpb_delta_pct"] <= STAGE_B_FLOOR_PCT
        row["quality_win_survives_param_margin"] = (
            row["bpb_delta_pct"] < 0.0 and row["credited_bpb_delta_pct"] < 0.0)
    return rows


def _write_card(path, *, name, bpb, sha="a" * 64, source=MATCHED_HOLDOUT_SOURCE,
                bpb_mode="full", sample_batches=None, config="arch-a8-kv4"):
    card = Scorecard(
        kind="bpb", name=name,
        provenance=Provenance(
            artifact=ArtifactRef(path="ckpt.pt", sha256=sha, kind="checkpoint",
                                 config=config),
            tokenizer=ArtifactRef(path="tok", sha256="0" * 64, kind="tokenizer"),
            seed=1, git_sha="deadbee", bpb_mode=bpb_mode,
            bpb_sample_batches=sample_batches),
        metrics={"bpb": bpb, "n_sources": 1.0},
        created_at="2026-08-25T00:00:00Z",
        items=[{"id": source, "bpb": bpb, "tokens": 1000}])
    return write_scorecard(path, card)


# ------------------------------------------------------ parameter discount ----

def test_a_parameter_surplus_can_explain_away_a_raw_bpb_win():
    """The grid's residual spread favours attention-sparse arms by up to 3.05%,
    which is the direction this phase hopes to find a winner. An arm 0.5%
    better on raw BPB while carrying 3.05% more parameters has not beaten the
    control -- it has outspent it."""
    credited = credited_bpb_delta_pct(-0.5, 3.05)

    assert credited == pytest.approx(-0.5 + 0.34 * 3.05)
    assert credited > 0.0, "a bought win must not read as a quality win"


def test_the_discount_is_neutral_for_a_parameter_matched_arm():
    assert credited_bpb_delta_pct(-0.4, 0.0) == pytest.approx(-0.4)


def test_a_smaller_arm_is_credited_rather_than_penalised():
    """A win from an arm carrying *fewer* parameters is worth more than it
    looks, not less."""
    assert credited_bpb_delta_pct(-0.4, -1.0) < -0.4


# --------------------------------------------------------------- frontier ----

def test_the_frontier_drops_a_shape_beaten_at_equal_cache_cost():
    """8 attention layers with 1 KV head, 4 with 2 and 2 with 4 all cost the
    same bytes per context token. At equal cost only the better-scoring shape
    is worth stage B's hours."""
    rows = [_row("a8-kv1", bpb=1.20, kv=1024),
            _row("a4-kv2", bpb=1.26, kv=1024),
            _row("a2-kv4", bpb=1.31, kv=1024)]

    assert [row["arm"] for row in pareto_frontier(rows)] == ["a8-kv1"]


def test_the_frontier_keeps_a_worse_arm_that_costs_less_cache():
    """The frontier is the trade, not the quality ranking: a cheaper cache that
    scores worse is exactly the point of the phase."""
    rows = [_row("a8-kv4", bpb=1.20, kv=4096),
            _row("a4-kv1", bpb=1.25, kv=512)]

    assert [row["arm"] for row in pareto_frontier(rows)] == ["a4-kv1", "a8-kv4"]


def test_the_frontier_is_ordered_cheapest_cache_first():
    rows = [_row("mid", bpb=1.22, kv=2048),
            _row("cheap", bpb=1.30, kv=512),
            _row("dear", bpb=1.18, kv=4096)]

    assert [row["arm"] for row in pareto_frontier(rows)] == ["cheap", "mid", "dear"]


# -------------------------------------------------------------- selection ----

def test_an_arm_worse_than_the_floor_does_not_advance():
    rows = _with_deltas([_row("a8-kv4", bpb=1.2000, kv=4096, is_control=True),
                         _row("a2-kv1", bpb=1.2500, kv=512)])

    decision = select_stage_b(rows)

    assert decision["eligible"] == []
    assert decision["selected"] == []
    assert decision["verdict"] == "no-advance"


def test_no_eligible_arm_is_a_recorded_negative_result():
    """A screen that finds nothing must say so, not be re-run at a looser
    threshold."""
    rows = _with_deltas([_row("a8-kv4", bpb=1.20, kv=4096, is_control=True),
                         _row("a2-kv1", bpb=1.40, kv=512)])

    decision = select_stage_b(rows)

    assert decision["verdict"] == "no-advance"
    assert "negative result" in decision["note"]
    assert decision["rule"]["floor_pct"] == STAGE_B_FLOOR_PCT


def test_the_floor_is_inclusive_at_exactly_the_preregistered_bound():
    """0.5% worse is 'no worse by more than 0.5%', so it passes."""
    control_bpb = 1.2000
    rows = _with_deltas([_row("a8-kv4", bpb=control_bpb, kv=4096, is_control=True),
                         _row("a2-kv1", bpb=control_bpb * 1.005, kv=512)])

    assert select_stage_b(rows)["eligible"] == ["a2-kv1"]


def test_the_control_never_occupies_a_stage_b_slot():
    """It runs in every stage anyway; spending a slot on it would drop a real
    candidate."""
    rows = _with_deltas([_row("a8-kv4", bpb=1.20, kv=4096, is_control=True),
                         _row("a2-kv1", bpb=1.20, kv=512)])

    decision = select_stage_b(rows)

    assert CONTROL.name not in decision["selected"]
    assert decision["selected"] == ["a2-kv1"]


def test_selection_is_capped_at_the_preregistered_width():
    """Six arms all inside the floor and all on the frontier -- each cheaper one
    scoring slightly worse, which is the trade the phase is buying.

    Deliberately off the 0.5% boundary; `test_the_floor_is_inclusive_at_exactly
    _the_preregistered_bound` owns that edge, and a fixture sitting on it here
    would fail on the last bit of a float rather than on the rule.
    """
    rows = _with_deltas(
        [_row("a8-kv4", bpb=1.2000, kv=4096, is_control=True)]
        + [_row(f"arm{i}", bpb=1.2050 - i * 0.0005, kv=512 * (i + 1))
           for i in range(6)])

    decision = select_stage_b(rows)

    assert len(decision["frontier"]) == 6
    assert len(decision["selected"]) == STAGE_B_MAX_ARMS
    assert decision["selected"] == ["arm0", "arm1", "arm2"], "cheapest cache first"
    assert decision["dropped_from_frontier"], "a silent cap reads as full coverage"


def test_an_advance_bought_with_parameters_is_labelled_as_such():
    """An arm may advance on its KV saving while its raw BPB win is inside the
    parameter margin. The report must not let that be quoted as beating the
    shipped ratio.

    The parameter counts are the grid's real ones: `a2-kv4` is the largest arm
    at 106,489,600 against the control's 104,908,288, a 1.51% surplus, which
    accounts for 0.51% of BPB at the discount exponent. A 0.4% raw win is
    inside that.
    """
    rows = _with_deltas([
        _row("a8-kv4", bpb=1.2000, kv=4096, parameters=104_908_288,
             is_control=True),
        _row("a2-kv4", bpb=1.1952, kv=2048, parameters=106_489_600)])

    decision = select_stage_b(rows)

    assert decision["selected"] == ["a2-kv4"]
    assert "parameter surplus" in decision["note"]
    assert rows[1]["bpb_delta_pct"] < 0
    assert rows[1]["quality_win_survives_param_margin"] is False


def test_a_genuine_quality_win_is_not_labelled_as_bought():
    rows = _with_deltas([
        _row("a8-kv4", bpb=1.2000, kv=4096, parameters=104_908_288,
             is_control=True),
        _row("a2-kv1", bpb=1.1500, kv=512, parameters=106_096_384)])

    decision = select_stage_b(rows)

    assert rows[1]["quality_win_survives_param_margin"] is True
    assert "note" not in decision


# ------------------------------------------------------------ measurement ----

def test_a_sampled_scorecard_is_refused_by_the_gate(tmp_path):
    """Every arm's in-training val_bpb is a bounded sample taken at step 1,500
    of 1,536, with the learning rate still an order of magnitude above its
    floor. Reading one as the arm's result would rank models mid-decay."""
    path = tmp_path / "card.json"
    _write_card(path, name="sampled", bpb=1.2, bpb_mode="sample",
                sample_batches=100)

    from daedalus.scorecard import load_scorecard

    with pytest.raises(ScorecardError, match="full pass"):
        arm_bpb(load_scorecard(path))


def test_a_scorecard_for_another_corpus_is_refused(tmp_path):
    """Reading whatever single item a card happens to carry would score an arm
    on a source it never trained on and call it the screen's answer."""
    path = tmp_path / "card.json"
    _write_card(path, name="wrong-corpus", bpb=1.2, source="stack-edu-python")

    from daedalus.scorecard import load_scorecard

    with pytest.raises(ScorecardError, match="no item for source"):
        arm_bpb(load_scorecard(path))


# ------------------------------------------------------------------ rows -----

def test_read_rows_refuses_a_table_without_the_control(tmp_path):
    """Every stage-A column is a delta against the control, so a table without
    it is unreadable rather than merely incomplete."""
    arm = ARMS_BY_NAME["a2-kv1"]
    _write_card(scorecard_path(arm, out_dir=str(tmp_path)),
                name="arch-stagea-a2-kv1-bpb", bpb=1.25, config=arm.config)

    with pytest.raises(ScorecardError, match="control"):
        read_rows([arm], out_dir=str(tmp_path))


def test_read_rows_joins_the_measured_column_to_the_analytic_ones(tmp_path):
    control, arm = CONTROL, ARMS_BY_NAME["a2-kv1"]
    _write_card(scorecard_path(control, out_dir=str(tmp_path)),
                name="c", bpb=1.2000, config=control.config)
    _write_card(scorecard_path(arm, out_dir=str(tmp_path)),
                name="a", bpb=1.2120, config=arm.config)

    rows = read_rows([control, arm], out_dir=str(tmp_path))
    measured = {row["arm"]: row for row in rows}

    assert measured[arm.name]["bpb_delta_pct"] == pytest.approx(1.0)
    assert measured[arm.name]["kv_bytes_per_context_token"] == \
        arm.kv_bytes_per_context_token
    assert measured[arm.name]["kv_saving_pct"] > 0
    assert measured[arm.name]["passes_floor"] is False
    assert measured[control.name]["bpb_delta_pct"] == pytest.approx(0.0)


def test_an_arm_that_was_never_scored_is_absent_rather_than_zero(tmp_path):
    control = CONTROL
    _write_card(scorecard_path(control, out_dir=str(tmp_path)),
                name="c", bpb=1.2, config=control.config)

    rows = read_rows([control, ARMS_BY_NAME["a2-kv1"]], out_dir=str(tmp_path))

    assert [row["arm"] for row in rows] == [control.name]


# ----------------------------------------------------------------- report ----

def test_the_report_carries_the_parameter_margin_and_scale_caveats():
    rows = _with_deltas([_row("a8-kv4", bpb=1.20, kv=4096, is_control=True),
                         _row("a2-kv1", bpb=1.21, kv=512)])

    report = build_report(rows)
    caveats = " ".join(report["caveats"])

    assert "parameter-matched only" in caveats
    assert "ranking at that scale" in caveats
    assert "Retrieval by depth" in caveats
    assert report["bpb_mode"] == "full"
    assert report["holdout_source"] == MATCHED_HOLDOUT_SOURCE


def test_the_markdown_table_shows_the_credited_column_beside_the_raw_one():
    rows = _with_deltas([
        _row("a8-kv4", bpb=1.2000, kv=4096, parameters=104_908_288,
             is_control=True),
        _row("a2-kv1", bpb=1.1940, kv=512, parameters=106_096_384)])

    markdown = render_markdown(build_report(rows))

    assert "credited %" in markdown
    assert "param surplus %" in markdown
    assert "Stage B selection" in markdown


# ---------------------------------------------------------------- scoring ----

def test_scoring_is_skipped_only_when_the_same_bytes_were_scored(tmp_path):
    path = tmp_path / "card.json"
    _write_card(path, name="c", bpb=1.2, sha="b" * 64)

    assert scored_from(path, "b" * 64) is True
    assert scored_from(path, "c" * 64) is False, \
        "a retrained arm must not keep the previous arm's number"
    assert scored_from(tmp_path / "missing.json", "b" * 64) is False


def test_score_arm_refuses_an_arm_with_no_checkpoint(tmp_path):
    """A missing checkpoint is an arm that never ran, and scoring must say so
    rather than skip it into a table that looks complete."""
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        score_arm(ARMS_BY_NAME["a2-kv1"], holdout_root=str(tmp_path),
                  run_root=str(tmp_path), out_dir=str(tmp_path))


def test_score_arm_measures_a_full_pass_over_the_matched_source(tmp_path):
    """The scorecard a later gate reads must record which corpus, which bytes,
    and that it was a full pass -- not a bound it never had."""
    arm = ARMS_BY_NAME["a2-kv1"]
    checkpoint = arm_checkpoint_path(arm, run_root=str(tmp_path))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")

    holdout = tmp_path / "holdout"
    for name in (MATCHED_HOLDOUT_SOURCE, "stack-edu-python"):
        (holdout / name).mkdir(parents=True)
        (holdout / name / "manifest.json").write_text(
            json.dumps({"total_tokens": 1000}))

    seen = []

    def fake_factory(_arm, _ckpt, **_kw):
        return lambda source_dir: (seen.append(source_dir.name), 1.25)[1]

    result = score_arm(arm, holdout_root=str(holdout), run_root=str(tmp_path),
                       out_dir=str(tmp_path / "cards"), device="cpu",
                       bpb_factory=fake_factory)

    assert seen == [MATCHED_HOLDOUT_SOURCE], "scored a source the arm never saw"
    card = json.loads((tmp_path / "cards" /
                       "arch-stagea-a2-kv1-bpb.json").read_text())
    assert card["provenance"]["bpb_mode"] == "full"
    assert card["provenance"]["bpb_sample_batches"] is None
    assert card["provenance"]["artifact"]["config"] == arm.config
    assert card["details"]["sources_requested"] == [MATCHED_HOLDOUT_SOURCE]
    assert result["checkpoint_sha256"] == \
        card["provenance"]["artifact"]["sha256"]
