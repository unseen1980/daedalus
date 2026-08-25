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
import math
from pathlib import Path

import pytest

from daedalus.scorecard import (ArtifactRef, Provenance, Scorecard,
                                ScorecardError, write_scorecard)
from scripts.architecture_report import (GATE_COLUMNS, MATCHED_HOLDOUT_SOURCE,
                                         PARAM_SCALING_EXPONENT,
                                         RETRIEVAL_GATE_TASKS,
                                         RETRIEVAL_MAX_DROP_POINTS,
                                         RETRIEVAL_MIN_ITEMS_PER_DEPTH,
                                         STAGE_B_FLOOR_PCT, STAGE_B_MAX_ARMS,
                                         arm_bpb, build_recommendation,
                                         build_report,
                                         credited_bpb_delta_pct, decode_check,
                                         export_check, gate_arm, gate_verdict,
                                         kv_check, pareto_frontier, read_rows,
                                         read_decode_passes, render_markdown,
                                         render_recommendation_markdown,
                                         retrieval_check, score_arm,
                                         scorecard_path, scored_from,
                                         select_stage_b)
from scripts.architecture_report import main as architecture_report_main
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


# =================================================== the recommendation gate ==
# What fails here does not look like a failure. It looks like a table with one
# strong column and four empty ones, read as a recommendation because the empty
# ones did not object. Every test below pins a refusal.

def _retrieval_card(path, *, task, depths, artifact_kind="gguf-q4_0",
                    per_depth=100):
    """A retrieval scorecard shaped as `daedalus.retrieval.summarize` writes."""
    metrics = {"exact_match": sum(depths.values()) / len(depths),
               "query_accuracy": 1.0, "n": float(per_depth * len(depths))}
    for depth, score in depths.items():
        metrics[f"exact_match_d{depth}"] = score
        metrics[f"n_d{depth}"] = float(per_depth)
    card = Scorecard(
        kind="retrieval", name=f"retrieval-{task}",
        provenance=Provenance(
            artifact=ArtifactRef(path="m.gguf", sha256="b" * 64,
                                 kind=artifact_kind),
            tokenizer=ArtifactRef(path="tok", sha256="0" * 64, kind="tokenizer"),
            seed=1, git_sha="deadbee", bpb_mode="not-applicable"),
        metrics=metrics, created_at="2026-08-25T00:00:00Z", item_count=1)
    return write_scorecard(path, card)


def _write_retrieval(root, arm_run, *, depths, artifact_kind="gguf-q4_0",
                     per_depth=100):
    for task in RETRIEVAL_GATE_TASKS:
        _retrieval_card(Path(root) / arm_run / f"retrieval-{task}.json",
                        task=task, depths=depths, artifact_kind=artifact_kind,
                        per_depth=per_depth)


def _scored(depths, *, items=100, artifact_kind="gguf-q4_0"):
    """The in-memory shape `read_retrieval` returns, for the pure-rule tests."""
    return {task: {"depths": {depth: {"exact_match": score, "n": items}
                              for depth, score in depths.items()},
                   "artifact_kind": artifact_kind}
            for task in RETRIEVAL_GATE_TASKS}


def _decode_report(path, *, models, file_mb=None, depths=(0, 2048)):
    """A decode_bench report with every model alternating inside each pass."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    passes = []
    for depth in depths:
        passes.append({
            "threads": 8, "n_gen": 128, "depth": depth, "rounds": 3,
            "models": {
                name: {"path": f"{name}.gguf",
                       "file_mb": (file_mb or {}).get(name),
                       "samples": [by_depth[depth]], "mean": by_depth[depth],
                       "stdev": None}
                for name, by_depth in models.items()}})
    path.write_text(json.dumps({"passes": passes, "note": None}))
    return path


def _gate_rows():
    """Control first, then one arm that clears BPB and cache on its own.

    The arm wins both measurable-for-free columns deliberately: these tests are
    about what happens to a shape that looks good on the evidence that exists,
    which is the only shape a missing column can mislead anyone about."""
    return _with_deltas([_row("a8-kv4", bpb=1.20, kv=4096, is_control=True),
                         _row("a4-kv1", bpb=1.19, kv=1024)])


# ------------------------------------------------- an unmeasured column ------

def test_a_strong_bpb_row_alone_is_not_a_recommendation():
    """The failure this gate exists for. An arm that clears the only column
    anyone measured must come out `unproven`, naming what is missing, rather
    than `recommended`."""
    control, arm = _gate_rows()

    entry = gate_arm(arm, control, retrieval={}, control_retrieval={},
                     decode_passes={}, arm_names=("a4-kv1",),
                     control_names=("a8-kv4",))

    assert entry["verdict"] == "unproven"
    assert entry["checks"]["bpb"]["status"] == "pass"
    assert set(entry["unproven"]) == {"retrieval", "export", "decode"}


def test_every_gate_column_is_required():
    """No column is optional. A rule satisfied by a majority would recommend a
    shape nobody had measured for export."""
    assert set(GATE_COLUMNS) == {"bpb", "retrieval", "kv", "export", "decode"}
    checks = {name: {"status": "pass"} for name in GATE_COLUMNS}
    assert gate_verdict(checks)["verdict"] == "recommended"
    for name in GATE_COLUMNS:
        partial = dict(checks, **{name: {"status": "unmeasured"}})
        assert gate_verdict(partial)["verdict"] == "unproven", name


def test_a_powerless_column_is_not_a_pass():
    """Measured-without-power is its own outcome: the instrument could not have
    detected the failure the column screens for."""
    checks = {name: {"status": "pass"} for name in GATE_COLUMNS}
    checks["retrieval"] = {"status": "no-power"}

    assert gate_verdict(checks)["verdict"] == "unproven"


def test_a_measured_failure_outranks_an_unmeasured_column():
    """`blocked` and `unproven` lead to different next actions -- closing a
    candidate versus running an evaluation -- so a failure must not be softened
    into merely unproven."""
    checks = {name: {"status": "pass"} for name in GATE_COLUMNS}
    checks["kv"] = {"status": "fail"}
    checks["decode"] = {"status": "unmeasured"}

    verdict = gate_verdict(checks)

    assert verdict["verdict"] == "blocked"
    assert verdict["failed"] == ["kv"]
    assert verdict["unproven"] == ["decode"]


# ------------------------------------------------------------- retrieval ----

def test_the_item_floor_is_derived_from_the_threshold():
    """Written as arithmetic on the threshold so the two cannot drift: one item
    must be worth no more than the gate it is asked to resolve."""
    assert RETRIEVAL_MIN_ITEMS_PER_DEPTH == \
        math.ceil(100.0 / RETRIEVAL_MAX_DROP_POINTS)
    assert 100.0 / RETRIEVAL_MIN_ITEMS_PER_DEPTH <= RETRIEVAL_MAX_DROP_POINTS


def test_a_depth_with_too_few_items_has_no_power():
    """With 20 items one item is 5 points against a 2-point gate, so the
    threshold is finer than the instrument and every arm passes or fails by a
    single item."""
    check = retrieval_check(_scored({2048: 0.50}, items=20),
                            _scored({2048: 0.55}, items=20))

    assert check["status"] == "no-power"
    cell = next(c for c in check["cells"] if c["depth"] == 2048)
    assert cell["status"] == "no-power"
    assert str(RETRIEVAL_MIN_ITEMS_PER_DEPTH) in cell["note"]


def test_a_control_at_the_floor_cannot_host_a_two_point_drop():
    """You cannot fall two points from one. Such a depth passes every arm,
    including one that emits nothing, so it is reported as powerless."""
    check = retrieval_check(_scored({2048: 0.0}), _scored({2048: 0.01}))

    assert check["status"] == "no-power"


def test_retrieval_fails_on_a_drop_past_the_gate_at_a_single_depth():
    """'No worse at any trained depth' is read at the finest grain the
    artifacts support: a strong shallow curve must not cover a deep regression."""
    arm = _scored({256: 0.90, 2048: 0.50})
    control = _scored({256: 0.88, 2048: 0.60})

    check = retrieval_check(arm, control)

    assert check["status"] == "fail"
    assert "2048" in check["note"]


def test_retrieval_is_evaluated_per_task_not_pooled():
    """Pooling would need a weighting rule this phase never preregistered, and
    would let a strong passkey curve cover an mqar regression."""
    arm = _scored({2048: 0.80})
    arm["mqar"]["depths"][2048]["exact_match"] = 0.50
    control = _scored({2048: 0.60})

    check = retrieval_check(arm, control)

    assert check["status"] == "fail"
    assert "mqar" in check["note"]


def test_retrieval_passes_when_every_cell_is_within_the_gate():
    depths = {256: 0.90, 512: 0.85, 1024: 0.80, 2048: 0.75}
    arm = _scored({depth: score - 0.01 for depth, score in depths.items()})

    check = retrieval_check(arm, _scored(depths))

    assert check["status"] == "pass"
    assert len(check["cells"]) == len(depths) * len(RETRIEVAL_GATE_TASKS)


def test_a_depth_the_arm_never_ran_is_unmeasured_rather_than_dropped():
    """Silently intersecting the depths would let an arm skip the deep end and
    still read as having held retention everywhere."""
    arm = _scored({256: 0.90})
    control = _scored({256: 0.90, 2048: 0.70})

    check = retrieval_check(arm, control)

    assert check["status"] == "unmeasured"
    assert any(cell["depth"] == 2048 and cell["status"] == "unmeasured"
               for cell in check["cells"])


# ---------------------------------------------------------------- export ----

def test_a_pytorch_score_is_not_export_evidence():
    """Scoring a checkpoint in PyTorch demonstrates nothing about whether stock
    llama.cpp can convert or load the shape, which is what the plan gates on."""
    check = export_check(_scored({2048: 0.8}, artifact_kind="checkpoint"))

    assert check["status"] == "unmeasured"
    assert "llama.cpp" in check["note"]


def test_a_gguf_backed_score_is_export_evidence():
    """A retrieval card whose artifact is a GGUF exists only because llama.cpp
    loaded that file and generated from it -- evidence, not a claim."""
    assert export_check(_scored({2048: 0.8}))["status"] == "pass"


# ---------------------------------------------------------------- decode ----

def test_decode_refuses_two_models_measured_in_separate_passes(tmp_path):
    """decode_bench alternates models within a pass precisely because absolutes
    move with box load. Reading across passes would manufacture a speedup that
    is a report about the box."""
    report = tmp_path / "decode.json"
    report.write_text(json.dumps({"passes": [
        {"threads": 8, "n_gen": 128, "depth": 0, "rounds": 3,
         "models": {"a4-kv1": {"path": "a.gguf", "file_mb": 60.0,
                               "samples": [30.0], "mean": 30.0,
                               "stdev": None}}},
        {"threads": 8, "n_gen": 128, "depth": 0, "rounds": 3,
         "models": {"a8-kv4": {"path": "c.gguf", "file_mb": 60.0,
                               "samples": [20.0], "mean": 20.0,
                               "stdev": None}}},
    ]}))
    control, arm = _gate_rows()

    check = decode_check(arm, control, read_decode_passes(report),
                         ("a4-kv1",), ("a8-kv4",))

    assert check["status"] == "unmeasured"
    assert "comparable" in check["note"]


def test_decode_fails_an_arm_slower_at_depth_zero(tmp_path):
    """The plan names depth zero because it is where a conv hybrid has least to
    gain: an arm slower on an empty context is slower for most chat turns no
    matter what its cache costs."""
    report = _decode_report(tmp_path / "decode.json", models={
        "a4-kv1": {0: 18.0, 2048: 19.0}, "a8-kv4": {0: 20.0, 2048: 15.0}})
    control, arm = _gate_rows()

    check = decode_check(arm, control, read_decode_passes(report),
                         ("a4-kv1",), ("a8-kv4",))

    assert check["status"] == "fail"
    assert check["depths"][0]["delta_pct"] == pytest.approx(-10.0)


def test_decode_reports_the_long_context_advantage_without_gating_on_it(tmp_path):
    """'Does not erase the benefit' is not 'must be faster', so the ratio's
    movement with depth is reported as the phase's finding rather than turned
    into a bound the plan never set."""
    report = _decode_report(tmp_path / "decode.json", models={
        "a4-kv1": {0: 20.0, 2048: 18.0}, "a8-kv4": {0: 20.0, 2048: 12.0}})
    control, arm = _gate_rows()

    check = decode_check(arm, control, read_decode_passes(report),
                         ("a4-kv1",), ("a8-kv4",))

    assert check["status"] == "pass"
    assert check["long_context_advantage_pct"] == pytest.approx(50.0)


def test_decode_is_found_under_the_run_name_as_well_as_the_grid_point(tmp_path):
    """The names in a decode report are whatever the operator typed at
    `--models`; rejecting a real measurement over a naming convention would
    throw the measurement away."""
    report = _decode_report(tmp_path / "decode.json", models={
        "arch-stagea-a4-kv1": {0: 20.0, 2048: 20.0},
        "arch-stagea-a8-kv4": {0: 20.0, 2048: 20.0}})
    control, arm = _gate_rows()

    check = decode_check(arm, control, read_decode_passes(report),
                         ("a4-kv1", "arch-stagea-a4-kv1"),
                         ("a8-kv4", "arch-stagea-a8-kv4"))

    assert check["status"] == "pass"


def test_a_decode_row_that_never_produced_a_number_is_not_a_measurement(tmp_path):
    """`decode_bench` writes `mean: null` when every round of a model failed or
    timed out. Reading that as present would compare an arm against nothing."""
    report = tmp_path / "decode.json"
    report.write_text(json.dumps({"passes": [
        {"threads": 8, "n_gen": 128, "depth": depth, "rounds": 3,
         "models": {"a4-kv1": {"path": "a.gguf", "file_mb": 60.0,
                               "samples": [], "mean": None, "stdev": None},
                    "a8-kv4": {"path": "c.gguf", "file_mb": 60.0,
                               "samples": [20.0], "mean": 20.0,
                               "stdev": None}}}
        for depth in (0, 2048)]}))
    control, arm = _gate_rows()

    check = decode_check(arm, control, read_decode_passes(report),
                         ("a4-kv1",), ("a8-kv4",))

    assert check["status"] == "unmeasured"


def test_an_oversized_artifact_erases_the_benefit(tmp_path):
    """Q4_0 bytes track parameters exactly, so a shape that buys its cache
    saving with a materially larger file has not bought anything."""
    report = _decode_report(
        tmp_path / "decode.json",
        models={"a4-kv1": {0: 20.0, 2048: 20.0},
                "a8-kv4": {0: 20.0, 2048: 20.0}},
        file_mb={"a4-kv1": 70.0, "a8-kv4": 60.0})
    control, arm = _gate_rows()

    check = decode_check(arm, control, read_decode_passes(report),
                         ("a4-kv1",), ("a8-kv4",))

    assert check["status"] == "fail"
    assert check["artifact_source"] == "measured"
    assert check["artifact_growth_pct"] == pytest.approx(100.0 * 10 / 60)


def test_the_artifact_column_says_whether_it_was_measured_or_computed():
    """A size column that silently switches between a file on disk and a
    parameter count is worse than one that has only ever been arithmetic."""
    control, arm = _gate_rows()

    check = decode_check(arm, control, {}, ("a4-kv1",), ("a8-kv4",))

    assert check["artifact_source"] == "analytic"
    assert check["artifact_MB"] == arm["q4_0_MB"]


# -------------------------------------------------------------------- kv ----

def test_kv_is_an_absolute_ceiling_not_a_delta():
    """The ceiling is what a deployment can afford, so a proxy control at a
    different depth can be over it while remaining the reference every other
    column is read against."""
    check = kv_check(_row("deep", bpb=1.2, kv=8192))

    assert check["status"] == "fail"
    assert check["at_or_under_preferred"] is False
    assert kv_check(_row("cheap", bpb=1.2, kv=4096))["at_or_under_preferred"]


# --------------------------------------------------------- the deliverable ---

def test_the_deliverable_excludes_an_unproven_arm_from_the_pareto_set(tmp_path):
    """A shape reaches the set only by clearing every column. An arm with a
    cheaper cache and better BPB that nobody measured must not displace one
    that was measured."""
    rows = _with_deltas([_row("a8-kv4", bpb=1.20, kv=4096, is_control=True),
                         _row("a4-kv1", bpb=1.19, kv=1024),
                         _row("a2-kv1", bpb=1.18, kv=512)])
    arms = [ARMS_BY_NAME[name] for name in ("a8-kv4", "a4-kv1", "a2-kv1")]
    for name in ("a8-kv4", "a4-kv1"):               # a2-kv1 goes unmeasured
        _write_retrieval(tmp_path / "retrieval", f"arch-stagea-{name}",
                         depths={256: 0.90, 2048: 0.80})
    report = _decode_report(
        tmp_path / "decode.json",
        models={name: {0: 20.0, 2048: 20.0}
                for name in ("a8-kv4", "a4-kv1", "a2-kv1")},
        file_mb={name: 60.0 for name in ("a8-kv4", "a4-kv1", "a2-kv1")})

    built = build_recommendation(rows, arms,
                                 retrieval_root=str(tmp_path / "retrieval"),
                                 decode_report=str(report))

    assert built["pareto_set"] == ["a4-kv1"]
    assert built["unproven"] == ["a2-kv1"]
    assert built["verdict"] == "recommend"


def test_no_recommendation_is_a_statement_about_the_evidence(tmp_path):
    """With nothing measured beyond BPB the phase recommends nothing, and says
    that this is about the evidence rather than about the shapes."""
    arms = [ARMS_BY_NAME["a8-kv4"], ARMS_BY_NAME["a4-kv1"]]

    built = build_recommendation(_gate_rows(), arms,
                                 retrieval_root=str(tmp_path / "missing"),
                                 decode_report=str(tmp_path / "missing.json"))

    assert built["pareto_set"] == []
    assert built["verdict"] == "no-recommendation"
    assert "evidence" in built["note"]


def test_the_markdown_names_every_column_for_every_arm(tmp_path):
    """A reader must see which column stopped an arm without opening the JSON,
    and must be told that a blank cell is not a pass."""
    arms = [ARMS_BY_NAME["a8-kv4"], ARMS_BY_NAME["a4-kv1"]]
    built = build_recommendation(_gate_rows(), arms,
                                 retrieval_root=str(tmp_path / "missing"),
                                 decode_report=str(tmp_path / "missing.json"))

    markdown = render_recommendation_markdown(built)

    for column in GATE_COLUMNS:
        assert f"- {column}: " in markdown
    assert "Neither is a pass" in markdown
    assert "Apple Silicon decode is pending" in markdown


def test_the_recommend_subcommand_writes_both_artifacts(tmp_path):
    """End to end through the CLI the controller launches, because a rule that
    is only ever called from a test is a rule the phase cannot run."""
    cards = tmp_path / "cards"
    for arm, bpb in (("a8-kv4", 1.20), ("a4-kv1", 1.19)):
        _write_card(cards / f"arch-stagea-{arm}-bpb.json",
                    name=f"arch-stagea-{arm}-bpb", bpb=bpb,
                    config=ARMS_BY_NAME[arm].config)
    for arm in ("a8-kv4", "a4-kv1"):
        _write_retrieval(tmp_path / "retrieval", f"arch-stagea-{arm}",
                         depths={256: 0.90, 2048: 0.80})
    decode = _decode_report(
        tmp_path / "decode.json",
        models={arm: {0: 20.0, 2048: 20.0} for arm in ("a8-kv4", "a4-kv1")},
        file_mb={arm: 60.0 for arm in ("a8-kv4", "a4-kv1")})

    assert architecture_report_main([
        "--report-root", str(tmp_path / "out"),
        "--scorecard-root", str(cards),
        "recommend",
        "--retrieval-root", str(tmp_path / "retrieval"),
        "--decode", str(decode)]) == 0

    report = json.loads(
        (tmp_path / "out" / "stagea-recommendation.json").read_text())
    assert report["pareto_set"] == ["a4-kv1"]
    assert report["gate"]["columns"] == list(GATE_COLUMNS)
    assert "recommendation gate" in \
        (tmp_path / "out" / "stagea-recommendation.md").read_text()
