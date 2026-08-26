"""Tests for phase 8's 250M-token probe gate.

The wording this gate implements -- "no arm improves code BPB by >=2% or moves
execution/syntax signal" -- has two readings that spend a 1B-token budget
differently, and an undefined "moves" that one item out of 378 would satisfy.
These tests pin the reading and the movement thresholds, so that a later change
to either is a visible change to a preregistered quantity rather than a quiet
one. The two cases that matter most are the ones that separate the readings:
an arm with BPB movement and no execution movement, and an arm with execution
movement and no BPB movement. Both continue.
"""

import pytest

from daedalus.code_gates import (
    CODE_BPB_IMPROVEMENT_PCT,
    EXECUTION_MOVE_POINTS,
    PASS_AT_1_MOVE_POINTS,
    SYNTAX_VALID_MOVE_POINTS,
    ExecutionScore,
    ProbeGateError,
    ProbeScore,
    arm_verdict,
    code_bpb_improvement_pct,
    execution_moves,
    execution_score,
    probes_250m_verdict,
)
from daedalus.scorecard import ArtifactRef, Provenance, Scorecard

ZERO_SHA = "0" * 64


# The untouched base as `runs/eval/code-base` measured it: HumanEval+ at zero
# with perfect syntax validity, MBPP+ at 3 items of 378 with 0.238 validity.
BASE = ProbeScore(
    name="base",
    code_bpb=1.2000,
    execution={
        "humaneval-plus": ExecutionScore(pass_at_1=0.0, pass_at_1_plus=0.0,
                                         syntax_valid=1.0, n=164),
        "mbpp-plus": ExecutionScore(pass_at_1=0.0079, pass_at_1_plus=0.0053,
                                    syntax_valid=0.2381, n=378),
    },
)


def _arm(name, *, code_bpb=1.2000, mbpp_syntax=0.2381, mbpp_pass=0.0079,
         humaneval_pass=0.0) -> ProbeScore:
    return ProbeScore(
        name=name,
        code_bpb=code_bpb,
        execution={
            "humaneval-plus": ExecutionScore(pass_at_1=humaneval_pass,
                                             pass_at_1_plus=humaneval_pass,
                                             syntax_valid=1.0, n=164),
            "mbpp-plus": ExecutionScore(pass_at_1=mbpp_pass,
                                        pass_at_1_plus=0.0053,
                                        syntax_valid=mbpp_syntax, n=378),
        },
    )


def _scorecard(**overrides) -> Scorecard:
    fields = {
        "kind": "code-execution",
        "name": "mbpp-plus",
        "provenance": Provenance(
            artifact=ArtifactRef(path="final/hero/checkpoint.pt",
                                 sha256=ZERO_SHA, kind="checkpoint"),
            tokenizer=ArtifactRef(path="<embedded>", sha256=ZERO_SHA,
                                  kind="tokenizer"),
            seed=20260824, git_sha="abc1234"),
        "metrics": {"pass@1": 0.0079, "pass@1_plus": 0.0053,
                    "syntax_valid": 0.2381, "n": 378.0},
        "item_count": 378,
        "created_at": "2026-08-26T15:29:45Z",
    }
    fields.update(overrides)
    return Scorecard(**fields)


# ---------------------------------------------------------- criterion A ---

def test_code_bpb_improvement_is_a_relative_reduction():
    assert code_bpb_improvement_pct(BASE, _arm("a", code_bpb=1.14)) == \
        pytest.approx(5.0)


def test_a_worse_arm_reports_a_negative_improvement_rather_than_zero():
    # Clamping to zero would make "no movement" and "actively worse" the same
    # entry in the record, and the second is the one worth seeing.
    assert code_bpb_improvement_pct(BASE, _arm("a", code_bpb=1.26)) < 0


def test_a_non_finite_measurement_is_refused_not_scored():
    with pytest.raises(ProbeGateError, match="non-finite"):
        code_bpb_improvement_pct(BASE, _arm("a", code_bpb=float("nan")))


def test_a_zero_base_bpb_is_refused_because_the_ratio_is_undefined():
    base = ProbeScore(name="base", code_bpb=0.0, execution=BASE.execution)
    with pytest.raises(ProbeGateError, match="undefined"):
        code_bpb_improvement_pct(base, _arm("a"))


# ---------------------------------------------------------- criterion B ---

def test_syntax_validity_moves_only_at_the_preregistered_bar():
    just_under = _arm("a", mbpp_syntax=0.2381 + (SYNTAX_VALID_MOVE_POINTS - 0.01) / 100)
    just_over = _arm("b", mbpp_syntax=0.2381 + SYNTAX_VALID_MOVE_POINTS / 100)
    assert execution_moves(BASE, just_under)["mbpp-plus"]["moved"] is False
    assert execution_moves(BASE, just_over)["mbpp-plus"]["moved"] is True


def test_an_arm_exactly_on_the_bpb_bar_clears_it():
    # 1.2000 -> 1.1760 is exactly 2%, and float subtraction puts it a few parts
    # in 1e16 under. "At least 2%" has to include 2%.
    exact = _arm("a", code_bpb=BASE.code_bpb * (1 - CODE_BPB_IMPROVEMENT_PCT / 100))
    assert arm_verdict(BASE, exact)["criterion_a_code_bpb"] is True


def test_one_extra_solved_item_does_not_move_the_signal():
    # 1/378 is 0.26 points against a 1.0-point bar. An unbounded "moves" would
    # authorise a 1B-token run on it.
    one_more = _arm("a", mbpp_pass=0.0079 + 1 / 378)
    assert execution_moves(BASE, one_more)["mbpp-plus"]["moved"] is False


def test_a_regression_is_recorded_but_never_counts_as_movement():
    worse = _arm("a", mbpp_syntax=0.10)
    entry = execution_moves(BASE, worse)["mbpp-plus"]["metrics"]["syntax_valid"]
    assert entry["regressed"] is True
    assert entry["moved"] is False


def test_the_stricter_pass_at_1_plus_counts_as_movement_on_its_own():
    arm = ProbeScore(
        name="a", code_bpb=1.2000,
        execution={"mbpp-plus": ExecutionScore(
            pass_at_1=0.0079, pass_at_1_plus=0.0053 + PASS_AT_1_MOVE_POINTS / 100,
            syntax_valid=0.2381, n=378)})
    base = ProbeScore(name="base", code_bpb=1.2000,
                      execution={"mbpp-plus": BASE.execution["mbpp-plus"]})
    assert execution_moves(base, arm)["mbpp-plus"]["moved"] is True


def test_an_arm_scored_on_different_benchmarks_is_refused():
    arm = ProbeScore(name="a", code_bpb=1.1,
                     execution={"mbpp-plus": BASE.execution["mbpp-plus"]})
    with pytest.raises(ProbeGateError, match="same benchmarks"):
        execution_moves(BASE, arm)


def test_a_different_item_count_is_refused_rather_than_compared():
    # A --task-limit on one side shrinks the denominator, which reads as a gain.
    arm = ProbeScore(
        name="a", code_bpb=1.1,
        execution={**dict(BASE.execution),
                   "mbpp-plus": ExecutionScore(pass_at_1=0.05, pass_at_1_plus=0.04,
                                               syntax_valid=0.5, n=100)})
    with pytest.raises(ProbeGateError, match="not comparable"):
        execution_moves(BASE, arm)


# ------------------------------------------------------- the two readings ---

def test_bpb_alone_qualifies_an_arm():
    verdict = arm_verdict(BASE, _arm("lr5e-4", code_bpb=1.14))
    assert verdict["criterion_a_code_bpb"] is True
    assert verdict["criterion_b_execution"] is False
    assert verdict["qualifies"] is True


def test_execution_movement_alone_qualifies_an_arm():
    verdict = arm_verdict(BASE, _arm("lr1e-3", mbpp_syntax=0.30))
    assert verdict["criterion_a_code_bpb"] is False
    assert verdict["criterion_b_execution"] is True
    assert verdict["qualifies"] is True


def test_one_qualifying_arm_continues_the_run():
    # The case that separates the two readings: no arm clears BPB, one moves
    # syntax validity. The loose reading continues; the strict one would stop.
    verdict = probes_250m_verdict(
        BASE, [_arm("lr5e-4"), _arm("lr1e-3", mbpp_syntax=0.30), _arm("lr2e-3")])
    assert verdict["continue"] is True
    assert verdict["qualifying"] == ["lr1e-3"]


def test_a_strong_bpb_arm_continues_without_any_execution_movement():
    # The other separating case, and the one the strict reading gets wrong:
    # 5% off code BPB with Python syntax validity unmoved.
    verdict = probes_250m_verdict(
        BASE, [_arm("lr5e-4", code_bpb=1.14), _arm("lr1e-3"), _arm("lr2e-3")])
    assert verdict["continue"] is True
    assert verdict["qualifying"] == ["lr5e-4"]


def test_the_gate_can_return_no():
    verdict = probes_250m_verdict(
        BASE, [_arm("lr5e-4", code_bpb=1.19),
               _arm("lr1e-3", mbpp_syntax=0.24),
               _arm("lr2e-3", code_bpb=1.25)])
    assert verdict["continue"] is False
    assert verdict["qualifying"] == []
    assert verdict["best_before_retention"] is None
    assert f"{CODE_BPB_IMPROVEMENT_PCT:g}%" in verdict["reason"]


def test_every_arm_is_reported_whether_it_qualified_or_not():
    verdict = probes_250m_verdict(
        BASE, [_arm("lr5e-4", code_bpb=1.14), _arm("lr1e-3"), _arm("lr2e-3")])
    assert [entry["arm"] for entry in verdict["arms"]] == \
        ["lr5e-4", "lr1e-3", "lr2e-3"]


def test_the_recorded_thresholds_are_the_ones_that_were_applied():
    verdict = probes_250m_verdict(BASE, [_arm("lr5e-4", code_bpb=1.14)])
    assert verdict["thresholds"] == {
        "code_bpb_improvement_pct": CODE_BPB_IMPROVEMENT_PCT,
        "execution_move_points": dict(EXECUTION_MOVE_POINTS)}


# ------------------------------------------------------------- selection ---

def test_qualifying_arms_are_ranked_by_code_bpb_lowest_first():
    verdict = probes_250m_verdict(
        BASE, [_arm("lr5e-4", code_bpb=1.16), _arm("lr1e-3", code_bpb=1.10),
               _arm("lr2e-3", code_bpb=1.13)])
    assert verdict["ranking"] == ["lr1e-3", "lr2e-3", "lr5e-4"]
    assert verdict["best_before_retention"] == "lr1e-3"


def test_pass_at_1_breaks_a_code_bpb_tie():
    verdict = probes_250m_verdict(
        BASE, [_arm("lr5e-4", code_bpb=1.10),
               _arm("lr1e-3", code_bpb=1.10, humaneval_pass=0.05)])
    assert verdict["ranking"] == ["lr1e-3", "lr5e-4"]


def test_a_non_qualifying_arm_is_never_ranked():
    verdict = probes_250m_verdict(
        BASE, [_arm("lr5e-4", code_bpb=1.19), _arm("lr1e-3", code_bpb=1.14)])
    assert verdict["ranking"] == ["lr1e-3"]


# ------------------------------------------------------------- refusals ---

def test_no_arms_is_refused():
    with pytest.raises(ProbeGateError, match="nothing to decide"):
        probes_250m_verdict(BASE, [])


def test_two_arms_under_one_name_are_refused():
    with pytest.raises(ProbeGateError, match="uniquely named"):
        probes_250m_verdict(BASE, [_arm("lr1e-3"), _arm("lr1e-3", code_bpb=1.1)])


# ------------------------------------------------------ scorecard reading ---

def test_an_execution_scorecard_reads_into_the_gate_shape():
    score = execution_score(_scorecard())
    assert (score.pass_at_1, score.syntax_valid, score.n) == (0.0079, 0.2381, 378)


def test_a_scorecard_of_another_kind_is_refused():
    with pytest.raises(ProbeGateError, match="not 'code-execution'"):
        execution_score(_scorecard(kind="bpb", metrics={"bpb": 1.2}))


def test_a_scorecard_missing_an_execution_metric_is_refused():
    with pytest.raises(ProbeGateError, match="syntax_valid"):
        execution_score(_scorecard(metrics={"pass@1": 0.0, "pass@1_plus": 0.0}))
