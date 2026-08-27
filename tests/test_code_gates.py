"""Tests for phase 8's two escalation gates.

The 250M wording -- "no arm improves code BPB by >=2% or moves execution/syntax
signal" -- has two readings that spend a 1B-token budget differently, and an
undefined "moves" that one item out of 378 would satisfy. The 1B wording --
"general BPB regression <=1.5%, five-task mean drop <=1 point, retrieval drop
<=2 points at every depth, code metrics improve" -- carries three exact numbers
and one clause with none at all.

These tests pin both readings and every threshold, so that a later change to any
of them is a visible change to a preregistered quantity rather than a quiet one.
The cases that matter most are the ones that separate the readings: at 250M, an
arm with BPB movement and no execution movement and an arm with the reverse,
both of which continue; at 1B, the same reverse case, which does *not*.
"""

import pytest

from daedalus.code_gates import (
    BRANCH_CODE_BPB_IMPROVEMENT_PCT,
    BRANCH_FIVE_TASK_DROP_POINTS_MAX,
    BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX,
    BRANCH_RETRIEVAL_DROP_POINTS_MAX,
    CODE_BPB_IMPROVEMENT_PCT,
    EXECUTION_MOVE_POINTS,
    PASS_AT_1_MOVE_POINTS,
    SYNTAX_VALID_MOVE_POINTS,
    BranchScore,
    ExecutionScore,
    ProbeGateError,
    ProbeScore,
    arm_verdict,
    branch_1b_verdict,
    code_bpb_improvement_pct,
    execution_moves,
    execution_regressions,
    execution_score,
    general_bpb_regression_pct,
    probes_250m_verdict,
    retrieval_drops,
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


# ========================================================= the 1B branch ===
#
# The untouched base's general-side numbers, in the shape
# `qat_recovery.collect_observation` already produces: a five-task mean in
# points and retrieval keyed `<task>:d<depth>` as a fraction.

BRANCH_BASE = BranchScore(
    name="base",
    code_bpb=1.2000,
    execution=BASE.execution,
    general_bpb=0.9000,
    five_task_mean=47.374,
    retrieval={"retrieval-passkey:d256": 0.83, "retrieval-passkey:d512": 0.81,
               "retrieval-mqar:d1024": 0.91, "retrieval-mqar:d2048": 0.86},
)


def _branch(*, code_bpb=1.14, general_bpb=0.9000, five_task_mean=47.374,
            retrieval=None, mbpp_syntax=0.2381, mbpp_pass=0.0079,
            humaneval_pass=0.0, name="code-branch-1b") -> BranchScore:
    """A branch that passes every clause, unless the caller breaks one."""

    arm = _arm(name, code_bpb=code_bpb, mbpp_syntax=mbpp_syntax,
               mbpp_pass=mbpp_pass, humaneval_pass=humaneval_pass)
    return BranchScore(
        name=name, code_bpb=arm.code_bpb, execution=arm.execution,
        general_bpb=general_bpb, five_task_mean=five_task_mean,
        retrieval=dict(BRANCH_BASE.retrieval if retrieval is None else retrieval),
    )


def _check(verdict, gate) -> dict:
    return next(entry for entry in verdict["checks"] if entry["gate"] == gate)


def test_a_branch_that_clears_every_clause_continues():
    verdict = branch_1b_verdict(BRANCH_BASE, _branch())
    assert verdict["continue"] is True
    assert verdict["failed"] == []
    assert verdict["gate"] == "branch_1b"


# ----------------------------------------- "code metrics improve", pinned ---

def test_code_bpb_is_required_and_execution_movement_cannot_substitute():
    # The case that separates this gate's reading from the 250M one. At 250M an
    # arm that moved MBPP+ syntax validity with no BPB movement qualifies; here
    # the same numbers stop the branch, because a `continue_if` clause is a
    # requirement and a further 2B tokens is what it authorises.
    verdict = branch_1b_verdict(BRANCH_BASE,
                                _branch(code_bpb=1.2000, mbpp_syntax=0.30))
    assert verdict["continue"] is False
    assert verdict["failed"] == ["code-bpb"]
    assert _check(verdict, "code-execution-regression")["passed"] is True
    # Still reported -- it is real evidence about the run, just not authority.
    assert _check(verdict, "code-execution-regression")["moved"] == ["mbpp-plus"]


def test_the_same_arm_qualifies_at_250m_and_is_stopped_at_1b():
    # Both readings applied to one set of numbers, so the divergence is a
    # property this file asserts rather than one a reader has to infer.
    arm = _arm("lr1e-3", code_bpb=1.2000, mbpp_syntax=0.30)
    assert arm_verdict(BASE, arm)["qualifies"] is True
    assert branch_1b_verdict(
        BRANCH_BASE, _branch(code_bpb=1.2000, mbpp_syntax=0.30))["continue"] is False


def test_the_bpb_bar_is_the_250m_stage_bar_and_not_the_final_one():
    # Borrowed, not invented. If this ever reads 5.0 it has silently become the
    # `final` gate and the plan's staged 1B -> 2B -> SFT design is gone.
    assert BRANCH_CODE_BPB_IMPROVEMENT_PCT == CODE_BPB_IMPROVEMENT_PCT == 2.0


def test_a_branch_exactly_on_the_bpb_bar_continues():
    exact = _branch(code_bpb=BRANCH_BASE.code_bpb
                    * (1 - BRANCH_CODE_BPB_IMPROVEMENT_PCT / 100))
    assert _check(branch_1b_verdict(BRANCH_BASE, exact), "code-bpb")["passed"] \
        is True


def test_execution_regression_past_its_own_bar_stops_the_branch():
    # A branch unlearning Python while its code BPB improves. The BPB clause
    # alone would continue it.
    collapsed = _branch(code_bpb=1.10, mbpp_syntax=0.2381 - 0.05)
    verdict = branch_1b_verdict(BRANCH_BASE, collapsed)
    assert verdict["continue"] is False
    assert verdict["failed"] == ["code-execution-regression"]
    fallen = _check(verdict, "code-execution-regression")["regressed"]
    assert [(entry["benchmark"], entry["metric"]) for entry in fallen] == \
        [("mbpp-plus", "syntax_valid")]


def test_a_regression_inside_the_bar_is_not_a_failure():
    # Symmetric with movement: a rise of less than two points is not movement,
    # so a fall of less than two points is not a regression this gate acts on.
    inside = _branch(code_bpb=1.10,
                     mbpp_syntax=0.2381 - (SYNTAX_VALID_MOVE_POINTS - 0.01) / 100)
    assert _check(branch_1b_verdict(BRANCH_BASE, inside),
                  "code-execution-regression")["passed"] is True


def test_execution_regressions_uses_the_same_thresholds_as_movement():
    fallen = execution_regressions(
        execution_moves(BASE, _arm("a", mbpp_pass=0.0079 - 1 / 378)))
    # 1/378 is 0.26 points against a 1.0-point bar, in the other direction.
    assert fallen == []


# ------------------------------------------------------- retention halves ---

def test_general_bpb_regression_is_positive_when_the_branch_is_worse():
    worse = _branch(general_bpb=0.9090)
    assert general_bpb_regression_pct(BRANCH_BASE, worse) == pytest.approx(1.0)


def test_general_bpb_regression_past_the_limit_stops_the_branch():
    verdict = branch_1b_verdict(BRANCH_BASE, _branch(general_bpb=0.9200))
    assert verdict["failed"] == ["general-bpb"]
    check = _check(verdict, "general-bpb")
    assert check["observed_regression_pct"] == pytest.approx(2.2222, abs=1e-3)
    assert check["limit_regression_pct"] == BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX


def test_a_branch_exactly_on_the_general_bpb_limit_continues():
    exact = _branch(general_bpb=BRANCH_BASE.general_bpb
                    * (1 + BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX / 100))
    assert _check(branch_1b_verdict(BRANCH_BASE, exact),
                  "general-bpb")["passed"] is True


def test_an_unmeasured_general_bpb_fails_rather_than_passing_silently():
    # The failure a retention gate exists to catch: the evaluation did not run.
    verdict = branch_1b_verdict(BRANCH_BASE, _branch(general_bpb=None))
    assert verdict["continue"] is False
    assert "not taken" in _check(verdict, "general-bpb")["reason"]


def test_a_non_finite_general_bpb_is_a_broken_run_not_a_regression():
    with pytest.raises(ProbeGateError, match="non-finite"):
        general_bpb_regression_pct(BRANCH_BASE,
                                   _branch(general_bpb=float("nan")))


def test_a_five_task_drop_past_one_point_stops_the_branch():
    verdict = branch_1b_verdict(BRANCH_BASE, _branch(five_task_mean=46.20))
    assert verdict["failed"] == ["five-task-mean"]
    assert _check(verdict, "five-task-mean")["observed_drop_points"] == \
        pytest.approx(1.174)


def test_a_branch_exactly_one_point_down_continues():
    exact = _branch(five_task_mean=BRANCH_BASE.five_task_mean
                    - BRANCH_FIVE_TASK_DROP_POINTS_MAX)
    assert _check(branch_1b_verdict(BRANCH_BASE, exact),
                  "five-task-mean")["passed"] is True


def test_an_unmeasured_five_task_mean_fails_rather_than_passing_silently():
    verdict = branch_1b_verdict(BRANCH_BASE, _branch(five_task_mean=None))
    assert verdict["continue"] is False
    assert "not measured" in _check(verdict, "five-task-mean")["reason"]


# -------------------------------------------------- retrieval, every depth ---

def test_retrieval_is_gated_per_depth_and_not_on_the_aggregate():
    # Loses 4 points at 2048, gains 4 at 256: the mean is flat and the gate
    # still says no, which is the whole reason it reads every depth.
    nets_out = dict(BRANCH_BASE.retrieval)
    nets_out["retrieval-mqar:d2048"] = 0.82
    nets_out["retrieval-passkey:d256"] = 0.87
    verdict = branch_1b_verdict(BRANCH_BASE, _branch(retrieval=nets_out))
    assert verdict["failed"] == ["retrieval"]
    check = _check(verdict, "retrieval")
    assert check["worst_key"] == "retrieval-mqar:d2048"
    assert check["worst_drop_points"] == pytest.approx(4.0)


def test_a_branch_exactly_two_points_down_at_one_depth_continues():
    exact = dict(BRANCH_BASE.retrieval)
    exact["retrieval-passkey:d512"] = 0.81 - BRANCH_RETRIEVAL_DROP_POINTS_MAX / 100
    assert _check(branch_1b_verdict(BRANCH_BASE, _branch(retrieval=exact)),
                  "retrieval")["passed"] is True


def test_a_depth_the_branch_never_measured_fails_the_gate():
    # "at every depth" must not quietly become "at every depth we measured":
    # dropping the deepest key and reporting the worst of the rest is exactly
    # how a long-context regression goes unseen.
    partial = {key: value for key, value in BRANCH_BASE.retrieval.items()
               if key != "retrieval-mqar:d2048"}
    verdict = branch_1b_verdict(BRANCH_BASE, _branch(retrieval=partial))
    assert verdict["failed"] == ["retrieval"]
    assert "retrieval-mqar:d2048" in _check(verdict, "retrieval")["reason"]


def test_a_branch_measured_at_extra_depths_is_not_penalised_for_them():
    # More coverage than the baseline is not a missing measurement.
    extra = dict(BRANCH_BASE.retrieval)
    extra["retrieval-passkey:d4096"] = 0.10
    assert _check(branch_1b_verdict(BRANCH_BASE, _branch(retrieval=extra)),
                  "retrieval")["passed"] is True


def test_no_retrieval_baseline_fails_rather_than_reporting_a_clean_sweep():
    base = BranchScore(name="base", code_bpb=BRANCH_BASE.code_bpb,
                       execution=BRANCH_BASE.execution,
                       general_bpb=BRANCH_BASE.general_bpb,
                       five_task_mean=BRANCH_BASE.five_task_mean)
    verdict = branch_1b_verdict(base, _branch())
    assert verdict["failed"] == ["retrieval"]
    assert "no retrieval baseline" in _check(verdict, "retrieval")["reason"]


def test_retrieval_drops_reports_every_key_it_compared():
    drops = retrieval_drops(BRANCH_BASE, _branch())
    assert sorted(drops["per_key"]) == sorted(BRANCH_BASE.retrieval)
    assert drops["missing"] == []


# ------------------------------------------------------------- reporting ---

def test_every_clause_is_reported_even_when_several_fail():
    verdict = branch_1b_verdict(
        BRANCH_BASE, _branch(code_bpb=1.2000, general_bpb=0.95,
                             five_task_mean=40.0))
    assert [entry["gate"] for entry in verdict["checks"]] == [
        "code-bpb", "code-execution-regression", "general-bpb",
        "five-task-mean", "retrieval"]
    assert verdict["failed"] == ["code-bpb", "general-bpb", "five-task-mean"]


def test_the_recorded_thresholds_are_the_ones_that_were_applied():
    verdict = branch_1b_verdict(BRANCH_BASE, _branch())
    assert verdict["thresholds"] == {
        "code_bpb_improvement_pct": BRANCH_CODE_BPB_IMPROVEMENT_PCT,
        "execution_move_points": dict(EXECUTION_MOVE_POINTS),
        "general_bpb_regression_pct_max": BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX,
        "five_task_drop_points_max": BRANCH_FIVE_TASK_DROP_POINTS_MAX,
        "retrieval_drop_points_max": BRANCH_RETRIEVAL_DROP_POINTS_MAX,
    }


def test_the_manifest_numbers_are_what_this_module_applies():
    # The three clauses that came with their own figures. A change here is a
    # change to a preregistered quantity and has to be visible.
    assert (BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX,
            BRANCH_FIVE_TASK_DROP_POINTS_MAX,
            BRANCH_RETRIEVAL_DROP_POINTS_MAX) == (1.5, 1.0, 2.0)


def test_a_branch_scored_on_different_benchmarks_is_refused_not_gated():
    # Non-comparable evidence raises, as it does at 250M; only an *absent*
    # measurement degrades to a failed clause.
    mismatched = BranchScore(
        name="code-branch-1b", code_bpb=1.10,
        execution={"mbpp-plus": BASE.execution["mbpp-plus"]},
        general_bpb=0.90, five_task_mean=47.374,
        retrieval=dict(BRANCH_BASE.retrieval))
    with pytest.raises(ProbeGateError, match="same benchmarks"):
        branch_1b_verdict(BRANCH_BASE, mismatched)
