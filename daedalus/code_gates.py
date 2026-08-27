"""Phase 8's escalation gates, as rules that can return no.

Two of them live here: `probes_250m_verdict`, which decides whether the three
250M-token probes justify a 1B-token branch, and `branch_1b_verdict`, which
decides whether that branch justifies the 2B extension and the post-training
that follows. Both are preregistered wordings with an ambiguity in them, and
both are disambiguated in this file *before* the run they gate has produced a
number -- the only time a threshold can be set without being an adjustment to a
result. The 250M reasoning is immediately below; the 1B reasoning sits above
`branch_1b_verdict`.

The preregistered wording is

    stop_if: no arm improves code BPB by >=2% or moves execution/syntax signal

and it has two readings that decide a 1B-token run differently:

  - **loose** (the natural English one): stop only when *every* arm fails
    *both* criteria, so one arm moving the execution/syntax signal alone is
    enough to continue even with no BPB movement;
  - **strict**: stop when no arm clears the BPB bar, *or* when no arm moves the
    execution signal, so continuing needs both.

This module implements the **loose** reading, and the reason is the same one
that put the execution/syntax signal in the gate at all. The code run manifest
records it: a 150M base scores 0.000 pass@1 on HumanEval+ and 0.008 on MBPP+,
so "a 250M-token probe gated on pass@1 alone would read every arm as identical
zero", and MBPP+ syntax validity at 0.238 "is the headroom to watch: it moves
before pass@1 does". The signal was added as the *more sensitive alternative*
to a BPB bar a short probe may not clear. Under the strict reading, adding a
more sensitive signal makes the gate harder to pass -- and would stop a run
whose code BPB improved 5% but which had not yet moved Python syntax validity.
That inverts the purpose of the addition, so it is not what the wording means.

The loose reading is easier to pass than a BPB-only gate would have been. That
is the cost, and it is why "moves execution/syntax signal" cannot stay
undefined: an unbounded "moves" is passed by one item out of 378, and a gate
that cannot return no is not a gate -- the same defect the DPO preference gate
was repaired for. So the movement thresholds below are named here, before any
arm has produced a number, which is the only time they can be set without being
an adjustment to a result.

What this module deliberately does *not* decide is the winner. The manifest's
`select_on` reads "code BPB and execution pass@1, subject to general
retention", and the retention bound for the *probe* stage is not preregistered
anywhere -- only the 1B branch gate's is. `probes_250m_verdict` therefore ranks
the qualifying arms and names `best_before_retention`; applying retention, and
choosing, stays with the caller that holds the general-side scorecards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from daedalus.scorecard import Scorecard, ScorecardError

#: Criterion A. Relative reduction in held-out code bits-per-byte against the
#: untouched base, in percent. Preregistered in the code run manifest.
CODE_BPB_IMPROVEMENT_PCT = 2.0

#: Criterion B, first half. Absolute percentage points of syntax validity. On
#: MBPP+ (n=378, base 0.238) two points is ~8 items; on HumanEval+ the base is
#: already 1.000, so this half can only be met by MBPP+ -- which is the
#: benchmark the manifest names as the one with headroom.
SYNTAX_VALID_MOVE_POINTS = 2.0

#: Criterion B, second half. Absolute percentage points of pass@1: ~2 items of
#: HumanEval+'s 164, ~4 of MBPP+'s 378. Set above one item deliberately, so a
#: single lucky solve cannot authorise a 1B-token run on its own.
PASS_AT_1_MOVE_POINTS = 1.0

#: The execution metrics criterion B reads, and the bar each must clear. Both
#: pass@1 variants count: `pass@1_plus` is the same items under EvalPlus's
#: extra inputs, so an arm that moves only the stricter one has still moved.
EXECUTION_MOVE_POINTS: Dict[str, float] = {
    "pass@1": PASS_AT_1_MOVE_POINTS,
    "pass@1_plus": PASS_AT_1_MOVE_POINTS,
    "syntax_valid": SYNTAX_VALID_MOVE_POINTS,
}


#: Slack on every "at least" comparison below, in the units of the threshold.
#:
#: A movement is a difference of two floats, so a measurement sitting *exactly*
#: on a preregistered bar can land a few parts in 1e16 under it: 0.2381 + 0.02
#: minus 0.2381 is 1.999999999999999 points, not 2. Without this, a bar written
#: as "at least two points" would silently mean "more than two points" for the
#: one value that most deserves to meet it. The slack is 1e-9 points against a
#: smallest real step of 1/378 = 0.26 points, so it cannot move any verdict a
#: benchmark could actually produce.
_BOUNDARY_TOLERANCE = 1e-9


def _meets(measured: float, threshold: float) -> bool:
    return measured >= threshold - _BOUNDARY_TOLERANCE


def _at_most(measured: float, limit: float) -> bool:
    """The retention direction of `_meets`, with the same boundary reasoning.

    A drop written as "no more than one point" has to include one point, and a
    difference of two floats lands a few parts in 1e16 either side of it.
    """

    return measured <= limit + _BOUNDARY_TOLERANCE


class ProbeGateError(ValueError):
    """Raised when two measurements are not comparable evidence."""


@dataclass(frozen=True)
class ExecutionScore:
    """One benchmark's execution result, as `scripts/code_eval.py` reports it."""

    pass_at_1: float
    pass_at_1_plus: float
    syntax_valid: float
    n: int

    def value(self, metric: str) -> float:
        return {"pass@1": self.pass_at_1,
                "pass@1_plus": self.pass_at_1_plus,
                "syntax_valid": self.syntax_valid}[metric]


@dataclass(frozen=True)
class ProbeScore:
    """What one arm -- or the untouched base -- measured, for this gate only."""

    name: str
    code_bpb: float
    execution: Mapping[str, ExecutionScore]


def execution_score(card: Scorecard) -> ExecutionScore:
    """Read a `code-execution` scorecard, or refuse it.

    Kind-checked rather than duck-typed: every scorecard in this program has a
    `metrics` dict, so reading a retrieval or BPB card here would find no
    `pass@1`, and the natural repair -- defaulting the missing metric to zero --
    would report the base's own score as a movement.
    """

    if card.kind != "code-execution":
        raise ProbeGateError(
            f"scorecard {card.name!r} is kind {card.kind!r}, not 'code-execution'; "
            f"this gate reads pass@1 and syntax validity")
    metrics = card.metrics
    missing = sorted({"pass@1", "pass@1_plus", "syntax_valid"} - set(metrics))
    if missing:
        raise ProbeGateError(
            f"scorecard {card.name!r} is missing execution metric(s) "
            f"{', '.join(missing)}; it has {sorted(metrics)}")
    return ExecutionScore(
        pass_at_1=float(metrics["pass@1"]),
        pass_at_1_plus=float(metrics["pass@1_plus"]),
        syntax_valid=float(metrics["syntax_valid"]),
        n=int(card.resolved_item_count()),
    )


def code_bpb_improvement_pct(base: ProbeScore, arm: ProbeScore) -> float:
    """Percent *reduction* in held-out code bits-per-byte. Higher is better."""

    for score in (base, arm):
        if not math.isfinite(score.code_bpb):
            raise ProbeGateError(
                f"{score.name} code BPB is {score.code_bpb!r}; a non-finite "
                f"measurement is a broken run, not a negative result")
    if base.code_bpb <= 0:
        raise ProbeGateError(
            f"base code BPB is {base.code_bpb!r}; a relative improvement "
            f"against it is undefined")
    return (base.code_bpb - arm.code_bpb) / base.code_bpb * 100.0


def execution_moves(base: ProbeScore, arm: ProbeScore) -> dict:
    """Per-benchmark, per-metric movement against the base, with its verdict.

    Regressions are recorded with their size and never count as movement: the
    criterion is that the signal *moved in the direction the gate is about*.
    A falling syntax validity is worth seeing in the payload -- it is how an
    arm that is unlearning Python announces itself -- but it is not evidence
    for spending 1B tokens.
    """

    if set(base.execution) != set(arm.execution):
        raise ProbeGateError(
            f"{arm.name} was scored on benchmarks {sorted(arm.execution)} and "
            f"the base on {sorted(base.execution)}; a gate cannot compare an "
            f"arm against a base that did not run the same benchmarks")

    moves: Dict[str, dict] = {}
    for benchmark in sorted(base.execution):
        base_score, arm_score = base.execution[benchmark], arm.execution[benchmark]
        if base_score.n != arm_score.n:
            # Not pairable: `scorecard.paired_outcomes` refuses the same
            # mismatch for the same reason. A --task-limit on one side turns a
            # smaller denominator into an apparent gain.
            raise ProbeGateError(
                f"{benchmark}: base scored {base_score.n} items and "
                f"{arm.name} scored {arm_score.n}; these are not comparable")
        metrics: Dict[str, dict] = {}
        for metric, threshold in sorted(EXECUTION_MOVE_POINTS.items()):
            delta = (arm_score.value(metric) - base_score.value(metric)) * 100.0
            metrics[metric] = {
                "base": base_score.value(metric),
                "arm": arm_score.value(metric),
                "delta_points": delta,
                "threshold_points": threshold,
                "moved": _meets(delta, threshold),
                "regressed": bool(delta < 0.0),
            }
        moves[benchmark] = {"n": base_score.n, "metrics": metrics,
                            "moved": any(entry["moved"]
                                         for entry in metrics.values())}
    return moves


def arm_verdict(base: ProbeScore, arm: ProbeScore) -> dict:
    """Whether one arm clears criterion A, criterion B, or neither."""

    improvement = code_bpb_improvement_pct(base, arm)
    moves = execution_moves(base, arm)
    bpb_cleared = _meets(improvement, CODE_BPB_IMPROVEMENT_PCT)
    execution_moved = any(entry["moved"] for entry in moves.values())
    return {
        "arm": arm.name,
        "code_bpb": arm.code_bpb,
        "code_bpb_base": base.code_bpb,
        "code_bpb_improvement_pct": improvement,
        "code_bpb_threshold_pct": CODE_BPB_IMPROVEMENT_PCT,
        "criterion_a_code_bpb": bpb_cleared,
        "criterion_b_execution": execution_moved,
        "qualifies": bool(bpb_cleared or execution_moved),
        "execution": moves,
    }


def _rank_key(verdict: dict):
    """Lowest code BPB first, then the largest total pass@1 movement.

    Both halves are what `select_on` names, in the order it names them. The
    pass@1 tie-break is summed across benchmarks rather than taken from one,
    because HumanEval+ and MBPP+ are both preregistered and neither is
    designated primary.
    """

    pass_movement = sum(
        entry["metrics"][metric]["delta_points"]
        for entry in verdict["execution"].values()
        for metric in ("pass@1", "pass@1_plus"))
    return (verdict["code_bpb"], -pass_movement, verdict["arm"])


def probes_250m_verdict(base: ProbeScore,
                        arms: Sequence[ProbeScore]) -> dict:
    """The gate: continue to the 1B branch, or stop and report the arms.

    Continue when *any* arm clears *either* criterion; stop only when no arm
    clears either. See the module docstring for why that reading and not the
    other one.
    """

    if not arms:
        raise ProbeGateError("no probe arms were supplied; this gate has "
                             "nothing to decide")
    duplicates = sorted({arm.name for arm in arms
                         if sum(1 for other in arms if other.name == arm.name) > 1})
    if duplicates:
        # Two arms under one name is a launcher defect -- most likely the same
        # run scored twice -- and silently ranking them would present it as
        # agreement between arms.
        raise ProbeGateError(f"probe arms are not uniquely named: "
                             f"{', '.join(duplicates)}")

    verdicts = [arm_verdict(base, arm) for arm in arms]
    qualifying = [verdict for verdict in verdicts if verdict["qualifies"]]
    ranking = [verdict["arm"] for verdict in sorted(qualifying, key=_rank_key)]
    best: Optional[str] = ranking[0] if ranking else None
    return {
        "gate": "probes_250m",
        "reading": "loose: continue when any arm clears either criterion",
        "continue": bool(qualifying),
        "reason": (f"{len(qualifying)} of {len(verdicts)} arm(s) cleared a "
                   f"criterion" if qualifying else
                   f"no arm improved code BPB by >={CODE_BPB_IMPROVEMENT_PCT:g}% "
                   f"or moved the execution/syntax signal"),
        "thresholds": {"code_bpb_improvement_pct": CODE_BPB_IMPROVEMENT_PCT,
                       "execution_move_points": dict(EXECUTION_MOVE_POINTS)},
        "arms": verdicts,
        "qualifying": [verdict["arm"] for verdict in qualifying],
        "ranking": ranking,
        # Named, not selected: general retention is not preregistered for this
        # stage, so the caller holding the general-side scorecards applies it.
        "best_before_retention": best,
    }


# ------------------------------------------------------------- the branch ---
#
# Phase 8's second gate, from the code run manifest:
#
#     branch_1b.continue_if: general BPB regression <=1.5%, five-task mean drop
#     <=1 point, retrieval drop <=2 points at every depth, code metrics improve
#
# Three of the four clauses carry their own numbers and are implemented as
# written. "code metrics improve" carries none -- no threshold, no metric list,
# no and/or -- and the readings spend a 2B-token extension differently, so it is
# disambiguated here. The reasoning is recorded in full as the
# `phase8-branch-1b-gate-wording` note in `runs/vast-program/events.jsonl`,
# written before the branch had produced any number; the short form is:
#
#   - `probes_250m` is a `stop_if` with an explicit "or"; this is a `continue_if`
#     with none. A continue-condition listed beside three hard retention bounds
#     is a requirement, so criterion A is **required** rather than one of two
#     ways to pass. Under the disjunction a 1B run with no BPB movement that
#     shifted MBPP+ syntax by two points would authorise a further 2B tokens.
#   - The bar is **borrowed, not invented**: 2.0%, the figure the *previous and
#     smaller* stage had to clear. The only other preregistered figure is the
#     `final` gate's 5.0%, and requiring the 1B midpoint to already meet the
#     end-of-program bar inverts the plan's staged 1B -> 2B -> SFT design. This
#     is the same rule `code_probe_report.py` used when the probe stage had no
#     retention bound of its own: reuse an adjacent preregistered number.
#   - "metrics" is plural, so BPB alone under-reads it -- but demanding strict
#     improvement in HumanEval+ pass@1, which the manifest records at 0.000 for
#     a 150M base, would stop the gate on an artifact of the base rather than on
#     the branch. So execution is checked for **regression** past the same
#     preregistered movement bars, and movement is reported without being
#     sufficient on its own.
#
# The cost is recorded rather than hidden: this is stricter than the 250M gate,
# and a branch that improved execution strongly while leaving code BPB flat is
# stopped by it. That direction is deliberate. The plan's degradation policy
# names stopping Daedalus-Code at 1B as an acceptable outcome and an
# unfinishable extension as the expensive mistake.

#: Retention. Percent *increase* in held-out general-replay BPB against the
#: untouched base, above which the branch does not continue.
BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX = 1.5

#: Retention. Absolute points of five-task mean the branch may lose.
BRANCH_FIVE_TASK_DROP_POINTS_MAX = 1.0

#: Retention. Absolute points of retrieval exact match the branch may lose, at
#: *every* depth rather than on the aggregate: a model that loses one depth and
#: gains another nets out flat while being worse at the thing this protects.
BRANCH_RETRIEVAL_DROP_POINTS_MAX = 2.0

#: Progress. Borrowed from `probes_250m` rather than invented -- see above.
BRANCH_CODE_BPB_IMPROVEMENT_PCT = CODE_BPB_IMPROVEMENT_PCT


@dataclass(frozen=True)
class BranchScore(ProbeScore):
    """What the 1B branch -- or the untouched base -- measured.

    A `ProbeScore` with the general-side measurements the 250M stage never took,
    so `code_bpb_improvement_pct` and `execution_moves` read it unchanged and
    the code half of this gate is literally the same code the probe gate ran.

    All three additions default to "not measured" rather than to a number: a
    retention gate whose input silently defaults is a gate that passes when the
    evaluation did not run, which is the one failure mode it exists to catch.
    """

    general_bpb: Optional[float] = None
    five_task_mean: Optional[float] = None
    #: `"<task>:d<depth>"` -> exact match in [0, 1], as
    #: `qat_recovery.collect_observation` already keys it, so one collector can
    #: feed phase 3's retention gate and this one.
    retrieval: Mapping[str, float] = field(default_factory=dict)


def general_bpb_regression_pct(base: BranchScore, branch: BranchScore) -> float:
    """Percent *increase* in held-out general-replay BPB. Positive is worse.

    The opposite sign convention to `code_bpb_improvement_pct`, because the two
    numbers are gated in opposite directions and a shared convention would put
    the reader one negation away from reading a retention failure as a win.
    """

    for score in (base, branch):
        if score.general_bpb is None:
            raise ProbeGateError(
                f"{score.name} has no general-replay BPB; general retention "
                f"cannot be measured against a number that was not taken")
        if not math.isfinite(score.general_bpb):
            raise ProbeGateError(
                f"{score.name} general BPB is {score.general_bpb!r}; a "
                f"non-finite measurement is a broken run, not a regression")
    if base.general_bpb <= 0:
        raise ProbeGateError(
            f"base general BPB is {base.general_bpb!r}; a relative regression "
            f"against it is undefined")
    return (branch.general_bpb - base.general_bpb) / base.general_bpb * 100.0


def execution_regressions(moves: Mapping[str, dict]) -> List[dict]:
    """Every execution metric that fell by at least its own movement bar.

    The same thresholds `execution_moves` counts upward movement at, applied
    downward. Reusing them is the point: a two-point rise in MBPP+ syntax
    validity is preregistered as meaningful, so a two-point fall is too, and
    inventing a separate regression bar here would be exactly the invention this
    gate's disambiguation avoids everywhere else.
    """

    fallen: List[dict] = []
    for benchmark in sorted(moves):
        for metric, entry in sorted(moves[benchmark]["metrics"].items()):
            if _meets(-entry["delta_points"], entry["threshold_points"]):
                fallen.append({"benchmark": benchmark, "metric": metric,
                               "base": entry["base"], "branch": entry["arm"],
                               "delta_points": entry["delta_points"],
                               "threshold_points": entry["threshold_points"]})
    return fallen


def retrieval_drops(base: BranchScore, branch: BranchScore) -> dict:
    """Per-key retrieval drop in points, the worst of them, and what is missing.

    Missing keys are returned rather than skipped. Scoring the branch at three
    of four depths and reporting the worst of those three is how a gate written
    as "at every depth" quietly becomes "at every depth we happened to measure".
    """

    if not base.retrieval:
        raise ProbeGateError(
            f"{base.name} has no retrieval baseline; there is nothing for "
            f"{branch.name}'s retrieval to be a drop against")
    missing = sorted(set(base.retrieval) - set(branch.retrieval))
    per_key = {}
    for key, base_value in sorted(base.retrieval.items()):
        if key not in branch.retrieval:
            continue
        drop = (float(base_value) - float(branch.retrieval[key])) * 100.0
        if not math.isfinite(drop):
            raise ProbeGateError(
                f"retrieval {key}: {base_value!r} -> "
                f"{branch.retrieval[key]!r} is not a finite drop")
        per_key[key] = {"base": float(base_value),
                        "branch": float(branch.retrieval[key]),
                        "drop_points": drop}
    worst = max(per_key, key=lambda key: per_key[key]["drop_points"], default=None)
    return {"per_key": per_key, "missing": missing, "worst_key": worst,
            "worst_drop_points": (per_key[worst]["drop_points"]
                                  if worst is not None else None)}


def branch_1b_verdict(base: BranchScore, branch: BranchScore) -> dict:
    """The 1B branch gate: continue to the extension and SFT, or stop.

    Every clause reports the number that decided it, and a clause whose
    measurement is absent fails with its reason rather than raising -- a report
    that names the three gates that passed and the one nobody ran is more useful
    than a traceback, and it still cannot continue.
    """

    checks: List[dict] = []

    improvement = code_bpb_improvement_pct(base, branch)
    checks.append({
        "gate": "code-bpb",
        "observed_improvement_pct": improvement,
        "limit_improvement_pct": BRANCH_CODE_BPB_IMPROVEMENT_PCT,
        "base": base.code_bpb, "branch": branch.code_bpb,
        "passed": _meets(improvement, BRANCH_CODE_BPB_IMPROVEMENT_PCT),
    })

    moves = execution_moves(base, branch)
    fallen = execution_regressions(moves)
    checks.append({
        "gate": "code-execution-regression",
        "regressed": fallen,
        "passed": not fallen,
        # Reported, never decisive: see the disambiguation above.
        "moved": sorted(name for name, entry in moves.items() if entry["moved"]),
    })

    try:
        regression = general_bpb_regression_pct(base, branch)
    except ProbeGateError as exc:
        checks.append({"gate": "general-bpb", "passed": False, "reason": str(exc)})
    else:
        checks.append({
            "gate": "general-bpb",
            "observed_regression_pct": regression,
            "limit_regression_pct": BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX,
            "base": base.general_bpb, "branch": branch.general_bpb,
            "passed": _at_most(regression, BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX),
        })

    if base.five_task_mean is None or branch.five_task_mean is None:
        absent = base.name if base.five_task_mean is None else branch.name
        checks.append({"gate": "five-task-mean", "passed": False,
                       "reason": f"{absent} five-task mean not measured"})
    else:
        drop = base.five_task_mean - branch.five_task_mean
        checks.append({
            "gate": "five-task-mean", "observed_drop_points": drop,
            "limit_drop_points": BRANCH_FIVE_TASK_DROP_POINTS_MAX,
            "base": base.five_task_mean, "branch": branch.five_task_mean,
            "passed": _at_most(drop, BRANCH_FIVE_TASK_DROP_POINTS_MAX),
        })

    try:
        retrieval = retrieval_drops(base, branch)
    except ProbeGateError as exc:
        checks.append({"gate": "retrieval", "passed": False, "reason": str(exc)})
    else:
        if retrieval["missing"]:
            checks.append({
                "gate": "retrieval", "passed": False,
                "reason": f"depths not measured: {retrieval['missing']}",
                **retrieval})
        else:
            checks.append({
                "gate": "retrieval",
                "limit_drop_points": BRANCH_RETRIEVAL_DROP_POINTS_MAX,
                "passed": _at_most(retrieval["worst_drop_points"],
                                   BRANCH_RETRIEVAL_DROP_POINTS_MAX),
                **retrieval})

    failed = [check["gate"] for check in checks if not check["passed"]]
    return {
        "gate": "branch_1b",
        "reading": ("code BPB improvement is required at the 250M stage's own "
                    "2% bar, execution is checked for regression only, and all "
                    "three retention bounds are as written"),
        "branch": branch.name,
        "continue": not failed,
        "reason": (f"{len(checks)} of {len(checks)} clauses passed"
                   if not failed else
                   f"{', '.join(failed)} did not pass"),
        "thresholds": {
            "code_bpb_improvement_pct": BRANCH_CODE_BPB_IMPROVEMENT_PCT,
            "execution_move_points": dict(EXECUTION_MOVE_POINTS),
            "general_bpb_regression_pct_max":
                BRANCH_GENERAL_BPB_REGRESSION_PCT_MAX,
            "five_task_drop_points_max": BRANCH_FIVE_TASK_DROP_POINTS_MAX,
            "retrieval_drop_points_max": BRANCH_RETRIEVAL_DROP_POINTS_MAX,
        },
        "checks": checks,
        "failed": failed,
        "execution": moves,
    }
