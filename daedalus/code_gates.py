"""Phase 8's 250M-token probe gate, as a rule that can return no.

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
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

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


class ProbeGateError(ValueError):
    """Raised when two probe measurements are not comparable evidence."""


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
