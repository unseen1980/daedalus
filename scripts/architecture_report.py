"""Phase 6 stage A scoring: what the fifteen arms bought, and what advances.

`architecture_sweep.py` deliberately stopped at launching. This is the other
half: read the finished arms on a held-out full pass, put the analytic columns
beside the measured one, and apply the stage B advancement rule. The rule is in
this file, committed, *before* any arm has a score -- which is the only order in
which a preregistered threshold means anything.

Four decisions carry the weight.

**The measurement is a full pass on the matched holdout.** Every arm logged a
`val_bpb` during training, and none of those numbers can decide this. The last
one lands at step 1,500 of 1,536, where the learning rate is still an order of
magnitude above its floor -- it measures a model mid-decay, not the model the
arm produced. It is also a bounded sample. So the arms are re-scored from their
final checkpoints, over every held-out window, and the scorecard records
`bpb_mode: full`; a sampled card is refused rather than read.

Matched means `fineweb-edu` alone. The arms trained on that source and nothing
else; the two other sources in the box's holdout root would measure transfer,
which is a real question and not this one, at three times the cost.

**The control is a grid point, and every column is a delta against it.**
`a8-kv4` is the shipped model's own ratio at the probe's width and depth. An
absolute BPB at 105M parameters over 100M tokens means very little; the same
number as a distance from the shipped ratio, measured under an identical
schedule, means what the phase is asking.

**A quality win inside the parameter margin is not a win.** The grid is
parameter-matched only to +/-1.5%: a conv block is dearer than an attention
block, so trading attention away *adds* parameters, and the sparsest arm carries
3.05% more than the densest. That bias points at exactly the arms a KV-savings
phase hopes will win. Every row therefore carries `credited_bpb_delta_pct`
alongside the raw delta -- the raw delta with the improvement its parameter
surplus could explain removed -- and `quality_win_survives_param_margin`, which
is false for an arm whose whole advantage is that it is bigger.

The discount uses `PARAM_SCALING_EXPONENT = 0.34`, Chinchilla's exponent on N.
That is deliberately an upper bound rather than an estimate: total loss carries
an irreducible term, so the true sensitivity of loss to parameters is strictly
smaller than 0.34, and over-crediting the surplus makes it *harder* for a sparse
arm to be called a winner. Conservative in the direction the bias runs.

**The floor is the plan's own gate, not a new one.** An arm advances only if its
raw BPB is no worse than the control by more than 0.5% -- the number phase 6 was
preregistered with, applied to the raw delta rather than the discounted one so
that no threshold is invented here after the fact. The discount is reported, and
carried into stage B -- where, contrary to what this file assumed before the
stage-B shape was worked out, the residual is *worse* rather than better. At
768 wide the conv-block premium grows faster than the model does, so the same
fixed-FFN grid lands 2.2% either side of its midpoint against stage A's 1.5%,
and solving the FFN per arm is no more available there than here (one step is
8.8% of a 159M model). The discount is therefore more load-bearing at stage B,
not less; `scripts/architecture_sweep.py` carries the arithmetic.

Among arms that clear the floor, what advances is the Pareto frontier on (KV
bytes down, BPB down), cheapest cache first. The floor already says "does not
lose quality"; among shapes that keep quality, the valuable one is the one whose
cache costs least. The frontier matters because KV cost ties: 8 attention layers
with 1 KV head, 4 with 2, and 2 with 4 all cost the same bytes per context
token, and at equal cost only the better-scoring shape is worth stage B's hours.

**Advancing is not recommending, and `recommend` is the other rule.** Stage A's
floor decides which arms are worth more GPU hours; it says nothing about whether
a shape may be put in front of a successor decision. The plan gates that on five
columns -- BPB, retrieval by depth, KV bytes, stock llama.cpp export and load,
and artifact size with decode -- and only one of them is measured by `score`. So
the recommendation rule reads the other four from the artifacts the evaluators
already write, and an unmeasured column is `unproven`, never a pass. That is the
whole point of having it: the failure mode of a phase like this is a table with
one strong column and four empty ones, read as a recommendation because the
empty ones did not object.

Two of the refusals are arithmetic on the plan's own thresholds rather than new
judgements. A "no worse by 2 points" retrieval gate cannot be evaluated with 20
items at a depth, because one item is 5 points and the threshold is finer than
the instrument; and a depth where the control itself scores under 2 points
cannot host a 2-point drop. Both are reported as powerless, which is a different
statement from passing.

**And the decision is handed on as a file, not as a sentence.** `select_stage_b`
writes which arms advance; stage B is the longest run this phase makes, at 2.5x
the tokens and 1.5x the width of the screen that chose them. Between the two sits
a `--arms a4-kv2,...` that somebody types. `advanced_selection` closes that gap:
stage B reads its arm list out of the committed stage-A report, so the arms that
run are the arms that were selected, a `no-advance` verdict actually stops stage
B rather than merely recommending that it stop, and the stage-B artifact records
which report chose it.

Subcommands: `score`, `report`, `recommend`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daedalus.arch_space import (MAX_KV_BYTES_PER_CONTEXT_TOKEN,  # noqa: E402
                                 PREFERRED_KV_BYTES_PER_CONTEXT_TOKEN)
from daedalus.scorecard import (ArtifactRef, ScorecardError, load_scorecard,  # noqa: E402
                                sha256_file)
from scripts.architecture_sweep import (ARMS, CONTROL, REPORT_ROOT,  # noqa: E402
                                        RUN_ROOT, SHAPES, STAGE_A, ArchArm,
                                        StageShape, arm_checkpoint_path,
                                        arm_run_name, arms_for,
                                        parameter_spread, selected_arms)
from scripts.bpb_eval import _git_short_sha, run_bpb_eval  # noqa: E402


# ============================================================ preregistered ===
# Named here, in the commit that precedes the first score. A threshold chosen
# after the numbers are on screen is not a threshold.

#: The one source the stage-A arms trained on, and therefore the only one whose
#: held-out BPB answers this screen's question.
MATCHED_HOLDOUT_SOURCE = "fineweb-edu"

#: Phase 6's own gate: an arm is out if its BPB is worse than the control's by
#: more than this, in percent. Applied to the raw delta -- see the module
#: docstring on why the parameter discount informs rather than gates.
STAGE_B_FLOOR_PCT = 0.5

#: How many non-control arms stage B re-runs at 150M over 250M tokens. Stage C
#: takes at most two finalists, so a wider stage B would buy resolution the next
#: stage cannot spend. The control runs in every stage regardless.
STAGE_B_MAX_ARMS = 3

#: Chinchilla's exponent on N in `L = E + A/N^0.34 + B/D^0.28`, used to discount
#: the parameter surplus. An upper bound, not an estimate: `E` is irreducible,
#: so `dlogL/dlogN` is strictly smaller than this in magnitude. Over-crediting
#: the surplus is the conservative error here.
PARAM_SCALING_EXPONENT = 0.34

#: Where the per-arm scorecards land, beside the sweep artifact rather than
#: inside the run directories -- a scorecard must never be mistaken for a
#: checkpoint.
SCORECARD_ROOT = f"{REPORT_ROOT}/scorecards"

#: Where `retrieval_eval.py` is pointed for this phase: one directory per arm
#: run, holding that arm's per-task retrieval scorecards.
RETRIEVAL_ROOT = f"{REPORT_ROOT}/retrieval"

#: The default `decode_bench.py` report. One file, not one per arm, and that is
#: load-bearing -- see `read_decode_passes`.
DECODE_REPORT = f"{REPORT_ROOT}/decode.json"


# ============================== preregistered: the recommendation gate ========
# Phase 6's own gate, named here in the commit that precedes the first finalist.
# Every threshold is either the plan's own number or arithmetic on it. None was
# chosen after seeing a table, and the two derived ones are written as
# expressions rather than literals so they cannot drift from the number they
# come from.

#: The context the arms train at, and therefore the deepest honest measurement
#: of the long-context benefit. Beyond it a decode number measures extrapolation.
TRAINED_CONTEXT = 2048

#: Retrieval: an arm is out if its exact match falls more than this many points
#: below the control's at any trained depth. The plan's number.
RETRIEVAL_MAX_DROP_POINTS = 2.0

#: The retrieval tasks whose depth curves the gate reads. `copy-control` is
#: deliberately absent: it is the prompt-formatter control, scored at a single
#: depth, and a formatter check is not a retention measurement.
RETRIEVAL_GATE_TASKS = ("passkey", "mqar")

#: How many items a depth needs before a 2-point threshold means anything. With
#: n items one item is worth 100/n points, so a threshold finer than that cannot
#: be crossed by less than one whole item: "no worse by 2 points" over 20 items
#: is not a strict test but a rounding of "no worse at all", passed and failed by
#: single items. Derived from the threshold rather than picked, so the two cannot
#: drift apart, and reported as powerless rather than passed -- the remedy is a
#: larger `--per-depth`, not a looser gate.
RETRIEVAL_MIN_ITEMS_PER_DEPTH = math.ceil(100.0 / RETRIEVAL_MAX_DROP_POINTS)

#: A depth where the control itself scores below the threshold cannot host a
#: drop of the threshold's size -- you cannot fall two points from one. Also
#: derived, for the same reason.
RETRIEVAL_POWER_FLOOR = RETRIEVAL_MAX_DROP_POINTS / 100.0

#: How far under the control an arm may measure on decode throughput or artifact
#: size before "does not erase the long-context benefit" stops being true.
#:
#: One number for both because one cause explains both: the grid is
#: parameter-matched only to +/-1.5% at stage A and +/-2.2% at stage B, so arms
#: differ by up to 4.4% in parameters by construction, and Q4_0 bytes track
#: parameters exactly while a memory-bound decode tracks them closely. A
#: tolerance under that would reject arms for the grid's own matching residual
#: rather than for a cost their shape imposes, which is the opposite of what
#: this gate is for.
MAX_DECODE_LOSS_PCT = 5.0
MAX_ARTIFACT_GROWTH_PCT = 5.0

#: A check's outcome. `unmeasured` and `no-power` are distinct from `fail` and
#: from each other, and none of the three is a pass: an arm cannot be
#: recommended on a column nobody measured, nor on one whose instrument is too
#: coarse to have detected the failure it is screening for.
CHECK_PASS = "pass"
CHECK_FAIL = "fail"
CHECK_UNMEASURED = "unmeasured"
CHECK_NO_POWER = "no-power"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# =================================================================== scoring ===

def scorecard_name(arm: ArchArm, tag: str = "stagea") -> str:
    return f"arch-{tag}-{arm.name}-bpb"


def scorecard_path(arm: ArchArm, *, tag: str = "stagea",
                   out_dir: str = SCORECARD_ROOT) -> Path:
    return Path(out_dir) / f"{scorecard_name(arm, tag)}.json"


def scored_from(path, checkpoint_sha: str) -> bool:
    """True when `path` already scores exactly these bytes.

    Keyed on the checkpoint digest rather than on the file existing, so a
    re-scored arm is skipped only when the thing that was scored is the thing
    that is there now. A rerun after `--refresh` retrained an arm must not
    silently keep the old arm's number.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        card = load_scorecard(path)
    except (ScorecardError, KeyError, ValueError, OSError):
        return False
    return card.provenance.artifact.sha256 == checkpoint_sha


def make_checkpoint_bpb_fn(arm: ArchArm, checkpoint, *, device: str,
                           seq_len: int, batch_size: int
                           ) -> Callable[[Path], float]:
    """Load one arm's final weights and return its per-source BPB callable.

    Imports live inside because this module's selection logic is pure and its
    tests must not need torch to exercise the rule that decides the phase.
    """
    from daedalus.config import PRESETS
    from daedalus.data import get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb
    from train import load_checkpoint

    tokenizer = get_tokenizer(None)
    model = Daedalus(PRESETS[arm.config]).to(device)
    load_checkpoint(str(checkpoint), model, map_location=device)
    model.eval()

    def bpb_fn(source_dir: Path) -> float:
        # max_batches=None is the full pass the gate requires.
        return evaluate_bpb(model, str(source_dir), seq_len, tokenizer, device,
                            batch_size=batch_size, max_batches=None)

    return bpb_fn


def _release(device: str) -> None:
    """Drop the previous arm's weights before the next 105M model lands.

    Fifteen arms scored in one process on a 24GB card is fifteen models plus
    fifteen sets of activations unless each is released.
    """
    import gc

    gc.collect()
    if not device.startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:                                   # pragma: no cover
        pass


def score_arm(arm: ArchArm, *, holdout_root: str, tag: str = "stagea",
              run_root: str = RUN_ROOT, out_dir: str = SCORECARD_ROOT,
              source: str = MATCHED_HOLDOUT_SOURCE, device: str = "cuda",
              seq_len: int = 2048, batch_size: int = 8, seed: int = 20260824,
              refresh: bool = False,
              bpb_factory: Callable[..., Callable[[Path], float]]
              = make_checkpoint_bpb_fn) -> dict:
    """Full-pass BPB for one finished arm, written as a scorecard."""

    checkpoint = arm_checkpoint_path(arm, tag, run_root)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"arm {arm.name} has no checkpoint at {checkpoint}; it either "
            "never ran or its run directory was moved")
    digest = sha256_file(checkpoint)
    path = scorecard_path(arm, tag=tag, out_dir=out_dir)
    if not refresh and scored_from(path, digest):
        return {"arm": arm.name, "scorecard": str(path), "skipped": "already-scored",
                "checkpoint_sha256": digest}

    bpb_fn = bpb_factory(arm, checkpoint, device=device, seq_len=seq_len,
                         batch_size=batch_size)
    try:
        written = run_bpb_eval(
            name=scorecard_name(arm, tag), holdout_root=holdout_root,
            out_dir=out_dir,
            artifact=ArtifactRef(path=str(checkpoint), sha256=digest,
                                 kind="checkpoint", config=arm.config),
            tokenizer_ref=ArtifactRef(path="<smollm2-default>", sha256="0" * 64,
                                      kind="tokenizer"),
            seed=seed, git_sha=_git_short_sha(), bpb_fn=bpb_fn,
            max_batches=None, sources=[source],
            runtime={"device": device, "seq_len": seq_len,
                     "batch_size": batch_size},
            details_extra={"phase": "phase6-architecture", "arm": arm.name,
                           "preset": arm.config, "tag": tag,
                           "run": arm_run_name(arm, tag)})
    finally:
        del bpb_fn
        _release(device)
    return {"arm": arm.name, "scorecard": str(written["scorecard"]),
            "checkpoint_sha256": digest}


def swept_arms(tag: str, arms: Sequence[ArchArm] = ARMS,
               report_root: str = REPORT_ROOT) -> Optional[List[ArchArm]]:
    """The arms `sweep-<tag>.json` records this stage as having trained.

    Stage A trains the whole grid, so its arm set and the shape's are the same
    thing and nothing here matters. Stage B does not: it trains what stage A
    *advanced*, four arms out of fifteen, and the shape has no idea which four.
    Scoring `arms_for(shape)` then walks into an arm that was never trained --
    `a8-kv2`, eligible but not selected -- and `score_arm` raises on the missing
    checkpoint, correctly, having already spent 26 GPU-minutes on the three arms
    it reached first and leaving the fourth unscored.

    The sweep artifact is the record of what ran, so it is what "this stage's
    arms" should mean. Returns None when there is no artifact, which keeps a
    stage that has not swept yet on the shape's grid and keeps stage A's
    behaviour byte-identical.
    """

    path = Path(report_root) / f"sweep-{tag}.json"
    if not path.exists():
        return None
    try:
        with path.open() as handle:
            swept = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    names = [entry.get("arm") for entry in (swept.get("arms") or ())]
    by_name = {arm.name: arm for arm in arms}
    # Order is preserved from the sweep, which already ran control-first, rather
    # than re-derived: the control has to be scored first for the same reason it
    # is trained first, and an unknown name is dropped rather than raising --
    # a renamed grid must not make an existing stage unscoreable.
    selected = [by_name[name] for name in names if name in by_name]
    return selected or None


def score_arms(arms: Sequence[ArchArm] = ARMS, **kwargs) -> dict:
    """Score every arm, control first, leaving each scorecard as it lands.

    Re-entrant for the same reason the sweep is: this is a GPU pass over fifteen
    checkpoints, and a session that ends mid-pass must cost only the arm it was
    on.
    """
    results = []
    for arm in arms:
        results.append(score_arm(arm, **kwargs))
    return {"scored": results}


# ================================================================= reporting ===

def credited_bpb_delta_pct(bpb_delta_pct: float, param_surplus_pct: float,
                           exponent: float = PARAM_SCALING_EXPONENT) -> float:
    """The raw delta with the improvement its parameter surplus could explain
    removed.

    An arm 3.05% larger than the control could buy about 1.04% of BPB with size
    alone under an exponent that is already an upper bound. An arm that came in
    0.5% better is therefore 0.54% *worse* once its extra parameters are paid
    for -- which is the reading this phase has to make, because the arms it
    hopes to advance are precisely the larger ones.
    """
    return bpb_delta_pct + exponent * param_surplus_pct


def arm_bpb(card, source: str = MATCHED_HOLDOUT_SOURCE) -> float:
    """The measured BPB for one source, refusing anything that is not a full
    pass over it.

    A bounded sample and a full pass are different measurements, and this one
    feeds a gate. Reading `metrics["bpb"]` would have taken whatever weighting
    the card happened to carry; reading the item asserts which corpus it is.
    """
    if card.provenance.bpb_mode != "full":
        raise ScorecardError(
            f"scorecard {card.name!r} was measured in bpb_mode "
            f"{card.provenance.bpb_mode!r}; a stage-A gate reads a full pass "
            "only")
    for item in card.items or ():
        if item.get("id") == source:
            return float(item["bpb"])
    raise ScorecardError(
        f"scorecard {card.name!r} has no item for source {source!r}; "
        f"it scored {[item.get('id') for item in card.items or ()]}")


def read_rows(arms: Sequence[ArchArm] = ARMS, *, tag: str = "stagea",
              out_dir: str = SCORECARD_ROOT,
              source: str = MATCHED_HOLDOUT_SOURCE) -> List[dict]:
    """One row per scored arm, measured column joined to analytic ones."""

    control_row = None
    rows: List[dict] = []
    for arm in arms:
        path = scorecard_path(arm, tag=tag, out_dir=out_dir)
        if not path.exists():
            continue
        card = load_scorecard(path)
        analytic = arm.describe()
        row = {
            "arm": arm.name,
            "preset": arm.config,
            "is_control": arm.is_control,
            "attention_layers": analytic["attention_layers"],
            "num_key_value_heads": analytic["num_key_value_heads"],
            "kv_bytes_per_context_token": analytic["kv_bytes_per_context_token"],
            "parameters": analytic["parameters"],
            "q4_0_MB": analytic["q4_0_MB"],
            "bpb": arm_bpb(card, source),
            "checkpoint_sha256": card.provenance.artifact.sha256,
            "scorecard": str(path),
        }
        rows.append(row)
        if arm.is_control:
            control_row = row

    if control_row is None:
        raise ScorecardError(
            f"no scorecard for the control arm {CONTROL.name!r} under "
            f"{out_dir}; every stage-A column is a delta against it, so a "
            "table without it is unreadable rather than merely incomplete")

    for row in rows:
        row["bpb_delta_pct"] = 100.0 * (row["bpb"] - control_row["bpb"]) \
            / control_row["bpb"]
        row["param_surplus_pct"] = 100.0 * (
            row["parameters"] - control_row["parameters"]) \
            / control_row["parameters"]
        row["param_explained_bpb_pct"] = -PARAM_SCALING_EXPONENT \
            * row["param_surplus_pct"]
        row["credited_bpb_delta_pct"] = credited_bpb_delta_pct(
            row["bpb_delta_pct"], row["param_surplus_pct"])
        row["kv_saving_pct"] = 100.0 * (
            control_row["kv_bytes_per_context_token"]
            - row["kv_bytes_per_context_token"]) \
            / control_row["kv_bytes_per_context_token"]
        row["passes_floor"] = row["bpb_delta_pct"] <= STAGE_B_FLOOR_PCT
        row["quality_win_survives_param_margin"] = (
            row["bpb_delta_pct"] < 0.0 and row["credited_bpb_delta_pct"] < 0.0)
    return rows


def pareto_frontier(rows: Sequence[dict]) -> List[dict]:
    """Rows no other row beats on both KV bytes and BPB, cheapest cache first.

    Needed because KV cost ties across the grid -- 8 attention layers with 1 KV
    head costs exactly what 4 with 2 and 2 with 4 do -- and at equal cache cost
    only the better-scoring shape is worth stage B's hours.
    """
    frontier = [
        row for row in rows
        if not any(
            other is not row
            and other["kv_bytes_per_context_token"] <= row["kv_bytes_per_context_token"]
            and other["bpb"] <= row["bpb"]
            and (other["kv_bytes_per_context_token"] < row["kv_bytes_per_context_token"]
                 or other["bpb"] < row["bpb"])
            for other in rows)
    ]
    return sorted(frontier, key=lambda row: (row["kv_bytes_per_context_token"],
                                             row["arm"]))


def select_stage_b(rows: Sequence[dict], *, floor_pct: float = STAGE_B_FLOOR_PCT,
                   max_arms: int = STAGE_B_MAX_ARMS) -> dict:
    """Apply the preregistered advancement rule and say what it decided."""

    eligible = [row for row in rows
                if not row["is_control"] and row["bpb_delta_pct"] <= floor_pct]
    frontier = pareto_frontier(eligible)
    selected = frontier[:max_arms]
    verdict = "advance" if selected else "no-advance"
    decision = {
        "rule": {
            "floor_pct": floor_pct,
            "max_arms": max_arms,
            "param_scaling_exponent": PARAM_SCALING_EXPONENT,
            "floor": "raw bpb_delta_pct <= floor_pct, phase 6's own gate",
            "order": "Pareto frontier on (kv_bytes down, bpb down), "
                     "cheapest cache first",
        },
        "eligible": [row["arm"] for row in eligible],
        "frontier": [row["arm"] for row in frontier],
        "selected": [row["arm"] for row in selected],
        "dropped_from_frontier": [row["arm"] for row in frontier[max_arms:]],
        "verdict": verdict,
    }
    if verdict == "no-advance":
        decision["note"] = (
            "no arm held the control's quality within the 0.5% floor, so no "
            "attention or KV-head reduction advances. The shipped ratio stands "
            "on this evidence; stage B does not run. Recorded as a negative "
            "result rather than re-run at a looser threshold.")
    else:
        unsurvived = [row["arm"] for row in selected
                      if row["bpb_delta_pct"] < 0
                      and not row["quality_win_survives_param_margin"]]
        if unsurvived:
            decision["note"] = (
                f"{unsurvived} score better than the control on raw BPB, but "
                "not by more than their parameter surplus could explain. They "
                "advance on the KV saving, not on a quality win, and must not "
                "be reported as beating the shipped ratio.")
    return decision


def report_path(tag: str = "stagea", root: str = REPORT_ROOT) -> Path:
    """Where `report` writes its machine-readable half, and where the next stage
    reads its arm list from. One function so the writer and the reader cannot
    disagree about the filename."""
    return Path(root) / f"{tag}-report.json"


def advanced_selection(*, from_tag: str = STAGE_A.tag,
                       report_root: str = REPORT_ROOT,
                       for_shape: Optional[str] = None) -> dict:
    """The arms a committed report advanced, ready to hand to the next stage.

    Stage B costs 2.5x stage A's tokens at 1.5x its width, and what it runs is
    decided entirely by a list of arm names. Retyping that list is a way to spend
    those hours on shapes the screen did not choose, and nothing downstream would
    notice: the run directories would be right, the schedule would be right, and
    the report would name arms that were never selected.

    Four refusals, each a way the handoff could produce a plausible wrong answer:

    - **No report.** Stage B's arm list is a stage-A *conclusion*; without one
      there is no default that is better than stopping. Scoring an unscored grid
      first is the remedy, not a fallback list.
    - **`no-advance`.** That verdict is preregistered: no arm held the control's
      quality inside the floor, so the shipped ratio stands and stage B does not
      run. A negative result that the next command can walk past is not a gate.
    - **A report for the shape being launched.** Reading stage B's own report to
      choose stage B's arms is circular -- it would re-run whatever already ran
      and call the result a scale-up.
    - **`advance` with nothing selected.** Verdict and list disagreeing means the
      file was hand-edited or written by another version; guessing which half is
      true is not a launcher's decision.

    A screen the deadline truncated is *not* refused -- the degradation policy
    prunes rather than blocks, and a frontier over the arms that finished is the
    best evidence there is. It is reported instead: `screened` carries what was
    scored against the full grid so a partial basis travels with the decision
    rather than being rediscovered from the row count later.
    """
    path = report_path(from_tag, report_root)
    if not path.exists():
        raise SystemExit(
            f"no report at {path}: the arms a later stage runs are a conclusion "
            f"of the {from_tag!r} screen, so score it and run `report` first. "
            "There is no default arm list that is better than stopping here.")
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    decision = (report or {}).get("stage_b")
    if not isinstance(decision, dict) or "verdict" not in decision:
        raise SystemExit(
            f"{path} carries no stage_b decision; it was not written by this "
            "module's `report` command")

    report_shape = (report.get("shape") or {}).get("name")
    if for_shape is not None and report_shape == for_shape:
        raise SystemExit(
            f"{path} is a {report_shape!r} report and {for_shape!r} is what is "
            "being launched: a stage cannot select its own arms from its own "
            "results. Point --arms-from-report at the screen that ran before it.")

    selected = list(decision.get("selected") or ())
    if decision["verdict"] != "advance":
        raise SystemExit(
            f"{path} records verdict {decision['verdict']!r}, so no arm "
            f"advances: {decision.get('note', 'see the report')} "
            "This is a preregistered negative result; re-run the screen or "
            "record it, but do not launch the next stage past it.")
    if not selected:
        raise SystemExit(
            f"{path} says {decision['verdict']!r} but selects no arm; verdict "
            "and selection disagree, and which of the two is true is not a "
            "question a launcher may answer by guessing")

    scored = len(report.get("rows") or ())
    grid = len(arms_for(SHAPES[report_shape])) if report_shape in SHAPES else None
    return {
        "report": str(path),
        "tag": report.get("tag", from_tag),
        "shape": report_shape,
        "created_at": report.get("created_at"),
        "control": report.get("control"),
        "verdict": decision["verdict"],
        "selected": selected,
        "frontier": list(decision.get("frontier") or ()),
        "eligible": list(decision.get("eligible") or ()),
        "dropped_from_frontier": list(decision.get("dropped_from_frontier") or ()),
        "rule": decision.get("rule"),
        "screened": {"scored": scored, "grid": grid,
                     "complete": grid is not None and scored == grid},
    }


def selection_notes(selection: dict) -> List[str]:
    """What a launcher that took its arm list from a report should say.

    Two facts, and a reader of the launched pass needs both: which report chose
    these arms, and whether it chose them over the whole grid or over the part of
    it that had been scored. Shared rather than restated at each call site
    because a warning that only some launchers print is worse than none -- it
    makes a partial basis look like a property of the launcher rather than of the
    evidence. The truncated case warns and continues, per the degradation policy,
    which prunes rather than blocks.
    """
    notes = []
    screened = selection.get("screened") or {}
    if not screened.get("complete"):
        notes.append(f"[architecture] warning: {selection['report']} scored "
                     f"{screened.get('scored')} of {screened.get('grid')} arms; "
                     f"these arms are the frontier of a partial screen")
    notes.append(f"[architecture] {selection['selected']} advanced by "
                 f"{selection['report']}")
    return notes


def build_report(rows: Sequence[dict], *, tag: str = "stagea",
                 shape: StageShape = STAGE_A,
                 source: str = MATCHED_HOLDOUT_SOURCE) -> dict:
    control = next(row for row in rows if row["is_control"])
    ordered = sorted(rows, key=lambda row: (row["kv_bytes_per_context_token"],
                                            row["arm"]))
    return {
        "tag": tag,
        "created_at": _utcnow(),
        "holdout_source": source,
        "bpb_mode": "full",
        "shape": {"name": shape.name, "seq_len": shape.seq_len,
                  "total_tokens": shape.total_tokens, "steps": shape.steps},
        "control": control["arm"],
        "control_bpb": control["bpb"],
        # Read off the arms this shape actually trained. The two stages differ
        # in width, and therefore in how far from parameter-matched the grid is
        # -- reporting stage A's spread on a stage-B table would under-state the
        # discount by a third.
        "parameter_spread": parameter_spread(arms_for(shape)),
        "rows": ordered,
        "stage_b": select_stage_b(rows),
        "caveats": [
            "The grid is parameter-matched only to +/-1.5%, and the residual "
            "favours attention-sparse arms: a conv block is dearer than an "
            "attention block, so cutting attention adds parameters. "
            "credited_bpb_delta_pct discounts that surplus at Chinchilla's "
            "0.34 exponent on N, which is an upper bound and therefore "
            "conservative against exactly the arms this phase hopes to "
            "advance.",
            f"A ranking measured on "
            f"{control['parameters'] / 1e6:.0f}M-parameter proxies over "
            f"{shape.total_tokens / 1e6:.0f}M tokens is a ranking at that "
            f"scale, and nothing here should be quoted as a property of a "
            f"larger successor.",
            "BPB is the only measured column. Retrieval by depth, GGUF "
            "export/load and decode shape are preregistered stage-6 gates that "
            "this pass does not measure; no arm is recommended on BPB alone.",
            f"Scored on {source} alone, the one source the arms trained on. "
            "Transfer to held-out code and other web text is unmeasured at "
            "stage A.",
        ],
    }


def render_markdown(report: dict) -> str:
    """The table a human reads, with the discount column beside the raw one."""

    lines = [
        f"# Phase 6 {report['shape']['name']}: attention x KV-head screen "
        f"({report['tag']})",
        "",
        f"Full-pass held-out BPB on `{report['holdout_source']}`, "
        f"{report['shape']['total_tokens']:,} tokens at "
        f"{report['shape']['seq_len']} context. Control `{report['control']}` "
        f"= {report['control_bpb']:.4f} BPB.",
        "",
        "| arm | attn | kv | KV B/tok | KV saved | params | BPB | vs ctrl % | "
        "param surplus % | credited % | floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in report["rows"]:
        mark = " (control)" if row["is_control"] else ""
        lines.append(
            f"| `{row['arm']}`{mark} | {row['attention_layers']} | "
            f"{row['num_key_value_heads']} | "
            f"{row['kv_bytes_per_context_token']} | "
            f"{row['kv_saving_pct']:+.0f}% | "
            f"{row['parameters'] / 1e6:.1f}M | {row['bpb']:.4f} | "
            f"{row['bpb_delta_pct']:+.2f} | {row['param_surplus_pct']:+.2f} | "
            f"{row['credited_bpb_delta_pct']:+.2f} | "
            f"{'pass' if row['passes_floor'] else 'FAIL'} |")

    stage_b = report["stage_b"]
    lines += [
        "",
        "## Stage B selection",
        "",
        f"- rule: raw delta <= {stage_b['rule']['floor_pct']}% of control, then "
        f"{stage_b['rule']['order']}, capped at {stage_b['rule']['max_arms']}",
        f"- eligible: {stage_b['eligible'] or 'none'}",
        f"- frontier: {stage_b['frontier'] or 'none'}",
        f"- **selected: {stage_b['selected'] or 'none'}**",
        f"- verdict: `{stage_b['verdict']}`",
    ]
    if stage_b.get("note"):
        lines += ["", f"> {stage_b['note']}"]
    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in report["caveats"]]
    return "\n".join(lines) + "\n"


# ========================================================== recommendation ====
# The other four columns, read from the artifacts their evaluators already
# write. Nothing here measures anything: the measuring is `retrieval_eval.py`,
# `decode_bench.py` and `export.py`, and a gate that re-derived their numbers
# would be a second opinion nobody asked for. What this does is join them, apply
# the plan's thresholds, and refuse to call a missing column a pass.


def retrieval_scorecard_path(arm: ArchArm, task: str, *, tag: str = "stagea",
                             root: str = RETRIEVAL_ROOT) -> Path:
    """Where this arm's retrieval scorecard for one task lives.

    Under the arm's *run* name rather than its grid-point name, because one arm
    name maps to two run directories across stages and a retrieval number is a
    property of a checkpoint, not of a grid point.
    """
    return Path(root) / arm_run_name(arm, tag) / f"retrieval-{task}.json"


def read_retrieval_depths(path) -> Optional[dict]:
    """`{depth: {exact_match, n}}` plus provenance, or None if not scored.

    The per-depth curve is read from the `exact_match_d<depth>` metrics
    `daedalus.retrieval.summarize` writes, and each depth's `n_d<depth>` comes
    with it -- the item count is not decoration here, it is what decides whether
    the depth can carry the gate's threshold at all.

    `artifact_kind` rides along because it is the export column's evidence: a
    retrieval scorecard whose artifact is a GGUF was produced by stock
    llama.cpp actually loading that file, which is exactly what the plan asks be
    demonstrated. Inventing a separate export-succeeded record would mean
    trusting a claim instead of a measurement.
    """
    path = Path(path)
    if not path.exists():
        return None
    card = load_scorecard(path)
    if card.kind != "retrieval":
        raise ScorecardError(
            f"scorecard {path} is kind {card.kind!r}; the retrieval column "
            "reads retrieval cards only")
    depths: Dict[int, dict] = {}
    for key, value in card.metrics.items():
        if not key.startswith("exact_match_d"):
            continue
        try:
            depth = int(key[len("exact_match_d"):])
        except ValueError:                              # pragma: no cover
            continue
        count = card.metrics.get(f"n_d{depth}")
        depths[depth] = {"exact_match": float(value),
                         "n": int(count) if count is not None else 0}
    runtime = card.provenance.runtime or {}
    return {
        "depths": depths,
        "artifact_kind": card.provenance.artifact.kind,
        "artifact_sha256": card.provenance.artifact.sha256,
        "backend": runtime.get("backend"),
        # Absent on every card written before the backend recorded it. Not
        # defaulted to the good value -- see `templated_cards`.
        "template_mode": runtime.get("template_mode"),
        "scorecard": str(path),
    }


#: The only mode in which a llama.cpp card measures a *base* model. `-st`
#: exits after one turn but still applies the chat template, so a card produced
#: that way scores the template rather than the model.
RAW_COMPLETION_MODE = "raw-completion"


def templated_cards(*records: dict) -> List[dict]:
    """The cards among these that cannot be shown to be raw completions.

    Only llama.cpp cards are judged: a torch card feeds the model its prompt
    directly and has no template to apply.

    A *missing* `template_mode` counts as templated rather than raw, which is
    the whole reason this is a separate function. Every card written before the
    backend recorded the field came from the pinned build, and that build can
    only run `-st` -- so "unstated" is not "unknown in principle", it is the
    templated path with no field to say so. Reading it the other way would let
    exactly the stage-A cards that motivated this check pass as measurements.
    """

    suspect = []
    for record in records:
        if not (record.get("artifact_kind") or "").startswith("gguf-"):
            continue
        if record.get("template_mode") != RAW_COMPLETION_MODE:
            suspect.append(record)
    return suspect


def read_retrieval(arm: ArchArm, *, tag: str = "stagea",
                   root: str = RETRIEVAL_ROOT,
                   tasks: Sequence[str] = RETRIEVAL_GATE_TASKS) -> dict:
    """Every gate task this arm has been scored on, keyed by task."""
    found = {}
    for task in tasks:
        record = read_retrieval_depths(
            retrieval_scorecard_path(arm, task, tag=tag, root=root))
        if record is not None:
            found[task] = record
    return found


def read_decode_passes(path) -> Dict[tuple, dict]:
    """`{(threads, depth): pass}` from one `decode_bench.py` report.

    Keyed by pass rather than flattened by model, because a decode number is
    only comparable to another from the *same* invocation -- `decode_bench`
    alternates models within a pass precisely so that concurrent box load hits
    both, and says in its own output that absolutes are not comparable across
    invocations. Flattening would make it trivial to read an arm measured on a
    quiet box against a control measured while a trainer was running, and
    produce a 30% "speedup" that is a report about the box.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as handle:
        report = json.load(handle)
    passes = {}
    for entry in report.get("passes") or ():
        passes[(int(entry["threads"]), int(entry["depth"]))] = entry
    return passes


def decode_entry(pass_record: Optional[dict], names: Sequence[str]
                 ) -> Optional[dict]:
    """This model's row in a pass, looked up by any of the names it may carry.

    A decode report is written by hand at the command line (`--models
    name=path`), so the name is whatever the operator typed. Both the grid-point
    name and the run name are accepted rather than mandating one, since
    rejecting a report over a naming convention would lose a real measurement.
    """
    if not pass_record:
        return None
    models = pass_record.get("models") or {}
    for name in names:
        entry = models.get(name)
        if entry and entry.get("mean") is not None:
            return entry
    return None


# ------------------------------------------------------------- the checks ----

def _check(status: str, note: str, **fields) -> dict:
    return {"status": status, "note": note, **fields}


def bpb_check(row: dict, *, floor_pct: float = STAGE_B_FLOOR_PCT) -> dict:
    """The plan's quality floor, on the raw delta stage A already computed."""
    delta = row["bpb_delta_pct"]
    passed = delta <= floor_pct
    return _check(
        CHECK_PASS if passed else CHECK_FAIL,
        f"BPB {delta:+.2f}% against the control, "
        f"{'within' if passed else 'outside'} the {floor_pct:g}% floor",
        bpb_delta_pct=delta, floor_pct=floor_pct,
        credited_bpb_delta_pct=row["credited_bpb_delta_pct"])


def kv_check(row: dict, *, ceiling: int = MAX_KV_BYTES_PER_CONTEXT_TOKEN,
             preferred: int = PREFERRED_KV_BYTES_PER_CONTEXT_TOKEN) -> dict:
    """The deployment constraint, and the only column that is pure arithmetic.

    An absolute ceiling, not a delta against the control. The shipped model sits
    at 6,144 exactly, which is why that is the ceiling -- and a proxy control at
    a different depth can therefore be over it while being the reference every
    other column is read against. That reads oddly in a table and is correct:
    the ceiling is a property of what a deployment can afford, not of what this
    stage happened to train.
    """
    kv = row["kv_bytes_per_context_token"]
    passed = kv <= ceiling
    return _check(
        CHECK_PASS if passed else CHECK_FAIL,
        f"{kv:,} KV bytes per context token, "
        f"{'at or under' if passed else 'over'} the {ceiling:,} ceiling"
        f"{' and at or under the preferred ' + format(preferred, ',') if kv <= preferred else ''}",
        kv_bytes_per_context_token=kv, ceiling=ceiling, preferred=preferred,
        at_or_under_preferred=kv <= preferred)


def retrieval_check(arm_tasks: dict, control_tasks: dict, *,
                    max_drop_points: float = RETRIEVAL_MAX_DROP_POINTS,
                    min_items: int = RETRIEVAL_MIN_ITEMS_PER_DEPTH,
                    power_floor: float = RETRIEVAL_POWER_FLOOR) -> dict:
    """Retention against the control at every trained depth, cell by cell.

    Evaluated per `(task, depth)` rather than pooled across tasks. Pooling would
    need a weighting rule this phase never preregistered, and it would let a
    strong passkey curve cover an mqar regression -- "no worse at any trained
    depth" read at the finest grain the artifacts support is both the stricter
    reading and the one that needs no new decision.

    A cell that cannot carry the threshold is `no-power`, not `pass`. Two ways
    that happens, both arithmetic: too few items for a 2-point difference to be
    less than one item, and a control already scoring under 2 points, which
    nothing can fall two points below.
    """
    if not arm_tasks:
        return _check(CHECK_UNMEASURED,
                      "no retrieval scorecard for this arm", cells=[])
    if not control_tasks:
        return _check(CHECK_UNMEASURED,
                      "the control has no retrieval scorecard, so there is "
                      "nothing to retain against", cells=[])

    # Before any cell is scored: a chat-templated card is not a weaker
    # measurement of retrieval, it is a measurement of something else. The
    # released base model scores the copy-control 1.0 through torch and 0.0
    # through a templated llama.cpp turn on the same weights, so these cards
    # read as a uniform zero whatever the architecture does -- and a uniform
    # zero is what the depth curve would look like if the arms genuinely could
    # not retrieve. Scoring them anyway produces `no-power` for the right
    # arithmetic and the wrong reason.
    templated = templated_cards(*arm_tasks.values(), *control_tasks.values())
    if templated:
        return _check(
            CHECK_UNMEASURED,
            f"{len(templated)} of {len(arm_tasks) + len(control_tasks)} "
            "retrieval cards were produced through llama.cpp without a "
            f"recorded {RAW_COMPLETION_MODE!r} mode, so the prompts reached "
            "the model wrapped in its chat template and the scores measure "
            "the template rather than retention. Re-run the retrieval pass "
            "against a llama-cli built with -DLLAMA_BUILD_UI=OFF, or a "
            "llama-completion binary.",
            cells=[], templated_scorecards=[path for path in
                                            (record.get("scorecard")
                                             for record in templated)
                                            if path])

    cells: List[dict] = []
    for task in RETRIEVAL_GATE_TASKS:
        arm_record = arm_tasks.get(task)
        control_record = control_tasks.get(task)
        if arm_record is None or control_record is None:
            missing = "the arm" if arm_record is None else "the control"
            cells.append({"task": task, "depth": None,
                          "status": CHECK_UNMEASURED,
                          "note": f"{missing} was not scored on {task}"})
            continue
        arm_depths = arm_record["depths"]
        control_depths = control_record["depths"]
        for depth in sorted(set(arm_depths) | set(control_depths)):
            arm_cell = arm_depths.get(depth)
            control_cell = control_depths.get(depth)
            if arm_cell is None or control_cell is None:
                missing = "the arm" if arm_cell is None else "the control"
                cells.append({"task": task, "depth": depth,
                              "status": CHECK_UNMEASURED,
                              "note": f"{missing} has no depth-{depth} items"})
                continue
            items = min(arm_cell["n"], control_cell["n"])
            drop_points = 100.0 * (control_cell["exact_match"]
                                   - arm_cell["exact_match"])
            common = {"task": task, "depth": depth,
                      "exact_match": arm_cell["exact_match"],
                      "control_exact_match": control_cell["exact_match"],
                      "drop_points": drop_points, "items": items}
            if items < min_items:
                cells.append({**common, "status": CHECK_NO_POWER,
                              "note": f"{items} items puts one item at "
                                      f"{100.0 / items if items else float('inf'):.1f} "
                                      f"points, coarser than the "
                                      f"{max_drop_points:g}-point threshold; "
                                      f"needs {min_items}"})
                continue
            if control_cell["exact_match"] < power_floor:
                cells.append({**common, "status": CHECK_NO_POWER,
                              "note": f"the control scores "
                                      f"{100.0 * control_cell['exact_match']:.1f} "
                                      f"points here, so it cannot be fallen "
                                      f"{max_drop_points:g} points below"})
                continue
            cells.append({**common,
                          "status": CHECK_PASS if drop_points <= max_drop_points
                          else CHECK_FAIL,
                          "note": f"{drop_points:+.1f} points against the "
                                  f"control over {items} items"})

    failed = [cell for cell in cells if cell["status"] == CHECK_FAIL]
    if failed:
        worst = max(failed, key=lambda cell: cell["drop_points"])
        return _check(CHECK_FAIL,
                      f"{worst['task']} at depth {worst['depth']} is "
                      f"{worst['drop_points']:.1f} points under the control, "
                      f"past the {max_drop_points:g}-point gate",
                      cells=cells, max_drop_points=max_drop_points)
    weak = [cell for cell in cells
            if cell["status"] in (CHECK_UNMEASURED, CHECK_NO_POWER)]
    if weak or not cells:
        return _check(CHECK_UNMEASURED if not cells else weak[0]["status"],
                      f"{len(weak)} of {len(cells)} task/depth cells could not "
                      f"carry the {max_drop_points:g}-point gate; retention is "
                      "not demonstrated at this scale",
                      cells=cells, max_drop_points=max_drop_points)
    return _check(CHECK_PASS,
                  f"no more than {max_drop_points:g} points under the control "
                  f"at any of {len(cells)} task/depth cells",
                  cells=cells, max_drop_points=max_drop_points)


def export_check(arm_tasks: dict) -> dict:
    """Did stock llama.cpp convert and load this arm?

    Answered from evidence rather than from a flag: a retrieval scorecard whose
    artifact kind is a GGUF exists only because `llama-cli` loaded that file and
    generated from it. A `checkpoint` artifact means the arm was scored in
    PyTorch, which demonstrates nothing about conversion -- so it reads as
    unmeasured, not as a failure. The remedy is to run the retrieval pass
    through the `llama-cpp` backend, which the phase wants anyway.
    """
    kinds = {record["artifact_kind"] for record in arm_tasks.values()}
    gguf = sorted(kind for kind in kinds if kind.startswith("gguf-"))
    if gguf:
        return _check(CHECK_PASS,
                      f"stock llama.cpp loaded and generated from {gguf}",
                      artifact_kinds=gguf)
    if not kinds:
        return _check(CHECK_UNMEASURED,
                      "nothing has been scored through a GGUF artifact for "
                      "this arm", artifact_kinds=[])
    return _check(CHECK_UNMEASURED,
                  f"scored only from {sorted(kinds)}; a PyTorch score is not "
                  "evidence that stock llama.cpp can convert or load this "
                  "shape", artifact_kinds=sorted(kinds))


def decode_check(row: dict, control_row: dict, passes: Dict[tuple, dict],
                 arm_names: Sequence[str], control_names: Sequence[str], *,
                 trained_context: int = TRAINED_CONTEXT,
                 max_decode_loss_pct: float = MAX_DECODE_LOSS_PCT,
                 max_artifact_growth_pct: float = MAX_ARTIFACT_GROWTH_PCT
                 ) -> dict:
    """Do artifact size and decode erase the long-context benefit?

    Three readings of the plan's sentence, in the order it makes them.

    *Artifact size.* Measured Q4_0 bytes when a decode pass recorded them,
    analytic bytes otherwise, and which one is used is stated -- a size column
    that silently switches between a file on disk and a parameter count is worse
    than one that has only ever been arithmetic.

    *Depth-zero decode.* The plan names it because it is the regime where a conv
    hybrid has least to gain: attention's KV reads barely show on an empty
    context, so an arm that is slower there is slower for most chat turns no
    matter what its cache costs.

    *The trained context.* Where the benefit is supposed to appear. "Does not
    erase" is not "must be faster", so this asks only that the arm is not worse
    at 2,048 -- and reports `long_context_advantage_pct`, how much the arm's
    ratio to the control improves between depth 0 and the trained context, as
    the evidence rather than as a threshold. That number is the phase's actual
    finding about KV cost, and turning it into a gate would mean inventing a
    bound the plan never set.

    Every comparison comes from a single pass, in which `decode_bench`
    alternated the two models. Across passes the absolutes are not comparable.
    """
    measured: Dict[int, dict] = {}
    for depth in (0, trained_context):
        for (threads, pass_depth), entry in sorted(passes.items()):
            if pass_depth != depth:
                continue
            arm_entry = decode_entry(entry, arm_names)
            control_entry = decode_entry(entry, control_names)
            if arm_entry is None or control_entry is None:
                continue
            measured[depth] = {
                "threads": threads,
                "arm_tok_s": arm_entry["mean"],
                "control_tok_s": control_entry["mean"],
                "arm_stdev": arm_entry.get("stdev"),
                "control_stdev": control_entry.get("stdev"),
                "ratio": arm_entry["mean"] / control_entry["mean"],
                "delta_pct": 100.0 * (arm_entry["mean"] - control_entry["mean"])
                / control_entry["mean"],
                "arm_file_mb": arm_entry.get("file_mb"),
                "control_file_mb": control_entry.get("file_mb"),
            }
            break

    arm_mb = (measured.get(0) or {}).get("arm_file_mb")
    control_mb = (measured.get(0) or {}).get("control_file_mb")
    if arm_mb and control_mb:
        artifact_source = "measured"
    else:
        arm_mb, control_mb = row["q4_0_MB"], control_row["q4_0_MB"]
        artifact_source = "analytic"
    artifact_growth_pct = 100.0 * (arm_mb - control_mb) / control_mb
    artifact_ok = artifact_growth_pct <= max_artifact_growth_pct
    common = {
        "artifact_source": artifact_source,
        "artifact_MB": arm_mb, "control_artifact_MB": control_mb,
        "artifact_growth_pct": artifact_growth_pct,
        "max_artifact_growth_pct": max_artifact_growth_pct,
        "max_decode_loss_pct": max_decode_loss_pct,
        "depths": measured,
    }
    if not artifact_ok:
        return _check(CHECK_FAIL,
                      f"the {artifact_source} Q4_0 artifact is "
                      f"{artifact_growth_pct:+.1f}% against the control, past "
                      f"the {max_artifact_growth_pct:g}% bound", **common)

    missing = [depth for depth in (0, trained_context) if depth not in measured]
    if missing:
        return _check(CHECK_UNMEASURED,
                      f"no decode pass measures this arm and the control "
                      f"together at depth {missing}; absolutes from separate "
                      "invocations are not comparable, so a pass containing "
                      "both is required", **common)

    slow = [depth for depth, entry in measured.items()
            if entry["delta_pct"] < -max_decode_loss_pct]
    if slow:
        worst = min(measured[depth]["delta_pct"] for depth in slow)
        return _check(CHECK_FAIL,
                      f"decode is {worst:.1f}% under the control at depth "
                      f"{slow}, past the {max_decode_loss_pct:g}% bound; the "
                      "cache saving does not pay for it", **common)

    advantage = 100.0 * (measured[trained_context]["ratio"]
                         / measured[0]["ratio"] - 1.0)
    return _check(CHECK_PASS,
                  f"artifact {artifact_growth_pct:+.1f}%, decode "
                  f"{measured[0]['delta_pct']:+.1f}% at depth 0 and "
                  f"{measured[trained_context]['delta_pct']:+.1f}% at "
                  f"{trained_context}; the ratio to the control moves "
                  f"{advantage:+.1f}% with depth",
                  long_context_advantage_pct=advantage, **common)


# ------------------------------------------------------------- the verdict ----

#: The five columns the plan gates a recommendation on, in the order it lists
#: them. All five are required: a recommendation is a statement about a shape a
#: successor might use, and four unmeasured columns do not become measured by
#: being outnumbered.
GATE_COLUMNS = ("bpb", "retrieval", "kv", "export", "decode")


def gate_verdict(checks: Dict[str, dict]) -> dict:
    """`recommended`, `blocked`, or `unproven`, and why.

    Three outcomes rather than two, because "we measured it and it failed" and
    "nobody measured it" are different facts about a shape and lead to different
    next actions -- one closes a candidate, the other names an evaluation still
    to run. Collapsing them is how a phase reports having screened a space it
    only partly measured.
    """
    failed = [name for name in GATE_COLUMNS
              if checks[name]["status"] == CHECK_FAIL]
    unproven = [name for name in GATE_COLUMNS
                if checks[name]["status"] in (CHECK_UNMEASURED, CHECK_NO_POWER)]
    if failed:
        verdict = "blocked"
        reason = (f"failed {failed} against the preregistered gate"
                  + (f"; {unproven} also unproven" if unproven else ""))
    elif unproven:
        verdict = "unproven"
        reason = (f"{unproven} not demonstrated, so this shape cannot be "
                  f"recommended on the {sorted(set(GATE_COLUMNS) - set(unproven))} "
                  "it does clear")
    else:
        verdict = "recommended"
        reason = "clears every preregistered column"
    return {"verdict": verdict, "failed": failed, "unproven": unproven,
            "reason": reason}


def gate_arm(row: dict, control_row: dict, *, retrieval: dict,
             control_retrieval: dict, decode_passes: Dict[tuple, dict],
             arm_names: Sequence[str], control_names: Sequence[str],
             floor_pct: float = STAGE_B_FLOOR_PCT) -> dict:
    """Every column for one arm, plus the verdict they add up to."""
    checks = {
        "bpb": bpb_check(row, floor_pct=floor_pct),
        "retrieval": retrieval_check(retrieval, control_retrieval),
        "kv": kv_check(row),
        "export": export_check(retrieval),
        "decode": decode_check(row, control_row, decode_passes, arm_names,
                               control_names),
    }
    return {"arm": row["arm"], "preset": row["preset"],
            "is_control": row["is_control"], "checks": checks,
            **gate_verdict(checks)}


def build_recommendation(rows: Sequence[dict], arms: Sequence[ArchArm], *,
                         tag: str = "stagea", shape: StageShape = STAGE_A,
                         retrieval_root: str = RETRIEVAL_ROOT,
                         decode_report: str = DECODE_REPORT,
                         source: str = MATCHED_HOLDOUT_SOURCE) -> dict:
    """The phase deliverable: a Pareto set of shapes that clear every column.

    A set, not a winner, and the plan says so. The columns trade against each
    other -- a cheaper cache usually costs quality -- so collapsing them to a
    ranking would mean weighting KV bytes against BPB, which is a decision that
    belongs to whoever picks the successor's size and not to this phase.
    """
    by_name = {arm.name: arm for arm in arms}
    control_row = next(row for row in rows if row["is_control"])
    control_arm = by_name[control_row["arm"]]
    control_retrieval = read_retrieval(control_arm, tag=tag,
                                       root=retrieval_root)
    decode_passes = read_decode_passes(decode_report)

    gated = []
    for row in rows:
        arm = by_name[row["arm"]]
        gated.append(gate_arm(
            row, control_row,
            retrieval=read_retrieval(arm, tag=tag, root=retrieval_root),
            control_retrieval=control_retrieval,
            decode_passes=decode_passes,
            arm_names=(arm.name, arm_run_name(arm, tag)),
            control_names=(control_arm.name,
                           arm_run_name(control_arm, tag))))

    verdicts = {entry["arm"]: entry["verdict"] for entry in gated}
    recommended = [row for row in rows
                   if not row["is_control"]
                   and verdicts[row["arm"]] == "recommended"]
    frontier = pareto_frontier(recommended)
    unproven = [entry["arm"] for entry in gated
                if entry["verdict"] == "unproven"]
    return {
        "tag": tag,
        "created_at": _utcnow(),
        "shape": {"name": shape.name, "seq_len": shape.seq_len,
                  "total_tokens": shape.total_tokens, "steps": shape.steps},
        "holdout_source": source,
        "control": control_row["arm"],
        "gate": {
            "columns": list(GATE_COLUMNS),
            "bpb_floor_pct": STAGE_B_FLOOR_PCT,
            "retrieval_max_drop_points": RETRIEVAL_MAX_DROP_POINTS,
            "retrieval_min_items_per_depth": RETRIEVAL_MIN_ITEMS_PER_DEPTH,
            "kv_ceiling": MAX_KV_BYTES_PER_CONTEXT_TOKEN,
            "kv_preferred": PREFERRED_KV_BYTES_PER_CONTEXT_TOKEN,
            "max_decode_loss_pct": MAX_DECODE_LOSS_PCT,
            "max_artifact_growth_pct": MAX_ARTIFACT_GROWTH_PCT,
            "trained_context": TRAINED_CONTEXT,
        },
        "arms": gated,
        "pareto_set": [row["arm"] for row in frontier],
        "unproven": unproven,
        "blocked": [entry["arm"] for entry in gated
                    if entry["verdict"] == "blocked"],
        "verdict": "recommend" if frontier else "no-recommendation",
        "note": (
            "no shape clears every preregistered column, so this phase "
            "recommends none. That is a statement about the evidence, not "
            "about the shapes: see `unproven` for the columns still to be "
            "measured." if not frontier else
            "the Pareto set is the deliverable; picking one of its members "
            "means weighting KV bytes against quality, which is the successor "
            "decision and not this phase's."),
        "caveats": [
            f"A ranking measured on "
            f"{control_row['parameters'] / 1e6:.0f}M-parameter proxies over "
            f"{shape.total_tokens / 1e6:.0f}M tokens is a ranking at that "
            f"scale. Depth, attention fraction and KV-head choices do not "
            f"extrapolate cleanly to a materially larger successor; the "
            f"deliverable is a set to sanity-check a decision, not a "
            f"configuration to copy.",
            "Parameter and byte accounting and KV bytes per context token "
            "transfer more reliably than the quality ranking does: the first "
            "is arithmetic and the second is a deployment constraint that "
            "binds harder as the model grows.",
            "Apple Silicon decode is pending the Mac run. The decode column "
            "here is this box's CPU, which fixes the shape of the curve and "
            "not the number a user would feel.",
        ],
    }


def render_recommendation_markdown(report: dict) -> str:
    gate = report["gate"]
    lines = [
        f"# Phase 6 {report['shape']['name']}: recommendation gate "
        f"({report['tag']})",
        "",
        f"Control `{report['control']}`. Five preregistered columns, all "
        f"required: BPB within {gate['bpb_floor_pct']}%, retrieval within "
        f"{gate['retrieval_max_drop_points']:g} points at every trained depth, "
        f"KV bytes at or under {gate['kv_ceiling']:,} "
        f"(preferred {gate['kv_preferred']:,}), stock llama.cpp export and "
        f"load, and artifact size with decode inside "
        f"{gate['max_artifact_growth_pct']:g}%/"
        f"{gate['max_decode_loss_pct']:g}%.",
        "",
        "| arm | BPB | retrieval | KV | export | decode | verdict |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    symbols = {CHECK_PASS: "pass", CHECK_FAIL: "**FAIL**",
               CHECK_UNMEASURED: "--", CHECK_NO_POWER: "n/p"}
    for entry in report["arms"]:
        mark = " (control)" if entry["is_control"] else ""
        cells = " | ".join(symbols[entry["checks"][column]["status"]]
                           for column in GATE_COLUMNS)
        lines.append(f"| `{entry['arm']}`{mark} | {cells} | "
                     f"{entry['verdict']} |")

    lines += [
        "",
        "`--` is unmeasured and `n/p` is measured-without-power. Neither is a "
        "pass: an arm is recommended only when every column is demonstrated.",
        "",
        "## Outcome",
        "",
        f"- **Pareto set: {report['pareto_set'] or 'none'}**",
        f"- unproven: {report['unproven'] or 'none'}",
        f"- blocked: {report['blocked'] or 'none'}",
        f"- verdict: `{report['verdict']}`",
        "",
        f"> {report['note']}",
        "",
        "## Why each arm landed where it did",
        "",
    ]
    for entry in report["arms"]:
        lines.append(f"- `{entry['arm']}` -- {entry['verdict']}: "
                     f"{entry['reason']}")
        for column in GATE_COLUMNS:
            check = entry["checks"][column]
            lines.append(f"  - {column}: {check['status']} -- {check['note']}")
    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in report["caveats"]]
    return "\n".join(lines) + "\n"


# ====================================================================== cli ====

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--scorecard-root", default=SCORECARD_ROOT)
    parser.add_argument("--shape", default=STAGE_A.name, choices=list(SHAPES),
                        help="which stage's arms and presets to read")
    parser.add_argument("--tag", default=None,
                        help="run-directory prefix; defaults to the one the "
                             "--shape owns")
    parser.add_argument("--source", default=MATCHED_HOLDOUT_SOURCE)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score")
    score.add_argument("--holdout-root", required=True,
                       help="the root holding one manifest-backed directory "
                            "per source; only --source is measured")
    score.add_argument("--arms", default=None,
                       help="comma-separated subset, control first")
    score.add_argument("--device", default="cuda")
    score.add_argument("--seq-len", type=int, default=STAGE_A.seq_len)
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--seed", type=int, default=20260824)
    score.add_argument("--refresh", action="store_true",
                       help="re-score arms whose scorecard already matches "
                            "their checkpoint")

    sub.add_parser("report")

    recommend = sub.add_parser("recommend")
    recommend.add_argument("--retrieval-root", default=RETRIEVAL_ROOT,
                           help="one directory per arm run, holding that arm's "
                                "retrieval-<task>.json scorecards")
    recommend.add_argument("--decode", default=DECODE_REPORT,
                           help="a single decode_bench.py report; arm and "
                                "control must appear in the same pass")

    args = parser.parse_args(argv)
    shape = SHAPES[args.shape]
    tag = shape.tag if args.tag is None else args.tag

    if args.command == "score":
        # `--arms` still wins; the sweep artifact only supplies the default, so
        # a stage is scored on the arms it trained rather than on its shape's
        # whole grid.
        stage_arms = (swept_arms(tag, arms_for(shape), args.report_root)
                      or arms_for(shape))
        report = score_arms(selected_arms(args.arms, stage_arms),
                            holdout_root=args.holdout_root,
                            tag=tag, run_root=args.run_root,
                            out_dir=args.scorecard_root, source=args.source,
                            device=args.device, seq_len=args.seq_len,
                            batch_size=args.batch_size, seed=args.seed,
                            refresh=args.refresh)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "report":
        rows = read_rows(arms_for(shape), tag=tag, out_dir=args.scorecard_root,
                         source=args.source)
        report = build_report(rows, tag=tag, shape=shape, source=args.source)
        json_path = report_path(tag, args.report_root)
        markdown_path = Path(args.report_root) / f"{tag}-report.md"
        _write_json(json_path, report)
        markdown_path.write_text(render_markdown(report))
        print(render_markdown(report))
        print(f"wrote {json_path} and {markdown_path}")
        return 0

    if args.command == "recommend":
        arms = arms_for(shape)
        rows = read_rows(arms, tag=tag, out_dir=args.scorecard_root,
                         source=args.source)
        report = build_recommendation(rows, arms, tag=tag, shape=shape,
                                      retrieval_root=args.retrieval_root,
                                      decode_report=args.decode,
                                      source=args.source)
        json_path = Path(args.report_root) / f"{tag}-recommendation.json"
        markdown_path = Path(args.report_root) / f"{tag}-recommendation.md"
        _write_json(json_path, report)
        markdown_path.write_text(render_recommendation_markdown(report))
        print(render_recommendation_markdown(report))
        print(f"wrote {json_path} and {markdown_path}")
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
