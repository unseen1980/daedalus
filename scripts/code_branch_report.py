"""Phase 8 gate 2's collector: four measurements read back as one branch score.

    python scripts/code_branch_report.py plan
    python scripts/code_branch_report.py verdict --json-out runs/code-probes/branch-1b-verdict.json

`daedalus/code_gates.py::branch_1b_verdict` is the rule that decides whether the
1B branch continues to the 2B extension and the post-training after it. It takes
two `BranchScore`s and nothing else. This is what builds them, and the split is
the same one the probe stage uses between `code_probes.py` and
`code_probe_report.py`: training writes weights, a scoring pass writes
scorecards, and the gate reads only scorecards.

**Four measurements, from three evaluators.** The probe stage needed two numbers
per model and this one needs five, because the branch gate is written as a trade
in both directions -- code down, general held. `code-bpb` and `general-bpb` come
from `code_probe_report`'s own scoring pass, the two execution cards from
`scripts/code_eval.py`, the five-task mean from an `eval.py` payload, and the
retrieval curve from `scripts/retrieval_eval.py`. Nothing here re-derives a
number: every value in a verdict written by this module is the value in the file
the verdict cites.

**The base half is already measured, and that is the point.** The 1B branch's
base is the released `hero` checkpoint -- the same bytes the probe pass scored
for code and general BPB, and the same bytes phase 3 scored for five tasks and
retrieval. So `--base-tasks` and `--base-retrieval` default to phase 3's
artifacts rather than asking for a re-measurement, and `assert_one_artifact`
proves they are the same bytes rather than assuming it. A base card from a
different checkpoint is the failure that produces a plausible retention number:
every clause of this gate is a difference against the base, and a base measured
on other weights moves all four at once.

**Pairing is checked before the gate runs, not inside it.** `execution_moves`
already refuses a mismatched item count, and `assert_paired` refuses two
execution cards from different harnesses. Retrieval needs its own version:
`assert_retrieval_paired` compares the seed, the per-depth item counts, and a
digest of the items with the *model-dependent* fields removed -- so a branch
scored at `--per-depth 10` against a base scored at 100, or with a different
seed behind the same item ids, is refused rather than differenced.

**The copy control is in the gate, deliberately.** It is a harness check that
scores 1.000 when prompts are well-formed, and phase 3's baseline records it at
`retrieval-copy-control:d0`. A gate written as "at every depth" that quietly
dropped it would pass while the thing that proves the other depths mean anything
was failing. If it is what falls, `retrieval_drops` names it as the worst key and
a reader sees immediately that the finding is about the harness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from daedalus.code_gates import (BranchScore, ProbeGateError,  # noqa: E402
                                 branch_1b_verdict)
from daedalus.scorecard import (Scorecard, ScorecardError,  # noqa: E402
                                item_digest, load_scorecard)
from scripts.code_branch import BRANCH_NAME  # noqa: E402
from scripts.code_probe_report import (BASE_MODEL, CODE_CARD,  # noqa: E402
                                       DATASETS, DEFAULT_BASE_EVAL_DIR,
                                       DEFAULT_EVAL_ROOT, GENERAL_CARD,
                                       PAIRED_RUNTIME_FIELDS,
                                       ProbeScoringError, ScoredModel,
                                       assert_paired, card_path,
                                       read_general_bpb, read_probe_score)
from scripts.qat_recovery import FIVE_TASKS, five_task_mean  # noqa: E402


#: The retrieval tasks the phase 2 evaluator writes one scorecard each for.
#: `copy-control` is a control rather than a model measurement -- see the module
#: docstring for why it is nonetheless in the gate.
RETRIEVAL_TASKS = ("passkey", "mqar", "copy-control")

#: Per-item fields that depend on the *model* rather than on the item. Excluded
#: from the identity digest two retrieval cards are paired on, because they are
#: exactly what differs between two models that scored the same items.
RETRIEVAL_OUTCOME_FIELDS = ("correct", "query_accuracy", "extracted", "response")

#: Where a model's five-task payload and retrieval cards live by default,
#: relative to the same `out_dir` its BPB and execution cards are written to.
TASKS_CARD = "tasks"

#: The base's general-side measurements, taken in phase 3 on the released `hero`
#: checkpoint. Defaults rather than requirements: `assert_one_artifact` checks
#: that whatever is passed scored the same bytes as the code-side cards, so an
#: operator who re-measures the base can point at the new cards and the check
#: still holds.
DEFAULT_BASE_TASKS = "runs/eval/baseline-hero-tasks.json"
DEFAULT_BASE_RETRIEVAL_DIR = "runs/eval/retrieval-base-n100"

DEFAULT_BRANCH_EVAL_DIR = f"{DEFAULT_EVAL_ROOT}/{BRANCH_NAME}"
DEFAULT_BRANCH_CHECKPOINT = f"runs/{BRANCH_NAME}/checkpoint.pt"
DEFAULT_BASE_CHECKPOINT = "/root/daedalus/final/hero/checkpoint.pt"
DEFAULT_VERDICT = "runs/code-probes/branch-1b-verdict.json"


class BranchScoringError(ValueError):
    """Raised when the inputs to the 1B gate are not its evidence."""


# ------------------------------------------------------------------- paths ---

def default_tasks_path(out_dir) -> str:
    return str(Path(out_dir) / f"{TASKS_CARD}.json")


def default_retrieval_paths(out_dir,
                            tasks: Sequence[str] = RETRIEVAL_TASKS) -> List[str]:
    return [str(Path(out_dir) / f"retrieval-{task}.json") for task in tasks]


def input_paths(model: ScoredModel, *, tasks: Optional[str] = None,
                retrieval: Optional[Sequence[str]] = None,
                datasets: Sequence[str] = DATASETS) -> Dict[str, str]:
    """Every file this gate reads for one model, labelled by what it supplies.

    Labelled rather than listed so `plan` can say which *clause* an absent file
    costs. "retrieval-mqar.json is missing" and "the retrieval clause cannot be
    measured" are the same fact, and only the second one is the one a session
    acts on.
    """

    paths = {CODE_CARD: str(card_path(model, CODE_CARD)),
             GENERAL_CARD: str(card_path(model, GENERAL_CARD))}
    for dataset in datasets:
        paths[dataset] = str(card_path(model, dataset))
    paths[TASKS_CARD] = tasks or default_tasks_path(model.out_dir)
    for path in (retrieval if retrieval is not None
                 else default_retrieval_paths(model.out_dir)):
        label = Path(path).stem
        if not label.startswith("retrieval-"):
            # The label is how `collect` tells a retrieval card from the rest,
            # so a card named otherwise would be read as neither and the
            # retrieval clause would fail as "not measured" while its file sat
            # on disk. `retrieval_eval.py` writes `retrieval-<task>.json`; this
            # is that contract, enforced where it is relied on.
            raise BranchScoringError(
                f"retrieval scorecard {path} is not named retrieval-<task>.json; "
                f"this gate identifies retrieval cards by that name")
        paths[label] = str(path)
    return paths


def missing_inputs(paths: Mapping[str, str]) -> List[str]:
    """The labels whose file is not on disk, in the order they were asked for."""

    return [label for label, path in paths.items() if not Path(path).exists()]


# --------------------------------------------------------------- retrieval ---

def retrieval_scores(cards: Mapping[str, Scorecard]) -> Dict[str, float]:
    """`{"<card name>:d<depth>": exact match}` across every retrieval card.

    Keyed exactly as `qat_recovery.collect_observation` keys it, and as
    `runs/qat-recovery/baseline.json` records it, so one baseline artifact can
    feed phase 3's retention gate and this one without a translation step in
    between -- a translation step being where a depth quietly goes missing.

    The undepthed `exact_match` aggregate is deliberately not a key. The gate is
    written "at every depth", and an aggregate alongside the depths would be
    counted as a fifth depth that cannot fail independently of the other four.
    """

    scores: Dict[str, float] = {}
    for name, card in sorted(cards.items()):
        if card.kind != "retrieval":
            raise BranchScoringError(
                f"scorecard {card.name!r} is kind {card.kind!r}, not "
                f"'retrieval'; this gate reads per-depth exact match from it")
        depths = {key: value for key, value in card.metrics.items()
                  if key.startswith("exact_match_d")}
        if not depths:
            raise BranchScoringError(
                f"retrieval scorecard {card.name!r} has no per-depth exact "
                f"match; it has {sorted(card.metrics)}. A retrieval clause "
                f"measured on the aggregate alone cannot fail at one depth.")
        for key, value in sorted(depths.items()):
            scores[f"{card.name}:{key[len('exact_match_'):]}"] = float(value)
    return scores


def retrieval_identity_digest(card: Scorecard) -> str:
    """A digest of *which* items were scored, with the model's answers removed.

    `scorecard.paired_outcomes` cannot be reused here: it excludes only the
    scored field, and a retrieval record also carries `extracted` and `response`,
    which differ between any two models and would make every honest pair look
    like a mismatch. What is left after `RETRIEVAL_OUTCOME_FIELDS` is removed is
    the item itself -- id, task, depth, needle position, prompt and expected
    answer -- all of which are determined by the seed and the generator, so this
    catches a re-seeded run behind identical item ids.
    """

    if card.items is None:
        raise BranchScoringError(
            f"retrieval scorecard {card.name!r} has no item sidecar, so the "
            f"items it scored cannot be compared against the other side's. Two "
            f"retrieval numbers whose items are unknown are not a difference.")
    return item_digest([{key: value for key, value in item.items()
                         if key not in RETRIEVAL_OUTCOME_FIELDS}
                        for item in card.items])


def assert_retrieval_paired(base: Scorecard, branch: Scorecard) -> None:
    """Refuse two retrieval cards that are not a difference of one thing.

    Every check here is a way to produce a finite, plausible, wrong drop:
    a different `--per-depth` changes the denominator, a different seed changes
    the needles behind the same item ids, and a different generation budget
    changes what counts as an answer.
    """

    if base.name != branch.name:
        raise BranchScoringError(
            f"retrieval cards name different tasks: {base.name!r} and "
            f"{branch.name!r}")
    if base.provenance.seed != branch.provenance.seed:
        raise BranchScoringError(
            f"{base.name}: base seed {base.provenance.seed} != branch seed "
            f"{branch.provenance.seed}; the items are generated from the seed, "
            f"so these two scored different needles")
    for field_name in PAIRED_RUNTIME_FIELDS:
        left = base.provenance.runtime.get(field_name)
        right = branch.provenance.runtime.get(field_name)
        if left != right:
            raise BranchScoringError(
                f"{base.name}: base was scored with {field_name}={left!r} and "
                f"the branch with {right!r}; these are not comparable")
    base_n = {key: value for key, value in base.metrics.items()
              if key.startswith("n_d")}
    branch_n = {key: value for key, value in branch.metrics.items()
                if key.startswith("n_d")}
    if base_n != branch_n:
        raise BranchScoringError(
            f"{base.name}: per-depth item counts differ ({base_n} vs "
            f"{branch_n}); a smaller denominator on one side is a different "
            f"measurement, not a drop")
    left_digest = retrieval_identity_digest(base)
    right_digest = retrieval_identity_digest(branch)
    if left_digest != right_digest:
        raise BranchScoringError(
            f"{base.name}: item identity digests differ ({left_digest} vs "
            f"{right_digest}); these cards scored different items")


# ------------------------------------------------------------- five tasks ---

def read_tasks_payload(path) -> dict:
    """An `eval.py` scorecard payload, or a refusal naming what is wrong."""

    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchScoringError(f"cannot read the tasks scorecard {path}: "
                                 f"{exc}") from exc
    if not isinstance(payload, dict):
        raise BranchScoringError(f"{path} is not an eval.py scorecard object")
    return payload


def tasks_artifact_sha256(payload: dict, path) -> str:
    """The checkpoint an `eval.py` payload scored, or a refusal.

    `eval.py` records a *list*, because it can score more than one checkpoint in
    a pass. A gate that took the first of several would be differencing a base
    it cannot name, so more than one is refused rather than indexed.
    """

    checkpoints = ((payload.get("provenance") or {}).get("checkpoints") or [])
    digests = sorted({entry.get("sha256") for entry in checkpoints
                      if isinstance(entry, dict) and entry.get("sha256")})
    if len(digests) != 1:
        raise BranchScoringError(
            f"{path} records {len(digests)} checkpoint digest(s) "
            f"({digests or 'none'}); this gate needs exactly one, so that the "
            f"five-task mean can be attributed to the artifact the other "
            f"clauses measured")
    return str(digests[0])


def read_five_task_mean(payload: dict, path) -> float:
    """The five-task mean in points, or a refusal naming the missing task.

    `qat_recovery.five_task_mean` returns None when a task is absent rather than
    averaging the rest, and this turns that None into a refusal: a mean over four
    tasks is smaller than a mean over five for a reason that has nothing to do
    with the model, and against a one-point drop limit it would read as a
    regression the branch did not cause.
    """

    mean = five_task_mean(payload)
    if mean is None:
        scores = payload.get("mean") if isinstance(payload.get("mean"), dict) \
            else payload.get("metrics") or {}
        missing = [task for task in FIVE_TASKS
                   if not isinstance(scores.get(task), (int, float))]
        raise BranchScoringError(
            f"{path} has no five-task mean: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} absent. A mean over the "
            f"remaining tasks would read as a drop the branch did not cause.")
    return float(mean)


def five_task_scores(payload: dict) -> Dict[str, float]:
    """The five headline per-task scores, in points, for the report only.

    Reported and never decisive: `branch_1b` gates the mean. Per-task movement
    is what a reader needs to tell "lost a point everywhere" from "lost four
    points on one task", and the `final` gate has a per-task review bar that
    this makes readable in advance.
    """

    scores = payload.get("mean") if isinstance(payload.get("mean"), dict) \
        else payload.get("metrics") or {}
    return {task: 100.0 * float(scores[task]) for task in FIVE_TASKS
            if isinstance(scores.get(task), (int, float))}


# ------------------------------------------------------------- collection ---

@dataclass(frozen=True)
class CollectedScore:
    """One model's gate inputs, plus where each of them came from."""

    score: BranchScore
    sha256: str
    cards: Dict[str, str]
    retrieval_cards: Dict[str, Scorecard] = field(default_factory=dict)
    execution_cards: Dict[str, Scorecard] = field(default_factory=dict)
    details: Dict[str, object] = field(default_factory=dict)


def assert_one_artifact(digests: Mapping[str, str], *, model: str,
                        expected: Optional[str] = None) -> str:
    """Every card of one model must have scored the same bytes.

    This is the check that lets phase 3's five-task and retrieval artifacts
    stand in as the base's general half. They were written weeks of GPU time
    before the code-side cards and by a different evaluator; what makes them the
    *same base* is that they record the same checkpoint digest, and that is
    provable rather than assumed.
    """

    distinct = sorted(set(digests.values()))
    if not distinct:
        raise BranchScoringError(f"{model}: no card recorded a checkpoint digest")
    if len(distinct) > 1:
        detail = ", ".join(f"{label}={digest[:12]}"
                           for label, digest in sorted(digests.items()))
        raise BranchScoringError(
            f"{model}: cards scored {len(distinct)} different checkpoints "
            f"({detail}). Every clause of this gate is a difference against one "
            f"artifact; mixing two moves all of them at once.")
    if expected is not None and distinct[0] != expected:
        raise BranchScoringError(
            f"{model}: cards scored {distinct[0][:12]} but the expected "
            f"checkpoint is {expected[:12]}")
    return distinct[0]


def collect(model: ScoredModel, *, tasks: Optional[str] = None,
            retrieval: Optional[Sequence[str]] = None,
            datasets: Sequence[str] = DATASETS,
            expected_sha256: Optional[str] = None) -> CollectedScore:
    """Read one model's five measurements back off disk as a `BranchScore`."""

    paths = input_paths(model, tasks=tasks, retrieval=retrieval,
                        datasets=datasets)
    absent = missing_inputs(paths)
    if absent:
        raise BranchScoringError(
            f"{model.name} has not been scored for {', '.join(absent)}; the "
            f"missing card(s) are "
            f"{', '.join(paths[label] for label in absent)}")

    try:
        probe = read_probe_score(model, datasets)
        general = read_general_bpb(model)
        execution_cards = {dataset: load_scorecard(paths[dataset])
                           for dataset in datasets}
        retrieval_cards = {
            Path(paths[label]).stem: load_scorecard(paths[label])
            for label in paths if label.startswith("retrieval-")}
    except (ScorecardError, ProbeScoringError, KeyError, ValueError) as exc:
        raise BranchScoringError(f"{model.name}: {exc}") from exc

    tasks_payload = read_tasks_payload(paths[TASKS_CARD])
    digests = {label: load_scorecard(paths[label]).provenance.artifact.sha256
               for label in (CODE_CARD, GENERAL_CARD)}
    digests.update({name: card.provenance.artifact.sha256
                    for name, card in execution_cards.items()})
    digests.update({name: card.provenance.artifact.sha256
                    for name, card in retrieval_cards.items()})
    digests[TASKS_CARD] = tasks_artifact_sha256(tasks_payload,
                                                paths[TASKS_CARD])
    digest = assert_one_artifact(digests, model=model.name,
                                 expected=expected_sha256)

    score = BranchScore(
        name=model.name,
        code_bpb=probe.code_bpb,
        execution=probe.execution,
        general_bpb=float(general["bpb"]),
        five_task_mean=read_five_task_mean(tasks_payload, paths[TASKS_CARD]),
        retrieval=retrieval_scores(retrieval_cards),
    )
    return CollectedScore(
        score=score, sha256=digest, cards=dict(paths),
        retrieval_cards=retrieval_cards, execution_cards=execution_cards,
        details={"checkpoint": model.checkpoint,
                 "general_bpb_share_covered": general.get("share_covered"),
                 "general_bpb_sources": general.get("sources"),
                 "five_task_scores": five_task_scores(tasks_payload)})


def harness_constraints(collected: CollectedScore) -> Dict[str, dict]:
    """What a matching pass must be run with, read off the cards themselves.

    Printed by `plan` so the branch's scoring pass is configured from the base's
    own provenance rather than from a remembered command line. The base's
    retrieval baseline was taken at `--per-depth 100`; the evaluator's default is
    10, and a branch scored at the default pairs against nothing.
    """

    constraints: Dict[str, dict] = {}
    for name, card in sorted({**collected.execution_cards,
                              **collected.retrieval_cards}.items()):
        entry: Dict[str, object] = {
            "seed": card.provenance.seed,
            "item_count": card.resolved_item_count(),
        }
        entry.update({field_name: card.provenance.runtime.get(field_name)
                      for field_name in PAIRED_RUNTIME_FIELDS})
        per_depth = {key: int(value) for key, value in sorted(card.metrics.items())
                     if key.startswith("n_d")}
        if per_depth:
            entry["per_depth_n"] = per_depth
        constraints[name] = entry
    return constraints


# ------------------------------------------------------------------- gate ---

def build_branch_verdict(base: CollectedScore, branch: CollectedScore) -> dict:
    """Pair the two sides, run the gate, and record what it read."""

    if base.sha256 == branch.sha256:
        raise BranchScoringError(
            f"the branch and the base are the same checkpoint "
            f"({base.sha256[:12]}); this gate measures what 1B tokens of "
            f"continued pretraining changed, and there is no difference here")
    for dataset, card in sorted(base.execution_cards.items()):
        if dataset not in branch.execution_cards:
            raise BranchScoringError(
                f"the branch has no {dataset} card; the base has one")
        assert_paired(card, branch.execution_cards[dataset])
    for name, card in sorted(base.retrieval_cards.items()):
        if name not in branch.retrieval_cards:
            raise BranchScoringError(
                f"the branch has no {name} card; the base has one, so this "
                f"gate's 'at every depth' cannot be measured")
        assert_retrieval_paired(card, branch.retrieval_cards[name])

    gate = branch_1b_verdict(base.score, branch.score)
    return {
        "schema": 1,
        "gate": gate,
        "continue": bool(gate["continue"]),
        "reason": gate["reason"],
        "models": {
            side.score.name: {"sha256": side.sha256, "cards": side.cards,
                              "details": side.details}
            for side in (base, branch)},
        "five_task_scores": {side.score.name: side.details["five_task_scores"]
                             for side in (base, branch)},
        "harness": {side.score.name: harness_constraints(side)
                    for side in (base, branch)},
    }


# -------------------------------------------------------------------- cli ---

def _models(a) -> tuple:
    base = ScoredModel(name=BASE_MODEL, checkpoint=a.base_checkpoint,
                       out_dir=a.base_out_dir)
    branch = ScoredModel(name=a.branch_name, checkpoint=a.branch_checkpoint,
                         out_dir=a.branch_out_dir)
    return base, branch


def _collect_args(a, model: ScoredModel, *, is_base: bool) -> CollectedScore:
    return collect(
        model,
        tasks=(a.base_tasks if is_base else a.branch_tasks),
        retrieval=(a.base_retrieval if is_base else a.branch_retrieval) or None,
        expected_sha256=(a.base_sha256 if is_base else None))


def _write_json(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _plan(a) -> int:
    base, branch = _models(a)
    ready = True
    for model, is_base in ((base, True), (branch, False)):
        paths = input_paths(
            model,
            tasks=(a.base_tasks if is_base else a.branch_tasks),
            retrieval=(a.base_retrieval if is_base else a.branch_retrieval)
            or None)
        absent = missing_inputs(paths)
        print(f"=== {model.name} ({len(paths) - len(absent)}/{len(paths)} "
              f"card(s) present) ===")
        for label, path in paths.items():
            mark = " " if label not in absent else "M"
            print(f"  [{mark}] {label:24s} {path}")
        if absent:
            ready = False
            print(f"  missing: {', '.join(absent)}")
            continue
        try:
            collected = _collect_args(a, model, is_base=is_base)
        except (BranchScoringError, ProbeGateError) as exc:
            ready = False
            print(f"  REFUSE: {exc}")
            continue
        print(f"  checkpoint {collected.sha256[:12]}, "
              f"code BPB {collected.score.code_bpb:.4f}, "
              f"general BPB {collected.score.general_bpb:.4f}, "
              f"five-task {collected.score.five_task_mean:.2f}, "
              f"{len(collected.score.retrieval)} retrieval depth(s)")
        for name, entry in harness_constraints(collected).items():
            print(f"      {name:24s} {json.dumps(entry, sort_keys=True)}")
    print("\nboth sides are ready" if ready else
          "\nnot ready: the branch is scored by the pass that writes the cards "
          "marked M above, with the harness settings printed for the base")
    return 0 if ready else 1


def _verdict(a) -> int:
    base, branch = _models(a)
    try:
        collected_base = _collect_args(a, base, is_base=True)
        collected_branch = _collect_args(a, branch, is_base=False)
        verdict = build_branch_verdict(collected_base, collected_branch)
    except (BranchScoringError, ProbeGateError, ProbeScoringError,
            ScorecardError, OSError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    _write_json(a.json_out, verdict)
    _print_verdict(verdict)
    print(f"\nwrote {a.json_out}")
    return 0


def _print_verdict(verdict: dict) -> None:
    gate = verdict["gate"]
    print(f"\n{gate['gate']}: {'continue' if gate['continue'] else 'STOP'} -- "
          f"{gate['reason']}")
    for check in gate["checks"]:
        mark = "pass" if check["passed"] else "FAIL"
        detail = check.get("reason") or ""
        for key in ("observed_improvement_pct", "observed_regression_pct",
                    "observed_drop_points", "worst_drop_points"):
            if check.get(key) is not None:
                detail = f"{key.replace('observed_', '')} {check[key]:+.3f}"
                break
        if check["gate"] == "code-execution-regression" and not detail:
            detail = (f"{len(check['regressed'])} metric(s) fell past their bar"
                      if check["regressed"] else "no metric fell past its bar")
        print(f"  {mark}  {check['gate']:28s} {detail}")


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def shared(parser):
        parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
        parser.add_argument("--base-out-dir", default=DEFAULT_BASE_EVAL_DIR)
        parser.add_argument("--base-tasks", default=DEFAULT_BASE_TASKS)
        parser.add_argument(
            "--base-retrieval", action="append", default=None,
            help=f"repeatable; defaults to {DEFAULT_BASE_RETRIEVAL_DIR}/"
                 f"retrieval-<task>.json for {', '.join(RETRIEVAL_TASKS)}")
        parser.add_argument(
            "--base-sha256", default=None,
            help="refuse unless every base card scored this checkpoint")
        parser.add_argument("--branch-name", default=BRANCH_NAME)
        parser.add_argument("--branch-checkpoint",
                            default=DEFAULT_BRANCH_CHECKPOINT)
        parser.add_argument("--branch-out-dir", default=DEFAULT_BRANCH_EVAL_DIR)
        parser.add_argument("--branch-tasks", default=None)
        parser.add_argument("--branch-retrieval", action="append", default=None)

    plan = sub.add_parser("plan", help="which cards exist, and what the branch "
                                       "pass must match")
    shared(plan)
    plan.set_defaults(fn=_plan)

    verdict = sub.add_parser("verdict", help="the 1B gate from existing cards")
    shared(verdict)
    verdict.add_argument("--json-out", default=DEFAULT_VERDICT)
    verdict.set_defaults(fn=_verdict)

    a = p.parse_args(argv)
    if a.base_retrieval is None:
        a.base_retrieval = default_retrieval_paths(DEFAULT_BASE_RETRIEVAL_DIR)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(_cli())
