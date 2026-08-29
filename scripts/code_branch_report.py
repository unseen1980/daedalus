"""Phase 8 gate 2's scoring pass and collector: five measurements, one score.

    python scripts/code_branch_report.py plan
    python scripts/code_branch_report.py score --device cuda
    python scripts/code_branch_report.py verdict --json-out runs/code-probes/branch-1b-verdict.json
    python scripts/code_branch_report.py stop-record   # only when the gate said stop

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

**`stop-record` is what a no is worth carrying forward.** The gate returns a
threshold answer and this cannot change it; what the aggregate it differenced
cannot say is whether an 8-point retrieval drop on 100 items is one-sided
disagreement or churn, and which of two general-replay sources the regression is
in. Both are read off the per-item sidecars and the per-source breakdown the
cards already carry, and both are marked as deciding nothing -- the same stance
`execution_moves` takes with `moved`. It refuses a branch the gate continued.

**`score` runs the branch's half of that pair, configured from the base's own
cards.** The base was measured across two phases and four evaluators, and the
knobs that fix what its numbers *are* -- `--per-depth`, the depth set, the seed,
the generation budget, the model config, `--task-limit` -- live in its
scorecards' provenance rather than in anyone's memory. `branch_pass_plan` reads
them back out and builds the branch's argv from them, so the pair is comparable
by construction instead of by a command line retyped three days later. The one
knob no scorecard records is MQAR's query count, which only ever appears inside
a prompt: `assert_items_reproduce` regenerates the items the settings imply and
refuses unless their identity digest is the base card's -- on the CPU, before
the checkpoint is loaded, rather than at the verdict after the GPU time is
spent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from daedalus.code_gates import (BranchScore, ProbeGateError,  # noqa: E402
                                 branch_1b_verdict)
from daedalus.scorecard import (Scorecard, ScorecardError,  # noqa: E402
                                item_digest, load_scorecard, sha256_file)
from scripts.code_branch import BRANCH_NAME, BRANCH_TOKENS  # noqa: E402
from scripts.code_probes import (DEFAULT_MIXTURE_RECORD,  # noqa: E402
                                 arm_is_complete)
from scripts.bpb_eval import per_source_bpb  # noqa: E402
from scripts.mcnemar import mcnemar  # noqa: E402
from scripts.code_probe_report import (BASE_MODEL, CODE_CARD,  # noqa: E402
                                       DATASETS, DEFAULT_BASE_EVAL_DIR,
                                       DEFAULT_EVAL_ROOT, GENERAL_CARD,
                                       PAIRED_RUNTIME_FIELDS,
                                       ProbeScoringError, ScoredModel,
                                       assert_paired, card_path,
                                       execution_command, pair_by_source,
                                       read_general_bpb, read_probe_score,
                                       score_bpb, score_execution, scored_from,
                                       scoring_plan)
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

#: Written beside the verdict rather than into it. The verdict is the gate's
#: output and is already on disk; rewriting it to add fields discovered
#: afterwards is how an immutable record stops being one.
DEFAULT_STOP_RECORD = "runs/code-probes/branch-1b-stop.json"


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
        bpb_cards = {label: load_scorecard(paths[label])
                     for label in (CODE_CARD, GENERAL_CARD)}
        # The breakdown behind `code_bpb`, carried from here so the verdict can
        # pair it without re-reading the card. Read inside the try because a
        # card whose aggregate has no breakdown is a refusal of this model's
        # inputs, not a crash in the middle of the gate.
        code_by_source = per_source_bpb(bpb_cards[CODE_CARD])
        execution_cards = {dataset: load_scorecard(paths[dataset])
                           for dataset in datasets}
        retrieval_cards = {
            Path(paths[label]).stem: load_scorecard(paths[label])
            for label in paths if label.startswith("retrieval-")}
    except (ScorecardError, ProbeScoringError, KeyError, ValueError) as exc:
        raise BranchScoringError(f"{model.name}: {exc}") from exc

    tasks_payload = read_tasks_payload(paths[TASKS_CARD])
    digests = {label: card.provenance.artifact.sha256
               for label, card in bpb_cards.items()}
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
                 "code_bpb_by_source": code_by_source,
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


# ----------------------------------------------------------- scoring pass ---

#: How many queries an MQAR item asks. The one harness knob no scorecard
#: records: `retrieval_eval.py` writes the backend, the device and the
#: generation budget into runtime and the per-depth counts into metrics, but the
#: query count only ever appears *inside* a prompt. So it is stated here as the
#: evaluator's own default -- which is what the base was scored with -- and
#: `assert_items_reproduce` proves it against the base's own items before the
#: checkpoint is loaded. A stated value checked before it costs anything is not
#: a guess; an unchecked one rewrites every MQAR item, and the pair would be
#: refused at the verdict, after the GPU time.
DEFAULT_N_QUERIES = 4

#: The control's items carry depth 0, so its per-depth count is
#: `--control-items` rather than `--per-depth`.
CONTROL_CARD = "retrieval-copy-control"


def _one(values: Sequence, what: str):
    """The single distinct value, or a refusal naming the disagreement."""

    distinct = list(dict.fromkeys(values))
    if len(distinct) != 1:
        raise BranchScoringError(
            f"the base's cards disagree on {what} ({distinct}); the branch's "
            f"pass is configured from them, and there is no single setting "
            f"here that would reproduce all of them")
    if distinct[0] is None:
        raise BranchScoringError(
            f"the base's cards record no {what}, so the branch's pass would "
            f"have to guess it")
    return distinct[0]


def per_depth_counts(card: Scorecard) -> Dict[int, int]:
    """`{depth: items}` off a retrieval card's `n_d<depth>` metrics."""

    counts = {int(key[len("n_d"):]): int(value)
              for key, value in card.metrics.items() if key.startswith("n_d")}
    if not counts:
        raise BranchScoringError(
            f"retrieval scorecard {card.name!r} has no per-depth item count, "
            f"so the --per-depth the branch must match cannot be read from it")
    return counts


@dataclass(frozen=True)
class RetrievalSettings:
    """The `retrieval_eval.py` configuration that reproduces the base's items."""

    depths: Tuple[int, ...]
    per_depth: int
    control_items: int
    seed: int
    max_new_tokens: int
    backend: str = "torch"
    n_queries: int = DEFAULT_N_QUERIES

    def to_dict(self) -> dict:
        return {"depths": list(self.depths), "per_depth": self.per_depth,
                "control_items": self.control_items, "seed": self.seed,
                "max_new_tokens": self.max_new_tokens, "backend": self.backend,
                "n_queries": self.n_queries}


def retrieval_settings_from(cards: Mapping[str, Scorecard], *,
                            n_queries: int = DEFAULT_N_QUERIES
                            ) -> RetrievalSettings:
    """The branch's retrieval harness, read off the base's own cards.

    Every value here is one a mismatch in would produce a finite, plausible,
    wrong drop: a smaller `--per-depth` is a different denominator, a different
    seed is different needles behind the same item ids, a shorter generation
    budget is a different definition of an answer.
    """

    if not cards:
        raise BranchScoringError(
            "there is no base retrieval card to configure the branch's pass "
            "from; a remembered command line is how a branch ends up scored at "
            "the evaluator's default --per-depth of 10 against a base scored "
            "at 100")
    control = cards.get(CONTROL_CARD)
    if control is None:
        raise BranchScoringError(
            f"the base has no {CONTROL_CARD} card, so --control-items would be "
            f"a guess -- and the control is in this gate, so a branch pass that "
            f"sized it differently could not be paired against the base at all")
    measured = {name: card for name, card in cards.items()
                if name != CONTROL_CARD}
    if not measured:
        raise BranchScoringError(
            "the base's only retrieval card is the copy control; there is no "
            "depth curve for the branch's pass to reproduce")

    backend = _one([card.provenance.runtime.get("backend")
                    for card in cards.values()], "the retrieval backend")
    if backend != "torch":
        raise BranchScoringError(
            f"the base's retrieval cards were scored through the {backend!r} "
            f"backend; this pass scores a PyTorch checkpoint, and a card from "
            f"another backend is a different harness rather than a different "
            f"model")
    counts = {name: per_depth_counts(card) for name, card in measured.items()}
    return RetrievalSettings(
        depths=_one([tuple(sorted(count)) for count in counts.values()],
                    "the depth set"),
        per_depth=int(_one([n for count in counts.values()
                            for n in count.values()],
                           "the per-depth item count")),
        control_items=int(_one(list(per_depth_counts(control).values()),
                               f"{CONTROL_CARD}'s item count")),
        seed=int(_one([card.provenance.seed for card in cards.values()],
                      "the seed")),
        max_new_tokens=int(_one([card.provenance.runtime.get("max_new_tokens")
                                 for card in cards.values()],
                                "max_new_tokens")),
        backend=str(backend),
        n_queries=int(n_queries),
    )


def generated_identity_digests(settings: RetrievalSettings, *,
                               tokenizer=None) -> Dict[str, str]:
    """The item identity digest these settings produce, per card name.

    Built by running the real `score_items` over empty completions rather than
    reading the identity fields off the items directly: the sidecar's fields are
    whatever that function puts in a record, and a hand-written second copy of
    that list would drift from it silently -- in the direction of a digest that
    never matches anything and a pass that can never start.
    """

    from daedalus.retrieval import make_all_items, score_items

    if tokenizer is None:
        from daedalus.data import get_tokenizer

        tokenizer = get_tokenizer()
    generated = make_all_items(tokenizer, depths=list(settings.depths),
                               per_depth=settings.per_depth,
                               seed=settings.seed,
                               n_queries=settings.n_queries,
                               control_items=settings.control_items)
    digests: Dict[str, str] = {}
    for task, items in generated.items():
        records = score_items(items, [""] * len(items))
        for item, record in zip(items, records):
            record["prompt"] = item.prompt
        digests[f"retrieval-{task}"] = item_digest(
            [{key: value for key, value in record.items()
              if key not in RETRIEVAL_OUTCOME_FIELDS} for record in records])
    return digests


def assert_items_reproduce(settings: RetrievalSettings,
                           cards: Mapping[str, Scorecard], *,
                           tokenizer=None) -> Dict[str, str]:
    """Refuse settings that would not regenerate the base's own items.

    The verdict already refuses two retrieval cards whose items differ. This is
    the same check moved to where it is free: item generation is seeded string
    assembly on the CPU, so a misconfigured pass is caught in seconds instead of
    after the branch has been generated from for an hour.
    """

    digests = generated_identity_digests(settings, tokenizer=tokenizer)
    for name, card in sorted(cards.items()):
        if name not in digests:
            raise BranchScoringError(
                f"the base has a {name} card, but these settings generate no "
                f"such task (only {', '.join(sorted(digests))})")
        if digests[name] != retrieval_identity_digest(card):
            raise BranchScoringError(
                f"{name}: these settings would score different items than the "
                f"base's card ({digests[name]} vs "
                f"{retrieval_identity_digest(card)}). Every setting but "
                f"--n-queries is read off the base's own cards, so this is "
                f"almost always the query count: the pass is configured for "
                f"{settings.n_queries} and the base's prompts say otherwise.")
    return digests


def retrieval_command(model: ScoredModel, settings: RetrievalSettings, *,
                      device: str = "cuda", config: str,
                      python: str = sys.executable) -> List[str]:
    """The one `retrieval_eval.py` invocation that writes all three cards."""

    # Resolved beside this file rather than relative to the working directory: a
    # phase the controller detaches is not guaranteed to be standing in the
    # repository root, and a retrieval pass that dies on "No such file" costs
    # the branch's scoring slot for a cwd.
    evaluator = Path(__file__).resolve().parent / "retrieval_eval.py"
    return [python, str(evaluator),
            "--backend", settings.backend,
            "--checkpoint", str(model.checkpoint),
            "--config", str(config),
            "--device", str(device),
            "--depths", ",".join(str(int(depth)) for depth in settings.depths),
            "--per-depth", str(settings.per_depth),
            "--n-queries", str(settings.n_queries),
            "--control-items", str(settings.control_items),
            "--max-new-tokens", str(settings.max_new_tokens),
            "--seed", str(settings.seed),
            "--out-dir", str(model.out_dir)]


def tasks_settings_from(payload: dict, path) -> dict:
    """`eval.py`'s harness, read off the base's own payload.

    `eval.py` has no task-selection flag -- it scores the same five -- so what
    has to match is the config the checkpoint is built with, the seed, and the
    per-task limit. The limit matters most: the base was scored on the full
    validation splits, and a branch scored on a 500-example subset would carry
    about two points of sampling noise against a one-point gate.
    """

    provenance = payload.get("provenance") or {}
    config = provenance.get("config")
    seed = provenance.get("seed")
    if not config or seed is None:
        raise BranchScoringError(
            f"{path} records no config/seed, so the branch's five-task pass "
            f"would have to be configured from memory")
    tasks = provenance.get("tasks") or {}
    if not tasks:
        raise BranchScoringError(
            f"{path} records no task provenance, so the --task-limit the "
            f"branch must match cannot be read from it")
    limits = list(dict.fromkeys((spec or {}).get("limit")
                                for spec in tasks.values()))
    if len(limits) != 1:
        raise BranchScoringError(
            f"{path} scored its tasks at {len(limits)} different limits "
            f"({limits}); there is no single --task-limit that reproduces it")
    return {"config": str(config), "seed": int(seed), "task_limit": limits[0],
            "device": provenance.get("device")}


def tasks_command(model: ScoredModel, settings: Mapping[str, object], *,
                  device: Optional[str] = None, out: Optional[str] = None,
                  python: str = sys.executable) -> List[str]:
    """The `eval.py` argv for the branch's five-task pass.

    `--no-wandb`, deliberately: this is a scoring pass whose output is a file
    the gate reads, and a run that can fail on a network call is a run that can
    lose an hour of scoring to something that has nothing to do with the model.
    """

    command = [python, str(Path(_ROOT) / "eval.py"),
               "--config", str(settings["config"]),
               "--checkpoints", str(model.checkpoint),
               "--device", str(device or settings.get("device") or "cuda"),
               "--seed", str(settings["seed"]),
               "--out", str(out or default_tasks_path(model.out_dir)),
               "--no-wandb"]
    if settings.get("task_limit"):
        command += ["--task-limit", str(int(settings["task_limit"]))]
    return command


def branch_pass_plan(model: ScoredModel, base: CollectedScore, *,
                     device: str = "cuda",
                     n_queries: int = DEFAULT_N_QUERIES) -> dict:
    """Every command the branch's scoring pass will run, and what fixed it.

    Assembled before anything is spent, and printed by `plan`, so the harness
    the branch will be measured with is reviewable against the base's
    provenance rather than after the fact against its cards.
    """

    config = _one([card.provenance.artifact.config
                   for card in (*base.execution_cards.values(),
                                *base.retrieval_cards.values())],
                  "the model config")
    execution_seed = int(_one([card.provenance.seed
                               for card in base.execution_cards.values()],
                              "the execution seed"))
    retrieval = retrieval_settings_from(base.retrieval_cards,
                                        n_queries=n_queries)
    tasks_path = base.cards[TASKS_CARD]
    tasks = tasks_settings_from(read_tasks_payload(tasks_path), tasks_path)
    return {
        "config": str(config),
        "device": device,
        "execution_seed": execution_seed,
        "retrieval": retrieval,
        "tasks": tasks,
        "commands": {
            "tasks": tasks_command(model, tasks, device=device),
            "retrieval": retrieval_command(model, retrieval, device=device,
                                           config=config),
            **{dataset: execution_command(model, dataset, device=device,
                                          config=str(config),
                                          seed=execution_seed)
               for dataset in sorted(base.execution_cards)},
        },
    }


def tasks_scored_from(path, checkpoint_sha: str) -> bool:
    """True when this `eval.py` payload already scores exactly these bytes.

    `scored_from`'s counterpart for the one input that is not a scorecard.
    """

    try:
        return tasks_artifact_sha256(read_tasks_payload(path),
                                     path) == checkpoint_sha
    except BranchScoringError:
        return False


def _run_subprocess(command: Sequence[str]) -> int:
    return subprocess.run([str(part) for part in command], cwd=_ROOT).returncode


def _run_step(runner: Optional[Callable[[Sequence[str]], int]],
              command: Sequence[str], *, what: str) -> None:
    code = (runner or _run_subprocess)(command)
    if code != 0:
        raise BranchScoringError(
            f"{what} exited {code}; command was "
            f"{' '.join(str(part) for part in command)}")


def score_branch(model: ScoredModel, *, base: CollectedScore, bpb_plan: dict,
                 pass_plan: Optional[dict] = None, device: str = "cuda",
                 batch_size: int = 8, n_queries: int = DEFAULT_N_QUERIES,
                 total_tokens: Optional[int] = BRANCH_TOKENS,
                 run_dir=None, refresh: bool = False, tokenizer=None,
                 runner: Optional[Callable[[Sequence[str]], int]] = None
                 ) -> dict:
    """Write the branch's five cards with the harness the base's cards fix.

    Re-entrant on the same terms as the probe pass: a card is reused only when
    the bytes it scored are the bytes on disk now, so a session that ends
    mid-pass costs the one measurement it was on rather than the whole set.
    """

    checkpoint = Path(model.checkpoint)
    if not checkpoint.exists():
        raise BranchScoringError(
            f"{model.name} has no checkpoint at {checkpoint}; it either never "
            f"ran or its run directory was moved")
    digest = sha256_file(checkpoint)
    if digest == base.sha256:
        raise BranchScoringError(
            f"{model.name}'s checkpoint is the base ({digest[:12]}); this gate "
            f"measures what 1B tokens of continued pretraining changed, so "
            f"scoring it would spend the pass to measure nothing")
    if total_tokens:
        directory = Path(run_dir) if run_dir else checkpoint.parent
        if not arm_is_complete(directory, int(total_tokens)):
            raise BranchScoringError(
                f"{directory} has not trained its {int(total_tokens):,}-token "
                f"budget. A checkpoint is written throughout a run, so its "
                f"existence says 'this can be resumed', not 'this is done' -- "
                f"and gating the branch on a model that is still training is a "
                f"verdict about a checkpoint nobody can name.")

    pass_plan = pass_plan or branch_pass_plan(model, base, device=device,
                                              n_queries=n_queries)
    # Before the checkpoint is loaded: seconds of CPU against an hour of GPU.
    assert_items_reproduce(pass_plan["retrieval"], base.retrieval_cards,
                           tokenizer=tokenizer)

    config = pass_plan["config"]
    # BPB first. It is the one pass that runs in-process, and it releases the
    # device in its own `finally`, so the subprocesses below never land on a
    # card that is still holding a model.
    bpb = score_bpb(model, plan=bpb_plan, config=config, device=device,
                    batch_size=batch_size, refresh=refresh)
    execution = score_execution(model, datasets=tuple(sorted(base.execution_cards)),
                                device=device, config=config,
                                seed=pass_plan["execution_seed"],
                                refresh=refresh, runner=runner)

    ran: Dict[str, dict] = {}
    tasks_path = default_tasks_path(model.out_dir)
    if refresh or not tasks_scored_from(tasks_path, digest):
        command = pass_plan["commands"]["tasks"]
        _run_step(runner, command, what=f"{model.name}'s five-task pass")
        ran["tasks"] = {"card": tasks_path, "command": list(command)}
    else:
        ran["tasks"] = {"card": tasks_path, "skipped": "already-scored"}

    cards = {name: str(card_path(model, name))
             for name in sorted(base.retrieval_cards)}
    stale = [name for name, path in cards.items()
             if refresh or not scored_from(path, digest)]
    if stale:
        # One invocation writes all three cards, so a single stale card
        # rescores the set rather than leaving two of them from other bytes.
        command = pass_plan["commands"]["retrieval"]
        _run_step(runner, command, what=f"{model.name}'s retrieval pass")
        ran["retrieval"] = {"cards": cards, "command": list(command),
                            "rescored": stale}
    else:
        ran["retrieval"] = {"cards": cards, "skipped": "already-scored"}

    return {"model": model.name, "checkpoint_sha256": digest, "config": config,
            "retrieval_settings": pass_plan["retrieval"].to_dict(),
            "tasks_settings": dict(pass_plan["tasks"]),
            "bpb": bpb, "execution": execution["execution"], "ran": ran}


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
        # The gate's code clause is a 5% improvement in a mixture-weighted mean
        # over the holdout's languages. Reported beside it, because at 250M
        # tokens that mean moved -23.6% overall and -2.3% on Python, and the two
        # execution benchmarks in the same verdict are Python-only.
        "code_bpb_by_source": pair_by_source(
            base.details["code_bpb_by_source"],
            branch.details["code_bpb_by_source"],
            measured_name=branch.score.name),
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


# ------------------------------------------------------ the stop's evidence ---
#
# A gate that returns no has done its job, and nothing below may change that
# answer. What it cannot do is say how to *read* the no, and for a branch this
# program stops rather than continues, that reading is what the final report
# carries forward.
#
# The two clauses that stopped the 1B branch fail for different reasons and want
# different evidence. Retrieval is 100 binary items per depth, so an 8-point drop
# is 8 items and the aggregate cannot say whether those 8 are one-sided or churn
# in both directions -- which is the difference between "code training damaged
# retrieval" and "this measurement cannot tell". General BPB is a
# mixture-weighted mean over sources, so a 2.26% regression can be one source
# moving or both, and the verdict already pairs the *code* mean by source for
# exactly that reason.
#
# Both are reported and neither is decisive, the same stance `execution_moves`
# takes with `moved`. The thresholds are preregistered, the numbers are in, and
# a paired test run after the fact is evidence about the finding, never a second
# chance at the gate.

def paired_retrieval_evidence(base: Scorecard,
                              branch: Scorecard) -> Dict[str, dict]:
    """Per-depth discordant counts and a McNemar p for one retrieval task.

    The pairing is the gate's own: `assert_retrieval_paired` refuses two cards
    that did not score the same items, and each item is then matched by `id`
    rather than by position, so the counts here are about the same needles the
    gate differenced. Keys are `retrieval_scores`' keys, so a row of this table
    sits beside the drop that produced it without a translation step.
    """

    assert_retrieval_paired(base, branch)
    if base.items is None or branch.items is None:
        raise BranchScoringError(
            f"{base.name}: a paired test needs both item sidecars; the "
            f"aggregate alone cannot say which items disagreed")

    branch_by_id = {item["id"]: item for item in branch.items}
    grouped: Dict[str, Tuple[List[int], List[int]]] = {}
    for item in base.items:
        other = branch_by_id.get(item["id"])
        if other is None:
            raise BranchScoringError(
                f"{base.name}: item {item['id']!r} is in the base's sidecar and "
                f"not the branch's; pairing what is left would compare two "
                f"different item sets")
        left, right = grouped.setdefault(f"{base.name}:d{item['depth']}",
                                         ([], []))
        left.append(int(item["correct"]))
        right.append(int(other["correct"]))

    evidence: Dict[str, dict] = {}
    for key, (left, right) in sorted(grouped.items()):
        result = mcnemar(left, right)
        evidence[key] = {
            "n": len(left),
            "base_correct": sum(left),
            "branch_correct": sum(right),
            # `mcnemar` labels its discordants by argument order; named here so
            # a reader never has to recover which side b01 was.
            "base_only": result["b01"],
            "branch_only": result["b10"],
            "n_discordant": result["n_discordant"],
            # The gate's sign convention: positive is the branch doing worse.
            # `mcnemar` reports branch-minus-base, so this is its negation.
            "drop_points": -result["diff_pts"],
            "p": result["p"],
            "resolved_at_p05": bool(result["p"] < 0.05),
            # Below ~10 disagreements the normal approximation behind `p` is
            # thin, as `scripts/mcnemar.py` says of its own output. Flagged
            # rather than withheld: the counts are still the finding.
            "thin": bool(result["n_discordant"] < 10),
        }
    return evidence


def build_stop_record(base: CollectedScore, branch: CollectedScore) -> dict:
    """The gate's no, the clauses that produced it, and how to read them.

    Built from the same cards as the verdict rather than from the verdict file,
    so the record cannot describe a gate result the evidence on disk does not
    produce. A branch the gate *continued* is refused: a stop record for it
    would misreport the program's own decision, and the file outlives the
    session that wrote it.
    """

    verdict = build_branch_verdict(base, branch)
    if verdict["continue"]:
        raise BranchScoringError(
            f"the 1B gate continues {branch.score.name}: {verdict['reason']}. "
            f"There is no stop to record.")

    failed = [check for check in verdict["gate"]["checks"]
              if not check["passed"]]
    retrieval: Dict[str, dict] = {}
    for name, card in sorted(base.retrieval_cards.items()):
        retrieval.update(paired_retrieval_evidence(
            card, branch.retrieval_cards[name]))

    try:
        general_by_source = pair_by_source(
            per_source_bpb(load_scorecard(base.cards[GENERAL_CARD])),
            per_source_bpb(load_scorecard(branch.cards[GENERAL_CARD])),
            measured_name=branch.score.name)
    except (ScorecardError, ProbeScoringError, KeyError, ValueError) as exc:
        raise BranchScoringError(
            f"the general-replay regression cannot be attributed to a source: "
            f"{exc}") from exc

    return {
        "schema": 1,
        "decision": "stop",
        "branch": branch.score.name,
        "reason": verdict["reason"],
        "reading": ("the gate is a preregistered threshold and has returned "
                    "its answer; everything under `evidence` says how to read "
                    "that answer and decides nothing"),
        "failed": failed,
        "evidence": {
            "retrieval_paired": retrieval,
            "general_bpb_by_source": general_by_source,
        },
        "models": verdict["models"],
        "verdict": {"gate": verdict["gate"]["gate"],
                    "continue": verdict["continue"],
                    "failed": verdict["gate"]["failed"]},
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


def _load_mixture(path) -> dict:
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchScoringError(f"cannot read the mixture record {path}: "
                                 f"{exc}") from exc
    if not isinstance(record, dict) or not record.get("holdout_root"):
        raise BranchScoringError(
            f"{path} names no holdout_root; the branch's two BPB cards are "
            f"aggregated over its sources")
    return record


def _print_pass(model: ScoredModel, base: CollectedScore, a) -> None:
    """What `score` would run for this branch, and what fixed each setting."""

    try:
        plan = branch_pass_plan(model, base, device=a.device,
                                n_queries=a.n_queries)
    except (BranchScoringError, ProbeScoringError) as exc:
        print(f"\nthe branch's pass cannot be configured: {exc}")
        return
    settings = plan["retrieval"]
    print(f"\n=== the branch's scoring pass ({model.name}) ===")
    print(f"  config {plan['config']}, execution seed "
          f"{plan['execution_seed']}, task limit "
          f"{plan['tasks']['task_limit'] or 'full splits'}")
    print(f"  retrieval {json.dumps(settings.to_dict(), sort_keys=True)}")
    for label, command in plan["commands"].items():
        print(f"  {label:16s} {' '.join(str(part) for part in command)}")


def _plan(a) -> int:
    base, branch = _models(a)
    collected_base: Optional[CollectedScore] = None
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
        if is_base:
            collected_base = collected
        print(f"  checkpoint {collected.sha256[:12]}, "
              f"code BPB {collected.score.code_bpb:.4f}, "
              f"general BPB {collected.score.general_bpb:.4f}, "
              f"five-task {collected.score.five_task_mean:.2f}, "
              f"{len(collected.score.retrieval)} retrieval depth(s)")
        for source, value in sorted(
                collected.details["code_bpb_by_source"].items()):
            print(f"      code BPB {source:43s} {value['bpb']:.5f} "
                  f"(weight {value['weight']:.3f})")
        for name, entry in harness_constraints(collected).items():
            print(f"      {name:24s} {json.dumps(entry, sort_keys=True)}")
    if collected_base is not None:
        _print_pass(branch, collected_base, a)
    print("\nboth sides are ready" if ready else
          "\nnot ready: the branch is scored by `score`, which writes the cards "
          "marked M above with the harness the base's own cards fix")
    return 0 if ready else 1


def _score(a) -> int:
    base, branch = _models(a)
    try:
        collected_base = _collect_args(a, base, is_base=True)
        record = _load_mixture(a.mixture_record)
        outcome = score_branch(
            branch, base=collected_base,
            bpb_plan=scoring_plan(record, record["holdout_root"]),
            device=a.device, batch_size=a.batch_size, n_queries=a.n_queries,
            total_tokens=a.total_tokens, run_dir=a.run_dir, refresh=a.refresh)
    except (BranchScoringError, ProbeGateError, ProbeScoringError,
            ScorecardError, OSError, ValueError, KeyError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, indent=2, sort_keys=True, default=str),
          flush=True)
    # The gate, on the cards this pass just wrote -- including the pairing
    # checks, which are the only place a scored branch is proved comparable.
    return _verdict(a)


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


def _stop_record(a) -> int:
    base, branch = _models(a)
    try:
        record = build_stop_record(_collect_args(a, base, is_base=True),
                                   _collect_args(a, branch, is_base=False))
    except (BranchScoringError, ProbeGateError, ProbeScoringError,
            ScorecardError, OSError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    _write_json(a.json_out, record)
    _print_stop_record(record)
    print(f"\nwrote {a.json_out}")
    return 0


def _print_stop_record(record: dict) -> None:
    print(f"\n{record['branch']}: STOP -- {record['reason']}")
    print(f"\n{len(record['failed'])} clause(s) failed; the evidence below "
          f"decides nothing.")
    print("\nretrieval, paired on the items both models answered:")
    print(f"  {'key':34s} {'n':>4s} {'base':>5s} {'branch':>6s} "
          f"{'base only':>9s} {'branch only':>11s} {'drop':>7s} {'p':>7s}")
    for key, row in sorted(record["evidence"]["retrieval_paired"].items()):
        flag = " thin" if row["thin"] else ""
        print(f"  {key:34s} {row['n']:4d} {row['base_correct']:5d} "
              f"{row['branch_correct']:6d} {row['base_only']:9d} "
              f"{row['branch_only']:11d} {row['drop_points']:+7.2f} "
              f"{row['p']:7.4f}{flag}")
    print("\ngeneral-replay BPB by source (improvement_pct is positive-is-"
          "better, so a retention regression reads negative):")
    for source, value in sorted(
            record["evidence"]["general_bpb_by_source"].items()):
        print(f"  {source:40s} {value['base']:.5f} -> {value['measured']:.5f} "
              f"({value['improvement_pct']:+.2f}%, weight {value['weight']:.3f})")


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
    print("\ncode BPB by source (the mean the code clause is measured on):")
    for source, value in sorted(verdict["code_bpb_by_source"].items()):
        print(f"  {source:52s} {value['base']:.5f} -> {value['measured']:.5f} "
              f"({value['improvement_pct']:+.2f}%, weight {value['weight']:.3f})")


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
                                       "pass would run")
    shared(plan)
    plan.add_argument("--device", default="cuda")
    plan.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    plan.set_defaults(fn=_plan)

    score = sub.add_parser("score", help="write the branch's cards, then run "
                                         "the gate on them")
    shared(score)
    score.add_argument("--mixture-record", default=DEFAULT_MIXTURE_RECORD)
    score.add_argument("--device", default="cuda")
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument(
        "--n-queries", type=int, default=DEFAULT_N_QUERIES,
        help="MQAR queries per item; the only retrieval setting not read off "
             "the base's cards, and checked against its items before anything "
             "is spent")
    score.add_argument(
        "--total-tokens", type=int, default=BRANCH_TOKENS,
        help="refuse to score until the run has trained this budget; 0 scores "
             "whatever is on disk")
    score.add_argument(
        "--run-dir", default=None,
        help="where the branch's metrics.jsonl is (default: beside its "
             "checkpoint)")
    score.add_argument("--refresh", action="store_true",
                       help="re-measure even when a card already scores these "
                            "exact bytes")
    score.add_argument("--json-out", default=DEFAULT_VERDICT)
    score.set_defaults(fn=_score)

    verdict = sub.add_parser("verdict", help="the 1B gate from existing cards")
    shared(verdict)
    verdict.add_argument("--json-out", default=DEFAULT_VERDICT)
    verdict.set_defaults(fn=_verdict)

    stop = sub.add_parser("stop-record",
                          help="a stopped branch's clauses and the paired "
                               "evidence for how to read them")
    shared(stop)
    stop.add_argument("--json-out", default=DEFAULT_STOP_RECORD)
    stop.set_defaults(fn=_stop_record)

    a = p.parse_args(argv)
    if a.base_retrieval is None:
        a.base_retrieval = default_retrieval_paths(DEFAULT_BASE_RETRIEVAL_DIR)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(_cli())
