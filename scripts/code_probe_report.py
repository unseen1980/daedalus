"""Score phase 8's finished 250M probe arms, and run the gate on the result.

    python scripts/code_probe_report.py plan
    python scripts/code_probe_report.py score --device cuda
    python scripts/code_probe_report.py verdict --json-out runs/code-probes/verdict.json

`scripts/code_probes.py` trains the three arms and stops there. Its own report
carries what the trainer saw -- loss, tokens, the in-run `val_bpb` -- and says in
its docstring that the gate's numbers come from somewhere else. This is that
somewhere else, and the split is the same one phase 6 uses between
`architecture_sweep.py` and `architecture_report.py`: training writes weights,
scoring reads them back with a full pass and a fixed harness, and the gate reads
only scorecards.

**Two BPB numbers, never one.** The preregistered gate is a trade -- code BPB
down, general BPB held -- and a single blended figure is the one number that
cannot express it, because a code gain and a replay regression move it in
opposite directions and it reports their sum. So each arm gets a `code-bpb` card
over the holdout's code sources and a `general-bpb` card over its general-replay
sources, weighted within each bucket by the shares the arm actually trained on.
The buckets come from the mixture record's own `buckets` block rather than from
a name prefix: the record is what composed the corpus, and a second opinion
about which source is code would be a second corpus definition free to drift
from the first.

**Every code aggregate carries its per-source breakdown.** The code card's
`bpb` is a mixture-weighted mean over the holdout's languages, and phase 8's
first two arms moved it by -23.6% while moving Python -- 55% of the corpus by
design -- by -2.3%; TypeScript moved -63.6%. Both execution benchmarks are
Python-only, so the aggregate and the pass@1 it sits beside are measuring
different languages. The verdict therefore reports `code_bpb_by_source`
wherever it reports `code_bpb`, and `_print_verdict` prints them together, so
the split is the number rather than a caveat added under a headline.

**`data/holdout/stack-edu-python` is not general retention.** The mixture record
says so in its own caveats, and the bucket split honours it -- that source is
served by the 65% code bucket and drawn from the same dataset the code corpus
streams, so scoring it as replay would credit code training as general
retention. It is not in this mixture's holdout at all; the check exists so a
later mixture that adds it cannot quietly change what "general BPB" means.

**Re-entrant, keyed on the checkpoint digest.** Scoring four checkpoints over
two benchmarks and two holdout buckets is a couple of GPU hours, and a session
that ends mid-pass must cost only the model it was on. A card is reused only
when the bytes it scored are the bytes on disk now, so a re-scored arm is never
answered with a previous arm's number.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from daedalus.code_gates import (ProbeGateError, ProbeScore,  # noqa: E402
                                 execution_score, probes_250m_verdict)
from daedalus.scorecard import (ArtifactRef, ScorecardError,  # noqa: E402
                                load_scorecard, sha256_file)
from scripts.bpb_eval import (_git_short_sha, discover_sources,  # noqa: E402
                              per_source_bpb, run_bpb_eval)
from scripts.code_probes import DEFAULT_MIXTURE_RECORD, DEFAULT_REPORT  # noqa: E402


# ============================================================ preregistered ===
# Named here, in the commit that precedes the first score. A bound chosen after
# the numbers are on screen is not a bound.

#: The general-BPB regression above which a qualifying arm is not carried into
#: the 1B branch, in percent.
#:
#: `select_on` reads "code BPB and execution pass@1, subject to general
#: retention" and `daedalus/code_gates.py` deliberately stops short of applying
#: the retention half, because no retention bound is preregistered for the
#: *probe* stage -- only for the branch. Something has to be applied here, and
#: inventing a probe-specific number after three arms have been measured is the
#: one thing that must not happen. So this borrows the branch's own bound: an
#: arm whose general BPB has already regressed further at 250M tokens than the
#: 1B gate will permit at 1B is not a candidate to spend 1B tokens on. It is a
#: reused preregistered figure rather than a new one, and it can only ever
#: remove a candidate the loose gate already qualified.
PROBE_RETENTION_REGRESSION_PCT_MAX = 1.5

#: The mixture buckets each BPB card aggregates over. Both names are keys of the
#: mixture record's `buckets` block; the third, `technical`, has no source in
#: this holdout and is reported as unscored rather than folded into either.
CODE_BUCKET = "code"
GENERAL_BUCKET = "general-replay"

#: Sources that are code no matter which bucket a future mixture files them
#: under. `codeprep.CODE_REPLAY_SOURCES` is the same list for the same reason;
#: it is restated as a *refusal* here because the failure it prevents is one
#: that produces a plausible number rather than an error.
NEVER_GENERAL_RETENTION = ("stack-edu-python",)

#: The execution benchmarks the gate reads, in the order the manifest names them.
DATASETS = ("humaneval-plus", "mbpp-plus")

#: The runtime fields two execution scorecards must agree on before their
#: difference means anything. `code_gates.execution_moves` already refuses a
#: mismatched item count; these are the knobs that change what a pass@1 *is*
#: without changing how many items produced it.
PAIRED_RUNTIME_FIELDS = ("backend", "max_new_tokens")

BASE_MODEL = "base"
DEFAULT_EVAL_ROOT = "runs/eval"
DEFAULT_BASE_EVAL_DIR = f"{DEFAULT_EVAL_ROOT}/code-base"
DEFAULT_VERDICT = "runs/code-probes/verdict.json"
CODE_CARD = "code-bpb"
GENERAL_CARD = "general-bpb"
CONFIG = "daedalus-150m"


class ProbeScoringError(ValueError):
    """Raised when the inputs to this pass are not the gate's evidence."""


# ------------------------------------------------------------------ buckets ---

def bucket_weights(record: dict, bucket: str,
                   present: Sequence[str]) -> dict:
    """One bucket's shares, restricted to the sources the holdout actually has.

    Renormalized *within* the bucket, because the bucket is the aggregate: a
    general-replay holdout carrying two of the bucket's six sources measures the
    replay distribution over those two, and saying so is the whole difference
    between a number and a claim. `share_covered` is how much of the bucket the
    holdout represents, and it travels into the scorecard so a reader never has
    to reconstruct it.
    """

    buckets = record.get("buckets")
    if not isinstance(buckets, dict) or bucket not in buckets:
        raise ProbeScoringError(
            f"the mixture record has no {bucket!r} bucket; it has "
            f"{sorted(buckets or ())}. The buckets block is what defines which "
            f"holdout source is code and which is replay.")
    members = {name: float(share) for name, share in buckets[bucket].items()}
    if not members:
        raise ProbeScoringError(f"bucket {bucket!r} is empty in the mixture record")

    have = set(present)
    kept = {name: share for name, share in members.items() if name in have}
    absent = sorted(name for name in members if name not in have)
    if not kept:
        raise ProbeScoringError(
            f"no source of the {bucket!r} bucket is in the holdout (it wants "
            f"{sorted(members)}); this aggregate would be measured over nothing")
    if bucket == GENERAL_BUCKET:
        # A code source filed as replay reports code training as retention, and
        # the number it produces is finite and plausible. The mixture record
        # already excludes this one by name; the refusal is here so a later
        # mixture that stops excluding it fails loudly rather than quietly.
        smuggled = sorted(set(kept) & set(NEVER_GENERAL_RETENTION))
        if smuggled:
            raise ProbeScoringError(
                f"{', '.join(smuggled)} is in the {bucket!r} bucket, but it is "
                f"code from the same dataset the code corpus streams; general "
                f"retention measured over it would credit code training as "
                f"retention")
    total = sum(kept.values())
    covered = total / sum(members.values())
    return {
        "bucket": bucket,
        "weights": {name: share / total for name, share in sorted(kept.items())},
        "sources": sorted(kept),
        "absent": absent,
        "share_covered": covered,
    }


def scoring_plan(record: dict, holdout_root: str) -> dict:
    """Which holdout sources feed which aggregate, and what feeds neither."""

    present = sorted(discover_sources(holdout_root))
    code = bucket_weights(record, CODE_BUCKET, present)
    general = bucket_weights(record, GENERAL_BUCKET, present)
    overlap = sorted(set(code["sources"]) & set(general["sources"]))
    if overlap:
        raise ProbeScoringError(
            f"{', '.join(overlap)} is in both the code and general-replay "
            f"buckets; one source cannot be both sides of this trade")
    claimed = set(code["sources"]) | set(general["sources"])
    return {
        "holdout_root": str(holdout_root),
        "present": present,
        CODE_CARD: code,
        GENERAL_CARD: general,
        # Not an error: the technical bucket has no holdout here, and a source
        # that belongs to neither aggregate is reported rather than folded into
        # whichever one it happens to resemble.
        "unscored": [name for name in present if name not in claimed],
        "caveats": list(record.get("caveats") or ()),
    }


# ------------------------------------------------------------------- inputs ---

@dataclass(frozen=True)
class ScoredModel:
    """One checkpoint this pass measures, and where its cards live."""

    name: str
    checkpoint: str
    out_dir: str

    @property
    def is_base(self) -> bool:
        return self.name == BASE_MODEL


def completed_arms(report: dict) -> List[dict]:
    """The gate report's arms, or a refusal naming what is not gate evidence.

    A smoke report has the same shape and different weights behind it, so the
    check is on the report's own `gate`/`smoke` fields -- which
    `code_probes.sweep` sets from the arms it actually ran, not from what it was
    asked to run.
    """

    if report.get("smoke") or report.get("gate") != "probes_250m":
        raise ProbeScoringError(
            f"report names gate {report.get('gate')!r} (smoke="
            f"{bool(report.get('smoke'))}); this pass scores the preregistered "
            f"250M probes and nothing else")
    arms = list(report.get("arms") or ())
    if not arms:
        raise ProbeScoringError("the probe report has no arms")
    broken = []
    for entry in arms:
        name = (entry.get("arm") or {}).get("name", "<unnamed>")
        if entry.get("error"):
            broken.append(f"{name}: {entry['error']}")
        elif not (entry.get("summary") or {}).get("complete"):
            broken.append(f"{name}: did not reach its budget")
    if broken:
        raise ProbeScoringError(
            "these arms are not scoreable evidence: " + "; ".join(broken))
    return arms


def models_for(report: dict, *, base_checkpoint: str,
               base_out_dir: str = DEFAULT_BASE_EVAL_DIR,
               eval_root: str = DEFAULT_EVAL_ROOT) -> List[ScoredModel]:
    """The base first, then each arm, each with its own card directory.

    Base first because every arm's number is a difference against it: a pass
    that dies partway through has then produced the one measurement without
    which none of the others mean anything.
    """

    models = [ScoredModel(name=BASE_MODEL, checkpoint=str(base_checkpoint),
                          out_dir=str(base_out_dir))]
    for entry in completed_arms(report):
        name = entry["arm"]["name"]
        run_dir = entry.get("run_dir") or os.path.join("runs", name)
        models.append(ScoredModel(
            name=name,
            checkpoint=os.path.join(run_dir, "checkpoint.pt"),
            out_dir=os.path.join(eval_root, name)))
    return models


def card_path(model: ScoredModel, name: str) -> Path:
    return Path(model.out_dir) / f"{name}.json"


def scored_from(path, checkpoint_sha: str) -> bool:
    """True when `path` already scores exactly these bytes.

    Keyed on the digest rather than on the file existing, so a re-scored model
    is skipped only when the thing that was scored is the thing that is there
    now -- an arm retrained after a repair must not be answered with the old
    arm's card.
    """

    path = Path(path)
    if not path.exists():
        return False
    try:
        card = load_scorecard(path)
    except (ScorecardError, KeyError, ValueError, OSError):
        return False
    return card.provenance.artifact.sha256 == checkpoint_sha


# ------------------------------------------------------------------ scoring ---

def make_checkpoint_bpb_fn(checkpoint, *, config: str, device: str,
                           seq_len: int, batch_size: int
                           ) -> Callable[[Path], float]:
    """Load one checkpoint and return its per-source full-pass BPB callable.

    Imports live inside so this module's plan and gate logic stay importable --
    and testable -- without torch.
    """

    from daedalus.config import PRESETS
    from daedalus.data import assert_shards_tokenizer, get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb
    from train import load_checkpoint

    tokenizer = get_tokenizer(None)
    model = Daedalus(PRESETS[config]).to(device)
    load_checkpoint(str(checkpoint), model, map_location=device)
    model.eval()

    def bpb_fn(source_dir: Path) -> float:
        # The tokenizer decodes the held-out ids to count the bytes BPB is per,
        # so it is part of the measurement. Checked against this source's own
        # manifest rather than assumed: a disagreement produces a finite,
        # plausible, wrong number, and this is the number the gate reads.
        assert_shards_tokenizer(str(source_dir), tokenizer, None)
        # max_batches=None is the full pass the gate requires.
        return evaluate_bpb(model, str(source_dir), seq_len, tokenizer, device,
                            batch_size=batch_size, max_batches=None)

    return bpb_fn


def _release(device: str) -> None:
    """Drop one model's weights before the next lands on the same card."""
    import gc

    gc.collect()
    if not device.startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:                                       # pragma: no cover
        pass


def score_bpb(model: ScoredModel, *, plan: dict, config: str = CONFIG,
              device: str = "cuda", seq_len: int = 2048, batch_size: int = 8,
              seed: int = 20260824, refresh: bool = False,
              bpb_factory: Callable[..., Callable[[Path], float]]
              = make_checkpoint_bpb_fn) -> dict:
    """Full-pass code and general BPB for one checkpoint, as two scorecards."""

    checkpoint = Path(model.checkpoint)
    if not checkpoint.exists():
        raise ProbeScoringError(
            f"{model.name} has no checkpoint at {checkpoint}; it either never "
            f"ran or its run directory was moved")
    digest = sha256_file(checkpoint)
    wanted = [name for name in (CODE_CARD, GENERAL_CARD)
              if refresh or not scored_from(card_path(model, name), digest)]
    if not wanted:
        return {"model": model.name, "skipped": "already-scored",
                "checkpoint_sha256": digest,
                "cards": {name: str(card_path(model, name))
                          for name in (CODE_CARD, GENERAL_CARD)}}

    bpb_fn = bpb_factory(checkpoint, config=config, device=device,
                         seq_len=seq_len, batch_size=batch_size)
    written: Dict[str, str] = {}
    try:
        for name in wanted:
            spec = plan[name]
            paths = run_bpb_eval(
                name=name, holdout_root=plan["holdout_root"],
                out_dir=model.out_dir,
                artifact=ArtifactRef(path=str(checkpoint), sha256=digest,
                                     kind="checkpoint", config=config),
                tokenizer_ref=ArtifactRef(path="<smollm2-default>",
                                          sha256="0" * 64, kind="tokenizer"),
                seed=seed, git_sha=_git_short_sha(), bpb_fn=bpb_fn,
                max_batches=None, sources=spec["sources"],
                weights=spec["weights"],
                runtime={"device": device, "seq_len": seq_len,
                         "batch_size": batch_size},
                details_extra={"phase": "phase8-code-probes", "model": model.name,
                               "bucket": spec["bucket"],
                               "bucket_share_covered": spec["share_covered"],
                               "bucket_absent": spec["absent"],
                               "caveats": plan["caveats"]})
            written[name] = str(paths["scorecard"])
    finally:
        del bpb_fn
        _release(device)
    return {"model": model.name, "checkpoint_sha256": digest, "cards": written}


def execution_command(model: ScoredModel, dataset: str, *, device: str,
                      config: str = CONFIG, seed: int = 20260824,
                      python: str = "python") -> List[str]:
    """The `scripts/code_eval.py` argv for one (model, benchmark) pair.

    Built as data rather than typed per model for the same reason the arms'
    train argv is: the base was scored by one of these commands weeks of GPU
    time ago, every arm is a difference against it, and a harness knob that
    differs between the two sides is a difference in the measurement rather
    than in the model. `assert_paired` checks the cards agree afterwards.
    """

    return [python, "scripts/code_eval.py", "--dataset", dataset,
            "--backend", "torch", "--checkpoint", model.checkpoint,
            "--config", config, "--device", device, "--seed", str(seed),
            "--out-dir", model.out_dir]


def score_execution(model: ScoredModel, *, datasets: Sequence[str] = DATASETS,
                    device: str = "cuda", config: str = CONFIG,
                    seed: int = 20260824, refresh: bool = False,
                    runner: Callable[[Sequence[str]], int] = None) -> dict:
    """Run each execution benchmark this model does not already have a card for.

    A subprocess, not an import: `code_eval` executes generated code under a
    resource-limited sandbox and one benchmark's harness state must not outlive
    it into the next model's pass.
    """

    def _run(command: Sequence[str]) -> int:
        return subprocess.run(list(command), cwd=_ROOT).returncode

    runner = runner or _run
    digest = sha256_file(model.checkpoint)
    results: Dict[str, dict] = {}
    for dataset in datasets:
        path = card_path(model, dataset)
        if not refresh and scored_from(path, digest):
            results[dataset] = {"skipped": "already-scored", "card": str(path)}
            continue
        command = execution_command(model, dataset, device=device,
                                    config=config, seed=seed)
        code = runner(command)
        if code != 0:
            raise ProbeScoringError(
                f"{model.name} {dataset} exited {code}; command was "
                f"{' '.join(command)}")
        results[dataset] = {"card": str(path), "command": command}
    return {"model": model.name, "checkpoint_sha256": digest,
            "execution": results}


# -------------------------------------------------------------------- gate ---

def assert_paired(base_card, arm_card) -> None:
    """Refuse two execution cards produced by different harnesses.

    Item count is checked by the gate itself. This checks the knobs that change
    what a pass@1 *means* without changing how many items produced it: a longer
    generation budget or a different backend on one side is a difference in the
    harness that the gate would read as a difference in the model.
    """

    for field in PAIRED_RUNTIME_FIELDS:
        left = base_card.provenance.runtime.get(field)
        right = arm_card.provenance.runtime.get(field)
        if left != right:
            raise ProbeScoringError(
                f"{arm_card.name}: base was scored with {field}={left!r} and "
                f"the arm with {right!r}; these are not comparable")
    if base_card.provenance.seed != arm_card.provenance.seed:
        raise ProbeScoringError(
            f"{arm_card.name}: base seed {base_card.provenance.seed} != arm "
            f"seed {arm_card.provenance.seed}")


def read_probe_score(model: ScoredModel,
                     datasets: Sequence[str] = DATASETS) -> ProbeScore:
    """One model's gate inputs, read back from its cards."""

    code = load_scorecard(card_path(model, CODE_CARD))
    if code.kind != "bpb":
        raise ProbeScoringError(
            f"{model.name}: {card_path(model, CODE_CARD)} is kind {code.kind!r}, "
            f"not 'bpb'")
    execution = {dataset: load_scorecard(card_path(model, dataset))
                 for dataset in datasets}
    return ProbeScore(name=model.name, code_bpb=float(code.metrics["bpb"]),
                      execution={name: execution_score(card)
                                 for name, card in execution.items()})


def pair_by_source(base_table: Dict[str, dict], measured_table: Dict[str, dict],
                   *, measured_name: str = "the measured model") -> dict:
    """Two per-source BPB tables as one paired breakdown, source by source.

    Keyed `base`/`measured` rather than `base`/`arm` because the 1B branch gate
    reads the same table for a model that is not an arm, and a key that means
    "arm" in one verdict and "branch" in the other is a key a reader has to
    translate.

    A source on one side only is refused rather than dropped. The aggregate is a
    weighted mean over whatever the card measured, so two aggregates over
    different source sets are already not comparable -- and a breakdown that
    silently omitted the difference would be the one place that fact was
    visible.
    """

    if sorted(base_table) != sorted(measured_table):
        raise ProbeScoringError(
            f"the base scored code sources {sorted(base_table)} and "
            f"{measured_name} scored {sorted(measured_table)}; these aggregates "
            f"are means over different holdouts and their difference is not a "
            f"difference between models")
    paired = {}
    for name in sorted(base_table):
        base_bpb = float(base_table[name]["bpb"])
        measured_bpb = float(measured_table[name]["bpb"])
        if base_bpb <= 0:
            raise ProbeScoringError(
                f"base BPB for source {name!r} is {base_bpb!r}; a relative "
                f"improvement against it is undefined")
        paired[name] = {
            "base": base_bpb,
            "measured": measured_bpb,
            "improvement_pct": (base_bpb - measured_bpb) / base_bpb * 100.0,
            "weight": measured_table[name]["weight"],
            "tokens": measured_table[name]["tokens"],
        }
    return paired


def code_bpb_by_source(base_model: ScoredModel, measured: ScoredModel) -> dict:
    """One model's code BPB per holdout source, paired against the base's own."""

    return pair_by_source(per_source_bpb(load_scorecard(card_path(base_model,
                                                                  CODE_CARD))),
                          per_source_bpb(load_scorecard(card_path(measured,
                                                                  CODE_CARD))),
                          measured_name=measured.name)


def read_general_bpb(model: ScoredModel) -> dict:
    """One model's general-replay BPB, with the coverage it was measured over."""

    card = load_scorecard(card_path(model, GENERAL_CARD))
    return {"bpb": float(card.metrics["bpb"]),
            "share_covered": card.details.get("bucket_share_covered"),
            "sources": card.details.get("sources_requested"),
            "card": str(card_path(model, GENERAL_CARD))}


def retention(base: dict, arm: dict) -> dict:
    """Percent *regression* in general BPB. Positive is worse."""

    if base["bpb"] <= 0:
        raise ProbeScoringError(
            f"base general BPB is {base['bpb']!r}; a relative regression "
            f"against it is undefined")
    pct = (arm["bpb"] - base["bpb"]) / base["bpb"] * 100.0
    return {"general_bpb": arm["bpb"], "general_bpb_base": base["bpb"],
            "regression_pct": pct,
            "threshold_pct": PROBE_RETENTION_REGRESSION_PCT_MAX,
            "retained": bool(pct <= PROBE_RETENTION_REGRESSION_PCT_MAX),
            "share_covered": arm["share_covered"]}


def build_verdict(models: Sequence[ScoredModel], *,
                  datasets: Sequence[str] = DATASETS,
                  plan: Optional[dict] = None) -> dict:
    """The gate, then retention, then the arm the 1B branch should start from."""

    base = next((model for model in models if model.is_base), None)
    if base is None:
        raise ProbeScoringError("no base model was scored; every arm's number "
                                "is a difference against it")
    arms = [model for model in models if not model.is_base]
    if not arms:
        raise ProbeScoringError("no arms were scored")

    base_score = read_probe_score(base, datasets)
    base_cards = {dataset: load_scorecard(card_path(base, dataset))
                  for dataset in datasets}
    arm_scores = []
    for arm in arms:
        for dataset in datasets:
            assert_paired(base_cards[dataset], load_scorecard(card_path(arm, dataset)))
        arm_scores.append(read_probe_score(arm, datasets))

    gate = probes_250m_verdict(base_score, arm_scores)

    base_general = read_general_bpb(base)
    retention_by_arm = {arm.name: retention(base_general, read_general_bpb(arm))
                        for arm in arms}
    # Beside `code_bpb` and `code_bpb_improvement_pct` in the same entry, not
    # under them: the aggregate is a 55%-Python mean and phase 8's probes moved
    # it almost entirely through TypeScript. A reader who sees only the
    # aggregate concludes something the breakdown does not support.
    by_source = {arm.name: code_bpb_by_source(base, arm) for arm in arms}
    for entry in gate["arms"]:
        entry["retention"] = retention_by_arm[entry["arm"]]
        entry["code_bpb_by_source"] = by_source[entry["arm"]]

    # Applied only to arms the gate already qualified, and only ever to remove
    # one. See PROBE_RETENTION_REGRESSION_PCT_MAX for why this bound and not a
    # new one.
    ranked = [name for name in gate["ranking"]]
    rejected = [name for name in ranked if not retention_by_arm[name]["retained"]]
    selected = next((name for name in ranked
                     if retention_by_arm[name]["retained"]), None)
    return {
        "schema": 1,
        "gate": gate,
        "code_bpb_base_by_source": per_source_bpb(
            load_scorecard(card_path(base, CODE_CARD))),
        "general_bpb_base": base_general,
        "retention": retention_by_arm,
        "retention_threshold_pct": PROBE_RETENTION_REGRESSION_PCT_MAX,
        "retention_rejected": rejected,
        "selected": selected,
        "reason": (f"{selected} is the lowest-code-BPB qualifying arm that "
                   f"holds general BPB within "
                   f"{PROBE_RETENTION_REGRESSION_PCT_MAX:g}%"
                   if selected else
                   ("no qualifying arm held general retention"
                    if ranked else gate["reason"])),
        "continue": bool(gate["continue"] and selected),
        "models": {model.name: {"checkpoint": model.checkpoint,
                                "cards": str(model.out_dir)}
                   for model in models},
        "plan": plan,
    }


# --------------------------------------------------------------------- cli ---

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


def _load(path: str, what: str) -> dict:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeScoringError(f"cannot read {what} {path}: {exc}") from exc


def _inputs(a) -> tuple:
    record = _load(a.mixture_record, "mixture record")
    report = _load(a.probes, "probe report")
    plan = scoring_plan(record, record["holdout_root"])
    models = models_for(report, base_checkpoint=a.base_checkpoint,
                        base_out_dir=a.base_out_dir, eval_root=a.eval_root)
    return record, report, plan, models


def _plan(a) -> int:
    try:
        _, _, plan, models = _inputs(a)
    except (ProbeScoringError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    for name in (CODE_CARD, GENERAL_CARD):
        spec = plan[name]
        print(f"{name}: {len(spec['sources'])} source(s), "
              f"{spec['share_covered'] * 100:.1f}% of the {spec['bucket']} bucket")
        for source, weight in spec["weights"].items():
            print(f"    {source:48s} {weight:.4f}")
        if spec["absent"]:
            print(f"    absent from the holdout: {', '.join(spec['absent'])}")
    if plan["unscored"]:
        print(f"scored by neither aggregate: {', '.join(plan['unscored'])}")
    print(f"\nmodels ({len(models)}):")
    for model in models:
        print(f"    {model.name:24s} {model.checkpoint} -> {model.out_dir}")
    return 0


def _score(a) -> int:
    try:
        _, _, plan, models = _inputs(a)
    except (ProbeScoringError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    scored = []
    for model in models:
        print(f"=== {model.name} ===", flush=True)
        try:
            bpb = score_bpb(model, plan=plan, device=a.device,
                            batch_size=a.batch_size, refresh=a.refresh)
            execution = score_execution(model, device=a.device,
                                        refresh=a.refresh)
        except (ProbeScoringError, OSError, ValueError) as exc:
            # Recorded, not raised: one model failing must not lose the passes
            # that already landed. The non-zero exit below is what the
            # controller sees.
            print(f"FAILED {model.name}: {exc!r}", file=sys.stderr, flush=True)
            scored.append({"model": model.name, "error": repr(exc)})
            continue
        scored.append({**bpb, **execution})
    failed = [entry for entry in scored if entry.get("error")]
    if failed:
        return 3
    try:
        verdict = build_verdict(models, plan=plan)
    except (ProbeGateError, ProbeScoringError, ScorecardError, KeyError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    _write_json(a.json_out, verdict)
    _print_verdict(verdict)
    print(f"\nwrote {a.json_out}")
    return 0


def _verdict(a) -> int:
    try:
        _, _, plan, models = _inputs(a)
        verdict = build_verdict(models, plan=plan)
    except (ProbeGateError, ProbeScoringError, ScorecardError, ValueError,
            KeyError, FileNotFoundError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    _write_json(a.json_out, verdict)
    _print_verdict(verdict)
    print(f"\nwrote {a.json_out}")
    return 0


def _print_verdict(verdict: dict) -> None:
    gate = verdict["gate"]
    print(f"\nbase code BPB {gate['arms'][0]['code_bpb_base']:.4f}, "
          f"general BPB {verdict['general_bpb_base']['bpb']:.4f}")
    for source, value in sorted(verdict["code_bpb_base_by_source"].items()):
        print(f"      {source:52s} {value['bpb']:.5f} "
              f"(weight {value['weight']:.3f})")
    for entry in gate["arms"]:
        retained = entry["retention"]
        print(f"  {entry['arm']:24s} code {entry['code_bpb']:.4f} "
              f"({entry['code_bpb_improvement_pct']:+.2f}%) "
              f"A={'y' if entry['criterion_a_code_bpb'] else 'n'} "
              f"B={'y' if entry['criterion_b_execution'] else 'n'} "
              f"general {retained['regression_pct']:+.2f}% "
              f"{'retained' if retained['retained'] else 'REGRESSED'}")
        # The aggregate above is a weighted mean; these are what it is a mean
        # of. Printed every time it is, because a gain concentrated in one
        # language is not the same result as an even one.
        for source, value in sorted(entry["code_bpb_by_source"].items()):
            print(f"      {source:52s} {value['measured']:.5f} "
                  f"({value['improvement_pct']:+.2f}%)")
    print(f"gate: {'continue' if gate['continue'] else 'STOP'} -- {gate['reason']}")
    print(f"selected: {verdict['selected'] or 'none'} -- {verdict['reason']}")


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def shared(parser):
        parser.add_argument("--probes", default=DEFAULT_REPORT)
        parser.add_argument("--mixture-record", default=DEFAULT_MIXTURE_RECORD)
        parser.add_argument("--base-checkpoint",
                            default="/root/daedalus/final/hero/checkpoint.pt")
        parser.add_argument("--base-out-dir", default=DEFAULT_BASE_EVAL_DIR)
        parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)

    plan = sub.add_parser("plan", help="which source feeds which aggregate")
    shared(plan)
    plan.set_defaults(fn=_plan)

    score = sub.add_parser("score", help="score every model, then run the gate")
    shared(score)
    score.add_argument("--device", default="cuda")
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--refresh", action="store_true",
                       help="re-measure even when a card already scores these "
                            "exact bytes")
    score.add_argument("--json-out", default=DEFAULT_VERDICT)
    score.set_defaults(fn=_score)

    verdict = sub.add_parser("verdict", help="the gate from existing cards only")
    shared(verdict)
    verdict.add_argument("--json-out", default=DEFAULT_VERDICT)
    verdict.set_defaults(fn=_verdict)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(_cli())
