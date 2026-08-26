"""Phase 7 steps 7 and 8: train the mixture arms that the selection rule reads.

`daedalus/mixture_opt.py` holds the arms, the derivation rule, the floors and the
selection; it costs nothing to run and is committed before any of it is measured.
This module is the half that spends GPU hours: it turns an arm into a `train.py`
invocation, runs it under the supervisor so an interruption continues it, and
records what the sampler actually drew.

Four decisions about how the arms are run, because each is a way a proxy like
this quietly stops measuring the thing it names.

**The probe is phase 4's, re-used rather than re-derived.** 200M tokens at
sequence 1024 in 131,072-token steps, `tok-probe-49152`, Muon 0.02 / Adam 3e-4,
100 warmup steps, 0.8 decay. That is exactly `scripts/tokenizer_lab.py`'s LM
probe, at the shipped vocabulary, and re-using it means the throughput, the
memory headroom and the schedule shape are all measured facts on this box rather
than estimates. It also keeps phase 4 and phase 7 separable: the tokenizer is
held at the shipped one here, and the data is held at one corpus there.

**Every arm shares one data root and one holdout.** A specialist is a weight of
1.0 on its source, not a different `--data-dir`. Arms pointed at different roots
differ in shard files, packing and holdout as well as in mixture, and "identical
apart from the mixture" stops being checkable and becomes a claim about two paths
looking similar.

**An arm whose epoch cap binds is refused, not trained.** `cap_weights_by_epochs`
silently reweights a source that cannot supply its share -- correct for a
production run, and fatal here, because the arm would train on a mixture that is
not the one it is named after and nothing downstream would know. `arm_preflight`
asks the same resolver the loader uses, before the GPU is touched.

**In-run `val_bpb` is not the comparison.** `train.py` weights the holdout by the
mixture each run samples, so the six arms' `val_bpb` columns are six models
scored on six different corpora. The comparison is `score` below, which drives
`scripts/bpb_eval.py` under `mixture_opt.evaluation_weights` -- the same
weighting for every arm, over every held-out window, from the final checkpoint
rather than from a bounded sample taken mid-decay. `val_bpb` stays on as the
divergence signal it is.

Subcommands: `arms`, `shape`, `run`, `sweep`, `score`, `derive`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daedalus.mixture_opt import (MixtureArm, candidate_arms,  # noqa: E402
                                  evaluation_weights, reference_arms,
                                  unrepresented_floored_domains)

#: Where `train.py` puts a run, and so the only place a supervisor may look for
#: its checkpoint. Asked of the trainer in `arm_checkpoint_path` rather than
#: composed here -- see phase 6 for what composing it cost.
RUN_ROOT = "runs"

#: Phase 7's artifacts, beside the headroom curve and the decontamination index.
REPORT_ROOT = "runs/corpus"


@dataclass(frozen=True)
class ProbeShape:
    """The run shape every arm shares. Shared *is* the experiment.

    Every field is identical across arms by construction, so two arms cannot
    differ in batch shape, schedule, learning rate or budget. Only the mixture
    weights change.
    """

    name: str
    config: str
    seq_len: int
    micro_batch: int
    batch_tokens: int
    total_tokens: int
    warmup_steps: int
    decay_frac: float
    muon_lr: float
    adam_lr: float
    max_source_epochs: float
    note: str = ""

    @property
    def steps(self) -> int:
        return -(-int(self.total_tokens) // int(self.batch_tokens))

    @property
    def grad_accum(self) -> int:
        return max(1, round(self.batch_tokens / (self.micro_batch * self.seq_len)))


#: Preregistered. The numbers are `scripts/tokenizer_lab.py`'s LM probe
#: constants, deliberately identical: same shape, same budget, same schedule, at
#: the shipped 49,152 vocabulary. 200,015,872 rather than a round 200M is 1,526
#: whole steps of 131,072 -- a partial final step trains on a truncated batch and
#: makes the step count a rounding artefact rather than a property of the plan.
PROBE = ProbeShape(
    name="probe",
    config="tok-probe-49152",
    seq_len=1024,
    micro_batch=16,
    batch_tokens=131_072,
    total_tokens=200_015_872,
    warmup_steps=100,
    decay_frac=0.8,
    muon_lr=0.02,
    adam_lr=3e-4,
    #: The shipped cap. Left at the production value rather than lowered,
    #: because `arm_preflight` refuses an arm the cap would bind on instead of
    #: quietly training a reweighted one -- so this bounds nothing here, and
    #: changing it would only change which arms are refused.
    max_source_epochs=4.0,
    note="phase 4's LM probe recipe at the shipped vocabulary, varying only the "
         "data mixture",
)

SHAPES = {PROBE.name: PROBE}

STAGES = ("reference", "candidates")


def stage_arms(stage: str, sources: Sequence[str],
               derived: Optional[dict] = None) -> List[MixtureArm]:
    if stage == "reference":
        if derived:
            raise SystemExit(
                "--derived-weights belongs to the candidates stage; the "
                "reference stage is what measures the excess loss it is "
                "derived from")
        return reference_arms(sources)
    if stage == "candidates":
        return candidate_arms(sources, derived)
    raise SystemExit(f"unknown stage {stage!r}; known: {list(STAGES)}")


def discover_sources(data_root) -> List[str]:
    """Every manifest-backed subdirectory of a mixture root, sorted.

    The same detection `train.resolve_mixture` performs, so the arms are built
    over exactly the sources the loader will find. Reading the blueprint instead
    would build arms naming seven sources that are not on this box, and a
    specialist for a source with no shards is an arm that cannot run.
    """
    root = Path(data_root)
    found = sorted(entry.name for entry in root.iterdir()
                   if entry.is_dir() and (entry / "manifest.json").exists())
    if not found:
        raise SystemExit(
            f"no source under {root} has a manifest.json; --data-dir must be a "
            f"mixture root, one subdirectory per source")
    return found


def arm_run_name(arm: MixtureArm, tag: str) -> str:
    return f"mix-{tag}-{arm.name}"


def arm_checkpoint_path(arm: MixtureArm, tag: str,
                        run_root: str = RUN_ROOT) -> Path:
    """The checkpoint this arm writes, asked of `train.py` rather than guessed."""
    from train import TrainArgs, checkpoint_path_for

    args = TrainArgs(run_name=arm_run_name(arm, tag), config=PROBE.config,
                     data_dir="", run_dir=None)
    resolved = Path(checkpoint_path_for(args))
    if run_root != RUN_ROOT:                      # tests may relocate the tree
        resolved = Path(run_root) / arm_run_name(arm, tag) / "checkpoint.pt"
    return resolved


def weight_args(arm: MixtureArm) -> List[str]:
    """`--mixture-weight` pairs, in a fixed source order.

    Sorted, and formatted to a fixed precision, so the same arm produces the
    same argv on every launch. `finished_run` compares commands exactly: a
    dict-ordering difference or a float repr that varied between Python builds
    would make a completed arm look like a different experiment and retrain it
    over its own checkpoint.
    """
    return [item for name in sorted(arm.weights)
            for item in ("--mixture-weight", f"{name}={arm.weights[name]:.6f}")]


def train_command(arm: MixtureArm, *, data_dir: str, run_name: str,
                  shape: ProbeShape = PROBE, device: str = "cuda",
                  val_dir: Optional[str] = None,
                  total_tokens: Optional[int] = None) -> List[str]:
    """The exact `train.py` invocation for one arm.

    Everything except the weights comes from the shape, so two arms cannot
    differ in seed, data order, batch shape, schedule or learning rate.
    Sequence length and tokens per step are flat rather than ramped for the
    reason phases 5 and 6 held theirs flat: a ramp makes the schedule mean
    something different early and late, and the schedule is held precisely so
    the mixture is the variable.
    """
    budget = int(shape.total_tokens if total_tokens is None else total_tokens)
    command = [
        sys.executable, "train.py",
        "--run-name", run_name,
        "--config", shape.config,
        "--data-dir", data_dir,
        "--total-tokens", str(budget),
        "--micro-batch", str(shape.micro_batch),
        "--seq-start", str(shape.seq_len), "--seq-end", str(shape.seq_len),
        "--tok-start", str(shape.batch_tokens),
        "--tok-end", str(shape.batch_tokens),
        "--muon-lr", f"{shape.muon_lr:g}",
        "--adam-lr", f"{shape.adam_lr:g}",
        "--warmup-steps", str(shape.warmup_steps),
        "--decay-frac", f"{shape.decay_frac:g}",
        "--device", device,
        "--hub-repo", "",
        "--no-wandb",
    ]
    command += weight_args(arm)
    if val_dir:
        command += ["--val-dir", val_dir]
    return command


#: Percentage points of L1 skew between an arm's asked-for shares and the shares
#: its sampler will draw. Anything above this and the arm is not the arm.
#: `summarize_mixture` rounds the figure to four decimals, so this is the
#: smallest bound that is not asking about representation error.
MAX_ARM_SKEW_PTS = 1e-3


def arm_preflight(arm: MixtureArm, *, data_dir: str,
                  shape: ProbeShape = PROBE,
                  total_tokens: Optional[int] = None) -> dict:
    """What this arm's sampler will actually draw, before any GPU is touched.

    Three refusals, separate because they are separate failures --
    `capped_sources` cannot tell any of them apart, since it flags a source at
    or past the cap whether or not anything was reweighted.

    **A source the arm names is not under the root.** `resolve_mixture` drops it
    and renormalizes the rest, and it sets `target_probs` *after* that
    renormalization -- so this is the one rewrite `l1_skew_pts` reports as zero.
    A baseline arm run against a root that had lost `stack-edu-python` would
    train on web alone and record a perfectly clean mixture summary. Zero-weight
    sources are checked too: they are exactly what makes a specialist arm the
    same experiment as the others, and a root missing one is not that root.

    **The sampled mixture is not the arm's mixture.** `cap_weights_by_epochs`
    clamps a source that cannot supply its share and water-fills the difference
    onto the others. Right for a production run; here it rewrites the one
    variable under test and leaves the arm named after a mixture it did not
    train on. `l1_skew_pts` is the exact measure of that, and it catches a
    source missing from the root -- which renormalizes the rest -- as well as
    the cap.

    **A source would be re-read past the cap.** In the all-capped regime the
    target mixture is *kept* and repetition is accepted instead, so there is no
    skew to see; the arm trains on its own mixture, many times over. That is not
    a rewrite, but it is not a usable proxy arm either: an arm whose advantage
    could be a fourth pass over its favourite source is not evidence about
    mixtures.

    Both are refusals rather than warnings, because the choice they force --
    shorten the budget, add data, or drop the arm -- is not one a launcher
    should make on its own at 200M tokens a time.
    """
    from train import mixture_preflight

    budget = int(shape.total_tokens if total_tokens is None else total_tokens)
    summary = mixture_preflight(data_dir, budget,
                                max_epochs=shape.max_source_epochs,
                                weights=arm.weights, verbose=False)
    missing = sorted(set(arm.weights) - set(summary["per_source"]))
    if missing:
        raise SystemExit(
            f"arm {arm.name!r} names {missing}, which {data_dir} has no shards "
            f"for. The mixture would be renormalized over what is left and the "
            f"summary would report no skew at all, because the target is taken "
            f"after that renormalization.")
    skew = float(summary["l1_skew_pts"])
    if skew > MAX_ARM_SKEW_PTS:
        raise SystemExit(
            f"arm {arm.name!r} would sample a mixture {skew:.4f} points of L1 "
            f"away from the one it names, at a {budget:,}-token budget: "
            f"{summary['per_source']}. Shorten the budget, add data for "
            f"{summary['capped_sources'] or 'the short sources'}, or drop the "
            f"arm -- but do not train it under this name.")
    seen = summary["max_epochs_seen"]
    if seen is not None and seen >= shape.max_source_epochs - 1e-6:
        raise SystemExit(
            f"arm {arm.name!r} would read {summary['most_repeated_source']} "
            f"{seen:.1f} times at a {budget:,}-token budget, at or past the "
            f"{shape.max_source_epochs:g}-epoch cap. The mixture is preserved "
            f"-- the corpus is simply too small for it -- but an arm whose "
            f"advantage could be a fourth pass over its own data is not "
            f"evidence about mixtures.")
    return summary


# =================================================================== running ===

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _weights_in(command: Sequence[str]) -> dict:
    """The `--mixture-weight` pairs a recorded command names."""
    parts, out = list(command), {}
    for index, part in enumerate(parts[:-1]):
        if part == "--mixture-weight":
            name, _, value = parts[index + 1].partition("=")
            out[name] = value
    return out


def finished_run(command: Sequence[str], ckpt) -> Optional[dict]:
    """The closed marker of an identical run that already finished, if any.

    Same four conditions as phase 6's: the marker must be this schema, its
    outcome must be `completed` rather than merely closed (a watchdog halt
    closes one too), the command must match exactly, and the checkpoint must
    actually exist -- the marker records what was intended.
    """
    from daedalus.supervise import INFLIGHT_SCHEMA, read_inflight

    ckpt = Path(ckpt)
    marker = read_inflight(str(ckpt.parent))
    if marker is None or marker.get("schema") != INFLIGHT_SCHEMA:
        return None
    if marker.get("outcome") != "completed":
        return None
    if marker.get("cmd") != list(command):
        return None
    return marker if ckpt.exists() else None


def foreign_run(command: Sequence[str], ckpt) -> Optional[dict]:
    """The mixture a marker in this run directory names, if it is not ours.

    The phase 6 hazard, in this phase's currency. There the discriminating
    argument was `--config` and a mistyped `--tag` could train a 159M arm over a
    finished 105M one; here it is the weights, and the way in is an arm
    *definition* that changed -- a blueprint share edited, a floor constant
    moved, a source added to the root -- while the run name stayed the same.

    `finished_run` cannot catch it, because a differing command is exactly what
    that guard lets through: a changed budget in the same directory is a rerun
    and must retrain. What separates the two is which argument changed. A
    different budget is a rerun; different weights under the same arm name are
    two experiments claiming one directory, and guessing which owns it is not a
    decision a launcher should make.
    """
    from daedalus.supervise import INFLIGHT_SCHEMA, read_inflight

    ckpt = Path(ckpt)
    marker = read_inflight(str(ckpt.parent))
    if marker is None or marker.get("schema") != INFLIGHT_SCHEMA:
        return None
    recorded, ours = _weights_in(marker.get("cmd") or ()), _weights_in(command)
    if not recorded or not ours or recorded == ours:
        return None
    return recorded


def run_arm(arm: MixtureArm, *, data_dir: str, tag: str,
            run_root: str = RUN_ROOT, device: str = "cuda",
            shape: ProbeShape = PROBE, val_dir: Optional[str] = None,
            total_tokens: Optional[int] = None, max_attempts: int = 3,
            stall_min: float = 20.0, refresh: bool = False) -> dict:
    """Train one arm under the supervisor, so an interruption continues it.

    `run_with_resume` reads the open in-flight marker beside the checkpoint, so
    a relaunch after the launching session died continues from where the arm got
    to rather than restarting it -- which is how phase 4 lost 60.3M tokens next
    to a checkpoint it never opened.

    A run that already *finished* needs the opposite guard and does not get it
    from the supervisor: its marker is closed, so `interrupted_marker` correctly
    declines to resume it, and `train.py` then starts at step 0 and overwrites
    the checkpoint on its first save. A completed identical run is therefore
    returned rather than re-entered, unless `refresh` asks for it deliberately.
    """
    from daedalus.supervise import run_with_resume, start_watchdog, stop_watchdog

    name = arm_run_name(arm, tag)
    budget = int(shape.total_tokens if total_tokens is None else total_tokens)
    command = train_command(arm, data_dir=data_dir, run_name=name, shape=shape,
                            device=device, val_dir=val_dir,
                            total_tokens=total_tokens)
    ckpt = arm_checkpoint_path(arm, tag, run_root)
    run_dir = ckpt.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    common = {**arm.describe(), "run": name, "run_dir": str(run_dir),
              "shape": shape.name, "total_tokens": budget,
              "steps": budget // shape.batch_tokens, "command": list(command)}

    occupant = foreign_run(command, ckpt)
    if occupant is not None:
        raise SystemExit(
            f"{run_dir} already holds a run of mixture {occupant}, but arm "
            f"{arm.name} is {_weights_in(command)}. Training here would "
            f"overwrite another experiment's checkpoint; use a --tag that "
            f"stage owns, or --refresh if the arm really was redefined.")
    if not refresh and finished_run(command, ckpt) is not None:
        # Recorded, not omitted: an artifact that drops a skipped arm and one
        # that never ran it look identical to a reader.
        return {**common, "skipped": "already-completed", "attempts": 0,
                "resumed": False, "returncodes": []}

    # Before the GPU, not after: a capped arm is a refusal, and finding that out
    # forty minutes in costs the forty minutes.
    common["preflight"] = arm_preflight(arm, data_dir=data_dir, shape=shape,
                                        total_tokens=total_tokens)

    watchdog = start_watchdog(name, str(run_dir), budget, stall_min=stall_min,
                              supervised=True)
    try:
        report = run_with_resume(
            list(command), str(ckpt),
            max_attempts=max_attempts, halt_marker=str(run_dir / "HALTED"),
            inflight_extra={"phase": "phase7-mixture", "arm": arm.name,
                            "weights": dict(arm.weights),
                            "shape": shape.name, "total_tokens": budget})
    finally:
        stop_watchdog(watchdog)
    return {**common, **report}


def sweep(*, data_dir: str, stage: str, tag: str, run_root: str = RUN_ROOT,
          report_root: str = REPORT_ROOT, device: str = "cuda",
          shape: ProbeShape = PROBE, val_dir: Optional[str] = None,
          total_tokens: Optional[int] = None,
          derived: Optional[dict] = None,
          refresh: bool = False) -> dict:
    """Every arm of one stage, baseline first.

    Re-entrant by design: arms that already finished are returned from their
    closed markers, so relaunching a sweep the deadline or a dead session cut
    short costs only the arms that have not run. The baseline is shared between
    the two stages under one tag, so the candidates sweep skips it rather than
    training it twice under two names.
    """
    sources = discover_sources(data_dir)
    arms = stage_arms(stage, sources, derived)
    header = {
        "phase": "phase7-mixture",
        "stage": stage,
        "tag": tag,
        "shape": asdict(shape),
        "steps": shape.steps,
        "data_dir": str(data_dir),
        "val_dir": str(val_dir) if val_dir else None,
        "sources": sources,
        # The yardstick, written beside the arms rather than left to be
        # re-derived at scoring time: every arm's aggregate BPB is computed
        # under these weights, and an artifact that does not carry them cannot
        # be checked for having used the same ones twice.
        "evaluation_weights": {name: round(value, 6) for name, value
                               in evaluation_weights(sources).items()},
        "unrepresented_floored_domains": unrepresented_floored_domains(sources),
        "total_tokens": int(shape.total_tokens if total_tokens is None
                            else total_tokens),
    }
    results: List[dict] = []
    for arm in arms:
        results.append(run_arm(arm, data_dir=data_dir, tag=tag,
                               run_root=run_root, device=device, shape=shape,
                               val_dir=val_dir, total_tokens=total_tokens,
                               refresh=refresh))
        # Rewritten after every arm, so a sweep cut short still leaves the arms
        # that finished, in the order they ran.
        _write_json(Path(report_root) / f"mixture-sweep-{stage}.json",
                    {**header, "arms": results})
    return {**header, "arms": results}


# =================================================================== scoring ===
# The arms are trained; this is the measurement the selection rule reads. Two
# things make it a different pass from `train.py`'s own `val_bpb` rather than a
# re-run of it, and both are stated in the module docstring: the in-run number
# weights the holdout by each arm's *own* mixture, and it is a bounded sample
# taken mid-decay. The comparison has to be one weighting for every arm, over
# every held-out window, from the final checkpoint.

#: Where the per-arm scorecards land: beside the sweep artifact, never inside a
#: run directory. A scorecard next to a checkpoint is a scorecard that can be
#: mistaken for one, and phase 6 put its own in the same place for the same
#: reason.
SCORECARD_ROOT = f"{REPORT_ROOT}/scorecards"

#: Scored at the context the arms trained at. A held-out BPB measured at a
#: longer context than the run ever saw is a measurement of extrapolation.
SCORE_SEQ_LEN = PROBE.seq_len

#: Windows per forward pass. Only affects wall-clock: BPB is an average over
#: whole non-overlapping windows, so the number is batch-size invariant.
SCORE_BATCH_SIZE = 8

#: The program's seed, carried into every scorecard's provenance.
SCORE_SEED = 20260824


def scorecard_name(arm: MixtureArm, tag: str) -> str:
    return f"mix-{tag}-{arm.name}-bpb"


def scorecard_path(arm: MixtureArm, *, tag: str,
                   out_dir: str = SCORECARD_ROOT) -> Path:
    return Path(out_dir) / f"{scorecard_name(arm, tag)}.json"


def sole_source(arm: MixtureArm) -> Optional[str]:
    """The single source an arm puts all its mass on, if it is one-hot.

    Asked of the weights rather than of the name. `specialist_name` builds
    `only-<source>`, and reading the role back out of a string would make an arm
    whose name matched that pattern -- or one whose weights were redefined while
    its name stayed -- score as something it is not.
    """
    carrying = [name for name, value in arm.weights.items() if value > 0.0]
    return carrying[0] if len(carrying) == 1 else None


def scoring_sources(arm: MixtureArm, sources: Sequence[str]) -> List[str]:
    """Which holdout sources this arm's *role* requires be measured.

    A specialist is read by exactly one number: `bpb_specialist_s(s)`, the
    achievable bound in `excess_loss`. Scoring it on the other two sources would
    answer what a one-source model loses elsewhere -- a real question, and not
    one this phase asks -- at three times the GPU hours per specialist.

    Every other arm is scored on every source, because both halves of the
    selection rule need that: the aggregate is a fixed-weight average over the
    whole corpus, and the per-source regression check compares a candidate to
    the baseline source by source.
    """
    only = sole_source(arm)
    return [only] if only is not None else list(dict.fromkeys(sources))


def already_scored(path, checkpoint_sha: str) -> bool:
    """True when `path` already scores exactly these checkpoint bytes.

    Keyed on the digest rather than on the file existing, so `--refresh` on a
    retrained arm cannot leave the old arm's number in place: the scorecard is
    reused only when the thing it scored is the thing that is there now.
    """
    from daedalus.scorecard import ScorecardError, load_scorecard

    path = Path(path)
    if not path.exists():
        return False
    try:
        card = load_scorecard(path)
    except (ScorecardError, KeyError, ValueError, OSError):
        return False
    return card.provenance.artifact.sha256 == checkpoint_sha


def make_checkpoint_bpb_fn(checkpoint, *, config: str, device: str,
                           seq_len: int, batch_size: int):
    """Load one arm's final weights and return its per-source BPB callable.

    Imported inside, like phase 6's equivalent, so the selection logic and its
    tests never need torch to exercise the rules that decide the phase.
    """
    from daedalus.config import PRESETS
    from daedalus.data import get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb
    from train import load_checkpoint

    # The shipped tokenizer, which is what packed these shards. Phase 4 varies
    # the vocabulary; phase 7 holds it and varies the data, so passing anything
    # else here would decode the held-out ids into the wrong bytes and report a
    # BPB for a corpus that does not exist.
    tokenizer = get_tokenizer(None)
    model = Daedalus(PRESETS[config]).to(device)
    load_checkpoint(str(checkpoint), model, map_location=device)
    model.eval()

    def bpb_fn(source_dir: Path) -> float:
        # max_batches=None: the full pass the selection rule requires.
        return evaluate_bpb(model, str(source_dir), seq_len, tokenizer, device,
                            batch_size=batch_size, max_batches=None)

    return bpb_fn


def _release(device: str) -> None:
    """Drop the previous arm's weights before the next model lands.

    Six arms scored in one process is six models plus six sets of activations
    unless each is released.
    """
    import gc

    gc.collect()
    if not device.startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:                                     # pragma: no cover
        pass


def score_arm(arm: MixtureArm, *, holdout_root: str, sources: Sequence[str],
              tag: str, run_root: str = RUN_ROOT,
              out_dir: str = SCORECARD_ROOT, shape: ProbeShape = PROBE,
              device: str = "cuda", seq_len: int = SCORE_SEQ_LEN,
              batch_size: int = SCORE_BATCH_SIZE, seed: int = SCORE_SEED,
              refresh: bool = False,
              bpb_factory=make_checkpoint_bpb_fn) -> dict:
    """Full-pass held-out BPB for one finished arm, written as a scorecard.

    The aggregate is computed under `evaluation_weights`, the same weighting for
    every arm scored over the whole corpus -- that is the entire reason this pass
    exists rather than reading `val_bpb` out of the training logs.
    """
    from daedalus.scorecard import ArtifactRef, sha256_file
    from scripts.bpb_eval import _git_short_sha, run_bpb_eval

    checkpoint = arm_checkpoint_path(arm, tag, run_root)
    if not checkpoint.exists():
        raise SystemExit(
            f"arm {arm.name!r} has no checkpoint at {checkpoint}: it either "
            f"never ran under --tag {tag!r} or its run directory moved. Run the "
            f"sweep before scoring it; there is no partial score worth having.")
    digest = sha256_file(checkpoint)
    path = scorecard_path(arm, tag=tag, out_dir=out_dir)
    wanted = scoring_sources(arm, sources)
    if not refresh and already_scored(path, digest):
        # Recorded rather than omitted: a pass that drops a skipped arm and one
        # that never scored it look identical to a reader.
        return {"arm": arm.name, "scorecard": str(path),
                "skipped": "already-scored", "checkpoint_sha256": digest,
                "sources": wanted}

    # Only an arm measured over the whole corpus gets the fixed weighting; a
    # specialist scored on its own source alone has no aggregate to weight, and
    # `summarize_bpb` refuses weights naming sources its records do not carry.
    whole_corpus = len(wanted) == len(list(dict.fromkeys(sources)))
    weights = evaluation_weights(sources) if whole_corpus else None

    bpb_fn = bpb_factory(checkpoint, config=shape.config, device=device,
                         seq_len=seq_len, batch_size=batch_size)
    try:
        written = run_bpb_eval(
            name=scorecard_name(arm, tag), holdout_root=holdout_root,
            out_dir=out_dir,
            artifact=ArtifactRef(path=str(checkpoint), sha256=digest,
                                 kind="checkpoint", config=shape.config),
            tokenizer_ref=ArtifactRef(path="<smollm2-default>", sha256="0" * 64,
                                      kind="tokenizer"),
            seed=seed, git_sha=_git_short_sha(), bpb_fn=bpb_fn,
            max_batches=None, weights=weights, sources=wanted,
            runtime={"device": device, "seq_len": seq_len,
                     "batch_size": batch_size},
            details_extra={"phase": "phase7-mixture", "arm": arm.name,
                           "tag": tag, "shape": shape.name,
                           "run": arm_run_name(arm, tag),
                           "basis": arm.basis,
                           "is_baseline": arm.is_baseline,
                           "train_weights": {name: round(value, 6) for name, value
                                             in sorted(arm.weights.items())},
                           "scored_role": "specialist" if not whole_corpus
                           else "mixture"})
    finally:
        del bpb_fn
        _release(device)
    return {"arm": arm.name, "scorecard": str(written["scorecard"]),
            "checkpoint_sha256": digest, "sources": wanted}


def score_arms(arms: Sequence[MixtureArm], **kwargs) -> dict:
    """Score every arm, baseline first, leaving each scorecard as it lands.

    Re-entrant for the reason the sweep is: this is a GPU pass over several
    finished checkpoints, and a session that ends mid-pass must cost only the
    arm it was on.
    """
    return {"scored": [score_arm(arm, **kwargs) for arm in arms]}


# ================================================================ derivation ===

def full_pass_bpb(card) -> Dict[str, float]:
    """`{source: bpb}` from a scorecard, refusing anything that is not a full
    pass.

    A bounded sample and a full pass are different measurements, and this one
    feeds a preregistered rule. Read from the items rather than from
    `metrics["bpb"]`, which is an aggregate under whatever weighting the card
    happened to carry.
    """
    from daedalus.scorecard import ScorecardError

    if card.provenance.bpb_mode != "full":
        raise SystemExit(
            f"scorecard {card.name!r} was measured in bpb_mode "
            f"{card.provenance.bpb_mode!r}; excess loss is a difference of two "
            f"held-out BPBs and a sampled one cannot be subtracted from a full "
            f"pass. Re-score it with --max-batches -1.")
    values = {}
    for item in card.items or ():
        try:
            values[str(item["id"])] = float(item["bpb"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScorecardError(
                f"scorecard {card.name!r} has an item without a readable "
                f"bpb: {item!r}") from exc
    if not values:
        raise SystemExit(
            f"scorecard {card.name!r} carries no per-source items; the "
            f"derivation reads per-source BPB, not an aggregate")
    return values


def _scored_card(arm: MixtureArm, *, tag: str, out_dir: str) -> tuple:
    """One arm's scorecard and its per-source BPB, or a refusal naming it."""
    from daedalus.scorecard import load_scorecard

    path = scorecard_path(arm, tag=tag, out_dir=out_dir)
    if not path.exists():
        raise SystemExit(
            f"arm {arm.name!r} has no scorecard at {path}. Excess loss is "
            f"defined against a specialist, so a source whose specialist has "
            f"not been scored has no measured excess and the derived mixture "
            f"cannot be built. Run `score --stage reference` first.")
    return path, load_scorecard(path)


def _holdout_key(card) -> Optional[str]:
    """A card's holdout root, as a path two cards can be compared on.

    Absolute and normalized, because `data/holdout` and
    `/workspace/daedalus/data/holdout` are the same corpus and refusing to
    subtract them would be a false alarm about a directory string.
    """
    root = (card.details or {}).get("holdout_root")
    return os.path.normpath(os.path.abspath(str(root))) if root else None


def _comparable(cards: Dict[str, object]) -> dict:
    """The holdout and context every card shares, or a refusal.

    Excess loss subtracts one arm's BPB from another's. Two numbers measured
    over different holdout roots, or at different context lengths, are not each
    other's units -- subtracting them produces a mixture derived from an
    arithmetic error, and every field in the artifact would still look right.
    """
    seen = {name: (_holdout_key(card), (card.provenance.runtime or {}).get("seq_len"))
            for name, card in cards.items()}
    distinct = sorted(set(seen.values()), key=lambda pair: (str(pair[0]), str(pair[1])))
    if len(distinct) > 1:
        raise SystemExit(
            f"the scorecards were not measured against one holdout: {seen}. "
            f"Excess loss is a difference of two BPBs, so they must share a "
            f"holdout root and a context length; re-score the odd ones out.")
    holdout_root, seq_len = distinct[0]
    return {"holdout_root": holdout_root, "seq_len": seq_len}


def derived_path(tag: str, root: str = REPORT_ROOT) -> Path:
    """Where `derive` writes, and where the candidates sweep reads its weights.

    Scoped by tag, like phase 6's stage reports: two stages sharing one filename
    is how a rerun quietly overwrites the evidence another stage was launched
    from.
    """
    return Path(root) / f"mixture-derived-{tag}.json"


def derive(*, sources: Sequence[str], tag: str,
           out_dir: str = SCORECARD_ROOT) -> dict:
    """The derived mixture, from the reference stage's scored arms.

    Every number in the rule -- the temperature, the ratio cap, the floors --
    is `daedalus/mixture_opt.py`'s, committed before the first arm was trained.
    This reads the measurements, applies them, and records what it read, so the
    derived weights can be recomputed from the artifact rather than trusted.
    """
    from daedalus.mixture_opt import (EXCESS_RATIO_CAP, EXCESS_TEMPERATURE,
                                      baseline_arm, derive_weights,
                                      domain_floors, domain_shares,
                                      excess_loss, excess_scores)

    sources = list(dict.fromkeys(sources))
    baseline = baseline_arm(sources)
    baseline_path, baseline_card = _scored_card(baseline, tag=tag, out_dir=out_dir)
    baseline_bpb = full_pass_bpb(baseline_card)
    missing = sorted(set(sources) - set(baseline_bpb))
    if missing:
        raise SystemExit(
            f"the baseline scorecard {baseline_path} scored "
            f"{sorted(baseline_bpb)} and not {missing}. Excess loss is measured "
            f"per source against the baseline, so a source the baseline was not "
            f"scored on cannot be tilted.")

    cards = {baseline.name: baseline_card}
    specialists: Dict[str, float] = {}
    provenance: Dict[str, dict] = {}
    for source in sources:
        arm = next(candidate for candidate in reference_arms(sources)
                   if sole_source(candidate) == source)
        path, card = _scored_card(arm, tag=tag, out_dir=out_dir)
        values = full_pass_bpb(card)
        if source not in values:
            raise SystemExit(
                f"the specialist scorecard {path} scored {sorted(values)}, "
                f"which does not include {source!r} -- the one source it is "
                f"read for. Re-score arm {arm.name!r}.")
        cards[arm.name] = card
        specialists[source] = values[source]
        provenance[source] = {
            "arm": arm.name, "scorecard": str(path),
            "checkpoint_sha256": card.provenance.artifact.sha256,
            "bpb": values[source]}

    measured_on = _comparable(cards)
    excess = excess_loss({name: baseline_bpb[name] for name in sources},
                         specialists)
    floors = domain_floors(baseline.weights)
    weights = derive_weights(baseline.weights, excess, floors=floors)
    return {
        "phase": "phase7-mixture",
        "stage": "candidates",
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat()
                              .replace("+00:00", "Z"),
        "weights": {name: round(value, 6) for name, value in sorted(weights.items())},
        "domain_shares": {domain: round(value, 6) for domain, value
                          in sorted(domain_shares(weights).items())},
        "rule": {
            "excess": "bpb_baseline(s) - bpb_specialist_s(s), on the same "
                      "full-pass holdout",
            "temperature": EXCESS_TEMPERATURE,
            "ratio_cap": EXCESS_RATIO_CAP,
            "floors": {domain: round(value, 6)
                       for domain, value in sorted(floors.items())},
            "unrepresented_floored_domains":
                unrepresented_floored_domains(sources),
        },
        "measured_on": measured_on,
        "baseline": {
            "arm": baseline.name, "scorecard": str(baseline_path),
            "checkpoint_sha256": baseline_card.provenance.artifact.sha256,
            "weights": {name: round(value, 6)
                        for name, value in sorted(baseline.weights.items())},
            "bpb": {name: baseline_bpb[name] for name in sources},
        },
        "specialists": provenance,
        "excess_loss": excess,
        "excess_scores": excess_scores(excess),
    }


# ====================================================================== cli ====

def _read_derived(path: Optional[str]) -> Optional[dict]:
    """Derived weights from a JSON file, accepting the artifact or bare shares.

    The derivation step writes a record with provenance around the weights;
    accepting `{"weights": {...}}` as well as `{...}` means the candidates sweep
    reads that artifact directly instead of someone retyping three numbers out
    of it, which is the step where a mixture stops being the derived one.
    """
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    weights = payload.get("weights", payload) if isinstance(payload, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise SystemExit(f"{path} has no mixture weights")
    return {str(name): float(value) for name, value in weights.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--tag", default="probe",
                        help="run-directory prefix shared by every arm, so the "
                             "baseline is trained once and both stages read it")
    parser.add_argument("--shape", default=PROBE.name, choices=list(SHAPES))
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("arms", "run", "sweep"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data-dir", required=True)
        cmd.add_argument("--stage", default="reference", choices=list(STAGES))
        cmd.add_argument("--derived-weights", default=None,
                         help="JSON file of derived shares, for the candidates "
                              "stage. Without it the derived arm is omitted "
                              "rather than stubbed with the blueprint")
        if name == "arms":
            continue
        if name == "run":
            cmd.add_argument("--arm", required=True)
        cmd.add_argument("--val-dir", default=None)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--total-tokens", type=int, default=None,
                         help="override the shape's budget (smokes only)")
        cmd.add_argument("--refresh", action="store_true",
                         help="re-train arms that already completed, "
                              "overwriting their checkpoints")

    sub.add_parser("shape")

    for name in ("score", "derive"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data-dir", required=True,
                         help="the mixture root the arms trained over; the "
                              "sources it holds are the sources the arms name")
        cmd.add_argument("--scorecard-root", default=SCORECARD_ROOT)
        if name == "derive":
            cmd.add_argument("--out", default=None,
                             help=f"default {derived_path('<tag>')}")
            continue
        cmd.add_argument("--val-dir", required=True,
                         help="the holdout root, one manifest-backed "
                              "subdirectory per source")
        cmd.add_argument("--stage", default="reference", choices=list(STAGES))
        cmd.add_argument("--derived-weights", default=None,
                         help="JSON file of derived shares, for scoring the "
                              "candidates stage")
        cmd.add_argument("--arm", default=None,
                         help="score this arm only; without it every arm of "
                              "the stage that has a checkpoint is scored")
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--seq-len", type=int, default=SCORE_SEQ_LEN)
        cmd.add_argument("--batch-size", type=int, default=SCORE_BATCH_SIZE)
        cmd.add_argument("--refresh", action="store_true",
                         help="re-score arms whose scorecard already matches "
                              "their checkpoint bytes")

    args = parser.parse_args(argv)
    shape = SHAPES[args.shape]

    if args.command == "shape":
        print(json.dumps({**asdict(shape), "steps": shape.steps,
                          "grad_accum": shape.grad_accum}, indent=2,
                         sort_keys=True))
        return 0

    # `derive` is the command that *produces* the derived weights, so it is the
    # one subcommand with no `--derived-weights` to read.
    derived = _read_derived(getattr(args, "derived_weights", None))
    sources = discover_sources(args.data_dir)

    if args.command == "arms":
        for arm in stage_arms(args.stage, sources, derived):
            print(json.dumps(arm.describe(), sort_keys=True))
        return 0

    if args.command == "run":
        arms = {arm.name: arm for arm in stage_arms(args.stage, sources, derived)}
        if args.arm not in arms:
            raise SystemExit(f"unknown arm {args.arm!r}; known: {sorted(arms)}")
        report = run_arm(arms[args.arm], data_dir=args.data_dir, tag=args.tag,
                         run_root=args.run_root, device=args.device,
                         shape=shape, val_dir=args.val_dir,
                         total_tokens=args.total_tokens, refresh=args.refresh)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "score":
        arms = stage_arms(args.stage, sources, derived)
        if args.arm is not None:
            by_name = {arm.name: arm for arm in arms}
            if args.arm not in by_name:
                raise SystemExit(
                    f"unknown arm {args.arm!r}; known: {sorted(by_name)}")
            arms = [by_name[args.arm]]
        report = score_arms(
            arms, holdout_root=args.val_dir, sources=sources, tag=args.tag,
            run_root=args.run_root, out_dir=args.scorecard_root, shape=shape,
            device=args.device, seq_len=args.seq_len,
            batch_size=args.batch_size, refresh=args.refresh)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "derive":
        record = derive(sources=sources, tag=args.tag,
                        out_dir=args.scorecard_root)
        out = Path(args.out) if args.out else derived_path(
            args.tag, args.report_root)
        _write_json(out, record)
        print(json.dumps({"weights": record["weights"],
                          "excess_loss": record["excess_loss"],
                          "excess_scores": record["excess_scores"],
                          "wrote": str(out)}, indent=2, sort_keys=True))
        return 0

    if args.command == "sweep":
        report = sweep(data_dir=args.data_dir, stage=args.stage, tag=args.tag,
                       run_root=args.run_root, report_root=args.report_root,
                       device=args.device, shape=shape, val_dir=args.val_dir,
                       total_tokens=args.total_tokens, derived=derived,
                       refresh=args.refresh)
        print(json.dumps({"stage": report["stage"],
                          "arms": [row["arm"] for row in report["arms"]],
                          "skipped": [row["arm"] for row in report["arms"]
                                      if row.get("skipped")]},
                         indent=2))
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
