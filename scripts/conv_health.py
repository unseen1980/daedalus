"""Phase 5: does a decay schedule stop ShortConv channel death, and at what cost?

Four schedules for the conv projections, one variable between them, read on the
functional instrument in `daedalus/conv_health.py` rather than on the shipped
weight proxy. The reason for both is the same and is recorded in
`runs/preflight/conv-death-fix-validated.md`: the fix under test *is* a change
to weight decay, so a magnitude metric can be satisfied by an arm where nothing
shrank as easily as by one where nothing died.

The arms and why these four:

    shipped-0.1        the control -- what the released model trained under
    weak-0.0133        7.5x weaker, constant
    warmup-0-to-0.1    nothing for the first 10%, shipped decay after
    weak-then-0.1      0.0133 ramping to shipped decay by 30%

There is no zero-decay arm, and that is deliberate. Decay 0 wins on death
outright (0.00% against baseline's 67.06%) and still cannot be the
recommendation, because it has no equilibrium: the same experiment measured
6.8x and 10.5x projection-norm growth in 600 steps, which is 0.5% of a real
run, against a decay whose stated purpose is keeping Muon stable in exactly the
overtrained regime a real run ends in. The two varying arms are the shapes that
can have both properties -- weak while the early race is being decided, shipped
strength once it is over.

Every arm carries `--conv-proj-wd`, the control included. Running the control
as the shipped *single*-group split would make the arms differ in optimizer
layout as well as in decay, and the whole comparison rests on one variable;
0.1 in its own group is numerically the shipped decay with the layout matched.

Subcommands: `arms`, `run`, `sweep`, `score`, `report`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


#: The established positive control: 600 steps at Muon lr 0.15 makes what
#: `hero` needed ~10,000 steps at 0.02 to show.
PROBE_STEPS = 600
PROBE_MUON_LR = 0.15
PROBE_SEQ_LEN = 256
PROBE_MICRO_BATCH = 8
PROBE_CONFIG = "conv-probe"

#: Fixed, non-overlapping held-out windows. Deterministic and never sampled:
#: a 0.01-nat ablation effect must not be confusable with which batches were
#: drawn, which is why the mechanism experiment scored on fixed windows too.
HOLDOUT_BATCHES = 16

#: Where `train.py` puts a run, and therefore the only place a supervisor may
#: look for its checkpoint. `train.py` resolves `runs/<run_name>` and has no CLI
#: flag to move it, so a launcher that watched anything else would hand
#: `run_with_resume` a path that never appears: the in-flight marker would sit
#: beside a file that is never written, every relaunch would start from step 0,
#: and no log would say so. Same layout as the phase 4 probes.
RUN_ROOT = "runs"

#: Sweep, score and verdict artifacts, kept out of the run directories so a
#: report is not mistaken for a checkpoint.
REPORT_ROOT = "runs/conv-health"


@dataclass(frozen=True)
class ConvArm:
    """One preregistered decay schedule."""

    name: str
    start: float
    end: Optional[float] = None
    ramp_frac: float = 0.0
    hold_frac: float = 0.0
    note: str = ""

    @property
    def is_control(self) -> bool:
        return self.name == "shipped-0.1"

    def train_flags(self) -> List[str]:
        flags = ["--conv-proj-wd", repr(self.start)]
        if self.end is not None:
            flags += ["--conv-proj-wd-end", repr(self.end),
                      "--conv-proj-wd-ramp-frac", repr(self.ramp_frac)]
            if self.hold_frac:
                flags += ["--conv-proj-wd-hold-frac", repr(self.hold_frac)]
        return flags


ARMS: Sequence[ConvArm] = (
    ConvArm("shipped-0.1", start=0.1,
            note="control: the decay the released model trained under, in its "
                 "own group so the arms differ only in the decay value"),
    ConvArm("weak-0.0133", start=0.0133,
            note="7.5x weaker constant; its own mechanism predicts the death "
                 "is postponed rather than prevented"),
    ConvArm("warmup-0-to-0.1", start=0.0, end=0.1, ramp_frac=0.10,
            note="no decay while the early race is decided, shipped decay after"),
    ConvArm("weak-then-0.1", start=0.0133, end=0.1, ramp_frac=0.30,
            note="weak early, shipped decay by 30% of the run"),
)

ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
CONTROL = ARMS[0]


def arm_run_name(arm: ConvArm, tag: str = "probe") -> str:
    return f"conv-{tag}-{arm.name}"


def arm_checkpoint_path(arm: ConvArm, tag: str = "probe",
                        run_root: str = RUN_ROOT) -> Path:
    """The checkpoint this arm writes, asked of `train.py` rather than guessed.

    `run_dir_for` is the trainer's own resolution, so the supervisor and the
    scorer cannot drift from where the run actually lands.
    """
    from train import TrainArgs, checkpoint_path_for

    args = TrainArgs(run_name=arm_run_name(arm, tag), config=PROBE_CONFIG,
                     data_dir="", run_dir=None)
    resolved = Path(checkpoint_path_for(args))
    if run_root != RUN_ROOT:                    # tests may relocate the tree
        resolved = Path(run_root) / arm_run_name(arm, tag) / "checkpoint.pt"
    return resolved


def probe_train_command(arm: ConvArm, *, data_dir: str, run_name: str,
                        total_tokens: int, device: str = "cuda",
                        config: str = PROBE_CONFIG,
                        muon_lr: float = PROBE_MUON_LR,
                        val_dir: Optional[str] = None) -> List[str]:
    """The exact `train.py` invocation for one arm.

    Every field except the decay flags is shared, and the shared fields are
    built here rather than per arm so an arm cannot quietly differ in seed,
    data order, batch shape or schedule. That identity is the experiment.
    """
    command = [
        sys.executable, "train.py",
        "--run-name", run_name,
        "--config", config,
        "--data-dir", data_dir,
        "--total-tokens", str(int(total_tokens)),
        "--micro-batch", str(PROBE_MICRO_BATCH),
        # Flat sequence length and batch: the ramp exists to buy throughput on
        # a long run and would make `lr x steps` mean something different early
        # and late, which is the axis the arms are compared on.
        "--seq-start", str(PROBE_SEQ_LEN), "--seq-end", str(PROBE_SEQ_LEN),
        "--tok-start", str(PROBE_MICRO_BATCH * PROBE_SEQ_LEN),
        "--tok-end", str(PROBE_MICRO_BATCH * PROBE_SEQ_LEN),
        "--muon-lr", repr(muon_lr),
        "--warmup-steps", "50",
        "--decay-frac", "0.2",
        "--device", device,
        "--hub-repo", "",
        "--no-wandb",
    ]
    if val_dir:
        command += ["--val-dir", val_dir]
    return command + arm.train_flags()


def probe_total_tokens(steps: int = PROBE_STEPS) -> int:
    return steps * PROBE_MICRO_BATCH * PROBE_SEQ_LEN


# =================================================================== running ===

def run_arm(arm: ConvArm, *, data_dir: str, run_root: str = RUN_ROOT,
            device: str = "cuda", steps: int = PROBE_STEPS, tag: str = "probe",
            config: str = PROBE_CONFIG, muon_lr: float = PROBE_MUON_LR,
            val_dir: Optional[str] = None, max_attempts: int = 3,
            stall_min: float = 20.0) -> dict:
    """Train one arm under the supervisor, so an interruption continues it.

    `run_with_resume` reads the open in-flight marker beside the checkpoint, so
    a relaunch after the launching session died continues from where the arm
    got to instead of restarting it -- which is how phase 4 lost 60.3M tokens
    next to a checkpoint it never opened.
    """
    from daedalus.supervise import run_with_resume, start_watchdog, stop_watchdog

    name = arm_run_name(arm, tag)
    total_tokens = probe_total_tokens(steps)
    command = probe_train_command(
        arm, data_dir=data_dir, run_name=name, total_tokens=total_tokens,
        device=device, config=config, muon_lr=muon_lr, val_dir=val_dir)

    ckpt = arm_checkpoint_path(arm, tag, run_root)
    run_dir = ckpt.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog = start_watchdog(name, str(run_dir), total_tokens,
                              stall_min=stall_min, supervised=True)
    try:
        report = run_with_resume(
            list(command), str(ckpt),
            max_attempts=max_attempts, halt_marker=str(run_dir / "HALTED"),
            inflight_extra={"phase": "phase5-conv-health", "arm": arm.name,
                            "schedule": asdict(arm), "steps": steps})
    finally:
        stop_watchdog(watchdog)
    return {"arm": arm.name, "run": name, "run_dir": str(run_dir),
            "steps": steps, "total_tokens": total_tokens,
            "command": list(command), **report}


def sweep(*, data_dir: str, run_root: str = RUN_ROOT,
          report_root: str = REPORT_ROOT, device: str = "cuda",
          steps: int = PROBE_STEPS, tag: str = "probe",
          config: str = PROBE_CONFIG, muon_lr: float = PROBE_MUON_LR,
          val_dir: Optional[str] = None,
          arms: Sequence[ConvArm] = ARMS) -> dict:
    """Every arm in order, control first.

    Control first so a sweep cut short still has the one arm without which none
    of the others can be read: an arm reading 0% dead means nothing until the
    baseline has been shown to die.
    """
    results = []
    for arm in arms:
        results.append(run_arm(arm, data_dir=data_dir, run_root=run_root,
                               device=device, steps=steps, tag=tag,
                               config=config, muon_lr=muon_lr, val_dir=val_dir))
        # Rewritten after every arm, so a sweep cut short still leaves the
        # arms that finished.
        _write_json(Path(report_root) / f"sweep-{tag}.json",
                    {"tag": tag, "steps": steps, "arms": results})
    return {"tag": tag, "steps": steps, "arms": results}


# =================================================================== scoring ===

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def holdout_windows(holdout_dir: str, *, seq_len: int = PROBE_SEQ_LEN,
                    batch_size: int = PROBE_MICRO_BATCH,
                    batches: int = HOLDOUT_BATCHES) -> List["object"]:
    """Fixed non-overlapping held-out windows, materialised once.

    `shuffle=False` with `stride=seq_len` visits each token exactly once in
    file order, so the same call returns the same windows every time and in
    every arm. Materialised rather than re-iterated because the ablation deltas
    are differences between passes over *identical* data -- re-deriving the
    windows per pass is one refactor away from silently not being identical.
    """
    from daedalus.data import make_loader

    loader = make_loader(holdout_dir, seq_len, batch_size, shuffle=False,
                         num_workers=0, stride=seq_len)
    windows = []
    for batch in loader:
        windows.append(batch)
        if len(windows) >= batches:
            break
    if not windows:
        raise ValueError(f"no held-out windows in {holdout_dir}")
    return windows


def held_out_loss(model, windows, device: str = "cuda") -> float:
    """Mean loss over the fixed windows. Deterministic by construction."""
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in windows:
            x = batch.to(device)
            _, loss, _ = model(x, targets=x)
            total += float(loss)
            count += 1
    return total / count


def score_arm(arm: ConvArm, *, run_root: str = RUN_ROOT, holdout_dir: str,
              device: str = "cuda", tag: str = "probe",
              config: str = PROBE_CONFIG,
              matched_k: Optional[Dict[int, int]] = None,
              threshold: Optional[float] = None) -> dict:
    """Channel health, projection norms and the ablation pair for one arm.

    `matched_k` is the per-layer size of the *control's* flagged set. The plan's
    rule is that a clean arm must lose measurably when its weakest
    baseline-sized channel set is removed, and that only means something if the
    size is the baseline's -- an arm choosing its own k could pick zero and pass
    by ablating nothing.
    """
    import torch

    from daedalus.config import PRESETS
    from daedalus.conv_health import (DEAD_THRESHOLD, ablated_model,
                                      conv_layers, dead_channels, model_health,
                                      projection_norms, weakest_alive)
    from daedalus.model import Daedalus
    from train import load_checkpoint

    threshold = DEAD_THRESHOLD if threshold is None else threshold
    name = arm_run_name(arm, tag)
    ckpt = arm_checkpoint_path(arm, tag, run_root)
    if not ckpt.exists():
        raise FileNotFoundError(f"arm {arm.name} has no checkpoint at {ckpt}")

    model = Daedalus(PRESETS[config]).to(device)
    load_checkpoint(str(ckpt), model)
    health = model_health(model, threshold=threshold)
    windows = holdout_windows(holdout_dir)

    baseline_loss = held_out_loss(model, windows, device)
    flagged = {layer.layer_index: dead_channels(layer)
               for layer in health.layers}
    with ablated_model(model, flagged):
        flagged_loss = held_out_loss(model, windows, device)

    # The matched control. On a clean arm the flagged set is empty and ablating
    # it proves nothing, so this is the load-bearing half of the pair.
    sizes = matched_k or {index: len(channels)
                          for index, channels in flagged.items()}
    matched = {}
    for layer_health in health.layers:
        k = int(sizes.get(layer_health.layer_index, 0))
        if k:
            matched[layer_health.layer_index] = weakest_alive(layer_health, k)
    if matched:
        with ablated_model(model, matched):
            matched_loss = held_out_loss(model, windows, device)
    else:
        # Nothing to ablate: the control flagged nothing anywhere, which the
        # verdict reads as an invalid sweep rather than as four clean arms.
        matched_loss = baseline_loss

    layers = dict(conv_layers(model))
    norms = {layer_health.layer_index:
             projection_norms(layers[layer_health.layer_index], layer_health)
             for layer_health in health.layers}
    alive_total = sum(n["alive_channels"] for n in norms.values())
    weighted = lambda key: sum(  # noqa: E731
        n[key] * n["alive_channels"] for n in norms.values()) / alive_total

    return {
        "arm": arm.name,
        "schedule": asdict(arm),
        "run": name,
        "checkpoint": str(ckpt),
        "threshold": threshold,
        "health": health.to_json(),
        "held_out_loss": baseline_loss,
        "ablate_flagged": {
            "channels": sum(len(v) for v in flagged.values()),
            "loss": flagged_loss,
            "delta": flagged_loss - baseline_loss,
        },
        "ablate_matched": {
            "channels": sum(len(v) for v in matched.values()),
            "per_layer_k": {str(k): v for k, v in sizes.items()},
            "loss": matched_loss,
            "delta": matched_loss - baseline_loss,
        },
        "projection_norms": {
            "per_layer": {str(k): v for k, v in norms.items()},
            "alive_weighted": {"in_proj": weighted("in_proj"),
                               "out_proj": weighted("out_proj"),
                               "kernel": weighted("kernel"),
                               "alive_channels": alive_total},
        },
    }


def score_all(*, run_root: str = RUN_ROOT, report_root: str = REPORT_ROOT,
              holdout_dir: str, device: str = "cuda", tag: str = "probe",
              config: str = PROBE_CONFIG,
              arms: Sequence[ConvArm] = ARMS,
              threshold: Optional[float] = None) -> dict:
    """Score the control first, then every other arm at the control's k.

    The ordering is the point: the matched-ablation size comes from the
    control's flagged set, so the control has to be scored before anything can
    be compared to it.
    """
    control = next(arm for arm in arms if arm.is_control)
    control_score = score_arm(control, run_root=run_root, device=device,
                              holdout_dir=holdout_dir, tag=tag, config=config,
                              threshold=threshold)
    matched_k = {int(layer["layer_index"]): int(layer["dead_channels"])
                 for layer in control_score["health"]["per_layer"]}

    scores = [control_score]
    for arm in arms:
        if arm.is_control:
            continue
        scores.append(score_arm(arm, run_root=run_root, device=device,
                                holdout_dir=holdout_dir, tag=tag,
                                config=config, matched_k=matched_k,
                                threshold=threshold))
    payload = {"tag": tag, "control": control.name, "matched_k": matched_k,
               "arms": scores}
    _write_json(Path(report_root) / f"scored-{tag}.json", payload)
    return payload


# ================================================================== verdict ====

#: Preregistered, and written here rather than applied by hand later. Phase 5
#: step 7: dead fraction under 1%, projection norms within 2x the control's
#: alive-channel baseline, held-out loss no worse by more than 0.5%.
MAX_DEAD_FRACTION = 0.01
MAX_NORM_RATIO = 2.0
MAX_LOSS_REGRESSION = 0.005

#: What the positive control has to show for the sweep to mean anything. The
#: fix note's own clause 2: "a fix arm that looks clean because *nothing* died
#: in the probe is not validation". 5% is its bar for material death, reused
#: here rather than picked once the arms had reported.
CONTROL_DEATH_FLOOR = 0.05


def verdict(scored: dict) -> dict:
    """Apply the preregistered rule to a scored sweep.

    The positive control is checked first and its failure invalidates the whole
    sweep rather than any single arm: an arm reading 0% dead is uninterpretable
    until the baseline has been shown to die, which is the clause the fix note
    added after a probe where nothing died in *any* arm would have read as four
    successes.
    """
    by_name = {score["arm"]: score for score in scored["arms"]}
    control = by_name[scored["control"]]
    control_dead = control["health"].get("dead_fraction")
    control_norms = control["projection_norms"]["alive_weighted"]
    control_loss = control["held_out_loss"]

    positive_control = {
        "arm": control["arm"],
        "dead_fraction": control_dead,
        "exhibits_death": bool(control_dead and control_dead >= CONTROL_DEATH_FLOOR),
        "flagged_ablation_delta": control["ablate_flagged"]["delta"],
    }

    arms = []
    for score in scored["arms"]:
        if score["arm"] == control["arm"]:
            continue
        dead = score["health"].get("dead_fraction")
        norms = score["projection_norms"]["alive_weighted"]
        norm_ratio = {key: norms[key] / control_norms[key]
                      for key in ("in_proj", "out_proj", "kernel")}
        loss_regression = (score["held_out_loss"] - control_loss) / control_loss
        checks = {
            "dead_fraction_under_1pc": dead is not None and dead < MAX_DEAD_FRACTION,
            "norms_within_2x": max(norm_ratio.values()) <= MAX_NORM_RATIO,
            "loss_not_worse": loss_regression <= MAX_LOSS_REGRESSION,
            # The matched control: a clean arm has to lose capacity when its
            # weakest baseline-sized set goes, or its 0% is a metric that never
            # fired rather than channels that stayed alive.
            "matched_ablation_bites": score["ablate_matched"]["delta"] > 0.0,
        }
        arms.append({
            "arm": score["arm"],
            "dead_fraction": dead,
            "norm_ratio": norm_ratio,
            "loss_regression": loss_regression,
            "matched_ablation_delta": score["ablate_matched"]["delta"],
            "checks": checks,
            "passes": all(checks.values()),
        })

    return {
        "positive_control": positive_control,
        "valid": positive_control["exhibits_death"],
        "arms": arms,
        "passing": [arm["arm"] for arm in arms if arm["passes"]] if
                   positive_control["exhibits_death"] else [],
        "thresholds": {"max_dead_fraction": MAX_DEAD_FRACTION,
                       "max_norm_ratio": MAX_NORM_RATIO,
                       "max_loss_regression": MAX_LOSS_REGRESSION},
    }


# ====================================================================== cli ====

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--tag", default="probe")
    parser.add_argument("--config", default=PROBE_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("arms")

    for name in ("run", "sweep"):
        cmd = sub.add_parser(name)
        if name == "run":
            cmd.add_argument("--arm", required=True, choices=list(ARMS_BY_NAME))
        cmd.add_argument("--data-dir", required=True)
        cmd.add_argument("--val-dir", default=None)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--steps", type=int, default=PROBE_STEPS)
        cmd.add_argument("--muon-lr", type=float, default=PROBE_MUON_LR)

    score = sub.add_parser("score")
    score.add_argument("--holdout-dir", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--threshold", type=float, default=None)

    sub.add_parser("report")

    args = parser.parse_args(argv)

    if args.command == "arms":
        for arm in ARMS:
            print(json.dumps(asdict(arm)))
        return 0

    if args.command == "run":
        report = run_arm(ARMS_BY_NAME[args.arm], data_dir=args.data_dir,
                         run_root=args.run_root, device=args.device,
                         steps=args.steps, tag=args.tag, config=args.config,
                         muon_lr=args.muon_lr, val_dir=args.val_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "sweep":
        report = sweep(data_dir=args.data_dir, run_root=args.run_root,
                       report_root=args.report_root, device=args.device,
                       steps=args.steps, tag=args.tag, config=args.config,
                       muon_lr=args.muon_lr, val_dir=args.val_dir)
        print(json.dumps({"arms": [a["arm"] for a in report["arms"]]}, indent=2))
        return 0

    if args.command == "score":
        scored = score_all(run_root=args.run_root,
                           report_root=args.report_root,
                           holdout_dir=args.holdout_dir,
                           device=args.device, tag=args.tag, config=args.config,
                           threshold=args.threshold)
        decision = verdict(scored)
        _write_json(Path(args.report_root) / f"verdict-{args.tag}.json", decision)
        print(json.dumps(decision, indent=2))
        return 0

    if args.command == "report":
        path = Path(args.report_root) / f"verdict-{args.tag}.json"
        if not path.exists():
            raise SystemExit(f"no verdict at {path}; run `score` first")
        print(path.read_text())
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
