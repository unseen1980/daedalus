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
class ProbeShape:
    """The run shape an arm sweep is measured at. Shared by every arm in it.

    Two are preregistered. The probe screens the four schedules cheaply at a
    toy width and an accelerant learning rate; the paired escalation re-runs
    the survivors at the shipped 150M shape, which is the only place a result
    about *this model's* channels can be established.

    What has to carry over between them is not the learning rate but the decay
    budget. Muon decays as `w *= (1 - lr*wd)` once per optimizer step, so what
    a channel actually experiences is `sum(lr_t) * wd` -- tokens do not appear.
    `decay_clock` below is that sum, and it is why `batch_tokens` is a declared
    field rather than a throughput knob: at hero's 512k tokens/step, 500M
    tokens is 954 steps and a clock of 29, which would very likely kill nothing
    and waste the run.
    """

    name: str
    config: str
    muon_lr: float
    seq_len: int
    micro_batch: int
    batch_tokens: int
    total_tokens: int
    warmup_steps: int
    decay_frac: float
    note: str = ""

    @property
    def steps(self) -> int:
        return -(-int(self.total_tokens) // int(self.batch_tokens))

    @property
    def decay_clock(self) -> float:
        """`sum(lr_t)` over the run, which multiplied by `wd` is the exponent.

        Approximated from the WSD shape: flat through warmup's midpoint and the
        stable phase, linear to zero across the decay tail. Reported so a shape
        can be checked against hero's before it is run rather than after.
        """
        steps = self.steps
        decay = self.decay_frac * steps
        stable = steps - decay - self.warmup_steps
        return self.muon_lr * (0.5 * self.warmup_steps + max(stable, 0.0)
                               + 0.5 * decay)


#: The screen. hidden 256 at lr 0.15 for 600 steps: the shape the 2026-08-11
#: mechanism experiment established, where lr is the accelerant and `wd` is the
#: variable.
PROBE_SHAPE = ProbeShape(
    name="probe", config="conv-probe", muon_lr=PROBE_MUON_LR,
    seq_len=PROBE_SEQ_LEN, micro_batch=PROBE_MICRO_BATCH,
    batch_tokens=PROBE_MICRO_BATCH * PROBE_SEQ_LEN,
    total_tokens=PROBE_STEPS * PROBE_MICRO_BATCH * PROBE_SEQ_LEN,
    warmup_steps=50, decay_frac=0.2,
    note="the established positive control, at a toy width")

#: The escalation the plan calls for: the shipped 150M shape, 500M tokens, Muon
#: lr 0.04.
#:
#: `batch_tokens` is hero's `--tok-start`, held flat. That choice is the one
#: that decides whether this run can answer anything: 500M tokens at 131,072
#: per step is 3,815 steps and a decay clock of ~116, so the shipped arm's
#: channels see `exp(-11.6)` of shrink against a dead threshold that needs
#: ~`exp(-4.6)`. Hero's own clock was ~1,890 over 59.9B tokens, so this is 6% of
#: it -- an acceleration, and an honest one, because the mechanism under test is
#: the clock. Flat rather than ramped for the reason the probe is flat: a ramp
#: makes `lr x steps` mean something different early and late, and that is the
#: axis the arms are compared on.
#:
#: warmup and decay are `train.py`'s shipped defaults rather than the probe's,
#: because the claim being escalated is about the regime the released model was
#: trained in.
PAIRED_SHAPE = ProbeShape(
    name="paired", config="daedalus-150m", muon_lr=0.04,
    seq_len=2048, micro_batch=8, batch_tokens=131_072,
    total_tokens=500_000_000, warmup_steps=300, decay_frac=0.45,
    note="the shipped 150M shape at the plan's 500M tokens and lr 0.04")

SHAPES = {shape.name: shape for shape in (PROBE_SHAPE, PAIRED_SHAPE)}


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


def selected_arms(names: Optional[str],
                  arms: Sequence[ConvArm] = ARMS) -> Sequence[ConvArm]:
    """A named subset of the arms, always including the control, control first.

    The escalation runs a subset -- the plan advances the top two schedules,
    not all four -- and every selection criterion it is read against is stated
    relative to the control: norms within 2x the control's *alive* channels,
    held-out loss no worse than the control's by 0.5%, and a matched ablation
    sized from the control's flagged set. A subset without it is not a cheaper
    experiment, it is an unreadable one, so the control is added rather than
    required of the caller and ordered first for the same reason `sweep` runs
    it first.
    """
    if not names:
        return arms
    by_name = {arm.name: arm for arm in arms}
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(by_name)}")
    control = next(arm for arm in arms if arm.is_control)
    chosen = [control] + [by_name[name] for name in wanted
                          if name != control.name]
    return chosen


def arm_checkpoint_path(arm: ConvArm, tag: str = "probe",
                        run_root: str = RUN_ROOT,
                        config: str = PROBE_CONFIG) -> Path:
    """The checkpoint this arm writes, asked of `train.py` rather than guessed.

    `run_dir_for` is the trainer's own resolution, so the supervisor and the
    scorer cannot drift from where the run actually lands.
    """
    from train import TrainArgs, checkpoint_path_for

    args = TrainArgs(run_name=arm_run_name(arm, tag), config=config,
                     data_dir="", run_dir=None)
    resolved = Path(checkpoint_path_for(args))
    if run_root != RUN_ROOT:                    # tests may relocate the tree
        resolved = Path(run_root) / arm_run_name(arm, tag) / "checkpoint.pt"
    return resolved


def probe_train_command(arm: ConvArm, *, data_dir: str, run_name: str,
                        total_tokens: int, device: str = "cuda",
                        config: Optional[str] = None,
                        muon_lr: Optional[float] = None,
                        shape: ProbeShape = PROBE_SHAPE,
                        val_dir: Optional[str] = None) -> List[str]:
    """The exact `train.py` invocation for one arm.

    Every field except the decay flags is shared, and the shared fields are
    built here rather than per arm so an arm cannot quietly differ in seed,
    data order, batch shape or schedule. That identity is the experiment.

    `config` and `muon_lr` override the shape's, which is how the CLI's
    per-invocation flags still work; everything else comes from the shape so
    the probe and the escalation cannot drift apart in a field nobody re-reads.
    """
    command = [
        sys.executable, "train.py",
        "--run-name", run_name,
        "--config", config or shape.config,
        "--data-dir", data_dir,
        "--total-tokens", str(int(total_tokens)),
        "--micro-batch", str(shape.micro_batch),
        # Flat sequence length and batch: the ramp exists to buy throughput on
        # a long run and would make `lr x steps` mean something different early
        # and late, which is the axis the arms are compared on.
        "--seq-start", str(shape.seq_len), "--seq-end", str(shape.seq_len),
        "--tok-start", str(shape.batch_tokens),
        "--tok-end", str(shape.batch_tokens),
        "--muon-lr", repr(shape.muon_lr if muon_lr is None else muon_lr),
        "--warmup-steps", str(shape.warmup_steps),
        "--decay-frac", repr(shape.decay_frac),
        "--device", device,
        "--hub-repo", "",
        "--no-wandb",
    ]
    if val_dir:
        command += ["--val-dir", val_dir]
    return command + arm.train_flags()


def probe_total_tokens(steps: int = PROBE_STEPS,
                       shape: ProbeShape = PROBE_SHAPE) -> int:
    return steps * shape.batch_tokens


# =================================================================== running ===

def run_arm(arm: ConvArm, *, data_dir: str, run_root: str = RUN_ROOT,
            device: str = "cuda", steps: Optional[int] = None,
            tag: str = "probe", config: Optional[str] = None,
            muon_lr: Optional[float] = None, shape: ProbeShape = PROBE_SHAPE,
            val_dir: Optional[str] = None, max_attempts: int = 3,
            stall_min: float = 20.0) -> dict:
    """Train one arm under the supervisor, so an interruption continues it.

    `run_with_resume` reads the open in-flight marker beside the checkpoint, so
    a relaunch after the launching session died continues from where the arm
    got to instead of restarting it -- which is how phase 4 lost 60.3M tokens
    next to a checkpoint it never opened.

    `steps` overrides the shape's token budget, for smokes. Left None the shape
    decides, which is what a preregistered run wants: the escalation's budget
    is 500M tokens because the plan says so, not because a caller passed a
    step count that happened to multiply out to it.
    """
    from daedalus.supervise import run_with_resume, start_watchdog, stop_watchdog

    name = arm_run_name(arm, tag)
    total_tokens = (shape.total_tokens if steps is None
                    else probe_total_tokens(steps, shape))
    command = probe_train_command(
        arm, data_dir=data_dir, run_name=name, total_tokens=total_tokens,
        device=device, config=config, muon_lr=muon_lr, shape=shape,
        val_dir=val_dir)

    ckpt = arm_checkpoint_path(arm, tag, run_root, config=config or shape.config)
    run_dir = ckpt.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog = start_watchdog(name, str(run_dir), total_tokens,
                              stall_min=stall_min, supervised=True)
    try:
        report = run_with_resume(
            list(command), str(ckpt),
            max_attempts=max_attempts, halt_marker=str(run_dir / "HALTED"),
            inflight_extra={"phase": "phase5-conv-health", "arm": arm.name,
                            "schedule": asdict(arm), "shape": shape.name,
                            "steps": total_tokens // shape.batch_tokens})
    finally:
        stop_watchdog(watchdog)
    return {"arm": arm.name, "run": name, "run_dir": str(run_dir),
            "shape": shape.name, "steps": total_tokens // shape.batch_tokens,
            "total_tokens": total_tokens, "command": list(command), **report}


def sweep(*, data_dir: str, run_root: str = RUN_ROOT,
          report_root: str = REPORT_ROOT, device: str = "cuda",
          steps: Optional[int] = None, tag: str = "probe",
          config: Optional[str] = None, muon_lr: Optional[float] = None,
          shape: ProbeShape = PROBE_SHAPE, val_dir: Optional[str] = None,
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
                               config=config, muon_lr=muon_lr, shape=shape,
                               val_dir=val_dir))
        # Rewritten after every arm, so a sweep cut short still leaves the
        # arms that finished.
        _write_json(Path(report_root) / f"sweep-{tag}.json",
                    {"tag": tag, "steps": shape.steps if steps is None else steps,
                     "shape": asdict(shape), "arms": results})
    return {"tag": tag, "steps": shape.steps if steps is None else steps,
            "shape": asdict(shape), "arms": results}


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


def matched_ablation_set(layers: Sequence["object"], sizes: Dict[int, int]):
    """The control-sized weakest-alive set per layer, and what could not be met.

    `weakest_alive` slices a list, so a layer with fewer alive channels than the
    control's `k` returns everything it has and no error. That is a different
    measurement from the one the matched control claims to be: on the
    2026-08-25 probe it turned three arms' "weakest baseline-sized alive set"
    into "every channel still alive", and the resulting 1.6-1.8 nat delta read
    as the arm passing the check. Removing every live channel hurting is a
    tautology.

    Returns `(matched, requested, short)` so the caller records the size beside
    the delta and the verdict can decline to credit a set that was never
    baseline-sized. Truncation is reported rather than raised: an arm too dead
    to supply `k` has already failed on its dead fraction, and the sweep should
    still report it.
    """
    from daedalus.conv_health import weakest_alive

    matched: Dict[int, List[int]] = {}
    short: Dict[int, dict] = {}
    requested = 0
    for layer_health in layers:
        k = int(sizes.get(layer_health.layer_index, 0))
        requested += k
        if not k:
            continue
        channels = weakest_alive(layer_health, k)
        matched[layer_health.layer_index] = channels
        if len(channels) < k:
            short[layer_health.layer_index] = {"requested": k,
                                               "delivered": len(channels)}
    return matched, requested, short


def score_arm(arm: ConvArm, *, run_root: str = RUN_ROOT, holdout_dir: str,
              device: str = "cuda", tag: str = "probe",
              config: Optional[str] = None,
              shape: ProbeShape = PROBE_SHAPE,
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
                                      projection_norms)
    from daedalus.model import Daedalus
    from train import load_checkpoint

    config = config or shape.config
    threshold = DEAD_THRESHOLD if threshold is None else threshold
    name = arm_run_name(arm, tag)
    ckpt = arm_checkpoint_path(arm, tag, run_root, config=config)
    if not ckpt.exists():
        raise FileNotFoundError(f"arm {arm.name} has no checkpoint at {ckpt}")

    model = Daedalus(PRESETS[config]).to(device)
    load_checkpoint(str(ckpt), model)
    health = model_health(model, threshold=threshold)
    # Scored at the shape the arm trained at: a held-out loss read at a
    # different sequence length is not the loss the run was optimising, and the
    # ablation deltas are differences between passes over identical windows.
    windows = holdout_windows(holdout_dir, seq_len=shape.seq_len,
                              batch_size=shape.micro_batch)

    baseline_loss = held_out_loss(model, windows, device)
    flagged = {layer.layer_index: dead_channels(layer)
               for layer in health.layers}
    with ablated_model(model, flagged):
        flagged_loss = held_out_loss(model, windows, device)

    # The matched control. On a clean arm the flagged set is empty and ablating
    # it proves nothing, so this is the load-bearing half of the pair.
    sizes = matched_k or {index: len(channels)
                          for index, channels in flagged.items()}
    matched, requested, short = matched_ablation_set(health.layers, sizes)
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
            "requested_channels": requested,
            "baseline_sized": not short,
            "short_layers": {str(k): v for k, v in short.items()},
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
              config: Optional[str] = None, shape: ProbeShape = PROBE_SHAPE,
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
                              shape=shape, threshold=threshold)
    matched_k = {int(layer["layer_index"]): int(layer["dead_channels"])
                 for layer in control_score["health"]["per_layer"]}

    scores = [control_score]
    for arm in arms:
        if arm.is_control:
            continue
        scores.append(score_arm(arm, run_root=run_root, device=device,
                                holdout_dir=holdout_dir, tag=tag,
                                config=config, shape=shape,
                                matched_k=matched_k, threshold=threshold))
    payload = {"tag": tag, "control": control.name, "matched_k": matched_k,
               "shape": asdict(shape),
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


def _matched_is_baseline_sized(ablation: dict) -> bool:
    """Did the matched ablation actually get the control's k channels?

    Reads the explicit flag when the scorer wrote one and falls back to
    comparing the delivered count against the requested one, so a score written
    before either field existed is not retroactively failed on a fact it never
    recorded.
    """
    if "baseline_sized" in ablation:
        return bool(ablation["baseline_sized"])
    requested = ablation.get("requested_channels")
    if requested is None:
        return True
    return int(ablation.get("channels", 0)) >= int(requested)


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
        ablation = score["ablate_matched"]
        checks = {
            "dead_fraction_under_1pc": dead is not None and dead < MAX_DEAD_FRACTION,
            "norms_within_2x": max(norm_ratio.values()) <= MAX_NORM_RATIO,
            "loss_not_worse": loss_regression <= MAX_LOSS_REGRESSION,
            # The matched control: a clean arm has to lose capacity when its
            # weakest baseline-sized set goes, or its 0% is a metric that never
            # fired rather than channels that stayed alive. The size is half the
            # claim -- an arm too dead to supply k weakest-*alive* channels
            # instead has all of them ablated, and "removing every live channel
            # hurts" is a tautology, not evidence. Uncredited rather than
            # crashed: that arm has already failed on its dead fraction, and the
            # sweep should still report it.
            "matched_ablation_bites": (_matched_is_baseline_sized(ablation)
                                       and ablation["delta"] > 0.0),
        }
        arms.append({
            "arm": score["arm"],
            "dead_fraction": dead,
            "norm_ratio": norm_ratio,
            "loss_regression": loss_regression,
            "matched_ablation_delta": ablation["delta"],
            "matched_ablation_channels": ablation.get("channels"),
            "matched_ablation_requested": ablation.get("requested_channels"),
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


# =================================================================== report ====

#: The phase deliverable. Written beside the verdicts it is assembled from,
#: never inside a run directory, for the reason `REPORT_ROOT` exists.
REPORT_NAME = "phase5-conv-decay.md"

#: Both preregistered stages, in the order they were run. The screen is cheap
#: and the escalation is the one a claim about this model can rest on, so a
#: reader has to see which stage a number came from.
STAGE_TITLES = {
    "probe": "Screen -- hidden 256, Muon lr 0.15, 600 steps",
    "paired": "Escalation -- the shipped 150M shape, 500M tokens, Muon lr 0.04",
}

#: Below this the flagged channels moved held-out loss by nothing a float can
#: distinguish from zero, which is the reading that makes "dead" mean dead.
FREE_ABLATION_NATS = 1e-4


def load_stage(report_root, tag: str) -> Optional[dict]:
    """One stage's verdict and the shape it was measured at, or None.

    A stage that was never scored is absent rather than empty: the report is
    assembled from artifacts on disk so that every number in it is the number
    in the file it cites, and inventing a stage would break exactly that.
    """
    root = Path(report_root)
    decision_path = root / f"verdict-{tag}.json"
    if not decision_path.exists():
        return None
    scored_path = root / f"scored-{tag}.json"
    shape = {}
    if scored_path.exists():
        shape = json.loads(scored_path.read_text()).get("shape") or {}
    return {"tag": tag, "verdict": json.loads(decision_path.read_text()),
            "shape": shape}


def recommendation(stages: Sequence[dict]) -> dict:
    """What the sweep licenses, derived from the verdicts rather than written.

    The decisive stage is the *last valid* one, which is the escalation when it
    ran: a screen at hidden 256 and lr 0.15 can rank schedules but cannot
    establish anything about this model's channels, and a stage whose positive
    control did not die measured nothing at all. If no stage is valid the
    sweep has no finding, which is a different answer from "no arm passed" and
    is reported as one.
    """
    valid = [stage for stage in stages if stage["verdict"].get("valid")]
    if not valid:
        return {
            "decisive_stage": None,
            "selected": None,
            "negative_result": True,
            "reason": ("no stage's positive control exhibited channel death, "
                       "so no arm in this sweep is readable"),
            "findings": [],
        }

    decisive = valid[-1]
    decision = decisive["verdict"]
    passing = decision.get("passing") or []
    findings = []

    control = decision["positive_control"]
    delta = abs(float(control.get("flagged_ablation_delta") or 0.0))
    if delta < FREE_ABLATION_NATS:
        findings.append({
            "kind": "control-death-is-free",
            "stage": decisive["tag"],
            "dead_fraction": control.get("dead_fraction"),
            "flagged_ablation_delta": control.get("flagged_ablation_delta"),
        })

    arms = decision.get("arms") or []

    # When no arm could supply the control's k anywhere, the ablation clause
    # decided nothing, and a reader counting four failed clauses would
    # overstate how much evidence the rule actually applied. Reported rather
    # than repaired: every one of these arms fails on its dead fraction too, so
    # crediting them changes no verdict, and rewriting a clause once it is
    # known which arms it rejects is the move the plan forbids.
    credited = [arm for arm in arms
                if (arm.get("checks") or {}).get("matched_ablation_bites")]
    if arms and not credited:
        decisive_on_ablation = [
            arm["arm"] for arm in arms
            if all(passed for name, passed in (arm.get("checks") or {}).items()
                   if name != "matched_ablation_bites")]
        findings.append({
            "kind": "ablation-clause-never-applied",
            "stage": decisive["tag"],
            "arms": [arm["arm"] for arm in arms],
            "decided_any_verdict": decisive_on_ablation,
        })

    survivors = [arm for arm in arms if arm.get("dead_fraction") is not None]
    if survivors and not passing:
        best = min(survivors, key=lambda arm: arm["dead_fraction"])
        findings.append({
            "kind": "no-arm-approaches-the-bar",
            "stage": decisive["tag"],
            "arm": best["arm"],
            "dead_fraction": best["dead_fraction"],
            "bar": MAX_DEAD_FRACTION,
        })

    # The trade the escalation existed to price: an arm can buy fewer dead
    # channels with projection growth, and growth is driven by the decay clock,
    # so its cost is only visible once the same arm has been read at both
    # shapes.
    for arm in arms:
        ratios = arm.get("norm_ratio") or {}
        if not ratios or max(ratios.values()) <= MAX_NORM_RATIO:
            continue
        earlier = _arm_in_stages(stages[:-1] if len(stages) > 1 else [],
                                 arm["arm"])
        findings.append({
            "kind": "death-traded-for-norm-growth",
            "stage": decisive["tag"],
            "arm": arm["arm"],
            "norm_ratio": ratios,
            "earlier_norm_ratio": (earlier or {}).get("norm_ratio"),
            "limit": MAX_NORM_RATIO,
        })

    if passing:
        return {
            "decisive_stage": decisive["tag"],
            "selected": passing[0],
            "negative_result": False,
            "reason": (f"{passing[0]} cleared every clause of the "
                       f"preregistered rule at the {decisive['tag']} stage"),
            "findings": findings,
        }
    return {
        "decisive_stage": decisive["tag"],
        "selected": None,
        "negative_result": True,
        "reason": ("no tested schedule cleared the preregistered rule; "
                   "recording the negative result rather than relaxing a bar "
                   "after seeing the numbers"),
        "findings": findings,
    }


def _arm_in_stages(stages: Sequence[dict], name: str) -> Optional[dict]:
    for stage in reversed(list(stages)):
        for arm in stage["verdict"].get("arms") or []:
            if arm.get("arm") == name:
                return arm
    return None


def _pct(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def _check_mark(passed: bool) -> str:
    return "pass" if passed else "FAIL"


def _ablation_cell(arm: dict) -> str:
    """The matched ablation's delta *and* the size that earns it a reading.

    A big delta from a set that could not be baseline-sized means every live
    channel was removed, which hurts by construction. Reporting the delta alone
    is how that reads as the check passing.
    """
    delivered = arm.get("matched_ablation_channels")
    requested = arm.get("matched_ablation_requested")
    delta = arm.get("matched_ablation_delta")
    size = ("n/a" if delivered is None or requested is None
            else f"{delivered}/{requested}")
    credited = (arm.get("checks") or {}).get("matched_ablation_bites")
    note = "credited" if credited else "uncredited"
    return f"{delta:+.3f} nats over {size} ({note})"


def _stage_table(stage: dict) -> List[str]:
    decision = stage["verdict"]
    rows = [
        "| arm | dead | in_proj | out_proj | kernel | held-out loss | "
        "matched ablation | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in decision.get("arms") or []:
        ratios = arm.get("norm_ratio") or {}
        rows.append(
            f"| `{arm['arm']}` | {_pct(arm.get('dead_fraction'))} | "
            f"{ratios.get('in_proj', float('nan')):.2f}x | "
            f"{ratios.get('out_proj', float('nan')):.2f}x | "
            f"{ratios.get('kernel', float('nan')):.2f}x | "
            f"{_pct(arm.get('loss_regression'))} | "
            f"{_ablation_cell(arm)} | "
            f"{_check_mark(bool(arm.get('passes')))} |")
    return rows


def _stage_section(stage: dict) -> List[str]:
    decision = stage["verdict"]
    control = decision["positive_control"]
    shape = stage.get("shape") or {}
    title = STAGE_TITLES.get(stage["tag"], stage["tag"])
    lines = [f"## {title}", ""]
    if shape:
        lines += [
            f"`{shape.get('config', '?')}` at "
            f"{int(shape.get('total_tokens', 0)):,} tokens, Muon lr "
            f"{shape.get('muon_lr', '?')}.",
            "",
        ]

    if not decision.get("valid"):
        lines += [
            f"**Unreadable.** The positive control `{control['arm']}` reached "
            f"{_pct(control.get('dead_fraction'))} dead channels, below the "
            f"{_pct(CONTROL_DEATH_FLOOR, 0)} floor this stage needs to show it "
            f"can detect death at all. No arm below is interpretable: a 0% "
            f"reading here is a metric that never fired, not channels that "
            f"stayed alive.",
            "",
        ]
        return lines + _stage_table(stage) + [""]

    lines += [
        f"Positive control `{control['arm']}` died at "
        f"{_pct(control.get('dead_fraction'))}, and removing every channel it "
        f"flagged moved held-out loss by "
        f"{control.get('flagged_ablation_delta'):+.2e} nats. The stage is "
        f"readable, and those channels were carrying nothing.",
        "",
        "Norm columns are the arm's alive-channel mean over the control's, "
        "held-out loss is relative to the control, and negative is better.",
        "",
    ]
    return lines + _stage_table(stage) + [""]


def _cross_shape_section(stages: Sequence[dict]) -> List[str]:
    """The same arm at both shapes, which is what the escalation bought.

    The decay clock is `sum(lr_t) * wd`, so an arm's behaviour is a function of
    the clock rather than of the learning rate, and a screen at 600 steps can
    only rank schedules. Whether the ranking survives a 6x longer clock is the
    question the escalation answers, and it is answered by this table.
    """
    if len(stages) < 2:
        return []
    names = [arm["arm"] for arm in stages[-1]["verdict"].get("arms") or []]
    rows = ["| arm | dead | max norm ratio |", "|---|---|---|"]

    # The control first, and it is not decoration. Every norm ratio in this
    # table is *against the control of its own stage*, and the control's own
    # dead fraction moved between the two shapes -- so an arm whose death fell
    # from one stage to the next may only have followed its baseline down.
    # Without this row the table invites reading that as the arm improving.
    control_cells, control_name = [], "control"
    for stage in stages:
        control = stage["verdict"].get("positive_control") or {}
        control_name = control.get("arm") or control_name
        control_cells.append(_pct(control.get("dead_fraction")))
    rows.append(f"| `{control_name}` (control) | "
                f"{' -> '.join(control_cells)} | 1.00x by definition |")

    for name in names:
        cells = []
        for stage in stages:
            arm = _arm_in_stages([stage], name)
            if arm is None:
                cells.append(("n/a", "n/a"))
                continue
            ratios = (arm.get("norm_ratio") or {}).values()
            cells.append((_pct(arm.get("dead_fraction")),
                          f"{max(ratios):.2f}x" if ratios else "n/a"))
        dead = " -> ".join(cell[0] for cell in cells)
        norm = " -> ".join(cell[1] for cell in cells)
        rows.append(f"| `{name}` | {dead} | {norm} |")
    return [
        "## The same arms at both shapes",
        "",
        "Left to right: " + " -> ".join(
            f"`{stage['tag']}`" for stage in stages) + ". Muon decays once per "
        "optimizer step, so what a channel experiences is `sum(lr_t) * wd` and "
        "not tokens. The escalation runs a ~6x longer decay clock than the "
        "screen, and an arm whose cost grows with the clock is an arm the "
        "screen would have passed.",
        "",
    ] + rows + [""]


def _findings_lines(advice: dict) -> List[str]:
    lines = []
    for finding in advice.get("findings") or []:
        if finding["kind"] == "control-death-is-free":
            lines.append(
                f"- At the `{finding['stage']}` stage the shipped decay left "
                f"{_pct(finding['dead_fraction'])} of conv channels dead and "
                f"removing all of them cost "
                f"{finding['flagged_ablation_delta']:+.2e} nats. The death is "
                f"real and the channels were not being used, so this is a "
                f"capacity-allocation result: a schedule that keeps them alive "
                f"has to show they then *earn* their place, not merely that "
                f"they are alive.")
        elif finding["kind"] == "no-arm-approaches-the-bar":
            lines.append(
                f"- The lowest dead fraction any arm reached at the "
                f"`{finding['stage']}` stage was {_pct(finding['dead_fraction'])} "
                f"(`{finding['arm']}`), against a {_pct(finding['bar'])} bar. "
                f"No tested schedule is close, so this is not a threshold that "
                f"a slightly different ramp would have cleared.")
        elif finding["kind"] == "ablation-clause-never-applied":
            decided = finding["decided_any_verdict"]
            consequence = (
                f"It was decisive for {', '.join('`' + name + '`' for name in decided)}, "
                f"which cleared every other clause -- so that rejection rests "
                f"on a set that was never baseline-sized and should be re-read "
                f"before it is relied on."
                if decided else
                "It decided no verdict: every arm it declined to credit also "
                "failed on its dead fraction, so the rule's outcome would be "
                "unchanged either way.")
            lines.append(
                f"- The matched-ablation clause could not be met by any arm at "
                f"the `{finding['stage']}` stage. An arm needs the control's "
                f"*per-layer* count of weakest-alive channels to spare, and a "
                f"control that killed most of a layer requests more than any "
                f"arm has left there. {consequence}")
        elif finding["kind"] == "death-traded-for-norm-growth":
            ratios = finding["norm_ratio"]
            worst = max(ratios, key=lambda key: ratios[key])
            earlier = finding.get("earlier_norm_ratio") or {}
            trend = ""
            if earlier.get(worst) is not None:
                trend = (f" -- at the screen the same arm read "
                         f"{earlier[worst]:.2f}x, so the cost grows with the "
                         f"decay clock rather than staying put")
            lines.append(
                f"- `{finding['arm']}` bought its lower dead fraction with "
                f"projection growth: {worst} reached {ratios[worst]:.2f}x the "
                f"control's alive-channel baseline against a "
                f"{finding['limit']:g}x limit{trend}. That is the equilibrium "
                f"objection that kept a zero-decay arm out of this sweep, now "
                f"measured on an arm that is in it.")
    return lines


def render_report(stages: Sequence[dict]) -> str:
    """The phase 5 deliverable: what was measured, and what it does not license."""
    advice = recommendation(stages)
    lines = [
        "# Phase 5: does a decay schedule stop ShortConv channel death?",
        "",
        "Four schedules for the conv projections, one variable between them, "
        "read on the coupled `in_proj` x kernel x `out_proj` instrument rather "
        "than on the shipped weight proxy. The fix under test *is* a change to "
        "weight decay, so a magnitude metric can be satisfied by an arm where "
        "nothing shrank as easily as by one where nothing died.",
        "",
        "**Scope: V2 only.** Every arm here is trained from initialization at "
        "a proxy shape. Nothing in this phase touches the released V1 weights, "
        "and no result below says a dead channel in the released model was "
        "revived -- a channel that collapsed during a 59.9B-token run is not "
        "brought back by choosing a different schedule for a future run.",
        "",
        "## The preregistered rule",
        "",
        f"> An arm is selected only when its dead fraction is under "
        f"{_pct(MAX_DEAD_FRACTION, 0)}, its alive-channel projection norms "
        f"stay within {MAX_NORM_RATIO:g}x the control's, its held-out loss is "
        f"no worse than the control's by more than "
        f"{_pct(MAX_LOSS_REGRESSION, 1)}, and removing its weakest "
        f"*baseline-sized* channel set measurably costs held-out loss.",
        "",
        f"A stage is readable only if the control itself dies by at least "
        f"{_pct(CONTROL_DEATH_FLOOR, 0)}. These four thresholds are constants "
        f"in `verdict()`, and `runs/conv-health/verdict-probe.json` records the "
        f"same values from before the escalation was launched, so none of them "
        f"moved after the results landed.",
        "",
    ]

    for stage in stages:
        lines += _stage_section(stage)
    lines += _cross_shape_section(stages)

    lines += ["## Verdict", ""]
    if advice["negative_result"]:
        lines += [f"**Negative result.** {advice['reason'].capitalize()}.", ""]
    else:
        lines += [f"**Selected: `{advice['selected']}`.** "
                  f"{advice['reason'].capitalize()}.", ""]
    findings = _findings_lines(advice)
    if findings:
        lines += findings + [""]

    lines += [
        "## What this licenses for V2",
        "",
        "- The instrument works and the shipped schedule's death is real, so a "
        "future from-scratch V2 can be measured on the same rule without "
        "re-establishing the baseline.",
        "- No schedule in this sweep is a recipe. A V2 candidate has to clear "
        "the dead-fraction bar *and* the norm bar at a decay clock at least as "
        "long as the escalation's; every arm here failed at least one.",
        "- The dead channels cost nothing to remove, so the honest framing of "
        "the opportunity is parameters that were paid for and not used, not "
        "quality that was lost. Any claimed gain from reviving them has to be "
        "shown as a held-out improvement, not as a higher alive count.",
    ]
    # Sections are built with their own trailing blank, so the document ends in
    # however many the last one left. `git diff --check` rejects a blank line at
    # EOF, which would make the report uncommittable by the approved wrapper.
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def write_report(report_root=REPORT_ROOT, tags: Sequence[str] = ("probe", "paired"),
                 out=None) -> Path:
    """Assemble the report from whichever stages were scored."""
    stages = [stage for stage in
              (load_stage(report_root, tag) for tag in tags)
              if stage is not None]
    if not stages:
        raise SystemExit(
            f"no verdict for {list(tags)} under {report_root}; run `score` first")
    path = Path(out) if out else Path(report_root) / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(render_report(stages))
    os.replace(temporary, path)
    return path


# ====================================================================== cli ====

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--tag", default="probe")
    parser.add_argument("--config", default=None,
                        help="override the shape's model preset")
    parser.add_argument("--shape", default=PROBE_SHAPE.name, choices=list(SHAPES),
                        help="which preregistered run shape to use")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("arms")

    for name in ("run", "sweep"):
        cmd = sub.add_parser(name)
        if name == "run":
            cmd.add_argument("--arm", required=True, choices=list(ARMS_BY_NAME))
        else:
            cmd.add_argument("--arms", default=None,
                             help="comma-separated subset, control first; "
                                  "default every preregistered arm")
        cmd.add_argument("--data-dir", required=True)
        cmd.add_argument("--val-dir", default=None)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--steps", type=int, default=None,
                         help="override the shape's token budget (smokes only)")
        cmd.add_argument("--muon-lr", type=float, default=None)

    score = sub.add_parser("score")
    score.add_argument("--holdout-dir", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--threshold", type=float, default=None)
    score.add_argument("--arms", default=None,
                       help="comma-separated subset; the control is required")

    report = sub.add_parser("report")
    report.add_argument("--tags", default="probe,paired",
                        help="stages to assemble, in order; missing ones are "
                             "skipped rather than invented")
    report.add_argument("--out", default=None)
    report.add_argument("--json", action="store_true",
                        help="print the single --tag verdict instead")

    sub.add_parser("shapes")

    args = parser.parse_args(argv)
    shape = SHAPES[getattr(args, "shape", PROBE_SHAPE.name)]

    if args.command == "arms":
        for arm in ARMS:
            print(json.dumps(asdict(arm)))
        return 0

    if args.command == "shapes":
        for name, candidate in SHAPES.items():
            print(json.dumps({**asdict(candidate), "steps": candidate.steps,
                              "decay_clock": candidate.decay_clock}))
        return 0

    if args.command == "run":
        report = run_arm(ARMS_BY_NAME[args.arm], data_dir=args.data_dir,
                         run_root=args.run_root, device=args.device,
                         steps=args.steps, tag=args.tag, config=args.config,
                         muon_lr=args.muon_lr, shape=shape,
                         val_dir=args.val_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "sweep":
        report = sweep(data_dir=args.data_dir, run_root=args.run_root,
                       report_root=args.report_root, device=args.device,
                       steps=args.steps, tag=args.tag, config=args.config,
                       muon_lr=args.muon_lr, shape=shape,
                       val_dir=args.val_dir, arms=selected_arms(args.arms))
        print(json.dumps({"arms": [a["arm"] for a in report["arms"]]}, indent=2))
        return 0

    if args.command == "score":
        scored = score_all(run_root=args.run_root,
                           report_root=args.report_root,
                           holdout_dir=args.holdout_dir,
                           device=args.device, tag=args.tag, config=args.config,
                           shape=shape, arms=selected_arms(args.arms),
                           threshold=args.threshold)
        decision = verdict(scored)
        _write_json(Path(args.report_root) / f"verdict-{args.tag}.json", decision)
        print(json.dumps(decision, indent=2))
        return 0

    if args.command == "report":
        if args.json:
            path = Path(args.report_root) / f"verdict-{args.tag}.json"
            if not path.exists():
                raise SystemExit(f"no verdict at {path}; run `score` first")
            print(path.read_text())
            return 0
        tags = tuple(tag.strip() for tag in args.tags.split(",") if tag.strip())
        path = write_report(args.report_root, tags=tags, out=args.out)
        print(path.read_text())
        print(f"wrote {path}")
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
