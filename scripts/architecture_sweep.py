"""Phase 6 stages A and B: what do attention layers and KV heads actually buy?

`daedalus/arch_space.py` answered the free half of this phase -- which shapes
are stock-LFM2-compatible, parameter-matched and inside the KV budget. It cannot
answer the half that costs GPU hours: whether a stack with a third of its layers
attending is *better* than one with a twelfth, and whether four KV heads earn
their cache over one. Stage A is the screen that ranks those, cheaply, before
stage B re-runs the survivors at a scale a recommendation can rest on.

The grid is `daedalus.config`'s fifteen `arch-a{n}-kv{k}` presets: five attention
fractions crossed with three KV-head counts, every other field held. Three
decisions about how it is *run* are worth stating, because each is a way a
screen like this quietly stops measuring what it claims to.

**The screen runs at the shipped 2,048 context, not at a cheaper 1,024.** Halving
the context would nearly double throughput and would bias the result in exactly
the direction this phase is hoping to find: attention is worth least at short
context, so a 1,024-token screen flatters every attention-sparse arm and would
recommend cuts that a real 2,048-token model cannot afford. The KV cache is the
whole subject, and its cost and its benefit both scale with context.

**The control is a grid point, not the shipped checkpoint.** `arch-a8-kv4` is the
shipped model's own ratio -- attention every third layer, four KV heads -- at the
probe's width and depth. Reading the arms against the released 160M model would
compare architectures *and* 55M parameters *and* two different training budgets.
Note that the shipped ratio at depth 24 costs 8,192 KV bytes per context token
against the shipped model's 6,144: at a fixed fraction, depth buys KV cost, which
is a finding about the successor rather than a defect in the control.

**A truncated sweep still reads as a curve.** Arms run control-first and then in
descending KV cost, so a sweep cut short by the deadline leaves a contiguous walk
down from the shipped ratio rather than a scatter of unrelated points. The
control runs first for the reason it does in phase 5: an arm's number means
nothing until the point it is measured against exists.

**The residual parameter spread favours the arms this phase hopes will win, and
that has to be carried into the scoring.** A conv block costs 1.05M parameters
where an attention block costs 0.59M-0.79M, so trading attention for convolution
*adds* parameters: the sparsest arm carries 3.05% more than the densest, and the
grid spans +/-1.5% about its midpoint (`parameter_spread`). Holding the FFN fixed
is the tightest matching available here -- solving it per arm would snap arms as
much as 4.5% apart, because one 256-wide step is 9.0% of this model -- but
tightest is not zero. The bias points toward attention-sparse arms, which is
exactly the direction a phase looking for KV savings would like to find, so a
stage-A quality win inside that margin is not a win. Every arm's exact parameter
count is in its sweep record for that reason.

**Stage B is the same grid at 150M over 250M tokens, and it is a `--shape`
rather than a second module.** What the two stages share is the whole
experiment: the arm names, the interleaved layout, the context, the tokens per
step, the learning rates and the schedule shape. What differs is width, FFN and
budget. Expressing that as one more `StageShape` plus one more preset family
keeps the shared half literally shared, so stage B cannot drift in a field
nobody re-reads -- which is the failure the generated presets exist to prevent.
The cost is that one arm name now maps to two run directories, and
`foreign_run` is what stops a mistyped `--tag` from resolving that ambiguity by
overwriting a finished arm.

Scoring lives in `architecture_report.py`, next to the evaluators it reads.

Subcommands: `arms`, `shapes`, `run`, `sweep`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daedalus.arch_space import (candidate_from_config,
                                 kv_bytes_per_context_token)
from daedalus.config import (ARCH_PROBE_ATTENTION_BLOCKS, ARCH_PROBE_CONTROL,
                             ARCH_PROBE_KV_HEADS, PRESETS,
                             arch_probe_preset_name, arch_stageb_preset_name)

#: Where `train.py` puts a run, and therefore the only place a supervisor may
#: look for its checkpoint. Asked of the trainer rather than composed here --
#: see `arm_checkpoint_path`, and `runs/conv-health` for what composing it cost.
RUN_ROOT = "runs"

#: Sweep and score artifacts, kept out of the run directories so a report is
#: never mistaken for a checkpoint.
REPORT_ROOT = "runs/architecture"


@dataclass(frozen=True)
class StageShape:
    """The run shape a stage's arms share. Shared *is* the experiment.

    Every field here is identical across arms by construction, so an arm cannot
    quietly differ in batch shape, schedule or learning rate. Only the preset
    changes, and the preset differs only in attention count and KV heads.
    """

    name: str
    seq_len: int
    micro_batch: int
    batch_tokens: int
    total_tokens: int
    warmup_steps: int
    decay_frac: float
    muon_lr: float
    adam_lr: float
    #: Which family of presets this stage's arms name. The grid points are the
    #: same two knobs at both stages; the preset behind a point is not.
    preset_family: str
    #: The run-directory prefix this stage owns. Derived from the shape rather
    #: than defaulted on the CLI so that `--shape stage-b` cannot land in
    #: `runs/arch-stagea-*` because a `--tag` was forgotten -- which would train
    #: a 159M arm on top of a finished 105M one.
    tag: str
    note: str = ""

    @property
    def steps(self) -> int:
        return -(-int(self.total_tokens) // int(self.batch_tokens))

    @property
    def grad_accum(self) -> int:
        return max(1, round(self.batch_tokens / (self.micro_batch * self.seq_len)))


#: Stage A, preregistered.
#:
#: The token budget and schedule are phase 4's probe recipe re-used rather than
#: re-derived: 1,536 optimizer steps at 65,536 tokens is within 7% of the 1,636
#: the tokenizer probes ran, so the warmup and decay fractions that produced a
#: fully-decayed WSD curve there produce one here. What changed is the context
#: -- 2,048 rather than 1,024 -- which is the axis this phase cannot economise
#: on, so the batch is halved in *windows* to keep the tokens per step, and
#: hence the schedule, where they were.
#:
#: The budget is a whole number of steps (1,536 x 65,536) rather than a round
#: 100M: a partial final step trains on a truncated batch and makes the arm's
#: step count depend on rounding rather than on the plan.
STAGE_A = StageShape(
    name="stage-a",
    seq_len=2048,
    micro_batch=8,
    batch_tokens=65_536,
    total_tokens=100_663_296,
    warmup_steps=100,
    decay_frac=0.8,
    muon_lr=0.02,
    adam_lr=3e-4,
    preset_family="stage-a",
    tag="stagea",
    note="the fifteen-point attention x KV-head screen at ~105M parameters",
)

#: Stage B, preregistered: the survivors re-run at the scale the plan allows a
#: recommendation to rest on.
#:
#: **250M tokens, as 3,840 whole steps of 65,536.** Exactly 2.5x stage A's
#: budget at the identical tokens per step, so the two stages differ in how long
#: the schedule runs and not in what a step is. A round 250,000,000 would leave
#: a truncated final batch and make the step count a rounding artefact.
#:
#: **Warmup scales with the run, decay does not.** 250 steps is stage A's 6.5%
#: of a 2.5x longer run, and `decay_frac` is unchanged, so the WSD curve has the
#: same shape at both stages -- a schedule that meant something different early
#: and late would confound the scale-up it is supposed to measure.
#:
#: **The learning rates are stage A's, which are also the shipped model's.** The
#: hero run trained this parameter count at `--muon-lr 0.02`, and stage A ran
#: its probes there too. Re-tuning between stages would be an unpreregistered
#: decision that moves the comparison, and there is no evidence asking for one.
#:
#: **The micro-batch halves to 4 and the accumulation doubles to 8.** Tokens per
#: step are unchanged; what changes is peak activation memory, which scales with
#: `micro_batch * hidden_size`. Stage A is known to fit at 8 x 512; 4 x 768 is
#: 25% *under* that known-good point, so a 24GB card holds the wider model with
#: headroom. Discovering otherwise costs an OOM two hours into a seven-hour
#: sweep, which is the expensive way to learn it.
STAGE_B = StageShape(
    name="stage-b",
    seq_len=2048,
    micro_batch=4,
    batch_tokens=65_536,
    total_tokens=251_658_240,
    warmup_steps=250,
    decay_frac=0.8,
    muon_lr=0.02,
    adam_lr=3e-4,
    preset_family="stage-b",
    tag="stageb",
    note="the stage-A survivors re-run at ~159M parameters over 250M tokens",
)

SHAPES = {shape.name: shape for shape in (STAGE_A, STAGE_B)}

#: How a grid point names its preset in each family. The two knobs are the same
#: at both stages; the width, FFN and therefore the preset are not.
PRESET_NAMERS = {
    "stage-a": arch_probe_preset_name,
    "stage-b": arch_stageb_preset_name,
}


@dataclass(frozen=True)
class ArchArm:
    """One grid point: a preset name plus the two knobs it varies."""

    name: str
    config: str
    num_attention_blocks: int
    num_key_value_heads: int

    @property
    def is_control(self) -> bool:
        return (self.num_attention_blocks,
                self.num_key_value_heads) == ARCH_PROBE_CONTROL

    @property
    def kv_bytes_per_context_token(self) -> int:
        return kv_bytes_per_context_token(PRESETS[self.config])

    def describe(self) -> dict:
        """The analytic columns, from the same functions that screened the space.

        Read off `arch_space` rather than recomputed so a number in a sweep
        artifact and the same number in the phase 6 screen cannot disagree.
        """
        from daedalus.arch_space import describe as describe_candidate

        record = describe_candidate(
            candidate_from_config(self.name, PRESETS[self.config]))
        record["arm"] = self.name
        record["preset"] = self.config
        record["is_control"] = self.is_control
        return record


def _build_arms(preset_family: str = "stage-a") -> List[ArchArm]:
    """The grid, control first and then descending KV cost.

    Ordering is not cosmetic. A sweep the deadline cuts short leaves whatever
    ran, and descending cost means that prefix is a contiguous walk down from
    the shipped ratio -- readable as a curve -- rather than an arbitrary subset
    whose gaps have to be explained.

    An arm's `name` is its grid point and is the same in every family, so
    `--arms a4-kv2` selects the same *shape* at either stage while `config`
    resolves to that stage's preset. The two stages then share one vocabulary of
    arm names, which is what lets stage A's decision be handed to stage B.
    """
    namer = PRESET_NAMERS[preset_family]
    arms = [
        ArchArm(name=f"a{blocks}-kv{kv}",
                config=namer(blocks, kv),
                num_attention_blocks=blocks, num_key_value_heads=kv)
        for blocks in ARCH_PROBE_ATTENTION_BLOCKS
        for kv in ARCH_PROBE_KV_HEADS
    ]
    control = next(arm for arm in arms if arm.is_control)
    rest = sorted((arm for arm in arms if not arm.is_control),
                  key=lambda arm: (-arm.kv_bytes_per_context_token, arm.name))
    return [control] + rest


ARMS_BY_FAMILY = {family: tuple(_build_arms(family))
                  for family in PRESET_NAMERS}

#: Stage A's grid stays the module-level default: it is what every existing
#: caller, artifact and scorecard means by "the arms".
ARMS: Sequence[ArchArm] = ARMS_BY_FAMILY["stage-a"]
ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
CONTROL = ARMS[0]


def arms_for(shape: "StageShape") -> Sequence[ArchArm]:
    """The grid at this stage's scale, control first."""
    return ARMS_BY_FAMILY[shape.preset_family]


def parameter_spread(arms: Sequence[ArchArm] = ARMS) -> dict:
    """How far from parameter-matched the grid actually is, and which way.

    Reported rather than asserted away. The residual is structural -- a conv
    block is dearer than an attention block, so cutting attention adds
    parameters -- and it points at the arms a KV-savings phase would like to
    see win. A later scoring slice that compares BPB across these arms without
    this number in front of it is comparing architecture plus up to 3% of model.
    """
    counts = {arm.name: PRESETS[arm.config].param_count()["total"]
              for arm in arms}
    smallest = min(counts, key=counts.get)
    largest = max(counts, key=counts.get)
    midpoint = (counts[smallest] + counts[largest]) / 2.0
    return {
        "per_arm": counts,
        "min": counts[smallest],
        "max": counts[largest],
        "min_arm": smallest,
        "max_arm": largest,
        "midpoint": midpoint,
        # Half-width about the midpoint, which is the form `arch_space`'s
        # `PARAM_MATCH_TOLERANCE` is stated in.
        "max_drift_from_midpoint": (counts[largest] - midpoint) / midpoint,
        "spread_over_min": (counts[largest] - counts[smallest]) / counts[smallest],
        "favours": largest,
    }


def selected_arms(names: Optional[str],
                  arms: Sequence[ArchArm] = ARMS) -> Sequence[ArchArm]:
    """A named subset, always including the control and always control-first.

    Same rule as phase 5's, for the same reason: every stage-A column is read
    relative to the control, so a subset without it is not a cheaper experiment
    but an unreadable one.
    """
    if not names:
        return arms
    by_name = {arm.name: arm for arm in arms}
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(by_name)}")
    control = next(arm for arm in arms if arm.is_control)
    return [control] + [by_name[name] for name in wanted
                        if name != control.name]


def arm_run_name(arm: ArchArm, tag: str = "stagea") -> str:
    return f"arch-{tag}-{arm.name}"


def arm_checkpoint_path(arm: ArchArm, tag: str = "stagea",
                        run_root: str = RUN_ROOT) -> Path:
    """The checkpoint this arm writes, asked of `train.py` rather than guessed.

    The phase 5 smoke found what composing it instead costs: the supervisor was
    handed a path that never appears, so the in-flight marker sat beside a file
    nothing ever wrote and every relaunch would have restarted from step 0 with
    no log saying so.
    """
    from train import TrainArgs, checkpoint_path_for

    args = TrainArgs(run_name=arm_run_name(arm, tag), config=arm.config,
                     data_dir="", run_dir=None)
    resolved = Path(checkpoint_path_for(args))
    if run_root != RUN_ROOT:                    # tests may relocate the tree
        resolved = Path(run_root) / arm_run_name(arm, tag) / "checkpoint.pt"
    return resolved


def train_command(arm: ArchArm, *, data_dir: str, run_name: str,
                  total_tokens: int, device: str = "cuda",
                  shape: StageShape = STAGE_A,
                  val_dir: Optional[str] = None) -> List[str]:
    """The exact `train.py` invocation for one arm.

    Everything except `--config` comes from the shape, so two arms cannot differ
    in seed, data order, batch shape, schedule or learning rate. Sequence length
    and batch tokens are flat rather than ramped for the reason phase 5's were:
    a ramp makes the schedule mean something different early and late, and the
    schedule is held constant precisely so the architecture is the variable.
    """
    command = [
        sys.executable, "train.py",
        "--run-name", run_name,
        "--config", arm.config,
        "--data-dir", data_dir,
        "--total-tokens", str(int(total_tokens)),
        "--micro-batch", str(shape.micro_batch),
        "--seq-start", str(shape.seq_len), "--seq-end", str(shape.seq_len),
        "--tok-start", str(shape.batch_tokens),
        "--tok-end", str(shape.batch_tokens),
        "--muon-lr", repr(shape.muon_lr),
        "--adam-lr", repr(shape.adam_lr),
        "--warmup-steps", str(shape.warmup_steps),
        "--decay-frac", repr(shape.decay_frac),
        "--device", device,
        "--hub-repo", "",
        "--no-wandb",
    ]
    if val_dir:
        command += ["--val-dir", val_dir]
    return command


# =================================================================== running ===

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def finished_run(command: Sequence[str], ckpt) -> Optional[dict]:
    """The closed marker of an identical run that already finished, if any.

    All four conditions matter. `outcome == "completed"` rather than
    `completed is True`, because `mark_inflight_done` closes the marker for a
    watchdog halt and for exhausted attempts too, and neither of those is a
    result worth keeping. The command must match exactly, so a changed budget,
    schedule or preset is a different experiment and runs. And the checkpoint
    must actually be there, because the marker records what was *intended*.
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


def _preset_in(command: Sequence[str]) -> Optional[str]:
    """The `--config` a recorded command names, or None."""
    parts = list(command)
    for index, part in enumerate(parts[:-1]):
        if part == "--config":
            return parts[index + 1]
    return None


def foreign_run(command: Sequence[str], ckpt) -> Optional[str]:
    """The preset a marker in this run directory names, if it is not ours.

    Two stages share one vocabulary of arm names, so `a4-kv2` is a stage-A run
    directory and a stage-B one depending only on the tag -- and a tag is a
    string on a command line. Get it wrong and the sweep does not fail: it finds
    a marker whose command differs (a different `--config`), correctly declines
    to treat it as a finished run of *this* experiment, and trains a 159M arm
    from step 0 straight over a finished 105M one. The stage-A result would be
    gone with nothing in any log to say which arm it had been.

    `finished_run` cannot catch it, because differing commands is exactly what
    that guard is built to let through -- a changed budget or schedule is a new
    experiment and must run. What separates the two cases is *which* argument
    changed: a different budget in the same run directory is a rerun, a
    different preset is another experiment's directory. Only the second is
    refused, and it is refused rather than worked around, because guessing which
    of two experiments owns a directory is not a decision a launcher should make.
    """
    from daedalus.supervise import INFLIGHT_SCHEMA, read_inflight

    ckpt = Path(ckpt)
    marker = read_inflight(str(ckpt.parent))
    if marker is None or marker.get("schema") != INFLIGHT_SCHEMA:
        return None
    recorded = _preset_in(marker.get("cmd") or ())
    ours = _preset_in(command)
    if recorded is None or ours is None or recorded == ours:
        return None
    return recorded


def run_arm(arm: ArchArm, *, data_dir: str, run_root: str = RUN_ROOT,
            device: str = "cuda", tag: Optional[str] = None,
            shape: StageShape = STAGE_A, total_tokens: Optional[int] = None,
            val_dir: Optional[str] = None, max_attempts: int = 3,
            stall_min: float = 20.0, refresh: bool = False) -> dict:
    """Train one arm under the supervisor, so an interruption continues it.

    `run_with_resume` reads the open in-flight marker beside the checkpoint, so
    a relaunch after the launching session died continues from where the arm got
    to rather than restarting it -- which is how phase 4 lost 60.3M tokens next
    to a checkpoint it never opened.

    A run that already *finished* needs the opposite guard, and does not get it
    from the supervisor. Its marker is closed, so `interrupted_marker` correctly
    declines to resume it -- and `train.py` then starts at step 0 and overwrites
    the checkpoint on its first save. That is the same lost-work failure from
    the other end: a relaunched sweep would destroy the fifteen finished arms
    that *are* the stage-A result, silently, in the course of reproducing them.
    So a completed identical run is returned rather than re-entered, unless
    `refresh` asks for it deliberately.
    """
    from daedalus.supervise import run_with_resume, start_watchdog, stop_watchdog

    tag = shape.tag if tag is None else tag
    name = arm_run_name(arm, tag)
    budget = int(shape.total_tokens if total_tokens is None else total_tokens)
    command = train_command(arm, data_dir=data_dir, run_name=name,
                            total_tokens=budget, device=device, shape=shape,
                            val_dir=val_dir)

    ckpt = arm_checkpoint_path(arm, tag, run_root)
    run_dir = ckpt.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    common = {"arm": arm.name, "preset": arm.config, "run": name,
              "run_dir": str(run_dir), "shape": shape.name,
              "total_tokens": budget, "steps": budget // shape.batch_tokens,
              "command": list(command)}
    occupant = foreign_run(command, ckpt)
    if occupant is not None:
        raise SystemExit(
            f"{run_dir} already holds a run of preset {occupant!r}, but arm "
            f"{arm.name} at shape {shape.name!r} is preset {arm.config!r}. "
            f"Training here would overwrite another experiment's checkpoint; "
            f"pass the --tag that stage owns instead.")
    if not refresh and finished_run(command, ckpt) is not None:
        # Recorded, not omitted: an artifact that drops a skipped arm and one
        # that never ran it look identical to a reader.
        return {**common, "skipped": "already-completed", "attempts": 0,
                "resumed": False, "returncodes": []}

    watchdog = start_watchdog(name, str(run_dir), budget, stall_min=stall_min,
                              supervised=True)
    try:
        report = run_with_resume(
            list(command), str(ckpt),
            max_attempts=max_attempts, halt_marker=str(run_dir / "HALTED"),
            inflight_extra={"phase": "phase6-architecture", "arm": arm.name,
                            "preset": arm.config, "shape": shape.name,
                            "attention_blocks": arm.num_attention_blocks,
                            "kv_heads": arm.num_key_value_heads,
                            "kv_bytes_per_context_token":
                                arm.kv_bytes_per_context_token,
                            "total_tokens": budget})
    finally:
        stop_watchdog(watchdog)
    return {**common, **report}


def sweep(*, data_dir: str, run_root: str = RUN_ROOT,
          report_root: str = REPORT_ROOT, device: str = "cuda",
          tag: Optional[str] = None, shape: StageShape = STAGE_A,
          total_tokens: Optional[int] = None, val_dir: Optional[str] = None,
          arms: Optional[Sequence[ArchArm]] = None,
          refresh: bool = False) -> dict:
    """Every arm in order, control first.

    Re-entrant by design: arms that already finished are returned from their
    closed markers, so relaunching a sweep the deadline or a dead session cut
    short costs only the arms that have not run.
    """
    tag = shape.tag if tag is None else tag
    arms = arms_for(shape) if arms is None else arms
    results = []
    for arm in arms:
        results.append(run_arm(arm, data_dir=data_dir, run_root=run_root,
                               device=device, tag=tag, shape=shape,
                               total_tokens=total_tokens, val_dir=val_dir,
                               refresh=refresh))
        # Rewritten after every arm, so a sweep cut short still leaves the arms
        # that finished, in the order they were run.
        _write_json(Path(report_root) / f"sweep-{tag}.json",
                    {"tag": tag, "shape": asdict(shape),
                     "total_tokens": int(shape.total_tokens
                                         if total_tokens is None
                                         else total_tokens),
                     # Written beside the results, not left to be re-derived:
                     # the scoring slice needs it to read any BPB difference.
                     "parameter_spread": parameter_spread(arms),
                     "arms": results})
    return {"tag": tag, "shape": asdict(shape),
            "parameter_spread": parameter_spread(arms), "arms": results}


# ====================================================================== cli ====

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--tag", default=None,
                        help="run-directory prefix; defaults to the one the "
                             "--shape owns, which is what keeps two stages out "
                             "of each other's checkpoints")
    parser.add_argument("--shape", default=STAGE_A.name, choices=list(SHAPES))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("arms")
    sub.add_parser("shapes")

    for name in ("run", "sweep"):
        cmd = sub.add_parser(name)
        if name == "run":
            cmd.add_argument("--arm", required=True, choices=list(ARMS_BY_NAME))
        else:
            cmd.add_argument("--arms", default=None,
                             help="comma-separated subset, control first; "
                                  "default every grid point")
        cmd.add_argument("--data-dir", required=True)
        cmd.add_argument("--val-dir", default=None)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--total-tokens", type=int, default=None,
                         help="override the shape's budget (smokes only)")
        cmd.add_argument("--refresh", action="store_true",
                         help="re-train arms that already completed, "
                              "overwriting their checkpoints")

    args = parser.parse_args(argv)
    shape = SHAPES[args.shape]

    if args.command == "arms":
        for arm in arms_for(shape):
            print(json.dumps(arm.describe(), sort_keys=True))
        return 0

    if args.command == "shapes":
        for candidate in SHAPES.values():
            print(json.dumps({**asdict(candidate), "steps": candidate.steps,
                              "grad_accum": candidate.grad_accum},
                             sort_keys=True))
        return 0

    if args.command == "run":
        chosen = {arm.name: arm for arm in arms_for(shape)}[args.arm]
        report = run_arm(chosen, data_dir=args.data_dir,
                         run_root=args.run_root, device=args.device,
                         tag=args.tag, shape=shape,
                         total_tokens=args.total_tokens, val_dir=args.val_dir,
                         refresh=args.refresh)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "sweep":
        report = sweep(data_dir=args.data_dir, run_root=args.run_root,
                       report_root=args.report_root, device=args.device,
                       tag=args.tag, shape=shape,
                       total_tokens=args.total_tokens, val_dir=args.val_dir,
                       arms=selected_arms(args.arms, arms_for(shape)),
                       refresh=args.refresh)
        print(json.dumps({"arms": [a["arm"] for a in report["arms"]],
                          "skipped": [a["arm"] for a in report["arms"]
                                      if a.get("skipped")]}, indent=2))
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
