"""Phase 6 stage A: what do attention layers and KV heads actually buy?

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

Scoring is deliberately not in this module yet. Stage A takes hours per sweep and
the metrics it is read on -- full-pass BPB, retrieval by depth, artifact size,
GGUF load -- are the phase's own evaluation slice; assembling them belongs beside
those evaluators, not inside the launcher.

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
                             arch_probe_preset_name)

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
    note="the fifteen-point attention x KV-head screen at ~105M parameters",
)

SHAPES = {shape.name: shape for shape in (STAGE_A,)}


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


def _build_arms() -> List[ArchArm]:
    """The grid, control first and then descending KV cost.

    Ordering is not cosmetic. A sweep the deadline cuts short leaves whatever
    ran, and descending cost means that prefix is a contiguous walk down from
    the shipped ratio -- readable as a curve -- rather than an arbitrary subset
    whose gaps have to be explained.
    """
    arms = [
        ArchArm(name=f"a{blocks}-kv{kv}",
                config=arch_probe_preset_name(blocks, kv),
                num_attention_blocks=blocks, num_key_value_heads=kv)
        for blocks in ARCH_PROBE_ATTENTION_BLOCKS
        for kv in ARCH_PROBE_KV_HEADS
    ]
    control = next(arm for arm in arms if arm.is_control)
    rest = sorted((arm for arm in arms if not arm.is_control),
                  key=lambda arm: (-arm.kv_bytes_per_context_token, arm.name))
    return [control] + rest


ARMS: Sequence[ArchArm] = tuple(_build_arms())
ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
CONTROL = ARMS[0]


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


def run_arm(arm: ArchArm, *, data_dir: str, run_root: str = RUN_ROOT,
            device: str = "cuda", tag: str = "stagea",
            shape: StageShape = STAGE_A, total_tokens: Optional[int] = None,
            val_dir: Optional[str] = None, max_attempts: int = 3,
            stall_min: float = 20.0) -> dict:
    """Train one arm under the supervisor, so an interruption continues it.

    `run_with_resume` reads the open in-flight marker beside the checkpoint, so
    a relaunch after the launching session died continues from where the arm got
    to rather than restarting it -- which is how phase 4 lost 60.3M tokens next
    to a checkpoint it never opened.
    """
    from daedalus.supervise import run_with_resume, start_watchdog, stop_watchdog

    name = arm_run_name(arm, tag)
    budget = int(shape.total_tokens if total_tokens is None else total_tokens)
    command = train_command(arm, data_dir=data_dir, run_name=name,
                            total_tokens=budget, device=device, shape=shape,
                            val_dir=val_dir)

    ckpt = arm_checkpoint_path(arm, tag, run_root)
    run_dir = ckpt.parent
    run_dir.mkdir(parents=True, exist_ok=True)
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
    return {"arm": arm.name, "preset": arm.config, "run": name,
            "run_dir": str(run_dir), "shape": shape.name,
            "total_tokens": budget, "steps": budget // shape.batch_tokens,
            "command": list(command), **report}


def sweep(*, data_dir: str, run_root: str = RUN_ROOT,
          report_root: str = REPORT_ROOT, device: str = "cuda",
          tag: str = "stagea", shape: StageShape = STAGE_A,
          total_tokens: Optional[int] = None, val_dir: Optional[str] = None,
          arms: Sequence[ArchArm] = ARMS) -> dict:
    """Every arm in order, control first."""
    results = []
    for arm in arms:
        results.append(run_arm(arm, data_dir=data_dir, run_root=run_root,
                               device=device, tag=tag, shape=shape,
                               total_tokens=total_tokens, val_dir=val_dir))
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
    parser.add_argument("--tag", default="stagea")
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

    args = parser.parse_args(argv)
    shape = SHAPES[args.shape]

    if args.command == "arms":
        for arm in ARMS:
            print(json.dumps(arm.describe(), sort_keys=True))
        return 0

    if args.command == "shapes":
        for candidate in SHAPES.values():
            print(json.dumps({**asdict(candidate), "steps": candidate.steps,
                              "grad_accum": candidate.grad_accum},
                             sort_keys=True))
        return 0

    if args.command == "run":
        report = run_arm(ARMS_BY_NAME[args.arm], data_dir=args.data_dir,
                         run_root=args.run_root, device=args.device,
                         tag=args.tag, shape=shape,
                         total_tokens=args.total_tokens, val_dir=args.val_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "sweep":
        report = sweep(data_dir=args.data_dir, run_root=args.run_root,
                       report_root=args.report_root, device=args.device,
                       tag=args.tag, shape=shape,
                       total_tokens=args.total_tokens, val_dir=args.val_dir,
                       arms=selected_arms(args.arms))
        print(json.dumps({"arms": [a["arm"] for a in report["arms"]]}, indent=2))
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
