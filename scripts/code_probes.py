"""The three preregistered 250M-token continued-pretraining probes.

    python scripts/code_probes.py command --arm code-probe-lr0.001
    python scripts/code_probes.py sweep --init-from /root/daedalus/final/hero/checkpoint.pt \
        --init-from-sha256 cfbf27dc... --json-out runs/code-probes/probes.json

Gate 1 of phase 8: three arms from the *same* base checkpoint at Muon `5e-4`,
`1e-3` and `2e-3`, Adam proportionally matched, identical data, order and seed.
Everything that is not the learning rate is therefore built once and shared, and
a test asserts the three argvs differ in exactly three arguments -- three shell
lines that were meant to match are not a comparison anybody can check.

**One sweep, not three phases.** The box has one GPU and the controller holds
one lease per lane, so three separately detached phases would not queue behind
each other -- the second would be *refused* while the first ran, and the box
would sit idle between turns waiting to be asked again. This is the shape phases
5 and 6 already used: one detached controller phase whose command is this sweep,
which runs the arms in sequence and is restart-safe between them.

**Restart-safe between arms and inside one.** Each arm goes through
`qat_recovery.launch_supervised` -- the watchdog, the halt marker and
`run_with_resume`, which adds `--resume` on a retry and refuses to continue a run
the watchdog stopped. A relaunch of the *sweep* skips the arms that already
reached their budget (read off `metrics.jsonl`, which is what actually says a run
finished) and resumes the one that was in flight. A restart that ignored a
600MB checkpoint beside it is how phase 4 lost 60.3M tokens.

**`--init-from`, never `--resume`.** `--resume` on attempt one restores the
finished pretraining run's step and token count, so the probe writes no metrics
row and exits 0 -- a probe that trained nothing looks exactly like one that
finished early. The base checkpoint is also hashed before the first arm: the
plan lists "corrupt or unverifiable released baseline" as a hard blocker, and
three arms from a checkpoint nobody hashed compare each other against nothing.

The mixture comes from `corpus mixture`'s record rather than from flags, because
the shares, the root and the epoch cap are one decision and typing any of them
again is how an arm trains on a mixture no artifact describes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.qat_recovery import (BATCH_TOKENS, DECAY_FRAC,  # noqa: E402
                                  MICRO_BATCH, SEQ_LEN, adam_lr_for,
                                  assert_no_resume, estimated_steps,
                                  warmup_steps_for)

#: What this phase's in-flight markers say they are. `boot_resume.py` continues
#: a run from that marker after a reboot and `session_keeper` reads it to know
#: the box is busy, so the phase name in it is provenance rather than decoration
#: -- and `qat_recovery.launch_supervised`, which is otherwise exactly this
#: function, hardcodes `phase3-qat-recovery`. Reusing it would have had a
#: phase 8 arm resumed after a reboot under phase 3's name.
PHASE = "phase8-code-probes"

#: Preregistered in `runs/vast-program/code-run-manifest.json` and in #15's
#: body, before any arm ran. Adam follows from the shipped Muon:Adam ratio,
#: which `adam_lr_for` reads off `train.py`'s own defaults -- the two optimizers
#: cover disjoint parameter sets, so scaling one alone changes which half of the
#: model moves rather than lowering "the" learning rate.
PROBE_MUON_LRS = (5e-4, 1e-3, 2e-3)

PROBE_TOKENS = 250_000_000

#: The config the released base is, so continued pretraining is the same model.
CONFIG = "daedalus-150m"

DEFAULT_MIXTURE_RECORD = "runs/codeprep/train-mixture.json"
DEFAULT_REPORT = "runs/code-probes/probes.json"

#: How far the capped mixture may sit from the one asked for at *this* budget.
#: The composed root was checked at whatever budget it was composed for; an arm
#: reads it at 250M, and a mixture that is a different experiment at that budget
#: must say so before an hour and a half of GPU goes into it.
MAX_L1_SKEW_PTS = 5.0

#: A run has finished when its metrics say so. A checkpoint on disk does not:
#: it is written throughout, so its existence is what a *resume* keys off, not
#: what a skip may.
COMPLETE_FRAC = 0.999


@dataclass(frozen=True)
class CodeProbe:
    """One preregistered arm. Everything but the rate is shared."""

    name: str
    muon_lr: float
    total_tokens: int = PROBE_TOKENS

    @property
    def adam_lr(self) -> float:
        return adam_lr_for(self.muon_lr)

    @property
    def warmup_steps(self) -> int:
        return warmup_steps_for(self.total_tokens)

    def to_dict(self) -> dict:
        return {**asdict(self), "adam_lr": self.adam_lr,
                "warmup_steps": self.warmup_steps,
                "estimated_steps": estimated_steps(self.total_tokens)}


def probe_arms(muon_lrs: Sequence[float] = PROBE_MUON_LRS,
               total_tokens: int = PROBE_TOKENS,
               tag: str = "probe",
               steps: Optional[int] = None,
               batch_tokens: int = BATCH_TOKENS) -> List[CodeProbe]:
    """The three arms, in preregistered order.

    `steps` overrides the budget and `tag` the run name, together, for a smoke.
    Both, because a shortened run under the gate's own name is the worst of the
    two mistakes available: it lands in `runs/code-probe-lr0.001`, it satisfies
    nothing, and the next sweep either resumes it as if it were the real arm or
    -- worse -- reads it as one that finished at a budget nobody chose. Left
    alone the preregistered budget decides, which is what a gate wants.
    """
    if steps is not None:
        total_tokens = int(steps) * int(batch_tokens)
    return [CodeProbe(name=f"code-{tag}-lr{lr:g}", muon_lr=lr,
                      total_tokens=total_tokens)
            for lr in muon_lrs]


def load_mixture(path=DEFAULT_MIXTURE_RECORD) -> dict:
    """`corpus mixture`'s record, checked for the parts an arm depends on.

    Refuses a record whose composed root is not there. The root is a farm of
    symlinks into two corpora, and a missing one is not an error the trainer
    reports usefully -- `resolve_mixture` renormalizes over whatever it finds,
    so a root that lost half its sources trains a perfectly healthy-looking arm
    on a mixture nothing describes.
    """
    with open(path) as handle:
        record = json.load(handle)
    weights = record.get("weights")
    train_root = record.get("train_root")
    if not weights or not train_root:
        raise ValueError(f"{path} has no composed mixture; run "
                         f"`scripts/codeprep.py corpus mixture` first")
    missing = sorted(
        name for name in weights
        if not os.path.exists(os.path.join(train_root, name, "manifest.json")))
    if missing:
        raise ValueError(
            f"{train_root} is missing {len(missing)} of the mixture's "
            f"{len(weights)} sources ({', '.join(missing)}). resolve_mixture "
            f"would renormalize over the rest and train an arm on a mixture "
            f"this record does not describe.")
    return record


def mixture_preflight_at(record: dict, total_tokens: int) -> dict:
    """What the sampler will really draw at *this* arm's budget.

    `train.py`'s own resolver, not a second implementation: the epoch cap moves
    shares, and the shares an arm trains at are the ones its report has to
    carry.
    """
    from train import mixture_preflight

    return mixture_preflight(record["train_root"], total_tokens,
                             weights=record["weights"], verbose=False)


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_base_checkpoint(path, expect_sha256: Optional[str]) -> str:
    """Hash the base checkpoint, and refuse one that is not the pinned artifact.

    Every arm's result is a difference against this file. A truncated or
    substituted checkpoint produces three arms that still train, still exit 0
    and still compare against each other -- the comparison would just be of a
    model nobody released.
    """
    if not os.path.exists(path):
        raise ValueError(f"no base checkpoint at {path!r}")
    digest = sha256_of(path)
    if expect_sha256 and digest != expect_sha256:
        raise ValueError(
            f"{path} hashes {digest}, not the pinned {expect_sha256}. Phase 8 "
            f"starts from the released base and nothing else.")
    return digest


def train_command(arm: CodeProbe, *, init_from: str, record: dict,
                  device: str = "cuda", micro_batch: int = MICRO_BATCH,
                  seq_len: int = SEQ_LEN, batch_tokens: int = BATCH_TOKENS,
                  val_every_steps: int = 250,
                  gradient_checkpointing: bool = False,
                  no_compile: bool = False,
                  hub_repo: Optional[str] = None,
                  python: str = "python") -> List[str]:
    """The exact `train.py` argv for one arm.

    Built as data so "identical data and order" is checkable by a test rather
    than by reading three shell lines: every arm differs only in `--run-name`,
    `--muon-lr` and `--adam-lr`.

    No `--seed`: `train.py` has no such flag and `TrainArgs.seed` defaults to 0,
    so the arms already share one. No `--qat-frac` either -- the probes are
    plain continued pretraining, and QAT is gate 6, applied once to the final
    checkpoint rather than to each candidate.
    """
    from daedalus.codeprep import mixture_weight_flags

    cmd = [
        python, "train.py",
        "--run-name", arm.name,
        "--config", CONFIG,
        "--data-dir", record["train_root"],
        # Weights only, never --resume. See the module docstring.
        "--init-from", init_from,
        "--total-tokens", str(arm.total_tokens),
        "--micro-batch", str(micro_batch),
        # Constant shape: continued pretraining starts from the end of a
        # finished schedule rather than replaying its batch and sequence ramp.
        "--seq-start", str(seq_len), "--seq-end", str(seq_len),
        "--tok-start", str(batch_tokens), "--tok-end", str(batch_tokens),
        "--muon-lr", f"{arm.muon_lr:g}",
        "--adam-lr", f"{arm.adam_lr:g}",
        "--warmup-steps", str(arm.warmup_steps),
        "--decay-frac", str(DECAY_FRAC),
        "--device", device,
        "--val-dir", record["holdout_root"],
        "--val-every-steps", str(val_every_steps),
    ]
    cmd += mixture_weight_flags(record["weights"])
    if gradient_checkpointing:
        cmd += ["--gradient-checkpointing"]
    if no_compile:
        cmd += ["--no-compile"]
    cmd += ["--hub-repo", hub_repo or ""]
    return cmd


def metrics_rows(run_dir) -> List[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue                     # a torn last line, not a verdict
    return rows


def arm_is_complete(run_dir, total_tokens: int,
                    frac: float = COMPLETE_FRAC) -> bool:
    """Whether this arm already trained its budget.

    Read off `metrics.jsonl` rather than off the checkpoint. The checkpoint is
    written throughout the run, so its existence says "this can be resumed", not
    "this is done" -- and a sweep that skipped on the checkpoint would skip the
    arm it was meant to resume.
    """
    rows = metrics_rows(run_dir)
    if not rows:
        return False
    return max(int(row.get("tokens") or 0) for row in rows) >= total_tokens * frac


def arm_summary(arm: CodeProbe, run_dir) -> dict:
    """The last metrics row's headline numbers, for the sweep's own report."""
    rows = metrics_rows(run_dir)
    if not rows:
        return {"rows": 0}
    last = rows[-1]
    return {
        "rows": len(rows),
        "tokens": int(last.get("tokens") or 0),
        "step": int(last.get("step") or 0),
        "loss": last.get("loss"),
        "elapsed_h": last.get("elapsed_h"),
        "val_bpb": last.get("val_bpb"),
        "val_bpb_per_source": last.get("val_bpb_per_source"),
        "skipped_updates": last.get("skipped_updates"),
        "complete": arm_is_complete(run_dir, arm.total_tokens),
    }


def launch_supervised(arm: CodeProbe, command: Sequence[str], *,
                      run_root: str = "runs", stall_min: float = 20.0,
                      max_attempts: int = 3) -> dict:
    """Run one arm under the watchdog and the resume supervisor.

    Nothing here sits inside a training loop: `run_with_resume` owns the
    subprocess and this returns when the arm is over. The three pieces catch
    different failures --

    - the **watchdog** catches divergence and stalls, which an arm cannot
      detect about itself;
    - the **halt marker** is what makes a watchdog stop stick. Without it the
      supervisor reads the watchdog's SIGTERM as an ordinary crash, resumes the
      diverged checkpoint with no watchdog left running, and trains a broken
      model for the rest of the budget before exiting 0;
    - **`--resume` on retry only.** Attempt one must not carry it -- that is
      what `assert_no_resume` enforces -- and a retry must, resuming this
      *arm's* checkpoint rather than the released one.

    A near-copy of `qat_recovery.launch_supervised` in shape, and deliberately
    not a call to it: that one writes phase 3's name into the in-flight marker,
    which is the record `boot_resume.py` continues a run from after a reboot.
    """
    from daedalus.supervise import (run_with_resume, start_watchdog,
                                    stop_watchdog)

    assert_no_resume(command)
    run_dir = Path(run_root) / arm.name
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog = start_watchdog(arm.name, str(run_dir), arm.total_tokens,
                              stall_min=stall_min, supervised=True)
    try:
        return run_with_resume(
            list(command), str(run_dir / "checkpoint.pt"),
            max_attempts=max_attempts, halt_marker=str(run_dir / "HALTED"),
            inflight_extra={"phase": PHASE, "arm": arm.name,
                            "muon_lr": arm.muon_lr,
                            "total_tokens": arm.total_tokens})
    finally:
        stop_watchdog(watchdog)


def run_arm(arm: CodeProbe, *, init_from: str, record: dict,
            run_root: str = "runs", max_attempts: int = 3,
            stall_min: float = 20.0, **command_kwargs) -> dict:
    """One arm, start to finish, under the watchdog and the resume supervisor."""
    command = train_command(arm, init_from=init_from, record=record,
                            **command_kwargs)
    assert_no_resume(command)
    run_dir = Path(run_root) / arm.name
    report = {"arm": arm.to_dict(), "command": command,
              "run_dir": str(run_dir)}
    supervised = launch_supervised(arm, command, run_root=run_root,
                                   stall_min=stall_min,
                                   max_attempts=max_attempts)
    report["supervisor"] = supervised
    report["summary"] = arm_summary(arm, run_dir)
    return report


def sweep(*, init_from: str, init_from_sha256: Optional[str] = None,
          mixture_record=DEFAULT_MIXTURE_RECORD,
          arms: Optional[Sequence[CodeProbe]] = None,
          run_root: str = "runs", json_out: Optional[str] = DEFAULT_REPORT,
          max_attempts: int = 3, max_l1_skew: float = MAX_L1_SKEW_PTS,
          **command_kwargs) -> dict:
    """Every arm in preregistered order, skipping the ones already finished.

    The report is written after each arm rather than at the end: a sweep that
    dies in arm three must not take arms one and two's numbers with it.
    """
    record = load_mixture(mixture_record)
    digest = assert_base_checkpoint(init_from, init_from_sha256)
    arms = list(arms if arms is not None else probe_arms())
    if not arms:
        raise ValueError("no arms to run")
    preflight = mixture_preflight_at(record, arms[0].total_tokens)
    if preflight["l1_skew_pts"] > max_l1_skew:
        raise ValueError(
            f"at {arms[0].total_tokens:,} tokens the mixture sits "
            f"{preflight['l1_skew_pts']:.2f} points from the one asked for, "
            f"past the {max_l1_skew:g}-point limit "
            f"({', '.join(preflight['capped_sources'])} cannot fill it inside "
            f"the epoch cap)")

    report = {
        "schema": 1,
        "gate": "probes_250m",
        "init_from": {"path": str(init_from), "sha256": digest},
        "mixture": {"record": str(mixture_record),
                    "train_root": record["train_root"],
                    "holdout_root": record["holdout_root"],
                    "weights": record["weights"],
                    "preflight": preflight},
        "arms": [],
    }

    def write() -> None:
        if not json_out:
            return
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        tmp = f"{json_out}.tmp"
        with open(tmp, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, json_out)

    write()
    for arm in arms:
        run_dir = Path(run_root) / arm.name
        if arm_is_complete(run_dir, arm.total_tokens):
            print(f"=== skip {arm.name}: already trained "
                  f"{arm.total_tokens:,} tokens ===", flush=True)
            report["arms"].append({"arm": arm.to_dict(), "skipped": True,
                                   "run_dir": str(run_dir),
                                   "summary": arm_summary(arm, run_dir)})
            write()
            continue
        print(f"=== {arm.name}: muon {arm.muon_lr:g}, adam {arm.adam_lr:g}, "
              f"{arm.total_tokens:,} tokens ===", flush=True)
        try:
            report["arms"].append(run_arm(
                arm, init_from=init_from, record=record, run_root=run_root,
                max_attempts=max_attempts, **command_kwargs))
        except Exception as exc:            # noqa: BLE001 - recorded, not raised
            # One arm failing is a negative result for that arm, not a reason to
            # lose the two beside it. The controller sees a non-zero exit below.
            print(f"FAILED {arm.name}: {exc!r}", file=sys.stderr, flush=True)
            report["arms"].append({"arm": arm.to_dict(), "error": repr(exc),
                                   "run_dir": str(run_dir),
                                   "summary": arm_summary(arm, run_dir)})
        write()
    return report


def _print_command(a) -> int:
    record = load_mixture(a.mixture_record)
    arms = {arm.name: arm for arm in probe_arms()}
    arm = arms.get(a.arm)
    if arm is None:
        print(f"REFUSE: no arm {a.arm!r}; this gate has {sorted(arms)}",
              file=sys.stderr)
        return 2
    print(" ".join(train_command(arm, init_from=a.init_from, record=record)))
    return 0


def _sweep(a) -> int:
    if a.steps and a.tag == "probe":
        print("REFUSE: --steps shortens the run, so it needs its own --tag. "
              "Under the gate's name a shortened arm lands in the gate's run "
              "directory, and the next sweep resumes it as the real arm or "
              "reads it as one that finished at a budget nobody chose.",
              file=sys.stderr)
        return 2
    arms = probe_arms(tag=a.tag, steps=a.steps)
    if a.steps:
        print(f"SMOKE: {a.steps} steps per arm under the name "
              f"{arms[0].name!r}, not the preregistered "
              f"{PROBE_TOKENS:,}-token gate", flush=True)
    try:
        report = sweep(init_from=a.init_from,
                       init_from_sha256=a.init_from_sha256,
                       mixture_record=a.mixture_record, arms=arms,
                       run_root=a.run_root, json_out=a.json_out,
                       max_attempts=a.max_attempts,
                       device=a.device, no_compile=a.no_compile,
                       val_every_steps=a.val_every_steps)
    except (OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    for entry in report["arms"]:
        summary = entry.get("summary") or {}
        note = "  SKIPPED" if entry.get("skipped") else ""
        if entry.get("error"):
            note = f"  ERROR {entry['error'][:60]}"
        print(f"  {entry['arm']['name']:24s} "
              f"muon {entry['arm']['muon_lr']:<8g} "
              f"{summary.get('tokens', 0) / 1e6:>8,.1f}M  "
              f"loss {summary.get('loss') or float('nan'):.4f}{note}")
    if a.json_out:
        print(f"\nwrote {a.json_out}")
    failed = [entry for entry in report["arms"] if entry.get("error")]
    incomplete = [entry for entry in report["arms"]
                  if not entry.get("error")
                  and not (entry.get("summary") or {}).get("complete")]
    if failed or incomplete:
        for entry in failed + incomplete:
            print(f"  - {entry['arm']['name']}: "
                  f"{entry.get('error') or 'did not reach its budget'}",
                  file=sys.stderr)
        return 3
    return 0


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    command = sub.add_parser("command", help="print one arm's train.py argv")
    command.add_argument("--arm", required=True)
    command.add_argument("--init-from", required=True)
    command.add_argument("--mixture-record", default=DEFAULT_MIXTURE_RECORD)
    command.set_defaults(fn=_print_command)

    run = sub.add_parser("sweep", help="every arm in order, resuming what was "
                                       "in flight and skipping what finished")
    run.add_argument("--init-from", required=True,
                     help="the released base checkpoint; never the SFT/DPO one")
    run.add_argument("--init-from-sha256", default=None,
                     help="pin it, so three arms cannot be compared against a "
                          "checkpoint nobody verified")
    run.add_argument("--mixture-record", default=DEFAULT_MIXTURE_RECORD)
    run.add_argument("--run-root", default="runs")
    run.add_argument("--json-out", default=DEFAULT_REPORT)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--tag", default="probe",
                     help="the run-name segment; a smoke uses its own so it "
                          "cannot be resumed into, or read as, the gate's arm")
    run.add_argument("--steps", type=int, default=None,
                     help="override the preregistered budget (smokes only); "
                          "requires --tag to move the run name with it")
    run.add_argument("--device", default="cuda")
    run.add_argument("--val-every-steps", type=int, default=250)
    run.add_argument("--no-compile", action="store_true")
    run.set_defaults(fn=_sweep)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(_cli())
