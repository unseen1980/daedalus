"""Phase 8 gate 2: the 1B-token branch, at the rate the 250M probes selected.

    python scripts/code_branch.py plan
    python scripts/code_branch.py launch

Gate 1 ran three 250M-token arms and `code_probe_report.py score` wrote the
verdict. This is what happens next, and the plan's wording for it is exact:
"train a fresh 1B branch from the base with the selected LR/mixture". Three
words in that sentence are the ones a hand-typed launch gets wrong.

**"fresh"** and **"from the base"** together mean `--init-from` the released
`hero` checkpoint -- *not* the selected probe arm's checkpoint sitting one
directory away with 250M tokens already in it. Continuing an arm would be a
staged 1.25B run reported as a 1B one, against a base the gate's numbers are
differences from. `assert_base_checkpoint` hashes the file against the pinned
digest and `refuse_probe_checkpoint` rejects a path inside a probe's run
directory by name, because the two mistakes look identical from the outside: a
run that trains, exits 0 and produces numbers nobody can interpret.

**"selected"** is read out of `verdict.json`, never retyped. The verdict already
applied the gate and general retention; this refuses to launch at all when it
says stop, so six hours of GPU cannot be spent against a gate that returned no.
The rate comes back from `probe_arms()` by name rather than from the verdict
payload, so a hand-edited verdict cannot smuggle in a rate no arm ran.

**The schedule is recomputed, not inherited.** `warmup_steps_for` at 1B is 95
steps, not the 250M probes' 23; copying an arm's argv and editing `--total-
tokens` leaves the run in a warmup sized for a quarter of the budget. The
mixture is re-preflighted at 1B too -- the epoch cap moves shares with the
budget, so the mixture an arm trained on is not automatically the mixture this
trains on, and a skew past the limit is a different experiment.

**The estimate is measured.** `--estimated-hours` feeds the controller's
deadline reserve, and a number typed from memory is how a run gets refused at
T+136h or, worse, admitted when it does not fit. It is derived from the probes'
own windowed throughput, slowest arm, with a margin.

Supervision is the launcher's (`--supervise-checkpoint`, `--watchdog-tokens`)
rather than a fourth copy of `launch_supervised`: this is the single-run phase
that capability was added for. Detached, so it outlives the session that starts
it.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.code_probes import (CodeProbe, DEFAULT_MIXTURE_RECORD,  # noqa: E402
                                 MAX_L1_SKEW_PTS, arm_is_complete,
                                 assert_base_checkpoint, load_mixture,
                                 metrics_rows, mixture_preflight_at,
                                 probe_arms, train_command)
from scripts.qat_recovery import assert_no_resume, estimated_steps  # noqa: E402
from scripts.vast_program import trainer_checkpoint_for  # noqa: E402

#: The preregistered budget. `runs/vast-program/code-run-manifest.json` names
#: this gate `branch_1b`; the plan's step 4 is "a fresh 1B-token branch".
BRANCH_TOKENS = 1_000_000_000

#: Deliberately not `code-probe-*`. A probe arm's name here would have
#: `arm_is_complete` read the finished 250M arm as this run's budget, and a
#: relaunch resume into it.
BRANCH_NAME = "code-branch-1b"

#: What the in-flight marker says this is, for `boot_resume.py` and the keeper.
PHASE = "phase8-branch-1b"

DEFAULT_VERDICT = "runs/code-probes/verdict.json"
DEFAULT_LOG = "runs/code-probes/branch-1b.log"
DEFAULT_STATE = "runs/vast-program/state.json"

#: Wall-clock margin on the measured projection. Validation passes, checkpoint
#: writes and one supervised retry are all real and none of them are in a
#: training window's throughput.
HOURS_MARGIN = 1.25

#: Attempts the supervisor may make. Matches the probes.
MAX_ATTEMPTS = 3


def branch_checkpoint(command: Sequence[str], *,
                      name: str = BRANCH_NAME) -> str:
    """The checkpoint this argv will write, asked of `train.py`.

    Not composed from `--run-root`: `train.py` has no `--run-dir` flag, so it
    always writes `runs/<run-name>/checkpoint.pt` relative to its own working
    directory, and a run root that moved the *plan's* idea of the path would
    hand the supervisor a marker beside a file that never appears -- every
    relaunch then restarts from step zero and nothing reports a problem. So
    `--run-root` here governs only where the *probe arms* are read from, which
    is a different directory and a different question.

    The fallback is the same rule spelled out, for an environment where
    `train.py` will not import (it pulls in torch). The launcher re-checks the
    agreement either way.
    """

    return trainer_checkpoint_for(command) or os.path.join("runs", name,
                                                           "checkpoint.pt")


class BranchRefused(ValueError):
    """Raised when the 1B branch must not be launched as asked."""


# ------------------------------------------------------------------ inputs ---

def load_verdict(path=DEFAULT_VERDICT) -> dict:
    """`verdict.json`, or a refusal naming what is missing from it."""

    try:
        with open(path) as handle:
            verdict = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchRefused(
            f"cannot read the probe verdict {path}: {exc}. Run "
            f"`scripts/code_probe_report.py score` first; this gate does not "
            f"decide the winner itself.") from exc
    if not isinstance(verdict, dict) or "continue" not in verdict:
        raise BranchRefused(
            f"{path} is not a probe verdict: it has no `continue` field. "
            f"`code_probe_report.py score` writes one; a probe *report* "
            f"(probes.json) is a different document.")
    return verdict


def selected_arm(verdict: dict, *, total_tokens: int = BRANCH_TOKENS,
                 name: str = BRANCH_NAME) -> CodeProbe:
    """The 1B arm the verdict authorises, or a refusal carrying its reason.

    The learning rate is looked up from `probe_arms()` by the selected *name*
    rather than taken from the verdict document. A verdict is a file on disk;
    resolving the rate through the preregistered arm list means the only rates
    this can ever launch are the three that were preregistered and ran.
    """

    if not verdict.get("continue"):
        raise BranchRefused(
            f"the 250M probe gate says stop: "
            f"{verdict.get('reason') or 'no reason recorded'}. The "
            f"preregistered response is to record the negative result and stop "
            f"phase 8 escalation, not to spend 1B tokens anyway.")
    chosen = verdict.get("selected")
    if not chosen:
        raise BranchRefused(
            f"the gate continued but selected no arm "
            f"({verdict.get('reason') or 'no reason recorded'}); there is no "
            f"rate to launch at.")
    rates = {arm.name: arm.muon_lr for arm in probe_arms()}
    if chosen not in rates:
        raise BranchRefused(
            f"the verdict selected {chosen!r}, which is not one of the "
            f"preregistered arms {sorted(rates)}. This gate launches at a rate "
            f"an arm actually ran at.")
    return CodeProbe(name=name, muon_lr=rates[chosen], total_tokens=total_tokens)


def refuse_probe_checkpoint(init_from, *, run_root="runs",
                            arms: Optional[Sequence[CodeProbe]] = None) -> None:
    """Refuse a base path that is inside a probe arm's run directory.

    The plan says this branch starts from the released base and never from a
    probe arm checkpoint, and the two are one `--init-from` apart. Continuing an
    arm trains fine and exits 0; what it produces is 1.25B staged tokens
    reported as a fresh 1B run, measured against a base it no longer started
    from.

    Checked by path rather than only by hash, so the refusal names the actual
    mistake instead of "this file is not the pinned digest".
    """

    resolved = os.path.realpath(str(init_from))
    for arm in (arms if arms is not None else probe_arms()):
        arm_dir = os.path.realpath(os.path.join(run_root, arm.name))
        if resolved == arm_dir or resolved.startswith(arm_dir + os.sep):
            raise BranchRefused(
                f"{init_from} is inside the probe arm {arm.name}'s run "
                f"directory. The 1B branch starts fresh from the released base "
                f"checkpoint; continuing an arm would be a staged 1.25B run "
                f"reported as a 1B one.")


# ------------------------------------------------------------- projection ---

def arm_throughput(run_dir) -> Optional[float]:
    """Median windowed tokens/second for one finished arm, or None.

    `tok_per_sec` and not `tokens / elapsed_h`: `elapsed_h` is wall-clock since
    *this process* started, so after a resume it restarts at zero while `tokens`
    continues from the checkpoint. The ratio then overstates throughput and the
    projection comes out short -- the direction that gets a run admitted inside
    the deadline reserve when it does not fit.

    Median over every window rather than the last one, because the first windows
    of a run carry `torch.compile` and are not the rate the other 99% runs at.
    """

    rates = [float(row["tok_per_sec"]) for row in metrics_rows(run_dir)
             if isinstance(row.get("tok_per_sec"), (int, float))
             and float(row["tok_per_sec"]) > 0]
    return statistics.median(rates) if rates else None


def projected_hours(*, total_tokens: int = BRANCH_TOKENS, run_root="runs",
                    arms: Optional[Sequence[CodeProbe]] = None,
                    margin: float = HOURS_MARGIN) -> dict:
    """How long the branch will take, from the probes' measured throughput.

    Slowest arm, not the mean: the estimate feeds a deadline reserve, and an
    optimistic one is refused at T+136h with the run half done.
    """

    arms = list(arms if arms is not None else probe_arms())
    measured = {}
    for arm in arms:
        rate = arm_throughput(Path(run_root) / arm.name)
        if rate is not None:
            measured[arm.name] = rate
    if not measured:
        raise BranchRefused(
            f"no probe arm under {run_root} has a usable tok_per_sec row, so "
            f"there is nothing to project this branch's hours from. Pass "
            f"--estimated-hours explicitly if you are launching without them.")
    slowest = min(measured.values())
    hours = total_tokens / slowest / 3600.0 * margin
    return {"tok_per_sec": measured, "slowest_tok_per_sec": slowest,
            "margin": margin, "hours": hours,
            "steps": estimated_steps(total_tokens)}


# ------------------------------------------------------------------- plan ---

def branch_plan(*, verdict_path=DEFAULT_VERDICT, init_from: str,
                init_from_sha256: Optional[str] = None,
                mixture_record=DEFAULT_MIXTURE_RECORD, run_root="runs",
                total_tokens: int = BRANCH_TOKENS,
                estimated_hours: Optional[float] = None,
                max_l1_skew: float = MAX_L1_SKEW_PTS,
                **command_kwargs) -> dict:
    """Everything the launch needs, with every refusal already applied.

    Assembled before the phase transition rather than inside the phase, so a
    refusal costs nothing and leaves no half-started phase in the ledger.
    """

    verdict = load_verdict(verdict_path)
    arm = selected_arm(verdict, total_tokens=total_tokens)
    refuse_probe_checkpoint(init_from, run_root=run_root)
    digest = assert_base_checkpoint(init_from, init_from_sha256)
    record = load_mixture(mixture_record)

    # At the branch's own budget: the epoch cap moves shares with the budget, so
    # the mixture the arms trained on is not automatically this one.
    preflight = mixture_preflight_at(record, total_tokens)
    if preflight["l1_skew_pts"] > max_l1_skew:
        raise BranchRefused(
            f"at {total_tokens:,} tokens the mixture sits "
            f"{preflight['l1_skew_pts']:.2f} points from the one asked for, "
            f"past the {max_l1_skew:g}-point limit "
            f"({', '.join(preflight['capped_sources'])} cannot fill it inside "
            f"the epoch cap). The probes' mixture was checked at "
            f"250M; this is a different draw.")

    command = train_command(arm, init_from=init_from, record=record,
                            **command_kwargs)
    assert_no_resume(command)

    # From the argv, so the completeness check reads the directory the trainer
    # will actually write rather than one composed beside it.
    checkpoint = branch_checkpoint(command, name=arm.name)
    run_dir = Path(checkpoint).parent
    if arm_is_complete(run_dir, arm.total_tokens):
        raise BranchRefused(
            f"{run_dir} already trained {arm.total_tokens:,} tokens. Score it "
            f"rather than launching it again.")

    projection = None
    if estimated_hours is None:
        projection = projected_hours(total_tokens=total_tokens,
                                     run_root=run_root)
        estimated_hours = projection["hours"]

    return {
        "schema": 1,
        "gate": "branch_1b",
        "phase": PHASE,
        "selected_probe": verdict["selected"],
        "verdict": {"path": str(verdict_path),
                    "reason": verdict.get("reason"),
                    "continue": bool(verdict.get("continue"))},
        "arm": arm.to_dict(),
        "init_from": {"path": str(init_from), "sha256": digest},
        "mixture": {"record": str(mixture_record),
                    "train_root": record["train_root"],
                    "holdout_root": record["holdout_root"],
                    "weights": record["weights"],
                    "preflight": preflight},
        "run_dir": str(run_dir),
        "supervise_checkpoint": str(checkpoint),
        "estimated_hours": float(estimated_hours),
        "projection": projection,
        "command": command,
    }


def lease_holder(state_path=DEFAULT_STATE) -> Optional[dict]:
    """The live main-lane lease payload, or None when nobody holds it.

    Checked here so a refusal says "phase X is running" up front, rather than
    the detached child taking the refusal in its own log where nothing reads it
    until someone wonders why the branch never started.
    """

    from scripts.vast_program import default_lease_name, process_is_alive

    path = Path(state_path).with_name(default_lease_name())
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    return payload if process_is_alive(pid, payload.get("start_ticks")) else None


def launch(plan: dict, *, state_path=DEFAULT_STATE, log_path=DEFAULT_LOG,
           max_attempts: int = MAX_ATTEMPTS, spawn=None) -> dict:
    """Detach the branch as a supervised main-lane phase."""

    from scripts.vast_program import detach_phase

    kwargs = {} if spawn is None else {"spawn": spawn}
    return detach_phase(
        state=state_path,
        phase=plan["phase"],
        command=plan["command"],
        log_path=log_path,
        estimated_hours=plan["estimated_hours"],
        max_attempts=max_attempts,
        supervise_checkpoint=plan["supervise_checkpoint"],
        # Divergence and a stall are what a six-hour run cannot notice about
        # itself, and the halt marker is what makes the watchdog's stop stick.
        watchdog_tokens=plan["arm"]["total_tokens"],
        **kwargs,
    )


# -------------------------------------------------------------------- cli ---

def _describe(plan: dict) -> None:
    arm = plan["arm"]
    print(f"gate      : {plan['gate']} -- {plan['verdict']['reason']}")
    print(f"selected  : {plan['selected_probe']} -> {arm['name']}")
    print(f"rate      : muon {arm['muon_lr']:g}, adam {arm['adam_lr']:g}")
    print(f"budget    : {arm['total_tokens']:,} tokens, "
          f"{arm['estimated_steps']:,} steps, "
          f"{arm['warmup_steps']:,} warmup")
    print(f"init-from : {plan['init_from']['path']} "
          f"({plan['init_from']['sha256'][:12]})")
    preflight = plan["mixture"]["preflight"]
    print(f"mixture   : {plan['mixture']['record']}, "
          f"L1 skew {preflight['l1_skew_pts']:.2f} pts at this budget")
    projection = plan["projection"]
    measured = ("measured" if projection else "given")
    print(f"estimate  : {plan['estimated_hours']:.2f}h ({measured})")
    if projection:
        print(f"            slowest probe arm "
              f"{projection['slowest_tok_per_sec']:,.0f} tok/s "
              f"x {projection['margin']:g} margin")
    print(f"supervises: {plan['supervise_checkpoint']}")
    print(f"\n{' '.join(plan['command'])}")


def _plan_args(a) -> dict:
    return branch_plan(
        verdict_path=a.verdict, init_from=a.init_from,
        init_from_sha256=a.init_from_sha256, mixture_record=a.mixture_record,
        run_root=a.run_root, total_tokens=a.total_tokens,
        estimated_hours=a.estimated_hours, device=a.device,
        val_every_steps=a.val_every_steps, no_compile=a.no_compile,
        hub_repo=a.hub_repo)


def _plan(a) -> int:
    try:
        plan = _plan_args(a)
    except (BranchRefused, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    _describe(plan)
    if a.json_out:
        _write_json(a.json_out, plan)
        print(f"\nwrote {a.json_out}")
    return 0


def _launch(a) -> int:
    try:
        plan = _plan_args(a)
    except (BranchRefused, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    holder = lease_holder(a.state)
    if holder is not None:
        print(f"REFUSE: the main lane's lease is held by live pid "
              f"{holder.get('pid')} (acquired {holder.get('acquired_at')}). "
              f"Watch {a.state} and its phase log; the branch is launched once "
              f"that phase is over.", file=sys.stderr)
        return 3
    _describe(plan)
    if a.json_out:
        _write_json(a.json_out, plan)
    started = launch(plan, state_path=a.state, log_path=a.log,
                     max_attempts=a.max_attempts)
    print(f"\ndetached phase {plan['phase']} pid {started['pid']} "
          f"log {started['log']}")
    return 0


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


def _common(parser) -> None:
    parser.add_argument("--verdict", default=DEFAULT_VERDICT)
    parser.add_argument(
        "--init-from", default="/root/daedalus/final/hero/checkpoint.pt",
        help="the released base checkpoint; never a probe arm's")
    parser.add_argument(
        "--init-from-sha256",
        default="cfbf27dccf93a07caa2b93cbd630e483c174d52aed8785d104edb7addeb0e153",
        help="the pinned digest from the code run manifest")
    parser.add_argument("--mixture-record", default=DEFAULT_MIXTURE_RECORD)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--total-tokens", type=int, default=BRANCH_TOKENS)
    parser.add_argument(
        "--estimated-hours", type=float, default=None,
        help="override the projection measured off the probes' throughput")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-every-steps", type=int, default=250)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--json-out", default=None)


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="what the branch would run, and why")
    _common(plan)
    plan.set_defaults(fn=_plan)

    run = sub.add_parser("launch", help="detach it as a supervised phase")
    _common(run)
    run.add_argument("--state", default=DEFAULT_STATE)
    run.add_argument("--log", default=DEFAULT_LOG)
    run.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    run.set_defaults(fn=_launch)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(_cli())
