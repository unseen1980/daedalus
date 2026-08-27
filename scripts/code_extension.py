"""Phase 8 step 5: the staged 2B extension, and what it refuses.

    python scripts/code_extension.py plan
    python scripts/code_extension.py launch

Gate 2 scored the 1B branch and `code_branch_report.py verdict` wrote the answer.
This is what the manifest authorises next, and its wording carries four
conditions that a hand-typed launch collapses into one command:

    extension_2b.only_if : the 1B gate passes and completion is projected
                           before T+136h
    extension_2b.how     : --init-from the 1B weights, lower LR, fresh WSD,
                           same replay floor
    extension_2b.report_as: staged adaptation, not one uninterrupted 3B schedule

**"the 1B gate"**, not the 250M one. Both verdicts live in
`runs/code-probes/`, both carry a `continue` field, and pointing `--verdict` at
the probe file would spend 2B tokens on the authority of a gate that only ever
authorised 1B. `load_branch_verdict` reads the *nested* gate name and refuses
anything that is not `branch_1b` on this branch.

**"--init-from the 1B weights"** is the inverse of the refusal gate 2's launcher
needed. There the danger was continuing a probe arm; here it is starting from the
released base again, which trains fine, exits 0, and produces a fresh 2B run
reported as a staged 3B one. `refuse_base_checkpoint` hashes the input against
the pinned base digest and refuses a match, `refuse_probe_checkpoint` still
rejects a probe arm, and `assert_branch_complete` refuses to extend a 1B run that
never reached its budget -- "1B + 2B" is a claim about the first stage as much as
the second.

**"lower LR"** carries no number, so one is fixed here and the reasoning is
recorded rather than buried: `EXTENSION_LR_FRAC` halves the rate **the branch
actually ran at**, read off its own `metrics.jsonl` as the peak of its schedule
rather than re-derived from a document, and cross-checked against the probe
verdict that selected it. Adam follows through `adam_lr_for`, because the two
optimizers cover disjoint parameter sets and scaling one alone changes which half
of the model moves instead of lowering "the" learning rate.

**"fresh WSD"** falls out of building the argv from a `CodeProbe` at this stage's
own budget: 2B tokens warm up over `warmup_steps_for(2B)` and decay to zero
inside this stage. Inheriting the branch's schedule -- 1B's warmup, or `--resume`
into a finished decay -- is the failure `assert_no_resume` and the warmup test
exist for.

**"same replay floor"** is checked at the extension's budget, not assumed from
the branch's. The epoch cap moves shares with the budget, and the only source
that caps here is a general-replay one, so the bucket this gate's retention
depends on is exactly the bucket the cap eats into. `replay_floor` reports the
target and effective share of the record's `general-replay` bucket and refuses a
shortfall past that bucket's pro-rata share of the mixture's own preregistered
L1 budget -- one point for a twenty-point bucket, which is a bar that can
actually fire, unlike the aggregate five it is derived from.

**The deadline clause is checked here, up front.** The controller refuses a phase
whose estimate crosses T+136h, but it does so inside the detached child, where
the refusal lands in a log nobody reads until the GPU has been idle for hours.
`assert_fits_deadline` asks the same `ProgramDeadline` the controller uses,
before anything is spawned, from a projection measured off the branch's own
throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.code_branch import (BRANCH_NAME, BRANCH_TOKENS,  # noqa: E402
                                 BranchRefused, DEFAULT_STATE, HOURS_MARGIN,
                                 MAX_ATTEMPTS, arm_throughput,
                                 branch_checkpoint, lease_holder, launch,
                                 load_verdict, refuse_probe_checkpoint,
                                 selected_arm)
from scripts.code_probes import (CodeProbe, DEFAULT_MIXTURE_RECORD,  # noqa: E402
                                 MAX_L1_SKEW_PTS, arm_is_complete,
                                 load_mixture, metrics_rows,
                                 mixture_preflight_at, sha256_of, train_command)
from scripts.qat_recovery import assert_no_resume, estimated_steps  # noqa: E402

#: The second stage's own budget. The manifest's `extension_2b` is "a further 2B
#: tokens"; with the branch's 1B before it the artifact has seen 3B, staged.
EXTENSION_TOKENS = 2_000_000_000

#: Named apart from the branch, so a relaunch resumes *this* stage rather than
#: reopening the finished 1B run's checkpoint and reading its budget as met.
EXTENSION_NAME = "code-ext-2b"

#: What the in-flight marker says this is, for `boot_resume.py` and the keeper.
PHASE = "phase8-extension-2b"

#: The manifest says "lower LR" and fixes no number, so this is the
#: disambiguation, chosen before the branch produced a number and recorded as the
#: `phase8-extension-lr` decision in `runs/vast-program/events.jsonl`:
#:
#:   - It must be *lower*: the branch runs a fully decayed WSD schedule
#:     (`DECAY_FRAC` 0.8, decaying to zero), so its weights arrive at this stage
#:     from a schedule that already annealed. Reopening at the same peak
#:     re-injects exactly the update size the decay just removed.
#:   - It must be lower by a stated factor rather than by taste, because a rate
#:     typed per stage is how two stages of one artifact become two experiments.
#:     A half is the smallest factor that is unambiguously "lower" and still
#:     leaves the stage able to move the model at all; a tenth would make 2B
#:     tokens of GPU an expensive no-op, and the plan's degradation policy names
#:     an unfinishable or pointless extension as the expensive mistake.
#:   - The rate it halves is the one the branch *ran at*, read from the branch's
#:     own metrics, so this cannot drift from what actually happened.
EXTENSION_LR_FRAC = 0.5

#: How far the branch's measured peak may sit from the rate its selected probe
#: arm was preregistered at before the two readings are treated as disagreeing.
#: Generous, because it exists to catch a *different rate*, not float noise: the
#: three preregistered rates are a factor of two apart.
LR_AGREEMENT_REL = 0.01

DEFAULT_BRANCH_VERDICT = "runs/code-probes/branch-1b-verdict.json"
DEFAULT_PROBE_VERDICT = "runs/code-probes/verdict.json"
DEFAULT_BRANCH_CHECKPOINT = f"runs/{BRANCH_NAME}/checkpoint.pt"
DEFAULT_LOG = "runs/code-probes/extension-2b.log"

#: The bucket the general-retention gates are protected by. The record names it.
REPLAY_BUCKET = "general-replay"

#: The released base's pinned digest, from the code run manifest. Here it is the
#: thing to refuse rather than the thing to require: see `refuse_base_checkpoint`.
BASE_SHA256 = "cfbf27dccf93a07caa2b93cbd630e483c174d52aed8785d104edb7addeb0e153"


class ExtensionRefused(ValueError):
    """Raised when the 2B extension must not be launched as asked."""


# ------------------------------------------------------------------ inputs ---

def load_branch_verdict(path=DEFAULT_BRANCH_VERDICT) -> dict:
    """Gate 2's verdict, or a refusal naming what it actually is.

    The two verdicts sit in one directory, both have a `continue` field, and a
    passing probe verdict would authorise this launch just as readily as a
    passing branch one -- while meaning something four times smaller. So the
    nested gate name is checked, not merely the presence of `continue`.
    """

    try:
        verdict = load_verdict(path)
    except BranchRefused as exc:
        raise ExtensionRefused(str(exc)) from exc

    gate = verdict.get("gate")
    name = gate.get("gate") if isinstance(gate, dict) else gate
    if name != "branch_1b":
        raise ExtensionRefused(
            f"{path} is a {name!r} verdict, not `branch_1b`. The 2B extension "
            f"is authorised by the 1B branch's gate; the 250M probe gate "
            f"authorised the 1B branch and nothing past it.")
    branch = gate.get("branch") if isinstance(gate, dict) else None
    if branch != BRANCH_NAME:
        raise ExtensionRefused(
            f"{path} scored {branch!r}, not {BRANCH_NAME!r}. This stage "
            f"extends the branch that gate passed, and a verdict about another "
            f"model does not authorise it.")
    if not verdict.get("continue"):
        raise ExtensionRefused(
            f"the 1B branch gate says stop: "
            f"{verdict.get('reason') or 'no reason recorded'}. The "
            f"preregistered response is to stop Daedalus-Code at 1B and report "
            f"it, not to spend a further 2B tokens.")
    return verdict


def branch_peak_lr(run_dir) -> Optional[float]:
    """The peak Muon rate the branch's schedule actually reached, or None.

    Read from the run rather than from the launch document, because those are
    two different claims and only this one survives a hand relaunch: `lr` is the
    scheduled Muon rate at each window, so under WSD its maximum is the peak the
    run warmed up to. A run that never left warmup would report the top of its
    ramp, which is why `estimated_steps` is not what this is compared against --
    the probe verdict is.
    """

    rates = [float(row["lr"]) for row in metrics_rows(run_dir)
             if isinstance(row.get("lr"), (int, float))
             and math.isfinite(float(row["lr"])) and float(row["lr"]) > 0]
    return max(rates) if rates else None


def extension_rate(*, branch_run_dir, probe_verdict_path=DEFAULT_PROBE_VERDICT,
                   frac: float = EXTENSION_LR_FRAC) -> dict:
    """The extension's Muon rate: half what the branch ran at, twice attested.

    Two independent readings of the same number -- the branch's own metrics and
    the probe verdict that selected the rate it was launched with -- and a
    refusal when they disagree. Either alone can be wrong in a way that is
    invisible: a verdict is a file that can be rewritten after the fact, and a
    run's metrics can carry a rate somebody launched by hand.
    """

    measured = branch_peak_lr(branch_run_dir)
    if measured is None:
        raise ExtensionRefused(
            f"{branch_run_dir} has no usable `lr` row, so the rate this stage "
            f"is meant to halve cannot be read off the run that set it. The "
            f"extension does not guess a rate.")

    selected = selected_arm(load_verdict(probe_verdict_path))
    if abs(measured - selected.muon_lr) > LR_AGREEMENT_REL * selected.muon_lr:
        raise ExtensionRefused(
            f"the branch's metrics peak at muon {measured:g} but "
            f"{probe_verdict_path} selected {selected.muon_lr:g}. One of the "
            f"two does not describe the run this extends, and 'half the rate "
            f"the branch ran at' is undefined until they agree.")

    return {"branch_muon_lr": measured, "frac": float(frac),
            "muon_lr": measured * float(frac),
            "measured_from": str(Path(branch_run_dir) / "metrics.jsonl"),
            "cross_checked_against": {"path": str(probe_verdict_path),
                                      "arm": selected.name,
                                      "muon_lr": selected.muon_lr}}


def extension_arm(rate: dict, *, total_tokens: int = EXTENSION_TOKENS,
                  name: str = EXTENSION_NAME) -> CodeProbe:
    """This stage as an arm, so its schedule is built at its own budget."""

    return CodeProbe(name=name, muon_lr=rate["muon_lr"],
                     total_tokens=total_tokens)


# --------------------------------------------------------------- the input ---

def refuse_base_checkpoint(init_from, *, base_sha256: str = BASE_SHA256) -> str:
    """Refuse the released base, and return the digest of what was given.

    The mirror image of gate 2's refusal. There, starting from a probe arm would
    have staged a 1.25B run reported as a fresh 1B one; here, starting from the
    base again produces a *fresh 2B* run reported as staged 3B adaptation. Both
    train, both exit 0, and both are only visible in the hash of the file the
    run started from -- so it is hashed.
    """

    if not os.path.exists(init_from):
        raise ExtensionRefused(
            f"no checkpoint at {init_from!r}; this stage continues the 1B "
            f"branch's weights and has nothing to start from without them.")
    digest = sha256_of(init_from)
    if base_sha256 and digest == base_sha256:
        raise ExtensionRefused(
            f"{init_from} is the released base checkpoint. This stage extends "
            f"the 1B branch -- `--init-from` its weights, not the base's. From "
            f"the base this would be a fresh 2B run reported as staged 3B "
            f"adaptation.")
    return digest


def assert_branch_complete(run_dir, *, total_tokens: int = BRANCH_TOKENS) -> int:
    """Refuse to extend a first stage that never reached its budget.

    "Staged 1B -> 2B" is a claim about both stages. Extending a branch that
    stopped at 400M tokens produces an artifact whose reported 3B is 2.4B, and
    whose gate-2 numbers were measured on weights this stage did not start from.
    """

    rows = metrics_rows(run_dir)
    trained = max((int(row.get("tokens") or 0) for row in rows), default=0)
    if not arm_is_complete(run_dir, total_tokens):
        raise ExtensionRefused(
            f"{run_dir} trained {trained:,} of {total_tokens:,} tokens, so the "
            f"first stage is not finished. Extending it would report a staged "
            f"{(trained + EXTENSION_TOKENS):,}-token artifact as "
            f"{(total_tokens + EXTENSION_TOKENS):,}, against gate-2 numbers "
            f"taken on a checkpoint this stage did not continue.")
    return trained


# ------------------------------------------------------------- the mixture ---

def replay_tolerance_pts(target_pts: float,
                         l1_tolerance_pts: float = MAX_L1_SKEW_PTS) -> float:
    """How far one bucket may drift: its pro-rata share of the L1 budget.

    Derived rather than invented, because no per-bucket tolerance is
    preregistered and the obvious substitute -- reusing the whole mixture's
    5-point L1 limit for a 20-point bucket -- is not a gate at all. A shortfall
    the cap takes out of one bucket shows up twice in the aggregate L1 skew
    (once where it is lost, once where it is redistributed), so any bucket
    shortfall past ~2.5 points is already refused by the mixture check and a
    bucket bar set at 5 could never fire first.

    Distributing the same budget pro rata keeps the two consistent -- the
    per-bucket allowances sum to exactly the aggregate one -- and makes each
    bucket's bar tighter than the aggregate in proportion to how much of the
    mixture it is. For the 20-point replay bucket that is 1.0 point.
    """

    return l1_tolerance_pts * float(target_pts) / 100.0


def replay_floor(record: dict, preflight: dict, *,
                 l1_tolerance_pts: float = MAX_L1_SKEW_PTS) -> dict:
    """The general-replay bucket's target and effective share at this budget.

    The manifest's "same replay floor" is a statement about the bucket the
    general-retention gates depend on, and at this budget it is the bucket the
    epoch cap actually reaches: the replay sources are the small ones, so a
    larger draw caps them first and the aggregate L1 skew can stay comfortable
    while replay alone thins.
    """

    buckets = record.get("buckets")
    if not isinstance(buckets, dict) or REPLAY_BUCKET not in buckets:
        raise ExtensionRefused(
            f"the mixture record has no {REPLAY_BUCKET!r} bucket, so it cannot "
            f"say what its replay floor is. This stage holds the same floor the "
            f"branch trained under, which means reading it from the same "
            f"record.")

    members = sorted(buckets[REPLAY_BUCKET])
    weights = record.get("weights") or {}
    per_source = preflight.get("per_source") or {}
    missing = [name for name in members if name not in per_source]
    if missing:
        raise ExtensionRefused(
            f"the preflight has no share for replay source(s) "
            f"{', '.join(missing)}; the floor cannot be measured over sources "
            f"the sampler did not resolve.")

    target = sum(float(weights.get(name, 0.0)) for name in members) * 100.0
    effective = sum(float(per_source[name]["effective_share"])
                    for name in members) * 100.0
    capped = sorted(name for name in members if per_source[name].get("capped"))
    shortfall = target - effective
    tolerance_pts = replay_tolerance_pts(target, l1_tolerance_pts)
    floor = {"bucket": REPLAY_BUCKET, "members": members,
             "target_pts": target, "effective_pts": effective,
             "shortfall_pts": shortfall, "capped_sources": capped,
             "tolerance_pts": tolerance_pts,
             "l1_tolerance_pts": float(l1_tolerance_pts)}

    declared = (record.get("corpus_shares") or {}).get(REPLAY_BUCKET)
    if declared is not None:
        floor["declared_pts"] = float(declared) * 100.0
        if abs(floor["declared_pts"] - target) > 0.5:
            raise ExtensionRefused(
                f"the record declares a {floor['declared_pts']:.2f}-point "
                f"replay bucket but its weights sum to {target:.2f} points. "
                f"The floor this stage is meant to hold is not the floor the "
                f"record describes.")

    if shortfall > tolerance_pts:
        raise ExtensionRefused(
            f"at {preflight['total_run_tokens']:,} tokens the replay bucket "
            f"draws {effective:.2f} of its {target:.2f} points "
            f"({shortfall:.2f} short, past the {tolerance_pts:.2f}-point "
            f"tolerance; {', '.join(capped) or 'no source'} capped). The "
            f"general-retention gates are measured against a model trained with "
            f"that replay, so a thinner floor is a different experiment.")
    return floor


# ------------------------------------------------------------- projection ---

def projected_hours(*, branch_run_dir, total_tokens: int = EXTENSION_TOKENS,
                    margin: float = HOURS_MARGIN) -> dict:
    """How long this stage takes, from the branch's own measured throughput.

    The branch is the same model, mixture, batch shape and box, and it has just
    run for hours -- so its windowed median is a better estimate than the 250M
    probes', and it is the only one that reflects whatever the box was actually
    doing this week. `arm_throughput` reads `tok_per_sec` rather than
    tokens/elapsed_h for the reason gate 2 documents: after a resume the ratio
    overstates the rate, which understates the hours, which is the direction that
    gets a run admitted inside the deadline reserve when it does not fit.
    """

    rate = arm_throughput(branch_run_dir)
    if rate is None:
        raise ExtensionRefused(
            f"{branch_run_dir} has no usable tok_per_sec row, so there is "
            f"nothing to project this stage's hours from. Pass "
            f"--estimated-hours explicitly if you are launching without them.")
    return {"tok_per_sec": rate, "margin": margin,
            "hours": total_tokens / rate / 3600.0 * margin,
            "steps": estimated_steps(total_tokens),
            "measured_from": str(Path(branch_run_dir) / "metrics.jsonl")}


def assert_fits_deadline(estimated_hours: float, *, state_path=DEFAULT_STATE,
                         now: Optional[datetime] = None) -> dict:
    """Refuse a stage that cannot finish before the finalization window opens.

    The manifest makes this an `only_if`, so it is answered before anything is
    spawned. The rule itself is the controller's -- same `ProgramDeadline`, same
    state file, same reserve -- rather than a second implementation of the
    program's clock.
    """

    from daedalus.program_state import ProgramDeadline

    try:
        with open(state_path) as handle:
            state = json.load(handle)
        started_at = datetime.fromisoformat(
            str(state["started_at"]).replace("Z", "+00:00"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ExtensionRefused(
            f"cannot read the program's start time from {state_path}: {exc}. "
            f"The extension's only_if is a deadline projection, and it is not "
            f"answerable without the deadline.") from exc

    deadline = ProgramDeadline(
        started_at,
        hard_hours=float(state.get("hard_hours", 144.0)),
        reserve_hours=float(state.get("reserve_hours", 8.0)))
    now = now or datetime.now(timezone.utc)
    remaining = (deadline.finalizes_at - now).total_seconds() / 3600.0
    fits = deadline.can_start(now, estimated_hours)
    verdict = {"stage": deadline.stage(now),
               "finalizes_at": deadline.finalizes_at.isoformat(),
               "expires_at": deadline.expires_at.isoformat(),
               "hours_to_finalization": remaining,
               "estimated_hours": float(estimated_hours), "fits": bool(fits)}
    if not fits:
        raise ExtensionRefused(
            f"a {estimated_hours:.2f}h stage does not fit the "
            f"{remaining:.2f}h left before finalization at "
            f"{verdict['finalizes_at']} (deadline stage: {verdict['stage']}). "
            f"The preregistered response is to stop Daedalus-Code at 1B rather "
            f"than start an extension that cannot finish and be evaluated.")
    return verdict


# ------------------------------------------------------------------- plan ---

def extension_plan(*, branch_verdict_path=DEFAULT_BRANCH_VERDICT,
                   probe_verdict_path=DEFAULT_PROBE_VERDICT,
                   init_from: str = DEFAULT_BRANCH_CHECKPOINT,
                   base_sha256: str = BASE_SHA256,
                   mixture_record=DEFAULT_MIXTURE_RECORD, run_root="runs",
                   total_tokens: int = EXTENSION_TOKENS,
                   estimated_hours: Optional[float] = None,
                   state_path=DEFAULT_STATE, now: Optional[datetime] = None,
                   max_l1_skew: float = MAX_L1_SKEW_PTS,
                   arms: Optional[Sequence[CodeProbe]] = None,
                   **command_kwargs) -> dict:
    """Everything the launch needs, with every refusal already applied."""

    verdict = load_branch_verdict(branch_verdict_path)

    branch_run_dir = Path(init_from).parent
    refuse_probe_checkpoint(init_from, run_root=run_root, arms=arms)
    digest = refuse_base_checkpoint(init_from, base_sha256=base_sha256)
    trained = assert_branch_complete(branch_run_dir)

    rate = extension_rate(branch_run_dir=branch_run_dir,
                          probe_verdict_path=probe_verdict_path)
    arm = extension_arm(rate, total_tokens=total_tokens)

    record = load_mixture(mixture_record)
    preflight = mixture_preflight_at(record, total_tokens)
    if preflight["l1_skew_pts"] > max_l1_skew:
        raise ExtensionRefused(
            f"at {total_tokens:,} tokens the mixture sits "
            f"{preflight['l1_skew_pts']:.2f} points from the one asked for, "
            f"past the {max_l1_skew:g}-point limit "
            f"({', '.join(preflight['capped_sources'])} cannot fill it inside "
            f"the epoch cap). The branch's mixture was checked at "
            f"{BRANCH_TOKENS:,}; this is a different draw.")
    floor = replay_floor(record, preflight, l1_tolerance_pts=max_l1_skew)

    command = train_command(arm, init_from=init_from, record=record,
                            **command_kwargs)
    assert_no_resume(command)

    checkpoint = branch_checkpoint(command, name=arm.name)
    run_dir = Path(checkpoint).parent
    if arm_is_complete(run_dir, arm.total_tokens):
        raise ExtensionRefused(
            f"{run_dir} already trained {arm.total_tokens:,} tokens. Score it "
            f"rather than launching it again.")
    if os.path.realpath(run_dir) == os.path.realpath(branch_run_dir):
        raise ExtensionRefused(
            f"this stage would write into the branch's own run directory "
            f"{run_dir}, overwriting the 1B checkpoint it starts from and the "
            f"metrics gate 2 was decided on.")

    projection = None
    if estimated_hours is None:
        projection = projected_hours(branch_run_dir=branch_run_dir,
                                     total_tokens=total_tokens)
        estimated_hours = projection["hours"]
    deadline = assert_fits_deadline(estimated_hours, state_path=state_path,
                                    now=now)

    return {
        "schema": 1,
        "gate": "extension_2b",
        "phase": PHASE,
        "branch_verdict": {"path": str(branch_verdict_path),
                           "reason": verdict.get("reason"),
                           "continue": bool(verdict.get("continue"))},
        "arm": arm.to_dict(),
        "rate": rate,
        "init_from": {"path": str(init_from), "sha256": digest,
                      "run": BRANCH_NAME, "tokens": trained},
        "mixture": {"record": str(mixture_record),
                    "train_root": record["train_root"],
                    "holdout_root": record["holdout_root"],
                    "weights": record["weights"],
                    "preflight": preflight,
                    "replay_floor": floor},
        "run_dir": str(run_dir),
        "supervise_checkpoint": str(checkpoint),
        "estimated_hours": float(estimated_hours),
        "projection": projection,
        "deadline": deadline,
        # The manifest's `report_as`, carried in the launch document so the
        # report is written from a record that already says what this was.
        "staged": {
            "stage": 2, "of": 2,
            "previous": {"run": BRANCH_NAME, "tokens": BRANCH_TOKENS,
                         "checkpoint": str(init_from), "sha256": digest},
            "this_stage_tokens": total_tokens,
            "cumulative_tokens": BRANCH_TOKENS + total_tokens,
            "report_as": ("staged adaptation -- 1B then 2B, each a fresh WSD "
                          "schedule, not one uninterrupted 3B run"),
        },
        "command": command,
    }


# -------------------------------------------------------------------- cli ---

def _describe(plan: dict) -> None:
    arm = plan["arm"]
    rate = plan["rate"]
    print(f"gate      : {plan['gate']} -- {plan['branch_verdict']['reason']}")
    print(f"stage     : {plan['staged']['stage']} of {plan['staged']['of']} -- "
          f"{plan['staged']['cumulative_tokens']:,} cumulative tokens, "
          f"{plan['staged']['report_as']}")
    print(f"rate      : muon {arm['muon_lr']:g} "
          f"({rate['frac']:g} x the branch's measured {rate['branch_muon_lr']:g}), "
          f"adam {arm['adam_lr']:g}")
    print(f"budget    : {arm['total_tokens']:,} tokens, "
          f"{arm['estimated_steps']:,} steps, "
          f"{arm['warmup_steps']:,} warmup")
    print(f"init-from : {plan['init_from']['path']} "
          f"({plan['init_from']['sha256'][:12]}, "
          f"{plan['init_from']['tokens']:,} tokens trained)")
    preflight = plan["mixture"]["preflight"]
    floor = plan["mixture"]["replay_floor"]
    print(f"mixture   : {plan['mixture']['record']}, "
          f"L1 skew {preflight['l1_skew_pts']:.2f} pts at this budget")
    print(f"replay    : {floor['effective_pts']:.2f} of "
          f"{floor['target_pts']:.2f} pts "
          f"({', '.join(floor['capped_sources']) or 'nothing'} capped)")
    projection = plan["projection"]
    measured = ("measured" if projection else "given")
    print(f"estimate  : {plan['estimated_hours']:.2f}h ({measured}), "
          f"{plan['deadline']['hours_to_finalization']:.2f}h to finalization")
    if projection:
        print(f"            branch {projection['tok_per_sec']:,.0f} tok/s "
              f"x {projection['margin']:g} margin")
    print(f"supervises: {plan['supervise_checkpoint']}")
    print(f"\n{' '.join(plan['command'])}")


def _plan_args(a) -> dict:
    return extension_plan(
        branch_verdict_path=a.verdict, probe_verdict_path=a.probe_verdict,
        init_from=a.init_from, base_sha256=a.base_sha256,
        mixture_record=a.mixture_record, run_root=a.run_root,
        total_tokens=a.total_tokens, estimated_hours=a.estimated_hours,
        state_path=a.state, device=a.device,
        val_every_steps=a.val_every_steps, no_compile=a.no_compile,
        hub_repo=a.hub_repo)


def _plan(a) -> int:
    try:
        plan = _plan_args(a)
    except (ExtensionRefused, BranchRefused, OSError, ValueError) as exc:
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
    except (ExtensionRefused, BranchRefused, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    holder = lease_holder(a.state)
    if holder is not None:
        print(f"REFUSE: the main lane's lease is held by live pid "
              f"{holder.get('pid')} (acquired {holder.get('acquired_at')}). "
              f"Watch {a.state} and its phase log; this stage is launched once "
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
    parser.add_argument("--verdict", default=DEFAULT_BRANCH_VERDICT,
                        help="gate 2's verdict; not the 250M probe one")
    parser.add_argument("--probe-verdict", default=DEFAULT_PROBE_VERDICT,
                        help="the document the branch's rate was selected from")
    parser.add_argument("--init-from", default=DEFAULT_BRANCH_CHECKPOINT,
                        help="the 1B branch's weights; never the base's")
    parser.add_argument("--base-sha256", default=BASE_SHA256,
                        help="the released base's digest, refused as an input")
    parser.add_argument("--mixture-record", default=DEFAULT_MIXTURE_RECORD)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--total-tokens", type=int, default=EXTENSION_TOKENS)
    parser.add_argument(
        "--estimated-hours", type=float, default=None,
        help="override the projection measured off the branch's throughput")
    parser.add_argument("--state", default=DEFAULT_STATE)
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

    plan = sub.add_parser("plan", help="what the stage would run, and why")
    _common(plan)
    plan.set_defaults(fn=_plan)

    run = sub.add_parser("launch", help="detach it as a supervised phase")
    _common(run)
    run.add_argument("--log", default=DEFAULT_LOG)
    run.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    run.set_defaults(fn=_launch)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(_cli())
