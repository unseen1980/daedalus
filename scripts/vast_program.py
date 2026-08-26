"""Deterministic controller for the unattended Vast research program.

The controller is intentionally small and boring: it owns phase state, deadline
gates, one-process lease ownership, and bounded command retries. Long-running
training is still delegated to the existing supervised launchers; this module is
the durable phase ledger that makes those launches auditable and restart-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from daedalus.program_state import (MAIN_LANE, ProgramDeadline,
                                    ProgramStateStore, _timestamp, valid_lane)


TERMINAL_STATUSES = {"completed", "halted"}

#: Above this estimate a phase outlives the session that starts it, so it must be
#: detached. A turn's own command timeout is ten minutes, so anything at or past
#: this bound was never going to survive in-session anyway -- it would only look
#: like it had until the session ended and killed the trainer with it.
DETACH_REQUIRED_HOURS = 0.25

#: Seconds between attempts of a *supervised* phase. `run-phase`'s own default
#: is zero, which is right for a quick command and wrong for a trainer: three
#: attempts in as many seconds spend the budget on a cause -- a filesystem
#: hiccup, a driver blip -- that has had no time to clear. Every
#: orchestrator-launched run in this program has used `daedalus.supervise`'s 60s.
SUPERVISED_BACKOFF_SEC = 60.0

#: Minutes without a metrics row before the watchdog calls a supervised run
#: stalled. Matches `qat_recovery`, `conv_health`, `mixture_opt` and
#: `architecture_sweep`, all of which pass 20.0.
SUPERVISED_STALL_MIN = 20.0


def default_lease_name(lane: str = MAIN_LANE) -> str:
    """The lease filename a lane owns.

    One lease per lane, because the lease answers "is someone already driving
    *this* work", and a CPU pass beside a GPU run is not the same work. The
    main lane keeps the original filename so a controller started before lanes
    existed and one started after contend for the same lock rather than both
    believing they own the box.
    """

    return "controller.lock" if lane == MAIN_LANE else f"controller-{lane}.lock"


class ControllerLeaseError(RuntimeError):
    """Raised when another live controller owns the program lease."""


class DeadlineRefused(RuntimeError):
    """Raised when a phase would violate the reserved finalization window."""


class PhaseFailed(RuntimeError):
    """Raised after a phase exhausts its bounded retry budget."""

    def __init__(self, phase: str, returncodes: list[int]):
        super().__init__(f"phase {phase!r} failed with return codes {returncodes}")
        self.phase = phase
        self.returncodes = returncodes


class TerminalStateError(RuntimeError):
    """Raised when attempting to mutate a completed or halted program."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def process_start_ticks(pid: int) -> Optional[int]:
    """Return Linux `/proc/<pid>/stat` start ticks, or None off Linux."""

    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        after_comm = text.rsplit(")", 1)[1]
        fields = after_comm.split()
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def running_in_own_session() -> bool:
    """True when this process leads its own session, as ``setsid`` makes it.

    What the detach guard actually wants to know is whether the phase will
    outlive the session that started it, and that is a property of the process,
    not of an argument. Asking the OS rather than trusting ``--detach`` matters
    because the detached child is deliberately handed an argv *without* the
    flag, so that it cannot fork again -- and a flag-based guard then refuses
    the very child the parent just detached, which made every phase at or past
    the bound un-startable.

    A session leader's children sit in its session rather than in the
    engineering session's process group, so the keeper's group kill cannot
    reach them. That is exactly the property ``--detach`` buys.
    """

    try:
        return os.getsid(0) == os.getpid()
    except (AttributeError, OSError):  # non-POSIX: the guard is advisory there
        return False


def process_is_alive(pid: int, start_ticks: Optional[int]) -> bool:
    """True only when pid exists and, when knowable, matches its start time."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    current_ticks = process_start_ticks(pid)
    return start_ticks is None or current_ticks is None or current_ticks == start_ticks


class VastProgramController:
    """Durable phase controller with a single live owner."""

    def __init__(
        self,
        *,
        state_path,
        lease_path=None,
        events_path=None,
        lane: str = MAIN_LANE,
        now: Callable[[], datetime] = _utcnow,
        runner: Optional[Callable[[Sequence[str]], object]] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.store = ProgramStateStore(state_path, events_path=events_path)
        self.state_path = Path(state_path)
        self.lane = valid_lane(lane)
        self.lease_path = (Path(lease_path) if lease_path is not None
                           else self.state_path.with_name(default_lease_name(self.lane)))
        self.now = now
        self.runner = runner or self._run_command
        self.sleeper = sleeper
        self._lease: Optional[dict] = None

    @staticmethod
    def _run_command(command: Sequence[str]) -> int:
        return subprocess.run(list(command)).returncode

    def initialize(
        self,
        *,
        base_sha: str,
        started_at: Optional[datetime] = None,
        hard_hours: float = 144.0,
        reserve_hours: float = 8.0,
    ) -> dict:
        return self.store.initialize(
            started_at=started_at or self.now(),
            base_sha=base_sha,
            hard_hours=hard_hours,
            reserve_hours=reserve_hours,
        )

    def _lease_payload(self) -> dict:
        pid = os.getpid()
        return {
            "schema": 1,
            "pid": pid,
            "start_ticks": process_start_ticks(pid),
            "acquired_at": _timestamp(self.now()),
        }

    def _lease_is_live(self, payload: Optional[dict]) -> bool:
        if not payload:
            return False
        try:
            pid = int(payload["pid"])
        except (KeyError, TypeError, ValueError):
            return False
        return process_is_alive(pid, payload.get("start_ticks"))

    def acquire_lease(self) -> dict:
        """Acquire the controller lease, replacing malformed or stale owners."""

        payload = self._lease_payload()
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            existing = _read_json(self.lease_path)
            if self._lease_is_live(existing):
                raise ControllerLeaseError(
                    f"active controller pid {existing.get('pid')} owns {self.lease_path}"
                )
            _write_json_atomic(self.lease_path, payload)
        else:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        self._lease = payload
        self.store._append_event({"kind": "lease_acquired", "at": payload["acquired_at"],
                                  "pid": payload["pid"], **self._lane_field()})
        return payload

    def release_lease(self) -> bool:
        """Release the lease only when this process still owns it."""

        existing = _read_json(self.lease_path)
        if not existing or self._lease is None:
            return False
        if existing.get("pid") != self._lease.get("pid"):
            return False
        if existing.get("start_ticks") != self._lease.get("start_ticks"):
            return False
        self.lease_path.unlink(missing_ok=True)
        self.store._append_event({"kind": "lease_released", "at": _timestamp(self.now()),
                                  "pid": os.getpid(), **self._lane_field()})
        self._lease = None
        return True

    def _lane_field(self) -> dict:
        """`{"lane": ...}` for a side lane, nothing for the main one, so a
        timeline written before lanes existed still reads identically."""

        return {} if self.lane == MAIN_LANE else {"lane": self.lane}

    def note(self, *, phase: str, kind: str = "decision",
             details: Optional[dict] = None) -> dict:
        """Append one record to the timeline without taking the lease.

        The lease exists so that two controllers cannot drive phases at once:
        its subject is the state snapshot and the transitions that rewrite it.
        A note writes neither. It is an append to a timeline that is append-only
        by construction, and `ProgramStateStore.append_event` makes that append
        atomic against the controller writing to the same file.

        Making a note wait for the lease would be the wrong reading of what the
        lease protects, and an expensive one. A long phase holds it for hours --
        phase 5's paired escalation held it for 8.6 -- and those hours are
        exactly when engineering work happens beside the run. Under a
        lease-taking note, every decision made in that window is unrecordable
        until the training finishes, which is how a decision ends up documented
        only in a commit message, or appended to `events.jsonl` out of band by
        something that is not the controller. Both happened during phase 5.

        A terminal program is *not* refused: why a run halted is the record most
        worth keeping, and it can only be written after the halt.
        """

        record = {"kind": kind, "at": _timestamp(self.now()), "phase": phase,
                  "pid": os.getpid(), "details": dict(details or {}),
                  **self._lane_field()}
        self.store.append_event(record)
        return record

    def _load_state_or_raise(self) -> dict:
        state = self.store.load()
        status = state.get("status")
        if status in TERMINAL_STATUSES:
            raise TerminalStateError(f"program already {status}")
        return state

    def _deadline(self, state: dict) -> ProgramDeadline:
        started_at = datetime.fromisoformat(state["started_at"].replace("Z", "+00:00"))
        return ProgramDeadline(
            started_at,
            hard_hours=float(state.get("hard_hours", 144.0)),
            reserve_hours=float(state.get("reserve_hours", 8.0)),
        )

    def _check_deadline(self, phase: str, estimated_hours: float) -> None:
        state = self._load_state_or_raise()
        deadline = self._deadline(state)
        now = self.now()
        stage = deadline.stage(now)
        if stage != "active" or not deadline.can_start(now, estimated_hours):
            # Recorded in this controller's own lane. A side lane refused by
            # the deadline must not rewrite the top-level phase to
            # `finalization` while the main lane's run is still going: the
            # progress branch would announce finalization hours early, on
            # behalf of a pass that never started.
            self.store.transition(
                phase="finalization",
                status="running",
                now=now,
                lane=self.lane,
                details={
                    "refused_phase": phase,
                    "deadline_stage": stage,
                    "estimated_hours": estimated_hours,
                    "finalizes_at": _timestamp(deadline.finalizes_at),
                    "expires_at": _timestamp(deadline.expires_at),
                },
            )
            raise DeadlineRefused(f"refusing {phase!r}: deadline stage is {stage}")

    @staticmethod
    def _returncode(result: object) -> int:
        if isinstance(result, int):
            return result
        return int(getattr(result, "returncode"))

    def run_phase(
        self,
        phase: str,
        command: Sequence[str],
        *,
        estimated_hours: float = 0.0,
        max_attempts: int = 1,
        backoff_sec: float = 0.0,
    ) -> dict:
        """Run one phase command with bounded retries and durable status."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        command = list(command)
        self._check_deadline(phase, estimated_hours)
        started_at = self.now()
        self.store.transition(
            phase=phase,
            status="running",
            now=started_at,
            lane=self.lane,
            # `started_at` explicitly, because `updated_at` stopped meaning
            # "when this phase began" as soon as a second lane could bump it --
            # and "how long has this been running" is the first question asked
            # of a phase that is still holding the lease.
            details={"command": command, "max_attempts": max_attempts,
                     "estimated_hours": estimated_hours,
                     "started_at": _timestamp(started_at)},
        )

        returncodes: list[int] = []
        for attempt in range(1, max_attempts + 1):
            started_at = self.now()
            rc = self._returncode(self.runner(command))
            returncodes.append(rc)
            self.store._append_event({
                "kind": "phase_attempt",
                "at": _timestamp(started_at),
                "phase": phase,
                "attempt": attempt,
                "returncode": rc,
                **self._lane_field(),
            })
            if rc == 0:
                details = {"command": command, "attempts": attempt, "returncodes": returncodes}
                self.store.transition(phase=phase, status="passed", now=self.now(),
                                      lane=self.lane, details=details)
                return details
            if attempt < max_attempts and backoff_sec > 0:
                self.sleeper(backoff_sec)

        # "failed", not "halted". `halted` is terminal -- it is how the program
        # records PROGRAM_HALTED -- so marking a phase halted wedged the whole
        # controller on the first non-zero exit, and every later phase, related
        # or not, raised TerminalStateError until someone transitioned the state
        # by hand. A malformed argv is not a hard blocker; the plan's rule is
        # that a failed phase preserves its artifacts and unrelated safe phases
        # continue. `failed` is already one of the progress publisher's
        # attention statuses, so this is still surfaced, just not fatal.
        details = {"command": command, "attempts": max_attempts, "returncodes": returncodes}
        self.store.transition(phase=phase, status="failed", now=self.now(),
                              lane=self.lane, details=details)
        raise PhaseFailed(phase, returncodes)


def supervised_runner(
    checkpoint,
    *,
    phase: str = "",
    lane: str = MAIN_LANE,
    max_attempts: int = 1,
    backoff_sec: float = SUPERVISED_BACKOFF_SEC,
    watchdog_tokens: int = 0,
    stall_min: float = SUPERVISED_STALL_MIN,
    report: Optional[dict] = None,
) -> Callable[[Sequence[str]], int]:
    """A phase runner that launches through the supervised trainer loop.

    Detaching a phase makes it outlive the session that started it. It does not
    make it *continue*. The plain runner is `subprocess.run` on the argv it was
    given, so for a trainer:

    - a retry re-runs the same argv, which starts at step zero and overwrites
      the checkpoint the previous attempt wrote;
    - a relaunch after the launching session, the controller or the box died is
      attempt one of a fresh process, so it does the same thing beside a
      checkpoint nobody opened -- how phase 4 lost 60.3M tokens;
    - no in-flight marker is written, and that marker is what
      `scripts/boot_resume.py` continues a run from after a reboot and what
      `session_keeper.supervised_job_probe` reads to know the box is busy. A
      bare-command trainer is invisible to both for its entire run.

    Every phase up to here escaped that only by going through an orchestrator
    (`qat_recovery`, `conv_health`, `architecture_sweep`, `mixture_opt`,
    `tokenizer_lab`) that wrapped `run_with_resume` itself. Phase 8's runs are
    the longest in the program and it has no orchestrator, so the launcher
    grows the capability instead of a sixth caller reimplementing it -- and
    getting it right by accident is not something to bet a 1B-token run on.

    The retry budget is handed *inward*: `run_with_resume` is the only loop that
    knows to add `--resume` and to stop rather than continue a run the watchdog
    halted, so the controller runs it once. `report`, when given, is filled with
    the supervisor's own attempt history for the timeline.
    """

    checkpoint = Path(checkpoint)

    def run(command: Sequence[str]) -> int:
        from daedalus.supervise import (TrainingFailed, run_with_resume,
                                        start_watchdog, stop_watchdog)

        run_dir = checkpoint.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        watchdog = None
        if watchdog_tokens > 0:
            watchdog = start_watchdog(run_dir.name, str(run_dir), watchdog_tokens,
                                      stall_min=stall_min, supervised=True)
        try:
            outcome = run_with_resume(
                list(command), str(checkpoint),
                max_attempts=max_attempts,
                backoff_sec=backoff_sec,
                # The marker `watchdog.py` writes for this run. Without it a
                # divergence halt reads as an ordinary crash and the diverged
                # checkpoint is resumed with no watchdog left running.
                halt_marker=str(run_dir / "HALTED"),
                inflight_extra={"phase": phase, "lane": lane})
        except TrainingFailed as exc:
            if report is not None:
                report.update({"attempts": exc.attempts,
                               "returncodes": list(exc.returncodes),
                               "halt": exc.halt})
            return (exc.returncodes[-1] if exc.returncodes else 1) or 1
        finally:
            stop_watchdog(watchdog)
        if report is not None:
            report.update(outcome)
        return 0

    return run


def detached_phase_argv(
    *,
    state,
    phase: str,
    command: Sequence[str],
    lease=None,
    base_sha: str = "",
    lane: str = MAIN_LANE,
    estimated_hours: float = 0.0,
    max_attempts: int = 1,
    backoff_sec: Optional[float] = None,
    supervise_checkpoint=None,
    watchdog_tokens: int = 0,
    stall_min: float = SUPERVISED_STALL_MIN,
) -> list[str]:
    """Rebuild this invocation for the detached controller, without ``--detach``.

    Rebuilt from the parsed values rather than filtered out of ``sys.argv`` so a
    phase command that happens to contain ``--detach`` cannot lose an argument
    or, worse, make the child detach again.
    """

    argv = [sys.executable, str(Path(__file__).resolve()), "--state", str(state)]
    if lease:
        argv += ["--lease", str(lease)]
    if base_sha:
        argv += ["--base-sha", str(base_sha)]
    if lane != MAIN_LANE:
        # Dropping it would detach the side pass into the main lane, where it
        # would take the GPU run's lease -- and be refused by it.
        argv += ["--lane", str(lane)]
    argv += [
        "run-phase",
        "--phase", str(phase),
        "--estimated-hours", str(float(estimated_hours)),
        "--max-attempts", str(int(max_attempts)),
    ]
    if backoff_sec is not None:
        argv += ["--backoff-sec", str(float(backoff_sec))]
    # The child is the process that actually runs the command, so supervision
    # dropped here leaves `--detach` working and `--supervise-checkpoint`
    # silently inert -- which is the combination a long training phase is
    # launched with, and the one whose absence is invisible until a relaunch
    # starts at step zero.
    if supervise_checkpoint:
        argv += ["--supervise-checkpoint", str(supervise_checkpoint)]
    if watchdog_tokens:
        argv += ["--watchdog-tokens", str(int(watchdog_tokens)),
                 "--stall-min", str(float(stall_min))]
    argv += ["--", *[str(part) for part in command]]
    return argv


def detach_phase(
    *,
    state,
    phase: str,
    command: Sequence[str],
    log_path,
    lease=None,
    base_sha: str = "",
    lane: str = MAIN_LANE,
    estimated_hours: float = 0.0,
    max_attempts: int = 1,
    backoff_sec: Optional[float] = None,
    supervise_checkpoint=None,
    watchdog_tokens: int = 0,
    stall_min: float = SUPERVISED_STALL_MIN,
    spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict:
    """Start the phase controller in its own session and return immediately.

    A phase must outlive the engineering session that starts it. Each session
    runs in its own process group so the keeper can reap the whole tree when it
    cuts one short, and anything the session launches inherits that group -- so
    a phase started in-session dies whenever the session ends, on a normal exit
    as readily as on a timeout, taking the trainer and its GPU work with it.
    That is how the phase 4 sweep lost its second arm at step 460 of 1636 and
    left the box idle. Detaching into a fresh session removes the phase from the
    session's lifetime: the turn ends, the keeper starts the next one, and the
    sweep keeps running. Ownership is unchanged -- the detached controller takes
    the same single-owner lease, so this cannot start a second writer.
    """

    argv = detached_phase_argv(
        state=state,
        phase=phase,
        command=command,
        lease=lease,
        base_sha=base_sha,
        lane=lane,
        estimated_hours=estimated_hours,
        max_attempts=max_attempts,
        backoff_sec=backoff_sec,
        supervise_checkpoint=supervise_checkpoint,
        watchdog_tokens=watchdog_tokens,
        stall_min=stall_min,
    )
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = spawn(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"pid": process.pid, "log": str(log_path), "argv": argv}


def take_lease(controller: "VastProgramController") -> dict:
    """Acquire the lane's lease, or exit saying who is already driving it.

    A busy lease is an *expected* answer, not a defect: a session that comes
    back while a detached ten-hour phase is still running asks for it every
    time. Uncaught, that answered with a traceback, which reads as "the
    controller is broken" rather than "this is already running" -- and the
    obvious response to a broken controller is to try again harder. So the
    refusal names the phase in progress and where to watch it.
    """

    try:
        return controller.acquire_lease()
    except ControllerLeaseError as exc:
        try:
            state = controller.store.load()
        except (OSError, ValueError):
            state = {}
        if controller.lane != MAIN_LANE:
            current = (state.get("lanes") or {}).get(controller.lane) or {}
        else:
            current = state
        since = (current.get("details") or {}).get("started_at") \
            or current.get("updated_at", "an unrecorded time")
        raise SystemExit(
            f"{exc}: lane {controller.lane!r} is running "
            f"{current.get('phase', 'an unrecorded phase')} "
            f"({current.get('status', 'unknown')}) since {since}. Watch it in "
            f"{controller.state_path} and its phase log; do not relaunch it. "
            f"A pass that does not contend for the same device can run beside "
            f"it under its own --lane."
        ) from exc


def _details(raw: str) -> dict:
    """`--details-json`, or a refusal naming what was wrong with it."""
    try:
        details = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --details-json: {exc}") from exc
    if not isinstance(details, dict):
        raise SystemExit("--details-json must decode to an object")
    return details


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="runs/vast-program/state.json")
    parser.add_argument("--lease", default=None)
    parser.add_argument("--base-sha", default="")
    parser.add_argument(
        "--lane", default=MAIN_LANE,
        help="which lane of the program this phase belongs to. The default "
             "'main' lane owns the box's phase and the top-level status; a "
             "named lane records a pass that runs beside it without "
             "contending for the same device, and takes its own lease")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--phase", default="bootstrap")
    init.add_argument("--status", default="running")

    set_base = sub.add_parser("set-base-sha")
    set_base.add_argument("base_sha")

    transition = sub.add_parser("transition")
    transition.add_argument("--phase", required=True)
    transition.add_argument("--status", required=True)
    transition.add_argument("--details-json", default="{}")

    note = sub.add_parser(
        "note", help="append a record to the timeline without the lease, so a "
                     "decision made beside a running phase is still recorded")
    note.add_argument("--phase", required=True)
    note.add_argument("--kind", default="decision")
    note.add_argument("--details-json", default="{}")

    run = sub.add_parser("run-phase")
    run.add_argument("--phase", required=True)
    run.add_argument("--estimated-hours", type=float, default=0.0)
    run.add_argument("--max-attempts", type=int, default=1)
    run.add_argument(
        "--backoff-sec", type=float, default=None,
        help=f"seconds between attempts (default 0 for a plain command, "
             f"{SUPERVISED_BACKOFF_SEC:g} under --supervise-checkpoint)")
    run.add_argument(
        "--supervise-checkpoint", default=None,
        help="run the phase through the supervised trainer loop, resuming this "
             "checkpoint on a retry or a relaunch instead of starting over, and "
             "leaving the in-flight marker boot resume and the keeper read")
    run.add_argument(
        "--watchdog-tokens", type=int, default=0,
        help="target tokens for a watchdog beside a --supervise-checkpoint run; "
             "omitted, the run has no divergence or stall detection")
    run.add_argument("--stall-min", type=float, default=SUPERVISED_STALL_MIN)
    run.add_argument(
        "--detach",
        action="store_true",
        help="own the phase from a new session so it outlives the caller",
    )
    run.add_argument("--log", default=None, help="detached phase log (default runs/vast-program/logs/<phase>.log)")
    run.add_argument("phase_command", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    try:
        lane = valid_lane(args.lane)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    controller = VastProgramController(state_path=args.state, lease_path=args.lease,
                                       lane=lane)
    if args.command == "init":
        controller.initialize(base_sha=args.base_sha)
        controller.store.transition(phase=args.phase, status=args.status, now=controller.now())
        return 0

    if args.command == "set-base-sha":
        if not Path(args.state).exists():
            raise SystemExit(f"state file does not exist: {args.state}")
        take_lease(controller)
        try:
            controller.store.set_base_sha(base_sha=args.base_sha, now=controller.now())
        finally:
            controller.release_lease()
        return 0

    if args.command == "note":
        # No lease and no state read: the timeline is append-only and a note is
        # an append. Deliberately usable while a detached phase owns the lease,
        # which is when most notes are written.
        controller.note(phase=args.phase, kind=args.kind,
                        details=_details(args.details_json))
        return 0

    if args.command == "transition":
        if not Path(args.state).exists():
            raise SystemExit(f"state file does not exist: {args.state}")
        details = _details(args.details_json)
        take_lease(controller)
        try:
            controller.store.transition(phase=args.phase, status=args.status,
                                        now=controller.now(), lane=lane, details=details)
        finally:
            controller.release_lease()
        return 0

    command = args.phase_command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("run-phase requires a command after --")

    if args.supervise_checkpoint and "--resume" in command:
        # `--resume` on attempt one restores the *finished* run's step and token
        # count, so the phase writes no metrics row and exits 0 -- a phase that
        # trained nothing looks exactly like one that finished early. The
        # supervisor adds `--resume` when continuing is the right answer;
        # `--init-from` is how a phase starts from someone else's weights.
        raise SystemExit(
            f"phase {args.phase!r} passes --resume to a supervised command: "
            f"--supervise-checkpoint adds --resume itself on a retry or a "
            f"relaunch, and an explicit one makes attempt one train nothing "
            f"and exit 0. Use --init-from to start from existing weights.")

    if (not args.detach and args.estimated_hours >= DETACH_REQUIRED_HOURS
            and not running_in_own_session()):
        raise SystemExit(
            f"phase {args.phase!r} is estimated at {args.estimated_hours:g}h, at or "
            f"past the {DETACH_REQUIRED_HOURS:g}h bound where a phase outlives the "
            f"session that starts it; re-run it with --detach"
        )

    if args.detach:
        # State and lease are left to the detached controller so the caller
        # never holds ownership it is about to walk away from.
        log_path = args.log or Path(args.state).with_name("logs") / f"{args.phase}.log"
        started = detach_phase(
            state=args.state,
            phase=args.phase,
            command=command,
            log_path=log_path,
            lease=args.lease,
            base_sha=args.base_sha,
            lane=lane,
            estimated_hours=args.estimated_hours,
            max_attempts=args.max_attempts,
            backoff_sec=args.backoff_sec,
            supervise_checkpoint=args.supervise_checkpoint,
            watchdog_tokens=args.watchdog_tokens,
            stall_min=args.stall_min,
        )
        print(f"detached phase {args.phase} lane {lane} pid {started['pid']} "
              f"log {started['log']}")
        return 0

    if not Path(args.state).exists():
        controller.initialize(base_sha=args.base_sha)

    attempts = args.max_attempts
    supervised: Optional[dict] = None
    if args.supervise_checkpoint:
        supervised = {}
        controller.runner = supervised_runner(
            args.supervise_checkpoint,
            phase=args.phase,
            lane=lane,
            max_attempts=args.max_attempts,
            backoff_sec=(SUPERVISED_BACKOFF_SEC if args.backoff_sec is None
                         else args.backoff_sec),
            watchdog_tokens=args.watchdog_tokens,
            stall_min=args.stall_min,
            report=supervised,
        )
        # One attempt at this level: the supervisor owns the retries, because it
        # is the only loop that adds `--resume` and that refuses to continue a
        # run the watchdog halted. Retrying it from out here would restart a
        # halted run and square the budget.
        attempts = 1

    take_lease(controller)
    try:
        controller.run_phase(
            args.phase,
            command,
            estimated_hours=args.estimated_hours,
            max_attempts=attempts,
            backoff_sec=args.backoff_sec or 0.0,
        )
    finally:
        # The phase details record the controller's attempts, which under
        # supervision is always one; without this the timeline says a run that
        # crashed twice and resumed twice went through first time. Empty when
        # the phase never ran -- a deadline refusal has nothing to report.
        if supervised:
            controller.note(phase=args.phase, kind="supervised_run",
                            details={"checkpoint": str(args.supervise_checkpoint),
                                     **supervised})
        controller.release_lease()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
