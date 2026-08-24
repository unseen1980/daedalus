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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from daedalus.program_state import ProgramDeadline, ProgramStateStore, _timestamp


TERMINAL_STATUSES = {"completed", "halted"}


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
        now: Callable[[], datetime] = _utcnow,
        runner: Optional[Callable[[Sequence[str]], object]] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.store = ProgramStateStore(state_path, events_path=events_path)
        self.state_path = Path(state_path)
        self.lease_path = Path(lease_path) if lease_path is not None else self.state_path.with_name("controller.lock")
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
        self.store._append_event({"kind": "lease_acquired", "at": payload["acquired_at"], "pid": payload["pid"]})
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
        self.store._append_event({"kind": "lease_released", "at": _timestamp(self.now()), "pid": os.getpid()})
        self._lease = None
        return True

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
            self.store.transition(
                phase="finalization",
                status="running",
                now=now,
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
        self.store.transition(
            phase=phase,
            status="running",
            now=self.now(),
            details={"command": command, "max_attempts": max_attempts, "estimated_hours": estimated_hours},
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
            })
            if rc == 0:
                details = {"command": command, "attempts": attempt, "returncodes": returncodes}
                self.store.transition(phase=phase, status="passed", now=self.now(), details=details)
                return details
            if attempt < max_attempts and backoff_sec > 0:
                self.sleeper(backoff_sec)

        details = {"command": command, "attempts": max_attempts, "returncodes": returncodes}
        self.store.transition(phase=phase, status="halted", now=self.now(), details=details)
        raise PhaseFailed(phase, returncodes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="runs/vast-program/state.json")
    parser.add_argument("--lease", default=None)
    parser.add_argument("--base-sha", default="")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--phase", default="bootstrap")
    init.add_argument("--status", default="running")

    run = sub.add_parser("run-phase")
    run.add_argument("--phase", required=True)
    run.add_argument("--estimated-hours", type=float, default=0.0)
    run.add_argument("--max-attempts", type=int, default=1)
    run.add_argument("--backoff-sec", type=float, default=0.0)
    run.add_argument("phase_command", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    controller = VastProgramController(state_path=args.state, lease_path=args.lease)
    if args.command == "init":
        controller.initialize(base_sha=args.base_sha)
        controller.store.transition(phase=args.phase, status=args.status, now=controller.now())
        return 0

    if not Path(args.state).exists():
        controller.initialize(base_sha=args.base_sha)
    controller.acquire_lease()
    try:
        command = args.phase_command
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            raise SystemExit("run-phase requires a command after --")
        controller.run_phase(
            args.phase,
            command,
            estimated_hours=args.estimated_hours,
            max_attempts=args.max_attempts,
            backoff_sec=args.backoff_sec,
        )
    finally:
        controller.release_lease()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
