"""Persistent state primitives for the unattended Vast research program."""

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:                                            # POSIX only; see `_exclusive`
    import fcntl
except ImportError:                             # pragma: no cover - not Linux
    fcntl = None


PROGRAM_STATE_SCHEMA = 1

#: The lane that owns the top-level `phase`/`status` pair -- the box's one
#: expensive resource, and what every existing reader of the snapshot means by
#: "the phase". Named rather than spelled `None` so the default is visible in
#: the events it writes.
MAIN_LANE = "main"

#: A lane name becomes part of a lease filename, so it is restricted to what is
#: safe there. Refused rather than sanitised: a lane silently renamed into
#: another lane's lease is two controllers sharing one lock.
_LANE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


def valid_lane(lane: str) -> str:
    """Return ``lane`` if it can safely name a lease file, else raise."""

    if lane == MAIN_LANE or _LANE_NAME.match(lane or ""):
        return lane
    raise ValueError(
        f"invalid lane {lane!r}: lanes name a lease file, so they must match "
        f"{_LANE_NAME.pattern}")


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: dict) -> None:
    """Replace ``path`` with ``payload`` so a crash never leaves a torn file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


#: Retained for callers written before the helper became part of the API.
_write_json_atomic = write_json_atomic


@contextmanager
def _exclusive(state_path: Path):
    """Serialize one read-modify-write of the snapshot across processes.

    `write_json_atomic` makes each *write* atomic, which is enough while one
    controller owns the file. It is not enough once a second lane writes the
    same snapshot beside the first: both read, both update their own keys, and
    the later write puts back a copy of the state that predates the earlier
    one. The main lane's phase would silently revert to whatever the side lane
    had read minutes before -- and the snapshot is what the progress branch
    publishes, so the first sign of it would be a heartbeat naming a phase that
    finished hours ago.

    A flock on a sidecar file, held across the read and the write, is the
    smallest thing that closes that. It is advisory and process-scoped, so it
    costs nothing when only one lane runs, which is still the normal case.
    Absent `fcntl` (non-POSIX) it degrades to no locking rather than failing:
    the program runs on Linux, and a single-lane caller there behaves exactly
    as it did before.
    """

    if fcntl is None:                           # pragma: no cover - not Linux
        yield
        return
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(state_path.name + ".lock")
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


class ProgramStateStore:
    """Atomic program snapshot accompanied by a durable event timeline."""

    def __init__(self, state_path, events_path: Optional[Path] = None):
        self.state_path = Path(state_path)
        self.events_path = (Path(events_path) if events_path is not None
                            else self.state_path.with_name("events.jsonl"))

    def append_event(self, event: dict) -> None:
        """Append one durable, machine-parsable entry to the phase timeline.

        One `os.write` on an `O_APPEND` descriptor, rather than a buffered
        `write` that may be split into several. The timeline has more than one
        writer whenever work happens beside a long phase -- the detached
        controller driving that phase, and whatever records a decision alongside
        it -- and a record split across two `write` calls can be interleaved with
        another process's, producing two lines that are each half a JSON object
        and a timeline that no longer parses.

        Linux takes the inode lock for a write to a regular file and resolves
        `O_APPEND`'s offset under it, so one `write` cannot interleave with
        another's or land on top of it. That property is what makes a lock
        unnecessary here, and it is why this is a single call rather than a
        convenient loop.
        """

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(event, sort_keys=True) + "\n").encode()
        fd = os.open(self.events_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _append_event(self, event: dict) -> None:
        """Retained for callers written before the helper became part of the API."""

        self.append_event(event)

    def load(self) -> dict:
        with self.state_path.open() as handle:
            return json.load(handle)

    def initialize(self, *, started_at: datetime, base_sha: str,
                   hard_hours: float = 144.0,
                   reserve_hours: float = 8.0) -> dict:
        state = {
            "schema": PROGRAM_STATE_SCHEMA,
            "started_at": _timestamp(started_at),
            "updated_at": _timestamp(started_at),
            "base_sha": base_sha,
            "hard_hours": hard_hours,
            "reserve_hours": reserve_hours,
            "phase": "not_started",
            "status": "initialized",
            "details": {},
        }
        write_json_atomic(self.state_path, state)
        self._append_event({
            "kind": "initialized",
            "at": _timestamp(started_at),
            "base_sha": base_sha,
        })
        return state

    def transition(self, *, phase: str, status: str, now: datetime,
                   details: Optional[dict] = None,
                   lane: str = MAIN_LANE) -> dict:
        """Record a phase transition for one lane of the program.

        The main lane keeps writing the top-level `phase`/`status` pair, which
        is what every existing reader -- the progress publisher, the resume
        guard, this module's own deadline check -- already means by "the
        phase". A named lane writes under `lanes` instead and leaves that pair
        alone, so a CPU pass running beside a GPU phase is recorded without
        claiming to *be* the phase. Both move `updated_at`: it means "when the
        ledger last changed", and a lane advancing is the program advancing.
        """

        lane = valid_lane(lane)
        details = dict(details or {})
        with _exclusive(self.state_path):
            state = self.load()
            if lane == MAIN_LANE:
                state.update({
                    "phase": phase,
                    "status": status,
                    "updated_at": _timestamp(now),
                    "details": details,
                })
            else:
                lanes = dict(state.get("lanes") or {})
                lanes[lane] = {
                    "phase": phase,
                    "status": status,
                    "updated_at": _timestamp(now),
                    "details": details,
                }
                state["lanes"] = lanes
                state["updated_at"] = _timestamp(now)
            write_json_atomic(self.state_path, state)
        event = {
            "kind": "transition",
            "at": _timestamp(now),
            "phase": phase,
            "status": status,
            "details": details,
        }
        if lane != MAIN_LANE:
            # Only when it is not the main lane, so a timeline written before
            # lanes existed and one written after read identically for the
            # phases that carry the box.
            event["lane"] = lane
        self._append_event(event)
        return state

    def set_base_sha(self, *, base_sha: str, now: datetime) -> dict:
        """Update the recorded source baseline without resetting run timing."""

        with _exclusive(self.state_path):
            state = self.load()
            state["base_sha"] = base_sha
            state["updated_at"] = _timestamp(now)
            write_json_atomic(self.state_path, state)
        self._append_event({
            "kind": "base_sha_updated",
            "at": _timestamp(now),
            "base_sha": base_sha,
        })
        return state


@dataclass(frozen=True)
class ProgramDeadline:
    """Hard deadline with a protected finalization window."""

    started_at: datetime
    hard_hours: float = 144.0
    reserve_hours: float = 8.0

    @property
    def expires_at(self) -> datetime:
        return self.started_at + timedelta(hours=self.hard_hours)

    @property
    def finalizes_at(self) -> datetime:
        return self.expires_at - timedelta(hours=self.reserve_hours)

    def stage(self, now: datetime) -> str:
        if now >= self.expires_at:
            return "expired"
        if now >= self.finalizes_at:
            return "finalizing"
        return "active"

    def can_start(self, now: datetime, estimated_hours: float) -> bool:
        return now + timedelta(hours=estimated_hours) <= self.finalizes_at