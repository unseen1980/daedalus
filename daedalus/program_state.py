"""Persistent state primitives for the unattended Vast research program."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


PROGRAM_STATE_SCHEMA = 1


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ProgramStateStore:
    """Atomic program snapshot accompanied by a durable event timeline."""

    def __init__(self, state_path, events_path: Optional[Path] = None):
        self.state_path = Path(state_path)
        self.events_path = (Path(events_path) if events_path is not None
                            else self.state_path.with_name("events.jsonl"))

    def _append_event(self, event: dict) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

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
        _write_json_atomic(self.state_path, state)
        self._append_event({
            "kind": "initialized",
            "at": _timestamp(started_at),
            "base_sha": base_sha,
        })
        return state

    def transition(self, *, phase: str, status: str, now: datetime,
                   details: Optional[dict] = None) -> dict:
        state = self.load()
        state.update({
            "phase": phase,
            "status": status,
            "updated_at": _timestamp(now),
            "details": dict(details or {}),
        })
        _write_json_atomic(self.state_path, state)
        self._append_event({
            "kind": "transition",
            "at": _timestamp(now),
            "phase": phase,
            "status": status,
            "details": dict(details or {}),
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