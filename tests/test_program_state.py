"""Tests for persistent Vast program state and deadline policy."""

import json
from datetime import datetime, timedelta, timezone


def test_deadline_reserves_the_final_eight_hours():
    from daedalus.program_state import ProgramDeadline

    started_at = datetime(2026, 8, 24, tzinfo=timezone.utc)
    deadline = ProgramDeadline(started_at, hard_hours=144, reserve_hours=8)

    assert deadline.stage(started_at + timedelta(hours=135)) == "active"
    assert deadline.stage(started_at + timedelta(hours=136)) == "finalizing"
    assert deadline.stage(started_at + timedelta(hours=144)) == "expired"


def test_new_phase_must_finish_before_finalization():
    from daedalus.program_state import ProgramDeadline

    started_at = datetime(2026, 8, 24, tzinfo=timezone.utc)
    deadline = ProgramDeadline(started_at, hard_hours=144, reserve_hours=8)
    now = started_at + timedelta(hours=130)

    assert deadline.can_start(now, estimated_hours=6)
    assert not deadline.can_start(now, estimated_hours=6.01)


def test_phase_transition_writes_state_and_append_only_event(tmp_path):
    from daedalus.program_state import ProgramStateStore

    started_at = datetime(2026, 8, 24, tzinfo=timezone.utc)
    transitioned_at = started_at + timedelta(minutes=5)
    store = ProgramStateStore(tmp_path / "state.json")

    initial = store.initialize(started_at=started_at, base_sha="abc123")
    updated = store.transition(
        phase="bootstrap",
        status="running",
        now=transitioned_at,
        details={"check": "tests"},
    )

    assert initial["schema"] == 1
    assert updated["phase"] == "bootstrap"
    assert updated["status"] == "running"
    assert json.loads((tmp_path / "state.json").read_text()) == updated
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == ["initialized", "transition"]
    assert events[-1]["details"] == {"check": "tests"}
    assert not (tmp_path / "state.json.tmp").exists()