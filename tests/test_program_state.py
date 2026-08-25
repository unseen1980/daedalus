"""Tests for persistent Vast program state and deadline policy."""

import json
import multiprocessing
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest


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


def test_a_lane_transition_leaves_the_main_phase_alone(tmp_path):
    """A pass beside the GPU run is recorded, not mistaken for the run.

    The top-level `phase`/`status` pair is what the progress publisher, the
    resume guard and the deadline check all mean by "the phase". A CPU pass
    writing it would announce that the box had moved on from a 10-hour training
    run that is still going.
    """
    from daedalus.program_state import ProgramStateStore

    started_at = datetime(2026, 8, 24, tzinfo=timezone.utc)
    store = ProgramStateStore(tmp_path / "state.json")
    store.initialize(started_at=started_at, base_sha="abc123")
    store.transition(phase="phase6-stageb-sweep", status="running",
                     now=started_at + timedelta(minutes=1))

    updated = store.transition(phase="phase6-evidence", status="running",
                               now=started_at + timedelta(minutes=2),
                               lane="evidence", details={"device": "cpu"})

    assert updated["phase"] == "phase6-stageb-sweep"
    assert updated["status"] == "running"
    assert updated["lanes"]["evidence"] == {
        "phase": "phase6-evidence",
        "status": "running",
        "updated_at": "2026-08-24T00:02:00Z",
        "details": {"device": "cpu"},
    }
    # `updated_at` means "when the ledger last changed", and a lane advancing is
    # the program advancing.
    assert updated["updated_at"] == "2026-08-24T00:02:00Z"
    events = [json.loads(line) for line in
              (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "lane" not in events[-2], "a main-lane record must read as it always did"
    assert events[-1]["lane"] == "evidence"


def test_a_lane_write_cannot_lose_the_main_lanes_phase(tmp_path):
    """Two lanes are two writers of one snapshot, so the read and the write
    have to be one step.

    `write_json_atomic` makes each write atomic, which is enough for a single
    controller and not enough for two: both read, both update their own keys,
    and the later write puts back a state that predates the earlier one. The
    slow read below is what a real interleaving looks like, made deterministic.
    """
    from daedalus.program_state import ProgramStateStore

    started_at = datetime(2026, 8, 24, tzinfo=timezone.utc)
    state = tmp_path / "state.json"
    ProgramStateStore(state).initialize(started_at=started_at, base_sha="abc123")

    side = ProgramStateStore(state)
    original_load = side.load

    def slow_load():
        value = original_load()
        time.sleep(0.5)          # the read-modify-write window, widened
        return value

    side.load = slow_load
    writer = threading.Thread(target=side.transition, kwargs={
        "phase": "phase6-evidence", "status": "running", "lane": "evidence",
        "now": started_at + timedelta(minutes=1)})
    writer.start()
    time.sleep(0.1)              # lands inside the side lane's window
    ProgramStateStore(state).transition(phase="phase6-stageb-sweep",
                                        status="running",
                                        now=started_at + timedelta(minutes=2))
    writer.join(timeout=30)

    recorded = json.loads(state.read_text())
    assert recorded["phase"] == "phase6-stageb-sweep", (
        "the side lane's write reverted the main lane's phase")
    assert recorded["lanes"]["evidence"]["phase"] == "phase6-evidence"


@pytest.mark.parametrize("lane", ["../escape", "Main", "with space", "", "a/b"])
def test_an_unsafe_lane_name_is_refused(lane):
    """A lane names a lease file. One that resolves elsewhere is two
    controllers sharing a lock, or a lock outside the run directory."""
    from daedalus.program_state import valid_lane

    with pytest.raises(ValueError, match="invalid lane"):
        valid_lane(lane)


def _append_many(path, marker, count, size):
    from daedalus.program_state import ProgramStateStore

    store = ProgramStateStore(path.parent / "state.json", events_path=path)
    for index in range(count):
        store.append_event({"kind": marker, "i": index, "pad": marker * size})


def test_two_writers_cannot_tear_each_others_timeline_records(tmp_path):
    """The timeline has two writers whenever work happens beside a long phase:
    the detached controller driving it, and whatever records a decision
    alongside. A record written as several `write` calls can interleave with
    another process's, leaving two lines that are each half an object -- and a
    timeline that no longer parses is not a timeline.

    The records are padded past a page so that a buffered writer would have to
    split them, which is the condition the guarantee is needed under.
    """
    events = tmp_path / "events.jsonl"
    writers = [multiprocessing.Process(target=_append_many,
                                       args=(events, marker, 40, 3000))
               for marker in ("a", "b")]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=60)

    lines = events.read_text().splitlines()
    decoded = [json.loads(line) for line in lines]      # raises on a torn line
    assert len(decoded) == 80
    assert sorted(event["kind"] for event in decoded) == ["a"] * 40 + ["b"] * 40