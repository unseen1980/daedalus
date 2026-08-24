"""Tests for the deterministic Vast program controller."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def test_controller_acquires_and_releases_lease(tmp_path):
    from scripts.vast_program import VastProgramController


    state = tmp_path / "state.json"
    lease = tmp_path / "controller.lock"
    controller = VastProgramController(state_path=state, lease_path=lease, now=lambda: NOW)

    controller.initialize(base_sha="abc123")
    acquired = controller.acquire_lease()


    assert acquired["pid"] == os.getpid()
    assert json.loads(lease.read_text())["start_ticks"] == acquired["start_ticks"]
    assert controller.release_lease() is True
    assert not lease.exists()


def test_live_lease_refuses_second_controller(tmp_path):
    from scripts.vast_program import ControllerLeaseError, VastProgramController


    lease = tmp_path / "controller.lock"
    first = VastProgramController(state_path=tmp_path / "state.json", lease_path=lease, now=lambda: NOW)
    second = VastProgramController(state_path=tmp_path / "state.json", lease_path=lease, now=lambda: NOW)

    first.initialize(base_sha="abc123")
    first.acquire_lease()

    with pytest.raises(ControllerLeaseError, match="active controller"):
        second.acquire_lease()


def test_stale_lease_is_replaced(tmp_path):
    from scripts.vast_program import VastProgramController


    lease = tmp_path / "controller.lock"
    lease.write_text(json.dumps({"pid": 999_999_999, "start_ticks": 1, "acquired_at": "old"}))
    controller = VastProgramController(state_path=tmp_path / "state.json", lease_path=lease, now=lambda: NOW)
    controller.initialize(base_sha="abc123")


    acquired = controller.acquire_lease()


    assert acquired["pid"] == os.getpid()
    assert json.loads(lease.read_text())["pid"] == os.getpid()


def test_phase_refuses_to_start_inside_finalization_window(tmp_path):
    from scripts.vast_program import DeadlineRefused, VastProgramController


    started = NOW - timedelta(hours=137)
    controller = VastProgramController(state_path=tmp_path / "state.json", now=lambda: NOW)
    controller.initialize(base_sha="abc123", started_at=started)


    with pytest.raises(DeadlineRefused, match="finalizing"):
        controller.run_phase("too-late", ["true"], estimated_hours=0.1)

    assert controller.store.load()["phase"] == "finalization"
    assert controller.store.load()["status"] == "running"


def test_phase_records_bounded_retries_then_passes(tmp_path):
    from scripts.vast_program import VastProgramController


    commands = []

    def runner(command):
        commands.append(command)
        return 1 if len(commands) == 1 else 0

    controller = VastProgramController(state_path=tmp_path / "state.json", now=lambda: NOW, runner=runner)
    controller.initialize(base_sha="abc123")


    result = controller.run_phase("unit", ["pytest", "tests/test_program_state.py"], max_attempts=2)


    assert result["attempts"] == 2
    assert result["returncodes"] == [1, 0]
    assert controller.store.load()["phase"] == "unit"
    assert controller.store.load()["status"] == "passed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events][-4:] == [
        "transition", "phase_attempt", "phase_attempt", "transition",
    ]


def test_phase_failure_marks_halted_without_extra_attempts(tmp_path):
    from scripts.vast_program import PhaseFailed, VastProgramController

    calls = []

    def runner(command):
        calls.append(command)
        return 7

    controller = VastProgramController(state_path=tmp_path / "state.json", now=lambda: NOW, runner=runner)
    controller.initialize(base_sha="abc123")


    with pytest.raises(PhaseFailed):
        controller.run_phase("gate", ["false"], max_attempts=2)

    assert len(calls) == 2
    state = controller.store.load()
    assert state["phase"] == "gate"
    assert state["status"] == "halted"
    assert state["details"]["returncodes"] == [7, 7]


@pytest.mark.parametrize("terminal", ["completed", "halted"])
def test_terminal_program_refuses_new_phase(tmp_path, terminal):
    from scripts.vast_program import TerminalStateError, VastProgramController

    controller = VastProgramController(state_path=tmp_path / "state.json", now=lambda: NOW)
    controller.initialize(base_sha="abc123")
    controller.store.transition(phase="final", status=terminal, now=NOW)

    with pytest.raises(TerminalStateError, match=terminal):
        controller.run_phase("again", ["true"])


def test_set_base_sha_preserves_started_at_and_records_event(tmp_path):
    from scripts.vast_program import VastProgramController

    controller = VastProgramController(state_path=tmp_path / "state.json", now=lambda: NOW)
    controller.initialize(base_sha="")
    started_at = controller.store.load()["started_at"]

    updated = controller.store.set_base_sha(base_sha="abc123", now=NOW + timedelta(minutes=5))

    assert updated["base_sha"] == "abc123"
    assert updated["started_at"] == started_at
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1] == {
        "kind": "base_sha_updated",
        "at": "2026-08-24T12:05:00Z",
        "base_sha": "abc123",
    }


def test_cli_transition_preserves_base_sha_and_started_at(tmp_path):
    from scripts.vast_program import main

    state = tmp_path / "state.json"

    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0
    original = json.loads(state.read_text())
    assert main([
        "--state", str(state),
        "transition",
        "--phase", "phase2-evaluation",
        "--status", "running",
        "--details-json", '{"reason":"operator-migration"}',
    ]) == 0

    updated = json.loads(state.read_text())
    assert updated["base_sha"] == "abc123"
    assert updated["started_at"] == original["started_at"]
    assert updated["phase"] == "phase2-evaluation"
    assert updated["status"] == "running"
    assert updated["details"] == {"reason": "operator-migration"}


def test_detached_phase_argv_round_trips_every_option(tmp_path):
    """The detached controller must be handed the same phase, verbatim."""

    from scripts.vast_program import detached_phase_argv

    argv = detached_phase_argv(
        state=tmp_path / "state.json",
        lease=tmp_path / "controller.lock",
        base_sha="abc123",
        phase="phase4-probe-sweep",
        estimated_hours=6.0,
        max_attempts=2,
        backoff_sec=30.0,
        command=["python", "scripts/tokenizer_lab.py", "sweep", "--device", "cuda"],
    )

    # The child must not re-detach, or every phase would fork forever.
    assert "--detach" not in argv
    assert argv[1].endswith("vast_program.py")
    assert "--state" in argv and str(tmp_path / "state.json") in argv
    assert argv[argv.index("--phase") + 1] == "phase4-probe-sweep"
    assert argv[argv.index("--estimated-hours") + 1] == "6.0"
    assert argv[argv.index("--max-attempts") + 1] == "2"
    # `--` keeps a phase command's own flags out of the controller's parser.
    assert argv[argv.index("--") + 1:] == [
        "python", "scripts/tokenizer_lab.py", "sweep", "--device", "cuda",
    ]


def test_detached_phase_survives_a_group_kill_of_its_launching_session(tmp_path):
    """The regression that cost phase 4 its second arm.

    An engineering session runs in its own process group so the keeper can reap
    the whole tree on timeout. A phase launched in-session inherits that group,
    so ending the session killed the running trainer. The detached phase must
    outlive exactly that kill.
    """

    import signal
    import subprocess
    import sys
    import time

    repo = Path(__file__).resolve().parents[1]
    marker = tmp_path / "phase-finished"
    pidfile = tmp_path / "detached.pid"

    # The phase itself: slow enough that the kill lands while it is running.
    phase_script = tmp_path / "phase.py"
    phase_script.write_text(
        "import pathlib, time\n"
        "time.sleep(3)\n"
        f"pathlib.Path({str(marker)!r}).write_text('ok')\n"
    )

    # Stands in for the engineering session: detaches a phase, then blocks.
    session = tmp_path / "session.py"
    session.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        "from scripts.vast_program import detach_phase\n"
        "started = detach_phase(\n"
        f"    state={str(tmp_path / 'state.json')!r},\n"
        "    phase='phase-detach-drill',\n"
        f"    command=[sys.executable, {str(phase_script)!r}],\n"
        f"    log_path={str(tmp_path / 'phase.log')!r},\n"
        ")\n"
        f"open({str(pidfile)!r}, 'w').write(str(started['pid']))\n"
        "time.sleep(600)\n"
    )

    launched = subprocess.Popen(
        [sys.executable, str(session)], start_new_session=True, cwd=str(repo)
    )
    try:
        deadline = time.monotonic() + 60
        while not pidfile.exists() and time.monotonic() < deadline:
            assert launched.poll() is None, "session died before detaching"
            time.sleep(0.1)
        assert pidfile.exists(), "phase never detached"

        detached_pid = int(pidfile.read_text())
        session_group = os.getpgid(launched.pid)
        assert os.getpgid(detached_pid) != session_group, "phase shares the session group"

        # Exactly what the keeper does to a session it cuts short.
        os.killpg(session_group, signal.SIGKILL)
        launched.wait(timeout=30)
        assert not marker.exists(), "phase finished before the kill proved anything"

        deadline = time.monotonic() + 60
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert marker.read_text() == "ok", "the detached phase was killed with its session"
    finally:
        if launched.poll() is None:
            launched.kill()
