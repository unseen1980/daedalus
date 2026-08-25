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


def test_phase_failure_marks_failed_without_extra_attempts(tmp_path):
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
    assert state["status"] == "failed"
    assert state["details"]["returncodes"] == [7, 7]


def test_a_failed_phase_does_not_wedge_the_whole_program(tmp_path):
    """A non-zero exit is a failed phase, not PROGRAM_HALTED.

    Marking it `halted` -- which is terminal -- meant one malformed argv left
    every later phase, related or not, raising TerminalStateError until the
    state was edited by hand. The plan's rule is the opposite: preserve the
    failed phase's artifacts and continue unrelated safe phases.
    """
    from scripts.vast_program import PhaseFailed, VastProgramController

    returncodes = iter([3, 0])
    controller = VastProgramController(state_path=tmp_path / "state.json",
                                       now=lambda: NOW,
                                       runner=lambda command: next(returncodes))
    controller.initialize(base_sha="abc123")

    with pytest.raises(PhaseFailed):
        controller.run_phase("typo", ["bad-argv"])

    controller.run_phase("unrelated", ["true"])

    assert controller.store.load()["status"] == "passed"


def test_a_deliberate_halt_is_still_terminal(tmp_path):
    """PROGRAM_HALTED must keep stopping everything; only the automatic
    per-phase failure was reclassified."""
    from scripts.vast_program import TerminalStateError, VastProgramController

    controller = VastProgramController(state_path=tmp_path / "state.json", now=lambda: NOW)
    controller.initialize(base_sha="abc123")
    controller.store.transition(phase="blocker", status="halted", now=NOW)

    with pytest.raises(TerminalStateError):
        controller.run_phase("anything", ["true"])


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


def test_a_note_records_a_decision_while_a_phase_owns_the_lease(tmp_path):
    """The lease's subject is the state snapshot and the transitions that
    rewrite it. A note writes neither, and making it wait would silence the
    record for exactly as long as a phase runs -- phase 5's escalation held the
    lease for 8.6 hours, which is when the work beside it was done.
    """
    from scripts.vast_program import VastProgramController, main

    state = tmp_path / "state.json"
    lease = tmp_path / "controller.lock"
    holder = VastProgramController(state_path=state, lease_path=lease,
                                   now=lambda: NOW)
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0
    holder.acquire_lease()

    assert main(["--state", str(state), "--lease", str(lease), "note",
                 "--phase", "phase6-stagea-scoring",
                 "--details-json", '{"decision": "read, not retyped"}']) == 0

    events = [json.loads(line) for line in
              (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "decision"
    assert events[-1]["phase"] == "phase6-stagea-scoring"
    assert events[-1]["details"] == {"decision": "read, not retyped"}
    # The lease is untouched: the note neither took it nor released someone
    # else's, so the phase that owns it is unaffected.
    assert json.loads(lease.read_text())["pid"] == os.getpid()


def test_a_note_leaves_the_phase_and_status_alone(tmp_path):
    """A note is a record, not a transition. One that moved `phase` would let a
    comment reassign what the program believes it is doing."""
    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0
    before = json.loads(state.read_text())

    assert main(["--state", str(state), "note", "--phase", "somewhere-else",
                 "--kind", "limitation", "--details-json", '{"note": "x"}']) == 0

    assert json.loads(state.read_text()) == before


def test_a_halted_program_can_still_be_annotated(tmp_path):
    """Why a run halted is the record most worth keeping, and it can only be
    written after the halt."""
    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0
    assert main(["--state", str(state), "transition", "--phase", "p",
                 "--status", "halted"]) == 0

    assert main(["--state", str(state), "note", "--phase", "p",
                 "--details-json", '{"why": "credentials"}']) == 0

    events = [json.loads(line) for line in
              (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["details"] == {"why": "credentials"}


def test_a_malformed_note_is_refused_rather_than_recorded_empty(tmp_path):
    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    for bad in ("{not json", '"a string"'):
        with pytest.raises(SystemExit):
            main(["--state", str(state), "note", "--phase", "p",
                  "--details-json", bad])

    kinds = [json.loads(line)["kind"] for line in
             (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "decision" not in kinds, "a refused note must leave no record"


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


def test_long_phase_refuses_to_run_inside_the_calling_session(tmp_path, monkeypatch):
    """A forgotten --detach must fail loudly, not silently lose GPU hours."""

    from scripts import vast_program
    from scripts.vast_program import main

    # Pinned rather than inherited: whether the *test runner* happens to lead
    # its own session is an accident of how pytest was invoked, and this test
    # is about the caller that does not.
    monkeypatch.setattr(vast_program, "running_in_own_session", lambda: False)

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    with pytest.raises(SystemExit, match="--detach"):
        main([
            "--state", str(state),
            "run-phase",
            "--phase", "phase4-probe-sweep",
            "--estimated-hours", "6.0",
            "--", "python", "scripts/tokenizer_lab.py", "sweep",
        ])

    # The phase never started, so the state is untouched and the lease is free.
    assert json.loads(state.read_text())["phase"] == "bootstrap"
    assert not (tmp_path / "controller.lock").exists()


def test_short_phase_still_runs_inline(tmp_path):
    """The guard bounds long phases only; a quick command keeps its output."""

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0
    assert main([
        "--state", str(state),
        "run-phase",
        "--phase", "phase4-budget",
        "--estimated-hours", "0.01",
        "--", "python", "-c", "pass",
    ]) == 0

    recorded = json.loads(state.read_text())
    assert recorded["phase"] == "phase4-budget"
    assert recorded["status"] == "passed"


def test_a_session_leader_may_run_a_long_phase_inline(tmp_path, monkeypatch):
    """The detached controller's own case, unit-sized.

    It leads its own session, so the phase it runs is already outside the
    engineering session's process group -- there is nothing left for `--detach`
    to buy, and the guard has nothing to refuse.
    """

    from scripts import vast_program
    from scripts.vast_program import main

    monkeypatch.setattr(vast_program, "running_in_own_session", lambda: True)

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0
    assert main([
        "--state", str(state),
        "--lease", str(tmp_path / "controller.lock"),
        "run-phase",
        "--phase", "phase4-gguf-check",
        "--estimated-hours", "6.0",
        "--", "python", "-c", "pass",
    ]) == 0

    recorded = json.loads(state.read_text())
    assert recorded["phase"] == "phase4-gguf-check"
    assert recorded["status"] == "passed"


def test_detaching_a_long_phase_actually_runs_it(tmp_path):
    """The guard must not refuse the child it just detached.

    `detached_phase_argv` deliberately drops `--detach` so the child cannot fork
    again -- which handed the child an invocation the guard reads as "a long
    phase running inside its calling session" and refuses. Every phase at or
    past the bound was therefore un-startable: the parent detached, the child
    exited 1 into its log, and the state never left the previous phase. Found
    launching phase 4's gguf-check, whose whole 0.25h estimate is why it was
    detached in the first place.
    """

    import sys
    import time

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    marker = tmp_path / "phase-ran"
    phase_script = tmp_path / "phase.py"
    phase_script.write_text(f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ok')\n")

    assert main([
        "--state", str(state),
        "--lease", str(tmp_path / "controller.lock"),
        "run-phase",
        "--phase", "phase4-gguf-check",
        "--estimated-hours", "6.0",
        "--detach",
        "--log", str(tmp_path / "phase.log"),
        "--", sys.executable, str(phase_script),
    ]) == 0

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        recorded = json.loads(state.read_text())
        if recorded["phase"] == "phase4-gguf-check" and recorded["status"] != "running":
            break
        time.sleep(0.1)

    recorded = json.loads(state.read_text())
    assert recorded["phase"] == "phase4-gguf-check", (
        f"the detached child never claimed the phase; log: "
        f"{(tmp_path / 'phase.log').read_text()}")
    assert recorded["status"] == "passed", (tmp_path / "phase.log").read_text()
    assert marker.exists(), "the phase command never ran"


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
