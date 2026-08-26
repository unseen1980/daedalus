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


def test_a_side_lane_runs_beside_the_phase_that_owns_the_box(tmp_path):
    """The controller has to be able to express what the schedule already asks
    for: a CPU pass beside a GPU run.

    Phase 6's evidence pass is ~400 stock llama-cli invocations per arm and
    nothing else; stage B is ten hours of GPU. Serialising them spends a third
    of a day of an idle CPU waiting for a device it never touches. Before
    lanes the second job had two options, and both were wrong: take the lease
    and be refused by the run in progress, or take no lease and vanish from the
    ledger entirely.
    """
    from scripts.vast_program import VastProgramController

    state = tmp_path / "state.json"
    main = VastProgramController(state_path=state, now=lambda: NOW,
                                 runner=lambda command: 0)
    main.initialize(base_sha="abc123")
    main.acquire_lease()
    main.store.transition(phase="phase6-stageb-sweep", status="running", now=NOW)

    side = VastProgramController(state_path=state, lane="evidence",
                                 now=lambda: NOW, runner=lambda command: 0)
    side.acquire_lease()          # its own lease, so the main one cannot refuse it
    side.run_phase("phase6-evidence", ["true"], estimated_hours=3.0)

    assert side.lease_path.name == "controller-evidence.lock"
    assert main.lease_path.exists(), "the side lane must not take the main lease"
    recorded = json.loads(state.read_text())
    assert recorded["phase"] == "phase6-stageb-sweep"
    assert recorded["status"] == "running"
    assert recorded["lanes"]["evidence"]["phase"] == "phase6-evidence"
    assert recorded["lanes"]["evidence"]["status"] == "passed"


def test_one_lane_still_admits_one_controller(tmp_path):
    """Lanes multiply the leases, not the writers per lane."""
    from scripts.vast_program import ControllerLeaseError, VastProgramController

    state = tmp_path / "state.json"
    VastProgramController(state_path=state, now=lambda: NOW).initialize(base_sha="abc")
    first = VastProgramController(state_path=state, lane="evidence", now=lambda: NOW)
    second = VastProgramController(state_path=state, lane="evidence", now=lambda: NOW)

    first.acquire_lease()

    with pytest.raises(ControllerLeaseError, match="active controller"):
        second.acquire_lease()


def test_a_busy_lane_answers_with_what_is_running_not_a_traceback(tmp_path):
    """A session that comes back while a detached ten-hour phase is still
    running asks for the lease every time. A traceback reads as "the controller
    is broken", and the obvious response to a broken controller is to try
    again harder -- which is the one thing that must not happen here."""
    from scripts.vast_program import VastProgramController, main

    state = tmp_path / "state.json"
    lease = tmp_path / "controller.lock"
    holder = VastProgramController(state_path=state, lease_path=lease, now=lambda: NOW)
    holder.initialize(base_sha="abc123")
    holder.acquire_lease()
    holder.store.transition(
        phase="phase6-stageb-sweep", status="running", now=NOW,
        details={"started_at": "2026-08-24T12:00:00Z"})
    # A side lane advancing bumps `updated_at`, so the phase's own start has to
    # be read from where it was recorded, not from the state's last change.
    holder.store.transition(phase="phase6-evidence", status="running",
                            now=NOW + timedelta(hours=3), lane="evidence")

    with pytest.raises(SystemExit) as refusal:
        main(["--state", str(state), "--lease", str(lease), "run-phase",
              "--phase", "phase6-evidence", "--estimated-hours", "0.01",
              "--", "true"])

    message = str(refusal.value)
    assert "phase6-stageb-sweep" in message
    assert "since 2026-08-24T12:00:00Z" in message
    assert "do not relaunch it" in message
    assert "--lane" in message, "the message must name the way out"
    # The refused phase left no mark: the running one still owns the record.
    assert json.loads(state.read_text())["phase"] == "phase6-stageb-sweep"


def test_a_side_lane_refused_by_the_deadline_leaves_the_phase_alone(tmp_path):
    """The refusal is the lane's, and so is the record of it.

    Writing `finalization` to the top-level pair would have the progress branch
    announce finalization on behalf of a pass that never started, hours before
    the main lane's run is due to end.
    """
    from scripts.vast_program import DeadlineRefused, VastProgramController

    state = tmp_path / "state.json"
    main = VastProgramController(state_path=state, now=lambda: NOW)
    main.initialize(base_sha="abc123", started_at=NOW - timedelta(hours=137))
    main.store.transition(phase="phase8-code-1b", status="running", now=NOW)

    side = VastProgramController(state_path=state, lane="evidence", now=lambda: NOW)
    with pytest.raises(DeadlineRefused):
        side.run_phase("phase6-evidence", ["true"], estimated_hours=3.0)

    recorded = json.loads(state.read_text())
    assert recorded["phase"] == "phase8-code-1b"
    assert recorded["status"] == "running"
    assert recorded["lanes"]["evidence"]["phase"] == "finalization"


def test_a_lane_still_stops_when_the_program_is_terminal(tmp_path):
    """PROGRAM_HALTED halts everything, including whatever runs beside it."""
    from scripts.vast_program import TerminalStateError, VastProgramController

    state = tmp_path / "state.json"
    main = VastProgramController(state_path=state, now=lambda: NOW)
    main.initialize(base_sha="abc123")
    main.store.transition(phase="blocker", status="halted", now=NOW)

    side = VastProgramController(state_path=state, lane="evidence", now=lambda: NOW)
    with pytest.raises(TerminalStateError, match="halted"):
        side.run_phase("phase6-evidence", ["true"])


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
    # The main lane is the default, so it is not spelled out.
    assert "--lane" not in argv
    assert argv[1].endswith("vast_program.py")
    assert "--state" in argv and str(tmp_path / "state.json") in argv
    assert argv[argv.index("--phase") + 1] == "phase4-probe-sweep"
    assert argv[argv.index("--estimated-hours") + 1] == "6.0"
    assert argv[argv.index("--max-attempts") + 1] == "2"
    # `--` keeps a phase command's own flags out of the controller's parser.
    assert argv[argv.index("--") + 1:] == [
        "python", "scripts/tokenizer_lab.py", "sweep", "--device", "cuda",
    ]


def test_detaching_a_side_lane_carries_the_lane_to_the_child(tmp_path):
    """Dropping `--lane` would detach the CPU pass into the main lane, where it
    would try to take the GPU run's lease and be refused by it -- from a
    detached child whose only trace is a log nobody is reading."""

    from scripts.vast_program import detached_phase_argv

    argv = detached_phase_argv(
        state=tmp_path / "state.json",
        phase="phase6-evidence",
        lane="evidence",
        command=["python", "scripts/architecture_evidence.py", "retrieval"],
    )

    assert argv[argv.index("--lane") + 1] == "evidence"
    assert argv.index("--lane") < argv.index("run-phase"), (
        "--lane is a controller option, not one of the phase command's")


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


# ------------------------------------------------------- supervised phases ---
#
# Detaching a phase makes it outlive the session that started it. It does not
# make it *continue*: the controller's plain runner re-runs the same argv, so a
# retry of a trainer starts at step zero and overwrites the checkpoint the
# previous attempt wrote, and no in-flight marker is left for `boot_resume` or
# the keeper's busy probe to find. Every phase so far avoided that only because
# it went through an orchestrator script that wrapped `run_with_resume` itself.


def _fake_trainer(tmp_path, *, argv_log, checkpoint, exit_codes, extra=""):
    """A stand-in trainer that records the argv of each attempt.

    Real enough for what the launcher must get right: it writes its checkpoint
    *before* it fails, which is the whole precondition for continuing one.
    """

    script = tmp_path / "fake_trainer.py"
    script.write_text(
        "import pathlib, sys\n"
        f"log = pathlib.Path({str(argv_log)!r})\n"
        "attempts = log.read_text().splitlines() if log.exists() else []\n"
        "log.write_text('\\n'.join(attempts + [' '.join(sys.argv[1:])]) + '\\n')\n"
        f"pathlib.Path({str(checkpoint)!r}).write_text('weights')\n"
        + extra
        + f"codes = {list(exit_codes)!r}\n"
        "sys.exit(codes[min(len(attempts), len(codes) - 1)])\n"
    )
    return script


def test_a_supervised_phase_resumes_its_own_checkpoint_on_retry(tmp_path):
    """The retry that costs a run: same argv, from step zero, over the weights.

    `--max-attempts 2` on a bare trainer relaunches it with no `--resume`, so
    attempt two starts at step zero and overwrites the checkpoint attempt one
    left. Under `--supervise-checkpoint` the retry continues it instead.
    """

    import sys

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    checkpoint = tmp_path / "runs" / "phase8-code-probe" / "checkpoint.pt"
    argv_log = tmp_path / "argv.log"
    trainer = _fake_trainer(tmp_path, argv_log=argv_log, checkpoint=checkpoint,
                            exit_codes=[1, 0])

    assert main([
        "--state", str(state),
        "--lease", str(tmp_path / "controller.lock"),
        "run-phase",
        "--phase", "phase8-code-probe",
        "--max-attempts", "2",
        "--backoff-sec", "0",
        "--supervise-checkpoint", str(checkpoint),
        "--", sys.executable, str(trainer),
    ]) == 0

    attempts = argv_log.read_text().splitlines()
    assert attempts == ["", f"--resume {checkpoint}"], (
        "the retry restarted the trainer instead of resuming it")
    assert json.loads(state.read_text())["status"] == "passed"


def test_a_supervised_phase_is_visible_to_the_keeper_while_it_runs(tmp_path):
    """No marker means the box reads as free while it is training.

    `supervised_job_probe` is how the keeper knows not to launch a session onto
    a busy GPU, and `boot_resume` is how a run survives a reboot. Both find work
    by its in-flight marker, and the marker is written by the supervised
    launcher -- not by `train.py` -- so a phase run as a bare command is
    invisible to both of them for its entire life.
    """

    import sys

    from scripts.vast_program import main

    repo = Path(__file__).resolve().parents[1]
    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    runs_root = tmp_path / "runs"
    checkpoint = runs_root / "phase8-code-1b" / "checkpoint.pt"
    seen = tmp_path / "probe.txt"
    trainer = _fake_trainer(
        tmp_path, argv_log=tmp_path / "argv.log", checkpoint=checkpoint,
        exit_codes=[0],
        extra=(f"sys.path.insert(0, {str(repo)!r})\n"
               "from daedalus.session_keeper import supervised_job_probe\n"
               f"pathlib.Path({str(seen)!r}).write_text("
               f"str(supervised_job_probe({str(runs_root)!r})()))\n"),
    )

    assert main([
        "--state", str(state),
        "--lease", str(tmp_path / "controller.lock"),
        "run-phase",
        "--phase", "phase8-code-1b",
        "--supervise-checkpoint", str(checkpoint),
        "--", sys.executable, str(trainer),
    ]) == 0

    assert seen.read_text() == "True", "the keeper read the box as free mid-run"
    marker = json.loads((checkpoint.parent / "inflight.json").read_text())
    assert marker["completed"] is True, "a finished run must not be resumed"
    assert marker["cmd"] == [sys.executable, str(trainer)]


def test_a_supervised_phase_continues_a_run_its_launcher_died_under(tmp_path):
    """The failure that cost phase 4 an arm, reached through the launcher.

    A session ending kills the trainer it started, leaving an open marker beside
    a checkpoint and no process to continue it. The relaunch is attempt one of a
    fresh supervisor, so `attempt > 1` is false and only the marker knows this
    run has already started.
    """

    import sys

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    run_dir = tmp_path / "runs" / "phase8-code-probe"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_text("60.3M tokens of training")
    argv_log = tmp_path / "argv.log"
    trainer = _fake_trainer(tmp_path, argv_log=argv_log, checkpoint=checkpoint,
                            exit_codes=[0])
    (run_dir / "inflight.json").write_text(json.dumps({
        "schema": 1,
        "run_dir": str(run_dir),
        "cmd": [sys.executable, str(trainer)],
        "ckpt_path": str(checkpoint),
        "completed": False,
        "outcome": None,
        # A pid that cannot be alive: the launcher is provably gone, which is
        # the bar for taking over its checkpoint.
        "supervisor_pid": 2 ** 30,
        "supervisor_start_ticks": 1,
    }))

    assert main([
        "--state", str(state),
        "--lease", str(tmp_path / "controller.lock"),
        "run-phase",
        "--phase", "phase8-code-probe",
        "--supervise-checkpoint", str(checkpoint),
        "--", sys.executable, str(trainer),
    ]) == 0

    assert argv_log.read_text().splitlines() == [f"--resume {checkpoint}"], (
        "attempt one of the relaunch ignored the checkpoint beside it")


def test_a_watchdog_halt_stops_a_supervised_phase_rather_than_resuming_it(tmp_path):
    """Retrying is the right answer to a crash and the wrong one to divergence.

    Without the halt marker the launcher reads the watchdog's SIGTERM as an
    ordinary crash and resumes the diverged checkpoint -- with no watchdog left
    running -- for the rest of the budget.
    """

    import sys

    from scripts.vast_program import PhaseFailed, main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    run_dir = tmp_path / "runs" / "phase8-code-1b"
    checkpoint = run_dir / "checkpoint.pt"
    argv_log = tmp_path / "argv.log"
    halt = json.dumps({"kind": "divergence", "reason": "loss went to nan"})
    trainer = _fake_trainer(
        tmp_path, argv_log=argv_log, checkpoint=checkpoint, exit_codes=[1],
        extra=f"pathlib.Path({str(run_dir / 'HALTED')!r}).write_text({halt!r})\n",
    )

    with pytest.raises(PhaseFailed):
        main([
            "--state", str(state),
            "--lease", str(tmp_path / "controller.lock"),
            "run-phase",
            "--phase", "phase8-code-1b",
            "--max-attempts", "3",
            "--backoff-sec", "0",
            "--supervise-checkpoint", str(checkpoint),
            "--", sys.executable, str(trainer),
        ])

    assert len(argv_log.read_text().splitlines()) == 1, "resumed a halted run"
    assert json.loads(state.read_text())["status"] == "failed"


def test_a_supervised_phase_records_what_the_supervisor_did(tmp_path):
    """The controller's own `attempts` counts *its* attempts, which under
    supervision is always one. Without the supervisor's report the timeline says
    a run that crashed twice and resumed twice went through first time."""

    import sys

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    checkpoint = tmp_path / "runs" / "phase8-code-probe" / "checkpoint.pt"
    trainer = _fake_trainer(tmp_path, argv_log=tmp_path / "argv.log",
                            checkpoint=checkpoint, exit_codes=[1, 0])

    assert main([
        "--state", str(state),
        "--lease", str(tmp_path / "controller.lock"),
        "run-phase",
        "--phase", "phase8-code-probe",
        "--max-attempts", "2",
        "--backoff-sec", "0",
        "--supervise-checkpoint", str(checkpoint),
        "--", sys.executable, str(trainer),
    ]) == 0

    events = [json.loads(line)
              for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    supervised = [event for event in events if event["kind"] == "supervised_run"]
    assert len(supervised) == 1
    assert supervised[0]["details"]["attempts"] == 2
    assert supervised[0]["details"]["resumed"] is True
    assert supervised[0]["details"]["returncodes"] == [1, 0]
    assert supervised[0]["details"]["checkpoint"] == str(checkpoint)


def test_a_supervised_phase_command_may_not_carry_its_own_resume(tmp_path):
    """`--resume` on attempt one restores the *finished* run's step and token
    count, so the phase trains nothing, writes no metrics row and exits 0. The
    supervisor adds `--resume` when it is the right answer; an explicit one is
    a phase that silently does nothing."""

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    with pytest.raises(SystemExit, match="--resume"):
        main([
            "--state", str(state),
            "run-phase",
            "--phase", "phase8-code-probe",
            "--supervise-checkpoint", "runs/phase8-code-probe/checkpoint.pt",
            "--", "python", "train.py",
            "--resume", "runs/hero/checkpoint.pt",
        ])


def test_the_supervised_checkpoint_is_read_out_of_the_trainer_command(tmp_path):
    """Composed by `train.py`'s own resolver, so the two sides cannot drift."""

    from scripts.vast_program import trainer_checkpoint_for

    assert trainer_checkpoint_for(
        ["python", "train.py", "--run-name", "phase8-code-probe"]
    ) == os.path.join("runs", "phase8-code-probe", "checkpoint.pt")
    assert trainer_checkpoint_for(
        ["python", "train.py", "--run-name=phase8-code-probe",
         "--run-dir=/data/phase8"]
    ) == os.path.join("/data/phase8", "checkpoint.pt")
    # Not a trainer, so there is nothing to check and nothing to refuse: a
    # sweep script or an evaluation pass names its own runs.
    assert trainer_checkpoint_for(
        ["python", "scripts/conv_health.py", "sweep"]) is None
    assert trainer_checkpoint_for(["python", "train.py"]) is None


def test_a_supervised_phase_is_refused_a_checkpoint_its_trainer_never_writes(tmp_path):
    """The failure is silent, which is why it is worth a refusal.

    A path `train.py` does not write leaves the in-flight marker beside a file
    that never appears: `resumed` stays False forever, every relaunch starts at
    step zero, and the run looks healthy throughout. The phase 5 smoke found it
    by composing the path instead of asking, and a hand-typed launch flag is a
    better chance to make the same mistake than an orchestrator ever was.
    """

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0

    with pytest.raises(SystemExit, match="restart from step zero"):
        main([
            "--state", str(state),
            "run-phase",
            "--phase", "phase8-code-probe",
            "--supervise-checkpoint", "runs/phase8-code-probe/rolling.pt",
            "--", "python", "train.py",
            "--run-name", "phase8-code-probe", "--data-dir", "data/code",
        ])

    # Refused before anything was claimed: the state and the lease are untouched.
    assert json.loads(state.read_text())["phase"] == "bootstrap"
    assert not (tmp_path / "controller.lock").exists()


def test_an_unsupervised_phase_leaves_no_marker_behind(tmp_path):
    """Most phases are scoring passes and report generators, not resumable runs.
    Marking one in flight would offer `boot_resume` a run to continue that has
    no checkpoint and no meaning."""

    from scripts.vast_program import main

    state = tmp_path / "state.json"
    assert main(["--state", str(state), "--base-sha", "abc123", "init"]) == 0
    assert main([
        "--state", str(state),
        "--lease", str(tmp_path / "controller.lock"),
        "run-phase",
        "--phase", "phase8-scorecard",
        "--", "python", "-c", "pass",
    ]) == 0

    assert not list(tmp_path.rglob("inflight.json"))
    assert json.loads(state.read_text())["status"] == "passed"


def test_detached_phase_argv_carries_supervision_to_the_child(tmp_path):
    """The detached child is the process that actually runs the command.

    Dropping the supervision options here would leave the parent's `--detach`
    working and its `--supervise-checkpoint` silently inert -- the exact
    combination a long training phase is launched with.
    """

    from scripts.vast_program import detached_phase_argv

    argv = detached_phase_argv(
        state=tmp_path / "state.json",
        phase="phase8-code-1b",
        command=["python", "train.py", "--run-name", "phase8-code-1b"],
        supervise_checkpoint="runs/phase8-code-1b/checkpoint.pt",
        watchdog_tokens=1_000_000_000,
        stall_min=30.0,
        max_attempts=3,
        backoff_sec=60.0,
    )

    assert argv[argv.index("--supervise-checkpoint") + 1] == \
        "runs/phase8-code-1b/checkpoint.pt"
    assert argv[argv.index("--watchdog-tokens") + 1] == "1000000000"
    assert argv[argv.index("--stall-min") + 1] == "30.0"
    assert argv[argv.index("--backoff-sec") + 1] == "60.0"
    assert argv.index("--supervise-checkpoint") > argv.index("run-phase"), (
        "the supervision options belong to run-phase, not to the controller")
    assert argv[argv.index("--") + 1:] == [
        "python", "train.py", "--run-name", "phase8-code-1b",
    ]
