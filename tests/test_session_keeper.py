"""Bounded relaunch behaviour for unattended Claude engineering sessions."""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daedalus.program_state import ProgramDeadline, ProgramStateStore
from daedalus.session_keeper import (
    ClaudeSessionLauncher,
    PlanGuard,
    KeeperAction,
    KeeperPolicy,
    KeeperState,
    LaunchOutcome,
    LaunchRequest,
    SessionKeeper,
    decide,
    failure_context_for,
    remaining_backoff,
    supervised_job_probe,
)


START = datetime(2026, 8, 24, 10, 0, tzinfo=timezone.utc)
POLICY = KeeperPolicy()
DEADLINE = ProgramDeadline(started_at=START)


def program_state(**overrides) -> dict:
    state = {
        "schema": 1,
        "started_at": START.isoformat().replace("+00:00", "Z"),
        "updated_at": START.isoformat().replace("+00:00", "Z"),
        "base_sha": "abc123",
        "hard_hours": 144.0,
        "reserve_hours": 8.0,
        "phase": "phase2-evaluation",
        "status": "running",
        "details": {},
    }
    state.update(overrides)
    return state


def decision(
    keeper_state: KeeperState, *, now=None, supervised_job_live=False, **state_overrides
) -> KeeperAction:
    return decide(
        program_state=program_state(**state_overrides),
        keeper_state=keeper_state,
        now=now or START + timedelta(hours=1),
        deadline=DEADLINE,
        policy=POLICY,
        supervised_job_live=supervised_job_live,
    )


class TestDecide:
    def test_starts_a_fresh_session_for_a_new_phase(self):
        action = decision(KeeperState())

        assert action.kind == "launch"
        assert action.resume_session_id is None
        assert (action.generation, action.attempt) == (1, 1)

    def test_resets_to_a_fresh_session_when_the_phase_changes(self):
        state = KeeperState(
            phase="phase1-control-plane",
            session_id="old-session",
            generation=2,
            attempt=3,
            consecutive_failures=3,
        )

        action = decision(state)

        assert action.kind == "launch"
        assert action.reason == "new phase"
        assert action.resume_session_id is None
        assert (action.generation, action.attempt) == (1, 1)

    def test_resumes_the_same_session_for_a_repair_continuation(self):
        state = KeeperState(
            phase="phase2-evaluation",
            session_id="s-1",
            attempt=1,
            consecutive_failures=1,
            last_exit_at=(START - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        )

        action = decision(state)

        assert action.kind == "launch"
        assert action.resume_session_id == "s-1"
        assert action.attempt == 2

    def test_escalates_to_an_independent_session_after_spent_continuations(self):
        state = KeeperState(
            phase="phase2-evaluation",
            session_id="s-1",
            attempt=POLICY.max_resume_attempts,
            consecutive_failures=POLICY.max_resume_attempts,
            last_exit_at=(START - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        )

        action = decision(state)

        assert action.kind == "launch"
        assert action.reason == "independent review session"
        assert action.resume_session_id is None
        assert (action.generation, action.attempt) == (2, 1)

    def test_blocks_after_every_independent_session_fails(self):
        state = KeeperState(
            phase="phase2-evaluation",
            session_id="s-2",
            generation=POLICY.max_generations,
            attempt=POLICY.max_resume_attempts,
            consecutive_failures=POLICY.max_resume_attempts,
            last_exit_at=(START - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        )

        action = decision(state)

        assert action.kind == "block"
        assert "repair continuations failed" in action.reason

    def test_a_recorded_blocker_is_never_relaunched_through(self):
        state = KeeperState(phase="phase2-evaluation", blocked_reason="hard blocker")

        action = decision(state)

        assert action == KeeperAction(
            kind="block", reason="hard blocker", generation=1, attempt=0
        )

    @pytest.mark.parametrize("status", ["complete", "completed"])
    def test_a_finished_program_stops_supervision(self, status):
        action = decision(KeeperState(), status=status)

        assert action.kind == "stop"
        assert status in action.reason

    @pytest.mark.parametrize("status", ["halted", "blocked", "failed"])
    def test_a_recorded_blocker_stops_supervision(self, status):
        action = decision(
            KeeperState(), status=status, details={"blocker": "needs an operator"}
        )

        assert action.kind == "stop"
        assert "blocker" in action.reason

    @pytest.mark.parametrize("status", ["halted", "failed"])
    def test_a_failed_phase_command_keeps_supervision_running(self, status):
        # `run-phase` writes `halted` when one command exits non-zero. That is
        # the repair contract's cue to engage, not to go offline.
        action = decision(
            KeeperState(), status=status, details={"attempts": 1, "returncodes": [1]}
        )

        assert action.kind == "launch"

    def test_no_experiment_session_starts_inside_the_finalization_window(self):
        now = START + timedelta(hours=137)

        action = decision(KeeperState(), now=now)

        assert action.kind != "launch"

    def test_no_engineering_session_starts_after_the_hard_deadline(self):
        now = START + timedelta(hours=145)

        action = decision(KeeperState(), now=now)

        assert action.kind == "stop"
        assert action.reason == "deadline stage expired"

    def test_waits_while_no_phase_is_active(self):
        action = decision(KeeperState(), phase="not_started")

        assert action.kind == "wait"
        assert action.wait_seconds == POLICY.backoff_base_sec

    def test_waits_out_the_exponential_backoff_before_relaunching(self):
        exited = START + timedelta(hours=1)
        state = KeeperState(
            phase="phase2-evaluation",
            session_id="s-1",
            attempt=1,
            consecutive_failures=2,
            last_exit_at=exited.isoformat().replace("+00:00", "Z"),
        )

        action = decision(state, now=exited + timedelta(seconds=10))

        assert action.kind == "wait"
        assert action.wait_seconds == pytest.approx(POLICY.backoff_for(2) - 10)


class TestBackoff:
    def test_backoff_grows_exponentially_and_stays_finite(self):
        policy = KeeperPolicy(backoff_base_sec=30.0, backoff_cap_sec=600.0)

        assert policy.backoff_for(0) == 0.0
        assert policy.backoff_for(1) == 30.0
        assert policy.backoff_for(2) == 60.0
        assert policy.backoff_for(3) == 120.0
        assert policy.backoff_for(20) == 600.0

    def test_a_first_attempt_owes_no_backoff(self):
        assert remaining_backoff(KeeperState(), START, POLICY) == 0.0


class TestKeeperState:
    def test_round_trips_through_a_dictionary(self):
        state = KeeperState(phase="phase2-evaluation", session_id="s-1", launches=4)

        assert KeeperState.from_dict(state.to_dict()) == state

    def test_ignores_unknown_persisted_fields(self):
        payload = KeeperState(phase="p").to_dict()
        payload["field_from_a_future_schema"] = True

        assert KeeperState.from_dict(payload).phase == "p"


class TestLauncherArgv:
    def _launcher(self, tmp_path: Path, **kwargs) -> ClaudeSessionLauncher:
        prompt = tmp_path / "phase2.md"
        prompt.write_text("do the phase 2 slice")
        return ClaudeSessionLauncher(repo=tmp_path, prompt_path=prompt, **kwargs)

    def test_runs_non_interactively_on_opus_at_xhigh_effort(self, tmp_path):
        argv = list(
            self._launcher(tmp_path).build_argv(
                LaunchRequest(phase="phase2-evaluation", generation=1, attempt=1)
            )
        )

        assert argv[:2] == ["claude", "-p"]
        assert argv[1 + argv.index("-p")] == "do the phase 2 slice"
        assert argv[argv.index("--model") + 1] == "opus"
        assert argv[argv.index("--effort") + 1] == "xhigh"
        assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
        assert "--resume" not in argv

    def test_a_continuation_resumes_the_recorded_session(self, tmp_path):
        argv = list(
            self._launcher(tmp_path).build_argv(
                LaunchRequest(
                    phase="phase2-evaluation",
                    generation=1,
                    attempt=2,
                    resume_session_id="s-1",
                )
            )
        )

        assert argv[argv.index("--resume") + 1] == "s-1"

    def test_the_versioned_plan_is_appended_as_a_system_prompt(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("plan")

        argv = list(
            self._launcher(tmp_path, system_prompt_path=plan).build_argv(
                LaunchRequest(phase="phase2-evaluation", generation=1, attempt=1)
            )
        )

        assert argv[argv.index("--append-system-prompt-file") + 1] == str(plan)

    def test_a_continuation_carries_the_explicit_failure_context(self, tmp_path):
        prompt = self._launcher(tmp_path).build_prompt(
            LaunchRequest(
                phase="phase2-evaluation",
                generation=1,
                attempt=2,
                resume_session_id="s-1",
                failure_context="exit=1 pytest collection error",
            )
        )

        assert "do the phase 2 slice" in prompt
        assert "pytest collection error" in prompt
        assert "Attempt 2 of generation 1" in prompt


class TestLauncherExecution:
    def _launcher(self, tmp_path: Path, runner) -> ClaudeSessionLauncher:
        prompt = tmp_path / "phase2.md"
        prompt.write_text("slice")
        return ClaudeSessionLauncher(
            repo=tmp_path, prompt_path=prompt, runner=runner, timeout_sec=5.0
        )

    def test_reports_the_session_id_claude_printed(self, tmp_path):
        def runner(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps({"session_id": "s-9", "is_error": False}), stderr=""
            )

        outcome = self._launcher(tmp_path, runner).launch(
            LaunchRequest(phase="p", generation=1, attempt=1)
        )

        assert outcome == LaunchOutcome(exit_code=0, session_id="s-9", tail="")

    def test_keeps_the_resumed_session_id_when_output_is_unparsable(self, tmp_path):
        def runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="not json", stderr="boom")

        outcome = self._launcher(tmp_path, runner).launch(
            LaunchRequest(phase="p", generation=1, attempt=2, resume_session_id="s-1")
        )

        assert (outcome.exit_code, outcome.session_id, outcome.tail) == (1, "s-1", "boom")

    def test_a_hung_session_times_out_instead_of_blocking_the_program(self, tmp_path):
        def runner(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 5.0)

        outcome = self._launcher(tmp_path, runner).launch(
            LaunchRequest(phase="p", generation=1, attempt=1)
        )

        assert outcome.exit_code == 124
        assert "timeout" in outcome.tail

    def test_a_missing_claude_binary_is_reported_not_raised(self, tmp_path):
        def runner(argv, **kwargs):
            raise OSError("No such file or directory")

        outcome = self._launcher(tmp_path, runner).launch(
            LaunchRequest(phase="p", generation=1, attempt=1)
        )

        assert outcome.exit_code == 127
        assert "could not launch Claude" in outcome.tail


class RecordingLauncher:
    """Launcher stub that returns queued outcomes and records every request."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def launch(self, request: LaunchRequest) -> LaunchOutcome:
        self.requests.append(request)
        return self.outcomes.pop(0) if self.outcomes else LaunchOutcome(exit_code=0)


@pytest.fixture()
def keeper_environment(tmp_path):
    store = ProgramStateStore(tmp_path / "state.json")
    store.initialize(started_at=START, base_sha="abc123")
    store.transition(phase="phase2-evaluation", status="running", now=START)
    return store, tmp_path / "keeper.json"


def build_keeper(store, keeper_path, launcher, *, progress, now=None, policy=POLICY):
    return SessionKeeper(
        store=store,
        keeper_state_path=keeper_path,
        launcher=launcher,
        policy=policy,
        progress_probe=lambda: progress[0],
        clock=lambda: now or START + timedelta(hours=1),
        sleeper=lambda seconds: None,
    )


class TestSessionKeeperStep:
    def test_a_progressing_session_clears_the_retry_budget(self, keeper_environment):
        store, keeper_path = keeper_environment
        progress = ["sha-1"]
        launcher = RecordingLauncher([LaunchOutcome(exit_code=0, session_id="s-1")])

        def advance(request):
            progress[0] = "sha-2"
            return LaunchOutcome(exit_code=0, session_id="s-1")

        launcher.launch = advance
        keeper = build_keeper(store, keeper_path, launcher, progress=progress)

        action = keeper.step()
        state = keeper.load_state()

        assert action.kind == "launch"
        assert state.consecutive_failures == 0
        assert state.session_id is None
        assert state.progress_fingerprint == "sha-2"

    def test_a_clean_exit_without_progress_counts_as_a_failure(self, keeper_environment):
        store, keeper_path = keeper_environment
        progress = ["sha-1"]
        launcher = RecordingLauncher([LaunchOutcome(exit_code=0, session_id="s-1")])
        keeper = build_keeper(store, keeper_path, launcher, progress=progress)

        keeper.step()
        state = keeper.load_state()

        assert state.consecutive_failures == 1
        assert state.session_id == "s-1"
        assert "progressed=False" in state.last_failure

    def test_a_failing_session_is_resumed_with_its_own_identifier(self, keeper_environment):
        store, keeper_path = keeper_environment
        progress = ["sha-1"]
        launcher = RecordingLauncher(
            [
                LaunchOutcome(exit_code=1, session_id="s-1", tail="pytest failed"),
                LaunchOutcome(exit_code=1, session_id="s-1", tail="pytest failed"),
            ]
        )
        keeper = build_keeper(
            store,
            keeper_path,
            launcher,
            progress=progress,
            policy=KeeperPolicy(backoff_base_sec=0.0),
        )

        keeper.step()
        keeper.step()

        assert launcher.requests[0].resume_session_id is None
        assert launcher.requests[1].resume_session_id == "s-1"
        assert "pytest failed" in launcher.requests[1].failure_context

    def test_repeated_failures_end_in_a_recorded_hard_blocker(self, keeper_environment):
        store, keeper_path = keeper_environment
        progress = ["sha-1"]
        policy = KeeperPolicy(
            max_resume_attempts=2, max_generations=2, backoff_base_sec=0.0
        )
        launcher = RecordingLauncher([LaunchOutcome(exit_code=1, session_id="s-1")] * 8)
        keeper = build_keeper(
            store, keeper_path, launcher, progress=progress, policy=policy
        )

        final = keeper.run(max_cycles=10)

        assert final.kind == "block"
        assert store.load()["status"] == "blocked"
        assert store.load()["details"]["blocker"] == final.reason
        assert keeper.load_state().blocked_reason == final.reason

    def test_supervision_stops_cleanly_once_the_program_finishes(self, keeper_environment):
        store, keeper_path = keeper_environment
        store.transition(phase="phase9-final", status="complete", now=START)
        launcher = RecordingLauncher([])
        keeper = build_keeper(store, keeper_path, launcher, progress=["sha-1"])

        action = keeper.run(max_cycles=3)

        assert action.kind == "stop"
        assert launcher.requests == []

    def test_every_launch_and_exit_reaches_the_durable_timeline(self, keeper_environment):
        store, keeper_path = keeper_environment
        launcher = RecordingLauncher([LaunchOutcome(exit_code=1, session_id="s-1")])
        keeper = build_keeper(store, keeper_path, launcher, progress=["sha-1"])

        keeper.step()

        kinds = [
            json.loads(line)["kind"]
            for line in store.events_path.read_text().splitlines()
        ]
        assert kinds[-2:] == ["session_launch", "session_exit"]

    def test_a_probe_failure_does_not_stop_supervision(self, keeper_environment):
        store, keeper_path = keeper_environment

        def broken_probe():
            raise RuntimeError("git missing")

        keeper = SessionKeeper(
            store=store,
            keeper_state_path=keeper_path,
            launcher=RecordingLauncher([LaunchOutcome(exit_code=0, session_id="s-1")]),
            progress_probe=broken_probe,
            clock=lambda: START + timedelta(hours=1),
            sleeper=lambda seconds: None,
        )

        action = keeper.step()

        assert action.kind == "launch"
        assert "probe-error" in keeper.load_state().progress_fingerprint


class TestPromptResolution:
    def _launcher(self, tmp_path: Path) -> ClaudeSessionLauncher:
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "phase2-evaluation.md").write_text("phase two work")
        fallback = tmp_path / "default.md"
        fallback.write_text("standing autonomy contract")
        return ClaudeSessionLauncher(
            repo=tmp_path, prompt_path=fallback, prompt_dir=prompts
        )

    def test_a_phase_with_its_own_prompt_uses_it(self, tmp_path):
        launcher = self._launcher(tmp_path)

        prompt = launcher.build_prompt(
            LaunchRequest(phase="phase2-evaluation", generation=1, attempt=1)
        )

        assert prompt == "phase two work"

    def test_a_phase_without_a_prompt_falls_back_to_the_standing_prompt(self, tmp_path):
        launcher = self._launcher(tmp_path)

        prompt = launcher.build_prompt(
            LaunchRequest(phase="phase7-corpus", generation=1, attempt=1)
        )

        assert prompt == "standing autonomy contract"


class TestCommandLine:
    def _cli(self):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "daedalus_session_keeper_cli", root / "scripts" / "session_keeper.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _program(self, tmp_path: Path) -> Path:
        store = ProgramStateStore(tmp_path / "state.json")
        store.initialize(started_at=datetime.now(timezone.utc), base_sha="abc123")
        store.transition(
            phase="phase2-evaluation",
            status="running",
            now=datetime.now(timezone.utc),
        )
        return tmp_path / "state.json"

    def _argv(self, tmp_path: Path, state: Path, claude_bin: str) -> list:
        return [
            "--repo", str(tmp_path),
            "--state", str(state),
            "--keeper-state", str(tmp_path / "keeper.json"),
            "--default-prompt", str(tmp_path / "default.md"),
            "--claude-bin", claude_bin,
            "--poll-interval-sec", "0",
            "--max-cycles", "1",
        ]

    def test_a_missing_standing_prompt_fails_before_any_launch(self, tmp_path):
        cli = self._cli()
        state = self._program(tmp_path)

        with pytest.raises(SystemExit):
            cli.main(self._argv(tmp_path, state, "claude"))

    def test_one_cycle_launches_the_configured_binary(self, tmp_path):
        cli = self._cli()
        state = self._program(tmp_path)
        (tmp_path / "default.md").write_text("standing prompt")
        marker = tmp_path / "launched.txt"
        fake = tmp_path / "fake-claude"
        fake.write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$@" > {marker}\n'
            'echo \'{"session_id": "s-cli"}\'\n'
        )
        fake.chmod(0o755)

        assert cli.main(self._argv(tmp_path, state, str(fake))) == 0

        recorded = marker.read_text()
        assert "standing prompt" in recorded
        assert "--permission-mode" in recorded and "dontAsk" in recorded
        assert json.loads((tmp_path / "keeper.json").read_text())["launches"] == 1


def write_plan_manifest(tmp_path: Path) -> tuple:
    import hashlib

    versioned = tmp_path / "versioned-plan.md"
    versioned.write_text("reviewable scope")
    execution = tmp_path / "execution-plan.md"
    execution.write_text("operational detail")
    manifest = tmp_path / "plan-hashes.txt"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256(plan.read_bytes()).hexdigest()}  {plan}\n"
            for plan in (versioned, execution)
        )
    )
    return PlanGuard(hashes_path=manifest), versioned, execution


class TestPlanGuard:
    def test_matching_plans_verify(self, tmp_path):
        guard, _, _ = write_plan_manifest(tmp_path)

        assert guard.verify() == (True, "")

    def test_a_silently_changed_plan_is_refused(self, tmp_path):
        guard, versioned, _ = write_plan_manifest(tmp_path)
        versioned.write_text("reviewable scope, quietly edited")

        verified, detail = guard.verify()

        assert verified is False
        assert versioned.name in detail

    def test_a_missing_plan_is_refused(self, tmp_path):
        guard, _, execution = write_plan_manifest(tmp_path)
        execution.unlink()

        verified, detail = guard.verify()

        assert verified is False
        assert "unreadable plan" in detail

    def test_an_unreadable_manifest_is_refused(self, tmp_path):
        guard = PlanGuard(hashes_path=tmp_path / "absent.txt")

        verified, detail = guard.verify()

        assert verified is False
        assert "unreadable plan manifest" in detail

    def test_materializes_both_plans_into_a_root_only_prompt(self, tmp_path):
        guard, _, _ = write_plan_manifest(tmp_path)
        destination = tmp_path / "context" / "plan-context.md"

        written = guard.materialize(destination)
        body = written.read_text()

        assert body.index("reviewable scope") < body.index("operational detail")
        assert oct(written.stat().st_mode)[-3:] == "600"
        assert not (tmp_path / "context" / "plan-context.md.tmp").exists()


class TestSessionKeeperPlanEnforcement:
    def test_a_verified_plan_is_appended_to_the_session(self, keeper_environment, tmp_path):
        store, keeper_path = keeper_environment
        guard, _, _ = write_plan_manifest(tmp_path)
        context = tmp_path / "plan-context.md"
        launcher = RecordingLauncher([LaunchOutcome(exit_code=0, session_id="s-1")])
        keeper = SessionKeeper(
            store=store,
            keeper_state_path=keeper_path,
            launcher=launcher,
            plan_guard=guard,
            plan_context_path=context,
            progress_probe=lambda: "sha-1",
            clock=lambda: START + timedelta(hours=1),
            sleeper=lambda seconds: None,
        )

        action = keeper.step()

        assert action.kind == "launch"
        assert launcher.requests[0].system_prompt_path == context
        assert "reviewable scope" in context.read_text()

    def test_a_changed_plan_blocks_instead_of_launching(self, keeper_environment, tmp_path):
        store, keeper_path = keeper_environment
        guard, versioned, _ = write_plan_manifest(tmp_path)
        versioned.write_text("scope changed without review")
        launcher = RecordingLauncher([LaunchOutcome(exit_code=0)])
        keeper = SessionKeeper(
            store=store,
            keeper_state_path=keeper_path,
            launcher=launcher,
            plan_guard=guard,
            plan_context_path=tmp_path / "plan-context.md",
            progress_probe=lambda: "sha-1",
            clock=lambda: START + timedelta(hours=1),
            sleeper=lambda seconds: None,
        )

        action = keeper.step()

        assert action.kind == "block"
        assert "changed plan" in action.reason
        assert launcher.requests == []
        assert store.load()["status"] == "blocked"


class TestRetryBudgetWithoutASessionId:
    """A turn that times out or never starts reports no session id."""

    def _spent(self, **overrides) -> KeeperState:
        state = KeeperState(
            phase="phase2-evaluation",
            session_id=None,
            attempt=POLICY.max_resume_attempts,
            consecutive_failures=POLICY.max_resume_attempts,
            last_exit_at=(START - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        )
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    def test_a_spent_budget_escalates_even_with_no_recorded_session(self):
        action = decision(self._spent())

        assert action.kind == "launch"
        assert action.reason == "independent review session"
        assert (action.generation, action.attempt) == (2, 1)

    def test_the_last_generation_blocks_rather_than_relaunching_forever(self):
        action = decision(self._spent(generation=POLICY.max_generations))

        assert action.kind == "block"

    def test_repeated_timeouts_end_in_a_blocker(self, keeper_environment):
        store, keeper_path = keeper_environment
        policy = KeeperPolicy(
            max_resume_attempts=2, max_generations=2, backoff_base_sec=0.0
        )
        launcher = RecordingLauncher(
            [LaunchOutcome(exit_code=124, session_id=None, tail="timeout")] * 10
        )
        keeper = build_keeper(
            store, keeper_path, launcher, progress=["sha-1"], policy=policy
        )

        final = keeper.run(max_cycles=12)

        assert final.kind == "block"
        assert len(launcher.requests) == policy.max_resume_attempts * policy.max_generations


class TestSupervisedJobBackpressure:
    def test_no_session_starts_while_a_supervised_job_owns_the_box(self):
        action = decision(KeeperState(), supervised_job_live=True)

        assert action.kind == "wait"
        assert action.reason == "supervised job in flight"
        assert action.wait_seconds == POLICY.busy_poll_sec

    def test_a_busy_box_neither_launches_nor_counts_a_failure(self, keeper_environment):
        store, keeper_path = keeper_environment
        launcher = RecordingLauncher([])
        keeper = SessionKeeper(
            store=store,
            keeper_state_path=keeper_path,
            launcher=launcher,
            busy_probe=lambda: True,
            progress_probe=lambda: "sha-1",
            clock=lambda: START + timedelta(hours=1),
            sleeper=lambda seconds: None,
        )

        action = keeper.step()

        assert action.kind == "wait"
        assert launcher.requests == []
        assert keeper.load_state().consecutive_failures == 0

    def test_a_completed_marker_does_not_hold_the_program_idle(self, tmp_path):
        run_dir = tmp_path / "hero"
        run_dir.mkdir()
        (run_dir / "inflight.json").write_text(
            json.dumps({"schema": 1, "completed": True, "supervisor_pid": 1})
        )

        assert supervised_job_probe(tmp_path)() is False

    def test_an_unknown_marker_does_not_hold_the_program_idle(self, tmp_path):
        run_dir = tmp_path / "hero"
        run_dir.mkdir()
        (run_dir / "inflight.json").write_text(json.dumps({"schema": 1, "cmd": ["x"]}))

        assert supervised_job_probe(tmp_path)() is False

    @pytest.mark.skipif(
        not Path("/proc").is_dir(), reason="process start ticks are read from /proc"
    )
    def test_a_live_supervisor_marks_the_box_busy(self, tmp_path):
        from daedalus.supervise import proc_start_ticks

        run_dir = tmp_path / "hero"
        run_dir.mkdir()
        (run_dir / "inflight.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "cmd": ["x"],
                    "supervisor_pid": os.getpid(),
                    "supervisor_start_ticks": proc_start_ticks(os.getpid()),
                }
            )
        )

        assert supervised_job_probe(tmp_path)() is True


class TestTimeoutKillsTheWholeTree:
    def test_a_timed_out_turn_does_not_leave_its_children_running(self, tmp_path):
        child_marker = tmp_path / "child.pid"
        script = tmp_path / "slow-claude"
        script.write_text(
            "#!/bin/bash\n"
            "sleep 120 &\n"
            f'echo $! > {child_marker}\n'
            "sleep 120\n"
        )
        script.chmod(0o755)
        prompt = tmp_path / "prompt.md"
        prompt.write_text("work")
        launcher = ClaudeSessionLauncher(
            repo=tmp_path,
            prompt_path=prompt,
            claude_bin=str(script),
            timeout_sec=2.0,
        )

        outcome = launcher.launch(LaunchRequest(phase="p", generation=1, attempt=1))

        assert outcome.exit_code == 124
        child_pid = int(child_marker.read_text().strip())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except OSError:
                break
            time.sleep(0.2)
        else:
            raise AssertionError(f"child {child_pid} survived the timeout kill")


class TestFinalizationWindow:
    def test_the_reserve_hands_the_program_to_finalization(self):
        action = decision(KeeperState(), now=START + timedelta(hours=137))

        assert action.kind == "finalize"
        assert action.reason == "reserved finalization window reached"

    def test_finalization_turns_still_run_inside_the_reserve(self):
        state = KeeperState(phase=POLICY.finalization_phase)

        action = decision(
            state,
            now=START + timedelta(hours=137),
            phase=POLICY.finalization_phase,
        )

        assert action.kind == "launch"

    def test_nothing_launches_after_the_hard_deadline(self):
        action = decision(
            KeeperState(phase=POLICY.finalization_phase),
            now=START + timedelta(hours=145),
            phase=POLICY.finalization_phase,
        )

        assert action.kind == "stop"
        assert action.reason == "deadline stage expired"

    def test_the_keeper_records_the_finalization_transition(self, keeper_environment):
        store, keeper_path = keeper_environment
        launcher = RecordingLauncher([])
        keeper = SessionKeeper(
            store=store,
            keeper_state_path=keeper_path,
            launcher=launcher,
            progress_probe=lambda: "sha-1",
            clock=lambda: START + timedelta(hours=137),
            sleeper=lambda seconds: None,
        )

        action = keeper.step()
        state = store.load()

        assert action.kind == "finalize"
        assert state["phase"] == POLICY.finalization_phase
        assert state["status"] == "running"
        assert state["details"]["previous_phase"] == "phase2-evaluation"
        assert launcher.requests == []


class TestAssignedSessionIdentity:
    """A session that dies before printing anything must still be resumable."""

    def test_a_fresh_session_is_launched_under_an_assigned_identifier(self, tmp_path):
        prompt = tmp_path / "prompt.md"
        prompt.write_text("work")
        launcher = ClaudeSessionLauncher(repo=tmp_path, prompt_path=prompt)

        argv = list(
            launcher.build_argv(
                LaunchRequest(
                    phase="p", generation=1, attempt=1, session_id="assigned-1"
                )
            )
        )

        assert argv[argv.index("--session-id") + 1] == "assigned-1"
        assert "--resume" not in argv

    def test_a_continuation_resumes_and_does_not_reassign(self, tmp_path):
        prompt = tmp_path / "prompt.md"
        prompt.write_text("work")
        launcher = ClaudeSessionLauncher(repo=tmp_path, prompt_path=prompt)

        argv = list(
            launcher.build_argv(
                LaunchRequest(
                    phase="p",
                    generation=1,
                    attempt=2,
                    resume_session_id="s-1",
                    session_id="s-1",
                )
            )
        )

        assert argv[argv.index("--resume") + 1] == "s-1"
        assert "--session-id" not in argv

    def test_the_identifier_is_recorded_before_the_session_starts(
        self, keeper_environment
    ):
        store, keeper_path = keeper_environment
        recorded = {}

        class CrashingLauncher:
            def launch(self, request):
                # Whatever happens next, the keeper already knows the id.
                recorded["state"] = json.loads(keeper_path.read_text())
                recorded["request"] = request
                raise RuntimeError("keeper died here")

        keeper = build_keeper(
            store, keeper_path, CrashingLauncher(), progress=["sha-1"]
        )

        with pytest.raises(RuntimeError):
            keeper.step()

        assert recorded["state"]["session_id"] == recorded["request"].session_id
        assert recorded["state"]["session_id"]

    def test_a_killed_session_leaves_a_resumable_identifier(self, keeper_environment):
        store, keeper_path = keeper_environment
        launcher = RecordingLauncher(
            [
                LaunchOutcome(exit_code=-9, session_id=None, tail="killed"),
                LaunchOutcome(exit_code=0, session_id=None),
            ]
        )
        keeper = build_keeper(
            store,
            keeper_path,
            launcher,
            progress=["sha-1"],
            policy=KeeperPolicy(backoff_base_sec=0.0),
        )

        keeper.step()
        assigned = keeper.load_state().session_id
        keeper.step()

        assert assigned is not None
        assert launcher.requests[1].resume_session_id == assigned


class TestStaleFailureContext:
    """A failure report is only useful while it still describes the situation."""

    def _state(self, exited_at: str) -> KeeperState:
        return KeeperState(
            phase="phase3-qat-recovery",
            last_failure="exit=-9 the workspace has not been trusted",
            last_exit_at=exited_at,
        )

    def test_a_current_failure_is_handed_to_the_next_turn(self):
        keeper_state = self._state("2026-08-24T14:36:02Z")

        context = failure_context_for(
            {"updated_at": "2026-08-24T14:30:00Z"}, keeper_state
        )

        assert "not been trusted" in context

    def test_a_failure_the_controller_moved_past_is_dropped(self):
        keeper_state = self._state("2026-08-24T14:36:02Z")

        context = failure_context_for(
            {"updated_at": "2026-08-24T14:39:18Z"}, keeper_state
        )

        assert context == ""

    def test_no_failure_means_no_context(self):
        assert failure_context_for({}, KeeperState()) == ""

    def test_the_keeper_does_not_resend_a_repaired_failure(self, keeper_environment):
        store, keeper_path = keeper_environment
        launcher = RecordingLauncher(
            [
                LaunchOutcome(exit_code=1, session_id="s-1", tail="already repaired"),
                LaunchOutcome(exit_code=0, session_id="s-1"),
            ]
        )
        keeper = build_keeper(
            store,
            keeper_path,
            launcher,
            progress=["sha-1"],
            policy=KeeperPolicy(backoff_base_sec=0.0),
        )

        keeper.step()
        # The operator repairs the cause and transitions the controller.
        store.transition(
            phase="phase2-evaluation",
            status="running",
            now=START + timedelta(hours=2),
            details={"reason": "operator repaired the cause"},
        )
        keeper.step()

        assert launcher.requests[1].failure_context == ""


class TestNestedSupervisedRuns:
    """Experiment drivers group their arms below the runs root."""

    def _marker(self, run_dir: Path, *, live: bool) -> None:
        import os

        from daedalus.supervise import proc_start_ticks

        run_dir.mkdir(parents=True, exist_ok=True)
        marker = {"schema": 1, "cmd": ["train.py"]}
        if live:
            marker["supervisor_pid"] = os.getpid()
            marker["supervisor_start_ticks"] = proc_start_ticks(os.getpid())
        (run_dir / "inflight.json").write_text(json.dumps(marker))

    @pytest.mark.skipif(
        not Path("/proc").is_dir(), reason="process start ticks are read from /proc"
    )
    def test_an_arm_nested_under_its_driver_marks_the_box_busy(self, tmp_path):
        self._marker(tmp_path / "qat-recovery" / "muon-5e-4", live=True)

        assert supervised_job_probe(tmp_path)() is True

    @pytest.mark.skipif(
        not Path("/proc").is_dir(), reason="process start ticks are read from /proc"
    )
    def test_a_top_level_run_still_marks_the_box_busy(self, tmp_path):
        self._marker(tmp_path / "hero", live=True)

        assert supervised_job_probe(tmp_path)() is True

    def test_a_nested_completed_arm_does_not_hold_the_program_idle(self, tmp_path):
        run_dir = tmp_path / "qat-recovery" / "muon-1e-3"
        run_dir.mkdir(parents=True)
        (run_dir / "inflight.json").write_text(
            json.dumps({"schema": 1, "completed": True, "supervisor_pid": 1})
        )

        assert supervised_job_probe(tmp_path)() is False
