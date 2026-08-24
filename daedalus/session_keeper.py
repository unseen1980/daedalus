"""Bounded Claude engineering sessions for the unattended Vast program.

``scripts/vast_program.py`` owns phases, gates, and the 144-hour deadline, but
nothing owned the Claude Code sessions that implement each slice: when a session
exited, was rate limited, or reached a turn cap, an operator had to relaunch it
by hand.  This module supplies the missing supervisor so a 144-hour program does
not depend on an attended laptop.

Responsibilities, taken from the program's autonomous repair contract:

* refuse to start engineering work once the program reaches a terminal status or
  crosses into the reserved finalization window;
* resume the *same* session for a bounded number of repair continuations so the
  failing turn keeps its context;
* escalate to a fresh independent session once those continuations are spent;
* record a hard blocker instead of relaunching forever;
* treat "exited cleanly but changed nothing" as a failure rather than progress.

:func:`decide` is a pure function, so every rule above is exercised in tests
without spawning Claude.
"""

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from daedalus.program_state import (
    ProgramDeadline,
    ProgramStateStore,
    write_json_atomic,
)


KEEPER_STATE_SCHEMA = 1

#: Program statuses after which no further engineering session may start.
TERMINAL_PROGRAM_STATUSES = frozenset(
    {"complete", "completed", "halted", "blocked", "failed"}
)

#: Phase names that carry no engineering work.
IDLE_PHASES = frozenset({"", "not_started"})


def _utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class KeeperPolicy:
    """Bounded-retry limits for one phase's engineering sessions."""

    max_resume_attempts: int = 3
    max_generations: int = 2
    backoff_base_sec: float = 30.0
    backoff_cap_sec: float = 600.0
    session_timeout_sec: float = 3600.0
    busy_poll_sec: float = 900.0
    finalization_phase: str = "phase9-finalization"

    def backoff_for(self, consecutive_failures: int) -> float:
        """Exponential backoff with a finite cap, per the reliability policy."""

        if consecutive_failures <= 0:
            return 0.0
        scaled = self.backoff_base_sec * (2 ** (consecutive_failures - 1))
        return float(min(scaled, self.backoff_cap_sec))


@dataclass(frozen=True)
class KeeperAction:
    """One decision: launch a session, wait, stop cleanly, or block."""

    kind: str
    reason: str
    wait_seconds: float = 0.0
    resume_session_id: Optional[str] = None
    generation: int = 1
    attempt: int = 1


@dataclass
class KeeperState:
    """Durable record of this phase's session attempts."""

    phase: str = ""
    session_id: Optional[str] = None
    generation: int = 1
    attempt: int = 0
    consecutive_failures: int = 0
    launches: int = 0
    last_exit_code: Optional[int] = None
    last_exit_at: Optional[str] = None
    last_failure: str = ""
    blocked_reason: str = ""
    progress_fingerprint: str = ""
    schema: int = KEEPER_STATE_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "KeeperState":
        if not payload:
            return cls()
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass(frozen=True)
class LaunchRequest:
    """Everything one Claude engineering turn needs to start."""

    phase: str
    generation: int
    attempt: int
    resume_session_id: Optional[str] = None
    failure_context: str = ""
    system_prompt_path: Optional[Path] = None


@dataclass(frozen=True)
class PlanGuard:
    """Refuse to launch a session against a silently changed program plan.

    ``hashes_path`` is an ordinary ``sha256sum`` manifest naming the versioned
    plan and the protected execution plan. Both are verified before every launch
    and concatenated, in manifest order, into a root-only prompt file so the
    session sees reviewable scope together with operational detail.
    """

    hashes_path: Path

    def entries(self):
        """Parsed ``(digest, path)`` pairs from the sha256sum manifest."""

        pairs = []
        for line in Path(self.hashes_path).read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            digest, _, name = stripped.partition(" ")
            name = name.strip().lstrip("*")
            if not digest or not name:
                raise ValueError(f"malformed plan hash line: {line!r}")
            pairs.append((digest.lower(), Path(name)))
        if not pairs:
            raise ValueError(f"no plan hashes recorded in {self.hashes_path}")
        return pairs

    def verify(self):
        """``(ok, detail)`` for the recorded plans, without printing content."""

        try:
            entries = self.entries()
        except (OSError, ValueError) as error:
            return False, f"unreadable plan manifest: {error}"
        for expected, plan_path in entries:
            try:
                actual = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            except OSError as error:
                return False, f"unreadable plan {plan_path}: {error}"
            if actual != expected:
                return False, f"plan {plan_path} does not match its recorded hash"
        return True, ""

    def materialize(self, destination) -> Path:
        """Write the verified concatenation to a fresh mode-0600 prompt file."""

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sections = []
        for _, plan_path in self.entries():
            sections.append(f"# Plan: {plan_path.name}\n\n{plan_path.read_text()}")
        temporary = destination.with_name(destination.name + ".tmp")
        handle = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(handle, "w") as stream:
            stream.write("\n\n".join(sections))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        return destination


@dataclass(frozen=True)
class LaunchOutcome:
    """Result of one engineering turn, free of credential material."""

    exit_code: int
    session_id: Optional[str] = None
    tail: str = ""


def remaining_backoff(
    keeper_state: KeeperState, now: datetime, policy: KeeperPolicy
) -> float:
    """Seconds still owed before the next relaunch may happen."""

    if keeper_state.consecutive_failures <= 0 or not keeper_state.last_exit_at:
        return 0.0
    owed = policy.backoff_for(keeper_state.consecutive_failures)
    elapsed = (now - _parse_timestamp(keeper_state.last_exit_at)).total_seconds()
    return max(0.0, owed - elapsed)


def decide(
    *,
    program_state: dict,
    keeper_state: KeeperState,
    now: datetime,
    deadline: ProgramDeadline,
    policy: KeeperPolicy,
    supervised_job_live: bool = False,
) -> KeeperAction:
    """Choose the next keeper action without performing any side effect."""

    if keeper_state.blocked_reason:
        return KeeperAction(
            kind="block",
            reason=keeper_state.blocked_reason,
            generation=keeper_state.generation,
            attempt=keeper_state.attempt,
        )

    status = str(program_state.get("status", ""))
    if status in TERMINAL_PROGRAM_STATUSES:
        return KeeperAction(kind="stop", reason=f"program status {status}")

    stage = deadline.stage(now)
    if stage == "expired":
        return KeeperAction(kind="stop", reason="deadline stage expired")

    phase = str(program_state.get("phase", ""))
    if stage == "finalizing" and phase != policy.finalization_phase:
        # The reserved window is for rescoring, uploads, and reports; it is the
        # one transition the keeper makes on its own rather than waiting for a
        # turn to make it.
        return KeeperAction(
            kind="finalize",
            reason="reserved finalization window reached",
            generation=1,
            attempt=1,
        )
    if phase in IDLE_PHASES:
        return KeeperAction(
            kind="wait",
            reason="no active phase",
            wait_seconds=policy.backoff_base_sec,
        )

    if supervised_job_live:
        # A long training or evaluation job owns the box; an engineering turn
        # would have nothing to do and its idleness would read as a failure.
        return KeeperAction(
            kind="wait",
            reason="supervised job in flight",
            wait_seconds=policy.busy_poll_sec,
            generation=keeper_state.generation,
            attempt=keeper_state.attempt,
        )

    if phase != keeper_state.phase:
        return KeeperAction(
            kind="launch",
            reason="new phase",
            generation=1,
            attempt=1,
        )

    if keeper_state.generation > policy.max_generations:
        return KeeperAction(
            kind="block",
            reason=(
                f"{policy.max_generations} independent sessions failed on "
                f"phase {phase}"
            ),
            generation=keeper_state.generation,
            attempt=keeper_state.attempt,
        )

    waiting = remaining_backoff(keeper_state, now, policy)
    if waiting > 0:
        return KeeperAction(
            kind="wait",
            reason="retry backoff",
            wait_seconds=waiting,
            generation=keeper_state.generation,
            attempt=keeper_state.attempt,
        )

    # The attempt budget is spent whether or not the failing turn managed to
    # report a session id: a turn that times out or never starts reports none.
    if keeper_state.attempt >= policy.max_resume_attempts:
        generation = keeper_state.generation + 1
        if generation > policy.max_generations:
            return KeeperAction(
                kind="block",
                reason=(
                    f"{policy.max_resume_attempts} repair continuations failed "
                    f"on phase {phase}"
                ),
                generation=keeper_state.generation,
                attempt=keeper_state.attempt,
            )
        return KeeperAction(
            kind="launch",
            reason="independent review session",
            generation=generation,
            attempt=1,
        )

    if keeper_state.session_id:
        return KeeperAction(
            kind="launch",
            reason="repair continuation",
            resume_session_id=keeper_state.session_id,
            generation=keeper_state.generation,
            attempt=keeper_state.attempt + 1,
        )

    return KeeperAction(
        kind="launch",
        reason="fresh session",
        generation=keeper_state.generation,
        attempt=keeper_state.attempt + 1,
    )


class ClaudeSessionLauncher:
    """Run one bounded, non-interactive Claude engineering turn."""

    def __init__(
        self,
        *,
        repo: Path,
        prompt_path: Path,
        prompt_dir: Optional[Path] = None,
        system_prompt_path: Optional[Path] = None,
        claude_bin: str = "claude",
        model: str = "opus",
        effort: str = "xhigh",
        permission_mode: str = "dontAsk",
        timeout_sec: float = 3600.0,
        runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    ):
        self.repo = Path(repo)
        self.prompt_path = Path(prompt_path)
        self.prompt_dir = Path(prompt_dir) if prompt_dir is not None else None
        self.system_prompt_path = (
            Path(system_prompt_path) if system_prompt_path is not None else None
        )
        self.claude_bin = claude_bin
        self.model = model
        self.effort = effort
        self.permission_mode = permission_mode
        self.timeout_sec = timeout_sec
        self.runner = runner or run_process_group

    def resolve_prompt_path(self, request: LaunchRequest) -> Path:
        """Phase-specific prompt when one exists, otherwise the standing prompt."""

        if self.prompt_dir is not None:
            candidate = self.prompt_dir / f"{request.phase}.md"
            if candidate.is_file():
                return candidate
        return self.prompt_path

    def build_prompt(self, request: LaunchRequest) -> str:
        """Phase prompt plus the explicit failure context for a continuation."""

        prompt = self.resolve_prompt_path(request).read_text()
        if not request.failure_context:
            return prompt
        return (
            f"{prompt}\n\n## Current failure context\n\n"
            f"Attempt {request.attempt} of generation {request.generation} on "
            f"phase {request.phase}. The previous turn ended without recorded "
            f"progress:\n\n{request.failure_context}\n\n"
            "Reproduce the failure narrowly, repair it, and push the tested fix "
            "before starting new work."
        )

    def build_argv(self, request: LaunchRequest) -> Sequence[str]:
        """Exact non-interactive Claude invocation for one engineering turn."""

        argv = [
            self.claude_bin,
            "-p",
            self.build_prompt(request),
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--permission-mode",
            self.permission_mode,
            "--output-format",
            "json",
        ]
        system_prompt = request.system_prompt_path or self.system_prompt_path
        if system_prompt is not None:
            argv += ["--append-system-prompt-file", str(system_prompt)]
        if request.resume_session_id:
            argv += ["--resume", request.resume_session_id]
        return argv

    def launch(self, request: LaunchRequest) -> LaunchOutcome:
        """Execute the turn, returning its exit code and reported session id."""

        argv = list(self.build_argv(request))
        try:
            completed = self.runner(
                argv,
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return LaunchOutcome(
                exit_code=124,
                session_id=request.resume_session_id,
                tail=f"session exceeded {self.timeout_sec:.0f}s timeout",
            )
        except OSError as error:
            return LaunchOutcome(
                exit_code=127,
                session_id=request.resume_session_id,
                tail=f"could not launch Claude: {error}",
            )

        session_id = _session_id_from_output(completed.stdout) or request.resume_session_id
        # A successful turn's stdout is the structured result, not diagnostics.
        diagnostics = (completed.stderr or "").strip()
        if not diagnostics and completed.returncode != 0:
            diagnostics = completed.stdout or ""
        tail = _tail(diagnostics)
        return LaunchOutcome(
            exit_code=int(completed.returncode),
            session_id=session_id,
            tail=tail,
        )


def run_process_group(
    argv,
    *,
    cwd=None,
    timeout=None,
    text: bool = True,
    capture_output: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``argv`` in its own process group so a timeout kills the whole tree.

    A Claude turn launches the approved wrapper, which launches llama.cpp on the
    GPU. Killing only the direct child would leave that GPU process holding the
    card while the next turn tries to score against it.
    """

    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            list(argv), timeout, output=stdout, stderr=stderr
        )
    completed = subprocess.CompletedProcess(
        list(argv), process.returncode, stdout, stderr
    )
    if check:
        completed.check_returncode()
    return completed


def _kill_process_group(process: subprocess.Popen) -> None:
    """Escalate from SIGTERM to SIGKILL across the launched process group."""

    try:
        group = os.getpgid(process.pid)
    except OSError:
        process.kill()
        return
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, signal_number)
        except OSError:
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def supervised_job_probe(runs_root) -> Callable[[], bool]:
    """True while a supervised training or evaluation run owns the box.

    Liveness that cannot be established reads as *not* busy: an unrecognised or
    stale marker must never leave the program permanently idle, and the
    supervised launcher refuses to start a second trainer on its own.
    """

    root = Path(runs_root)

    def probe() -> bool:
        from daedalus.supervise import read_inflight, supervisor_is_live

        if not root.is_dir():
            return False
        for marker_path in sorted(root.glob("*/inflight.json")):
            marker = read_inflight(str(marker_path.parent))
            if not marker or marker.get("completed"):
                continue
            if supervisor_is_live(marker) is True:
                return True
        return False

    return probe


def _session_id_from_output(stdout: Optional[str]) -> Optional[str]:
    """Read the session id Claude reports in ``--output-format json`` mode."""

    if not stdout:
        return None
    for candidate in (stdout, *reversed(stdout.strip().splitlines())):
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("session_id"):
            return str(payload["session_id"])
    return None


def _tail(text: str, limit: int = 600) -> str:
    collapsed = text.strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[-limit:]


class SessionKeeper:
    """Keep exactly one bounded engineering session alive for the active phase."""

    def __init__(
        self,
        *,
        store: ProgramStateStore,
        keeper_state_path,
        launcher: ClaudeSessionLauncher,
        policy: KeeperPolicy = KeeperPolicy(),
        progress_probe: Optional[Callable[[], str]] = None,
        plan_guard: Optional[PlanGuard] = None,
        plan_context_path=None,
        busy_probe: Optional[Callable[[], bool]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.store = store
        self.keeper_state_path = Path(keeper_state_path)
        self.launcher = launcher
        self.policy = policy
        self.progress_probe = progress_probe
        self.busy_probe = busy_probe
        self.plan_guard = plan_guard
        self.plan_context_path = (
            Path(plan_context_path) if plan_context_path is not None else None
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper

    def load_state(self) -> KeeperState:
        if not self.keeper_state_path.exists():
            return KeeperState()
        with self.keeper_state_path.open() as handle:
            return KeeperState.from_dict(json.load(handle))

    def save_state(self, state: KeeperState) -> None:
        write_json_atomic(self.keeper_state_path, state.to_dict())

    def _deadline(self, program_state: dict) -> ProgramDeadline:
        return ProgramDeadline(
            started_at=_parse_timestamp(program_state["started_at"]),
            hard_hours=float(program_state.get("hard_hours", 144.0)),
            reserve_hours=float(program_state.get("reserve_hours", 8.0)),
        )

    def _supervised_job_live(self) -> bool:
        if self.busy_probe is None:
            return False
        try:
            return bool(self.busy_probe())
        except Exception:  # an unreadable marker must not stop supervision
            return False

    def _fingerprint(self) -> str:
        if self.progress_probe is None:
            return ""
        try:
            return str(self.progress_probe())
        except Exception as error:  # a probe failure must not stop supervision
            return f"probe-error:{error}"

    def step(self) -> KeeperAction:
        """Run one decide-and-act cycle."""

        program_state = self.store.load()
        keeper_state = self.load_state()
        now = self.clock()
        action = decide(
            program_state=program_state,
            keeper_state=keeper_state,
            now=now,
            deadline=self._deadline(program_state),
            policy=self.policy,
            supervised_job_live=self._supervised_job_live(),
        )

        if action.kind == "wait":
            self.sleeper(action.wait_seconds)
            return action
        if action.kind == "stop":
            return action
        if action.kind == "block":
            self._block(program_state, keeper_state, action, now)
            return action
        if action.kind == "finalize":
            self._finalize(program_state, now)
            return action

        system_prompt_path = None
        if self.plan_guard is not None:
            verified, detail = self.plan_guard.verify()
            if not verified:
                refused = KeeperAction(
                    kind="block",
                    reason=f"refusing to launch against a changed plan: {detail}",
                    generation=action.generation,
                    attempt=action.attempt,
                )
                self._block(program_state, keeper_state, refused, now)
                return refused
            if self.plan_context_path is not None:
                system_prompt_path = self.plan_guard.materialize(self.plan_context_path)

        self._launch(program_state, keeper_state, action, now, system_prompt_path)
        return action

    def run(self, *, max_cycles: Optional[int] = None) -> KeeperAction:
        """Supervise until the program stops, blocks, or the cycle cap is hit."""

        cycles = 0
        action = KeeperAction(kind="wait", reason="not started")
        while max_cycles is None or cycles < max_cycles:
            action = self.step()
            cycles += 1
            if action.kind in {"stop", "block"}:
                break
        return action

    def _finalize(self, program_state: dict, now: datetime) -> None:
        """Hand the program to its finalization phase inside the reserve."""

        details = dict(program_state.get("details") or {})
        details["reason"] = "reserved finalization window reached"
        details["previous_phase"] = str(program_state.get("phase", ""))
        self.store.transition(
            phase=self.policy.finalization_phase,
            status="running",
            now=now,
            details=details,
        )

    def _block(
        self,
        program_state: dict,
        keeper_state: KeeperState,
        action: KeeperAction,
        now: datetime,
    ) -> None:
        keeper_state.blocked_reason = action.reason
        self.save_state(keeper_state)
        if program_state.get("status") == "blocked":
            return
        details = dict(program_state.get("details") or {})
        details["blocker"] = action.reason
        details["last_session_failure"] = keeper_state.last_failure
        self.store.transition(
            phase=str(program_state.get("phase", "")),
            status="blocked",
            now=now,
            details=details,
        )

    def _launch(
        self,
        program_state: dict,
        keeper_state: KeeperState,
        action: KeeperAction,
        now: datetime,
        system_prompt_path: Optional[Path] = None,
    ) -> None:
        phase = str(program_state.get("phase", ""))
        request = LaunchRequest(
            phase=phase,
            generation=action.generation,
            attempt=action.attempt,
            resume_session_id=action.resume_session_id,
            failure_context=keeper_state.last_failure,
            system_prompt_path=system_prompt_path,
        )
        before = self._fingerprint()
        self.store.append_event(
            {
                "kind": "session_launch",
                "at": _utc_timestamp(now),
                "phase": phase,
                "reason": action.reason,
                "generation": action.generation,
                "attempt": action.attempt,
                "resumed": bool(action.resume_session_id),
            }
        )

        outcome = self.launcher.launch(request)
        after = self._fingerprint()
        made_progress = self.progress_probe is None or after != before
        succeeded = outcome.exit_code == 0 and made_progress

        finished = self.clock()
        keeper_state.phase = phase
        keeper_state.generation = action.generation
        keeper_state.attempt = action.attempt
        keeper_state.launches += 1
        keeper_state.session_id = outcome.session_id
        keeper_state.last_exit_code = outcome.exit_code
        keeper_state.last_exit_at = _utc_timestamp(finished)
        keeper_state.progress_fingerprint = after

        if succeeded:
            keeper_state.consecutive_failures = 0
            keeper_state.last_failure = ""
            keeper_state.attempt = 0
            keeper_state.generation = 1
            keeper_state.session_id = None
        else:
            keeper_state.consecutive_failures += 1
            keeper_state.last_failure = (
                f"exit={outcome.exit_code} progressed={made_progress} "
                f"{outcome.tail}".strip()
            )

        self.save_state(keeper_state)
        self.store.append_event(
            {
                "kind": "session_exit",
                "at": _utc_timestamp(finished),
                "phase": phase,
                "exit_code": outcome.exit_code,
                "progressed": made_progress,
                "succeeded": succeeded,
                "generation": action.generation,
                "attempt": action.attempt,
            }
        )


def git_progress_probe(repo) -> Callable[[], str]:
    """Fingerprint source progress as ``HEAD sha`` plus the working-tree state."""

    repo_path = Path(repo)

    def probe() -> str:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        return f"{head.stdout.strip()}|{dirty.stdout.strip()}"

    return probe
