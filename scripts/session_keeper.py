"""Keep one bounded Claude engineering session alive for the active phase.

The deterministic controller owns phases, gates, and the deadline; this service
owns the engineering sessions that implement them, so a session that exits, is
rate limited, or reaches a turn cap is relaunched without an attended laptop.
Credentials are inherited from the protected runtime environment and are never
echoed, logged, or written into program state.
"""

import argparse
import sys
import time
from pathlib import Path

from daedalus.program_state import ProgramStateStore
from daedalus.session_keeper import (
    ClaudeSessionLauncher,
    KeeperPolicy,
    PlanGuard,
    SessionKeeper,
    filesystem_activity_probe,
    git_progress_probe,
    supervised_job_probe,
)


def build_keeper(args) -> SessionKeeper:
    """Assemble the keeper from validated command-line preconditions."""

    repo = Path(args.repo).resolve()
    prompt_path = Path(args.default_prompt)
    if not prompt_path.is_file():
        raise SystemExit(f"missing standing prompt {prompt_path}")

    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else None
    system_prompt_path = (
        Path(args.system_prompt_file) if args.system_prompt_file else None
    )
    if system_prompt_path is not None and not system_prompt_path.is_file():
        raise SystemExit(f"missing system prompt {system_prompt_path}")

    plan_guard = PlanGuard(hashes_path=Path(args.plan_hashes)) if args.plan_hashes else None
    if plan_guard is not None:
        verified, detail = plan_guard.verify()
        if not verified:
            raise SystemExit(f"plan verification failed: {detail}")

    policy = KeeperPolicy(
        max_resume_attempts=args.max_resume_attempts,
        max_generations=args.max_generations,
        backoff_base_sec=args.backoff_base_sec,
        backoff_cap_sec=args.backoff_cap_sec,
        session_timeout_sec=args.session_timeout_sec,
        busy_poll_sec=args.busy_poll_sec,
    )
    launcher = ClaudeSessionLauncher(
        repo=repo,
        prompt_path=prompt_path,
        prompt_dir=prompt_dir,
        system_prompt_path=system_prompt_path,
        claude_bin=args.claude_bin,
        model=args.model,
        effort=args.effort,
        permission_mode=args.permission_mode,
        timeout_sec=args.session_timeout_sec,
        idle_timeout_sec=args.idle_timeout_sec,
        progress_probe=filesystem_activity_probe(repo),
        poll_interval_sec=args.idle_poll_sec,
    )
    return SessionKeeper(
        store=ProgramStateStore(args.state),
        keeper_state_path=args.keeper_state,
        launcher=launcher,
        policy=policy,
        progress_probe=git_progress_probe(repo),
        busy_probe=supervised_job_probe(args.runs_root or repo / "runs"),
        plan_guard=plan_guard,
        plan_context_path=args.plan_context,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--keeper-state", required=True)
    parser.add_argument("--prompt-dir")
    parser.add_argument("--default-prompt", required=True)
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--runs-root")
    parser.add_argument("--busy-poll-sec", type=float, default=900.0)
    parser.add_argument("--plan-hashes")
    parser.add_argument("--plan-context")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--model", default="opus")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--permission-mode", default="dontAsk")
    parser.add_argument("--session-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--idle-timeout-sec", type=float, default=5400.0)
    parser.add_argument("--idle-poll-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=60.0)
    parser.add_argument("--max-resume-attempts", type=int, default=3)
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--backoff-base-sec", type=float, default=30.0)
    parser.add_argument("--backoff-cap-sec", type=float, default=600.0)
    parser.add_argument("--max-cycles", type=int)
    args = parser.parse_args(argv)

    keeper = build_keeper(args)
    cycles = 0
    while args.max_cycles is None or cycles < args.max_cycles:
        try:
            action = keeper.step()
        except Exception as error:  # noqa: BLE001 - the service retries next poll
            print(f"session-keeper: {type(error).__name__}", flush=True)
            time.sleep(max(args.poll_interval_sec, 1.0))
            cycles += 1
            continue
        print(f"session-keeper: {action.kind} ({action.reason})", flush=True)
        cycles += 1
        if action.kind in {"stop", "block"}:
            return 0 if action.kind == "stop" else 1
        if action.kind != "wait":
            time.sleep(max(args.poll_interval_sec, 1.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
