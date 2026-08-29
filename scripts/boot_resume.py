"""Resume an approved in-flight training run after a box reboot."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daedalus.supervise import (
    INFLIGHT,
    INFLIGHT_SCHEMA,
    _read_halt,
    note_boot_resume,
    read_inflight,
    run_with_resume,
    supervisor_is_live,
    trainer_is_live,
)


MAX_BOOT_RESUMES = 5


def _pending_run_dirs(runs_root: str) -> list[str]:
    pending = []
    for path in sorted(Path(runs_root).rglob(INFLIGHT)):
        run_dir = str(path.parent)
        marker = read_inflight(run_dir)
        if marker is None or marker.get("schema") != INFLIGHT_SCHEMA:
            continue
        if marker.get("completed") is True:
            continue
        command = marker.get("cmd")
        checkpoint = marker.get("ckpt_path")
        if not isinstance(command, list) or not command:
            continue
        if not isinstance(checkpoint, str) or not os.path.isfile(checkpoint):
            continue
        pending.append(run_dir)
    return pending


def resume_pending(runs_root: str, *, resumer: Callable = None) -> dict:
    """Resume the sole pending run and refuse to guess among several."""
    pending = _pending_run_dirs(runs_root)
    if not pending:
        return {"status": "skipped", "reason": "no_pending_runs"}
    if len(pending) > 1:
        return {
            "status": "blocked",
            "reason": "multiple_pending_runs",
            "count": len(pending),
        }
    action = resumer or resume_run
    result = action(pending[0])
    return result if isinstance(result, dict) else {"status": "completed"}


def resume_run(
    run_dir: str,
    *,
    run: Callable = run_with_resume,
    supervisor_live: Callable = supervisor_is_live,
    trainer_live: Callable = trainer_is_live,
    max_boot_resumes: int = MAX_BOOT_RESUMES,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Resume one valid marker, or return a reason it was left alone."""
    log = log or (lambda message: print(message, flush=True))
    marker = read_inflight(run_dir)
    if marker is None:
        return {"status": "skipped", "reason": "missing_marker"}
    if marker.get("schema") != INFLIGHT_SCHEMA:
        return {"status": "skipped", "reason": "unsupported_schema"}
    if marker.get("completed") is True:
        return {"status": "skipped", "reason": "completed"}
    if _read_halt(marker.get("halt_marker")) is not None:
        return {"status": "skipped", "reason": "watchdog_halt"}
    if supervisor_live(marker) is not False:
        return {"status": "skipped", "reason": "supervisor_live_or_unknown"}
    if trainer_live(run_dir):
        return {"status": "skipped", "reason": "trainer_live"}

    cmd = marker.get("cmd")
    checkpoint = marker.get("ckpt_path")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(arg, str) for arg in cmd):
        return {"status": "skipped", "reason": "invalid_command"}
    if not isinstance(checkpoint, str) or not os.path.isfile(checkpoint):
        return {"status": "skipped", "reason": "missing_checkpoint"}
    if int(marker.get("boot_resumes", 0)) >= max_boot_resumes:
        return {"status": "skipped", "reason": "boot_resume_limit"}

    boot_number = note_boot_resume(run_dir)
    report = run(
        list(cmd),
        checkpoint,
        max_attempts=int(marker.get("max_attempts", 10)),
        backoff_sec=float(marker.get("backoff_sec", 60.0)),
        max_backoff_sec=float(marker.get("max_backoff_sec", 900.0)),
        halt_marker=marker.get("halt_marker"),
        force_resume=True,
        inflight_extra=marker.get("extra"),
        log=log,
    )
    return {"status": "completed", "boot_resume": boot_number, "report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--max-boot-resumes", type=int, default=MAX_BOOT_RESUMES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pending = _pending_run_dirs(args.runs_root)
    if not pending:
        result = {"status": "skipped", "reason": "no_pending_runs"}
    elif len(pending) > 1:
        result = {
            "status": "blocked",
            "reason": "multiple_pending_runs",
            "count": len(pending),
        }
    elif args.dry_run:
        result = {"status": "dry_run", "candidate": pending[0]}
    else:
        result = resume_run(
            pending[0],
            max_boot_resumes=args.max_boot_resumes,
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())