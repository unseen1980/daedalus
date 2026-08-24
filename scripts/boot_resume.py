"""Resume an approved in-flight training run after a box reboot."""

import os
from typing import Callable, Optional

from daedalus.supervise import (
    INFLIGHT_SCHEMA,
    _read_halt,
    note_boot_resume,
    read_inflight,
    run_with_resume,
    supervisor_is_live,
    trainer_is_live,
)


MAX_BOOT_RESUMES = 5


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