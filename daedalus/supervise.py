"""Supervised training subprocesses: retry a crashed run from its checkpoint.

`train.py` checkpoints every 30 min, so a crash costs at most that -- but only
if something restarts it. Nothing did: `abl_arch.run_training` used
`subprocess.run(check=True)` and a blip discarded a 12 h arm, and `hero` is a
four-day run with no supervisor at all, where an unnoticed death costs
$0.449/h until someone looks.

This is the shared primitive. `abl_arch.py` currently carries its own copy of
the same loop (added first, and left alone while it was about to run
unattended); converging it onto this module is a follow-up, not a thing to do
the evening it matters.

**`run_with_resume` only survives a crash of the *trainer*, not of the box.**
It is a loop inside a Python process, so a reboot, a stop/start, or an OOM kill
of the launcher takes the supervisor down with the run it was supervising. On
this instance nothing brings either back: `/root/onstart.sh` is one line
(`entrypoint.sh`), there is no crontab, and `/etc/supervisor/conf.d` holds only
the stock image services. The 2026-08-08 wedge ended exactly that way -- the
operator rebooted and every process, agent included, was gone.

The in-flight marker below is what closes that. It records the *exact* argv a
run was launched with, so `scripts/boot_resume.py` can re-enter this function
after a reboot without ever constructing a training command of its own. That
distinction is the safety property: the marker only exists because an approved
launcher already ran, so boot resume can continue `hero` but can never start it
(`tests/test_hero_gate_safety.py`).
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional


def _read_halt(path: Optional[str]) -> Optional[dict]:
    """The watchdog's halt marker, or None. Unreadable or half-written counts
    as absent: this decides whether to keep training, so it fails towards the
    behaviour that existed before the marker did."""
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


INFLIGHT = "inflight.json"
INFLIGHT_SCHEMA = 1


def inflight_path(run_dir: str) -> str:
    return os.path.join(run_dir, INFLIGHT)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: str, payload: dict) -> None:
    """Write via tmp+rename so a reboot mid-write cannot leave a half-marker.

    A torn `inflight.json` reads as absent (see `read_inflight`), which means a
    resumable run silently stops being resumable -- the one outcome this whole
    mechanism exists to prevent. `os.replace` is atomic within a filesystem, so
    a reader sees either the old marker or the new one, never a prefix.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def proc_start_ticks(pid: int) -> Optional[int]:
    """Field 22 of `/proc/<pid>/stat` -- when this process started, in clock
    ticks since boot.

    A pid on its own is not an identity. Linux reuses pids within seconds, so
    "is pid 1133711 alive" can be answered `True` by a process that has nothing
    to do with the one that wrote the marker. `(pid, starttime)` is unique for
    the life of the boot, which is exactly the window that matters here.

    Parsed from the last `)` rather than by splitting on whitespace: field 2 is
    the comm, it is parenthesised, and it may contain spaces.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        return int(raw[raw.rindex(")") + 1:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def supervisor_is_live(marker: Optional[dict]) -> Optional[bool]:
    """Is the process that launched and is retrying this run still alive?

    Distinct from `trainer_is_live`, and the distinction is the whole point.
    `run_with_resume` sleeps up to `max_backoff_sec` (900 s) *between* attempts,
    and in that window there is no `train.py` at all while the run is perfectly
    healthy and about to continue. Anything that reacts to "no trainer" by
    launching one would put a second trainer on a 32.6 GB card next to the one
    the live supervisor is about to start.

    Returns `None`, not `False`, when the marker predates this field -- an
    unknown answer must not read as "the launcher is gone". `hero` was launched
    on 2026-08-11T10:22:07Z from code that did not record it, so the caller has
    to have a story for `None` (see `scripts/train_watch.py`, which falls back
    to requiring a sustained absence longer than the backoff ceiling).
    """
    if not marker:
        return None
    pid = marker.get("supervisor_pid")
    ticks = marker.get("supervisor_start_ticks")
    if not isinstance(pid, int) or not isinstance(ticks, int):
        return None
    return proc_start_ticks(pid) == ticks


def read_inflight(run_dir: str) -> Optional[dict]:
    """The in-flight marker for `run_dir`, or None if absent/unreadable.

    Unreadable counts as absent, and here that fails *towards not resuming*.
    That is the opposite of `_read_halt`'s bias and deliberately so: a marker
    with no `cmd` cannot be acted on anyway, and an idle box is visible in
    `HEARTBEAT.md` within minutes, whereas guessing a training command is how
    you overwrite a 90-hour checkpoint.
    """
    try:
        with open(inflight_path(run_dir)) as f:
            marker = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) else None


def write_inflight(run_dir: str, cmd: List[str], ckpt_path: str,
                   extra: Optional[dict] = None, **kw) -> Optional[str]:
    """Record what it would take to continue this run after the box restarts.

    Never raises. A run that trains fine but cannot write its marker should
    keep training -- losing the insurance is bad, losing the run to a failed
    `open()` is worse.
    """
    try:
        os.makedirs(run_dir, exist_ok=True)
        previous = read_inflight(run_dir) or {}
        payload = {
            "schema": INFLIGHT_SCHEMA,
            "run_name": os.path.basename(os.path.normpath(run_dir)),
            "run_dir": run_dir,
            "cmd": list(cmd),
            "ckpt_path": ckpt_path,
            "cwd": os.getcwd(),
            "started_at": previous.get("started_at") or _utcnow(),
            "updated_at": _utcnow(),
            # Survives across boots: how many times a *reboot* (not a crash)
            # has restarted this run. `boot_resume` caps it so a box stuck in a
            # reboot loop cannot spend the balance one restart at a time.
            "boot_resumes": int(previous.get("boot_resumes", 0)),
            "completed": False,
            "outcome": None,
            # Who is retrying this run. Written here because this function runs
            # *in* the launcher process, so `os.getpid()` is precisely the
            # process whose death leaves the run unsupervised. See
            # `supervisor_is_live`: without it, a watcher cannot tell a dead
            # launcher from one asleep in its inter-attempt backoff.
            "supervisor_pid": os.getpid(),
            "supervisor_start_ticks": proc_start_ticks(os.getpid()),
        }
        payload.update(kw)
        if extra:
            payload["extra"] = extra
        path = inflight_path(run_dir)
        _write_json_atomic(path, payload)
        return path
    except Exception as e:            # noqa: BLE001 - insurance must not bite
        print(f"[supervise] WARNING: could not write in-flight marker: {e}",
              flush=True)
        return None


def mark_inflight_done(run_dir: str, outcome: str) -> None:
    """Close the marker so boot resume leaves a finished run alone.

    Called for *both* endings. A success obviously must not be restarted; a
    `TrainingFailed` must not either, because that path is reached only when
    the watchdog halted the run or every attempt was spent, and both are
    decisions the boot guard has no standing to overturn.
    """
    try:
        marker = read_inflight(run_dir)
        if marker is None:
            return
        marker["completed"] = True
        marker["outcome"] = outcome
        marker["updated_at"] = _utcnow()
        _write_json_atomic(inflight_path(run_dir), marker)
    except Exception as e:            # noqa: BLE001
        print(f"[supervise] WARNING: could not close in-flight marker: {e}",
              flush=True)


def note_boot_resume(run_dir: str) -> int:
    """Increment and return the reboot-restart count for this run."""
    marker = read_inflight(run_dir)
    if marker is None:
        return 0
    n = int(marker.get("boot_resumes", 0)) + 1
    marker["boot_resumes"] = n
    marker["updated_at"] = _utcnow()
    try:
        _write_json_atomic(inflight_path(run_dir), marker)
    except Exception as e:            # noqa: BLE001
        print(f"[supervise] WARNING: could not bump boot_resumes: {e}",
              flush=True)
    return n


def trainer_is_live(run_dir: str) -> bool:
    """Is a `train.py` for *this* run actually running right now?

    Deliberately not a bare `os.kill(pid, 0)`. After a reboot the pid in
    `train.pid` is stale and Linux hands out low pids again immediately, so a
    liveness check that only asks "does this pid exist" will happily conclude
    that systemd is the trainer and skip the resume. Same idiom as
    `credit_watch.budget_from_running_trainer`: confirm the cmdline is a
    `train.py` *and* names this run.
    """
    try:
        with open(os.path.join(run_dir, "train.pid")) as f:
            pid = int(f.read().strip())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = [a for a in f.read().split(b"\0") if a]
    except (OSError, ValueError):
        return False
    if not any(a.endswith(b"train.py") for a in argv):
        return False
    name = os.path.basename(os.path.normpath(run_dir)).encode()
    return any(a == name for a in argv)


def start_watchdog(run_name: str, run_dir: str, target_tokens: int,
                   stall_min: float, supervised: bool = True,
                   log: Optional[Callable] = None):
    """Spawn `watchdog.py` alongside a supervised run.

    Shared by `hero.py` and `abl_arch.py` so the flags cannot drift apart --
    `--supervised` in particular, without which a watchdog exits on the first
    recoverable crash and leaves the rest of a multi-day run unwatched.

    Never raises: an unstarted watchdog costs detection, and killing the job it
    was meant to protect costs the job.
    """
    log = log or (lambda m: print(m, flush=True))
    cmd = [sys.executable, "watchdog.py",
           "--run-name", run_name, "--run-dir", run_dir,
           "--target-tokens", str(target_tokens),
           "--stall-min", str(stall_min)]
    if supervised:
        cmd.append("--supervised")
    try:
        proc = subprocess.Popen(cmd)
        log(f"[supervise] watchdog pid {proc.pid}: {' '.join(cmd)}")
        return proc
    except Exception as e:
        log(f"[supervise] WARNING: could not start watchdog ({e}); continuing")
        return None


def stop_watchdog(proc, grace_sec: float = 30.0) -> None:
    """Terminate a watchdog started above. A failed run must not leave one
    polling a dead directory for the rest of the night."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()


class TrainingFailed(RuntimeError):
    """Every attempt failed. Carries the attempt history for the report."""

    def __init__(self, message: str, attempts: int, returncodes: List[int],
                 halt: Optional[dict] = None):
        super().__init__(message)
        self.attempts = attempts
        self.returncodes = returncodes
        # Set when the run was stopped by watchdog.py rather than by dying on
        # its own: {"kind": "divergence"|"stall", "reason": str, "at": float}.
        self.halt = halt


def run_with_resume(cmd: List[str], ckpt_path: str, max_attempts: int = 10,
                    backoff_sec: float = 60.0, max_backoff_sec: float = 900.0,
                    runner: Optional[Callable] = None,
                    sleeper: Optional[Callable] = None,
                    log: Optional[Callable] = None,
                    halt_marker: Optional[str] = None,
                    force_resume: bool = False,
                    record_inflight: bool = True,
                    inflight_extra: Optional[dict] = None) -> dict:
    """Run `cmd`, restarting it with `--resume <ckpt_path>` if it dies.

    Returns a report dict: attempts, resumed, returncodes. Raises
    TrainingFailed once `max_attempts` is exhausted, or as soon as
    `halt_marker` shows watchdog.py deliberately stopped the run.

    Backoff is exponential and capped. A crash that repeats instantly --
    a bad flag, a corrupt checkpoint, an OOM that will recur -- would
    otherwise burn every attempt in seconds and leave the box idle for the
    rest of the night; spacing them out means a transient cause (a
    filesystem hiccup, a driver blip) has time to clear, and a permanent one
    still terminates rather than looping forever.

    **`halt_marker` is what makes a watchdog halt stick.** Retrying is the
    right answer to a crash and the wrong one to a divergence: watchdog.py
    SIGTERMs the trainer and *exits*, so without this check the supervisor
    reads the resulting non-zero exit as a crash, resumes the diverged
    checkpoint with no watchdog left running, and trains a broken model for
    the rest of the run before exiting 0. On `hero` that is up to three days
    and ~$30 spent after the point where the loss had already gone.

    **`force_resume` is what `scripts/boot_resume.py` needs, and getting it
    wrong is the expensive failure in this file.** The `attempt > 1` rule below
    is right for a fresh launch and catastrophically wrong for a restart after
    a reboot: attempt 1 of a *new* supervisor process would relaunch `train.py`
    with no `--resume`, which starts from step 0 and overwrites the rolling
    checkpoint on its first save. At hour 90 of `hero` that silently destroys
    ~$40 of training and looks, from the outside, exactly like a healthy run.
    So a boot resume passes `force_resume=True` and attempt 1 carries
    `--resume` like every later one -- still only when the checkpoint actually
    exists, so the "no checkpoint yet" case is unchanged.

    `runner`/`sleeper`/`log` are injectable so this is testable without
    spawning processes or waiting.
    """
    runner = runner or (lambda c: subprocess.run(c).returncode)
    sleeper = sleeper or time.sleep
    log = log or (lambda m: print(m, flush=True))

    run_dir = os.path.dirname(os.path.abspath(ckpt_path))
    if record_inflight:
        write_inflight(run_dir, cmd, ckpt_path, extra=inflight_extra,
                       max_attempts=max_attempts, backoff_sec=backoff_sec,
                       max_backoff_sec=max_backoff_sec,
                       halt_marker=halt_marker)

    returncodes: List[int] = []
    resumed = False
    for attempt in range(1, max_attempts + 1):
        this_cmd = list(cmd)
        # Resume only from a checkpoint that exists. A first attempt that dies
        # before the 30-minute mark leaves none, and claiming a resume that
        # did not happen corrupts the record the writeup depends on.
        this_resume = (attempt > 1 or force_resume) and os.path.exists(ckpt_path)
        if this_resume:
            this_cmd += ["--resume", ckpt_path]
            resumed = True
        log(f"[supervise] attempt {attempt}/{max_attempts}"
            f"{' (resuming)' if this_resume else ''}: {' '.join(this_cmd)}")

        rc = runner(this_cmd)
        returncodes.append(rc)
        if rc == 0:
            if record_inflight:
                mark_inflight_done(run_dir, "completed")
            return {"attempts": attempt, "resumed": resumed,
                    "returncodes": returncodes}

        # Checked before the retry decision and before the backoff sleep: a
        # halted run must not cost another 15 minutes of the box on its way to
        # a conclusion that is already known.
        halt = _read_halt(halt_marker)
        if halt is not None:
            log(f"[supervise] watchdog halted this run ({halt.get('kind')}): "
                f"{halt.get('reason')} -- not resuming")
            if record_inflight:
                mark_inflight_done(run_dir, f"halted:{halt.get('kind')}")
            raise TrainingFailed(
                f"halted by watchdog after {attempt} attempt(s): "
                f"{halt.get('reason')}",
                attempts=attempt, returncodes=returncodes, halt=halt)

        if attempt == max_attempts:
            break
        delay = min(backoff_sec * (2 ** (attempt - 1)), max_backoff_sec)
        log(f"[supervise] attempt {attempt} exited {rc}; retrying in {delay:.0f}s")
        sleeper(delay)

    if record_inflight:
        mark_inflight_done(run_dir, "attempts_exhausted")
    raise TrainingFailed(
        f"training failed {max_attempts} times (returncodes {returncodes})",
        attempts=max_attempts, returncodes=returncodes)
