"""Daedalus watchdog (AGENT.md SS3, item 5): detect divergence, stalls, or
completion in a training run, then halt training and write the reason to
STATUS.md. Never terminates the instance (AGENT.md SS0.1, SS7) -- it stops
only the training process it identifies by PID, and only after confirming
that PID is the one train.py itself wrote to runs/<RUN_NAME>/train.pid.

This is a separate, external process from train.py by design: it reads
runs/<RUN_NAME>/metrics.jsonl and the pidfile rather than sharing any
in-process state, so it keeps working across restarts, crashes, or if it's
simply invoked later by a different agent session.

Remediation policy (AGENT.md SS6 -- roll back two checkpoints, halve the
Muon lr, resume) is *not* auto-executed here: watchdog.py's own job
description is detect + halt + report, not decide-and-resume. A halted run
with a clear reason in STATUS.md is what a human or the next agent session
acts on; auto-resuming a diverged run unattended is exactly the kind of
ambiguous, costly call AGENT.md SS0.5 reserves for a human decision gate.
"""
import argparse
import json
import os
import signal
import time
from typing import Optional


def read_metrics(run_dir: str) -> list:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------- divergence ---

def records_since_resume(records: list) -> list:
    """The tail of `records` belonging to the current attempt.

    `run_with_resume` restarts `train.py --resume <ckpt>`, and the resumed
    process continues from the *checkpoint's* step -- so `step` in
    metrics.jsonl jumps backwards at every restart. The loss either side of
    that jump describes two different points on the curve, and comparing
    across it is comparing the model to its own future.

    That is not hypothetical, and it is not cheap. The rolling checkpoint is
    written on a 30-minute gate whose first fire is at step 1, so a hard crash
    in the opening half hour resumes correctly from step 1 -- and then reports
    the step-20 loss (9.32 on `abl-arch` arm 1's real curve) against a running
    mean taken from just before the crash (3.71 at the 30-minute mark, i.e. a
    7.42 threshold). 9.32 > 7.42, so a *successful* recovery was scored as a
    divergence, and divergence is deliberately terminal: watchdog SIGTERMs the
    trainer, writes the halt marker, and `run_with_resume` refuses to resume
    past it. `hero` would have died ~35 minutes into 5.6 days, blaming the
    learning rate, with the box billing until someone noticed.

    Truncating here rather than widening the factor keeps a genuine divergence
    terminal: a real blow-up inside one attempt still trips on the very next
    record, and the NaN check upstream is not gated on history at all.
    """
    steps = [r.get("step") for r in records]
    for i in range(len(records) - 1, 0, -1):
        prev, cur = steps[i - 1], steps[i]
        if prev is not None and cur is not None and cur < prev:
            return records[i:]
    return records


def detect_divergence(records: list, window: int = 20, factor: float = 2.0) -> Optional[str]:
    """Flags the latest loss if it's NaN/inf, or > factor * the running mean
    of the preceding `window` losses (AGENT.md SS6: "2x the running mean").
    Needs at least a few prior points to judge -- returns None too early to
    tell, not a false positive. The history is scoped to the current attempt
    (`records_since_resume`), so a restart re-arms the detector instead of
    tripping it.
    """
    if not records:
        return None
    latest = records[-1].get("loss")
    if latest is None:
        return None
    if latest != latest or latest in (float("inf"), float("-inf")):  # NaN check
        return f"loss is non-finite at step {records[-1].get('step')}: {latest}"

    attempt = records_since_resume(records)
    history = [r["loss"] for r in attempt[:-1][-window:]
              if r.get("loss") is not None and r["loss"] == r["loss"]]
    if len(history) < min(5, window):
        return None  # not enough history to judge yet
    running_mean = sum(history) / len(history)
    if running_mean > 0 and latest > factor * running_mean:
        return (f"loss diverged at step {records[-1].get('step')}: {latest:.4f} > "
               f"{factor}x running mean {running_mean:.4f} (over last {len(history)} points)")
    return None


# ------------------------------------------------------------------ stall ---

def detect_stall(run_dir: str, max_gap_sec: float, now: Optional[float] = None,
                 since: Optional[float] = None) -> Optional[str]:
    """No metrics.jsonl progress in over `max_gap_sec` -- file-mtime based,
    so it works regardless of whether the process is technically still
    alive (hung) or has exited without cleaning up.

    The clock runs from the newer of metrics.jsonl and train.pid, because a
    *restart* resets what "no progress" means. train.py rewrites its pidfile at
    startup and does not write metrics until its first interval, so after a
    crash the sequence is: supervisor backs off (up to 15 min), relaunches,
    torch.compile takes several minutes, first metrics line lands. Measured
    from metrics.jsonl alone that gap includes all the dead time before the new
    process even existed, so a late-attempt restart would trip a 30-minute
    threshold within seconds of starting -- and since a halt is now terminal
    (see `write_halt_marker`), that false positive would end the run rather
    than merely pause it. The pidfile mtime gives each attempt its own grace
    period, while a process that hangs *after* writing it still stalls on time.

    `since` is the same argument one level up: a watchdog can only vouch for
    time it was watching. Launchers start it *before* the trainer (they then
    block on the training subprocess), and `import torch` alone costs seconds,
    so the first poll reliably precedes the first pidfile write. Relaunching a
    job whose run dir already holds hours-old metrics -- `hero` attempt two --
    would otherwise report a stall against the previous attempt's silence
    within a second of starting, and now that a halt is terminal that is an
    ending rather than a blip. Passing the watchdog's own start time gives a
    fresh watch a full window to see progress; a hang that begins after it is
    still caught on time.
    """
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    now = time.time() if now is None else now
    last = os.path.getmtime(path)
    try:
        last = max(last, os.path.getmtime(os.path.join(run_dir, "train.pid")))
    except OSError:
        pass                                  # no pidfile: metrics alone it is
    if since is not None:
        last = max(last, since)
    gap = now - last
    if gap > max_gap_sec:
        return f"no metrics progress in {gap/60:.1f} min (threshold {max_gap_sec/60:.1f} min)"
    return None


# -------------------------------------------------------------- liveness ---

def read_pidfile(run_dir: str) -> Optional[int]:
    path = os.path.join(run_dir, "train.pid")
    if not os.path.exists(path):
        return None
    try:
        return int(open(path).read().strip())
    except (ValueError, OSError):
        return None


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    return True


def detect_crash(run_dir: str, records: list, target_tokens: Optional[int] = None) -> Optional[str]:
    """The pidfile is gone (or its process is dead) while the run hadn't
    reached its token target and no completion was recorded -- distinct
    from a stall (process still alive but stuck) or a clean finish."""
    pid = read_pidfile(run_dir)
    if pid is not None and is_process_alive(pid):
        return None
    if pid is None and not records:
        return None  # never started, or already cleaned up after a normal finish
    if target_tokens is not None and records:
        if records[-1].get("tokens", 0) >= target_tokens:
            return None  # reached target; pidfile is just gone because fit() cleaned it up
    if pid is not None:
        return f"training process (pid {pid}) is no longer running and the run did not reach its token target"
    return None


# ------------------------------------------------------------- completion ---

def detect_completion(records: list, target_tokens: Optional[int] = None) -> Optional[str]:
    if not records or target_tokens is None:
        return None
    tokens = records[-1].get("tokens", 0)
    if tokens >= target_tokens:
        return f"reached target: {tokens:,} / {target_tokens:,} tokens at step {records[-1].get('step')}"
    return None


# --------------------------------------------------------------- actions ---

def halt_training(pid: Optional[int], grace_sec: float = 30.0,
                  sleep_fn=time.sleep) -> bool:
    """SIGTERM the training process, escalate to SIGKILL after a grace
    period. Only ever targets the specific training PID -- never touches
    instance lifecycle (AGENT.md SS0.1, SS7)."""
    if pid is None or not is_process_alive(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    sleep_fn(grace_sec)
    if is_process_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def write_status_halt(status_path: str, run_name: str, reason: str,
                      remediation: Optional[str] = None,
                      finished: bool = False) -> str:
    """Append the operator-facing note. `finished=True` for a run that reached
    its token target: heading it "WATCHDOG HALT" reads, on a phone, as a job
    that was stopped -- and every arm of `abl-arch` and `hero` ends this way,
    so the false alarm would be the *normal* case."""
    heading = ("WATCHDOG: `%s` finished" % run_name if finished
               else "WATCHDOG HALT: `%s`" % run_name)
    note = (
        f"\n\n---\n\n"
        f"# {heading}\n\n"
        f"**Reason:** {reason}\n\n"
    )
    if remediation:
        note += f"**Suggested remediation (AGENT.md SS6, not auto-applied):** {remediation}\n\n"
    note += (
        "Observed by `watchdog.py`, which stopped watching this run. "
        if finished else
        "Training was halted by `watchdog.py`. "
    )
    note += (
        "The instance itself was **not** "
        "touched (AGENT.md SS0.1) -- it is idle, waiting for a human or the "
        "next agent session to decide how to proceed.\n"
    )
    existing = ""
    if os.path.exists(status_path):
        with open(status_path) as f:
            existing = f.read()
    with open(status_path, "w") as f:
        f.write(existing + note)
    return status_path


DIVERGENCE_REMEDIATION = ("stop, roll back two checkpoints, halve the Muon "
                          "learning rate, resume (AGENT.md SS6)")

HALT_MARKER = "watchdog-halt.json"


def halt_marker_path(run_dir: str) -> str:
    return os.path.join(run_dir, HALT_MARKER)


def write_halt_marker(run_dir: str, kind: str, reason: str) -> Optional[str]:
    """Record the halt where the *supervisor* will see it, not only where a
    human will.

    This is what makes a halt stick. `watchdog.py` stops the training process
    and exits; `hero.py` wraps train.py in `run_with_resume`, which sees a
    non-zero exit and restarts it from the last checkpoint. Without a marker
    the sequence on a divergence at hour 20 of `hero` is: watchdog SIGTERMs
    the trainer, appends "WATCHDOG HALT" to the *bottom* of STATUS.md, and
    exits -- then the supervisor resumes the diverged checkpoint, now with no
    watchdog running, and trains a broken model for the remaining three days
    before exiting 0. Nothing raises, the launcher reports "finished in 2
    attempt(s)", and ~$30 is spent on a model that had already diverged.

    Never raises: a marker that cannot be written must not take down a run
    that is otherwise fine. The cost of that fallback is a resume the operator
    has to catch by eye, which is exactly today's behaviour.
    """
    path = halt_marker_path(run_dir)
    payload = {"kind": kind, "reason": reason, "at": time.time()}
    try:
        os.makedirs(run_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)   # never leave a half-written marker
        return path
    except OSError as e:
        print(f"watchdog: could not write halt marker {path}: {e}")
        return None


def read_halt_marker(run_dir: str) -> Optional[dict]:
    try:
        with open(halt_marker_path(run_dir)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def clear_halt_marker(run_dir: str) -> bool:
    """Drop a previous halt so a deliberately relaunched run is not blocked by
    it. Only ever called by a launcher at startup -- never between a
    supervisor's own retries, which is the case the marker exists to stop."""
    try:
        os.remove(halt_marker_path(run_dir))
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- driver ---

def check_once(run_dir: str, run_name: str, status_path: str = "STATUS.md",
               target_tokens: Optional[int] = None, stall_sec: float = 1800.0,
               divergence_window: int = 20, divergence_factor: float = 2.0,
               now: Optional[float] = None, do_halt: bool = True,
               crash_is_terminal: bool = True,
               since: Optional[float] = None) -> Optional[dict]:
    """Runs one detection pass. Returns None if everything looks healthy (or
    there's nothing to check yet), else a dict describing what was found and
    what action was taken.

    `crash_is_terminal=False` is for a run under a supervisor that restarts
    crashes from the checkpoint (`hero.py`). A dead training PID is then a
    normal, recoverable event owned by the supervisor rather than something to
    report and stop watching for -- see `_cli`.
    """
    records = read_metrics(run_dir)

    completion = detect_completion(records, target_tokens)
    if completion:
        write_status_halt(status_path, run_name, f"run finished: {completion}",
                          finished=True)
        return {"kind": "completion", "reason": completion, "halted_pid": None}

    divergence = detect_divergence(records, divergence_window, divergence_factor)
    if divergence:
        pid = read_pidfile(run_dir)
        halted = halt_training(pid) if do_halt else False
        write_status_halt(status_path, run_name, divergence, DIVERGENCE_REMEDIATION)
        write_halt_marker(run_dir, "divergence", divergence)
        return {"kind": "divergence", "reason": divergence, "halted_pid": pid if halted else None}

    crash = detect_crash(run_dir, records, target_tokens)
    if crash:
        if not crash_is_terminal:
            # A supervisor owns this case: it will restart training from the
            # last checkpoint, and we keep polling so the *new* process is
            # watched too. Reporting it here as well would append a "WATCHDOG
            # HALT" to STATUS.md for something that self-heals in a minute.
            return {"kind": "crash_supervised", "reason": crash, "halted_pid": None}
        write_status_halt(status_path, run_name, crash)
        return {"kind": "crash", "reason": crash, "halted_pid": None}

    stall = detect_stall(run_dir, stall_sec, now, since=since)
    if stall:
        pid = read_pidfile(run_dir)
        halted = halt_training(pid) if do_halt else False
        write_status_halt(status_path, run_name, stall)
        write_halt_marker(run_dir, "stall", stall)
        return {"kind": "stall", "reason": stall, "halted_pid": pid if halted else None}

    return None


# --------------------------------------------------------------------- cli ---

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=os.environ.get("RUN_NAME", "train"))
    p.add_argument("--run-dir", default=None)
    p.add_argument("--status-path", default="STATUS.md")
    p.add_argument("--target-tokens", type=int, default=None)
    p.add_argument("--stall-min", type=float, default=30.0)
    p.add_argument("--poll-sec", type=float, default=120.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--supervised", action="store_true",
                   help="a supervisor (hero.py) restarts crashes from the "
                        "checkpoint. Keep watching across a restart instead of "
                        "exiting on a dead PID -- otherwise a recoverable crash "
                        "at hour 5 leaves the remaining three days unwatched.")
    args = p.parse_args()

    run_dir = args.run_dir or os.path.join("runs", args.run_name)
    print(f"watchdog: watching {run_dir} (stall threshold {args.stall_min} min"
          f"{', supervised' if args.supervised else ''})")

    started_at = time.time()
    reported = None
    while True:
        result = check_once(run_dir, args.run_name, args.status_path,
                            target_tokens=args.target_tokens,
                            stall_sec=args.stall_min * 60,
                            crash_is_terminal=not args.supervised,
                            since=started_at)
        if result:
            # A supervised crash repeats every poll until the supervisor's
            # backoff elapses (up to 15 min, i.e. ~7 passes). Say it once.
            if result["kind"] != "crash_supervised" or result["reason"] != reported:
                print(f"watchdog: {result['kind']} -- {result['reason']}")
                reported = result["reason"]
            if result["kind"] in ("divergence", "stall", "crash", "completion"):
                break
        elif reported is not None:
            print("watchdog: training is healthy again")
            reported = None
        if args.once:
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    _cli()
