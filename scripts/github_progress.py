"""Publish a sanitized program heartbeat to a dedicated GitHub branch."""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


STATE_FIELDS = ("schema", "phase", "status", "started_at", "updated_at", "lanes")
METRIC_FIELDS = (
    "step",
    "tokens",
    "loss",
    "val_bpb",
    "tok_per_sec",
    "qat_active",
    "qat_rel_rmse",
    "grad_norm",
    "peak_mem_GB",
)
GPU_FIELDS = ("utilization_pct", "memory_used_mb", "memory_total_mb")
PROGRESS_FILES = ("STATUS.md", "status.json", "recent-metrics.json", "timeline.jsonl")

#: Statuses that mean the program stopped advancing on its own.
ATTENTION_STATUSES = frozenset({"blocked", "halted", "failed"})

#: What `ops/vast/install_supervisor.sh` copies onto the box, and where. A
#: session calls the *installed* wrapper and supervisor runs the *installed*
#: service scripts, so a committed change to any of these is inert until an
#: operator reinstalls -- and nothing said so. `ops/vast/run-approved` gained
#: the `branch` command phase 8 step 1 needs, and the only symptom available to
#: a session was `unapproved command branch`: the box's last deliverable sat
#: blocked on a one-line operator step while the heartbeat read `passed`. The
#: quieter half is worse. A command that exists in both copies but whose
#: *behaviour* changed -- a new guard in `commit-push` -- fails nothing and
#: enforces nothing, and the repository's tests pass either way, because they
#: exercise the committed copy and sessions run the installed one.
INSTALLED_CONTROL_PLANE = (
    ("ops/vast/run-approved", "/usr/local/bin/daedalus-approved"),
    ("ops/vast/daedalus_progress.sh",
     "/opt/supervisor-scripts/daedalus_progress.sh"),
    ("ops/vast/daedalus_resume.sh",
     "/opt/supervisor-scripts/daedalus_resume.sh"),
    ("ops/vast/daedalus_session_keeper.sh",
     "/opt/supervisor-scripts/daedalus_session_keeper.sh"),
    ("ops/vast/supervisord.conf", "/etc/supervisor/conf.d/daedalus.conf"),
)

#: The one step that reconciles them, quoted verbatim in the banner so the
#: reader never has to find it.
INSTALL_COMMAND = "bash ops/vast/install_supervisor.sh"

#: A blocker summary is operator-written prose, so it is bounded and scrubbed
#: before reaching a branch anyone can read.
BLOCKER_SUMMARY_LIMIT = 400
_PROTECTED_PATH = re.compile(r"/root/\S*")
_SECRET_LIKE = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|[A-Za-z0-9_-]{40,})"
)


def sanitize_blocker(value) -> str:
    """A short, credential-free description of why the program needs a human."""

    if isinstance(value, dict):
        text = str(value.get("summary") or value.get("reason") or "")
    else:
        text = str(value or "")
    text = " ".join(text.split())
    text = _PROTECTED_PATH.sub("<protected-path>", text)
    text = _SECRET_LIKE.sub("<redacted>", text)
    if len(text) > BLOCKER_SUMMARY_LIMIT:
        text = text[:BLOCKER_SUMMARY_LIMIT].rstrip() + "..."
    return text


def _committed_bytes(repository, path: str, runner=subprocess.run):
    """`path` as HEAD holds it, or None when HEAD does not carry it.

    Never raises. This is a maintenance check riding along inside the
    heartbeat, and a heartbeat that stops publishing because `git` was briefly
    unavailable would trade the thing that matters for the thing that does not.
    """

    try:
        result = runner(["git", "-C", str(repository), "show", f"HEAD:{path}"],
                        capture_output=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    return result.stdout


def stale_installed_files(source_repo, *, pairs=INSTALLED_CONTROL_PLANE,
                          committed=_committed_bytes) -> list:
    """Which control-plane files on the box are not what HEAD says they are.

    Compared against **HEAD, not the working tree**, and that is the whole
    design. An uncommitted edit to the wrapper is a session's work in progress;
    reporting it would ask an operator to install code nobody has reviewed or
    pushed, which is precisely what an operator-only install step exists to
    prevent. A session that could make the heartbeat demand its own uncommitted
    wrapper would be one edit away from widening its own permissions. A
    committed change is pushed, and readable in the pull request, before anyone
    is asked to install it.

    A file HEAD does not carry is skipped: a checkout predating one of these is
    not a box that needs reinstalling. A file HEAD carries and the box does not
    *is* reported -- it has never been installed at all.
    """

    stale = []
    for repo_path, installed_path in pairs:
        want = committed(source_repo, repo_path)
        if want is None:
            continue
        try:
            have = Path(installed_path).read_bytes()
        except OSError:
            have = None
        if have != want:
            stale.append(repo_path)
    return sorted(stale)


def build_deadline_view(state: dict, now: datetime) -> dict:
    """Elapsed and remaining time against the hard deadline and its reserve."""

    started = state.get("started_at")
    if not started:
        return {}
    try:
        started_at = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
    except ValueError:
        return {}
    hard_hours = float(state.get("hard_hours", 144.0))
    reserve_hours = float(state.get("reserve_hours", 8.0))
    elapsed_hours = (now - started_at).total_seconds() / 3600.0
    finalizes_in = hard_hours - reserve_hours - elapsed_hours
    remaining_hours = hard_hours - elapsed_hours
    if remaining_hours <= 0:
        stage = "expired"
    elif finalizes_in <= 0:
        stage = "finalizing"
    else:
        stage = "active"
    return {
        "stage": stage,
        "elapsed_hours": round(elapsed_hours, 2),
        "remaining_hours": round(remaining_hours, 2),
        "finalization_in_hours": round(finalizes_in, 2),
        "hard_hours": hard_hours,
        "reserve_hours": reserve_hours,
    }


def build_attention_view(state: dict, *, stale_install=()) -> dict:
    """Whether a human has to act, and the scrubbed reason when one does.

    Side lanes count. A pass running beside the main phase fails the same ways
    the main phase does, and a heartbeat that reads `running` because the GPU
    run is fine, while the CPU lane beside it died hours ago, is exactly the
    silence this file exists to prevent.

    A stale installed control plane counts for the same reason, and it is the
    one blocker no session can clear for itself: the program keeps running,
    every phase reports `passed`, and the capability a phase needs is simply
    absent from the box. It is appended to a real blocker rather than
    replacing it -- a halted run outranks a reinstall -- but it is never
    dropped, because the two are usually the same stall seen from both ends.
    """

    status = str(state.get("status", ""))
    details = state.get("details") or {}
    blocker = sanitize_blocker(details.get("blocker"))
    required = status in ATTENTION_STATUSES or bool(
        details.get("user_action_required")
    )
    for name, lane in sorted((state.get("lanes") or {}).items()):
        lane = lane if isinstance(lane, dict) else {}
        lane_details = lane.get("details") or {}
        if str(lane.get("status", "")) not in ATTENTION_STATUSES and not \
                lane_details.get("user_action_required"):
            continue
        required = True
        if not blocker:
            blocker = sanitize_blocker(lane_details.get("blocker")) or (
                f"lane {name}: {lane.get('phase', 'unknown')} "
                f"{lane.get('status', 'unknown')}")
    if stale_install:
        required = True
        notice = (f"the box runs control-plane code HEAD has moved past "
                  f"({', '.join(stale_install)}); run {INSTALL_COMMAND} on the "
                  f"instance")
        lead = blocker.rstrip()
        if lead and not lead.endswith((".", "!", "?")):
            lead += "."
        blocker = f"{lead} Also: {notice}" if lead else notice
    return {"user_action_required": bool(required),
            "blocker": sanitize_blocker(blocker)}


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _select(values: Optional[dict], fields) -> dict:
    source = values or {}
    return {field: source[field] for field in fields if field in source}


def build_public_snapshot(
    state: dict,
    *,
    source_branch: str,
    source_sha: str,
    now: datetime,
    metrics: Optional[dict] = None,
    gpu: Optional[dict] = None,
    stale_install=(),
) -> dict:
    """Return only fields explicitly approved for the public progress branch."""
    stale_install = sorted(stale_install)
    snapshot = _select(state, STATE_FIELDS)
    snapshot.update({
        "heartbeat_at": _timestamp(now),
        "source_branch": source_branch,
        "source_sha": source_sha,
        "metrics": _select(metrics, METRIC_FIELDS),
        "gpu": _select(gpu, GPU_FIELDS),
        "deadline": build_deadline_view(state, now),
        "attention": build_attention_view(state, stale_install=stale_install),
        # Repository-relative paths of committed files, so this names what to
        # reinstall without publishing anything about the box's layout.
        "stale_control_plane": stale_install,
    })
    return snapshot


def read_latest_jsonl(path) -> dict:
    """Return the newest complete JSON object, ignoring a torn writer tail."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _render_deadline(deadline: dict) -> str:
    if not deadline:
        return ""
    return (
        f"- Elapsed: `{deadline['elapsed_hours']}h` of "
        f"`{deadline['hard_hours']}h`\n"
        f"- Finalization window opens in: "
        f"`{deadline['finalization_in_hours']}h`\n"
        f"- Deadline stage: `{deadline['stage']}`\n"
    )


def _render_lanes(lanes) -> str:
    """One line per lane running beside the main phase, or nothing.

    Nothing, not an empty section: for most of the program only the main lane
    runs, and a permanent empty heading trains a reader to skip the place where
    the second lane will appear.
    """

    if not isinstance(lanes, dict) or not lanes:
        return ""
    rows = "".join(
        f"- `{name}`: `{(lane or {}).get('phase', 'unknown')}` "
        f"({(lane or {}).get('status', 'unknown')})\n"
        for name, lane in sorted(lanes.items())
    )
    return f"\n## Lanes\n\n{rows}"


def _render_status(snapshot: dict) -> str:
    metrics = json.dumps(snapshot.get("metrics", {}), indent=2, sort_keys=True)
    gpu = json.dumps(snapshot.get("gpu", {}), indent=2, sort_keys=True)
    attention = snapshot.get("attention", {})
    banner = ""
    if attention.get("user_action_required"):
        reason = attention.get("blocker") or "see the phase timeline"
        banner = f"> **Action required.** {reason}\n\n"
    return (
        "# Daedalus Vast Program Status\n\n"
        f"{banner}"
        f"- Heartbeat: `{snapshot.get('heartbeat_at', 'unknown')}`\n"
        f"- Phase: `{snapshot.get('phase', 'unknown')}`\n"
        f"- Status: `{snapshot.get('status', 'unknown')}`\n"
        f"- Source: `{snapshot.get('source_branch', 'unknown')}` "
        f"at `{snapshot.get('source_sha', 'unknown')}`\n"
        f"{_render_deadline(snapshot.get('deadline', {}))}"
        f"{_render_lanes(snapshot.get('lanes'))}"
        "\n## Latest Metrics\n\n"
        f"```json\n{metrics}\n```\n\n"
        "## GPU\n\n"
        f"```json\n{gpu}\n```\n"
    )


def write_snapshot(worktree, snapshot: dict) -> None:
    """Atomically replace current views and append one immutable heartbeat."""
    root = Path(worktree)
    root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    _write_text_atomic(root / "status.json", encoded)
    _write_text_atomic(
        root / "recent-metrics.json",
        json.dumps(snapshot.get("metrics", {}), indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(root / "STATUS.md", _render_status(snapshot))
    with (root / "timeline.jsonl").open("a") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def commit_and_push(worktree, *, branch: str, message: str,
                    runner=subprocess.run) -> bool:
    """Commit only heartbeat artifacts and push only to a progress branch."""
    if not branch.startswith("vast/progress-"):
        raise ValueError(f"refusing non-progress branch {branch!r}")
    root = str(Path(worktree))
    runner(["git", "-C", root, "add", "--", *PROGRESS_FILES], check=True)
    diff = runner(["git", "-C", root, "diff", "--cached", "--quiet"],
                  check=False)
    if diff.returncode == 0:
        return False
    if diff.returncode != 1:
        raise subprocess.CalledProcessError(diff.returncode, diff.args)
    runner(["git", "-C", root, "commit", "-m", message], check=True)
    runner([
        "git", "-C", root, "push", "origin",
        f"HEAD:refs/heads/{branch}",
    ], check=True)
    return True


def _git_reader(repository, field: str) -> str:
    command = {
        "branch": ["branch", "--show-current"],
        "sha": ["rev-parse", "--short=12", "HEAD"],
    }[field]
    result = subprocess.run(
        ["git", "-C", str(repository), *command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _gpu_reader() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        utilization, memory_used, memory_total = (
            int(value.strip()) for value in result.stdout.splitlines()[0].split(",")
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    return {
        "utilization_pct": utilization,
        "memory_used_mb": memory_used,
        "memory_total_mb": memory_total,
    }


def publish_once(
    *,
    source_repo,
    progress_worktree,
    state_path,
    progress_branch: str,
    metrics_path=None,
    now: Optional[datetime] = None,
    git_reader=_git_reader,
    gpu_reader=_gpu_reader,
    install_reader=stale_installed_files,
    committer=commit_and_push,
) -> dict:
    """Publish one sanitized heartbeat and return the public snapshot."""
    with Path(state_path).open() as handle:
        state = json.load(handle)
    heartbeat_at = now or datetime.now(timezone.utc)
    snapshot = build_public_snapshot(
        state,
        source_branch=git_reader(source_repo, "branch"),
        source_sha=git_reader(source_repo, "sha"),
        now=heartbeat_at,
        metrics=read_latest_jsonl(metrics_path) if metrics_path else {},
        gpu=gpu_reader(),
        stale_install=install_reader(source_repo),
    )
    write_snapshot(progress_worktree, snapshot)
    committer(
        progress_worktree,
        branch=progress_branch,
        message=f"progress: {snapshot['heartbeat_at']}",
    )
    return snapshot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--progress-worktree", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--branch", default="vast/progress-20260824")
    parser.add_argument("--interval-sec", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    while True:
        succeeded = True
        try:
            publish_once(
                source_repo=args.source_repo,
                progress_worktree=args.progress_worktree,
                state_path=args.state,
                metrics_path=args.metrics,
                progress_branch=args.branch,
            )
        except Exception as error:  # noqa: BLE001 - service retries next interval
            succeeded = False
            print(f"github-progress: {type(error).__name__}", flush=True)
        if args.once:
            return 0 if succeeded else 1
        time.sleep(max(args.interval_sec, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())