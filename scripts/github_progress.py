"""Publish a sanitized program heartbeat to a dedicated GitHub branch."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


STATE_FIELDS = ("schema", "phase", "status", "started_at", "updated_at")
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
) -> dict:
    """Return only fields explicitly approved for the public progress branch."""
    snapshot = _select(state, STATE_FIELDS)
    snapshot.update({
        "heartbeat_at": _timestamp(now),
        "source_branch": source_branch,
        "source_sha": source_sha,
        "metrics": _select(metrics, METRIC_FIELDS),
        "gpu": _select(gpu, GPU_FIELDS),
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


def _render_status(snapshot: dict) -> str:
    metrics = json.dumps(snapshot.get("metrics", {}), indent=2, sort_keys=True)
    gpu = json.dumps(snapshot.get("gpu", {}), indent=2, sort_keys=True)
    return (
        "# Daedalus Vast Program Status\n\n"
        f"- Heartbeat: `{snapshot.get('heartbeat_at', 'unknown')}`\n"
        f"- Phase: `{snapshot.get('phase', 'unknown')}`\n"
        f"- Status: `{snapshot.get('status', 'unknown')}`\n"
        f"- Source: `{snapshot.get('source_branch', 'unknown')}` "
        f"at `{snapshot.get('source_sha', 'unknown')}`\n\n"
        "## Latest Metrics\n\n"
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