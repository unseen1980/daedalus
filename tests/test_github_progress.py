"""Tests for the sanitized GitHub progress heartbeat."""

import json
import subprocess
from datetime import datetime, timezone


def test_public_snapshot_uses_a_strict_field_whitelist():
    from scripts.github_progress import build_public_snapshot

    state = {
        "schema": 1,
        "phase": "bootstrap",
        "status": "running",
        "started_at": "2026-08-24T00:00:00Z",
        "updated_at": "2026-08-24T01:00:00Z",
        "secret": "ghp_not-for-github",
        "details": {"endpoint": "private-host", "token": "private-token"},
    }
    metrics = {
        "step": 20,
        "tokens": 2_621_440,
        "loss": 3.5,
        "val_bpb": 0.91,
        "tok_per_sec": 42_000.0,
        "api_key": "private-token",
    }

    snapshot = build_public_snapshot(
        state,
        source_branch="vast/daedalus-improvements-20260824",
        source_sha="abc123",
        now=datetime(2026, 8, 24, 1, tzinfo=timezone.utc),
        metrics=metrics,
        gpu={"utilization_pct": 75, "memory_used_mb": 1024,
             "memory_total_mb": 24564, "serial": "private"},
    )

    encoded = json.dumps(snapshot)
    assert snapshot["phase"] == "bootstrap"
    assert snapshot["metrics"]["loss"] == 3.5
    assert snapshot["gpu"]["utilization_pct"] == 75
    assert "secret" not in encoded
    assert "private" not in encoded
    assert "endpoint" not in encoded
    assert "api_key" not in encoded


def test_write_snapshot_updates_public_files_and_timeline(tmp_path):
    from scripts.github_progress import write_snapshot

    snapshot = {
        "schema": 1,
        "phase": "bootstrap",
        "status": "running",
        "heartbeat_at": "2026-08-24T01:00:00Z",
        "source_branch": "vast/daedalus-improvements-20260824",
        "source_sha": "abc123",
        "metrics": {"loss": 3.5, "step": 20},
        "gpu": {"utilization_pct": 75},
    }

    write_snapshot(tmp_path, snapshot)

    assert json.loads((tmp_path / "status.json").read_text()) == snapshot
    assert json.loads((tmp_path / "recent-metrics.json").read_text()) == snapshot["metrics"]
    status = (tmp_path / "STATUS.md").read_text()
    assert "bootstrap" in status and "abc123" in status
    timeline = [json.loads(line) for line in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    assert timeline == [snapshot]
    assert list(tmp_path.glob("*.tmp")) == []


def test_commit_and_push_stages_only_progress_files(tmp_path):
    from scripts.github_progress import commit_and_push

    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1 if "--quiet" in command else 0)

    changed = commit_and_push(
        tmp_path,
        branch="vast/progress-20260824",
        message="progress: 2026-08-24T01:00:00Z",
        runner=runner,
    )

    assert changed is True
    assert calls[0][0] == [
        "git", "-C", str(tmp_path), "add", "--",
        "STATUS.md", "status.json", "recent-metrics.json", "timeline.jsonl",
    ]
    assert calls[-1][0] == [
        "git", "-C", str(tmp_path), "push", "origin",
        "HEAD:refs/heads/vast/progress-20260824",
    ]


def test_latest_metrics_ignores_a_torn_final_line(tmp_path):
    from scripts.github_progress import read_latest_jsonl

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 10, "loss": 4.0}\n{"step": 20, "loss": 3.5}\n{"step"')

    assert read_latest_jsonl(path) == {"step": 20, "loss": 3.5}


def test_publish_once_writes_and_commits_one_heartbeat(tmp_path):
    from scripts.github_progress import publish_once

    source = tmp_path / "source"
    progress = tmp_path / "progress"
    source.mkdir()
    state_path = tmp_path / "state.json"
    metrics_path = tmp_path / "metrics.jsonl"
    state_path.write_text(json.dumps({
        "schema": 1,
        "phase": "bootstrap",
        "status": "running",
        "started_at": "2026-08-24T00:00:00Z",
        "updated_at": "2026-08-24T01:00:00Z",
    }))
    metrics_path.write_text('{"step": 20, "loss": 3.5}\n')
    commits = []

    snapshot = publish_once(
        source_repo=source,
        progress_worktree=progress,
        state_path=state_path,
        metrics_path=metrics_path,
        progress_branch="vast/progress-20260824",
        now=datetime(2026, 8, 24, 1, tzinfo=timezone.utc),
        git_reader=lambda _, field: {
            "branch": "vast/daedalus-improvements-20260824",
            "sha": "abc123",
        }[field],
        gpu_reader=lambda: {"utilization_pct": 75},
        committer=lambda *args, **kwargs: commits.append((args, kwargs)) or True,
    )

    assert snapshot["source_sha"] == "abc123"
    assert json.loads((progress / "status.json").read_text()) == snapshot
    assert len(commits) == 1
    assert commits[0][1]["branch"] == "vast/progress-20260824"


def test_main_once_publishes_requested_paths(monkeypatch, tmp_path):
    from scripts import github_progress

    calls = []
    monkeypatch.setattr(
        github_progress,
        "publish_once",
        lambda **kwargs: calls.append(kwargs) or {"status": "running"},
    )

    result = github_progress.main([
        "--source-repo", str(tmp_path / "source"),
        "--progress-worktree", str(tmp_path / "progress"),
        "--state", str(tmp_path / "state.json"),
        "--metrics", str(tmp_path / "metrics.jsonl"),
        "--branch", "vast/progress-20260824",
        "--once",
    ])

    assert result == 0
    assert calls == [{
        "source_repo": str(tmp_path / "source"),
        "progress_worktree": str(tmp_path / "progress"),
        "state_path": str(tmp_path / "state.json"),
        "metrics_path": str(tmp_path / "metrics.jsonl"),
        "progress_branch": "vast/progress-20260824",
    }]

class TestDeadlineAndAttention:
    """A heartbeat has to say how much time is left and whether a human is needed."""

    def _state(self, **overrides) -> dict:
        state = {
            "schema": 1,
            "started_at": "2026-08-24T10:00:00Z",
            "updated_at": "2026-08-24T12:00:00Z",
            "hard_hours": 144.0,
            "reserve_hours": 8.0,
            "phase": "phase2-evaluation",
            "status": "running",
            "details": {},
        }
        state.update(overrides)
        return state

    def _at(self, hours: float):
        from datetime import datetime, timedelta, timezone

        return datetime(2026, 8, 24, 10, tzinfo=timezone.utc) + timedelta(hours=hours)

    def test_reports_elapsed_and_remaining_against_the_hard_deadline(self):
        from scripts.github_progress import build_deadline_view

        view = build_deadline_view(self._state(), self._at(10))

        assert view["stage"] == "active"
        assert view["elapsed_hours"] == 10.0
        assert view["remaining_hours"] == 134.0
        assert view["finalization_in_hours"] == 126.0

    def test_marks_the_reserve_and_the_expiry(self):
        from scripts.github_progress import build_deadline_view

        assert build_deadline_view(self._state(), self._at(137))["stage"] == "finalizing"
        assert build_deadline_view(self._state(), self._at(145))["stage"] == "expired"

    def test_a_state_without_a_start_time_reports_no_deadline(self):
        from scripts.github_progress import build_deadline_view

        assert build_deadline_view({"status": "running"}, self._at(1)) == {}

    def test_a_blocked_program_asks_for_a_human(self):
        from scripts.github_progress import build_attention_view

        view = build_attention_view(
            self._state(status="blocked", details={"blocker": "wrapper is stale"})
        )

        assert view == {"user_action_required": True, "blocker": "wrapper is stale"}

    def test_a_running_program_asks_for_nobody(self):
        from scripts.github_progress import build_attention_view

        assert build_attention_view(self._state())["user_action_required"] is False

    def test_a_blocker_summary_is_bounded_and_scrubbed(self):
        from scripts.github_progress import BLOCKER_SUMMARY_LIMIT, sanitize_blocker

        scrubbed = sanitize_blocker(
            {
                "summary": "reinstall /root/.config/daedalus/runtime.env using "
                "ghp_0123456789abcdef0123456789abcdef0123 " + "x" * 600
            }
        )

        assert "/root/" not in scrubbed
        assert "ghp_" not in scrubbed
        assert len(scrubbed) <= BLOCKER_SUMMARY_LIMIT + 3

    def test_the_status_page_leads_with_the_action_banner(self):
        from scripts.github_progress import _render_status, build_public_snapshot

        snapshot = build_public_snapshot(
            self._state(status="blocked", details={"blocker": "wrapper is stale"}),
            source_branch="vast/daedalus-improvements-20260824",
            source_sha="abc1234",
            now=self._at(10),
        )
        rendered = _render_status(snapshot)

        assert "> **Action required.** wrapper is stale" in rendered
        assert "Deadline stage: `active`" in rendered
