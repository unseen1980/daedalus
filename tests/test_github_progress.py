"""Tests for the sanitized GitHub progress heartbeat."""

import json
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