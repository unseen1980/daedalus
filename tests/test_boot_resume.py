"""Tests for restarting an approved in-flight run after a box reboot."""

import json


def _write_marker(tmp_path, **overrides):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_text("checkpoint")
    marker = {
        "schema": 1,
        "run_name": "boot-test",
        "run_dir": str(run_dir),
        "cwd": str(tmp_path),
        "cmd": ["python", "train.py", "--run-name", "boot-test"],
        "ckpt_path": str(checkpoint),
        "boot_resumes": 0,
        "completed": False,
        "max_attempts": 3,
        "backoff_sec": 1.0,
        "max_backoff_sec": 2.0,
        "halt_marker": None,
    }
    marker.update(overrides)
    (run_dir / "inflight.json").write_text(json.dumps(marker))
    return run_dir, marker


def test_incomplete_run_forces_resume_on_the_first_attempt(tmp_path):
    from scripts import boot_resume

    run_dir, marker = _write_marker(tmp_path)
    checkpoint = run_dir / "checkpoint.pt"

    calls = []

    def run(cmd, ckpt_path, **kwargs):
        calls.append((cmd, ckpt_path, kwargs))
        return {"attempts": 1, "resumed": True, "returncodes": [0]}

    result = boot_resume.resume_run(
        str(run_dir),
        run=run,
        supervisor_live=lambda _: False,
        trainer_live=lambda _: False,
    )

    assert result["status"] == "completed"
    assert calls[0][0] == marker["cmd"]
    assert calls[0][1] == str(checkpoint)
    assert calls[0][2]["force_resume"] is True
    assert json.loads((run_dir / "inflight.json").read_text())["boot_resumes"] == 1


def test_existing_watchdog_halt_prevents_child_launch(tmp_path):
    from scripts import boot_resume

    halt = tmp_path / "halt.json"
    halt.write_text(json.dumps({"kind": "divergence", "reason": "loss diverged"}))
    run_dir, _ = _write_marker(tmp_path, halt_marker=str(halt))
    calls = []

    result = boot_resume.resume_run(
        str(run_dir),
        run=lambda *args, **kwargs: calls.append((args, kwargs)),
        supervisor_live=lambda _: False,
        trainer_live=lambda _: False,
    )

    assert result == {"status": "skipped", "reason": "watchdog_halt"}
    assert calls == []


def test_live_supervisor_is_never_duplicated(tmp_path):
    from scripts import boot_resume

    run_dir, _ = _write_marker(tmp_path)
    calls = []

    result = boot_resume.resume_run(
        str(run_dir),
        run=lambda *args, **kwargs: calls.append((args, kwargs)),
        supervisor_live=lambda _: True,
        trainer_live=lambda _: False,
    )

    assert result == {"status": "skipped", "reason": "supervisor_live_or_unknown"}
    assert calls == []


def test_boot_resume_limit_prevents_restart(tmp_path):
    from scripts import boot_resume

    run_dir, _ = _write_marker(tmp_path, boot_resumes=5)
    calls = []

    result = boot_resume.resume_run(
        str(run_dir),
        run=lambda *args, **kwargs: calls.append((args, kwargs)),
        supervisor_live=lambda _: False,
        trainer_live=lambda _: False,
    )

    assert result == {"status": "skipped", "reason": "boot_resume_limit"}
    assert calls == []


def test_multiple_pending_runs_are_blocked_without_launching(tmp_path):
    from scripts import boot_resume

    first = tmp_path / "first"
    second = tmp_path / "second"
    for run_dir in (first, second):
        run_dir.mkdir()
        checkpoint = run_dir / "checkpoint.pt"
        checkpoint.write_text("checkpoint")
        (run_dir / "inflight.json").write_text(json.dumps({
            "schema": 1,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "cmd": ["python", "train.py", "--run-name", run_dir.name],
            "ckpt_path": str(checkpoint),
            "completed": False,
        }))
    calls = []

    result = boot_resume.resume_pending(
        str(tmp_path),
        resumer=lambda run_dir: calls.append(run_dir),
    )

    assert result == {
        "status": "blocked",
        "reason": "multiple_pending_runs",
        "count": 2,
    }
    assert calls == []


def test_main_dry_run_reports_candidate_without_mutating(tmp_path, capsys):
    from scripts import boot_resume

    run_dir, _ = _write_marker(tmp_path)

    result = boot_resume.main([
        "--runs-root", str(tmp_path),
        "--dry-run",
    ])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output == {
        "status": "dry_run",
        "candidate": str(run_dir),
    }
    assert json.loads((run_dir / "inflight.json").read_text())["boot_resumes"] == 0