"""Tests for reproducible Vast supervisor configuration."""

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_wrappers_are_valid_and_pass_logging_argument():
    wrappers = [
        ROOT / "ops/vast/daedalus_progress.sh",
        ROOT / "ops/vast/daedalus_resume.sh",
    ]
    for wrapper in wrappers:
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        assert '. "${utils}/logging.sh" ""' in wrapper.read_text()


def test_supervisor_config_keeps_progress_alive_and_resume_one_shot():
    config = (ROOT / "ops/vast/supervisord.conf").read_text()
    installer = (ROOT / "ops/vast/install_supervisor.sh").read_text()

    assert "[program:daedalus_progress]" in config
    assert "command=/opt/supervisor-scripts/daedalus_progress.sh" in config
    assert "autorestart=unexpected" in config
    assert "[program:daedalus_resume]" in config
    assert "command=/opt/supervisor-scripts/daedalus_resume.sh" in config
    assert "autorestart=false" in config
    assert "supervisorctl status daedalus_progress" in installer
    assert "supervisorctl status daedalus_resume || true" in installer


def test_approved_command_broker_refuses_commit_from_main(tmp_path):
    wrapper = ROOT / "ops/vast/run-approved"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    installer = (ROOT / "ops/vast/install_supervisor.sh").read_text()
    assert "/usr/local/bin/daedalus-approved" in installer
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repository, check=True)
    (repository / "README.md").write_text("initial\n")
    runtime = tmp_path / "runtime.env"
    runtime.write_text("")
    environment = os.environ.copy()
    environment.update({
        "DAEDALUS_REPO": str(repository),
        "DAEDALUS_RUNTIME_ENV": str(runtime),
    })

    result = subprocess.run(
        [
            str(wrapper), "commit-push",
            "--message", "test: forbidden",
            "--", "README.md",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "refusing branch" in result.stderr