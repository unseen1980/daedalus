"""Tests for reproducible Vast supervisor configuration."""

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

    assert "[program:daedalus_progress]" in config
    assert "command=/opt/supervisor-scripts/daedalus_progress.sh" in config
    assert "autorestart=unexpected" in config
    assert "[program:daedalus_resume]" in config
    assert "command=/opt/supervisor-scripts/daedalus_resume.sh" in config
    assert "autorestart=false" in config