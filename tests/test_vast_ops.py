"""Tests for reproducible Vast supervisor configuration."""

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_wrappers_are_valid_and_pass_logging_argument():
    wrappers = [
        ROOT / "ops/vast/bootstrap.sh",
        ROOT / "ops/vast/daedalus_progress.sh",
        ROOT / "ops/vast/daedalus_resume.sh",
        ROOT / "ops/vast/run-approved",
    ]
    for wrapper in wrappers:
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
    for wrapper in wrappers[1:3]:
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


def test_phase_prompt_files_exist_and_do_not_embed_known_secret_paths():
    prompt_dir = ROOT / "ops/vast/prompts"
    prompts = sorted(prompt_dir.glob("*.md"))

    assert {path.name for path in prompts} >= {"phase1-control-plane.md", "phase2-evaluation.md"}
    for prompt in prompts:
        text = prompt.read_text()
        assert "--permission-mode dontAsk" in text
        assert "/Users/I335123" not in text
        assert "108.250.144.200" not in text
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in text


def test_approved_command_broker_exposes_only_safe_phase1_commands():
    wrapper = (ROOT / "ops/vast/run-approved").read_text()

    for command in ["format)", "phase)", "hash)", "safe-log)", "pr-draft)", "pr-edit)"]:
        assert command in wrapper
    for forbidden in ["pr merge", "pr close", "push --force", "reset --hard"]:
        assert forbidden not in wrapper


def test_safe_log_rejects_secret_and_parent_paths(tmp_path):
    wrapper = ROOT / "ops/vast/run-approved"
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "vast/daedalus-improvements-20260824"], cwd=repository, check=True)
    runtime = tmp_path / "runtime.env"
    runtime.write_text("")
    environment = os.environ.copy()
    environment.update({
        "DAEDALUS_REPO": str(repository),
        "DAEDALUS_RUNTIME_ENV": str(runtime),
    })

    result = subprocess.run(
        [str(wrapper), "safe-log", "../secret.log"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "refusing" in result.stderr


def test_safe_log_allows_portal_log_paths():
    wrapper = (ROOT / "ops/vast/run-approved").read_text()

    assert "/var/log/portal/*.log)" in wrapper


def test_phase_command_runs_controller_with_repository_on_pythonpath(tmp_path):
    wrapper = ROOT / "ops/vast/run-approved"
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "activate").write_text("")
    runtime = tmp_path / "runtime.env"
    runtime.write_text("")
    state = tmp_path / "state.json"
    environment = os.environ.copy()
    environment.update({
        "DAEDALUS_REPO": str(ROOT),
        "DAEDALUS_RUNTIME_ENV": str(runtime),
        "DAEDALUS_VENV": str(fake_venv),
    })

    result = subprocess.run(
        [str(wrapper), "phase", "--state", str(state), "init", "--phase", "bootstrap", "--status", "running"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert state.exists()


def test_pr_status_script_renders_draft_commands_without_merge_or_close():
    script = ROOT / "scripts/pr_status.py"
    subprocess.run(["python", "-m", "py_compile", str(script)], check=True)
    text = script.read_text()

    assert "--draft" in text
    assert "pr merge" not in text
    assert "pr close" not in text