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


def test_progress_once_command_uses_github_progress_publisher():
    wrapper = (ROOT / "ops/vast/run-approved").read_text()

    assert "progress-once)" in wrapper
    assert "scripts/github_progress.py" in wrapper
    assert "--once" in wrapper
    assert "DAEDALUS_PROGRESS_WORKTREE" in wrapper


def test_pr_status_script_renders_draft_commands_without_merge_or_close():
    script = ROOT / "scripts/pr_status.py"
    subprocess.run(["python", "-m", "py_compile", str(script)], check=True)
    text = script.read_text()

    assert "--draft" in text
    assert "pr merge" not in text
    assert "pr close" not in text

def test_session_keeper_is_supervised_and_never_spins_on_a_hard_blocker():
    wrapper = ROOT / "ops/vast/daedalus_session_keeper.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text()
    config = (ROOT / "ops/vast/supervisord.conf").read_text()
    installer = (ROOT / "ops/vast/install_supervisor.sh").read_text()

    assert '. "${utils}/logging.sh" ""' in text
    assert "scripts/session_keeper.py" in text
    # The engineering session gets the Claude token only; runtime credentials
    # stay with the approved wrapper.
    assert "claude.env" in text
    assert "runtime.env" not in text
    assert "[program:daedalus_session_keeper]" in config
    assert "command=/opt/supervisor-scripts/daedalus_session_keeper.sh" in config
    assert "exitcodes=0,1" in config
    assert "supervisorctl status daedalus_session_keeper" in installer


def test_standing_engineering_prompt_carries_the_repair_contract():
    prompt = (ROOT / "ops/vast/prompts/default.md").read_text()

    assert "/usr/local/bin/daedalus-approved" in prompt
    assert "focused" in prompt
    assert "Do not ask" in prompt


def test_phase_three_prompt_preregisters_its_gates():
    prompt = (ROOT / "ops/vast/prompts/phase3-qat-recovery.md").read_text()

    assert "--init-from" in prompt and "never" in prompt
    assert "qat_frac=1.0" in prompt
    assert "50% reduction" in prompt
    assert "Do not tune a threshold after seeing an outcome." in prompt
    assert "/usr/local/bin/daedalus-approved" in prompt


def test_pr_edit_uses_the_rest_endpoint_the_run_token_can_reach():
    wrapper = (ROOT / "ops/vast/run-approved").read_text()

    # gh pr edit resolves org fields and fails with only the repo scope.
    assert "gh pr edit" not in wrapper
    assert "gh api -X PATCH" in wrapper
    assert "/pulls/${number}" in wrapper


def test_evaluation_entry_points_are_reachable_through_the_wrapper():
    wrapper = (ROOT / "ops/vast/run-approved").read_text()

    for subcommand in (
        "eval-retrieval",
        "eval-quant",
        "eval-code",
        "eval-bpb",
        "eval-tasks",
    ):
        assert f"    {subcommand})" in wrapper


def test_session_keeper_verifies_both_plans_before_every_launch():
    wrapper = (ROOT / "ops/vast/daedalus_session_keeper.sh").read_text()
    cli = (ROOT / "scripts/session_keeper.py").read_text()

    assert "--plan-hashes" in wrapper and "plan-hashes.txt" in wrapper
    assert "--plan-context" in wrapper and "claude-plan-context.md" in wrapper
    assert "plan verification failed" in cli


def test_session_keeper_yields_the_box_to_supervised_jobs():
    wrapper = (ROOT / "ops/vast/daedalus_session_keeper.sh").read_text()

    assert "--runs-root" in wrapper
    assert "--busy-poll-sec" in wrapper
    # Evaluation and training turns outlast a code slice.
    # A phase 3 turn drives three one-hour arms plus scoring.
    assert "DAEDALUS_SESSION_TIMEOUT_SEC:-28800" in wrapper


def test_finalization_prompt_forbids_new_experiments_and_merges():
    prompt = (ROOT / "ops/vast/prompts/phase9-finalization.md").read_text()

    assert "No new experiment may start." in prompt
    assert "Never merge." in prompt
    assert "immutable final artifacts" in prompt
    assert "/usr/local/bin/daedalus-approved" in prompt


def test_session_keeper_refuses_to_run_beside_an_orphaned_session():
    wrapper = (ROOT / "ops/vast/daedalus_session_keeper.sh").read_text()

    # A keeper that dies leaves its session running; a restart must not add a
    # second writer to the same working tree.
    assert 'pgrep -f "^${claude_bin} -p"' in wrapper
    assert "waiting" in wrapper


def test_runbook_covers_the_traps_an_operator_will_hit():
    runbook = (ROOT / "ops/vast/RUNBOOK.md").read_text()

    # Each of these has already cost the program time.
    assert "Supervisord treats that exit as expected" in runbook
    assert "supervisorctl start daedalus_session_keeper" in runbook
    assert "install_supervisor.sh" in runbook
    assert "Operation not permitted" in runbook


def test_phase_four_prompt_compares_bpb_not_token_perplexity():
    prompt = (ROOT / "ops/vast/prompts/phase4-tokenizer-lab.md").read_text()

    # Token-level perplexity is not comparable across vocabularies.
    assert "never token-level perplexity" in prompt
    assert "24,576" in prompt and "32,768" in prompt and "40,960" in prompt
    assert "do not adjust it after seeing the numbers" in prompt
    assert "/usr/local/bin/daedalus-approved" in prompt
