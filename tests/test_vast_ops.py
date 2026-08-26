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


def test_reload_service_refuses_everything_outside_this_program(tmp_path):
    """`supervisorctl` also drives caddy, the portal and the tunnel manager --
    the instance's management and auth surface. Restarting one of those to
    reload a publisher is a way to lose the box."""
    wrapper = ROOT / "ops/vast/run-approved"
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "vast/daedalus-improvements-20260824"],
                   cwd=repository, check=True)
    runtime = tmp_path / "runtime.env"
    runtime.write_text("")
    environment = os.environ.copy()
    environment.update({"DAEDALUS_REPO": str(repository),
                        "DAEDALUS_RUNTIME_ENV": str(runtime)})

    for service in ("caddy", "instance_portal", "tunnel_manager", "", "all"):
        result = subprocess.run([str(wrapper), "reload-service", service],
                                capture_output=True, text=True, env=environment)
        assert result.returncode != 0
        assert "refusing to control service" in result.stderr

    wrapper_text = wrapper.read_text()
    assert "supervisorctl restart" in wrapper_text
    for forbidden in ("supervisorctl stop", "supervisorctl shutdown"):
        assert forbidden not in wrapper_text, (
            "the wrapper may reload this program's services, not stop them")


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


SOURCE_BRANCH = "vast/daedalus-improvements-20260824"
CODE_BRANCH = "vast/daedalus-code-20260824"


def _repo_on(tmp_path, branch, name="repo"):
    """A git repository with one commit, checked out on `branch`."""
    repository = tmp_path / name
    repository.mkdir()
    subprocess.run(["git", "init", "-q", "-b", branch], cwd=repository, check=True)
    (repository / "README.md").write_text("initial\n")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(["git", "-c", "user.email=t@e.st", "-c", "user.name=t",
                    "commit", "-qm", "initial"], cwd=repository, check=True)
    return repository


def _wrapper_env(repository, tmp_path):
    runtime = tmp_path / "runtime.env"
    runtime.write_text("")
    environment = os.environ.copy()
    environment.update({"DAEDALUS_REPO": str(repository),
                        "DAEDALUS_RUNTIME_ENV": str(runtime)})
    return environment


def _branch_command(repository, tmp_path, target):
    return subprocess.run([str(ROOT / "ops/vast/run-approved"), "branch", target],
                          capture_output=True, text=True,
                          env=_wrapper_env(repository, tmp_path))


def test_the_code_branch_is_created_from_the_optimization_branchs_tested_sha(
        tmp_path):
    """Phase 8 step 1. Everything downstream already works from whichever source
    branch the checkout is on; the missing capability was getting there, and a
    session cannot run `git checkout` itself. The printed SHA is the parent the
    code run manifest has to record."""
    repository = _repo_on(tmp_path, SOURCE_BRANCH)
    parent = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            capture_output=True, text=True, check=True).stdout.strip()

    created = _branch_command(repository, tmp_path, CODE_BRANCH)

    assert created.returncode == 0, created.stderr
    assert created.stdout.strip() == parent
    assert subprocess.run(["git", "branch", "--show-current"], cwd=repository,
                          capture_output=True, text=True,
                          check=True).stdout.strip() == CODE_BRANCH
    # Idempotent: a session that is already there gets the SHA, not a second
    # branch or an error.
    again = _branch_command(repository, tmp_path, CODE_BRANCH)
    assert again.returncode == 0 and again.stdout.strip() == parent
    # And back, because a shared fix belongs on the optimization branch first.
    back = _branch_command(repository, tmp_path, SOURCE_BRANCH)
    assert back.returncode == 0, back.stderr
    assert subprocess.run(["git", "branch", "--show-current"], cwd=repository,
                          capture_output=True, text=True,
                          check=True).stdout.strip() == SOURCE_BRANCH


def test_a_code_branch_already_on_the_remote_is_continued_not_forked(tmp_path):
    """A checkout that has lost the local branch -- or never had it -- must
    continue the pushed one. Branching from here instead forks the code branch
    at whatever the optimization branch has reached, and nothing says so until
    a push is rejected, by which time the work is committed onto it."""
    repository = _repo_on(tmp_path, SOURCE_BRANCH)
    subprocess.run(["git", "checkout", "-q", "-b", "pushed-earlier"],
                   cwd=repository, check=True)
    (repository / "codeprep.py").write_text("# earlier code-branch work\n")
    subprocess.run(["git", "add", "codeprep.py"], cwd=repository, check=True)
    subprocess.run(["git", "-c", "user.email=t@e.st", "-c", "user.name=t",
                    "commit", "-qm", "earlier"], cwd=repository, check=True)
    remote_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                capture_output=True, text=True,
                                check=True).stdout.strip()
    subprocess.run(["git", "update-ref", f"refs/remotes/origin/{CODE_BRANCH}",
                    remote_sha], cwd=repository, check=True)
    subprocess.run(["git", "checkout", "-q", SOURCE_BRANCH], cwd=repository, check=True)
    subprocess.run(["git", "branch", "-qD", "pushed-earlier"], cwd=repository, check=True)

    result = _branch_command(repository, tmp_path, CODE_BRANCH)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == remote_sha
    assert (repository / "codeprep.py").exists()


def test_only_the_two_source_branches_can_be_switched_to(tmp_path):
    repository = _repo_on(tmp_path, SOURCE_BRANCH)

    for target in ("main", "vast/progress-20260824", "feature/anything", ""):
        result = _branch_command(repository, tmp_path, target)
        assert result.returncode != 0
        assert "refusing branch" in result.stderr


def test_the_optimization_branch_is_never_created_only_switched_to(tmp_path):
    """It exists and is what every other branch is built on, so "create it" is
    never the right answer to it being missing -- an empty one would silently
    become the base of the stacked pull request."""
    repository = _repo_on(tmp_path, CODE_BRANCH)

    result = _branch_command(repository, tmp_path, SOURCE_BRANCH)

    assert result.returncode != 0
    assert f"refusing to create {SOURCE_BRANCH}" in result.stderr


def test_switching_with_modified_tracked_files_is_refused(tmp_path):
    """A modified tracked file follows the checkout, which is how work meant for
    one branch lands in a commit on the other."""
    repository = _repo_on(tmp_path, SOURCE_BRANCH)
    (repository / "README.md").write_text("edited but not committed\n")

    result = _branch_command(repository, tmp_path, CODE_BRANCH)

    assert result.returncode != 0
    assert "modified tracked files" in result.stderr
    # Untracked artifacts are not the same thing: every phase leaves them, and
    # refusing on those would mean never switching at all.
    subprocess.run(["git", "checkout", "--", "README.md"], cwd=repository, check=True)
    (repository / "runs").mkdir()
    (repository / "runs" / "metrics.jsonl").write_text("{}\n")
    assert _branch_command(repository, tmp_path, CODE_BRANCH).returncode == 0


def test_a_branch_switch_cannot_reach_the_default_branch(tmp_path):
    """From `main`, the wrapper refuses before it can move anything -- the same
    guard `commit-push` uses, for the same reason."""
    repository = _repo_on(tmp_path, "main")

    result = _branch_command(repository, tmp_path, CODE_BRANCH)

    assert result.returncode != 0
    assert "refusing branch main" in result.stderr


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


def test_pr_find_is_a_read_and_is_how_a_session_learns_the_number():
    """`pr-edit` needs a number no engineering session could reach -- the PR is
    opened by the controller and its URL lives on the progress branch -- so a
    current body kept being written and never applied. The lookup has to be a
    GET: the alternative to guessing a number must not itself be able to open or
    retarget a pull request."""

    wrapper = (ROOT / "ops/vast/run-approved").read_text()

    assert "    pr-find)" in wrapper
    section = wrapper.split("    pr-find)", 1)[1].split(";;", 1)[0]
    assert "gh api" in section
    assert "select(.head.ref ==" in section
    for forbidden in ("-X PATCH", "-X POST", "-X PUT", "-X DELETE", "pr create"):
        assert forbidden not in section, (
            f"pr-find must be read-only; found {forbidden!r}")


def test_pr_find_refuses_to_answer_from_a_branch_it_may_not_touch(tmp_path):
    wrapper = ROOT / "ops/vast/run-approved"
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repository,
                   check=True)
    runtime = tmp_path / "runtime.env"
    runtime.write_text("")
    environment = os.environ.copy()
    environment.update({"DAEDALUS_REPO": str(repository),
                        "DAEDALUS_RUNTIME_ENV": str(runtime)})

    result = subprocess.run([str(wrapper), "pr-find"], capture_output=True,
                            text=True, env=environment)

    assert result.returncode != 0
    assert "refusing branch main" in result.stderr


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
    # Five sessions kept pr-body.md current and could not apply it.
    assert "daedalus-approved pr-find" in runbook


def test_phase_four_prompt_compares_bpb_not_token_perplexity():
    prompt = (ROOT / "ops/vast/prompts/phase4-tokenizer-lab.md").read_text()

    # Token-level perplexity is not comparable across vocabularies.
    assert "never token-level perplexity" in prompt
    assert "24,576" in prompt and "32,768" in prompt and "40,960" in prompt
    assert "do not adjust it after seeing the numbers" in prompt
    assert "/usr/local/bin/daedalus-approved" in prompt


def test_session_keeper_bounds_a_turn_on_silence_as_well_as_lifetime():
    wrapper = (ROOT / "ops/vast/daedalus_session_keeper.sh").read_text()
    cli = (ROOT / "scripts/session_keeper.py").read_text()

    assert "--idle-timeout-sec" in wrapper
    # Liveness is filesystem activity: an hour of training changes no git state.
    assert "filesystem_activity_probe" in cli


def test_phase_five_prompt_requires_functional_ablation():
    prompt = (ROOT / "ops/vast/prompts/phase5-conv-health.md").read_text()

    # Norms above a threshold are not evidence a channel carries signal.
    assert "matched functional ablation" in prompt
    assert "positive control" in prompt
    assert "does not revive the released model" in prompt
    assert "/usr/local/bin/daedalus-approved" in prompt


def test_phase_seven_prompt_targets_the_successor_budget():
    prompt = (ROOT / "ops/vast/prompts/phase7-corpus.md").read_text()

    # A single assumed budget would be wrong; report a curve instead.
    assert "curve across budgets" in prompt
    assert "200B" in prompt
    assert "shortfall" in prompt
    assert "/usr/local/bin/daedalus-approved" in prompt


def test_phase_six_prompt_states_proxy_limits_and_the_target():
    prompt = (ROOT / "ops/vast/prompts/phase6-architecture.md").read_text()

    # No successor size is decided; report a Pareto set, not one configuration.
    assert "no successor size is decided" in prompt.lower()
    assert "not a configuration to copy" in prompt
    assert "KV bytes per context token" in prompt
    assert "/usr/local/bin/daedalus-approved" in prompt
