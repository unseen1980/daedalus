"""Tests for the approved wrapper's evaluation subcommands.

Every shell action in this program goes through `daedalus-approved`, so an
evaluation that cannot be launched through the wrapper cannot be run at all.
These commands are the launch surface, and the guard that matters most is the
one that keeps a run from writing into the released-artifact tree: those files
are the immutable baseline every gate in Phases 3-8 is measured against, and an
evaluation that overwrote one would invalidate the whole program silently.

Kept in its own file rather than added to test_vast_ops.py, which another
session was editing concurrently.
"""

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "ops/vast/run-approved"


@pytest.fixture
def environment(tmp_path):
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "activate").write_text("")
    runtime = tmp_path / "runtime.env"
    runtime.write_text("")
    values = os.environ.copy()
    values.update({
        "DAEDALUS_REPO": str(ROOT),
        "DAEDALUS_RUNTIME_ENV": str(runtime),
        "DAEDALUS_VENV": str(fake_venv),
        "DAEDALUS_RELEASED_ROOT": "/root/daedalus",
    })
    return values


def _run(environment, *args):
    return subprocess.run([str(WRAPPER), *args], capture_output=True, text=True,
                          env=environment)


def test_wrapper_is_valid_shell_after_the_evaluation_commands():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_exposes_every_evaluation_entry_point():
    text = WRAPPER.read_text()

    for command in ("eval-retrieval)", "eval-quant)", "eval-code)",
                    "eval-bpb)", "eval-tasks)"):
        assert command in text
    for script in ("scripts/retrieval_eval.py", "scripts/gguf_eval.py",
                   "scripts/code_eval.py", "scripts/bpb_eval.py", "eval.py"):
        assert script in text


def test_evaluation_commands_still_refuse_merging_and_force_pushes():
    text = WRAPPER.read_text()

    for forbidden in ("pr merge", "pr close", "push --force", "reset --hard"):
        assert forbidden not in text


@pytest.mark.slow
def test_eval_bpb_reaches_its_script(environment):
    result = _run(environment, "eval-bpb", "--help")

    assert result.returncode == 0, result.stderr
    assert "--holdout-root" in result.stdout


@pytest.mark.slow
def test_eval_retrieval_reaches_its_script(environment):
    result = _run(environment, "eval-retrieval", "--help")

    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout


@pytest.mark.slow
def test_a_released_artifact_may_be_read_as_input(environment):
    # The whole point of the released tree is to be scored. Reading from it must
    # stay ordinary; only writing into it is refused.
    result = _run(environment, "eval-retrieval", "--backend", "llama-cpp",
                  "--gguf", "/root/daedalus/gguf/hero-base-f16.gguf",
                  "--out-dir", "runs/eval/retrieval-base", "--help")

    assert result.returncode == 0, result.stderr
    assert "refusing" not in result.stderr


@pytest.mark.slow
def test_an_output_directory_inside_the_released_tree_is_refused(environment):
    result = _run(environment, "eval-retrieval", "--backend", "oracle",
                  "--out-dir", "/root/daedalus/gguf")

    assert result.returncode != 0
    assert "released" in result.stderr


@pytest.mark.slow
def test_an_equals_form_output_path_is_refused_too(environment):
    result = _run(environment, "eval-tasks",
                  "--out=/root/daedalus/final/hero/results.json")

    assert result.returncode != 0
    assert "released" in result.stderr


@pytest.mark.slow
def test_a_per_item_sidecar_inside_the_released_tree_is_refused(environment):
    result = _run(environment, "eval-tasks", "--per-item",
                  "/root/daedalus/final/hero/items.json")

    assert result.returncode != 0
    assert "released" in result.stderr


@pytest.mark.slow
def test_the_guard_does_not_fire_on_an_unrelated_absolute_output(environment):
    result = _run(environment, "eval-bpb", "--out-dir", "/tmp/eval-out", "--help")

    assert result.returncode == 0, result.stderr
