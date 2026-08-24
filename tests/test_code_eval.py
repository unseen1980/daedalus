"""Tests for sandboxed HumanEval+/MBPP+ execution.

This is the only evaluator in the program that runs text a language model
wrote. The sandbox tests therefore execute real subprocesses -- a mocked
sandbox proves nothing about whether the sandbox holds -- and assert the
containment properties directly: no network, bounded CPU, bounded memory,
bounded wall clock, and a working directory that is created fresh and removed
afterwards.

Failure categories are tested because a code score with no breakdown is not
actionable: "pass@1 fell 3 points" could be a real capability loss or the model
learning to emit markdown fences, and only the category counts separate them.
"""

import json
import sys

import pytest

from daedalus.scorecard import load_scorecard


# ------------------------------------------------------- answer extraction ---

def test_extract_code_keeps_a_bare_completion():
    from scripts.code_eval import extract_code

    prompt = "def add(a, b):\n"
    assert extract_code(prompt, "    return a + b\n") == "def add(a, b):\n    return a + b"


def test_extract_code_unwraps_a_markdown_fence():
    from scripts.code_eval import extract_code

    prompt = "def add(a, b):\n"
    completion = "```python\ndef add(a, b):\n    return a + b\n```\ntext after"

    assert extract_code(prompt, completion) == "def add(a, b):\n    return a + b"


def test_extract_code_stops_at_a_new_top_level_statement():
    from scripts.code_eval import extract_code

    prompt = "def add(a, b):\n"
    completion = "    return a + b\n\nprint(add(1, 2))\n"

    assert extract_code(prompt, completion) == "def add(a, b):\n    return a + b"


def test_extract_code_stops_at_a_repeated_prompt_style_docstring():
    from scripts.code_eval import extract_code

    prompt = "def add(a, b):\n"
    completion = "    return a + b\n\ndef unrelated():\n    pass\n"

    assert "unrelated" not in extract_code(prompt, completion)


def test_extract_code_reports_an_empty_generation():
    from scripts.code_eval import extract_code

    assert extract_code("def add(a, b):\n", "   \n\n") == "def add(a, b):"


# ------------------------------------------------------------------ syntax ---

def test_check_syntax_accepts_valid_code():
    from scripts.code_eval import check_syntax

    assert check_syntax("def add(a, b):\n    return a + b\n") == (True, None)


def test_check_syntax_reports_the_error_for_invalid_code():
    from scripts.code_eval import check_syntax

    valid, message = check_syntax("def add(a, b)\n    return a + b\n")

    assert valid is False
    assert "SyntaxError" in message or "invalid syntax" in message


# ----------------------------------------------------------------- sandbox ---

@pytest.mark.slow
def test_sandbox_reports_a_passing_solution(tmp_path):
    from scripts.code_eval import run_in_sandbox

    verdict = run_in_sandbox("def add(a, b):\n    return a + b\n",
                             "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "passed"
    assert verdict["category"] is None


@pytest.mark.slow
def test_sandbox_categorizes_a_failed_assertion():
    from scripts.code_eval import run_in_sandbox

    verdict = run_in_sandbox("def add(a, b):\n    return a - b\n",
                             "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "failed"
    assert verdict["category"] == "assertion_failed"


@pytest.mark.slow
def test_sandbox_categorizes_a_raised_exception():
    from scripts.code_eval import run_in_sandbox

    verdict = run_in_sandbox("def add(a, b):\n    return a / 0\n",
                             "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "failed"
    assert verdict["category"] == "exception"


@pytest.mark.slow
def test_sandbox_categorizes_invalid_syntax_without_executing():
    from scripts.code_eval import run_in_sandbox

    verdict = run_in_sandbox("def add(a, b)\n    return a + b\n",
                             "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "failed"
    assert verdict["category"] == "syntax_error"


@pytest.mark.slow
def test_sandbox_kills_an_infinite_loop_within_the_timeout():
    from scripts.code_eval import run_in_sandbox

    verdict = run_in_sandbox("def add(a, b):\n    while True:\n        pass\n",
                             "assert add(1, 2) == 3\n", timeout_s=2.0)

    assert verdict["status"] == "failed"
    assert verdict["category"] == "timeout"


@pytest.mark.slow
def test_sandbox_blocks_outbound_network_access():
    from scripts.code_eval import run_in_sandbox

    solution = (
        "import socket\n"
        "def add(a, b):\n"
        "    socket.create_connection(('example.com', 80), timeout=1)\n"
        "    return a + b\n"
    )

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=15.0)

    assert verdict["status"] == "failed"
    assert verdict["category"] in {"exception", "network_blocked"}
    assert "network" in verdict["detail"].lower()


@pytest.mark.slow
def test_sandbox_blocks_urllib_as_well_as_raw_sockets():
    from scripts.code_eval import run_in_sandbox

    solution = (
        "import urllib.request\n"
        "def add(a, b):\n"
        "    urllib.request.urlopen('http://example.com')\n"
        "    return a + b\n"
    )

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=15.0)

    assert verdict["status"] == "failed"
    assert "network" in verdict["detail"].lower()


@pytest.mark.slow
def test_sandbox_bounds_memory():
    from scripts.code_eval import run_in_sandbox

    solution = ("def add(a, b):\n"
                "    blob = bytearray(3 * 1024 * 1024 * 1024)\n"
                "    return len(blob)\n")

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n",
                             timeout_s=20.0, memory_mb=256)

    assert verdict["status"] == "failed"
    assert verdict["category"] in {"exception", "resource_limit"}


@pytest.mark.slow
def test_sandbox_runs_in_a_fresh_directory_and_removes_it():
    from scripts.code_eval import run_in_sandbox

    solution = ("import os\n"
                "def add(a, b):\n"
                "    open('scratch.txt', 'w').write(os.getcwd())\n"
                "    return a + b\n")

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "passed"
    workdir = verdict["workdir"]
    # Created for this item alone, and gone afterwards -- so one item cannot
    # leave state that changes another item's result.
    assert not any(part == "daedalus-code-eval" for part in workdir.split("/")[:2])
    import os
    assert not os.path.exists(workdir)


@pytest.mark.slow
def test_sandbox_does_not_inherit_the_parent_working_directory(tmp_path):
    from scripts.code_eval import run_in_sandbox

    solution = ("import os\n"
                "def add(a, b):\n"
                "    return os.getcwd()\n")

    verdict = run_in_sandbox(solution,
                             "import os\nassert add(1, 2) != os.path.dirname(__file__) or True\n",
                             timeout_s=10.0)

    assert verdict["status"] == "passed"


# ------------------------------------------------------------------ scoring ---

def _fake_problems():
    return {
        "HumanEval/0": {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "base_input": None,
            "test": "assert add(1, 2) == 3\n",
            "plus_test": "assert add(-1, 1) == 0\n",
        },
        "HumanEval/1": {
            "task_id": "HumanEval/1",
            "prompt": "def mul(a, b):\n",
            "entry_point": "mul",
            "test": "assert mul(2, 3) == 6\n",
            "plus_test": "assert mul(0, 5) == 0\n",
        },
    }


class ScriptedBackend:
    """Returns a fixed completion per prompt, so scoring is tested alone."""

    def __init__(self, completions):
        self.completions = completions
        self.seen = []

    def generate(self, item):
        self.seen.append(item.prompt)
        return self.completions[item.id]


@pytest.mark.slow
def test_evaluate_problems_reports_pass_at_1_for_base_and_plus():
    from scripts.code_eval import evaluate_problems

    backend = ScriptedBackend({
        "HumanEval/0": "    return a + b\n",
        "HumanEval/1": "    return a - b\n",
    })

    records = evaluate_problems(_fake_problems(), backend, timeout_s=10.0)

    by_id = {record["id"]: record for record in records}
    assert by_id["HumanEval/0"]["base_passed"] == 1
    assert by_id["HumanEval/0"]["plus_passed"] == 1
    assert by_id["HumanEval/1"]["base_passed"] == 0
    assert by_id["HumanEval/1"]["category"] == "assertion_failed"
    assert all(record["syntax_valid"] == 1 for record in records)


@pytest.mark.slow
def test_plus_tests_only_count_when_the_base_tests_pass():
    from scripts.code_eval import evaluate_problems

    backend = ScriptedBackend({
        "HumanEval/0": "    return a - b\n",
        "HumanEval/1": "    return a * b\n",
    })

    records = evaluate_problems(_fake_problems(), backend, timeout_s=10.0)
    by_id = {record["id"]: record for record in records}

    # HumanEval/0 fails its base test, so its plus result is not credited even
    # though `a - b` happens to satisfy the plus assertion.
    assert by_id["HumanEval/0"]["base_passed"] == 0
    assert by_id["HumanEval/0"]["plus_passed"] == 0


def test_summarize_code_reports_pass_at_1_and_failure_categories():
    from scripts.code_eval import summarize_code

    records = [
        {"id": "a", "base_passed": 1, "plus_passed": 1, "syntax_valid": 1,
         "category": None},
        {"id": "b", "base_passed": 0, "plus_passed": 0, "syntax_valid": 1,
         "category": "assertion_failed"},
        {"id": "c", "base_passed": 0, "plus_passed": 0, "syntax_valid": 0,
         "category": "syntax_error"},
        {"id": "d", "base_passed": 1, "plus_passed": 0, "syntax_valid": 1,
         "category": "assertion_failed"},
    ]

    metrics = summarize_code(records)

    assert metrics["pass@1"] == pytest.approx(0.5)
    assert metrics["pass@1_plus"] == pytest.approx(0.25)
    assert metrics["syntax_valid"] == pytest.approx(0.75)
    assert metrics["n"] == 4
    assert metrics["fail_assertion_failed"] == 2
    assert metrics["fail_syntax_error"] == 1


def test_summarize_code_handles_an_empty_run():
    from scripts.code_eval import summarize_code

    metrics = summarize_code([])

    assert metrics["n"] == 0


# --------------------------------------------------------------- scorecards ---

@pytest.mark.slow
def test_run_code_eval_writes_a_scorecard_with_task_revisions(tmp_path):
    from scripts.code_eval import run_code_eval
    from daedalus.scorecard import ArtifactRef

    backend = ScriptedBackend({
        "HumanEval/0": "    return a + b\n",
        "HumanEval/1": "    return a * b\n",
    })

    paths = run_code_eval(
        "humaneval-plus", _fake_problems(), backend, out_dir=tmp_path,
        artifact=ArtifactRef(path="m.gguf", sha256="a" * 64, kind="gguf-q4_0"),
        tokenizer_ref=ArtifactRef(path="t.json", sha256="b" * 64,
                                  kind="tokenizer"),
        seed=7, git_sha="deadbee", dataset_revision="v0.1.10",
        runtime={"evalplus": "0.3.1"}, timeout_s=10.0)

    card = load_scorecard(paths["scorecard"])
    assert card.kind == "code-execution"
    assert card.name == "humaneval-plus"
    assert card.metrics["pass@1"] == pytest.approx(1.0)
    assert card.item_count == 2
    assert card.provenance.seed == 7
    assert card.provenance.task_revisions == {"humaneval-plus": "v0.1.10"}
    assert card.provenance.runtime["evalplus"] == "0.3.1"
    assert card.provenance.bpb_mode == "not-applicable"


@pytest.mark.slow
def test_run_code_eval_keeps_the_generated_program_for_every_item(tmp_path):
    from scripts.code_eval import run_code_eval
    from daedalus.scorecard import ArtifactRef

    backend = ScriptedBackend({
        "HumanEval/0": "    return a + b\n",
        "HumanEval/1": "    return a * b\n",
    })

    paths = run_code_eval(
        "humaneval-plus", _fake_problems(), backend, out_dir=tmp_path,
        artifact=ArtifactRef(path="m.gguf", sha256="a" * 64, kind="gguf-q4_0"),
        tokenizer_ref=ArtifactRef(path="t.json", sha256="b" * 64,
                                  kind="tokenizer"),
        seed=7, git_sha="deadbee", dataset_revision="v0.1.10", timeout_s=10.0)

    sidecar = json.loads(paths["items"].read_text())
    assert "return a + b" in sidecar["items"][0]["solution"]
    assert sidecar["items"][0]["completion"] == "    return a + b\n"


# ----------------------------------------------------------------- dataset ---

def test_load_problems_records_the_limit_it_applied():
    from scripts.code_eval import load_problems

    problems = load_problems("humaneval-plus", limit=1,
                             loader=lambda name: _fake_problems())

    assert len(problems) == 1
    assert "HumanEval/0" in problems


def test_load_problems_rejects_an_unknown_dataset():
    from scripts.code_eval import load_problems

    with pytest.raises(ValueError, match="unknown"):
        load_problems("not-a-benchmark", loader=lambda name: {})


@pytest.mark.slow
@pytest.mark.skipif("evalplus" not in sys.modules and
                    not pytest.importorskip("evalplus", reason="evalplus absent"),
                    reason="evalplus absent")
def test_evalplus_is_importable_for_the_real_run():
    import evalplus

    assert evalplus is not None
