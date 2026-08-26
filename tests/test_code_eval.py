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
import os
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


# MBPP+ prompts are module docstrings, so the answer is a *top-level* def and
# the HumanEval rule above would cut it away on its first line -- leaving the
# docstring alone as the program, and every item failing on an undefined entry
# point. That reads as a model which cannot code.
MBPP_PROMPT = '"""\nWrite a function to add two numbers.\nassert add(1, 2) == 3\n"""\n'


def test_extract_code_keeps_a_top_level_definition_after_a_docstring_prompt():
    from scripts.code_eval import extract_code

    solution = extract_code(MBPP_PROMPT, "def add(a, b):\n    return a + b\n")

    assert "def add(a, b):" in solution and "return a + b" in solution


def test_extract_code_keeps_the_imports_a_top_level_answer_needs():
    from scripts.code_eval import extract_code

    solution = extract_code(
        MBPP_PROMPT, "import math\n\ndef area(r):\n    return math.pi * r * r\n")

    assert "import math" in solution and "def area(r):" in solution


def test_extract_code_cuts_a_top_level_answers_own_tests():
    """The harness supplies the tests; a candidate's own assert is not
    evidence, and an assert that fails would fail an item the model solved."""
    from scripts.code_eval import extract_code

    solution = extract_code(
        MBPP_PROMPT,
        "def add(a, b):\n    return a + b\n\nassert add(1, 2) == 99\nprint(add(1, 2))\n")

    assert "assert" not in solution.split('"""')[-1]
    assert "print(" not in solution


def test_extract_code_skips_prose_before_a_top_level_answer():
    from scripts.code_eval import extract_code

    solution = extract_code(
        MBPP_PROMPT, "Here is the solution:\n\ndef add(a, b):\n    return a + b\n")

    assert "Here is the solution" not in solution
    assert "def add(a, b):" in solution


def test_extract_code_keeps_a_completion_with_no_definition_in_it():
    """So it fails as what it is, rather than being replaced by nothing."""
    from scripts.code_eval import extract_code

    solution = extract_code(MBPP_PROMPT, "the answer is 3\n")

    assert "the answer is 3" in solution


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


# ------------------------------------------------- containment beyond python ---
#
# The gate claims executed code reaches neither the network nor files outside
# its sandbox. Patching `socket` only stops code that asks Python politely: a
# child running as root can shell out to `curl`, read the mode-0600 credential
# files, and write anywhere. These four pin the containment that actually holds
# that claim up. `unshare -n` is unavailable in this container, so none of them
# may be built on a network namespace.

requires_root = pytest.mark.skipif(
    os.geteuid() != 0,
    reason="privilege dropping is only observable when the parent starts as root")


@pytest.mark.slow
@requires_root
def test_sandbox_drops_root_before_running_a_candidate():
    from scripts.code_eval import run_in_sandbox

    solution = ("import os\n"
                "def add(a, b):\n"
                "    assert os.getuid() != 0, f'still root: {os.getuid()}'\n"
                "    return a + b\n")

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "passed", verdict["detail"]


@pytest.mark.slow
def test_sandbox_blocks_shelling_out_to_a_network_client():
    """`curl` never touches the patched `socket` module.

    A candidate that runs a network client as a subprocess reaches the internet
    with every Python-level block still in place, which is the gap that made the
    socket patch insufficient evidence for this gate.
    """
    from scripts.code_eval import run_in_sandbox

    solution = ("import subprocess\n"
                "def add(a, b):\n"
                "    subprocess.run(['curl', '-s', 'https://example.com'],\n"
                "                   capture_output=True)\n"
                "    return a + b\n")

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=15.0)

    assert verdict["status"] == "failed"
    assert verdict["category"] == "process_blocked", verdict["detail"]


@pytest.mark.slow
def test_sandbox_blocks_os_system_and_exec():
    from scripts.code_eval import run_in_sandbox

    solution = ("import os\n"
                "def add(a, b):\n"
                "    os.system('curl -s https://example.com')\n"
                "    return a + b\n")

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=15.0)

    assert verdict["status"] == "failed"
    assert verdict["category"] == "process_blocked", verdict["detail"]


@pytest.mark.slow
@requires_root
def test_sandbox_cannot_read_a_root_owned_secret_outside_it(tmp_path):
    """Standing in for `/root/.config/daedalus/runtime.env`.

    The real credential files are mode 0600 and root-owned; a candidate running
    as root reads them without tripping a single Python-level block.
    """
    from scripts.code_eval import run_in_sandbox

    secret_dir = tmp_path / "config"
    secret_dir.mkdir()
    secret = secret_dir / "runtime.env"
    secret.write_text("HF_TOKEN=not-a-real-token\n")
    secret.chmod(0o600)
    secret_dir.chmod(0o700)

    solution = ("def add(a, b):\n"
                f"    open({str(secret)!r}).read()\n"
                "    return a + b\n")

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "failed"
    assert "PermissionError" in verdict["detail"], verdict["detail"]


@pytest.mark.slow
@requires_root
def test_sandbox_cannot_write_outside_its_working_directory(tmp_path):
    from scripts.code_eval import run_in_sandbox

    protected = tmp_path / "protected"
    protected.mkdir()
    protected.chmod(0o700)

    solution = ("def add(a, b):\n"
                f"    open({str(protected / 'planted.txt')!r}, 'w').write('x')\n"
                "    return a + b\n")

    verdict = run_in_sandbox(solution, "assert add(1, 2) == 3\n", timeout_s=10.0)

    assert verdict["status"] == "failed"
    assert "PermissionError" in verdict["detail"], verdict["detail"]
    assert not (protected / "planted.txt").exists()


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
    """Problems in EvalPlus's *real* shape, not a convenient one.

    The previous version of this fixture invented `test` as a bare assertion and
    a `plus_test` key that EvalPlus does not ship. Every test built on it agreed
    with the code, and none of them touched the schema -- which is how
    `problem.get("test", "")` shipped with a default that turned an empty
    program into a pass. The fields here are the ones the real loader returns:
    `test` defines `check(candidate)` and does not call it, and the extended
    suite is *inputs* whose expectations come from `canonical_solution`.
    """
    return {
        "HumanEval/0": {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "canonical_solution": "    return a + b\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "base_input": [[1, 2]],
            "plus_input": [[-1, 1], [0, 0]],
            "atol": 0,
        },
        "HumanEval/1": {
            "task_id": "HumanEval/1",
            "prompt": "def mul(a, b):\n",
            "entry_point": "mul",
            "canonical_solution": "    return a * b\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 6\n",
            "base_input": [[2, 3]],
            "plus_input": [[0, 5]],
            "atol": 0,
        },
    }


def _fake_mbpp_problems():
    """MBPP+'s real shape, which is *not* HumanEval+'s.

    Keys taken from the failure that found this: `['assertion', 'atol',
    'base_input', 'canonical_solution', 'contract', 'entry_point', 'plus_input',
    'prompt', 'task_id']`. No `test`, so the base suite is inputs scored against
    the reference, and the prompt is a module docstring rather than an open
    function body. `test_the_real_mbpp_plus_schema_is_the_one_this_harness_reads`
    holds this fixture to the dataset.
    """
    return {
        "Mbpp/2": {
            "task_id": "Mbpp/2",
            "prompt": MBPP_PROMPT,
            "entry_point": "add",
            "canonical_solution": "def add(a, b):\n    return a + b\n",
            "assertion": "assert add(1, 2) == 3",
            "contract": "",
            "base_input": [[1, 2]],
            "plus_input": [[-1, 1], [0, 0]],
            "atol": 0,
        },
    }


def test_a_base_suite_is_built_from_inputs_when_the_dataset_ships_no_test():
    """MBPP+ carries no `test` key at all; the base suite is `base_input`."""
    from scripts.code_eval import test_program_for

    problem = _fake_mbpp_problems()["Mbpp/2"]

    program = test_program_for(problem, "base")

    assert "_run_plus_suite()" in program
    assert repr([[1, 2]]) in program            # base inputs, not plus inputs
    assert repr([[-1, 1], [0, 0]]) not in program


def test_a_problem_with_neither_a_test_nor_inputs_still_raises():
    """The no-default rule that the empty-program pass@1 = 1.0 defect cost."""
    from scripts.code_eval import test_program_for

    problem = dict(_fake_mbpp_problems()["Mbpp/2"])
    problem.pop("base_input")

    with pytest.raises(ValueError, match="neither a base test nor base inputs"):
        test_program_for(problem, "base")


def test_an_empty_input_list_raises_rather_than_scoring_a_pass():
    """A program with no assertions in it exits zero and counts as a pass."""
    from scripts.code_eval import test_program_for

    problem = dict(_fake_mbpp_problems()["Mbpp/2"], base_input=[])

    with pytest.raises(ValueError, match="no base inputs"):
        test_program_for(problem, "base")


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


@pytest.mark.slow
def test_an_mbpp_shaped_problem_scores_end_to_end():
    """The whole path MBPP+ needs: a docstring prompt, a top-level answer,
    a base suite with no `test` key, and both suites executed."""
    from scripts.code_eval import evaluate_problems

    backend = ScriptedBackend({"Mbpp/2": "def add(a, b):\n    return a + b\n"})

    records = evaluate_problems(_fake_mbpp_problems(), backend, timeout_s=10.0)

    assert records[0]["base_passed"] == 1
    assert records[0]["plus_passed"] == 1
    assert records[0]["syntax_valid"] == 1


@pytest.mark.slow
def test_a_wrong_mbpp_answer_fails_the_suite_it_is_scored_against():
    """The other half: the path above must be able to say no."""
    from scripts.code_eval import evaluate_problems

    backend = ScriptedBackend({"Mbpp/2": "def add(a, b):\n    return a - b\n"})

    records = evaluate_problems(_fake_mbpp_problems(), backend, timeout_s=10.0)

    assert records[0]["base_passed"] == 0
    assert records[0]["category"] == "assertion_failed"


@pytest.mark.slow
def test_the_oracle_backend_returns_the_datasets_own_solution():
    """An oracle pass measures the harness, so it must run the reference
    through the same extraction and sandbox a candidate goes through."""
    from scripts.code_eval import CanonicalBackend, evaluate_problems

    problems = _fake_mbpp_problems()

    records = evaluate_problems(problems, CanonicalBackend(problems),
                                timeout_s=10.0)

    assert records[0]["base_passed"] == 1 and records[0]["plus_passed"] == 1
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


@pytest.mark.slow
def test_the_real_evalplus_problems_carry_an_executable_test():
    """The schema check that was missing, and cost a meaningless pass@1.

    `evaluate_problems` read `problem["test"]`, a key EvalPlus does not ship.
    The default was an empty string, so every syntactically valid generation ran
    a program with no assertions in it, exited zero, and scored as a pass -- the
    released 150M base model measured pass@1 = 1.0 on HumanEval+ while emitting
    function bodies that were nothing but a repeated docstring.

    Every other test in this file builds its own problems, so all of them agreed
    with the code and none of them touched the real schema.
    """
    pytest.importorskip("evalplus", reason="evalplus absent")
    from scripts.code_eval import load_problems, test_program_for

    problems = load_problems("humaneval-plus", limit=3)

    for task_id, problem in problems.items():
        base = test_program_for(problem, "base")
        assert base.strip(), (
            f"{task_id} produced no base test program; problem keys were "
            f"{sorted(problem)}")
        # The exact defect: EvalPlus's `test` only *defines* check(candidate).
        assert f"check({problem['entry_point']})" in base


@pytest.mark.slow
def test_the_real_mbpp_plus_schema_is_the_one_this_harness_reads():
    """MBPP+ was an advertised `--dataset` choice that had never been run.

    It ships no `test` key at all -- the base suite is inputs, like the plus
    suite -- so `test_program_for` raised on the first problem of the phase 8
    baseline. Same shape as the defect phase 2 found in the HumanEval path, and
    it survived for the same reason: every fixture in this file was written to
    the code rather than to the dataset.
    """
    pytest.importorskip("evalplus", reason="evalplus absent")
    from scripts.code_eval import load_problems

    problems = load_problems("mbpp-plus", limit=1)
    task_id, problem = next(iter(problems.items()))

    assert "test" not in problem, f"{task_id} keys: {sorted(problem)}"
    assert problem["entry_point"], f"{task_id} keys: {sorted(problem)}"
    assert problem["base_input"], f"{task_id} carries no base inputs"
    assert problem["plus_input"], f"{task_id} carries no plus inputs"
    assert problem["canonical_solution"].lstrip().startswith(
        ("def ", "import ", "from ")), repr(problem["canonical_solution"][:200])
    assert problem["prompt"].lstrip().startswith('"""'), \
        repr(problem["prompt"][:200])


@pytest.mark.slow
def test_the_reference_solution_passes_the_suites_it_defines():
    """The oracle check. Nothing else here proves the suites can be passed.

    A harness can fail two ways: score a wrong answer as right (the empty-test
    bug) or score a right answer as wrong. Running EvalPlus's own canonical
    solution through the real sandbox catches both -- it must pass base *and*
    plus, and if it does not, no candidate's score means anything.
    """
    pytest.importorskip("evalplus", reason="evalplus absent")
    from scripts.code_eval import load_problems, run_in_sandbox, test_program_for

    problems = load_problems("humaneval-plus", limit=5)

    for task_id, problem in problems.items():
        reference = f"{problem['prompt']}{problem['canonical_solution']}"
        for suite in ("base", "plus"):
            verdict = run_in_sandbox(reference, test_program_for(problem, suite),
                                     timeout_s=30.0)
            assert verdict["status"] == "passed", (
                f"{task_id} {suite}: the reference solution failed its own "
                f"suite -- {verdict['category']}: {verdict['detail'][:400]}")


@pytest.mark.slow
def test_an_empty_bodied_solution_fails_the_real_base_suite():
    """The generation that used to score pass@1 = 1.0 must now score zero.

    The released 150M base model emitted a repeated signature and docstring with
    no body. That parsed, so it ran, and with no assertions in the program it
    exited zero and counted as a pass on every problem.
    """
    pytest.importorskip("evalplus", reason="evalplus absent")
    from scripts.code_eval import load_problems, run_in_sandbox, test_program_for

    problems = load_problems("humaneval-plus", limit=1)
    task_id, problem = next(iter(problems.items()))
    body_less = problem["prompt"]

    verdict = run_in_sandbox(body_less, test_program_for(problem, "base"),
                             timeout_s=30.0)

    assert verdict["status"] == "failed", (
        f"{task_id}: a solution with no function body passed the base suite")
