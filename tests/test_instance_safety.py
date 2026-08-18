"""The one rule with no exceptions: this agent never terminates the box.

AGENT.md SS0.1 and SS7 are explicit -- teardown is operator-controlled, and the
rule holds on job completion, on unrecoverable failure, and on an idle box
burning money. `VAST_API_KEY` is deliberately not provided for this reason.

Nothing enforced that. It held by absence, which is exactly the kind of
invariant that survives right up until someone adds a well-meaning "clean up
when we're done" call. `watchdog.py` is the obvious place for it to creep in:
its whole job is to notice a run has ended badly, and stopping the instance is
the intuitive next step. It must halt training and write STATUS.md instead.

This is a static check over the source, not a behavioural one -- the failure it
guards against costs the operator their box, so it should fail at review time
rather than at runtime.
"""
import ast
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# Commands that would end, suspend or reboot the instance. Matched against
# *executable string literals in code* only -- not comments and not docstrings,
# which legitimately discuss the operator rebooting the box after the RAM
# incident, and not attribute calls like `executor.shutdown()`.
FORBIDDEN_TOKENS = {
    "vastai", "shutdown", "poweroff", "reboot", "halt", "telinit",
}
FORBIDDEN_SUBSTRINGS = ("vastai", "vast.ai/api")


def _python_sources():
    skip_dirs = {".git", "wandb", "runs", "data", "__pycache__", "vendor",
                 ".pytest_cache", "node_modules"}
    for path in REPO.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        # This file necessarily contains the forbidden strings.
        if path.name == "test_instance_safety.py":
            continue
        yield path


def _docstring_nodes(tree):
    """Every string Constant that is a module/class/function docstring."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _code_string_literals(path):
    """String literals that are actually data, not prose."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            yield node.lineno, node.value


def _process_spawning_calls(tree):
    """Call nodes that actually execute something: `subprocess.*`, `os.system`,
    `os.exec*`, `os.spawn*`. Scoping to these is what separates a real
    termination call from `executor.shutdown()` or a `"init"` label."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            root = f.value
            mod = root.id if isinstance(root, ast.Name) else ""
            if mod == "subprocess" or (mod == "os" and (
                    f.attr == "system" or f.attr.startswith(("exec", "spawn", "popen")))):
                yield node
        elif isinstance(f, ast.Name) and f.id in {"run", "check_output", "Popen"}:
            yield node


def _literals_in(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def test_no_module_can_terminate_the_instance():
    offenders = []
    for path in _python_sources():
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)

        # 1. The vast.ai CLI/API must not appear anywhere in executable code.
        for lineno, value in _code_string_literals(path):
            if any(sub in value.lower() for sub in FORBIDDEN_SUBSTRINGS):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {value!r}")

        # 2. No process-spawning call may name a command that ends the box.
        for call in _process_spawning_calls(tree):
            for value in _literals_in(call):
                first = value.lower().replace("/", " ").split()
                if first and first[0] in FORBIDDEN_TOKENS:
                    offenders.append(
                        f"{path.relative_to(REPO)}:{call.lineno}: spawns {value!r}")

    assert not offenders, (
        "AGENT.md SS0.1/SS7: this agent must never destroy, stop or reboot the "
        "instance. Found:\n  " + "\n  ".join(offenders))


def test_the_guard_actually_catches_a_termination_call(tmp_path):
    """A guard that cannot fail is not a guard."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""Docstring mentioning reboot and shutdown is fine."""\n'
        "import subprocess\n"
        'subprocess.run(["vastai", "destroy", "instance"])\n'
    )
    hits = [v for _, v in _code_string_literals(bad)
            if any(sub in v.lower() for sub in FORBIDDEN_SUBSTRINGS)]
    assert hits == ["vastai"]

    worse = tmp_path / "worse.py"
    worse.write_text('import subprocess\nsubprocess.run(["shutdown", "-h", "now"])\n')
    tree = ast.parse(worse.read_text())
    spawned = [v for call in _process_spawning_calls(tree) for v in _literals_in(call)]
    assert any(v.lower().split()[0] in FORBIDDEN_TOKENS for v in spawned)

    # And it must NOT fire on the things this repo legitimately does.
    fine = tmp_path / "fine.py"
    fine.write_text(
        "import subprocess\n"
        "ex.shutdown(wait=False)\n"
        'wandb_calls.append("init")\n'
        'subprocess.run(["git", "push", "origin", "main"])\n'
    )
    tree = ast.parse(fine.read_text())
    spawned = [v for call in _process_spawning_calls(tree) for v in _literals_in(call)]
    assert not any(v.lower().split()[0] in FORBIDDEN_TOKENS for v in spawned)


def test_watchdog_halts_training_rather_than_the_box():
    """The specific module most likely to grow an instance-killing call."""
    src = (REPO / "watchdog.py").read_text()
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "_exit" not in attrs, "watchdog must not os._exit the world"
    # It is allowed -- and required -- to stop *training* and report.
    assert "STATUS.md" in src or "status" in src.lower(), \
        "watchdog must write the reason somewhere the operator can read it"
    assert names or attrs  # sanity: the file parsed to something


@pytest.mark.parametrize("var", ["VAST_API_KEY"])
def test_no_module_reads_the_deliberately_absent_api_key(var):
    """AGENT.md SS1: its absence is intentional. A module that reads it is
    either about to use it or about to complain that it is missing, and both
    are wrong."""
    offenders = [str(p.relative_to(REPO)) for p in _python_sources()
                 if var in p.read_text(encoding="utf-8", errors="replace")]
    assert not offenders, f"{var} is deliberately absent; read by {offenders}"
