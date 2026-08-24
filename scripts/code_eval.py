"""HumanEval+ / MBPP+ pass@1 under a real sandbox.

Every other evaluator in this program reads logits. This one *runs code a model
wrote*, on the same box that holds the checkpoints, the corpus and the run
credentials. So the sandbox is the primary artifact here, not the score:

  - **No network.** A preamble replaces `socket.socket`, `socket.create_connection`,
    `socket.getaddrinfo` and `urllib.request.urlopen` with functions that raise.
    Benchmark solutions never need the network, so anything that reaches for it
    is either a hallucinated import or something worse, and either way must fail
    loudly rather than succeed quietly.
  - **Bounded CPU, memory, file size and wall clock.** `setrlimit` caps address
    space and CPU seconds inside the child; `subprocess` bounds wall clock from
    outside, because a process blocked in a syscall burns no CPU and would sit
    past an RLIMIT_CPU bound forever.
  - **A fresh directory per item, removed afterwards.** Two items must not be
    able to influence each other through the filesystem, and a benchmark run
    must not leave writable litter beside the checkpoints.
  - **Isolated interpreter** (`-I`): no user site-packages, no `PYTHON*`
    environment, and no importing from the working directory.

Scoring follows EvalPlus's own rule: the `plus` tests are only credited when the
base tests already pass, so a solution cannot score on the harder suite while
failing the easier one.

Failure categories exist because a bare pass@1 delta is not actionable. A model
that starts wrapping answers in markdown fences and a model that has genuinely
lost coding ability both show up as "pass@1 fell"; `syntax_error` versus
`assertion_failed` versus `timeout` tells them apart, which is what Phase 8's
gate needs to interpret its own result.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus.scorecard import (  # noqa: E402
    ArtifactRef,
    Provenance,
    Scorecard,
    sha256_file,
    write_scorecard,
)


DATASETS = ("humaneval-plus", "mbpp-plus")

# Generous next to a benchmark solution (milliseconds) and tight next to an
# unattended sweep. EvalPlus's own default is in this range.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MEMORY_MB = 1024
DEFAULT_MAX_NEW_TOKENS = 384

# Injected ahead of every candidate program. Kept as source rather than a
# module import so the child needs nothing on its path but the standard library.
_SANDBOX_PREAMBLE = '''\
import builtins, socket

class _NetworkBlocked(RuntimeError):
    pass

def _blocked(*args, **kwargs):
    raise _NetworkBlocked(
        "network access is disabled in the Daedalus code sandbox")

# A *subclass*, not a function. `ssl` does `class SSLSocket(socket.socket)` at
# import time, so replacing the name with a function turns any import of ssl
# into "TypeError: function() argument 'code' must be code, not str" -- the
# connection is blocked either way, but the report would name the wrong cause,
# which is exactly the misdiagnosis the failure categories exist to prevent.
class _BlockedSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise _NetworkBlocked(
            "network access is disabled in the Daedalus code sandbox")

    def connect_ex(self, *args, **kwargs):
        raise _NetworkBlocked(
            "network access is disabled in the Daedalus code sandbox")

socket.socket = _BlockedSocket
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.gethostbyname = _blocked
try:
    import urllib.request
    urllib.request.urlopen = _blocked
    urllib.request.urlretrieve = _blocked
except Exception:
    pass

# No interactive input: a solution that blocks on stdin would otherwise consume
# the whole wall-clock budget and be reported as a timeout.
def _no_input(*args, **kwargs):
    raise RuntimeError("stdin is not available in the Daedalus code sandbox")
builtins.input = _no_input
'''


class SandboxError(RuntimeError):
    """Raised when the sandbox itself could not be established."""


@dataclass
class PromptItem:
    """The minimal shape the shared generation backends consume."""

    id: str
    prompt: str


# -------------------------------------------------------- answer extraction ---

_FENCE_RE = re.compile(r"```(?:python|py)?\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(prompt: str, completion: str) -> str:
    """Turn a raw completion into the program that will be executed.

    Base models continue past the function they were asked for -- into a new
    top-level statement, a second definition, or prose. Executing that trailing
    text would fail items the model actually solved, so the completion is cut at
    the first line that leaves the function body. A fenced block, when present,
    is taken as the model's own answer boundary.
    """

    fenced = _FENCE_RE.search(completion)
    if fenced:
        body = fenced.group(1)
        # A fenced block usually restates the signature; when it does, it is
        # the whole program, not a continuation of the prompt.
        if body.lstrip().startswith(("def ", "from ", "import ", "class ")):
            return body.rstrip()
        completion = body

    kept: List[str] = []
    for line in completion.split("\n"):
        stripped = line.strip()
        if stripped and not line[:1].isspace() and not line.startswith(("@", ")")):
            break                      # a new top-level statement: stop here
        kept.append(line)
    body = "\n".join(kept).rstrip()
    return f"{prompt.rstrip()}\n{body}".rstrip() if body.strip() else prompt.rstrip()


def check_syntax(code: str):
    """(is_valid, message). Parsed, never executed."""

    try:
        ast.parse(code)
    except SyntaxError as error:
        return False, f"SyntaxError: {error.msg} (line {error.lineno})"
    return True, None


# ----------------------------------------------------------------- sandbox ---

def _limits(memory_mb: int, cpu_seconds: int) -> Callable[[], None]:
    def apply() -> None:
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        # 16 MB of output at most: a runaway writer must not fill the disk the
        # checkpoints live on.
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _cpu_seconds_for(timeout_s: float) -> int:
    """CPU budget, deliberately looser than the wall clock.

    Both bounds are needed and they must not race. Wall clock is the one that
    catches the common case (a spin loop, a blocked read); RLIMIT_CPU is the
    backstop for a child that escapes the parent's timer. Setting them equal
    means SIGXCPU almost always lands first, and every infinite loop gets
    reported as a generic `exception` -- which is what this code did until the
    timeout test caught it.
    """

    return max(2, int(timeout_s * 2) + 5)


# Signals worth naming: a child killed by one of these did not "raise an
# exception", and reporting it as one would hide a resource failure.
_SIGNAL_CATEGORY = {24: "timeout",        # SIGXCPU
                    9: "resource_limit",  # SIGKILL (usually the OOM killer)
                    25: "resource_limit"} # SIGXFSZ


def _categorize(stderr: str, returncode: int = 1) -> str:
    if returncode < 0 and -returncode in _SIGNAL_CATEGORY:
        return _SIGNAL_CATEGORY[-returncode]
    if "_NetworkBlocked" in stderr or "network access is disabled" in stderr:
        return "network_blocked"
    if "MemoryError" in stderr or "Cannot allocate memory" in stderr:
        return "resource_limit"
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax_error"
    if "AssertionError" in stderr:
        return "assertion_failed"
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return "import_error"
    return "exception"


def run_in_sandbox(solution: str, test_code: str, *,
                   timeout_s: float = DEFAULT_TIMEOUT_S,
                   memory_mb: int = DEFAULT_MEMORY_MB) -> Dict[str, object]:
    """Execute `solution` followed by `test_code` under every containment."""

    valid, message = check_syntax(solution)
    if not valid:
        # Refused before execution: there is nothing to run, and reporting it as
        # a runtime exception would blur a formatting failure into a logic one.
        return {"status": "failed", "category": "syntax_error",
                "detail": message, "workdir": ""}

    workdir = tempfile.mkdtemp(prefix="daedalus-code-eval-")
    program = Path(workdir) / "candidate.py"
    program.write_text(f"{_SANDBOX_PREAMBLE}\n{solution}\n\n{test_code}\n")
    # A minimal environment: no proxies to reach through, no PYTHON* overrides,
    # and a TMPDIR inside the item's own directory.
    environment = {"PATH": "/usr/bin:/bin", "TMPDIR": workdir,
                   "HOME": workdir, "LC_ALL": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        # `start_new_session` puts the child in its own process group, so a
        # timeout can kill the *group*. Without it, a candidate that spawned a
        # helper would leave that helper running after the parent gave up --
        # and an unattended sweep would accumulate them.
        process = subprocess.Popen(
            [sys.executable, "-I", str(program)],
            cwd=workdir, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
            preexec_fn=_limits(memory_mb, _cpu_seconds_for(timeout_s)),
        )
    except OSError as error:                    # sandbox could not be established
        shutil.rmtree(workdir, ignore_errors=True)
        raise SandboxError(f"could not start the code sandbox: {error}") from error

    try:
        _, stderr = process.communicate(timeout=timeout_s)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            process.kill()
        process.communicate()
        return {"status": "failed", "category": "timeout",
                "detail": f"exceeded {timeout_s}s wall clock", "workdir": workdir}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if returncode == 0:
        return {"status": "passed", "category": None, "detail": "",
                "workdir": workdir}
    stderr = stderr or ""
    return {"status": "failed", "category": _categorize(stderr, returncode),
            "detail": stderr.strip()[-2000:] or f"exited with {returncode}",
            "workdir": workdir}


# ---------------------------------------------------------------- evaluation ---

def evaluate_problems(problems: Dict[str, dict], backend, *,
                      timeout_s: float = DEFAULT_TIMEOUT_S,
                      memory_mb: int = DEFAULT_MEMORY_MB) -> List[dict]:
    """Generate one greedy completion per problem and execute it."""

    records: List[dict] = []
    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        completion = backend.generate(PromptItem(id=task_id, prompt=prompt))
        solution = extract_code(prompt, completion)
        syntax_valid, syntax_message = check_syntax(solution)

        base = run_in_sandbox(solution, problem.get("test", ""),
                              timeout_s=timeout_s, memory_mb=memory_mb)
        base_passed = int(base["status"] == "passed")
        plus_passed = 0
        plus = None
        if base_passed and problem.get("plus_test"):
            # EvalPlus's rule: the extended suite only counts on top of a
            # passing base suite, so `plus` can never exceed `base`.
            plus = run_in_sandbox(solution, problem["plus_test"],
                                  timeout_s=timeout_s, memory_mb=memory_mb)
            plus_passed = int(plus["status"] == "passed")

        records.append({
            "id": task_id,
            "base_passed": base_passed,
            "plus_passed": plus_passed,
            "syntax_valid": int(syntax_valid),
            "category": base["category"] if not base_passed
            else (plus["category"] if plus and not plus_passed else None),
            "detail": base["detail"] if not base_passed else "",
            "syntax_error": syntax_message,
            "completion": completion,
            "solution": solution,
        })
    return records


def summarize_code(records: List[dict]) -> Dict[str, float]:
    """pass@1 for the base and plus suites, syntax validity, failure counts."""

    if not records:
        return {"n": 0.0, "pass@1": 0.0, "pass@1_plus": 0.0, "syntax_valid": 0.0}
    total = len(records)
    metrics = {
        "n": float(total),
        "pass@1": sum(record["base_passed"] for record in records) / total,
        "pass@1_plus": sum(record["plus_passed"] for record in records) / total,
        "syntax_valid": sum(record["syntax_valid"] for record in records) / total,
    }
    for record in records:
        category = record.get("category")
        if category:
            key = f"fail_{category}"
            metrics[key] = metrics.get(key, 0.0) + 1
    return metrics


# ---------------------------------------------------------------- datasets ---

def _evalplus_loader(name: str) -> Dict[str, dict]:
    if name == "humaneval-plus":
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    from evalplus.data import get_mbpp_plus
    return get_mbpp_plus()


def load_problems(name: str, *, limit: Optional[int] = None,
                  loader: Callable[[str], Dict[str, dict]] = _evalplus_loader
                  ) -> Dict[str, dict]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    problems = loader(name)
    if limit:
        problems = dict(list(problems.items())[:limit])
    return problems


# --------------------------------------------------------------- scorecards ---

def run_code_eval(name: str, problems: Dict[str, dict], backend, *, out_dir,
                  artifact: ArtifactRef, tokenizer_ref: ArtifactRef, seed: int,
                  git_sha: str, dataset_revision: str,
                  runtime: Optional[dict] = None,
                  timeout_s: float = DEFAULT_TIMEOUT_S,
                  memory_mb: int = DEFAULT_MEMORY_MB) -> Dict[str, Path]:
    records = evaluate_problems(problems, backend, timeout_s=timeout_s,
                                memory_mb=memory_mb)
    card = Scorecard(
        kind="code-execution",
        name=name,
        provenance=Provenance(
            artifact=artifact, tokenizer=tokenizer_ref, seed=seed,
            git_sha=git_sha, bpb_mode="not-applicable",
            task_revisions={name: dataset_revision},
            runtime=dict(runtime or {})),
        metrics=summarize_code(records),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        items=records,
        details={"timeout_s": timeout_s, "memory_mb": memory_mb},
    )
    return write_scorecard(Path(out_dir) / f"{name}.json", card)


# --------------------------------------------------------------------- cli ---

def _git_short_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="humaneval-plus")
    parser.add_argument("--backend", choices=("torch", "llama-cpp"), required=True)
    parser.add_argument("--gguf")
    parser.add_argument("--llama-cli", default="/opt/llama.cpp/build/bin/llama-cli")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", default="daedalus-150m")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-revision", default="unknown")
    parser.add_argument("--out-dir", default="runs/eval/code")
    args = parser.parse_args(argv)

    from scripts.retrieval_eval import LlamaCppBackend, TorchBackend

    runtime = {"backend": args.backend, "max_new_tokens": args.max_new_tokens}
    try:
        import evalplus
        runtime["evalplus"] = getattr(evalplus, "__version__", "unknown")
    except ImportError:
        runtime["evalplus"] = "absent"

    if args.backend == "llama-cpp":
        if not args.gguf:
            parser.error("--gguf is required for the llama-cpp backend")
        backend = LlamaCppBackend(args.gguf, args.llama_cli, threads=args.threads,
                                  n_ctx=args.n_ctx,
                                  max_new_tokens=args.max_new_tokens,
                                  seed=args.seed)
        artifact = ArtifactRef(path=args.gguf, sha256=sha256_file(args.gguf),
                               kind="gguf-q4_0" if "q4" in Path(args.gguf).name.lower()
                               else "gguf-f16")
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for the torch backend")
        from daedalus.config import PRESETS
        from daedalus.data import get_tokenizer
        from daedalus.model import Daedalus
        from train import load_checkpoint

        model = Daedalus(PRESETS[args.config]).to(args.device)
        load_checkpoint(args.checkpoint, model, map_location=args.device)
        if args.device.startswith("cuda"):
            model = model.half()
        backend = TorchBackend(model=model, tokenizer=get_tokenizer(),
                               device=args.device,
                               max_new_tokens=args.max_new_tokens, eos_id=0)
        artifact = ArtifactRef(path=args.checkpoint,
                               sha256=sha256_file(args.checkpoint),
                               kind="checkpoint", config=args.config)

    tokenizer_ref = (ArtifactRef(path=args.tokenizer,
                                 sha256=sha256_file(args.tokenizer),
                                 kind="tokenizer")
                     if args.tokenizer
                     else ArtifactRef(path="<embedded>", sha256="0" * 64,
                                      kind="tokenizer"))

    problems = load_problems(args.dataset, limit=args.limit)
    paths = run_code_eval(args.dataset, problems, backend, out_dir=args.out_dir,
                          artifact=artifact, tokenizer_ref=tokenizer_ref,
                          seed=args.seed, git_sha=_git_short_sha(),
                          dataset_revision=args.dataset_revision,
                          runtime=runtime, timeout_s=args.timeout_s,
                          memory_mb=args.memory_mb)
    payload = json.loads(Path(paths["scorecard"]).read_text())
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    print(f"wrote {paths['scorecard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
