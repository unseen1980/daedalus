"""Score the retrieval tasks through PyTorch FP16 and stock llama.cpp GGUF.

Both paths answer the *same* generated items, so the FP16-vs-Q4_0 comparison is
paired at the item level rather than a difference of two aggregates. That
matters here more than usual: retrieval accuracy at 150M is low and lumpy, and
an unpaired difference of two ~10% scores has an error bar wide enough to hide
the entire effect a quantization gate is trying to detect.

Two design choices are worth stating outright.

**The control gate runs before the model.** `verify_items` answers every
generated item with `OracleBackend`, which reads the answer out of the prompt
text. If that is not 100%, the items are malformed and no model run can be
interpreted, so the evaluation aborts instead of producing a plausible-looking
zero. This is the plan's "synthetic controls score 100% where expected" check,
enforced at the point where it is cheap.

**llama.cpp flags are probed, not assumed.** The binary is built from a pinned
upstream commit that this code does not control, and llama-cli's flag set moves
(`-no-cnv` arrived in 2025; `--no-display-prompt` has been renamed before). A
wrong flag is not a soft failure -- llama-cli exits non-zero and an unattended
sweep loses its arm. So the backend reads `--help` once and includes only the
optional flags that binary advertises, while the flags it cannot run without
(`-m`, `-f`, `-n`, `-c`, `--temp`) are required and their absence is an error.
Nothing about the model or the GGUF is modified: this is stock llama.cpp,
invoked as a user would.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus.retrieval import (  # noqa: E402
    DEPTHS,
    OracleBackend,
    RetrievalItem,
    make_all_items,
    score_items,
    summarize,
)
from daedalus.scorecard import (  # noqa: E402
    ArtifactRef,
    Provenance,
    Scorecard,
    sha256_file,
    write_scorecard,
)


# One generation is a few dozen tokens on a 150M model; a minute is ~50x the
# measured cost and exists only to turn a wedged binary into a traceback, the
# same reasoning as export.py's bounds.
GENERATION_TIMEOUT_S = 60.0
HELP_TIMEOUT_S = 30.0

# Flags the backend cannot work without. Their absence means the binary is not
# llama-cli, and guessing further would waste an unattended run.
_REQUIRED_FLAGS = ("-m", "-f", "-n", "-c", "--temp")

# Flags that improve determinism or output hygiene but whose spelling has
# changed across upstream releases. Included only when `--help` advertises them.
_OPTIONAL_FLAGS = ("--top-k", "-s", "-t", "-ngl", "--no-warmup", "-no-cnv",
                   "--no-display-prompt")


class ControlFailure(RuntimeError):
    """Raised when the synthetic controls do not score 100%."""


def parse_depths(value: str):
    return tuple(int(part) for part in value.split(",") if part.strip())


def verify_items(items: Sequence[RetrievalItem]) -> Dict[str, float]:
    """Answer every item from its own prompt; refuse to continue below 100%.

    A failure here is a formatter bug -- a dropped needle, a query whose binding
    is absent, an answer that cannot be extracted -- not a model result.
    """

    records = score_items(items, OracleBackend().generate_all(items))
    metrics = summarize(items, records)
    if metrics["exact_match"] < 1.0:
        broken = [record["id"] for record in records if not record["correct"]]
        raise ControlFailure(
            f"synthetic control scored {metrics['exact_match']:.3f}, expected "
            f"1.0; {len(broken)} malformed item(s), first: {broken[:5]}")
    return metrics


# -------------------------------------------------------- llama.cpp backend ---

class LlamaCppBackend:
    """Greedy generation through an unmodified `llama-cli`."""

    def __init__(self, gguf_path, binary, *, threads: int = 8, n_ctx: int = 4096,
                 max_new_tokens: int = 24, seed: int = 20260824,
                 timeout_s: float = GENERATION_TIMEOUT_S,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.gguf_path = Path(gguf_path)
        self.binary = Path(binary)
        self.threads = threads
        self.n_ctx = n_ctx
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.timeout_s = timeout_s
        self.runner = runner
        self._supported: Optional[set] = None

    # -- flag probing --------------------------------------------------------
    def supported_flags(self) -> set:
        if self._supported is not None:
            return self._supported
        result = self.runner([str(self.binary), "--help"], capture_output=True,
                             text=True, timeout=HELP_TIMEOUT_S)
        help_text = (result.stdout or "") + (result.stderr or "")
        missing = [flag for flag in _REQUIRED_FLAGS
                   if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])",
                                    help_text)]
        if missing:
            raise RuntimeError(
                f"{self.binary} does not advertise required flags {missing}; "
                "this does not look like llama-cli")
        self._supported = {
            flag for flag in _REQUIRED_FLAGS + _OPTIONAL_FLAGS
            if re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text)
        }
        return self._supported

    def _command(self, prompt_file: str) -> List[str]:
        supported = self.supported_flags()
        command = [str(self.binary), "-m", str(self.gguf_path),
                   "-f", prompt_file,
                   "-n", str(self.max_new_tokens),
                   "-c", str(self.n_ctx),
                   # Temperature zero is the greedy path in llama.cpp; top-k 1
                   # pins it even on builds whose temp-0 handling changed.
                   "--temp", "0"]
        for flag, value in (("--top-k", "1"), ("-s", str(self.seed)),
                            ("-t", str(self.threads)), ("-ngl", "0")):
            if flag in supported:
                command += [flag, value]
        for flag in ("--no-warmup", "-no-cnv", "--no-display-prompt"):
            if flag in supported:
                command.append(flag)
        return command

    # -- generation ----------------------------------------------------------
    def generate(self, item: RetrievalItem) -> str:
        # The prompt goes in a file, never in argv: these prompts run to 2048
        # tokens and contain newlines, and an argv round trip through a shell
        # would be one quoting bug away from silently scoring a truncated
        # context.
        handle, prompt_file = tempfile.mkstemp(prefix="daedalus-retrieval-",
                                               suffix=".txt")
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(item.prompt)
            result = self.runner(self._command(prompt_file), capture_output=True,
                                 text=True, timeout=self.timeout_s)
        finally:
            os.unlink(prompt_file)
        if result.returncode != 0:
            raise RuntimeError(
                f"llama-cli exited {result.returncode} for item {item.id}: "
                f"{(result.stderr or '').strip()[-500:]}")
        return _clean_completion(result.stdout or "")

    def generate_all(self, items: Sequence[RetrievalItem]) -> List[str]:
        return [self.generate(item) for item in items]


_NOISE = re.compile(
    r"\[end of text\]|^llama_perf.*$|^llama_print_timings.*$|^main:.*$",
    flags=re.MULTILINE)


def _clean_completion(text: str) -> str:
    """Drop llama-cli's end-of-text marker and any timing lines on stdout."""

    return _NOISE.sub("", text).strip()


# ------------------------------------------------------------ torch backend ---

class TorchBackend:
    """Greedy decoding from a PyTorch model, matching the GGUF path's sampling."""

    def __init__(self, model, tokenizer, *, device: str = "cpu",
                 max_new_tokens: int = 24, eos_id: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.eos_id = eos_id

    def generate(self, item: RetrievalItem) -> str:
        import torch

        try:
            ids = self.tokenizer.encode(item.prompt, add_special_tokens=False)
        except TypeError:
            ids = self.tokenizer.encode(item.prompt)
        was_training = getattr(self.model, "training", False)
        self.model.eval()
        generated: List[int] = []
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                window = torch.tensor([ids], device=self.device)
                logits, _, _ = self.model(window, targets=None, return_logits=True)
                next_id = int(torch.argmax(logits[0, -1]).item())
                if self.eos_id is not None and next_id == self.eos_id:
                    break
                generated.append(next_id)
                ids.append(next_id)
        self.model.train(was_training)
        return self.tokenizer.decode(generated).strip()

    def generate_all(self, items: Sequence[RetrievalItem]) -> List[str]:
        return [self.generate(item) for item in items]


# -------------------------------------------------------------- scorecards ---

def run_retrieval(items_by_task: Dict[str, List[RetrievalItem]], backend, *,
                  out_dir, artifact: ArtifactRef, tokenizer_ref: ArtifactRef,
                  seed: int, git_sha: str,
                  runtime: Optional[dict] = None,
                  verify: bool = True) -> Dict[str, Dict[str, Path]]:
    """Score every task and write one scorecard (plus item sidecar) per task."""

    out_dir = Path(out_dir)
    if verify:
        for items in items_by_task.values():
            verify_items(items)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    written: Dict[str, Dict[str, Path]] = {}
    for task, items in items_by_task.items():
        records = score_items(items, backend.generate_all(items))
        # The exact prompt rides with each outcome: a retrieval number is only
        # checkable if the text that produced it can be read back.
        for item, record in zip(items, records):
            record["prompt"] = item.prompt
        card = Scorecard(
            kind="retrieval",
            name=f"retrieval-{task}",
            provenance=Provenance(
                artifact=artifact, tokenizer=tokenizer_ref, seed=seed,
                git_sha=git_sha, bpb_mode="not-applicable",
                runtime=dict(runtime or {}),
            ),
            metrics=summarize(items, records),
            created_at=created_at,
            items=records,
        )
        written[task] = write_scorecard(out_dir / f"retrieval-{task}.json", card)
    return written


# --------------------------------------------------------------------- cli ---

def _git_short_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _artifact_ref(path: Optional[str], kind: str) -> ArtifactRef:
    if path is None:
        return ArtifactRef(path="<none>", sha256="0" * 64, kind=kind)
    return ArtifactRef(path=path, sha256=sha256_file(path), kind=kind)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "llama-cpp", "oracle"),
                        required=True)
    parser.add_argument("--gguf", help="GGUF artifact for the llama-cpp backend")
    parser.add_argument("--llama-cli", default="/opt/llama.cpp/build/bin/llama-cli")
    parser.add_argument("--checkpoint", help="Daedalus .pt for the torch backend")
    parser.add_argument("--config", default="daedalus-150m")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer directory or file recorded in provenance")
    parser.add_argument("--depths", default=",".join(str(d) for d in DEPTHS))
    parser.add_argument("--per-depth", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=4)
    parser.add_argument("--control-items", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="runs/eval/retrieval")
    args = parser.parse_args(argv)

    from daedalus.data import get_tokenizer

    tokenizer = get_tokenizer()
    items = make_all_items(tokenizer, depths=parse_depths(args.depths),
                           per_depth=args.per_depth, seed=args.seed,
                           n_queries=args.n_queries,
                           control_items=args.control_items)

    runtime = {"backend": args.backend, "device": args.device,
               "max_new_tokens": args.max_new_tokens}
    if args.backend == "llama-cpp":
        if not args.gguf:
            parser.error("--gguf is required for the llama-cpp backend")
        backend = LlamaCppBackend(args.gguf, args.llama_cli, threads=args.threads,
                                  n_ctx=args.n_ctx,
                                  max_new_tokens=args.max_new_tokens,
                                  seed=args.seed)
        artifact = _artifact_ref(args.gguf,
                                 "gguf-q4_0" if "q4" in Path(args.gguf).name.lower()
                                 else "gguf-f16")
        runtime["llama_cli"] = str(args.llama_cli)
        runtime["threads"] = args.threads
    elif args.backend == "torch":
        if not args.checkpoint:
            parser.error("--checkpoint is required for the torch backend")
        import torch
        from daedalus.config import PRESETS
        from daedalus.model import Daedalus
        from train import load_checkpoint

        model = Daedalus(PRESETS[args.config]).to(args.device)
        load_checkpoint(args.checkpoint, model, map_location=args.device)
        if args.device.startswith("cuda"):
            model = model.half()
        backend = TorchBackend(model=model, tokenizer=tokenizer,
                               device=args.device,
                               max_new_tokens=args.max_new_tokens, eos_id=0)
        artifact = _artifact_ref(args.checkpoint, "checkpoint")
        artifact.config = args.config
        runtime["torch"] = torch.__version__
    else:
        backend = OracleBackend()
        artifact = _artifact_ref(None, "checkpoint")

    written = run_retrieval(
        items, backend, out_dir=args.out_dir, artifact=artifact,
        tokenizer_ref=_artifact_ref(args.tokenizer, "tokenizer"),
        seed=args.seed, git_sha=_git_short_sha(), runtime=runtime)

    for task, paths in sorted(written.items()):
        print(f"wrote {paths['scorecard']}")
        print(json.dumps(json.loads(Path(paths["scorecard"]).read_text())["metrics"],
                         indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
