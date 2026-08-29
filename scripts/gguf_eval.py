"""Paired FP16 vs Q4_0 evaluation of the *same* items through stock llama.cpp.

The number this program is trying to move -- the roughly 6% perplexity penalty
Q4_0 costs the released model -- is a difference between two measurements, and
the gates that decide Phase 3 are written at 3% and 1%. Computing two aggregate
perplexities and subtracting them is not good enough at that resolution: each
aggregate carries its own sampling error, and the error of the difference is
larger than either.

`llama-perplexity` makes a much better measurement available for free. It prints
a *running* perplexity after every chunk:

    [1]4.0000,[2]4.5000,[3]4.2500,

Those are cumulative means of the per-chunk negative log-likelihood, so the
sequence can be inverted exactly:

    nll_k = k * ln(P_k) - (k - 1) * ln(P_{k-1})

That recovers each chunk's own NLL. Because both artifacts read the identical
text file with the identical context size, chunk k is literally the same tokens
in both runs, and the penalty becomes a paired per-chunk delta -- with a
per-chunk sign test alongside it, which no aggregate can provide.

The pairing is enforced, not trusted: `compare_quantization` refuses two
scorecards whose chunk counts differ, whose text file or context size differ, or
which turn out to be two artifacts of the same precision. Every one of those
would produce a confident, meaningless penalty.

Nothing here modifies llama.cpp or the GGUF files. `llama-perplexity` is invoked
exactly as a user would, and the released artifacts are opened read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus.scorecard import (  # noqa: E402
    ArtifactRef,
    Provenance,
    Scorecard,
    ScorecardError,
    sha256_file,
    write_scorecard,
)


# Measured on this model class: a 529 KB text file at -c 512 takes ~50 s. An
# hour is ~72x that -- wide enough that a loaded box cannot trip it, tight
# enough that a wedged binary becomes a traceback instead of an idle GPU
# billing overnight. Same reasoning as export.py's PPL_TIMEOUT_S.
PPL_TIMEOUT_S = 3600.0

_CHUNK_RE = re.compile(r"\[(\d+)\]([0-9]*\.?[0-9]+)")
_FINAL_RE = re.compile(r"Final estimate: PPL = ([\d.]+)")

_PRECISION_OF = {"gguf-f16": "f16", "gguf-q4_0": "q4_0", "gguf-q6_k": "q6_k"}


# ------------------------------------------------------------------ parsing ---

def parse_perplexity_chunks(output: str) -> List[float]:
    """The running perplexity after each chunk, in order."""

    pairs = [(int(index), float(value)) for index, value in _CHUNK_RE.findall(output)]
    if not pairs:
        raise ValueError(
            "no per-chunk estimates found in llama-perplexity output; expected "
            f"lines like '[1]4.0000,[2]4.5000,'. Got:\n{output[:400]}")
    pairs.sort(key=lambda pair: pair[0])
    if [index for index, _ in pairs] != list(range(1, len(pairs) + 1)):
        raise ValueError(
            f"chunk indices are not contiguous from 1: {[i for i, _ in pairs][:10]}")
    return [value for _, value in pairs]


def running_perplexities(chunk_nll: Sequence[float]) -> List[float]:
    """The inverse of `chunk_nlls`; exists so the inversion is testable."""

    running, total = [], 0.0
    for index, nll in enumerate(chunk_nll, start=1):
        total += nll
        running.append(math.exp(total / index))
    return running


def chunk_nlls(running: Sequence[float]) -> List[float]:
    """Recover each chunk's own NLL from llama-perplexity's running means."""

    if not running:
        raise ValueError("no running perplexities to invert")
    if any(value <= 0 for value in running):
        raise ValueError(
            f"running perplexity must be positive, got {min(running)}")
    nlls, previous_total = [], 0.0
    for index, value in enumerate(running, start=1):
        total = index * math.log(value)
        nlls.append(total - previous_total)
        previous_total = total
    return nlls


# ------------------------------------------------------------------ running ---

def run_perplexity(gguf_path, text_file, *, binary, n_ctx: int = 512,
                   threads: int = 8, timeout_s: float = PPL_TIMEOUT_S,
                   runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
                   ) -> Dict[str, object]:
    """One stock `llama-perplexity` pass, returning per-chunk NLL and the total."""

    command = [str(binary), "-m", str(gguf_path), "-f", str(text_file),
               "-c", str(n_ctx), "-t", str(threads), "-ngl", "0", "--no-warmup"]
    result = runner(command, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-perplexity exited {result.returncode} for {gguf_path}: "
            f"{(result.stderr or '').strip()[-500:]}")
    output = (result.stdout or "") + (result.stderr or "")
    nlls = chunk_nlls(parse_perplexity_chunks(output))
    final = _FINAL_RE.search(output)
    return {
        "chunk_nll": nlls,
        # The final estimate is llama.cpp's own; when it is present it is the
        # authority, and the reconstructed mean is checked against it below.
        "perplexity": float(final.group(1)) if final
        else math.exp(sum(nlls) / len(nlls)),
        "reconstructed_perplexity": math.exp(sum(nlls) / len(nlls)),
        "command": command,
    }


# --------------------------------------------------------------- scorecards ---

def perplexity_scorecard(*, name: str, artifact: ArtifactRef,
                         tokenizer_ref: ArtifactRef, chunk_nll: Sequence[float],
                         seed: int, git_sha: str, text_file: str, n_ctx: int,
                         runtime: Optional[dict] = None,
                         created_at: Optional[str] = None) -> Scorecard:
    """One artifact's perplexity, with each chunk kept as a pairable item."""

    items = [{"id": f"chunk-{index}", "nll": float(nll)}
             for index, nll in enumerate(chunk_nll)]
    mean_nll = sum(item["nll"] for item in items) / len(items)
    return Scorecard(
        kind="paired-quant",
        name=name,
        provenance=Provenance(
            artifact=artifact, tokenizer=tokenizer_ref, seed=seed,
            git_sha=git_sha, bpb_mode="not-applicable",
            runtime=dict(runtime or {})),
        metrics={"perplexity": math.exp(mean_nll), "mean_nll": mean_nll},
        created_at=created_at or datetime.now(timezone.utc).isoformat()
        .replace("+00:00", "Z"),
        items=items,
        details={"text_file": str(text_file), "n_ctx": int(n_ctx)},
    )


def compare_quantization(left: Scorecard, right: Scorecard) -> Dict[str, object]:
    """Paired FP16-vs-quantized damage, or a refusal to compare.

    `left` is the higher-precision reference (FP16), `right` the quantized
    artifact. Every refusal below corresponds to a way two runs can look
    comparable and not be.
    """

    left_precision = _PRECISION_OF.get(left.provenance.artifact.kind)
    right_precision = _PRECISION_OF.get(right.provenance.artifact.kind)
    if left_precision is None or right_precision is None:
        raise ScorecardError(
            "quantization comparison needs two GGUF artifacts, got "
            f"{left.provenance.artifact.kind!r} and "
            f"{right.provenance.artifact.kind!r}")
    if left_precision == right_precision:
        raise ScorecardError(
            f"both scorecards were produced at precision {left_precision!r}; "
            "a quantization penalty needs two different precisions")
    for key in ("text_file", "n_ctx"):
        if left.details.get(key) != right.details.get(key):
            # Different text or context size means chunk k is not the same
            # tokens on both sides, so the pairing is a fiction.
            raise ScorecardError(
                f"{key} differs between scorecards "
                f"({left.details.get(key)!r} vs {right.details.get(key)!r}); "
                "the two runs did not score the same text")

    from daedalus.scorecard import paired_outcomes

    paired = paired_outcomes(left, right, field="nll")
    perplexity_left = math.exp(paired["mean_left"])
    perplexity_right = math.exp(paired["mean_right"])
    deltas = paired["per_item_delta"]
    return {
        "n_chunks": paired["n"],
        "text_file": left.details.get("text_file"),
        "n_ctx": left.details.get("n_ctx"),
        "fp16_sha256": left.provenance.artifact.sha256,
        "quantized_sha256": right.provenance.artifact.sha256,
        "quantized_precision": right_precision,
        "perplexity_fp16": perplexity_left,
        "perplexity_q4_0": perplexity_right,
        "q4_penalty_pct": (perplexity_right / perplexity_left - 1) * 100,
        "mean_nll_delta": paired["mean_delta"],
        # A per-chunk sign test: a penalty carried by nearly every chunk is a
        # different finding from one carried by a handful of outliers, and only
        # the paired view can tell them apart.
        "chunks_worse": sum(1 for delta in deltas if delta > 0),
        "chunks_better": sum(1 for delta in deltas if delta < 0),
        "chunks_tied": sum(1 for delta in deltas if delta == 0),
        "max_chunk_delta": max(deltas) if deltas else 0.0,
    }


def write_comparison(path, comparison: Dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


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
    parser.add_argument("--fp16-gguf", required=True)
    parser.add_argument("--quantized-gguf", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--llama-perplexity",
                        default="/opt/llama.cpp/build/bin/llama-perplexity")
    parser.add_argument("--llama-cpp-commit", default="unknown")
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out-dir", default="runs/eval/quant")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    git_sha = _git_short_sha()
    runtime = {"llama_cpp_commit": args.llama_cpp_commit,
               "llama_perplexity": args.llama_perplexity,
               "threads": args.threads}
    tokenizer_ref = (ArtifactRef(path=args.tokenizer,
                                 sha256=sha256_file(args.tokenizer),
                                 kind="tokenizer")
                     if args.tokenizer
                     else ArtifactRef(path="<embedded-in-gguf>", sha256="0" * 64,
                                      kind="tokenizer"))

    cards = {}
    for label, path, kind in (("fp16", args.fp16_gguf, "gguf-f16"),
                              ("quantized", args.quantized_gguf, "gguf-q4_0")):
        measured = run_perplexity(path, args.text_file,
                                  binary=args.llama_perplexity,
                                  n_ctx=args.n_ctx, threads=args.threads)
        card = perplexity_scorecard(
            name=f"perplexity-{label}",
            artifact=ArtifactRef(path=path, sha256=sha256_file(path), kind=kind),
            tokenizer_ref=tokenizer_ref, chunk_nll=measured["chunk_nll"],
            seed=args.seed, git_sha=git_sha, text_file=args.text_file,
            n_ctx=args.n_ctx, runtime=runtime)
        write_scorecard(out_dir / f"perplexity-{label}.json", card)
        cards[label] = card
        print(f"{label}: PPL {measured['perplexity']:.4f} over "
              f"{len(measured['chunk_nll'])} chunks")

    comparison = compare_quantization(cards["fp16"], cards["quantized"])
    write_comparison(out_dir / "quant-comparison.json", comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
