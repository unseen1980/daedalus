"""Held-out bits-per-byte by source: code languages, and the general replay.

Phase 8 is a trade. Continued pretraining on code is *supposed* to move code
bits-per-byte down, and the only question that matters is what it costs
everywhere else -- which is why the gate reads two numbers at once (code BPB
improves >=5%, general BPB regresses <=1.5%) and why both must be measured on
holdouts split by whole document/repository rather than by packed window.

Two decisions are worth stating.

**Bits per byte, not perplexity.** Byte-normalised likelihood is the only
likelihood comparable across tokenizers, and Phase 4 exists precisely to change
the tokenizer. A per-token number would make the V1 and V2 columns of the final
report incomparable while looking like they could be subtracted.

**Every aggregate is named.** A code holdout token-weighted across languages
lets Python's volume swallow a collapse in Rust; equal weighting lets a tiny
language dominate; the training mixture's weights answer a third question again.
Rather than pick one silently, all three are emitted -- `bpb_token_weighted`,
`bpb_equal_weight`, and `bpb` under whatever weighting was actually requested --
with `details.weighting` naming which one `bpb` is. Per-language values sit
alongside as the real deliverable, and as pairable items so two checkpoints can
be compared language by language.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
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


def discover_sources(root) -> Dict[str, int]:
    """Every manifest-backed subdirectory of `root`, with its token count."""

    root = Path(root)
    sources: Dict[str, int] = {}
    for entry in sorted(root.iterdir()):
        manifest = entry / "manifest.json"
        if not (entry.is_dir() and manifest.exists()):
            continue
        payload = json.loads(manifest.read_text())
        if "total_tokens" not in payload:
            raise ValueError(
                f"{manifest} has no total_tokens; a source without a token "
                "count cannot be weighted or reported")
        sources[entry.name] = int(payload["total_tokens"])
    if not sources:
        raise ValueError(
            f"no source under {root} has a manifest.json; expected one "
            "subdirectory per language (or per replay source)")
    return sources


def evaluate_sources(root, *, bpb_fn: Callable[[Path], float]) -> List[dict]:
    """Measure BPB for each source, keeping it as its own pairable item."""

    records = []
    for name, tokens in discover_sources(root).items():
        value = float(bpb_fn(Path(root) / name))
        if not math.isfinite(value):
            raise ValueError(
                f"bits-per-byte for source {name!r} is not finite ({value}); "
                "refusing to record it")
        records.append({"id": name, "bpb": value, "tokens": tokens})
    return records


def parse_weights(pairs) -> Dict[str, float]:
    weights = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"expected name=fraction, got {pair!r}")
        name, value = pair.split("=", 1)
        weights[name] = float(value)
    return weights


def summarize_bpb(records: List[dict],
                  weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Per-source BPB plus all three aggregates, each under its own name."""

    if not records:
        return {"n_sources": 0.0}
    metrics: Dict[str, float] = {"n_sources": float(len(records))}
    for record in records:
        metrics[f"bpb_{record['id']}"] = record["bpb"]
        metrics[f"tokens_{record['id']}"] = float(record["tokens"])

    metrics["bpb_equal_weight"] = (sum(record["bpb"] for record in records)
                                   / len(records))
    total_tokens = sum(record["tokens"] for record in records)
    metrics["bpb_token_weighted"] = (
        sum(record["bpb"] * record["tokens"] for record in records) / total_tokens
        if total_tokens else float("nan"))

    if weights:
        known = {record["id"] for record in records}
        unknown = sorted(set(weights) - known)
        if unknown:
            # Silently dropping a weight would renormalise the aggregate to a
            # different mixture than the caller asked for, which is exactly the
            # kind of quiet reweighting this module exists to prevent.
            raise ValueError(
                f"weights name sources absent from the holdout: {unknown}; "
                f"holdout has {sorted(known)}")
        total_weight = sum(weights.values())
        metrics["bpb"] = sum(record["bpb"] * weights.get(record["id"], 0.0)
                             for record in records) / total_weight
    else:
        metrics["bpb"] = metrics["bpb_equal_weight"]
    return metrics


def run_bpb_eval(*, name: str, holdout_root, out_dir, artifact: ArtifactRef,
                 tokenizer_ref: ArtifactRef, seed: int, git_sha: str,
                 bpb_fn: Callable[[Path], float],
                 max_batches: Optional[int] = None,
                 weights: Optional[Dict[str, float]] = None,
                 runtime: Optional[dict] = None) -> Dict[str, Path]:
    """Score one holdout root and write its scorecard."""

    records = evaluate_sources(holdout_root, bpb_fn=bpb_fn)
    card = Scorecard(
        kind="bpb",
        name=name,
        provenance=Provenance(
            artifact=artifact, tokenizer=tokenizer_ref, seed=seed,
            git_sha=git_sha,
            bpb_mode="full" if max_batches is None else "sample",
            bpb_sample_batches=max_batches,
            runtime=dict(runtime or {})),
        metrics=summarize_bpb(records, weights),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        items=records,
        details={
            "holdout_root": str(holdout_root),
            "weighting": "explicit" if weights else "equal",
            "weights": dict(weights) if weights else None,
        },
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
    parser.add_argument("--name", required=True,
                        help="scorecard name, e.g. code-bpb or general-replay-bpb")
    parser.add_argument("--holdout-root", required=True,
                        help="directory with one manifest-backed subdirectory "
                             "per language (or per replay source)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="daedalus-150m")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=-1,
                        help="-1 (default) scores every held-out window and is "
                             "recorded as a full pass; a positive value is "
                             "recorded as a bounded sample")
    parser.add_argument("--weight", action="append", default=[],
                        help="name=fraction; repeatable. Without any, the "
                             "headline bpb is equal-weighted across sources")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out-dir", default="runs/eval/bpb")
    args = parser.parse_args(argv)

    max_batches = None if args.max_batches < 0 else args.max_batches

    from daedalus.config import PRESETS
    from daedalus.data import get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb
    from train import load_checkpoint

    tokenizer = get_tokenizer()
    model = Daedalus(PRESETS[args.config]).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    def bpb_fn(source_dir: Path) -> float:
        return evaluate_bpb(model, str(source_dir), args.seq_len, tokenizer,
                            args.device, batch_size=args.batch_size,
                            max_batches=max_batches)

    tokenizer_ref = (ArtifactRef(path=args.tokenizer,
                                 sha256=sha256_file(args.tokenizer),
                                 kind="tokenizer")
                     if args.tokenizer
                     else ArtifactRef(path="<smollm2-default>", sha256="0" * 64,
                                      kind="tokenizer"))

    paths = run_bpb_eval(
        name=args.name, holdout_root=args.holdout_root, out_dir=args.out_dir,
        artifact=ArtifactRef(path=args.checkpoint,
                             sha256=sha256_file(args.checkpoint),
                             kind="checkpoint", config=args.config),
        tokenizer_ref=tokenizer_ref, seed=args.seed, git_sha=_git_short_sha(),
        bpb_fn=bpb_fn, max_batches=max_batches,
        weights=parse_weights(args.weight) or None,
        runtime={"device": args.device, "seq_len": args.seq_len,
                 "batch_size": args.batch_size})

    payload = json.loads(Path(paths["scorecard"]).read_text())
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    print(f"wrote {paths['scorecard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
