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
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus.scorecard import (  # noqa: E402
    ArtifactRef,
    Provenance,
    Scorecard,
    sha256_file,
    write_scorecard,
)


def discover_sources(root, sources: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Every manifest-backed subdirectory of `root`, with its token count.

    `sources` narrows the scan to a named subset. Phase 6 needs that: its stage-A
    arms trained on one source, and the box's holdout root carries three, so
    scoring an arm over all of them would answer a question the screen never
    asked and cost three times the GPU hours to get it wrong.

    A requested name that is not present raises rather than being skipped. A
    scorecard that quietly measured two of the three sources it was asked for
    describes a holdout that does not exist -- the same failure mode
    `summarize_bpb` already refuses for weights naming absent sources.
    """

    root = Path(root)
    found: Dict[str, int] = {}
    for entry in sorted(root.iterdir()):
        manifest = entry / "manifest.json"
        if not (entry.is_dir() and manifest.exists()):
            continue
        payload = json.loads(manifest.read_text())
        if "total_tokens" not in payload:
            raise ValueError(
                f"{manifest} has no total_tokens; a source without a token "
                "count cannot be weighted or reported")
        found[entry.name] = int(payload["total_tokens"])
    if not found:
        raise ValueError(
            f"no source under {root} has a manifest.json; expected one "
            "subdirectory per language (or per replay source)")
    if sources is None:
        return found
    wanted = list(dict.fromkeys(sources))
    missing = [name for name in wanted if name not in found]
    if missing:
        raise ValueError(
            f"requested source(s) {missing} are not under {root}; "
            f"it has {sorted(found)}")
    return {name: found[name] for name in sorted(wanted)}


def evaluate_sources(root, *, bpb_fn: Callable[[Path], float],
                     sources: Optional[Sequence[str]] = None) -> List[dict]:
    """Measure BPB for each source, keeping it as its own pairable item."""

    records = []
    for name, tokens in discover_sources(root, sources).items():
        value = float(bpb_fn(Path(root) / name))
        if not math.isfinite(value):
            raise ValueError(
                f"bits-per-byte for source {name!r} is not finite ({value}); "
                "refusing to record it")
        records.append({"id": name, "bpb": value, "tokens": tokens})
    return records


def build_holdout(mixture_root, holdout_root, train_root,
                  holdout_frac: float = 0.02,
                  weights: Optional[Dict[str, float]] = None) -> dict:
    """Materialize the train/holdout split this evaluator requires, and record
    what was reserved.

    `--holdout-root` is required and there is no default place one comes from.
    Phase 3 found what that costs: the corpus fetched onto the box is train
    shards only, `make_mixture_holdout_split` had never been run against it, and
    full-pass BPB -- third in the preregistered selection order -- went
    unmeasured for want of a directory. Folding the build into scoring makes the
    split a consequence of asking for the measurement rather than a prerequisite
    someone has to remember.

    Shards are hardlinked, not copied, so a split of the 3.0 GB corpus on this
    box costs inodes rather than gigabytes, and re-running is a no-op: the
    underlying helper reuses a destination only when it is the same file by
    (device, inode), so this is safe to call before every scoring pass.

    Sources with a single shard cannot be split and are skipped by the helper.
    They are named in `skipped` rather than dropped silently, because a
    scorecard that omits a source it did not measure describes a different
    mixture than the one it claims to.

    That is not a hypothetical. On this box seven of the corpus's ten sources
    arrived as a single shard each, so no holdout can be carved from them at
    all, and the aggregate covers the three that remain. Naming the skips is
    necessary but not sufficient -- a reader counting names cannot tell whether
    the gap is 5% of the mixture or half of it. `mixture_share_covered` answers
    that directly, so a partial measurement cannot be read as a whole one.
    """
    from daedalus.data import make_mixture_holdout_split

    splits = make_mixture_holdout_split(str(mixture_root), str(train_root),
                                        str(holdout_root),
                                        holdout_frac=holdout_frac)
    present = set(discover_sources(mixture_root))
    sources = {
        name: {
            "train_tokens": int(split["train"]["total_tokens"]),
            "holdout_tokens": int(split["holdout"]["total_tokens"]),
            "train_shards": len(split["train"]["shards"]),
            "holdout_shards": len(split["holdout"]["shards"]),
        }
        for name, split in splits.items()
    }
    if weights is None:
        from daedalus.dataprep import MIXTURE
        weights = {spec.key: spec.share for spec in MIXTURE}
    on_disk = sum(weights.get(name, 0.0) for name in present)
    covered = sum(weights.get(name, 0.0) for name in sources)
    return {
        "mixture_root": str(mixture_root),
        "holdout_frac": holdout_frac,
        "sources": sources,
        "skipped": sorted(present - set(sources)),
        "mixture_share_covered": covered / on_disk if on_disk else 0.0,
    }


def recovery_exposure(mixture_root, *, run_tokens: int,
                      weights: Optional[Dict[str, float]] = None,
                      max_epochs: float = 4.0) -> dict:
    """How many times a run of `run_tokens` over `mixture_root` covered each
    source -- and therefore each of that source's holdout shards.

    A holdout carved out of the training corpus *after* a run has already
    trained on it is not held out from that run. `MixtureBatchSource` draws
    whole micro-batches from a source chosen by mixture weight, and within a
    source `ShardBatchSource` samples windows **with replacement** across every
    shard it has -- so reserving tail shards afterwards reserves nothing the run
    did not already see. What can be said exactly is *how much* it saw: the
    expected number of times the run covered a token of source `s` is
    `run_tokens * prob_s / tokens_on_disk_s`, which is the epoch count
    `resolve_mixture` already computes for the training preflight.

    The shares used are post-cap, because those are the shares the run actually
    sampled. Using the pre-cap blueprint would overstate exposure for exactly
    the short sources `cap_weights_by_epochs` exists to protect.

    This does not make a contaminated holdout clean. It makes the contamination
    a per-source number, which matters here because it is wildly uneven: a
    source with 403k tokens on disk and a 2% share is covered thousands of times
    by a budget that barely grazes a billion-token one, and an equal-weighted
    aggregate weights those two the same.
    """
    from train import resolve_mixture

    names, _target_probs, probs, tokens_on_disk = resolve_mixture(
        str(mixture_root), run_tokens, max_epochs, weights, verbose=False)

    sources: Dict[str, dict] = {}
    for name in names:
        on_disk = int(tokens_on_disk[name])
        drawn = float(run_tokens) * float(probs[name])
        sources[name] = {
            "share": float(probs[name]),
            "tokens_on_disk": on_disk,
            "tokens_drawn": drawn,
            "epochs": drawn / on_disk if on_disk else float("inf"),
        }
    repeated = sorted(name for name, s in sources.items() if s["epochs"] > 1.0)
    return {
        "run_tokens": int(run_tokens),
        "max_epochs_cap": max_epochs,
        "sources": sources,
        "repeated_sources": repeated,
        "max_epochs_seen": max((s["epochs"] for s in sources.values()),
                               default=0.0),
    }


def tokenizer_artifact(path) -> ArtifactRef:
    """An `ArtifactRef` for a tokenizer given as a file *or* a directory.

    A saved HF tokenizer is a directory, and `sha256_file` on one raises
    IsADirectoryError. Hashing `tokenizer.json` inside it is the right digest
    anyway: it is the file that carries the vocabulary and the merges, which is
    what "which tokenizer scored this" means.
    """
    path = Path(path)
    target = path / "tokenizer.json" if path.is_dir() else path
    if not target.exists():
        raise FileNotFoundError(
            f"{path} is not a tokenizer file and has no tokenizer.json")
    return ArtifactRef(path=str(path), sha256=sha256_file(target),
                       kind="tokenizer")


def parse_weights(pairs) -> Dict[str, float]:
    weights = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"expected name=fraction, got {pair!r}")
        name, value = pair.split("=", 1)
        weights[name] = float(value)
    return weights


#: The aggregates `summarize_bpb` writes under the same `bpb_` prefix as the
#: per-source values. Read back as source names they would appear in a
#: breakdown as two extra languages, one of which is an average of the real
#: ones -- which is how an aggregate gets counted twice.
BPB_AGGREGATE_METRICS = ("bpb_equal_weight", "bpb_token_weighted")


def per_source_bpb(card: Scorecard) -> Dict[str, Dict[str, Optional[float]]]:
    """Read a `bpb` scorecard back as the per-source table behind its aggregate.

    The inverse of `summarize_bpb`, and it lives beside it so the two keep
    agreeing about how a source is named. It exists because the aggregate alone
    invites a conclusion the breakdown does not support: phase 8's first two
    probe arms moved overall code BPB by -23.6%, of which TypeScript was -63.6%
    and Python -2.3%, on a corpus that is 55% Python. "The model got much better
    at code" and "the model got much better at TypeScript" are different claims,
    and only the second one is in the numbers.

    `weight` is the share the aggregate actually applied -- renormalized over
    the sources present, and an equal share when the card was written without
    explicit weights, which is what `summarize_bpb` does in that case. Reporting
    the raw requested weight instead would not add up to the `bpb` on the card.
    """

    if card.kind != "bpb":
        raise ValueError(
            f"scorecard {card.name!r} is kind {card.kind!r}, not 'bpb'; there "
            f"is no per-source bits-per-byte to read from it")
    metrics = card.metrics
    sources = sorted(key[len("bpb_"):] for key in metrics
                     if key.startswith("bpb_")
                     and key not in BPB_AGGREGATE_METRICS)
    if not sources:
        raise ValueError(
            f"scorecard {card.name!r} carries no per-source bits-per-byte; it "
            f"has {sorted(metrics)}. Its aggregate cannot be broken down, and "
            f"an aggregate reported without its breakdown is the thing this "
            f"function exists to prevent.")
    requested = (card.details or {}).get("sources_requested")
    if requested is not None and sorted(requested) != sources:
        raise ValueError(
            f"scorecard {card.name!r} says it scored {sorted(requested)} but "
            f"carries per-source values for {sources}; the aggregate and the "
            f"breakdown describe different holdouts")

    weights = (card.details or {}).get("weights") or {}
    total_weight = sum(float(value) for value in weights.values())
    return {
        name: {
            "bpb": float(metrics[f"bpb_{name}"]),
            "tokens": (float(metrics[f"tokens_{name}"])
                       if f"tokens_{name}" in metrics else None),
            "weight": (float(weights.get(name, 0.0)) / total_weight
                       if total_weight > 0 else 1.0 / len(sources)),
        }
        for name in sources
    }


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
                 sources: Optional[Sequence[str]] = None,
                 runtime: Optional[dict] = None,
                 details_extra: Optional[dict] = None) -> Dict[str, Path]:
    """Score one holdout root and write its scorecard."""

    records = evaluate_sources(holdout_root, bpb_fn=bpb_fn, sources=sources)
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
            # Named even when it is None, so a reader can tell "every source
            # under the root" from "the one source this arm trained on" without
            # having to know what the root happened to contain that day.
            "sources_requested": list(sources) if sources else None,
            **(details_extra or {}),
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
    parser.add_argument("--build-holdout-from", default=None,
                        help="mixture root to carve --holdout-root out of "
                             "first, by reserving whole tail shards per source. "
                             "Hardlinks and is idempotent, so it is safe to "
                             "pass on every scoring pass")
    parser.add_argument("--train-root", default=None,
                        help="where --build-holdout-from writes the "
                             "complementary train split (default: "
                             "<holdout-root>-train)")
    parser.add_argument("--holdout-frac", type=float, default=0.02,
                        help="fraction of each source's tokens to reserve")
    parser.add_argument("--exposure-tokens", type=int, default=0,
                        help="token budget of a run that already trained on "
                             "--build-holdout-from. Records how many times that "
                             "run covered each source, since a holdout carved "
                             "out afterwards is not held out from it")
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
    parser.add_argument("--source", action="append", default=[],
                        help="score only this holdout source; repeatable. "
                             "Without any, every manifest-backed source under "
                             "--holdout-root is scored")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out-dir", default="runs/eval/bpb")
    args = parser.parse_args(argv)

    max_batches = None if args.max_batches < 0 else args.max_batches
    weights = parse_weights(args.weight) or None

    details_extra: Dict[str, object] = {}
    if args.build_holdout_from:
        train_root = args.train_root or f"{args.holdout_root.rstrip('/')}-train"
        details_extra["holdout_build"] = build_holdout(
            args.build_holdout_from, args.holdout_root, train_root,
            holdout_frac=args.holdout_frac, weights=weights)
    if args.exposure_tokens:
        source_root = args.build_holdout_from or args.holdout_root
        details_extra["exposure"] = recovery_exposure(
            source_root, run_tokens=args.exposure_tokens, weights=weights)

    from daedalus.config import PRESETS
    from daedalus.data import assert_shards_tokenizer, get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb
    from train import load_checkpoint

    # `--tokenizer` decodes the held-out ids, it does not merely label the
    # scorecard. Bits per byte is nats/token converted through the *bytes those
    # tokens stand for*, and the byte count comes from decoding them -- so
    # decoding a 32,768-vocabulary holdout with SmolLM2 counts the wrong bytes
    # and reports a BPB for a corpus that does not exist. Harmless while every
    # artifact shared one tokenizer; wrong from Phase 4 onward, which is
    # exactly when BPB became the metric that decides between vocabularies.
    tokenizer = get_tokenizer(args.tokenizer)
    # Which makes it part of the measurement, so it is checked against the
    # shards rather than trusted: a holdout that records the vocabulary it was
    # packed under and disagrees produces a finite, plausible, wrong BPB, and
    # BPB is what the phase gates are decided on.
    assert_shards_tokenizer(args.holdout_root, tokenizer, args.tokenizer)
    model = Daedalus(PRESETS[args.config]).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.eval()

    def bpb_fn(source_dir: Path) -> float:
        return evaluate_bpb(model, str(source_dir), args.seq_len, tokenizer,
                            args.device, batch_size=args.batch_size,
                            max_batches=max_batches)

    tokenizer_ref = (tokenizer_artifact(args.tokenizer) if args.tokenizer
                     else ArtifactRef(path="<smollm2-default>", sha256="0" * 64,
                                      kind="tokenizer"))

    paths = run_bpb_eval(
        name=args.name, holdout_root=args.holdout_root, out_dir=args.out_dir,
        artifact=ArtifactRef(path=args.checkpoint,
                             sha256=sha256_file(args.checkpoint),
                             kind="checkpoint", config=args.config),
        tokenizer_ref=tokenizer_ref, seed=args.seed, git_sha=_git_short_sha(),
        bpb_fn=bpb_fn, max_batches=max_batches,
        weights=weights, sources=args.source or None,
        runtime={"device": args.device, "seq_len": args.seq_len,
                 "batch_size": args.batch_size},
        details_extra=details_extra or None)

    payload = json.loads(Path(paths["scorecard"]).read_text())
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    print(f"wrote {paths['scorecard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
