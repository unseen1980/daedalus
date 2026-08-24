"""Fetch a proportional slice of the pretraining corpus, not all 34 GB of it.

Phase 3 fine-tunes the released model on the distribution it was trained on,
which means the original tokenized shards -- same SmolLM2 tokenizer, same
decontamination, same per-source manifests. `daedalus.data.download_shards`
exists but snapshots the whole dataset repo, and the recovery probes need on
the order of a gigabyte to run three 100M-token arms plus a 300M follow-up
over identical data.

Two properties matter more than the saving:

**The slice keeps the mixture's shape.** Downloading "the first N files" would
silently retrain the model on whichever sources happen to sort first.
`MixtureBatchSource` renormalizes over the sources it finds on disk, so a
lopsided download does not fail -- it trains on a different distribution and
reports success. Per-source budgets come from `dataprep.MIXTURE`'s shares.

**Shards are taken from the front.** `select_holdout_shards` reserves holdout
shards from the *end* of each source's list, so taking the slice from the front
leaves that split reachable and reproducible rather than overlapping a holdout
with training data.

The rewritten local manifest lists only the shards actually present. A manifest
naming a file that was not downloaded is not a smaller corpus, it is a
`FileNotFoundError` inside `np.memmap` at the first batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_REPO = "Unseen1980/daedalus-corpus"


def mixture_shares() -> Dict[str, float]:
    """Source key -> intended share, read from the corpus definition itself."""
    from daedalus.dataprep import MIXTURE
    return {spec.key: float(spec.share) for spec in MIXTURE}


def group_by_source(files: Sequence[str]) -> Dict[str, List[str]]:
    """Repo files grouped by their top-level directory (the source key)."""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for path in files:
        parts = path.split("/")
        if len(parts) < 2:
            continue
        grouped[parts[0]].append(path)
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def per_source_targets(present: Sequence[str], target_tokens: int,
                       shares: Optional[Dict[str, float]] = None
                       ) -> Dict[str, int]:
    """Split a token budget across the sources actually in the repo.

    Renormalized over what is present, so a corpus missing a source asks the
    others for proportionally more rather than quietly under-filling the
    budget. A source the mixture never mentions gets nothing: it is not part
    of the distribution the released model was trained on.
    """
    shares = shares or mixture_shares()
    weights = {k: shares[k] for k in present if shares.get(k, 0) > 0}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(
            f"none of the sources present {sorted(present)} appear in the "
            f"corpus mixture; refusing to guess a distribution")
    return {k: int(target_tokens * w / total) for k, w in weights.items()}


def select_shards(manifest: dict, target_tokens: int) -> List[dict]:
    """Whole shards from the front of the list, until the budget is covered.

    Whole shards because a partial `.bin` has no manifest entry describing its
    real length, and `ShardDataset` derives its window count from the file it
    memory-maps. At least one shard is always taken: a source that rounds to
    zero would drop out of the mixture entirely.
    """
    shards = manifest.get("shards") or []
    if not shards:
        return []
    taken, seen = [], 0
    for shard in shards:
        taken.append(shard)
        seen += int(shard.get("tokens") or 0)
        if seen >= target_tokens:
            break
    return taken


def local_manifest(manifest: dict, shards: Sequence[dict]) -> dict:
    """The source manifest rewritten to describe only what was downloaded."""
    out = {k: v for k, v in manifest.items() if k not in ("shards", "total_tokens")}
    out["shards"] = list(shards)
    out["total_tokens"] = sum(int(s.get("tokens") or 0) for s in shards)
    out["subset_of"] = {
        "shards": len(manifest.get("shards") or []),
        "total_tokens": int(manifest.get("total_tokens") or 0),
    }
    return out


def plan_fetch(repo_files: Sequence[str], manifests: Dict[str, dict],
               target_tokens: int,
               shares: Optional[Dict[str, float]] = None) -> dict:
    """What would be downloaded, without downloading anything.

    Separated from the transfer so the shape of the slice can be reviewed --
    and tested -- before an hour of bandwidth is spent on it.
    """
    grouped = group_by_source(repo_files)
    present = [k for k in grouped if k in manifests]
    targets = per_source_targets(present, target_tokens, shares)

    plan, total = {}, 0
    for source in sorted(targets):
        chosen = select_shards(manifests[source], targets[source])
        tokens = sum(int(s.get("tokens") or 0) for s in chosen)
        total += tokens
        plan[source] = {
            "target_tokens": targets[source],
            "selected_tokens": tokens,
            "shards": [s["file"] for s in chosen],
            "available_shards": len(manifests[source].get("shards") or []),
        }
    return {"target_tokens": target_tokens, "planned_tokens": total,
            "sources": plan}


# ------------------------------------------------------------------ transfer ---

def _api(token: Optional[str]):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def list_repo(repo: str, token: Optional[str]) -> List[str]:
    return _api(token).list_repo_files(repo, repo_type="dataset")


def _download(repo: str, filename: str, out_root: Path,
              token: Optional[str]) -> Path:
    from huggingface_hub import hf_hub_download
    destination = out_root / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetched = hf_hub_download(repo_id=repo, filename=filename,
                              repo_type="dataset", token=token)
    # Copy rather than symlink into the cache: the shards outlive the cache,
    # and a training run that memory-maps a symlink into a directory something
    # else may prune is a failure mode with no error message.
    import shutil
    shutil.copyfile(fetched, destination)
    return destination


def fetch(repo: str, out_root, target_tokens: int, token: Optional[str],
          dry_run: bool = False) -> dict:
    out_root = Path(out_root)
    files = list_repo(repo, token)
    grouped = group_by_source(files)

    manifests: Dict[str, dict] = {}
    for source, paths in grouped.items():
        manifest_path = f"{source}/manifest.json"
        if manifest_path not in paths:
            continue
        if dry_run:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=repo, filename=manifest_path,
                                    repo_type="dataset", token=token)
        else:
            local = _download(repo, manifest_path, out_root, token)
        manifests[source] = json.loads(Path(local).read_text())

    plan = plan_fetch(files, manifests, target_tokens)
    if dry_run:
        return plan

    for source, entry in plan["sources"].items():
        for filename in entry["shards"]:
            _download(repo, f"{source}/{filename}", out_root, token)
        chosen = [s for s in manifests[source]["shards"]
                  if s["file"] in set(entry["shards"])]
        (out_root / source / "manifest.json").write_text(
            json.dumps(local_manifest(manifests[source], chosen), indent=2) + "\n")
    (out_root / "fetch-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", default="data/shards")
    parser.add_argument("--target-tokens", type=int, default=1_200_000_000,
                        help="enough for three 100M arms plus a 300M follow-up "
                             "without any source exceeding one epoch")
    parser.add_argument("--list", action="store_true",
                        help="show what the repo holds and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the slice that would be fetched")
    args = parser.parse_args(argv)

    token = (os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))

    if args.list:
        grouped = group_by_source(list_repo(args.repo, token))
        for source, paths in grouped.items():
            bins = [p for p in paths if p.endswith(".bin")]
            print(f"{source:28s} {len(bins):4d} shards  "
                  f"{'manifest' if f'{source}/manifest.json' in paths else 'NO MANIFEST'}")
        if not grouped:
            print(f"{args.repo} has no per-source directories")
        return 0

    plan = fetch(args.repo, args.out, args.target_tokens, token,
                 dry_run=args.dry_run)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.dry_run:
        print(f"\nfetched ~{plan['planned_tokens']:,} tokens into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
