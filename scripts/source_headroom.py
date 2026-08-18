#!/usr/bin/env python
"""Can each source still deliver its remaining token budget, or has it run out?

This exists because `stack-edu-python` ran out of documents mid-build and
nothing could be done about it. At 2026-08-10 ~20:5xZ it reported:

    stream exhausted at 1,210,964,651 tokens of a 1,350,000,000 budget
    -- this source has no more documents

That is a *permanent* 139M shortfall: no retry, no extra headroom and no amount
of waiting produces a document the dataset does not contain. It also fed
straight into the `hero` mixture gate, which refuses above 10.0 pts of skew and
clears 60B by only 0.64%, so "which source might dry up next" stopped being
trivia and became the input that decides whether a 5.9-day run can launch.

The thing worth noticing is that it was **predictable from metadata alone,
before a single document was streamed.** `stack-edu-python` reads 10 parquet
files totalling 1.79 GB; at the density it actually achieved that is ~1.21B
tokens, so a 1.35B budget was never satisfiable. One Hub metadata call would
have said so in about a second.

So this asks that question for every source, up front and re-runnably:

  * how many files the source's config actually resolves to, and how far the
    saved stream position has advanced through them;
  * a **lower bound** on the tokens still reachable in the files not yet
    touched, using the density the source itself has measured so far;
  * a verdict per source -- MET / SAFE / AT_RISK / EXHAUSTED / UNKNOWN -- and a
    non-zero exit if anything is short.

Two deliberate choices about honesty:

`shard_idx` is taken as the **maximum** over the whole nested `stream_state`,
because HF's resumable state nests several positions (an inner
`examples_iterable` plus a `previous_state`) and the furthest one is the real
position. Verified against both live shapes: `stack-edu-python` reduces to 10
of 10 files (correctly EXHAUSTED) and `finepdfs-edu` to 0 of 100.

The density is a **lower** bound, not an estimate. Tokens so far are divided by
the bytes of every file *including* the partially-consumed current one, and the
remaining bytes *exclude* that file's untouched remainder. Both errors point the
same way, so a SAFE verdict understates the real headroom and cannot be an
artefact of optimistic rounding. Density is measured from the source's own
output, so per-source filters (`fineweb-edu`'s `int_score >= 3`) and the
100k-char document cap are already priced in rather than assumed away.

File lists come from `datasets.load_dataset_builder`, which is the same
resolution `dataprep` streams through, not a path-prefix guess -- a heuristic
that silently mismatched a config would report SAFE for a source it had not
actually looked at, which is the one failure this must not have.

Usage:
  python scripts/source_headroom.py                       # live topup budgets
  python scripts/source_headroom.py --source-budget fineweb-edu=5625000000 ...
  python scripts/source_headroom.py --manifest data/manifest.json --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daedalus.dataprep import MIXTURE, SourceSpec            # noqa: E402


# Verdicts, worst-first so a run's overall exit status is the worst it saw.
EXHAUSTED = "EXHAUSTED"   # stream past its last file, still short of budget
AT_RISK = "AT_RISK"       # what is left cannot cover what is still needed
UNKNOWN = "UNKNOWN"       # no measured density yet -- cannot bound anything
SAFE = "SAFE"             # remaining files cover the shortfall, with margin
MET = "MET"               # already at or over budget

_SHORT = (EXHAUSTED, AT_RISK)


def furthest_shard_idx(stream_state: Any) -> Optional[int]:
    """The furthest `shard_idx` anywhere in a saved HF stream position.

    The state nests: `hf_state.examples_iterable.examples_iterable` carries one
    position and a sibling `previous_state` carries another, and which of the
    two is ahead differs between sources. Walking the whole structure and
    taking the max is the only reduction that is right for both -- see the
    module docstring for the two live shapes it was checked against.

    Returns None when the state holds no position at all (a source that has
    never run, or a stub from a test).
    """
    best: Optional[int] = None
    stack: List[Any] = [stream_state]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            idx = node.get("shard_idx")
            if isinstance(idx, int) and not isinstance(idx, bool):
                best = idx if best is None else max(best, idx)
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return best


@dataclass
class Headroom:
    key: str
    verdict: str
    tokens_so_far: int
    budget: int
    tokens_needed: int
    files_total: int
    files_consumed: int          # == furthest shard_idx, 0 when still on file 0
    bytes_total: int
    bytes_remaining: int         # strictly excludes the current file's tail
    density_tok_per_byte: Optional[float]
    tokens_remaining_lower: Optional[int]
    cover_ratio: Optional[float]  # tokens_remaining_lower / tokens_needed
    max_epochs: int
    note: str = ""

    def line(self) -> str:
        need = f"{self.tokens_needed:,}" if self.tokens_needed > 0 else "-"
        have = ("-" if self.tokens_remaining_lower is None
                else f"{self.tokens_remaining_lower:,}")
        cover = ("-" if self.cover_ratio is None
                 else ("inf" if self.cover_ratio == float("inf")
                       else f"{self.cover_ratio:,.0f}x"))
        return (f"{self.key:24s} {self.verdict:9s} "
                f"files {self.files_consumed:>3d}/{self.files_total:<3d} "
                f"need {need:>15s}  reachable>= {have:>17s}  {cover:>7s}"
                + (f"  {self.note}" if self.note else ""))


def assess(key: str, tokens_so_far: int, budget: int, file_sizes: List[int],
           shard_idx: Optional[int], max_epochs: int = 1,
           safety_margin: float = 1.10) -> Headroom:
    """Verdict for one source from its counts -- no network, no manifest.

    `file_sizes` must be in the order `datasets` resolved them, because
    `shard_idx` indexes into exactly that list.

    `safety_margin` is applied to what is still needed, so a source scraping
    past its budget by a fraction of a percent does not read as comfortable.
    """
    n_files = len(file_sizes)
    bytes_total = sum(file_sizes)
    needed = max(0, budget - tokens_so_far)
    consumed = 0 if shard_idx is None else max(0, min(shard_idx, n_files))

    # Bytes already touched includes the current, only-partly-read file; the
    # remainder excludes it. Both push the density and the headroom down.
    bytes_touched = sum(file_sizes[:consumed + 1]) if consumed < n_files else bytes_total
    bytes_remaining = sum(file_sizes[consumed + 1:]) if consumed < n_files else 0

    density: Optional[float] = None
    reachable: Optional[int] = None
    if tokens_so_far > 0 and bytes_touched > 0:
        density = tokens_so_far / bytes_touched
        reachable = int(density * bytes_remaining)

    note = ""
    if needed == 0:
        verdict = MET
    elif max_epochs > 1:
        # Re-read sources (everyday-conversations) loop the stream, so running
        # off the last file is not terminal for them. Their shortfall is a
        # recorded, accepted deviation rather than something to alarm on.
        verdict = SAFE
        note = f"re-reads stream up to {max_epochs} epochs"
    elif consumed >= n_files:
        verdict = EXHAUSTED
        note = "stream past its last file -- shortfall is permanent"
    elif reachable is None:
        verdict = UNKNOWN
        note = "no tokens yet -- cannot measure density"
    elif reachable >= needed * safety_margin:
        verdict = SAFE
    else:
        verdict = AT_RISK
        note = "remaining files cannot cover the shortfall"

    cover: Optional[float] = None
    if reachable is not None and needed > 0:
        cover = reachable / needed
    elif needed == 0:
        cover = float("inf")

    return Headroom(key=key, verdict=verdict, tokens_so_far=tokens_so_far,
                    budget=budget, tokens_needed=needed, files_total=n_files,
                    files_consumed=consumed, bytes_total=bytes_total,
                    bytes_remaining=bytes_remaining,
                    density_tok_per_byte=density,
                    tokens_remaining_lower=reachable, cover_ratio=cover,
                    max_epochs=max_epochs, note=note)


def resolve_files(spec: SourceSpec) -> List[int]:
    """Sizes of the files `spec` streams, in the order `datasets` yields them.

    Authoritative rather than a path guess: the builder performs the same
    config/`data_files` resolution `dataprep` gets, and sizes come from the
    Hub's file metadata keyed by the resolved paths.
    """
    from datasets import load_dataset_builder
    from huggingface_hub import HfApi

    kwargs = dict(spec.load_kwargs)
    builder = load_dataset_builder(spec.dataset, spec.config,
                                   revision=spec.revision, **kwargs)
    resolved = builder.config.data_files
    files = list(resolved.get(spec.split) or [])
    if not files:                      # some builders key splits differently
        files = [f for group in resolved.values() for f in group]

    api = HfApi()
    info = api.dataset_info(spec.dataset, revision=spec.revision,
                            files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}

    out = []
    for f in files:
        # Resolved entries are hf:// URLs or repo-relative paths; match on the
        # longest suffix that is a known repo file.
        name = str(f).split("@", 1)[-1]
        name = name.split("/", 1)[-1] if name.startswith("/") else name
        hit = sizes.get(name)
        if hit is None:
            cand = [k for k in sizes if str(f).endswith(k)]
            hit = sizes[max(cand, key=len)] if cand else 0
        out.append(hit)
    return out


def load_manifest(path: str) -> Dict[str, dict]:
    with open(path) as f:
        m = json.load(f)
    srcs = m.get("sources", [])
    if isinstance(srcs, dict):
        return {k: v for k, v in srcs.items()}
    return {s["key"]: s for s in srcs}


def parse_budgets(pairs: List[str]) -> Dict[str, int]:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k.strip()] = int(float(v))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--source-budget", action="append", default=[],
                    metavar="KEY=TOKENS",
                    help="budget for one source; repeatable. Defaults to the "
                         "live 60B top-up budgets recorded in the manifest's "
                         "target_tokens split by blueprint share.")
    ap.add_argument("--target-tokens", type=float, default=None,
                    help="corpus target; per-source budgets are share x this "
                         "when --source-budget is not given")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to these source keys")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    budgets = parse_budgets(args.source_budget)

    if not budgets:
        target = args.target_tokens
        if target is None:
            with open(args.manifest) as f:
                target = float(json.load(f).get("target_tokens") or 0)
        budgets = {s.key: int(round(s.share * target)) for s in MIXTURE}

    specs = [s for s in MIXTURE if not args.only or s.key in args.only]
    rows: List[Headroom] = []
    for spec in specs:
        entry = manifest.get(spec.key, {})
        tokens = int(entry.get("tokens") or 0)
        budget = int(budgets.get(spec.key, 0))
        try:
            sizes = resolve_files(spec)
        except Exception as e:                       # a Hub hiccup is not a verdict
            print(f"{spec.key:24s} {'UNKNOWN':9s} could not resolve files: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue
        rows.append(assess(spec.key, tokens, budget, sizes,
                           furthest_shard_idx(entry.get("stream_state")),
                           max_epochs=spec.max_epochs))

    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        print(f"{'source':24s} {'verdict':9s} {'files':>7s} "
              f"{'still needed':>20s}  {'reachable (lower bd)':>28s}")
        for r in sorted(rows, key=lambda r: (r.verdict not in _SHORT, r.key)):
            print(r.line())
        short = [r.key for r in rows if r.verdict in _SHORT]
        print()
        print(f"{len(rows)} sources checked; "
              + (f"SHORT: {', '.join(short)}" if short else "none short"))

    return 1 if any(r.verdict in _SHORT for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
