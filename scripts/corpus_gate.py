#!/usr/bin/env python
"""Decide Phase 7's acceptance list from the corpus on disk, not from prose.

Phase 7's acceptance is five claims: no known exact evaluation contamination,
complete scored-split coverage, no source above four epochs at the target
budget, an L1 mixture skew within five points with no all-capped fallback, and
every source reproducible from a revision-pinned manifest.

Every one of those is a property of files that exist -- a frozen n-gram index, a
scan artifact, ten `manifest.json`s -- so every one of them is read here rather
than asserted by the session that also did the work. That is the same discipline
`scripts/gate_check.py` applies to Phase 2, and for the same reason: the phases
downstream spend real budget on "the corpus is clean", and a paragraph saying so
is not evidence.

The exit status is the verdict, so a controller can gate on it without parsing
prose.

**The one trap this file exists to avoid.** `l1_skew_pts` measures how far the
sampled mixture lands from the blueprint, and the acceptance bound is five
points. But `cap_weights_by_epochs` has two failure modes and the skew only sees
one. When the epoch cap binds it reweights, and the skew rises -- visible. When
*no* allocation can satisfy the cap, because every source is over the limit, it
returns the target shares unchanged and accepts the repetition; the skew is then
**0.00 by construction**, its best possible value, at the one budget where
repetition is bounded by nothing at all. A gate that read the skew alone would
hand its cleanest verdict to the corpus that most deserves a refusal. So the
skew criterion carries the fallback guard with it, and `train.py` already
records the discriminator: after a successful cap every source sits at or below
the limit, so `max_epochs_seen > max_epochs` is true precisely in the unbounded
case and never otherwise.

Two criteria therefore read the same allocation, deliberately and once:
"does any source repeat too often" and "is the blueprint's mixture actually
delivered" are different questions about it, and computing the allocation twice
is how two criteria come to disagree about the same corpus.

A budget is required rather than defaulted. The epoch cap is only defined at
one, the operator has not fixed a successor size, and
`runs/corpus/headroom-curve.md` deliberately reports a curve rather than pick
one; a default here would smuggle that choice back in as an argument nobody
typed.

Usage
-----
    python scripts/corpus_gate.py --shards-root data/shards --budget-tokens 59900000000
    python scripts/corpus_gate.py --budget-tokens 60000000000 \
        --weights-from runs/corpus/mixture-derived-probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GATE_SCHEMA = 1

#: The plan's numbers, named here so a threshold cannot drift into an argument
#: default and be relaxed by a caller after seeing a result.
MAX_EPOCHS = 4.0
MAX_SKEW_PTS = 5.0

#: `summarize_mixture` rounds epochs to three decimals, so a source pinned at
#: exactly the cap can read fractionally over it. Anything the plan would call a
#: violation is orders of magnitude larger than this.
_EPOCH_TOL = 1e-3

#: The three disjoint indices `scripts/contam_scan.py` scores a corpus against.
#: All three must come back empty: `filtered` is the negative control that says
#: the pipeline removed what it indexed, and the other two are the exposure the
#: pipeline never looked for.
_INDEX_NAMES = ("filtered", "split_gap", "limit_gap")


def _verdict(criterion: str, passed: bool, observed: dict,
             evidence: Sequence[str], detail: str = "") -> dict:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "observed": observed,
        "evidence": [str(path) for path in evidence],
        "detail": detail,
    }


def _read_json(path) -> dict:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return payload


# ------------------------------------------------------------ the criteria ---

def decontam_index_verdict(path) -> dict:
    """The frozen index must cover every scored task, at its scored split.

    Read against `eval.TASK_SPLITS` rather than a list written here, so adding a
    task to the evaluator makes this gate demand coverage for it instead of
    quietly continuing to pass on the five it used to know about.

    Two ways coverage is incomplete, and both were real on the released build:
    a per-task `limit` (2,000 items, which covered 19.9% of HellaSwag), and an
    index built against a *different* split from the one scored (ARC-Easy and
    OpenBookQA were indexed on `validation` and are scored on `test`). Neither
    shows up as an error anywhere; both show up here.
    """

    import eval as E

    try:
        payload = _read_json(path)
    except (OSError, ValueError) as exc:
        return _verdict("decontam-index-complete", False, {}, [path], str(exc))

    provenance = payload.get("provenance") or {}
    tasks = provenance.get("tasks") or {}
    limit = provenance.get("limit")
    complete = bool(provenance.get("complete"))
    problems = list(payload.get("problems") or [])

    observed: Dict[str, dict] = {}
    failures: List[str] = []
    for name, scored_split in sorted(E.TASK_SPLITS.items()):
        entry = tasks.get(name)
        if not isinstance(entry, dict):
            observed[name] = {"indexed": False, "scored_split": scored_split}
            failures.append(f"{name} is not in the index")
            continue
        indexed_split = entry.get("split")
        items = entry.get("items")
        observed[name] = {"indexed": True, "split": indexed_split,
                          "scored_split": scored_split, "items": items}
        if indexed_split != scored_split:
            failures.append(
                f"{name} was indexed on {indexed_split!r} but is scored on "
                f"{scored_split!r}")
        if not items:
            failures.append(f"{name} indexed {items!r} items")

    if limit is not None:
        failures.append(f"a per-task limit of {limit} was applied")
    if not complete:
        failures.append("the index does not record itself as complete")
    for problem in problems:
        failures.append(f"the index recorded a problem: {problem}")

    observed["limit"] = limit
    observed["complete"] = complete
    observed["ngrams"] = provenance.get("ngrams")
    observed["digest"] = provenance.get("digest")
    return _verdict("decontam-index-complete", not failures, observed, [path],
                    "; ".join(failures))


def contamination_verdict(path, supply: Optional[Dict[str, int]] = None) -> dict:
    """The corpus must hold no document matching a scored evaluation item.

    A hit is decisive: `contam_scan` matches exact 13-grams, so a document it
    flags really does contain evaluation text. A *zero* is weaker than it looks,
    because the scan reads systematically-spaced windows rather than every
    token, so the pass detail carries the sampled fraction and the Wilson upper
    bound on the rate. "Zero in a 1.3% sample" and "zero in the corpus" are
    different claims and only the first one is ever measured.

    `supply` is what makes the verdict a statement about *this* corpus. A scan
    artifact carries no reference to the shard tree it read, so without the
    check below a scan of any corpus, or of an earlier state of this one, reads
    as evidence about the one being gated. `per_source[].source_tokens` is the
    extent the scan believed each source had; when that disagrees with what the
    source holds now, part of the corpus was never in front of the scanner and
    the clean verdict does not cover it.
    """

    try:
        payload = _read_json(path)
    except (OSError, ValueError) as exc:
        return _verdict("corpus-contamination", False, {}, [path], str(exc))

    totals = payload.get("totals") or {}
    if not totals:
        return _verdict("corpus-contamination", False, {}, [path],
                        "the scan artifact records no totals; it cannot decide "
                        "whether the corpus is clean")

    observed: Dict[str, object] = {
        "corpus_tokens": totals.get("corpus_tokens"),
        "docs_scanned": totals.get("docs"),
        "sampled_frac": totals.get("sampled_frac"),
    }
    failures: List[str] = []
    for name in _INDEX_NAMES:
        docs = totals.get(f"docs_{name}")
        observed[f"docs_{name}"] = docs
        observed[f"doc_rate_{name}_upper95"] = totals.get(
            f"doc_rate_{name}_upper95")
        if docs is None:
            failures.append(f"the scan did not report docs_{name}")
        elif docs:
            failures.append(f"{docs} document(s) hit the {name!r} index")

    if supply is not None:
        scanned = {row.get("source"): row.get("source_tokens")
                   for row in payload.get("per_source") or []}
        observed["scan_source_tokens"] = scanned
        for name in sorted(supply):
            if name not in scanned:
                failures.append(f"the scan did not cover {name!r}")
            elif scanned[name] != supply[name]:
                failures.append(
                    f"the scan saw {name!r} at {scanned[name]:,} tokens and "
                    f"the corpus holds {supply[name]:,}, so "
                    f"{abs(supply[name] - scanned[name]):,} were never scanned")
        extra = sorted(name for name in scanned if name not in supply)
        if extra:
            failures.append(
                f"the scan covered {extra}, which this corpus does not hold")

    sampled = totals.get("sampled_frac")
    bound = max((totals.get(f"doc_rate_{name}_upper95") or 0.0)
                for name in _INDEX_NAMES)
    detail = "; ".join(failures) if failures else (
        f"zero hits over {sampled:.4%} of the corpus by tokens; the 95% upper "
        f"bound on the document rate is {bound:.2e}, so this bounds the "
        f"contamination rate rather than proving it zero"
        if isinstance(sampled, float) else "")
    return _verdict("corpus-contamination", not failures, observed, [path],
                    detail)


def epoch_cap_verdict(summary: dict, *, max_epochs: float = MAX_EPOCHS,
                      evidence: Sequence[str] = ()) -> dict:
    """No source may be re-read more than `max_epochs` times at this budget."""

    per_source = summary.get("per_source") or {}
    epochs = {name: row["epochs"] for name, row in per_source.items()
              if row.get("epochs") is not None}
    over = sorted((name for name, value in epochs.items()
                   if value > max_epochs + _EPOCH_TOL),
                  key=lambda name: -epochs[name])
    observed = {
        "epochs": {name: epochs[name] for name in sorted(epochs)},
        "max_epochs_seen": summary.get("max_epochs_seen"),
        "most_repeated_source": summary.get("most_repeated_source"),
        "capped_sources": summary.get("capped_sources"),
        "budget_tokens": summary.get("total_run_tokens"),
        "max_epochs": max_epochs,
    }
    detail = "; ".join(f"{name} would be read {epochs[name]:.3g}x"
                       for name in over)
    return _verdict("epoch-cap", not over, observed, evidence, detail)


def mixture_skew_verdict(summary: dict, *, max_skew_pts: float = MAX_SKEW_PTS,
                         max_epochs: float = MAX_EPOCHS,
                         evidence: Sequence[str] = ()) -> dict:
    """The blueprint's mixture must actually be delivered, and provably so.

    The fallback guard is not a second epoch check wearing a different name.
    The epoch criterion above asks whether repetition is bounded; this one asks
    whether the skew number is *readable*, and in the all-capped case it is not
    -- it reads 0.00 whatever the corpus looks like. So a corpus that trips the
    fallback fails here even though its skew is nominally perfect, and the
    detail says which of the two it was.
    """

    skew = summary.get("l1_skew_pts")
    seen = summary.get("max_epochs_seen")
    all_capped = seen is not None and seen > max_epochs + _EPOCH_TOL
    over_skew = skew is None or skew > max_skew_pts

    observed = {
        "l1_skew_pts": skew,
        "max_skew_pts": max_skew_pts,
        "all_capped_fallback": all_capped,
        "max_epochs_seen": seen,
        "target_shares": {name: row.get("target_share")
                          for name, row in sorted(
                              (summary.get("per_source") or {}).items())},
        "effective_shares": {name: row.get("effective_share")
                             for name, row in sorted(
                                 (summary.get("per_source") or {}).items())},
    }
    failures: List[str] = []
    if all_capped:
        failures.append(
            f"every source is over the {max_epochs:g}-epoch limit, so the "
            f"target shares were kept unchanged and the skew reads "
            f"{skew} by construction rather than by measurement")
    if over_skew:
        failures.append(f"the sampled mixture is {skew} pts from target, "
                        f"over the {max_skew_pts:g}-pt bound")
    return _verdict("mixture-skew", not failures, observed, evidence,
                    "; ".join(failures))


def manifest_provenance_verdict(shards_root, sources: Sequence[str]) -> dict:
    """Every source must be rebuildable from its own manifest alone.

    Pinned means the manifest names the bytes it read: either an explicit
    `source_revision`, or the commit the Hub actually served, recorded at build
    time in `source_release`. A manifest with neither describes a build against
    "whatever the default branch pointed at that day", which is not a revision.

    Reproducible additionally means the transformation is recorded -- the
    filters that ran and the git sha of the tree that ran them. Both live in
    `daedalus/dataprep.py::source_provenance`; this reads for them rather than
    trusting that the code that writes them ran.
    """

    observed: Dict[str, dict] = {}
    evidence: List[str] = []
    failures: List[str] = []
    for name in sorted(sources):
        path = Path(shards_root) / name / "manifest.json"
        evidence.append(str(path))
        try:
            payload = _read_json(path)
        except (OSError, ValueError) as exc:
            observed[name] = {"readable": False}
            failures.append(f"{name}: {exc}")
            continue
        release = payload.get("source_release") or {}
        revision = payload.get("source_revision")
        # `sha` is what `dataprep.resolve_source_release` writes -- the commit
        # the Hub served on the day the source was built. This read `resolved_
        # commit`, a name no writer produces, so the criterion could not pass on
        # a correctly provenanced manifest; and it looked right for months
        # because the corpus it was run against carries no `source_release` at
        # all, so it failed for the reason it was meant to and not for this one.
        # A one-source rebuild under `--tokenizer` is what surfaced it: full
        # provenance on disk, `no source_revision and no resolved commit` in the
        # verdict. The alias is kept for a manifest written to the other shape.
        resolved = release.get("sha") or release.get("resolved_commit")
        row = {
            "readable": True,
            "source_revision": revision,
            "resolved_commit": resolved,
            "license": release.get("license"),
            "has_filters": isinstance(payload.get("filters"), dict),
            "builder_git_sha": payload.get("builder_git_sha"),
        }
        observed[name] = row
        missing = []
        if not (revision or resolved):
            missing.append("no source_revision and no resolved commit")
        if not row["has_filters"]:
            missing.append("no filters block")
        if not row["builder_git_sha"]:
            missing.append("no builder_git_sha")
        if missing:
            failures.append(f"{name}: " + ", ".join(missing))

    if not sources:
        return _verdict("manifest-provenance", False, observed, evidence,
                        "no source was named, so nothing was checked")
    return _verdict("manifest-provenance", not failures, observed, evidence,
                    "; ".join(failures))


# ------------------------------------------------------------------ the gate --

def source_supply(shards_root, name: str, *,
                  local: bool = False) -> tuple[int, str]:
    """`(tokens, which figure that was)` for one source.

    A shard directory on a work box is routinely a *fetch* of a built source
    rather than the source itself -- `scripts/fetch_corpus_subset.py` pulls a
    few shards so a proxy can train, and records what it took them from in
    `subset_of`. `total_tokens` is then the tokens present here, which is the
    right number for "what can this box train on" and the wrong one for "how
    often would a successor re-read this source".

    Reading the local figure for the epoch question understates every source's
    supply by however much of it was left on the Hub -- on this box, 10x to 30x
    -- and turns a corpus that is comfortably inside the cap into one that
    appears to blow through it tenfold. So the built figure is the default, and
    which one was used is recorded per source rather than inferred from the
    verdict.
    """

    payload = _read_json(Path(shards_root) / name / "manifest.json")
    present = int(payload["total_tokens"])
    built = (payload.get("subset_of") or {}).get("total_tokens")
    if local or not built:
        return present, "total_tokens"
    return int(built), "subset_of.total_tokens"


def mixture_allocation(shards_root: str, *, budget_tokens: int,
                       max_epochs: float = MAX_EPOCHS,
                       weights: Optional[Dict[str, float]] = None,
                       local_supply: bool = False,
                       drop_sources: Sequence[str] = ()) -> dict:
    """What this corpus would deliver at `budget_tokens`.

    `drop_sources` removes a source from the mixture and lets `resolve_mixture`
    renormalize the rest, which is how the plan's "remove the tiny dialogue
    source" is asked as a question rather than answered by retyping nine
    weights out of the blueprint -- the step at which a mixture stops being the
    blueprint's.

    The cap and the summary are `train.py`'s own, not a second implementation:
    a gate that modelled the allocation differently from the trainer would
    eventually pass a corpus the trainer refuses, or the reverse. What differs
    is only the supply figure fed to them -- see `source_supply` -- so
    `resolve_mixture` is asked for the renormalized target shares alone, with
    no budget, and the capping is done here against the built totals.

    `cap_weights_by_epochs` narrates the all-capped case on stdout. This
    command's stdout is the verdict document, so that line is captured; nothing
    is lost, because `all_capped_fallback` is recorded in the verdict itself.
    """

    import contextlib
    import io

    from train import (cap_weights_by_epochs, resolve_mixture,
                       summarize_mixture)

    if drop_sources:
        if weights is None:
            from daedalus.dataprep import MIXTURE
            weights = {source.key: source.share for source in MIXTURE}
        weights = {name: share for name, share in weights.items()
                   if name not in set(drop_sources)}

    names, target_probs, _, _ = resolve_mixture(
        shards_root, None, max_epochs=max_epochs, weights=weights,
        verbose=False)
    supply: Dict[str, int] = {}
    basis: Dict[str, str] = {}
    present: Dict[str, int] = {}
    for name in names:
        supply[name], basis[name] = source_supply(shards_root, name,
                                                  local=local_supply)
        present[name], _ = source_supply(shards_root, name, local=True)

    narration = io.StringIO()
    with contextlib.redirect_stdout(narration):
        probs = cap_weights_by_epochs(target_probs, supply, budget_tokens,
                                      max_epochs)
    summary = summarize_mixture(names, target_probs, probs, supply,
                                budget_tokens, max_epochs)
    summary["supply_basis"] = basis
    summary["tokens_present_locally"] = present
    return summary


def run_gate(*, shards_root: str, budget_tokens: int,
             decontam_index: str, contam_scan: str,
             max_epochs: float = MAX_EPOCHS,
             max_skew_pts: float = MAX_SKEW_PTS,
             weights: Optional[Dict[str, float]] = None,
             weights_source: Optional[str] = None,
             local_supply: bool = False,
             drop_sources: Sequence[str] = ()) -> dict:
    summary = mixture_allocation(shards_root, budget_tokens=budget_tokens,
                                 max_epochs=max_epochs, weights=weights,
                                 local_supply=local_supply,
                                 drop_sources=drop_sources)
    sources = sorted(summary.get("per_source") or {})
    allocation_evidence = [str(Path(shards_root) / name / "manifest.json")
                           for name in sources]

    supply = {name: row["tokens_on_disk"]
              for name, row in (summary.get("per_source") or {}).items()
              if row.get("tokens_on_disk") is not None}

    criteria = [
        decontam_index_verdict(decontam_index),
        contamination_verdict(contam_scan, supply),
        epoch_cap_verdict(summary, max_epochs=max_epochs,
                          evidence=allocation_evidence),
        mixture_skew_verdict(summary, max_skew_pts=max_skew_pts,
                             max_epochs=max_epochs,
                             evidence=allocation_evidence),
        manifest_provenance_verdict(shards_root, sources),
    ]
    return {
        "schema": GATE_SCHEMA,
        "gate": "phase7-corpus",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": all(item["passed"] for item in criteria),
        "shards_root": str(shards_root),
        "budget_tokens": int(budget_tokens),
        "sources": sources,
        "dropped_sources": sorted(drop_sources),
        "supply": "local" if local_supply else "built",
        "weights_source": weights_source or "daedalus.dataprep.MIXTURE",
        "allocation": summary,
        "criteria": criteria,
    }


def write_verdict(path, verdict: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", default="data/shards")
    parser.add_argument(
        "--budget-tokens", type=int, required=True,
        help="the successor budget the epoch cap is evaluated at. Required: "
             "epochs are only defined against one, and a default would pick a "
             "successor size the operator has deliberately not fixed")
    parser.add_argument("--max-epochs", type=float, default=MAX_EPOCHS)
    parser.add_argument("--max-skew-pts", type=float, default=MAX_SKEW_PTS)
    parser.add_argument(
        "--weights-from", default=None,
        help="a mixture artifact (`derive` or the phase 7 verdict) whose "
             "weights the allocation uses; without it the blueprint mixture "
             "in daedalus.dataprep is resolved")
    parser.add_argument(
        "--drop-source", action="append", default=[], dest="drop_sources",
        metavar="SOURCE",
        help="leave this source out of the mixture and renormalize the rest; "
             "repeatable. The plan's step 4 (remove the tiny dialogue source "
             "from general pretraining) is asked with this")
    parser.add_argument(
        "--local-supply", action="store_true",
        help="count only the tokens present under --shards-root, ignoring the "
             "`subset_of` figure a fetched shard directory records. Answers "
             "'what can this box train on' rather than 'what does the corpus "
             "hold', which is not the question the epoch cap asks")
    parser.add_argument("--decontam-index",
                        default="runs/corpus/decontam-index.json")
    parser.add_argument("--contam-scan",
                        default="runs/preflight/contam-exposure.json")
    parser.add_argument("--out", default="runs/corpus/phase7-gate.json")
    args = parser.parse_args(argv)

    from scripts.mixture_opt import _read_derived

    verdict = run_gate(
        shards_root=args.shards_root, budget_tokens=args.budget_tokens,
        decontam_index=args.decontam_index, contam_scan=args.contam_scan,
        max_epochs=args.max_epochs, max_skew_pts=args.max_skew_pts,
        weights=_read_derived(args.weights_from),
        weights_source=args.weights_from,
        local_supply=args.local_supply,
        drop_sources=args.drop_sources)
    write_verdict(args.out, verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    for item in verdict["criteria"]:
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['criterion']}"
              f"{': ' + item['detail'] if item['detail'] else ''}",
              file=sys.stderr)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
