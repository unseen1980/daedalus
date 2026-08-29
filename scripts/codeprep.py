"""Build and check the code corpus's decontamination index and admission gate.

    python scripts/codeprep.py decontam build
    python scripts/codeprep.py decontam verify --json-out runs/codeprep/decontam.json
    python scripts/codeprep.py corpus probe --rows 2000 --json-out runs/codeprep/probe.json

`corpus probe` is the measurement the corpus build's assumptions rest on. Every
row-shape assumption in `daedalus/codeprep.py` -- which key holds the
repository, which licence strings exist -- fails silently and in the same
direction: the gate refuses every row and the build writes an empty shard
directory with a zero exit. The probe reads real rows of each language config
and reports what was actually there, so that failure happens in two minutes
rather than after a night of streaming.

`build` loads every item of every code benchmark phase 8 is gated on, writes the
sorted n-gram set and its provenance sidecar, and prints the digest a code
corpus manifest should pin. `verify` re-reads a written index, recomputes the
digest, and checks its coverage -- and its `n` -- against what is scored today.

The two are different questions and they go stale differently: `build` asks
"did this build cover the benchmarks", `verify` asks "does the index this
manifest names still cover them, and will the corpus filter actually look its
n-grams up". See `daedalus/codeprep.py` for why the second half of that matters,
and `scripts/decontam_index.py` for the general index this one sits beside.
"""
import argparse
import json
import os
import sys
from typing import Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from daedalus.codeprep import (CODE_LANGUAGE_SHARES,  # noqa: E402
                               DEFAULT_CODE_INDEX_PATH, DEFAULT_CODE_N,
                               DEFAULT_HOLDOUT_CAP_TOKENS,
                               DEFAULT_HOLDOUT_FRAC, DEFAULT_INTERLEAVE_PASSES,
                               GITHUB_CODE_LANGUAGES, INTERLEAVED_CONFIG,
                               IncompleteIndex, IndexDigestMismatch,
                               GITHUB_CODE_DATASET, GITHUB_CODE_REVISION,
                               MIN_BUCKET_SHARE, CODE_BYTES_PER_TOKEN,
                               CORPUS_SHARES, SPLIT_SALT, build_code_index,
                               bucket_supply, checkpoint_every_for,
                               code_coverage_problems,
                               code_manifest_record, code_token_budget,
                               config_near_misses, config_row_counts,
                               corpus_specs, github_code_configs, load_index,
                               merge_gate_manifest, missing_configs,
                               plan_problems, probe_languages, probe_problems,
                               probe_record, sidecar_path, source_plan,
                               write_index)


def _report(provenance: dict) -> str:
    lines = [f"  {provenance['ngrams']:,} {provenance['n']}-grams  "
             f"{provenance['digest']}",
             f"  evalplus {provenance.get('evalplus')}"]
    for name, meta in sorted((provenance.get("benchmarks") or {}).items()):
        lines.append(f"  {name:16s} {meta['items']:>5,} items")
        for field, count in sorted((meta.get("fields") or {}).items()):
            short = (meta.get("short_fields") or {}).get(field, 0)
            # Reported on the same line as the field it belongs to, because
            # "4,102 n-grams from canonical_solution" and "and 91 solutions were
            # too short to contribute any" are one fact about coverage, and
            # separating them is how the second gets dropped from a summary.
            note = "" if not short else f"   ({short:,} too short to index)"
            lines.append(f"      {field:12s} {count:>8,} n-grams{note}")
        for field, short in sorted((meta.get("short_fields") or {}).items()):
            if field not in (meta.get("fields") or {}):
                lines.append(f"      {field:12s} {'-':>8}          "
                             f"({short:,} too short to index)")
    return "\n".join(lines)


def _build(a) -> int:
    print(f"loading every item of every code benchmark at n={a.n} ...",
          flush=True)
    try:
        ngrams, provenance = build_code_index(n=a.n)
    except IncompleteIndex as e:
        print("REFUSE: the index would not cover what phase 8 is gated on:",
              file=sys.stderr)
        for problem in e.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    write_index(a.out, ngrams, provenance)
    print(f"wrote {a.out} and {sidecar_path(a.out)}")
    print(_report(provenance))
    print(f"\npin it with:\n  --code-index {a.out} "
          f"--code-index-digest {provenance['digest']}")
    return 0


def _verify(a) -> int:
    try:
        ngrams, provenance = load_index(a.out, expect_digest=a.expect_digest)
    except (OSError, IndexDigestMismatch) as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    print(f"{a.out}: {len(ngrams):,} {provenance['n']}-grams")
    print(_report(provenance))
    problems = code_coverage_problems(provenance)
    if a.json_out:
        # Written before the verdict is returned, so a failing verify still
        # leaves the record of *why* rather than only a non-zero exit.
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump({"path": a.out, "provenance": provenance,
                       "problems": problems}, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {a.json_out}")
    if problems:
        print("\nthis index no longer covers what phase 8 is gated on:",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 3
    print("\ncoverage: complete against today's code benchmarks")
    return 0


def _configs(a) -> int:
    print(f"listing parquet directories on {a.dataset} @ {a.revision} ...",
          flush=True)
    try:
        available = github_code_configs(dataset=a.dataset, revision=a.revision)
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        print(f"REFUSE: could not list the repository: {e!r}", file=sys.stderr)
        return 2
    print(f"  {len(available):,} directories carry parquet files")
    for name, count in available.items():
        print(f"      {name:24s} {count:>5,} files")

    # Only meaningful against the dataset the buckets are defined over; another
    # repository has its own layout and "missing" would be noise about it.
    missing = (missing_configs(available)
               if a.dataset == GITHUB_CODE_DATASET else {})
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump({"dataset": a.dataset, "revision": a.revision,
                       "available": available, "missing": missing,
                       "near_misses": {config: config_near_misses(config, available)
                                       for configs in missing.values()
                                       for config in configs}},
                      f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {a.json_out}")
    if missing:
        print("\nbuckets this module names a directory that does not exist for:",
              file=sys.stderr)
        for bucket, configs in missing.items():
            for config in configs:
                near = config_near_misses(config, available)
                hint = (f" -- did you mean {', '.join(near)}?" if near else
                        " -- no near miss; this config was never converted")
                print(f"  - {bucket}/{config}{hint}", file=sys.stderr)
        return 3
    if a.dataset != GITHUB_CODE_DATASET:
        # Not "every bucket's directory exists": the buckets were never checked
        # against this repository, and saying so would be the reassuring kind of
        # success line -- `codeparrot/github-code-clean` printed exactly that
        # while carrying none of the four languages the listing was run to find.
        print(f"\nlisted {a.dataset}; its layout was not checked against this "
              f"module's buckets, which are defined over {GITHUB_CODE_DATASET}")
        return 0
    print("\nevery bucket's directory exists on this revision")
    return 0


def _probe_report(report: dict) -> str:
    lines = [f"  holdout {report['holdout_frac']:.1%} of repositories, salt "
             f"{report['split_salt']!r}, {report['rows_per_config']:,} rows per config"]
    for bucket, entry in sorted(report["languages"].items()):
        lines.append(f"  {bucket:24s} share {entry['share']:.0%}")
        for record in entry["configs"]:
            if not record.get("resolved"):
                lines.append(f"      {record['config']:16s} UNRESOLVED  "
                             f"{record.get('error', 'no rows')}")
                continue
            admitted = record["admitted"]
            lines.append(
                f"      {record['config']:16s} {record['rows_read']:>6,} rows  "
                f"train {admitted['train']:>6,}  holdout {admitted['holdout']:>5,}  "
                f"repos {record['repositories']['train'] + record['repositories']['holdout']:>6,}")
            kept = record.get("languages_kept")
            if kept:
                # The yield, on the line that reports the filter that caused it.
                # A share is only affordable if the amplification is, and the
                # two numbers are one fact about whether this bucket can be
                # served from this directory at all.
                amplification = record.get("stream_amplification")
                megabytes = sum(record.get("admitted_bytes", {}).values()) / 1e6
                lines.append(
                    f"          kept {', '.join(kept)}: "
                    f"{'no rows' if amplification is None else f'{amplification:,.1f} rows streamed per row kept'}"
                    f", {megabytes:,.1f} MB admitted")
            field = ", ".join(sorted(record["repository_fields"])) or "NONE"
            lines.append(f"          repository field: {field}")
            seen = record.get("languages") or {}
            top = ", ".join(f"{name}={count:,}" for name, count
                            in list(seen.items())[:8])
            lines.append(f"          languages ({len(seen)}): {top or 'none reported'}")
            licenses = ", ".join(f"{name or '<empty>'}={count:,}" for name, count
                                 in sorted(record["licenses"].items(),
                                           key=lambda kv: -kv[1])[:8])
            lines.append(f"          licences: {licenses}")
            refused = ", ".join(f"{name}={count:,}" for name, count
                                in sorted(record["refused"].items()) if count)
            lines.append(f"          refused: {refused or 'nothing'}")
    return "\n".join(lines)


def _probe(a) -> int:
    if a.config and a.language:
        print("REFUSE: --config probes named directories and --language probes "
              "the mixture's buckets; pass one or the other", file=sys.stderr)
        return 2
    what = a.config or (a.language or sorted(GITHUB_CODE_LANGUAGES))
    kept = f", keeping {', '.join(a.keep_language)}" if a.keep_language else ""
    print(f"probing {len(what)} "
          f"{'directory' if a.config else 'bucket'}(s) at {a.rows:,} rows each"
          f"{kept} ...", flush=True)
    try:
        report = probe_languages(a.language, configs=a.config, rows=a.rows,
                                 holdout_frac=a.holdout_frac,
                                 keep_languages=a.keep_language)
    except ValueError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    problems = probe_problems(report)
    report["problems"] = problems
    print(_probe_report(report))
    if a.json_out:
        # Written before the verdict, so a failing probe leaves the record of
        # *what was actually in the rows* rather than only a non-zero exit --
        # which is the entire output anyone needs to fix the assumption.
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {a.json_out}")
    if problems:
        print("\nthe corpus build would not do what it says on these sources:",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 3
    print("\nevery bucket resolved, and every licence value is one this gate "
          "classifies")
    return 0


def _plan_report(plan: dict) -> str:
    interleaved = plan["interleaved"]
    lines = [
        f"  budget {plan['interleave_passes']:g} pass(es) of "
        f"{interleaved['config']} per byte of code corpus; buckets under "
        f"{plan['min_bucket_share']:.1%} are dropped",
        f"  measured over {interleaved['rows_read'] or 0:,} rows, "
        f"{(interleaved['admitted_bytes'] or 0) / 1e6:,.1f} MB admitted, "
        f"{interleaved['stream_amplification']} rows streamed per row kept",
        "",
        f"  {'bucket':24s} {'plan':>7s} {'source':>13s} {'share':>7s} "
        f"{'needs':>9s}  why",
    ]
    for bucket, entry in sorted(plan["buckets"].items(),
                                key=lambda kv: -kv[1]["plan_share"]):
        required = entry.get("required_passes")
        lines.append(
            f"  {bucket:24s} {entry['plan_share']:>6.1%} "
            f"{entry['source']:>13s} {entry['share']:>6.1%} "
            f"{'-' if required is None else f'{required:>8.1f}x'}  "
            f"{entry['reason']}")
    lines.append("")
    lines.append(f"  redistributed {plan['redistributed']:.1%} onto the buckets "
                 f"with directories of their own")
    return "\n".join(lines)


def _plan(a) -> int:
    try:
        with open(a.configs_json) as f:
            available = json.load(f).get("available") or {}
        with open(a.probe_json) as f:
            report = json.load(f)
    except OSError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    record = probe_record(report, a.interleaved)
    if record is None:
        # Not an empty directory: a report that never measured it. Reading the
        # two apart is the whole point of the fallback measurement.
        print(f"REFUSE: {a.probe_json} contains no probe of {a.interleaved!r}; "
              f"run `corpus probe --config {a.interleaved}` first",
              file=sys.stderr)
        return 2
    try:
        plan = source_plan(available=available, interleaved=record,
                           passes=a.passes, min_share=a.min_share)
    except ValueError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    problems = plan_problems(plan)
    plan["problems"] = problems
    plan["evidence"] = {"configs": a.configs_json, "probe": a.probe_json}
    print(_plan_report(plan))
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(plan, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {a.json_out}")
    if problems:
        print("\nthis plan cannot be built as written:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 3
    return 0


#: Training budgets phase 8's gates are set at, in total tokens. The code
#: portion of each is what a code directory has to supply.
DEFAULT_CODE_BUDGETS = (250_000_000, 1_000_000_000, 3_000_000_000)


def _headroom(a) -> int:
    # Phase 7 already answers "can this source fill its share without being
    # re-read past the cap", down to the verdict strings and the shortfall
    # arithmetic. What phase 8 lacks is the *supply*, because there are no shard
    # manifests to count yet -- so it measures one and borrows the rest.
    from scripts.source_headroom import EPOCH_CAP, Supply, epoch_curve

    try:
        with open(a.plan_json) as f:
            plan = json.load(f)
        probes: dict = {}
        for path in a.probe_json:
            with open(path) as f:
                report = json.load(f)
            for entry in (report.get("languages") or {}).values():
                for record in entry.get("configs") or []:
                    if record.get("config"):
                        probes[record["config"]] = record
    except OSError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    configs = sorted({config for entry in (plan.get("buckets") or {}).values()
                      for config in (entry.get("configs") or [])
                      if entry.get("source") == "directories"}
                     | {entry["source_config"]
                        for entry in (plan.get("buckets") or {}).values()
                        if entry.get("source") == "interleaved"})
    if a.rows_json and os.path.exists(a.rows_json) and not a.remeasure:
        with open(a.rows_json) as f:
            row_counts = json.load(f)
        print(f"row counts from {a.rows_json}")
    else:
        print(f"reading parquet footers for {len(configs)} directory(s) ...",
              flush=True)
        row_counts = config_row_counts(configs, revision=a.revision)
        if a.rows_json:
            os.makedirs(os.path.dirname(a.rows_json) or ".", exist_ok=True)
            with open(a.rows_json, "w") as f:
                json.dump(row_counts, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"wrote {a.rows_json}")

    supply = bucket_supply(plan=plan, row_counts=row_counts, probes=probes,
                           bytes_per_token=a.bytes_per_token)
    supplies = {bucket: Supply(key=bucket, unique_tokens=entry["unique_tokens"],
                               realized_tokens=0, basis=entry["basis"])
                for bucket, entry in supply.items()}
    mixture = [(bucket, entry["share"], 1) for bucket, entry in supply.items()
               if entry["share"] > 0]
    # The code *portion* of each budget, since the rest of the mixture is
    # technical prose and general replay and no code directory supplies it.
    code_share = CORPUS_SHARES["code"]
    curve = epoch_curve(supplies, mixture,
                        budgets=[int(round(total * code_share))
                                 for total in a.budget_tokens],
                        epoch_cap=a.epoch_cap)

    print(f"\n  {a.bytes_per_token:g} bytes/token of code, {code_share:.0%} of "
          f"each budget is code, epoch cap {a.epoch_cap:g}")
    for bucket, entry in sorted(supply.items(),
                                key=lambda kv: -kv[1]["share"]):
        print(f"  {bucket:24s} share {entry['share']:6.1%}  "
              f"{entry['unique_tokens'] / 1e6:9,.1f}M unique tokens   "
              f"{entry['basis']}")
    for point in curve:
        totals = point["totals"]
        print(f"\n  code budget {point['budget'] / 1e6:,.0f}M tokens  "
              f"({point['budget'] / code_share / 1e6:,.0f}M total)  "
              f"{totals['verdict']}")
        for row in point["sources"]:
            print(f"      {row.line()}")

    bounded = sorted({config for entry in supply.values()
                      for config in entry["partial"]})
    if bounded:
        print(f"\n  {', '.join(bounded)} was counted from some of its files, so "
              f"every bucket drawn from it is a floor rather than a "
              f"measurement; re-run with --remeasure to close it")

    short = [point for point in curve if point["totals"]["verdict"] != "SUPPORTED"]
    record = {"bytes_per_token": a.bytes_per_token, "epoch_cap": a.epoch_cap,
              "code_share": code_share, "supply": supply,
              "row_counts": row_counts,
              "budgets": [{"budget": point["budget"],
                           "totals": {key: value for key, value
                                      in point["totals"].items()},
                           "sources": [vars(row) for row in point["sources"]]}
                          for point in curve],
              "evidence": {"plan": a.plan_json, "probes": list(a.probe_json)}}
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        print(f"\nwrote {a.json_out}")
    if short:
        print(f"\n{len(short)} budget(s) cannot be filled inside the epoch cap:",
              file=sys.stderr)
        for point in short:
            totals = point["totals"]
            binding = supply.get(totals["binding_source"]) or {}
            floor = (" -- on a lower-bound supply, so re-measure "
                     f"{', '.join(binding['partial'])} before believing it"
                     if binding.get("lower_bound") else "")
            print(f"  - {point['budget'] / 1e6:,.0f}M code tokens: "
                  f"{totals['binding_source']} would be read "
                  f"{totals['binding_epochs']:,.1f} times{floor}",
                  file=sys.stderr)
        return 3
    return 0


#: The general corpus's frozen index. Unioned with the code one rather than
#: replaced by it: a code corpus is still training data for a model scored on
#: the five general tasks, and dropping the general filter to add the code one
#: would decontaminate against the benchmarks phase 8 added and re-contaminate
#: against the ones it inherited.
DEFAULT_EVAL_INDEX_PATH = "data/decontam/eval-index-13gram.txt.gz"


def _load_union_index(a) -> Tuple[set, dict]:
    """`(ngrams, record)` -- the one set the corpus is filtered against.

    Both indexes are the same kind of file at the same `n` (see
    `daedalus/codeprep.py`'s module docstring), so the union is a set union and
    the predicate that consumes it does not know there were two. What the
    manifest carries is both provenance records, because "filtered against some
    index" is the claim the released corpus could not improve on.
    """
    from daedalus.eval_index import coverage_problems as eval_coverage_problems
    from daedalus.eval_index import load_index as load_eval_index
    from daedalus.eval_index import manifest_record as eval_manifest_record

    code_ngrams, code_provenance = load_index(
        a.code_index, expect_digest=a.code_index_digest)
    problems = code_coverage_problems(code_provenance)
    if problems:
        raise ValueError(f"{a.code_index} does not cover what phase 8 is gated "
                         f"on: {'; '.join(problems)}")
    record = {"code": code_manifest_record(code_provenance, path=a.code_index)}
    ngrams = set(code_ngrams)

    if a.eval_index:
        eval_ngrams, eval_provenance = load_eval_index(
            a.eval_index, expect_digest=a.eval_index_digest)
        problems = eval_coverage_problems(eval_provenance)
        if problems:
            raise ValueError(f"{a.eval_index} does not cover what this model is "
                             f"scored on: {'; '.join(problems)}")
        record["eval"] = eval_manifest_record(eval_provenance, path=a.eval_index)
        ngrams |= eval_ngrams
    else:
        # Recorded, not omitted: a corpus filtered against the code benchmarks
        # alone is a different artifact from one filtered against both, and an
        # absent field reads as an older manifest.
        record["eval"] = None
    record["ngrams"] = len(ngrams)
    return ngrams, record


def _build_order(holdout, train):
    """Holdout before train, source by source.

    Both sides share a `near_dup_group`, and the near-duplicate filter is a
    window over stream order rather than the whole corpus. Running the small
    side first means the train pass meets a filter that still holds the holdout
    documents, so a training file that near-duplicates a holdout file is dropped
    from *training* -- the direction that protects the measurement. The reverse
    order drops the holdout copy instead, which shrinks the holdout and leaves
    the leak where it was.

    Paired by position rather than re-sorted, because `corpus_specs` derives
    both sides from the same plan in the same order; a pairing that disagreed
    with that would put one bucket's holdout beside another's train.
    """
    if [(s.key, s.config) for s in holdout] != [(s.key, s.config) for s in train]:
        raise ValueError("the two sides of the split describe different sources")
    return [source for pair in zip(holdout, train) for source in pair]


def _uncap_exhaustion(stats: dict, max_docs: Optional[int]) -> dict:
    """Under a document cap, "this source has no more documents" is not known.

    `run_source` records exhaustion from the stream ending, and under
    `max_docs` the stream ends at the cap -- so a 25-document smoke manifested
    `Java-all`, a directory with millions of rows left in it, as exhausted. That
    is the reassuring kind of false record: a later reader takes it as a
    measurement of the source rather than of the cap.
    """
    if max_docs and stats.pop("exhausted", None):
        stats["stopped_by_doc_cap"] = True
    return stats


def _source_row(source, stats: dict) -> str:
    achieved = stats.get("achieved_fraction") or 0.0
    note = ""
    if stats.get("error"):
        note = f"  ERROR {stats['error'][:60]}"
    elif stats.get("exhausted"):
        note = "  EXHAUSTED"
    elif stats.get("stopped_by_doc_cap"):
        note = "  CAPPED"
    elif stats.get("incomplete"):
        note = "  INCOMPLETE"
    return (f"  {source.split:8s} {source.key:28s} {source.config:16s} "
            f"{stats.get('tokens', 0) / 1e6:>9,.1f}M of "
            f"{source.token_budget / 1e6:>8,.1f}M  {achieved:>6.1%}{note}")


def _build_corpus(a) -> int:
    from daedalus.data import get_tokenizer
    from daedalus.dataprep import (DedupState, _recover_source_stats,
                                   resolve_source_release, run_source)

    try:
        with open(a.plan_json) as f:
            plan = json.load(f)
        supply = {}
        if a.headroom_json and os.path.exists(a.headroom_json):
            with open(a.headroom_json) as f:
                supply = json.load(f).get("supply") or {}
    except OSError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    try:
        sides = {split: corpus_specs(
            plan=plan, split=split, total_tokens=a.total_tokens, supply=supply,
            holdout_frac=a.holdout_frac, salt=a.split_salt,
            holdout_cap_tokens=a.holdout_cap_tokens)
            for split in ("holdout", "train")}
        sources = _build_order(sides["holdout"], sides["train"])
    except ValueError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    code_tokens = code_token_budget(a.total_tokens)
    print(f"{a.total_tokens / 1e6:,.0f}M total tokens, "
          f"{CORPUS_SHARES['code']:.0%} code = {code_tokens / 1e6:,.0f}M, "
          f"holdout {a.holdout_frac:.1%} of repositories capped at "
          f"{a.holdout_cap_tokens / 1e6:,.1f}M per bucket")
    if not supply:
        print(f"  no per-directory supply from {a.headroom_json}; multi-directory "
              f"buckets are split evenly")
    for source in sources:
        print(f"  {source.split:8s} {source.key:28s} {source.config:16s} "
              f"{source.token_budget / 1e6:>8,.1f}M  {source.basis}")
    if a.dry_run:
        print("\n--dry-run: nothing was streamed")
        return 0

    try:
        ngrams, decontam = _load_union_index(a)
    except (OSError, ValueError, IndexDigestMismatch) as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    print(f"\ndecontamination: {decontam['ngrams']:,} n-grams "
          f"({decontam['code']['ngrams']:,} code + "
          f"{(decontam['eval'] or {}).get('ngrams', 0):,} general)")

    tokenizer = get_tokenizer(a.tokenizer)
    dedup = DedupState()
    manifest = {
        "schema": 1,
        "total_tokens": a.total_tokens,
        "code_tokens": code_tokens,
        "corpus_shares": dict(CORPUS_SHARES),
        "out_root": a.out_root,
        "holdout_frac": a.holdout_frac,
        "split_salt": a.split_salt,
        "holdout_cap_tokens": a.holdout_cap_tokens,
        "tokenizer": a.tokenizer,
        "decontam_index": decontam,
        "evidence": {"plan": a.plan_json, "headroom": a.headroom_json},
        "max_docs_per_source": a.max_docs_per_source,
        "sources": [],
    }

    def write_manifest() -> None:
        os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)
        tmp = a.manifest + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, a.manifest)

    rss_exceeded = False
    for source in sources:
        out_dir = source.out_dir(a.out_root)
        split_root = os.path.join(a.out_root, source.split)
        recovered = _recover_source_stats(source.key, split_root) or {}
        gate_path = os.path.join(out_dir, "gate.json")
        prior_gate = None
        if os.path.exists(gate_path):
            try:
                with open(gate_path) as f:
                    prior_gate = json.load(f)
            except (OSError, json.JSONDecodeError):
                prior_gate = None

        if rss_exceeded:
            # `_run_group_worker`'s rule, for the same reason: a process that
            # has already tripped its resident cap is holding that memory, and
            # the eighth dataprep incident was a raw C-level malloc failure
            # moments after one source's graceful cap trip took the next one
            # down hard enough to lose every other source's in-flight progress.
            # Leaving now exits non-zero, and the controller's next attempt is a
            # fresh process that resumes each source from its shards.
            print(f"=== skip {source.split}/{source.key}: an earlier source "
                  f"exceeded the RSS cap in this process ===", flush=True)
            manifest["sources"].append({
                **{k: v for k, v in vars(source).items()
                   if k not in ("spec", "gate")},
                **{k: v for k, v in recovered.items() if k != "drops"},
                "gate": prior_gate,
                "error": "skipped: an earlier source exceeded the RSS cap in "
                         "this process"})
            write_manifest()
            continue

        if a.resume and recovered.get("tokens", 0) >= source.token_budget > 0:
            print(f"=== skip {source.split}/{source.key}: "
                  f"{recovered['tokens']:,} tokens already on disk meets its "
                  f"{source.token_budget:,} budget ===", flush=True)
            manifest["sources"].append({
                **{k: v for k, v in vars(source).items()
                   if k not in ("spec", "gate")},
                **{k: v for k, v in recovered.items() if k != "drops"},
                "achieved_fraction": recovered["tokens"] / source.token_budget,
                "gate": prior_gate, "resumed": True})
            write_manifest()
            continue

        resume: dict = {}
        if a.resume and recovered.get("tokens", 0) > 0 and recovered.get("shards"):
            if recovered.get("stream_state") or recovered.get("n_seen"):
                resume = {"resume_skip": recovered.get("n_seen", 0),
                          "resume_seed": recovered,
                          "stream_state": recovered.get("stream_state")}
                how = ("stream-position restore (O(1))"
                       if recovered.get("stream_state")
                       else f"replaying {recovered.get('n_seen', 0):,} documents")
                print(f"resuming {source.key}: {recovered['tokens']:,} tokens "
                      f"already flushed, continuing via {how}", flush=True)
            else:
                # Shards with no recorded position: appending to them would
                # duplicate documents into the corpus silently, which is worse
                # than paying for the source again. Same call `run_dataprep`
                # makes.
                print(f"{source.key}: {recovered['tokens']:,} tokens on disk "
                      f"with no resume point -- rebuilding this source",
                      flush=True)

        print(f"=== {source.split}/{source.key} ({source.config}) "
              f"budget={source.token_budget:,} tokens ===", flush=True)
        try:
            stats = run_source(
                source.spec, tokenizer, source.token_budget, out_dir, dedup,
                ngrams, shard_tokens=a.shard_tokens,
                max_docs=a.max_docs_per_source,
                rss_limit_gb=a.rss_limit_gb,
                rss_check_every=a.rss_check_every,
                # Derived from this source's own budget unless overridden: the
                # shared 50,000-document default never fires at all on a
                # holdout pass, which yields ~2% of what it streams. See
                # `checkpoint_every_for`.
                checkpoint_every=(a.checkpoint_every
                                  or checkpoint_every_for(source.token_budget)),
                progress_every=(a.progress_every
                                or max(1, checkpoint_every_for(
                                    source.token_budget) // 2)),
                resume_skip=resume.get("resume_skip", 0),
                resume_seed=resume.get("resume_seed"),
                resume_stream_state=resume.get("stream_state"),
                release=resolve_source_release(source.spec),
                tokenizer_name=a.tokenizer)
        except Exception as e:                  # noqa: BLE001 - recorded, not raised
            print(f"FAILED {source.key}: {e!r}", file=sys.stderr, flush=True)
            stats = dict(recovered) or {"key": source.key, "tokens": 0}
            stats["error"] = repr(e)
        stats = _uncap_exhaustion(stats, a.max_docs_per_source)
        if "WorkerMemoryExceeded" in str(stats.get("error") or ""):
            rss_exceeded = True

        # Merged only when the resume restored a stream *position*. The replay
        # fallback re-reads the prefix from row zero and the gate is consulted
        # on every replayed row (`_DocumentStream.__iter__` filters before it
        # skips), so this attempt's manifest already covers the whole source --
        # folding the prior one onto it would count the prefix twice and report
        # a licence histogram no corpus has.
        gate = merge_gate_manifest(
            prior_gate if resume.get("stream_state") else None,
            source.gate.manifest())
        os.makedirs(out_dir, exist_ok=True)
        with open(gate_path, "w") as f:
            json.dump(gate, f, indent=2, sort_keys=True)
            f.write("\n")
        manifest["sources"].append({
            **{k: v for k, v in vars(source).items()
               if k not in ("spec", "gate")},
            **{k: v for k, v in stats.items()
               if k not in ("stream_state", "drops")},
            "drops": stats.get("drops"),
            "gate": gate,
            "resumed": bool(resume)})
        write_manifest()
        print(_source_row(source, stats), flush=True)

    print(f"\nwrote {a.manifest}")
    for entry in manifest["sources"]:
        source = next(s for s in sources
                      if s.key == entry["key"] and s.split == entry["split"])
        print(_source_row(source, entry))
    total = sum(entry.get("tokens", 0) for entry in manifest["sources"]
                if entry["split"] == "train")
    print(f"\n  {total / 1e6:,.1f}M train tokens of a {code_tokens / 1e6:,.1f}M "
          f"code budget")

    failed = [entry for entry in manifest["sources"] if entry.get("error")]
    if failed:
        print(f"\n{len(failed)} source(s) failed:", file=sys.stderr)
        for entry in failed:
            print(f"  - {entry['split']}/{entry['key']}: {entry['error']}",
                  file=sys.stderr)
        return 3
    if a.max_docs_per_source:
        # A document-capped run is a smoke: every source is short by
        # construction, so the shortfall check below would fail it for doing
        # exactly what it was asked to do.
        print("\n--max-docs-per-source was set: this is a smoke, not a corpus")
        return 0
    short = [entry for entry in manifest["sources"]
             if entry.get("tokens", 0) < entry["token_budget"] * 0.99]
    if short:
        # `headroom` said every bucket's directories hold its share inside the
        # epoch cap. A source that stops short anyway has falsified that, and
        # the budget it was measured against is the one downstream training
        # reads -- so this exits non-zero rather than logging a line under a
        # corpus that will be used as if it were whole.
        print(f"\n{len(short)} source(s) stopped short of the budget headroom "
              f"said they could fill:", file=sys.stderr)
        for entry in short:
            print(f"  - {entry['split']}/{entry['key']}: "
                  f"{entry.get('tokens', 0):,} of {entry['token_budget']:,} "
                  f"tokens{' (stream exhausted)' if entry.get('exhausted') else ''}",
                  file=sys.stderr)
        return 3
    return 0


def _mixture(a) -> int:
    """Compose the one root a continued-pretraining arm reads, and its weights.

    The code corpus is 65% of that root and is built by `corpus build`; the
    other 35% is the original pretraining data already on this box. Neither is
    moved: `--data-dir` needs one directory holding every source, so this links
    both corpora into one and prints the shares that make the result 65/15/20.

    Everything it decides is written to `--json-out`, including the epochs each
    source is read for at the budget, because a mixture is only checkable
    against the thing it was composed for.
    """
    from daedalus.codeprep import (code_train_sources, compose_mixture_root,
                                   mixture_weight_flags, replay_buckets,
                                   resolve_source_dirs, training_mixture)

    try:
        with open(a.manifest) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    try:
        code = code_train_sources(manifest)
    except ValueError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    replay_names = sorted(name for members in replay_buckets().values()
                          for name in members)
    replay_roots = [root for root in (a.general_train_root, a.general_root)
                    if root]
    replay_dirs, missing = resolve_source_dirs(replay_names, roots=replay_roots)
    try:
        record = training_mixture(code_sources=code, present=sorted(replay_dirs))
    except ValueError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    train_dirs = {key: os.path.join(a.code_root, "train", key) for key in code}
    train_dirs.update(replay_dirs)
    holdout_dirs = {
        key: os.path.join(a.code_root, "holdout", key) for key in code
        if os.path.exists(os.path.join(a.code_root, "holdout", key,
                                       "manifest.json"))}
    # The replay sources only. The code side's holdout is the one the build
    # carved by repository above, and a general root that happened to carry a
    # same-named directory must not shadow it.
    holdout_dirs.update(resolve_source_dirs(
        sorted(name for name in record["weights"] if name not in code),
        roots=[a.general_holdout_root])[0])

    print(f"{len(code)} code source(s) at {CORPUS_SHARES['code']:.0%}, "
          f"{len(replay_dirs)} replayed at "
          f"{1 - CORPUS_SHARES['code']:.0%} "
          f"({CORPUS_SHARES['technical']:.0%} technical + "
          f"{CORPUS_SHARES['general-replay']:.0%} general)")
    for bucket, members in sorted(record["buckets"].items()):
        print(f"\n  {bucket} ({record['corpus_shares'][bucket]:.0%})")
        for name in sorted(members):
            source_dir = train_dirs.get(name, "")
            print(f"      {name:34s} {record['weights'][name]:7.4f}   "
                  f"{source_dir}")
    for bucket, names in sorted((record.get("absent") or {}).items()):
        print(f"\n  {bucket}: {', '.join(names)} not on disk; the bucket's "
              f"share stays in the bucket")
    if missing:
        print(f"  looked under {replay_roots}")

    if a.dry_run:
        print("\n--dry-run: nothing was linked")
        return 0

    try:
        composed = {
            "train": compose_mixture_root(out_root=a.out_root,
                                          sources=train_dirs, split="train"),
            "holdout": compose_mixture_root(out_root=a.out_root,
                                            sources=holdout_dirs,
                                            split="holdout"),
        }
    except (OSError, ValueError) as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    train_root = os.path.join(a.out_root, "train")
    holdout_root = os.path.join(a.out_root, "holdout")

    # `train.py`'s own resolver, not a second implementation of it: what this
    # prints has to be what the run will actually sample, including the epoch
    # cap it applies. Imported here because it pulls in torch, which the rest of
    # this script does not need.
    from train import mixture_preflight

    preflight = mixture_preflight(train_root, a.total_tokens,
                                  weights=record["weights"], verbose=False)
    print(f"\n  at {a.total_tokens / 1e6:,.0f}M tokens: l1 skew "
          f"{preflight['l1_skew_pts']:.2f} pts, most repeated "
          f"{preflight['most_repeated_source']} at "
          f"{preflight['max_epochs_seen']:.2f} epochs")
    for name, row in sorted(preflight["per_source"].items()):
        capped = "  CAPPED" if row["capped"] else ""
        print(f"      {name:34s} {row['target_share']:7.4f} -> "
              f"{row['effective_share']:7.4f}  "
              f"{(row.get('epochs') or 0):5.2f} epochs{capped}")
    print(f"\n  holdout: {len(composed['holdout'])} source(s) under "
          f"{holdout_root}")
    for bucket in sorted(record["buckets"]):
        unscored = sorted(name for name in record["buckets"][bucket]
                          if name not in composed["holdout"])
        if unscored:
            print(f"      {bucket}: no holdout for {', '.join(unscored)}")

    record.update({
        "total_tokens": a.total_tokens,
        "out_root": a.out_root,
        "train_root": train_root,
        "holdout_root": holdout_root,
        "composed": composed,
        "preflight": preflight,
        "evidence": {"code_manifest": a.manifest,
                     "replay_roots": replay_roots,
                     "general_holdout_root": a.general_holdout_root},
        "train_flags": ["--data-dir", train_root, "--val-dir", holdout_root,
                        *mixture_weight_flags(record["weights"])],
        "caveats": [
            # Recorded here because this is the artifact the probe scoring reads
            # its source list from, and the gate it feeds says "general BPB
            # regression <=1.5%".
            "data/holdout/stack-edu-python is GitHub Python from the same "
            "dataset and revision the code bucket streams, and the code "
            "corpus is repository-split against its own holdout only -- so "
            "that source cannot serve as a general-retention measurement for "
            "a model being trained on code. Score general BPB over the "
            "general-text sources separately from it.",
        ],
    })
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        print(f"\nwrote {a.json_out}")

    print("\ntrain on it with:\n  " + " ".join(record["train_flags"]))
    if preflight["l1_skew_pts"] > a.max_l1_skew:
        print(f"\nthe realised mixture is {preflight['l1_skew_pts']:.2f} points "
              f"from the one asked for, past the {a.max_l1_skew:g}-point limit: "
              f"{', '.join(preflight['capped_sources'])} cannot fill "
              f"{a.total_tokens:,} tokens inside the epoch cap",
              file=sys.stderr)
        return 3
    return 0


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    decontam = sub.add_parser(
        "decontam", help="the frozen HumanEval+/MBPP+ n-gram index")
    action = decontam.add_subparsers(dest="action", required=True)

    # `--out` on each action rather than on `decontam`, so it can be written
    # after the verb (`decontam build --out ...`) the way every other script
    # here takes its flags. On the parent it would have to precede the verb,
    # which reads as a typo and is one.
    b = action.add_parser("build")
    b.add_argument("--out", default=DEFAULT_CODE_INDEX_PATH)
    b.add_argument("--n", type=int, default=DEFAULT_CODE_N,
                   help="n-gram length. The corpus filter looks up "
                        f"{DEFAULT_CODE_N}-grams; another value builds an index "
                        f"that unions cleanly and matches nothing, and `verify` "
                        f"refuses it.")
    b.set_defaults(fn=_build)

    v = action.add_parser("verify")
    v.add_argument("--out", default=DEFAULT_CODE_INDEX_PATH)
    v.add_argument("--expect-digest", default=None)
    v.add_argument("--json-out", default=None)
    v.set_defaults(fn=_verify)

    corpus = sub.add_parser(
        "corpus", help="the licensed, repository-split code corpus")
    corpus_action = corpus.add_subparsers(dest="action", required=True)

    probe = corpus_action.add_parser(
        "probe", help="read real rows and report what the gate found in them")
    probe.add_argument(
        "--language", action="append", choices=sorted(GITHUB_CODE_LANGUAGES),
        help=f"code bucket to probe, repeatable (default: all "
             f"{len(CODE_LANGUAGE_SHARES)})")
    probe.add_argument(
        "--config", action="append",
        help="probe this parquet directory by name instead of the mixture's "
             "buckets, repeatable -- how a directory with no share yet (the "
             "interleaved 'all-all', a candidate substitute) gets measured")
    probe.add_argument(
        "--keep-language", action="append",
        help="keep only rows of this language, repeatable -- how a bucket with "
             "no directory of its own is measured against the interleaved "
             "'all-all'. Reports rows streamed per row kept, which is what "
             "decides whether the bucket's share is affordable. Needs --config")
    probe.add_argument("--rows", type=int, default=2_000,
                       help="rows to read per config (default 2,000)")
    probe.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    probe.add_argument("--json-out", default=None)
    probe.set_defaults(fn=_probe)

    plan = corpus_action.add_parser(
        "plan", help="the mixture this revision can actually serve, from the "
                     "directory listing and the interleaved probe")
    plan.add_argument("--configs-json", default="runs/codeprep/github-code-configs.json",
                      help="`corpus configs --json-out` output: which "
                           "directories the revision really carries")
    plan.add_argument("--probe-json", default="runs/codeprep/allall-yield.json",
                      help=f"`corpus probe --config {INTERLEAVED_CONFIG} "
                           f"--json-out` output: what the interleaved directory "
                           f"admits per language")
    plan.add_argument("--interleaved", default=INTERLEAVED_CONFIG,
                      help="directory the bucketless buckets fall back to")
    plan.add_argument(
        "--passes", type=float, default=DEFAULT_INTERLEAVE_PASSES,
        help=f"interleaved bytes the build may admit per byte of code corpus "
             f"(default {DEFAULT_INTERLEAVE_PASSES:g}). Every bucket's "
             f"`required_passes` is recorded, so another budget re-derives the "
             f"mixture without re-reading a row")
    plan.add_argument("--min-share", type=float, default=MIN_BUCKET_SHARE,
                      help=f"below this share of the code mixture a bucket is "
                           f"dropped by name rather than carried "
                           f"(default {MIN_BUCKET_SHARE:.3f})")
    plan.add_argument("--json-out", default=None)
    plan.set_defaults(fn=_plan)

    headroom = corpus_action.add_parser(
        "headroom", help="whether each planned bucket's directories hold the "
                         "share the plan gives it, inside the epoch cap")
    headroom.add_argument("--plan-json", default="runs/codeprep/source-plan.json",
                          help="`corpus plan --json-out` output")
    headroom.add_argument(
        "--probe-json", action="append",
        default=None,
        help="probe reports supplying each directory's admitted bytes per row, "
             "repeatable (default: the per-directory and interleaved yields)")
    headroom.add_argument("--rows-json", default="runs/codeprep/config-rows.json",
                          help="cached parquet footer row counts; measured and "
                               "written when absent")
    headroom.add_argument("--remeasure", action="store_true",
                          help="re-read the footers even if the cache exists")
    headroom.add_argument("--revision", default=GITHUB_CODE_REVISION)
    headroom.add_argument("--bytes-per-token", type=float,
                          default=CODE_BYTES_PER_TOKEN,
                          help=f"measured code fertility of the tokenizer the "
                               f"budgets are counted in "
                               f"(default {CODE_BYTES_PER_TOKEN:g}, phase 4's "
                               f"49152-smollm2 reading)")
    headroom.add_argument("--budget-tokens", type=int, action="append",
                          default=None,
                          help="total training budget in tokens, repeatable "
                               "(default 250M, 1B, 3B -- phase 8's three gates)")
    headroom.add_argument("--epoch-cap", type=float, default=4.0,
                          help="the plan's ceiling: no source read more than "
                               "this many times")
    headroom.add_argument("--json-out", default=None)
    headroom.set_defaults(fn=_headroom)

    build = corpus_action.add_parser(
        "build", help="stream the planned directories into licensed, "
                      "repository-split, decontaminated shards")
    build.add_argument("--plan-json", default="runs/codeprep/source-plan.json",
                       help="`corpus plan --json-out` output: the mixture this "
                            "revision can actually serve")
    build.add_argument("--headroom-json", default="runs/codeprep/headroom.json",
                       help="`corpus headroom --json-out` output, for the "
                            "per-directory supply a multi-directory bucket's "
                            "budget is divided on")
    build.add_argument("--total-tokens", type=int, default=1_000_000_000,
                       help="total training budget in tokens; the code corpus "
                            f"built is {CORPUS_SHARES['code']:.0%} of it")
    build.add_argument("--out-root", default="data/code-shards",
                       help="shards go to <root>/train/<key> and "
                            "<root>/holdout/<key>, so each side is an ordinary "
                            "mixture root")
    build.add_argument("--manifest", default="data/code-shards/manifest.json")
    build.add_argument("--code-index", default=DEFAULT_CODE_INDEX_PATH)
    build.add_argument("--code-index-digest", default=None,
                       help="pin the code index this corpus is filtered "
                            "against, as `decontam build` prints it")
    build.add_argument("--eval-index", default=DEFAULT_EVAL_INDEX_PATH,
                       help="the general corpus's frozen index, unioned with "
                            "the code one; empty string to filter against the "
                            "code benchmarks alone")
    build.add_argument("--eval-index-digest", default=None)
    build.add_argument("--tokenizer", default=None,
                       help="tokenizer the ids are packed under (default: the "
                            "released SmolLM2 vocabulary Daedalus-Code inherits)")
    build.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC,
                       help="fraction of *repositories* held out")
    build.add_argument("--holdout-cap-tokens", type=int,
                       default=DEFAULT_HOLDOUT_CAP_TOKENS,
                       help="ceiling on one bucket's holdout tokens")
    build.add_argument("--split-salt", default=SPLIT_SALT,
                       help="mixed into the repository hash; changing it is a "
                            "different split and cannot be resumed onto an "
                            "existing tree")
    build.add_argument("--shard-tokens", type=int, default=100_000_000)
    build.add_argument("--max-docs-per-source", type=int, default=None,
                       help="stop each source after this many documents -- a "
                            "smoke, and reported as one")
    build.add_argument("--rss-limit-gb", type=float, default=8.0)
    build.add_argument("--rss-check-every", type=int, default=5_000)
    build.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="documents between durable checkpoints; 0 derives it from each "
             "source's own budget, because a cadence in yielded documents "
             "never fires on a holdout pass and a fixed small one leaves "
             "hundreds of shards behind on a large source")
    build.add_argument("--progress-every", type=int, default=0,
                       help="documents between progress lines; 0 derives it "
                            "the same way, so a multi-hour source is legible "
                            "instead of silent")
    build.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="rebuild every source from row zero. The default continues each "
             "source from the shards and stream position already on disk, "
             "which is what makes a relaunch after a crash cost the remainder "
             "rather than the whole corpus")
    build.add_argument("--dry-run", action="store_true",
                       help="print the budgets and exit without streaming")
    build.set_defaults(fn=_build_corpus, resume=True)

    mixture = corpus_action.add_parser(
        "mixture", help="link the built code corpus and the original "
                        "pretraining data into the one root a run reads, and "
                        "print the shares that make it 65/15/20")
    mixture.add_argument("--manifest", default="data/code-shards/manifest.json",
                         help="`corpus build --manifest` output; an unfinished "
                              "build is refused rather than composed")
    mixture.add_argument("--code-root", default="data/code-shards",
                         help="where `corpus build` wrote <root>/train/<key>")
    mixture.add_argument(
        "--general-train-root", default="data/shards-train",
        help="the holdout-disjoint carve of the original corpus, searched "
             "first: data/shards still holds the windows data/holdout scores")
    mixture.add_argument("--general-root", default="data/shards",
                         help="the original corpus, for the sources that were "
                              "never carved because they have one shard")
    mixture.add_argument("--general-holdout-root", default="data/holdout",
                         help="the general BPB holdout every phase so far has "
                              "been scored on")
    mixture.add_argument("--out-root", default="data/code-mixture",
                         help="the composed root: <root>/train and "
                              "<root>/holdout, each a directory of symlinks")
    mixture.add_argument("--total-tokens", type=int, default=250_000_000,
                         help="the budget the epochs and the cap are reported "
                              "at -- one probe, not the whole phase")
    mixture.add_argument("--max-l1-skew", type=float, default=5.0,
                         help="how far the capped mixture may sit from the one "
                              "asked for, in percentage points, before this "
                              "fails")
    mixture.add_argument("--json-out", default="runs/codeprep/train-mixture.json")
    mixture.add_argument("--dry-run", action="store_true",
                         help="print the shares without linking anything")
    mixture.set_defaults(fn=_mixture)

    configs = corpus_action.add_parser(
        "configs", help="list the parquet directories the revision really has")
    configs.add_argument(
        "--dataset", default=GITHUB_CODE_DATASET,
        help="repository to list; another one is how a substitute source for a "
             "language this dataset never converted gets chosen")
    configs.add_argument("--revision", default=GITHUB_CODE_REVISION)
    configs.add_argument("--json-out", default=None)
    configs.set_defaults(fn=_configs)

    a = p.parse_args(argv)
    if getattr(a, "fn", None) is _headroom:
        # Repeatable flags cannot carry a list default -- argparse appends to it
        # rather than replacing it, so one `--probe-json` would silently mean
        # three.
        if a.probe_json is None:
            a.probe_json = ["runs/codeprep/directory-yield.json",
                            "runs/codeprep/allall-yield.json"]
        if a.budget_tokens is None:
            a.budget_tokens = list(DEFAULT_CODE_BUDGETS)
    if getattr(a, "action", None) == "verify" and not os.path.exists(
            sidecar_path(a.out)):
        print(f"REFUSE: no index at {a.out} (build it first)", file=sys.stderr)
        return 2
    return a.fn(a)


def _exit(code: int):
    """Leave with `code` as the verdict, without running interpreter teardown.

    Neither live probe so far exited with the verdict it had computed. Both
    printed their report, wrote their JSON, and then:

        Fatal Python error: PyGILState_Release: thread state ... must be
        current when releasing
        Python runtime state: finalizing

    -- an abort, exit -6. A `datasets` streaming iterator leaves a background
    thread behind and `probe_source` breaks out of the loop rather than
    exhausting it, so the interpreter finalizes underneath a thread that is
    still holding GIL state. The measurement is complete before any of that
    happens; the only thing lost is the exit code.

    The exit code is not a detail here -- it is what the controller ledger
    records. The source probe's "four of ten directories did not resolve" (3)
    and the all-all coverage probe's "nothing wrong with these rows" (0) were
    both filed as the same -6, so `state.json` says a measurement that found no
    problem failed, and says nothing about the one that found four.

    Flush first, because `os._exit` does not: the report on stdout and the
    refusal on stderr *are* the output. Everything written to disk is closed by
    its own `with` block well before this, so skipping `atexit` costs a
    `datasets` cache cleanup -- a temp file under `HF_HOME` -- and no result.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _exit(_cli())
