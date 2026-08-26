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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from daedalus.codeprep import (CODE_LANGUAGE_SHARES,  # noqa: E402
                               DEFAULT_CODE_INDEX_PATH, DEFAULT_CODE_N,
                               DEFAULT_HOLDOUT_FRAC, DEFAULT_INTERLEAVE_PASSES,
                               GITHUB_CODE_LANGUAGES, INTERLEAVED_CONFIG,
                               IncompleteIndex, IndexDigestMismatch,
                               GITHUB_CODE_DATASET, GITHUB_CODE_REVISION,
                               MIN_BUCKET_SHARE, build_code_index,
                               code_coverage_problems, config_near_misses,
                               github_code_configs, load_index, missing_configs,
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
