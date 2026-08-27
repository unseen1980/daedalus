"""Phase 8 step 6's SFT sources, resolved against real rows.

The table in `daedalus/code_sft.py` names what to *ask* for. This resolves it:
which datasets answer, which licence each declares, which row key carries the
sub-dataset a row came from, how much of each stream survives the admission
gate, and how many admitted conversations ship a test that could be executed.

Every one of those is a guess until a row is read, and each fails in the same
silent direction -- a `source_field` the rows do not have, or a licence string
nobody classified, refuses every row and leaves a build that writes an empty
file and exits zero. `codeprep`'s corpus probe exists for the same reason and
this is deliberately the same shape.

Read the JSON rather than the exit code when the process ends in `-6`: pyarrow
aborts in `PyGILState_Release` during interpreter finalization, after everything
is written. That trap is recorded in the phase 8 PR body and it applies here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus.code_sft import (DEFAULT_MAX_LEN, SFT_SOURCES,  # noqa: E402
                               default_execute, probe_problems, probe_sources)
from daedalus.codeprep import (DEFAULT_CODE_INDEX_PATH,  # noqa: E402
                               code_coverage_problems)
from daedalus.codeprep import load_index as load_code_index

#: The five-task index, as `scripts/codeprep.py` defaults to it. A second
#: literal rather than an import because that module pulls `dataprep` and
#: `datasets` in behind it, which is a slow price for a path string on every
#: `--help`. A test pins the two together so the copy cannot drift.
DEFAULT_EVAL_INDEX_PATH = "data/decontam/eval-index-13gram.txt.gz"


def _load_indexes(a) -> dict:
    """`{name: ngrams}`, so a contamination hit reports *which* benchmark set it
    came from. Not a union: `codeprep`'s build unions them because its predicate
    takes one set, while `code_sft.contamination_hit` returns the index name and
    the two indexes cover different benchmarks -- a code-index hit and a
    five-task hit are different findings about a conversation."""

    if a.no_decontam:
        return {}
    indexes = {}
    ngrams, provenance = load_code_index(a.code_index,
                                         expect_digest=a.code_index_digest)
    problems = code_coverage_problems(provenance)
    if problems:
        raise ValueError(f"{a.code_index} does not cover what phase 8 is gated "
                         f"on: {'; '.join(problems)}")
    indexes["code"] = ngrams
    if a.eval_index:
        from daedalus.eval_index import coverage_problems, load_index

        eval_ngrams, eval_provenance = load_index(
            a.eval_index, expect_digest=a.eval_index_digest)
        problems = coverage_problems(eval_provenance)
        if problems:
            raise ValueError(f"{a.eval_index} does not cover what this model is "
                             f"scored on: {'; '.join(problems)}")
        indexes["eval"] = eval_ngrams
    return indexes


def _record_lines(record: dict) -> list:
    key = record["key"]
    if record.get("error"):
        return [f"  {key:26s} UNREACHABLE {record['error'][:120]}"]
    if not record.get("resolved"):
        return [f"  {key:26s} UNRESOLVED  yielded no rows"]
    report = record["report"]
    lines = [
        f"  {key:26s} half {record['half']:8s} "
        f"licence {record['declared_license']!r} -> {record['license_verdict']}",
        f"      {record['rows_offered']:>7,} offered  "
        f"{record['rows_kept']:>7,} kept ({record.get('kept_share') or 0:.1%})  "
        f"{report['admitted']:>7,} admitted "
        f"({report['admitted'] / max(record['rows_kept'], 1):.1%} of kept)",
    ]
    refused = ", ".join(f"{name}={count:,}" for name, count
                        in sorted(report["refusals"].items()) if count)
    lines.append(f"      refused: {refused or 'nothing'}")
    rows = ", ".join(f"{name}={count:,}" for name, count
                     in sorted(record["row_refusals"].items()) if count)
    lines.append(f"      not this source: {rows or 'nothing'}")
    shares = ", ".join(f"{bucket}={share:.1%}" for bucket, share
                       in sorted(report["code_language_shares"].items()))
    lines.append(f"      code language shares: {shares or 'no carried code'}")
    checked = report["syntax_checked_share"]
    lines.append(
        f"      syntax-checked {('%.1f%%' % (100 * checked)) if checked is not None else 'n/a'} "
        f"of {report['checked_code_bytes'] + report['unchecked_code_bytes']:,} "
        f"assistant code bytes; {report['unknown_language_bytes']:,} untagged")
    share = record.get("shipped_test_share")
    lines.append(
        f"      ships a runnable test: {record['shipped_tests']:,} of "
        f"{record['rows_kept']:,} kept, "
        f"{record['shipped_tests_admitted']:,} of {report['admitted']:,} "
        f"admitted{'' if share is None else f' ({share:.1%})'}; "
        f"executed: {'yes' if record['executed'] else 'no'}")
    if not record["executed"]:
        why = ", ".join(f"{name}={count:,}" for name, count
                        in sorted(record["no_test_reasons"].items()) if count)
        lines.append(f"      no test because: {why or 'nothing'}")
    supervised = report["supervised_tokens"]
    if supervised is None:
        lines.append("      supervised tokens: not counted (no tokenizer)")
    else:
        counted = report["supervised_tokens_counted_for"]
        lines.append(f"      supervised tokens: {supervised:,} over "
                     f"{counted:,} examples "
                     f"({supervised / max(counted, 1):,.0f} per example)")
    values = list((record.get("source_values") or {}).items())
    if values:
        top = ", ".join(f"{name}={count:,}" for name, count in values[:6])
        lines.append(f"      {record['source_field']} values ({len(values)}): {top}")
    return lines


def _probe(a) -> int:
    chosen = [s for s in SFT_SOURCES if not a.source or s.key in a.source]
    unknown = sorted(set(a.source or ()) - {s.key for s in SFT_SOURCES})
    if unknown:
        print(f"REFUSE: unknown source key(s) {unknown}; the table carries "
              f"{sorted(s.key for s in SFT_SOURCES)}", file=sys.stderr)
        return 2

    try:
        indexes = _load_indexes(a)
    except (OSError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2

    tokenizer = None
    if not a.no_tokenizer:
        from daedalus.data import get_tokenizer

        tokenizer = get_tokenizer()

    print(f"probing {len(chosen)} source(s) at {a.rows:,} rows each; "
          f"decontamination {'off' if not indexes else ', '.join(sorted(indexes))}; "
          f"tokenizer {'off' if tokenizer is None else 'on'}; "
          f"execution {'on' if a.execute else 'off'}", flush=True)

    report = probe_sources(chosen, rows=a.rows, indexes=indexes,
                           tokenizer=tokenizer, max_len=a.max_len,
                           execute=default_execute if a.execute else None,
                           timeout_s=a.timeout_s, memory_mb=a.memory_mb)
    problems = probe_problems(report)
    report["problems"] = problems

    for record in report["sources"]:
        print("\n".join(_record_lines(record)))
    print("\n  alternatives measured and not used:")
    for entry in report["alternatives"]:
        print(f"      {entry['dataset']:48s} {entry['declared_license']!r} -> "
              f"{entry['license_verdict']}")

    if a.json_out:
        # Written before the verdict, so a failing probe still leaves the record
        # of what was in the rows -- which is the entire output anyone needs to
        # fix the assumption that failed.
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {a.json_out}")

    if problems:
        print("\nan SFT build would not do what it says on these sources:",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 3
    print("\nevery source resolved, declares a permissive licence, and admits "
          "conversations through the gate")
    return 0


def _cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser(
        "probe", help="resolve the SFT source table against real rows")
    probe.add_argument("--rows", type=int, default=2_000,
                       help="rows *offered* per source, before the source "
                            "filter. The code half keeps about a tenth of "
                            "them, so a small number measures it poorly")
    probe.add_argument("--source", action="append", default=[],
                       help="probe only this table key; repeatable")
    probe.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                       help="post.py's token budget, so `over_token_budget` is "
                            "counted at the budget the trainer will use")
    probe.add_argument("--code-index", default=DEFAULT_CODE_INDEX_PATH)
    probe.add_argument("--code-index-digest", default=None)
    probe.add_argument("--eval-index", default=DEFAULT_EVAL_INDEX_PATH,
                       help="the five-task index; pass an empty string to "
                            "filter against the code benchmarks alone")
    probe.add_argument("--eval-index-digest", default=None)
    probe.add_argument("--no-decontam", action="store_true",
                       help="skip both indexes. Measures a yield no build will "
                            "see; for checking a source's shape only")
    probe.add_argument("--no-tokenizer", action="store_true",
                       help="skip the token budget check and the supervised "
                            "token counts")
    probe.add_argument("--execute", action="store_true",
                       help="run each shipped test in the code_eval sandbox. "
                            "Costs a process per test; off by default, and the "
                            "record says which of the two shares was measured")
    probe.add_argument("--timeout-s", type=float, default=30.0)
    probe.add_argument("--memory-mb", type=int, default=1024)
    probe.add_argument("--json-out", default=None)
    probe.set_defaults(fn=_probe)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(_cli())
