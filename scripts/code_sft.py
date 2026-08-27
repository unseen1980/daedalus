"""Phase 8 step 6's SFT sources: resolved against real rows, then built.

The table in `daedalus/code_sft.py` names what to *ask* for. `probe` resolves
it: which datasets answer, which licence each declares, which row key carries
the sub-dataset a row came from, how much of each stream survives the admission
gate, and how many admitted conversations ship a test that could be executed.

Every one of those is a guess until a row is read, and each fails in the same
silent direction -- a `source_field` the rows do not have, or a licence string
nobody classified, refuses every row and leaves a build that writes an empty
file and exits zero. `codeprep`'s corpus probe exists for the same reason and
this is deliberately the same shape.

`build` is the pass the probe authorises: the same rows through the same gate,
with the conversations that pass written once to a training file, a held-out
file and a manifest carrying every count the plan asks step 6 to track. It runs
the shipped tests it finds rather than only reporting that they exist, which is
the difference between "syntax-checked" and the plan's "syntax-checked *and*
execution-tested" -- and the reason it belongs in a detached phase rather than
in a training loop.

The `-6` trap recorded in the phase 8 PR body is closed here rather than worked
around: pyarrow aborts in `PyGILState_Release` during interpreter finalization,
after everything is written, so the process now exits on its own return code
without finalizing. See the `__main__` block.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus.code_sft import (DEFAULT_CODE_SHARE,  # noqa: E402
                               DEFAULT_HOLDOUT_EXAMPLES, DEFAULT_MAX_LEN,
                               DEFAULT_SUPERVISED_TOKENS, SFT_SOURCES,
                               CodeSFTError, build_dataset, default_execute,
                               probe_problems, probe_sources)
from daedalus.codeprep import (DEFAULT_CODE_INDEX_PATH,  # noqa: E402
                               code_coverage_problems)
from daedalus.codeprep import load_index as load_code_index

#: The five-task index, as `scripts/codeprep.py` defaults to it. A second
#: literal rather than an import because that module pulls `dataprep` and
#: `datasets` in behind it, which is a slow price for a path string on every
#: `--help`. A test pins the two together so the copy cannot drift.
DEFAULT_EVAL_INDEX_PATH = "data/decontam/eval-index-13gram.txt.gz"


def _index_provenance(path: str, provenance) -> dict:
    """What a manifest needs to re-derive which index filtered a build.

    The n-grams themselves say nothing about where they came from, and a built
    corpus outlives the command that wrote it -- phase 7 paid for that once,
    when establishing which sources predated a split change meant rebuilding an
    index and matching a gram count that happened to be in a log.
    """

    fields = provenance if isinstance(provenance, dict) else {}
    return {"path": path, "digest": fields.get("digest"), "n": fields.get("n"),
            "ngrams": fields.get("ngrams"), "built_at": fields.get("built_at")}


def _load_indexes(a):
    """`({name: ngrams}, {name: provenance})`, so a contamination hit reports
    *which* benchmark set it came from. Not a union: `codeprep`'s build unions
    them because its predicate takes one set, while `code_sft.contamination_hit`
    returns the index name and the two indexes cover different benchmarks -- a
    code-index hit and a five-task hit are different findings about a
    conversation."""

    if a.no_decontam:
        return {}, {}
    indexes, provenance = {}, {}
    ngrams, record = load_code_index(a.code_index,
                                     expect_digest=a.code_index_digest)
    problems = code_coverage_problems(record)
    if problems:
        raise ValueError(f"{a.code_index} does not cover what phase 8 is gated "
                         f"on: {'; '.join(problems)}")
    indexes["code"] = ngrams
    provenance["code"] = _index_provenance(a.code_index, record)
    if a.eval_index:
        from daedalus.eval_index import coverage_problems, load_index

        eval_ngrams, eval_record = load_index(
            a.eval_index, expect_digest=a.eval_index_digest)
        problems = coverage_problems(eval_record)
        if problems:
            raise ValueError(f"{a.eval_index} does not cover what this model is "
                             f"scored on: {'; '.join(problems)}")
        indexes["eval"] = eval_ngrams
        provenance["eval"] = _index_provenance(a.eval_index, eval_record)
    return indexes, provenance


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
    chosen, refusal = _chosen_sources(a)
    if refusal:
        print(f"REFUSE: {refusal}", file=sys.stderr)
        return 2

    try:
        indexes, _ = _load_indexes(a)
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


def _chosen_sources(a):
    """`(sources, refusal)` for the `--source` filter both subcommands take."""

    unknown = sorted(set(a.source or ()) - {s.key for s in SFT_SOURCES})
    if unknown:
        return None, (f"unknown source key(s) {unknown}; the table carries "
                      f"{sorted(s.key for s in SFT_SOURCES)}")
    return [s for s in SFT_SOURCES if not a.source or s.key in a.source], None


def _build(a) -> int:
    chosen, refusal = _chosen_sources(a)
    if refusal:
        print(f"REFUSE: {refusal}", file=sys.stderr)
        return 2

    try:
        indexes, provenance = _load_indexes(a)
    except (OSError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2

    from daedalus.data import get_tokenizer

    tokenizer = get_tokenizer()
    print(f"building {len(chosen)} source(s) into {a.out_dir}: "
          f"{a.supervised_tokens:,} supervised tokens at {a.code_share:.0%} "
          f"code, {a.holdout_examples:,} held out per half; "
          f"decontamination {', '.join(sorted(indexes))}; "
          f"execution {'off' if a.no_execute else 'on'}", flush=True)

    try:
        manifest = build_dataset(
            chosen, out_dir=a.out_dir, tokenizer=tokenizer, indexes=indexes,
            index_provenance=provenance,
            supervised_tokens=a.supervised_tokens, code_share=a.code_share,
            holdout_examples=a.holdout_examples,
            max_offered_rows=a.max_offered_rows, max_len=a.max_len,
            execute=None if a.no_execute else default_execute,
            timeout_s=a.timeout_s, memory_mb=a.memory_mb, seed=a.seed,
            overwrite=a.overwrite, progress_every=a.progress_every,
            log=lambda line: print(line, flush=True))
    except CodeSFTError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2

    for record in manifest["sources"]:
        print("\n".join(_record_lines(record)))
        print(f"      wrote {record['written_train']:,} training and "
              f"{record['written_holdout']:,} held-out conversations; "
              f"{record['rows_after_budget']:,} rows arrived after this half "
              f"was full")

    print(f"\n  {manifest['rows_offered']:,} rows offered, "
          f"{manifest['overlapping_rows']:,} claimed by more than one source")
    for half, block in sorted(manifest["halves"].items()):
        print(f"  {half:8s} {block['supervised_tokens']:>10,} of "
              f"{block['budget']:,} supervised tokens "
              f"({block['train_examples']:,} conversations, "
              f"{block['holdout_examples']:,} held out)"
              f"{'  TRIMMED to hold the share' if block['trimmed'] else ''}")
    share = manifest["realized_code_share"]
    print(f"  realized code share "
          f"{'n/a' if share is None else f'{share:.1%}'} of "
          f"{manifest['train_supervised_tokens']:,} supervised tokens over "
          f"{manifest['train_examples']:,} conversations")
    print(f"\nwrote {os.path.join(a.out_dir, 'manifest.json')}")

    if manifest["problems"]:
        print("\nthe built set is not what its manifest asks for:",
              file=sys.stderr)
        for problem in manifest["problems"]:
            print(f"  - {problem}", file=sys.stderr)
        return 3
    print("\nevery source resolved and both halves filled their share of the "
          "budget")
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

    build = sub.add_parser(
        "build", help="write the SFT set the sources admit, with its manifest")
    build.add_argument("--out-dir", default="data/code-sft")
    build.add_argument("--supervised-tokens", type=int,
                       default=DEFAULT_SUPERVISED_TOKENS,
                       help="training budget, in supervised tokens. Not a "
                            "preregistered quantity -- the plan names step 6's "
                            "filter, not its size -- so it is a flag and the "
                            "realized figure goes in the manifest")
    build.add_argument("--code-share", type=float, default=DEFAULT_CODE_SHARE,
                       help="the code half's share of supervised tokens. "
                            "Borrowed from the corpus's preregistered 65%%; "
                            "this is the invariant the budget gives way to")
    build.add_argument("--holdout-examples", type=int,
                       default=DEFAULT_HOLDOUT_EXAMPLES,
                       help="conversations held out per half, off the head of "
                            "the same stream so the trainer cannot reach them")
    build.add_argument("--max-offered-rows", type=int, default=None,
                       help="stop the pass after this many rows however full "
                            "the halves are; for smokes, which must not be "
                            "able to write a full set into the gate's path")
    build.add_argument("--source", action="append", default=[],
                       help="build only this table key; repeatable. A build "
                            "still needs a source for every weighted half")
    build.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                       help="post.py's token budget, so an example admitted "
                            "here is one the trainer's encoder will accept")
    build.add_argument("--code-index", default=DEFAULT_CODE_INDEX_PATH)
    build.add_argument("--code-index-digest", default=None)
    build.add_argument("--eval-index", default=DEFAULT_EVAL_INDEX_PATH)
    build.add_argument("--eval-index-digest", default=None)
    build.add_argument("--no-execute", action="store_true",
                       help="admit shipped tests without running them. The "
                            "plan asks for execution-tested conversations, so "
                            "this measures a set nothing should train on")
    build.add_argument("--timeout-s", type=float, default=30.0)
    build.add_argument("--memory-mb", type=int, default=1024)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--progress-every", type=int, default=20_000)
    build.add_argument("--overwrite", action="store_true",
                       help="replace a built set rather than refusing to write "
                            "a second one beside it")
    # A build's indexes and tokenizer are not optional: the first decides
    # whether the set carries the benchmarks this phase is gated on, and the
    # second is what the mixture is measured in. The probe's flags for skipping
    # them exist to check a source's *shape*, and nothing should train on the
    # result, so they are not offered here.
    build.set_defaults(fn=_build, no_decontam=False)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    # `os._exit`, not `SystemExit`, for post.py's reason and this module's own
    # recorded one: pyarrow's parquet reader keeps a worker thread alive behind
    # a half-consumed streaming dataset, and finalizing the interpreter beside
    # it aborts in `PyGILState_Release` with exit code -6 -- *after* the JSON
    # and the shards are on disk. Everything durable is written by the time
    # `_cli` returns, so the only thing that abort can still lose is the exit
    # status, and the controller reads a phase's exit status to decide whether
    # the work happened. Not inside `_cli`: the tests call that directly, and
    # `os._exit` there would take pytest with it.
    _code = _cli()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
