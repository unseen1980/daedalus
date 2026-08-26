"""Build and check the frozen decontamination index.

    python scripts/decontam_index.py build
    python scripts/decontam_index.py verify

`build` loads every scored task at every scored split with no per-task limit,
writes the sorted n-gram set and its provenance sidecar, and prints the digest
a corpus manifest should pin. `verify` re-reads a written index, recomputes the
digest, and checks its coverage against today's `eval.TASK_LOADERS` --
answering "does the index this corpus was filtered against still cover what the
model is scored on", which is a different question from the one `build`
answered and the one that goes stale.

See `daedalus/eval_index.py` for why the index is frozen rather than rebuilt at
the start of every corpus build, and `scripts/contam_scan.py` for the
measurement of what the *unfrozen*, limited index let through.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from daedalus.eval_index import (DEFAULT_INDEX_PATH, DEFAULT_N,  # noqa: E402
                                 IncompleteIndex, IndexDigestMismatch,
                                 build_index, coverage_problems, load_index,
                                 read_provenance, sidecar_path, write_index)


def _report(provenance: dict) -> str:
    lines = [f"  {provenance['ngrams']:,} {provenance['n']}-grams  "
             f"{provenance['digest']}"]
    for name, meta in sorted((provenance.get("tasks") or {}).items()):
        lines.append(f"  {name:14s} {meta['items']:>7,} items  "
                     f"{meta['candidates']:>7,} candidates  "
                     f"split={meta['split']}  repo={meta.get('repo')}")
    return "\n".join(lines)


def _build(a) -> int:
    print(f"loading every scored task at every scored split "
          f"(limit={a.limit!r}) ...", flush=True)
    try:
        ngrams, provenance = build_index(n=a.n, limit=a.limit,
                                         allow_partial=a.allow_partial)
    except IncompleteIndex as e:
        print("REFUSE: the index would not cover what this model is scored on:",
              file=sys.stderr)
        for problem in e.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    write_index(a.out, ngrams, provenance, allow_partial=a.allow_partial)
    print(f"wrote {a.out} and {sidecar_path(a.out)}")
    print(_report(provenance))
    print(f"\npin it with:\n  --eval-index {a.out} "
          f"--eval-index-digest {provenance['digest']}")
    return 0


def _verify(a) -> int:
    try:
        ngrams, provenance = load_index(a.out, expect_digest=a.expect_digest)
    except (OSError, IndexDigestMismatch) as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    print(f"{a.out}: {len(ngrams):,} {provenance['n']}-grams")
    print(_report(provenance))
    problems = coverage_problems(provenance)
    if problems:
        print("\nthis index no longer covers what the model is scored on:",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 3
    print("\ncoverage: complete against today's scored tasks and splits")
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump({"path": a.out, "provenance": provenance,
                       "problems": problems}, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {a.json_out}")
    return 0


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_INDEX_PATH)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=DEFAULT_N)
    b.add_argument("--limit", type=int, default=None,
                   help="per-task item cap. Only for reconstructing a "
                        "historical index; needs --allow-partial and the "
                        "result is marked incomplete.")
    b.add_argument("--allow-partial", action="store_true")
    b.set_defaults(fn=_build)

    v = sub.add_parser("verify")
    v.add_argument("--expect-digest", default=None)
    v.add_argument("--json-out", default=None)
    v.set_defaults(fn=_verify)

    a = p.parse_args(argv)
    if a.command == "verify" and not os.path.exists(sidecar_path(a.out)):
        print(f"REFUSE: no index at {a.out} (build it first)", file=sys.stderr)
        return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(_cli())
