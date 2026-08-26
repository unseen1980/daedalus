"""Build and check the code corpus's frozen decontamination index.

    python scripts/codeprep.py decontam build
    python scripts/codeprep.py decontam verify --json-out runs/codeprep/decontam.json

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

from daedalus.codeprep import (DEFAULT_CODE_INDEX_PATH,  # noqa: E402
                               DEFAULT_CODE_N, IncompleteIndex,
                               IndexDigestMismatch, build_code_index,
                               code_coverage_problems, load_index,
                               sidecar_path, write_index)


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

    a = p.parse_args(argv)
    if getattr(a, "action", None) == "verify" and not os.path.exists(
            sidecar_path(a.out)):
        print(f"REFUSE: no index at {a.out} (build it first)", file=sys.stderr)
        return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(_cli())
