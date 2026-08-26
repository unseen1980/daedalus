"""The code corpus's frozen decontamination index.

Why a second index
------------------
`daedalus/eval_index.py` freezes the n-grams of the five multiple-choice tasks
the general model is scored on. It contains nothing from HumanEval+ or MBPP+,
because nothing was ever scored on them when it was built. Phase 8 is scored on
exactly those two, and its corpus is 65% GitHub code -- so filtering the code
corpus against the general index alone would decontaminate it against the
benchmarks it is *not* judged by and leave the ones it is judged by untouched.

The exposure is not incidental overlap either, which is what makes this worth a
separate artifact rather than a footnote. HumanEval and MBPP are public
repositories; their prompts, their reference solutions and their assertions are
copied verbatim into tutorials, leaderboard harnesses, "LLM benchmark" scratch
repos and solution sets, all of which are ordinary permissively-licensed Python
and all of which a code corpus will happily ingest. A phase 8 gate reads
"HumanEval+/MBPP+ pass@1 improves over untouched base"; an unfiltered corpus can
deliver that by memorisation and the gate cannot tell the difference.

Same `n` as the general index, deliberately
-------------------------------------------
Both indexes are consumed by the same predicate. `dataprep.DedupState.keep`
calls `daedalus.data.is_contaminated(text, index)` at its default `n=13`, over a
single set, so a code corpus build filters against `general | code`. An index
built at a different `n` would load without complaint, union without complaint,
and match nothing at all -- every one of its n-grams the wrong length to be
looked up. `DEFAULT_CODE_N` is therefore pinned to the general index's `n` and
`code_coverage_problems` refuses a mismatch rather than trusting the caller to
notice.

What is indexed, and what that does not cover
---------------------------------------------
Per item: the `prompt`, the `canonical_solution`, the two concatenated (the
`reference` -- what EvalPlus itself executes, and what a solutions repo
contains), and HumanEval+'s `test`, which carries the assertions with their
literal expected values and is the single most distinctive text in the dataset.
MBPP+ ships no `test`; its assertion travels inside its prompt, so it is covered
by the prompt's n-grams (see `test_the_real_mbpp_plus_schema_is_the_one_this_
harness_reads` in tests/test_code_eval.py).

A field shorter than `n` whitespace tokens yields no n-grams and is not
filtered by this index. That is common for `canonical_solution` -- a fair number
of MBPP+ references are one line -- and it is not a defect to fix by lowering
`n`: `return min(list1)` is three tokens that occur in every Python corpus ever
built, and an index that matched it would empty the corpus rather than clean it.
The honest position is that short solutions are unfilterable at 13-grams, so
`short_fields` in the provenance *counts* them per benchmark and per field. An
item whose `reference` is too short to index is a different matter -- that item
is unfilterable outright -- and is refused.

Refused, not warned
-------------------
Following `eval_index.py`: a benchmark missing, a benchmark that loaded zero
items, a benchmark whose item count is not the one it is scored at, or an item
that contributes no n-grams at all is an `IncompleteIndex`, not a log line. The
failure this closes is a *reassuring* index -- one that is written, digested,
recorded in a manifest, and filters less than the manifest claims. The on-disk
format, the atomic write, the digest and the load-time verification are
`eval_index`'s, reused unchanged, so the two indexes are the same kind of object
and a manifest carries them the same way.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

# Re-exported deliberately, not incidentally: the format, the atomic write, the
# digest and the load-time verification are shared, so a caller reads and writes
# a code index through this module without needing to know it is an
# `eval_index` file underneath. `_utcnow` comes across too so both sidecars
# stamp `built_at` in one format -- a manifest holds them side by side.
from daedalus.eval_index import (DEFAULT_N, IncompleteIndex,  # noqa: F401
                                 IndexDigestMismatch, _utcnow, index_digest,
                                 load_index, read_provenance, sidecar_path,
                                 write_index)

SCHEMA = 1

#: Pinned to the general index's `n`, not merely defaulted to it. See the module
#: docstring: the two sets are unioned and looked up by one predicate.
DEFAULT_CODE_N = DEFAULT_N

DEFAULT_CODE_INDEX_PATH = "data/decontam/code-index-13gram.txt.gz"

CODE_BENCHMARKS = ("humaneval-plus", "mbpp-plus")

#: Items in each benchmark, measured 2026-08-26 against EvalPlus 0.3.1 -- the
#: same counts the phase 8 baseline scorecards in `runs/eval/code-base` were
#: written at (`item_count` 164 and 378).
#:
#: Here for the reason `eval_index.EXPECTED_ITEMS` is: a dataset that downloads
#: short comes back short rather than failing, and an index built from it would
#: be smaller, be marked complete, get a digest, and filter the code corpus
#: while its own provenance asserted full coverage. A legitimate upstream change
#: fails this loudly and names both numbers, which is the intended reading --
#: what the size of a benchmark we gate on is is not something to discover
#: afterwards from a corpus.
CODE_EXPECTED_ITEMS = {"humaneval-plus": 164, "mbpp-plus": 378}

#: Emitted in this order so the provenance's per-field counts read the same way
#: every build. `reference` is the one an item is *required* to contribute.
INDEXED_FIELDS = ("prompt", "solution", "reference", "test")

REQUIRED_FIELD = "reference"


def item_texts(problem: dict) -> Dict[str, str]:
    """The indexable texts of one EvalPlus problem, keyed by field name.

    Absent and blank fields are omitted rather than yielded empty, so
    "this benchmark ships no `test`" is visible in the provenance as a missing
    key instead of as a zero that could equally mean "indexed, matched nothing".
    """
    texts: Dict[str, str] = {}
    prompt = problem.get("prompt") or ""
    solution = problem.get("canonical_solution") or ""
    if prompt.strip():
        texts["prompt"] = prompt
    if solution.strip():
        texts["solution"] = solution
    if prompt.strip() and solution.strip():
        # What EvalPlus executes as the reference, and what a solutions repo
        # holds: the signature and docstring followed by the body.
        texts["reference"] = prompt + solution
    test = problem.get("test") or ""
    if test.strip():
        texts["test"] = test
    return texts


def _default_code_loader(name: str):
    from scripts.code_eval import load_problems

    return load_problems(name)


def _evalplus_version() -> str:
    try:
        import evalplus
    except Exception:                       # noqa: BLE001 - reported, not raised
        return "absent"
    return getattr(evalplus, "__version__", "unknown")


def build_code_index(n: int = DEFAULT_CODE_N,
                     loader: Optional[Callable[[str], Dict[str, dict]]] = None,
                     benchmarks: Optional[Tuple[str, ...]] = None,
                     expected_items: Optional[Dict[str, int]] = None,
                     now: Callable[[], str] = _utcnow,
                     version: Optional[Callable[[], str]] = None,
                     ) -> Tuple[Set[str], dict]:
    """`(ngrams, provenance)` over every item of every code benchmark.

    Raises `IncompleteIndex`, carrying every problem found rather than the
    first, when the index would not cover what phase 8 is gated on.
    """
    from daedalus.data import ngram_set

    loader = loader or _default_code_loader
    benchmarks = tuple(benchmarks) if benchmarks is not None else CODE_BENCHMARKS
    expected_items = (CODE_EXPECTED_ITEMS if expected_items is None
                      else expected_items)
    version = version or _evalplus_version

    problems_report: List[str] = []
    ngrams: Set[str] = set()
    benchmarks_meta: Dict[str, dict] = {}

    for name in benchmarks:
        try:
            problems = loader(name)
        except Exception as exc:            # noqa: BLE001 - a load failure is a gap
            # `load_all_tasks`-style tolerance is wrong here for the same reason
            # it is wrong in `eval_index`: a benchmark that failed to download
            # would produce an index that filters nothing against it and says
            # nothing about it.
            problems_report.append(
                f"benchmark {name!r} failed to load: {exc!r}")
            continue
        if not problems:
            problems_report.append(f"benchmark {name!r} contributed no items")
            continue

        wanted = expected_items.get(name)
        if wanted is not None and len(problems) != wanted:
            problems_report.append(
                f"benchmark {name!r} loaded {len(problems):,} items but is "
                f"scored at {wanted:,}")

        fields: Dict[str, int] = {}
        short_fields: Dict[str, int] = {}
        unindexable: List[str] = []
        for task_id, problem in problems.items():
            texts = item_texts(problem)
            if REQUIRED_FIELD not in texts:
                unindexable.append(f"{task_id} (no {REQUIRED_FIELD})")
                continue
            for field in INDEXED_FIELDS:
                text = texts.get(field)
                if text is None:
                    continue
                grams = ngram_set(text, n)
                if grams:
                    fields[field] = fields.get(field, 0) + len(grams)
                    ngrams |= grams
                else:
                    short_fields[field] = short_fields.get(field, 0) + 1
                    if field == REQUIRED_FIELD:
                        unindexable.append(
                            f"{task_id} ({REQUIRED_FIELD} is shorter than "
                            f"{n} tokens)")
        if unindexable:
            shown = ", ".join(sorted(unindexable)[:5])
            more = "" if len(unindexable) <= 5 else f" and {len(unindexable) - 5} more"
            problems_report.append(
                f"benchmark {name!r} has {len(unindexable):,} item(s) this "
                f"index cannot filter at all: {shown}{more}")

        benchmarks_meta[name] = {
            "items": len(problems),
            "fields": dict(sorted(fields.items())),
            "short_fields": dict(sorted(short_fields.items())),
        }

    if problems_report:
        raise IncompleteIndex(problems_report)

    provenance = {
        "schema": SCHEMA,
        "kind": "code",
        "n": n,
        "limit": None,
        # No partial mode. The general index has one because reproducing the
        # released corpus's 2,000-item filter was how its exposure was measured;
        # there is no historical code index to reproduce, so "complete" here is
        # a statement rather than a switch, and `write_index` refuses anything
        # else.
        "complete": True,
        "ngrams": len(ngrams),
        "digest": index_digest(ngrams),
        "built_at": now(),
        "evalplus": version(),
        "benchmarks": benchmarks_meta,
    }
    return ngrams, provenance


def code_coverage_problems(provenance: dict,
                           benchmarks: Optional[Tuple[str, ...]] = None,
                           expected_items: Optional[Dict[str, int]] = None,
                           expected_n: int = DEFAULT_CODE_N,
                           ) -> List[str]:
    """Everything wrong with a code index *as described by its own sidecar*.

    Run against a file the corpus build was handed, months after the build that
    wrote it: the question is not "did this build work" but "does the index this
    manifest names still cover what phase 8 is gated on, at the `n` the filter
    will look it up with".
    """
    benchmarks = tuple(benchmarks) if benchmarks is not None else CODE_BENCHMARKS
    expected_items = (CODE_EXPECTED_ITEMS if expected_items is None
                      else expected_items)

    problems: List[str] = []
    if provenance.get("schema") != SCHEMA:
        problems.append(f"unknown schema {provenance.get('schema')!r}")
    if provenance.get("kind") != "code":
        problems.append(
            f"index kind is {provenance.get('kind')!r}, not 'code'; a general "
            f"eval index does not cover HumanEval+ or MBPP+")
    if not provenance.get("complete"):
        problems.append("index is partial")
    got_n = provenance.get("n")
    if got_n != expected_n:
        # The silent one. Unioned into the general set and looked up at the
        # filter's `n`, an index built at another length matches nothing.
        problems.append(
            f"index is {got_n!r}-gram but the corpus filter looks up "
            f"{expected_n}-grams; it would union cleanly and match nothing")

    meta = provenance.get("benchmarks") or {}
    for name in benchmarks:
        entry = meta.get(name)
        if not entry:
            problems.append(f"benchmark {name!r} is not in the index")
            continue
        items = entry.get("items")
        if not items:
            problems.append(f"benchmark {name!r} contributed no items")
        elif expected_items.get(name) not in (None, items):
            problems.append(
                f"benchmark {name!r} was indexed at {items:,} items but is now "
                f"scored at {expected_items[name]:,}")
        if not (entry.get("fields") or {}):
            problems.append(f"benchmark {name!r} indexed no field")
    return problems


def code_manifest_record(provenance: dict, path: Optional[str] = None) -> dict:
    """The compact form a code corpus manifest carries for the index it used.

    Mirrors `eval_index.manifest_record` so a manifest can hold both under one
    shape, and keeps `short_fields` -- the part of the benchmark this index does
    not cover is exactly what a later reader needs and cannot recompute.
    """
    meta = provenance.get("benchmarks") or {}
    return {
        "digest": provenance.get("digest"),
        "path": path,
        "kind": "code",
        "n": provenance.get("n"),
        "ngrams": provenance.get("ngrams"),
        "complete": bool(provenance.get("complete")),
        "built_at": provenance.get("built_at"),
        "evalplus": provenance.get("evalplus"),
        "items": {name: entry.get("items") for name, entry in sorted(meta.items())},
        "short_fields": {name: dict(entry.get("short_fields") or {})
                         for name, entry in sorted(meta.items())},
    }
