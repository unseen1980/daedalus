"""The code corpus's admission gate and its frozen decontamination index.

Two things live here, and they answer the two questions phase 8 asks of every
row of GitHub it is about to tokenize: *may this text be in the corpus at all*
(the licence gate, `RepositoryGate`), and *is this text one of the benchmark
answers we are about to be graded on* (the decontamination index). The first
half is below the second, under "the admission gate".

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

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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


# ======================================================= the admission gate ===
#
# The plan gives phase 8's corpus two properties that the general corpus never
# needed, and both are decided per *row*, before a document is ever tokenized:
#
#   "Filter to approved permissive licenses and record repository identity,
#    commit/revision, and license. Split train/holdout by repository, not file
#    or packed window."
#
# `RepositoryGate` is both, as one callable, because they are asked of the same
# row at the same moment and because `dataprep.SourceSpec.filter_fn` is exactly
# one row predicate. Making it one object also makes the *record* possible: a
# free function would answer each row and remember nothing, and "which
# repositories, under which licences, went into this shard directory" is a
# question the manifest has to answer afterwards.
#
# Two failure modes shape everything below.
#
# **A licence deny-list fails open.** Refusing the copyleft strings you thought
# of admits every string you did not -- a new upstream value, a typo, a `None`
# from a row whose licence column was never populated -- and the artifact that
# results is a model whose training data cannot be described. So the gate is an
# allow-list: `permissive` is a closed set, everything else is refused, and an
# unrecognised value is refused *and counted separately* from a recognised
# copyleft one. The counters are the point: `unknown_license` climbing is how a
# schema change announces itself, where a deny-list would simply have let it
# through.
#
# **A split that is not a pure function of the repository leaks.** The holdout
# exists so that phase 8's code BPB means something, and it is worthless the
# moment one repository's files land on both sides -- two files from the same
# project share idioms, helpers, licence headers and often whole functions, so a
# leaked repository inflates the holdout score exactly like training on it
# would. `repository_split` is therefore a hash of the repository name and
# nothing else: not stream order, not a counter, not `hash()` (which is salted
# per process, so a resumed build would re-partition every repository it saw and
# the leak would arrive silently on the second attempt). Given the same name it
# returns the same side in every process, on every attempt, for the life of the
# program -- which is what lets the train pass and the holdout pass be two
# independent streams that cannot overlap.
#
# A row whose repository cannot be identified is refused rather than defaulted.
# There is no side to put it on: `want="train"` and `want="holdout"` would both
# have to guess, and if they guessed the same way the same document would enter
# both splits, which is the leak this exists to prevent. Refusing is the only
# answer that keeps the two passes disjoint, and `no_repository` counts it.

#: Licences a phase 8 shard may contain. An allow-list, deliberately: see above.
#: Public-domain dedications (`cc0-1.0`, `unlicense`, `0bsd`) are included
#: because they impose strictly less than MIT does.
PERMISSIVE_LICENSES = frozenset({
    "0bsd", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "isc",
    "mit", "unlicense",
})

#: Licences known to be *refused*, kept by name so that "we know this one and it
#: does not qualify" is distinguishable in the counters from "we have never seen
#: this string". The weak/file-level copyleft entries (`mpl-2.0`, `epl-1.0`,
#: `lgpl-*`) are here rather than above because the plan says *permissive*, and
#: the reciprocal obligation they carry is the thing permissive means the
#: absence of. `artistic-2.0` is refused on the same reading.
KNOWN_NON_PERMISSIVE_LICENSES = frozenset({
    "agpl-3.0", "artistic-2.0", "epl-1.0", "epl-2.0", "gpl-2.0", "gpl-3.0",
    "lgpl-2.1", "lgpl-3.0", "mpl-2.0", "osl-3.0",
})

#: Row keys that have carried a repository's identity in the code datasets this
#: program has read, most specific first. Several rather than one because the
#: gate is pointed at more than one dataset and a wrong guess here is silent:
#: every row would answer "no repository", every row would be refused, and the
#: build would produce an empty corpus with a clean exit. `gate.manifest()`
#: reports which key actually answered, and `scripts/codeprep.py corpus probe`
#: is how that is checked against real rows before a build is launched.
REPOSITORY_FIELDS = ("repo_name", "repository_name", "max_stars_repo_name",
                     "repo", "repo_id")

#: Mixed into the repository hash so the phase 8 split is this program's own.
#: Pinned to the code branch's name: re-deriving the split later reproduces it
#: only if this string is reproduced with it, so it is recorded in every
#: manifest the gate writes.
SPLIT_SALT = "vast/daedalus-code-20260824"

#: Fraction of *repositories* -- not documents, not tokens -- held out.
DEFAULT_HOLDOUT_FRAC = 0.02

SPLITS = ("train", "holdout")

#: How many distinct repository names one gate records verbatim before it stops
#: keeping the list and keeps only the counts. Bounded because a code source at
#: phase 8's scale streams repositories in the millions, and an unbounded set of
#: names inside a `dataprep` worker is exactly the "accumulate instead of
#: stream" growth that its RSS caps exist to catch -- a gate that OOMs the
#: worker it was added to protect provenance for is a bad trade. The counts and
#: the licence histogram stay exact past the cap; only the enumeration stops.
DEFAULT_MAX_REPOSITORIES = 200_000


#: The plan's code-language shares, over the 65% of phase 8's mixture that is
#: code. Keys are *buckets* rather than languages because the plan groups some
#: ("JavaScript/TypeScript 12%", "C/C++ 10%", "shell/SQL/other 4%") and a bucket
#: is the unit a token budget is set on.
CODE_LANGUAGE_SHARES = {
    "python": 0.55,
    "javascript-typescript": 0.12,
    "c-cpp": 0.10,
    "rust": 0.08,
    "go": 0.06,
    "java": 0.05,
    "shell-sql-other": 0.04,
}

assert abs(sum(CODE_LANGUAGE_SHARES.values()) - 1.0) < 1e-9, \
    "code language shares must sum to 1.0"

#: The plan's continued-pretraining mixture: what the 65% above is 65% *of*.
CORPUS_SHARES = {"code": 0.65, "technical": 0.15, "general-replay": 0.20}

assert abs(sum(CORPUS_SHARES.values()) - 1.0) < 1e-9, \
    "corpus shares must sum to 1.0"

#: Candidate `codeparrot/github-code` parquet directories per bucket -- what to
#: *ask* for, not what exists. The names carry real spelling risk (`GO` is
#: upper-case in this dataset's own vocabulary, `C++` and `C#` contain
#: characters a path may escape), and a directory that does not resolve fails by
#: yielding no rows, which is indistinguishable from a language with no
#: permissively licensed code in it. `scripts/codeprep.py corpus probe` resolves
#: each one against the real dataset and says which answered; nothing should
#: build a token budget on this table before it has.
GITHUB_CODE_LANGUAGES = {
    "python": ("Python-all",),
    "javascript-typescript": ("JavaScript-all", "TypeScript-all"),
    "c-cpp": ("C-all", "C++-all"),
    "rust": ("Rust-all",),
    "go": ("GO-all",),
    "java": ("Java-all",),
    "shell-sql-other": ("Shell-all", "SQL-all"),
}

assert set(GITHUB_CODE_LANGUAGES) == set(CODE_LANGUAGE_SHARES), \
    "every code bucket needs a source and every source needs a share"

#: The `language` values each bucket is, in the interleaved directory's own
#: vocabulary rather than the plan's. Compared case-folded (`normalize_language`),
#: because this dataset writes `GO` and `FORTRAN` upper-case and `Rust` and
#: `Python` not, and an exact match against the plan's spelling would refuse
#: every row -- which is what naming `GO-all` from a language list already did to
#: four directories.
GITHUB_CODE_BUCKET_LANGUAGES = {
    "python": ("Python",),
    "javascript-typescript": ("JavaScript", "TypeScript"),
    "c-cpp": ("C", "C++"),
    "rust": ("Rust",),
    "go": ("GO",),
    "java": ("Java",),
    "shell-sql-other": ("Shell", "SQL"),
}

assert set(GITHUB_CODE_BUCKET_LANGUAGES) == set(CODE_LANGUAGE_SHARES), \
    "every code bucket needs its language names and every one needs a share"

#: The directory that carries every language at once, and the only source on the
#: pinned revision for the four buckets with no directory of their own.
INTERLEAVED_CONFIG = "all-all"

#: Below this share of the code mixture a bucket is dropped rather than carried.
#: A bucket reduced to a fraction of a percent is a few million tokens of a
#: language in a multi-billion-token corpus: too little to teach the language and
#: enough to make a model card claim it. Dropping it by name is the honest
#: version of the same outcome.
MIN_BUCKET_SHARE = 0.005

#: Interleaved-directory bytes the build may admit per byte of code corpus it
#: produces. One pass means the fallback stream costs as much again as the entire
#: rest of the code corpus, which is already a large concession for the 18% of
#: the mixture that has no directory of its own.
#:
#: It is a parameter rather than a verdict: `source_plan` records every bucket's
#: `required_passes`, so the same measurement re-derives the mixture at any other
#: budget without re-reading a row.
DEFAULT_INTERLEAVE_PASSES = 1.0

#: Bytes per token the released 49,152-entry SmolLM2 tokenizer achieves on code
#: -- the tokenizer Daedalus-Code inherits, so the one its budgets are counted
#: in. Measured by phase 4's fertility pass over 327 code documents
#: (`runs/tokenizer-lab/measurements.json`, `49152-smollm2`/`fertility`/`code`),
#: not assumed at the ~4.0 that general text gives: code is denser, and taking
#: the general figure would understate every directory's supply by 40%.
CODE_BYTES_PER_TOKEN = 2.862

GITHUB_CODE_DATASET = "codeparrot/github-code"

#: The parquet-converted revision, as `dataprep.MIXTURE`'s `stack-edu-python`
#: reads it -- the same dataset, the same access path, so phase 8's Python
#: bucket and the general corpus's code share are drawn the same way.
GITHUB_CODE_REVISION = "refs/convert/parquet"


def github_code_data_files(config: str) -> str:
    """The `data_files` glob for one `codeparrot/github-code` config."""
    return f"{config}/partial-train/*.parquet"


def normalize_license(value) -> str:
    """One lowercase licence identifier out of whatever a row carries.

    Rows have carried a string, a list (multi-licensed projects), and `None`
    (the column exists but was never populated). All three normalize to
    something the allow-list can be asked about, and the empty string is what
    "no licence stated" becomes -- which the gate refuses, because an
    unlicensed public repository is not a permissively licensed one. Default
    copyright is the most restrictive answer, not the most permissive.

    A list normalizes to its first entry rather than to a joined string so that
    the value in the counters is a licence someone can look up.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().lower()


def normalize_language(value) -> str:
    """One case-folded language name out of whatever a row carries.

    Case-folded because this dataset's own vocabulary is inconsistent about it
    -- `GO` and `FORTRAN` are upper-case where `Python` and `Rust` are not --
    and an exact-match language filter against the wrong spelling fails exactly
    the way an exact-match *directory* name did: every row refused, an empty
    shard directory, a zero exit. That mistake has already been made once here
    (`GO-all`), and it cost a probe to find.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().lower()


def license_verdict(value) -> str:
    """`"permissive"`, `"non-permissive"` or `"unknown"` for one licence value.

    Three outcomes, not two, because only `"unknown"` is *news*. A build whose
    `non_permissive` counter is large is working as designed; one whose
    `unknown_license` counter is large has met a vocabulary this module does not
    know, and the honest response is to look at the strings -- which is why they
    are kept verbatim in the histogram -- rather than to widen the allow-list on
    a guess.
    """
    key = normalize_license(value)
    if key in PERMISSIVE_LICENSES:
        return "permissive"
    if key in KNOWN_NON_PERMISSIVE_LICENSES:
        return "non-permissive"
    return "unknown"


def repository_of(row: dict) -> Optional[str]:
    """`owner/name` for this row, lowercased, or None when it has none.

    Lowercased because GitHub treats `Owner/Repo` and `owner/repo` as one
    project while `blake2b` does not: two spellings of one repository would hash
    to different buckets and land on both sides of the split, which is the leak
    the split exists to prevent. Case is not part of a repository's identity
    here, so it is not part of what is hashed.
    """
    for key in REPOSITORY_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def repository_field_of(row: dict) -> Optional[str]:
    """Which of `REPOSITORY_FIELDS` answered for this row, for the manifest."""
    for key in REPOSITORY_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return key
    return None


def repository_bucket(repository: str, *, salt: str = SPLIT_SALT) -> float:
    """This repository's position in `[0, 1)`, stable across processes and runs.

    `blake2b` rather than `hash()`: Python's string hash is randomized per
    interpreter unless `PYTHONHASHSEED` is set, so a `hash()`-based split
    silently re-partitions every repository whenever the build restarts. Phase
    8's build is long enough to restart, and the resulting overlap between the
    two passes would be invisible -- the shards would look ordinary, the holdout
    would simply read better than it should.
    """
    digest = hashlib.blake2b(f"{salt}\x00{repository}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def repository_split(repository: str, *, holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                     salt: str = SPLIT_SALT) -> str:
    """`"holdout"` or `"train"` for one repository -- a total, pure function.

    Total on purpose: every repository gets a side, so the two passes together
    admit each repository exactly once, and neither can drop one by disagreeing
    about it.
    """
    if not 0.0 < holdout_frac < 1.0:
        # A zero holdout is a corpus whose code BPB cannot be measured on
        # unseen repositories, and a one is a corpus with no training data.
        # Both are almost certainly a units mistake (2 for 2%), and both fail
        # far more expensively later than here.
        raise ValueError(
            f"holdout_frac must be strictly between 0 and 1, got {holdout_frac!r}")
    return "holdout" if repository_bucket(repository, salt=salt) < holdout_frac else "train"


#: The reasons a row is refused, in the order the gate checks them. Fixed so a
#: manifest always carries every key -- an absent counter and a zero one read
#: identically otherwise, and "this build refused nothing for licence reasons"
#: is a claim worth being able to make.
REFUSAL_REASONS = ("other_language", "no_repository", "non_permissive",
                   "unknown_license", "other_split")


@dataclass
class RepositoryGate:
    """`SourceSpec.filter_fn` for one side of one code source.

    Constructed once per source per split and handed to `dataprep.run_source`,
    which calls it with every raw row before anything is tokenized. It answers
    the row and remembers what it answered, so the shard directory it fills can
    be manifested with the repositories and licences that produced it.

    Two gates over the same source with `want="train"` and `want="holdout"`, the
    same `holdout_frac` and the same `salt` partition it: `repository_split` is
    total and pure, so every repository is admitted by exactly one of them and
    no document can reach both. That is what makes the holdout a second
    independent stream rather than a second reading of the first.

    `languages`, when given, additionally keeps only rows of those languages.
    Four of the plan's seven buckets -- Rust, Go, Shell and SQL, 18% of the code
    mixture -- have no per-language directory on the pinned revision, and the
    interleaved `all-all` directory that does carry them carries everything
    else too. A language filter is the only way to read a bucket out of it, and
    it is deliberately *not* the default: a filter over a directory that is
    already one language costs a comparison per row and hides a misspelling.
    """

    want: str
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC
    salt: str = SPLIT_SALT
    max_repositories: int = DEFAULT_MAX_REPOSITORIES
    #: Language names to keep, case-folded, or None to keep every language.
    languages: Optional[frozenset] = None
    seen: int = 0
    admitted: int = 0
    refusals: Dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in REFUSAL_REASONS})
    #: Every distinct licence string met, verbatim, with its row count -- the
    #: admitted ones and the refused ones alike. Unbounded in principle and
    #: tiny in practice: a licence vocabulary is tens of strings, and if it ever
    #: is not, that is the finding.
    licenses: Dict[str, int] = field(default_factory=dict)
    #: Which `REPOSITORY_FIELDS` key answered, with its row count. More than one
    #: entry means the source is not shaped the way it was thought to be.
    repository_fields: Dict[str, int] = field(default_factory=dict)
    _repositories: Set[str] = field(default_factory=set, repr=False)
    repositories_truncated: bool = False

    def __post_init__(self) -> None:
        if self.want not in SPLITS:
            raise ValueError(f"want must be one of {SPLITS}, got {self.want!r}")
        # Validated here rather than at the first row, so a units mistake is a
        # refusal at construction instead of an empty corpus hours later.
        repository_split("", holdout_frac=self.holdout_frac, salt=self.salt)
        if self.languages is not None:
            wanted = frozenset(normalize_language(name)
                               for name in self.languages) - {""}
            if not wanted:
                # An empty allow-list refuses every row, which is
                # indistinguishable from a language with no permissive code in
                # it. `languages=None` is how "every language" is asked for.
                raise ValueError(
                    "languages was given but names no language; pass None to "
                    "keep every language")
            self.languages = wanted

    def __call__(self, row: dict) -> bool:
        self.seen += 1
        if self.languages is not None and \
                normalize_language(row.get("language")) not in self.languages:
            # Checked before the licence is tallied, not after: a gate scoped to
            # one language out of an interleaved directory must manifest *that
            # language's* licence vocabulary. Tallying every row first would
            # describe the directory instead, and the manifest would report a
            # licence mix no document in the shard was drawn from.
            self.refusals["other_language"] += 1
            return False
        license_key = normalize_license(row.get("license"))
        self.licenses[license_key] = self.licenses.get(license_key, 0) + 1

        repository = repository_of(row)
        if repository is None:
            self.refusals["no_repository"] += 1
            return False
        field_key = repository_field_of(row)
        if field_key:
            self.repository_fields[field_key] = \
                self.repository_fields.get(field_key, 0) + 1

        verdict = license_verdict(row.get("license"))
        if verdict != "permissive":
            self.refusals["non_permissive" if verdict == "non-permissive"
                          else "unknown_license"] += 1
            return False

        if repository_split(repository, holdout_frac=self.holdout_frac,
                            salt=self.salt) != self.want:
            self.refusals["other_split"] += 1
            return False

        self.admitted += 1
        if len(self._repositories) < self.max_repositories:
            self._repositories.add(repository)
        elif repository not in self._repositories:
            self.repositories_truncated = True
        return True

    @property
    def repositories(self) -> List[str]:
        """The admitted repository names recorded so far, sorted.

        Empty of refused ones by construction: a repository this gate never
        admitted a row from does not belong in this shard directory's manifest,
        whichever side of the split it is on.
        """
        return sorted(self._repositories)

    @property
    def repositories_count(self) -> int:
        """How many distinct repositories were recorded -- which past
        `max_repositories` is the cap, not the number admitted. `rows_admitted`
        is the count that keeps growing."""
        return len(self._repositories)

    def manifest(self) -> dict:
        """What this gate admitted, in the shape a source manifest carries.

        `split` and the two parameters beside it are written every time because
        the split is only reproducible from them: the same repository name under
        a different `salt` or `holdout_frac` is a different side, and a manifest
        that records the outcome without the parameters records something nobody
        can re-derive.
        """
        return {
            "split": self.want,
            "holdout_frac": self.holdout_frac,
            "split_salt": self.salt,
            "split_fn": "blake2b-64(salt\\0repository) / 2**64",
            # Recorded for the reason the split parameters are: a shard read out
            # of the interleaved directory under a language filter is a
            # different corpus from the same directory read whole, and nothing
            # in the shards themselves says which one this was.
            "languages": (None if self.languages is None
                          else sorted(self.languages)),
            "rows_seen": self.seen,
            "rows_admitted": self.admitted,
            "refused": dict(self.refusals),
            "licenses": dict(sorted(self.licenses.items())),
            "permissive_licenses": sorted(PERMISSIVE_LICENSES),
            "repository_fields": dict(sorted(self.repository_fields.items())),
            "repositories": self.repositories_count,
            "repositories_truncated": self.repositories_truncated,
            "repository_names": self.repositories,
        }


def probe_source(config: str, *, rows: int = 2_000,
                 stream: Optional[Callable[[str], object]] = None,
                 holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                 salt: str = SPLIT_SALT,
                 languages: Optional[Sequence[str]] = None) -> dict:
    """Run the gate over real rows of one source and report what it found.

    Every field name and licence string in this module is a *guess* about a
    dataset until a row of it has been read, and each guess fails silently in
    the same direction: `repo_name` spelled wrong makes `repository_of` answer
    None for every row, the gate refuses all of them, and the build finishes
    early with an empty shard directory and a zero exit. A licence vocabulary
    that is not the one assumed does the same thing more quietly still -- it
    refuses most rows and keeps a biased remainder.

    So this is not a dry run of the build; it is the measurement the build's
    assumptions rest on. It reports the columns a row actually has, which
    repository field answered, every licence string met with its count, and how
    the admitted repositories divided -- and it is cheap enough (a couple of
    thousand streamed rows) to run before every build rather than once.

    A config that does not resolve is reported as `resolved: false` with the
    error rather than raised, because the useful output is the whole table: one
    misspelled directory should not hide the six that were right.

    `languages` scopes the gate to those languages, which is how a bucket with
    no directory of its own gets measured against the interleaved `all-all`.
    What that measurement is *for* is `stream_amplification` and
    `admitted_bytes`: reading Rust out of a directory that is 0.27% Rust means
    streaming, decompressing and gate-checking some hundreds of rows per row
    kept, and whether that is affordable at the bucket's preregistered share is
    a question about the rate, not about whether the rows exist. Estimating it
    from a language histogram taken over *unadmitted* rows would be wrong in the
    one direction that matters, since the licence gate refuses about a third of
    this dataset and there is no reason its refusal rate is uniform by language.
    """
    record: dict = {"dataset": GITHUB_CODE_DATASET, "config": config,
                    "data_files": github_code_data_files(config),
                    "rows_requested": rows,
                    "languages_kept": (None if languages is None
                                       else sorted(normalize_language(name)
                                                   for name in languages))}
    gate = RepositoryGate(want="train", holdout_frac=holdout_frac, salt=salt,
                          max_repositories=rows, languages=languages)
    holdout = RepositoryGate(want="holdout", holdout_frac=holdout_frac, salt=salt,
                             max_repositories=rows, languages=languages)
    columns: Dict[str, int] = {}
    seen_languages: Dict[str, int] = {}
    admitted_languages: Dict[str, Dict[str, int]] = {}
    admitted_bytes = {"train": 0, "holdout": 0}
    samples: List[dict] = []
    try:
        for index, row in enumerate(stream(config) if stream is not None
                                    else _stream_github_code(config)):
            if index >= rows:
                break
            for key in row:
                columns[key] = columns.get(key, 0) + 1
            # What is *in* the directory, as opposed to what its name says. The
            # per-language directories this module reaches for turned out not to
            # exist for four of the plan's buckets, which makes the interleaved
            # `all-all` the fallback -- and whether a language is reachable there
            # at a usable rate is a question only the histogram answers.
            language = str(row.get("language") or "").strip()
            if language:
                seen_languages[language] = seen_languages.get(language, 0) + 1
            train_ok, holdout_ok = gate(row), holdout(row)
            chars = len(row.get("code") or "")
            if train_ok:
                admitted_bytes["train"] += chars
            elif holdout_ok:
                admitted_bytes["holdout"] += chars
            if train_ok or holdout_ok:
                # Per language, so one pass over the interleaved directory sizes
                # every bucket drawn from it. Four separate filtered probes would
                # answer the same question by streaming the same rows four
                # times, and the rows are the expensive part.
                entry = admitted_languages.setdefault(
                    normalize_language(language), {"rows": 0, "bytes": 0})
                entry["rows"] += 1
                entry["bytes"] += chars
            if (train_ok or holdout_ok) and len(samples) < 3:
                samples.append({"repository": repository_of(row),
                                "license": normalize_license(row.get("license")),
                                "split": "train" if train_ok else "holdout",
                                "chars": chars})
    except Exception as exc:                    # noqa: BLE001 - reported per config
        record.update({"resolved": False, "error": repr(exc)})
        return record

    train_manifest = gate.manifest()
    kept = gate.admitted + holdout.admitted
    record.update({
        "resolved": gate.seen > 0,
        "rows_read": gate.seen,
        "columns": dict(sorted(columns.items())),
        "languages": dict(sorted(seen_languages.items(),
                                 key=lambda kv: (-kv[1], kv[0]))),
        # What survived the gate, per language. `languages` above counts rows
        # *offered*; a share can only be budgeted from rows kept, and there is
        # no reason the licence gate -- which refuses about a third of this
        # dataset -- refuses at the same rate in every language.
        "admitted_languages": dict(sorted(admitted_languages.items(),
                                          key=lambda kv: (-kv[1]["rows"], kv[0]))),
        "repository_fields": train_manifest["repository_fields"],
        "licenses": train_manifest["licenses"],
        "admitted": {"train": gate.admitted, "holdout": holdout.admitted},
        "admitted_bytes": dict(admitted_bytes),
        # Rows that must be streamed per row kept. The number a token budget
        # divides by, and the reason a language reachable in principle can still
        # be unaffordable: `None` rather than a division by zero when nothing
        # survived, since "infinitely expensive" is already said by `problem`.
        "stream_amplification": (round(gate.seen / kept, 2) if kept else None),
        "repositories": {"train": gate.repositories_count,
                         "holdout": holdout.repositories_count},
        # The two gates refuse each other's rows under `other_split`, so the
        # licence and repository refusals are the ones that describe the source.
        "refused": {reason: train_manifest["refused"][reason]
                    for reason in REFUSAL_REASONS if reason != "other_split"},
        "samples": samples,
    })
    if gate.seen and not kept:
        # The silent failure, named. Every one of these has the same shape --
        # rows arrived and none survived -- and none of them raises.
        cause = ("the repository field, the licence vocabulary or the split is "
                 "not what this module assumes")
        if languages is not None:
            # With a filter on, the likeliest cause is a language name spelled
            # the way the plan spells it rather than the way the dataset does,
            # and `refused.other_language` says which of the two it was.
            cause = ("no row of {} survived; {:,} were refused on language "
                     "alone, so check the spelling against the `languages` "
                     "histogram before concluding the language is absent"
                     .format(", ".join(record["languages_kept"]),
                             train_manifest["refused"]["other_language"]))
        record["problem"] = "read {:,} rows and admitted none: {}".format(
            gate.seen, cause)
    return record


def github_code_configs(*, dataset: str = GITHUB_CODE_DATASET,
                        revision: str = GITHUB_CODE_REVISION,
                        api=None) -> Dict[str, int]:
    """`{directory: parquet files}` that actually exist on the revision.

    Asked of the repository rather than guessed, because guessing has already
    cost one probe: four of the ten directories in `GITHUB_CODE_LANGUAGES` --
    18% of the plan's code mixture -- came back `DataFilesNotFoundError`, and a
    `data_files` glob that matches nothing raises only because `datasets`
    happens to check. A build would have had every reason to read it as "this
    language has no permissively licensed code".

    The auto-converted parquet branch is a *subset*: the converter has size
    limits and does not convert every config of a large dataset, so absence here
    is as likely to mean "never converted" as "misspelled". Either way the
    remedy is the same -- read the list.

    `dataset` is a parameter rather than a constant because the answer this
    returns is how a substitute source gets chosen: `codeparrot/github-code`
    turned out to carry no Go, Rust, Shell or SQL directory at all, and the
    question that follows -- which permissively licensed dataset does -- is the
    same question asked of a different repository.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    counts: Dict[str, int] = {}
    for path in api.list_repo_files(dataset, repo_type="dataset",
                                    revision=revision):
        if not path.endswith(".parquet"):
            continue
        head = path.split("/", 1)[0]
        counts[head] = counts.get(head, 0) + 1
    return dict(sorted(counts.items()))


def missing_configs(available: Dict[str, int],
                    languages: Optional[Dict[str, Tuple[str, ...]]] = None,
                    ) -> Dict[str, List[str]]:
    """`{bucket: [config, ...]}` this module names that the revision does not
    carry, and the near-misses that suggest what it should have named instead.
    """
    languages = languages if languages is not None else GITHUB_CODE_LANGUAGES
    missing: Dict[str, List[str]] = {}
    for bucket, configs in sorted(languages.items()):
        absent = [config for config in configs if config not in available]
        if absent:
            missing[bucket] = absent
    return missing


def config_near_misses(config: str, available: Dict[str, int]) -> List[str]:
    """Directories that differ from `config` only in case or spacing.

    `GO-all` against a real `Go-all` is the whole class of failure here, and it
    is invisible to an exact lookup.
    """
    def fold(name: str) -> str:
        return name.replace(" ", "").replace("_", "-").lower()

    wanted = fold(config)
    return sorted(name for name in available if fold(name) == wanted
                  and name != config)


def _stream_github_code(config: str):
    """Streaming rows of one `codeparrot/github-code` config.

    Isolated so `probe_source` can be tested without the Hub, and shaped like
    `dataprep._stream_rows` so both reach the dataset the same way.
    """
    from datasets import load_dataset

    return load_dataset(GITHUB_CODE_DATASET, split="train", streaming=True,
                        revision=GITHUB_CODE_REVISION,
                        data_files=github_code_data_files(config))


def probe_languages(languages: Optional[Sequence[str]] = None, *,
                    rows: int = 2_000,
                    configs: Optional[Sequence[str]] = None,
                    stream: Optional[Callable[[str], object]] = None,
                    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                    salt: str = SPLIT_SALT,
                    keep_languages: Optional[Sequence[str]] = None) -> dict:
    """`probe_source` over every config of every requested bucket.

    `configs` probes named directories instead, under the bucket key
    `"unbucketed"`. That is how a directory nobody has assigned a share to yet
    -- the interleaved `all-all`, a candidate substitute source -- gets the same
    measurement as one that has one, without first pretending it is part of the
    mixture.

    `keep_languages` narrows those directories to a language, and belongs only
    to the `configs` path: a bucket's own directory is already one language, so
    a filter over it can only ever refuse rows it should have kept.
    """
    if keep_languages and not configs:
        raise ValueError(
            "keep_languages narrows a named directory and needs `configs`; a "
            "bucket's own directory is already the language it is named for")
    if configs:
        return {"holdout_frac": holdout_frac, "split_salt": salt,
                "rows_per_config": rows,
                "languages": {"unbucketed": {
                    "share": 0.0,
                    "configs": [probe_source(config, rows=rows, stream=stream,
                                             holdout_frac=holdout_frac, salt=salt,
                                             languages=keep_languages)
                                for config in configs]}}}
    buckets = list(languages or GITHUB_CODE_LANGUAGES)
    unknown = sorted(set(buckets) - set(GITHUB_CODE_LANGUAGES))
    if unknown:
        raise ValueError(f"unknown code bucket(s) {unknown}; known buckets are "
                         f"{sorted(GITHUB_CODE_LANGUAGES)}")
    report = {"holdout_frac": holdout_frac, "split_salt": salt,
              "rows_per_config": rows, "languages": {}}
    for bucket in buckets:
        report["languages"][bucket] = {
            "share": CODE_LANGUAGE_SHARES[bucket],
            "configs": [probe_source(config, rows=rows, stream=stream,
                                     holdout_frac=holdout_frac, salt=salt)
                        for config in GITHUB_CODE_LANGUAGES[bucket]],
        }
    return report


def probe_problems(report: dict) -> List[str]:
    """Everything in a probe that would make a build quietly produce nothing."""
    problems: List[str] = []
    for bucket, entry in sorted((report.get("languages") or {}).items()):
        for record in entry.get("configs") or []:
            name = f"{bucket}/{record.get('config')}"
            if not record.get("resolved"):
                problems.append(
                    f"{name} did not resolve: "
                    f"{record.get('error', 'it yielded no rows at all')}")
                continue
            if record.get("problem"):
                problems.append(f"{name} {record['problem']}")
            fields = record.get("repository_fields") or {}
            if not fields:
                problems.append(f"{name} has no repository field this module reads")
            elif len(fields) > 1:
                problems.append(
                    f"{name} answers to more than one repository field "
                    f"({', '.join(sorted(fields))}); the source is not shaped "
                    f"the way this module assumes")
            unknown = sorted(key for key in (record.get("licenses") or {})
                             if license_verdict(key) == "unknown")
            if unknown:
                # Not fatal -- an unknown licence is refused, which is the safe
                # direction -- but it is the finding the probe exists to surface.
                problems.append(
                    f"{name} carries {len(unknown)} licence value(s) this "
                    f"module does not classify: {', '.join(repr(u) for u in unknown[:8])}")
    return problems


def probe_record(report: dict, config: str) -> Optional[dict]:
    """The one directory's record inside a `probe_languages` report.

    A report keys its records by *bucket*, and the interleaved directory was
    probed with `--config`, which files it under `"unbucketed"`. Searching by the
    config name finds it either way, so the plan does not have to know which of
    the two ways its own evidence was gathered.
    """
    for entry in (report.get("languages") or {}).values():
        for record in entry.get("configs") or []:
            if record.get("config") == config:
                return record
    return None


def interleaved_bucket_yield(record: dict, *,
                             buckets: Optional[Dict[str, Tuple[str, ...]]] = None,
                             ) -> dict:
    """What one pass over the interleaved directory admits, per bucket.

    The measurement a fallback share is budgeted from. `record` is a
    `probe_source` record for `INTERLEAVED_CONFIG`, and the number taken from it
    is each bucket's fraction of the *admitted* bytes -- what survived the
    licence gate -- because a share can only be filled from rows the corpus is
    allowed to contain. The histogram of rows offered says something else and
    overstates every bucket, unevenly: the gate refuses about a third of this
    dataset and nothing makes it refuse at the same rate in every language.

    Refuses a record probed before per-language yield was recorded rather than
    defaulting it to zero, which would silently drop every fallback bucket.
    """
    buckets = buckets if buckets is not None else GITHUB_CODE_BUCKET_LANGUAGES
    admitted = record.get("admitted_languages")
    if admitted is None:
        raise ValueError(
            f"the probe of {record.get('config')!r} recorded no per-language "
            f"yield, so no fallback share can be budgeted from it; re-probe it")
    total = sum(entry["bytes"] for entry in admitted.values())
    measured: Dict[str, dict] = {}
    for bucket, names in sorted(buckets.items()):
        wanted = sorted({normalize_language(name) for name in names})
        rows = sum(admitted.get(name, {}).get("rows", 0) for name in wanted)
        admitted_bytes = sum(admitted.get(name, {}).get("bytes", 0)
                             for name in wanted)
        measured[bucket] = {
            "languages": wanted,
            "rows": rows,
            "bytes": admitted_bytes,
            "byte_fraction": (admitted_bytes / total) if total else 0.0,
        }
    return {
        "config": record.get("config"),
        "rows_read": record.get("rows_read"),
        "admitted_bytes": total,
        # Rows streamed per row kept, over the whole directory. The fallback
        # pays this on top of its own rarity: the budget below is counted in
        # admitted bytes, and this is what those bytes cost to read.
        "stream_amplification": record.get("stream_amplification"),
        "buckets": measured,
    }


def source_plan(*, available: Dict[str, int], interleaved: dict,
                passes: float = DEFAULT_INTERLEAVE_PASSES,
                shares: Optional[Dict[str, float]] = None,
                configs: Optional[Dict[str, Tuple[str, ...]]] = None,
                buckets: Optional[Dict[str, Tuple[str, ...]]] = None,
                min_share: float = MIN_BUCKET_SHARE) -> dict:
    """The code mixture this revision can actually serve, and at what shares.

    The plan's seven buckets assume seven per-language directories. Four of the
    ten directories they name do not exist on the parquet-converted revision --
    Go, Rust, Shell and SQL were never converted, with no near miss on spelling
    -- which is 18% of the code portion with no source of its own. The
    interleaved `all-all` directory does carry all of them, so the open question
    was never "are the rows there" but "what does reaching them cost", and that
    is a rate this takes from a measurement rather than from the plan's table.

    Three outcomes per bucket, in one pass over the same evidence:

    * every directory it names exists -- served from them at its plan share;
    * no directory, but the interleaved yield reaches at least `min_share` under
      the budget -- served from `INTERLEAVED_CONFIG` filtered to its languages,
      at whatever share the yield reaches, capped at the plan's;
    * no directory and the yield does not reach `min_share` -- dropped by name.

    What the capped and dropped buckets cannot serve is redistributed
    proportionally over the buckets that have their own directories, following
    `dataprep.GATED_SUBSTITUTION_NOTES` for the general corpus's gated sources.
    It cannot go to the fallback buckets: they are capped *because* the rows are
    not there, so handing them a larger target only moves where the shortfall is
    discovered from this function to a build that quietly comes up short.

    The result sums to 1.0 over the code portion and every input to it is
    recorded, so the decision is re-derivable rather than asserted.
    """
    shares = shares if shares is not None else CODE_LANGUAGE_SHARES
    configs = configs if configs is not None else GITHUB_CODE_LANGUAGES
    buckets = buckets if buckets is not None else GITHUB_CODE_BUCKET_LANGUAGES
    if passes < 0:
        raise ValueError(f"interleave passes must not be negative, got {passes}")
    measured = interleaved_bucket_yield(interleaved, buckets=buckets)

    entries: Dict[str, dict] = {}
    for bucket, plan_share in sorted(shares.items()):
        named = list(configs.get(bucket, ()))
        absent = [config for config in named if config not in available]
        entry: dict = {"plan_share": plan_share, "configs": named,
                       "missing_configs": absent}
        if not absent:
            entry.update({
                "source": "directories", "share": plan_share,
                "reason": "every directory this bucket names exists on the "
                          "revision",
            })
            entries[bucket] = entry
            continue
        yielded = measured["buckets"][bucket]
        fraction = yielded["byte_fraction"]
        reachable = passes * fraction
        entry.update({
            "source_config": INTERLEAVED_CONFIG,
            "languages": yielded["languages"],
            "interleaved_rows": yielded["rows"],
            "interleaved_bytes": yielded["bytes"],
            "yield_fraction": fraction,
            "reachable_share": reachable,
            # What the budget would have to be for this bucket to reach its plan
            # share. The one number that makes the drop re-derivable: nothing
            # here is unreachable in principle, only at a price, and this is the
            # price.
            "required_passes": (plan_share / fraction) if fraction else None,
        })
        if reachable < min_share:
            entry.update({
                "source": "dropped", "share": 0.0,
                "reason": f"{fraction:.3%} of the interleaved directory reaches "
                          f"{reachable:.3%} of the code mixture at {passes:g} "
                          f"pass(es), under the {min_share:.1%} floor",
            })
        else:
            entry.update({
                "source": "interleaved", "share": min(plan_share, reachable),
                "reason": f"{fraction:.3%} of the interleaved directory reaches "
                          f"{min(plan_share, reachable):.3%} of the code "
                          f"mixture at {passes:g} pass(es)",
            })
        entries[bucket] = entry

    shortfall = sum(entry["plan_share"] - entry["share"]
                    for entry in entries.values())
    absorbers = {bucket: entry for bucket, entry in entries.items()
                 if entry["source"] == "directories"}
    weight = sum(entry["plan_share"] for entry in absorbers.values())
    for bucket, entry in absorbers.items():
        entry["redistributed"] = (shortfall * entry["plan_share"] / weight
                                  if weight else 0.0)
        entry["share"] = entry["plan_share"] + entry["redistributed"]

    for entry in entries.values():
        entry["share"] = round(entry["share"], 6)

    return {
        "interleave_passes": passes,
        "min_bucket_share": min_share,
        "interleaved": {
            "config": measured["config"],
            "available": INTERLEAVED_CONFIG in available,
            "rows_read": measured["rows_read"],
            "admitted_bytes": measured["admitted_bytes"],
            "stream_amplification": measured["stream_amplification"],
            # The fraction of what the fallback stream reads that it keeps.
            # Its reciprocal is the read amplification the budget buys.
            "kept_fraction": round(sum(entry["yield_fraction"]
                                       for entry in entries.values()
                                       if entry["source"] == "interleaved"), 6),
        },
        "buckets": entries,
        "shares": {bucket: entry["share"] for bucket, entry in entries.items()
                   if entry["share"] > 0},
        "redistributed": round(shortfall, 6),
    }


def _parquet_row_count(url: str) -> int:
    """Rows in one parquet file, read from its footer.

    Isolated the way `_stream_github_code` is, so the supply measurement can be
    tested without the Hub.
    """
    import fsspec
    import pyarrow.parquet as pq

    with fsspec.open(url) as handle:
        return pq.ParquetFile(handle).metadata.num_rows


def config_row_counts(configs: Sequence[str], *,
                      dataset: str = GITHUB_CODE_DATASET,
                      revision: str = GITHUB_CODE_REVISION,
                      api=None,
                      count_rows: Optional[Callable[[str], int]] = None,
                      ) -> Dict[str, dict]:
    """How many rows each directory holds, from parquet footers.

    The one number the probe cannot supply. A probe measures the *rate* a
    directory admits bytes at, over the first few thousand rows; a supply is
    that rate times how many rows are actually there, and nothing in a stream
    says how many that is until it ends.

    Cheap enough to run before a build rather than after one: a footer is a few
    kilobytes at the end of a file, so a directory costs ten range requests
    instead of a pass over gigabytes. Phase 7 learned the same lesson the
    expensive way -- `stack-edu-python` was 139M tokens short of its share, and
    one metadata call would have said so before a document was streamed.

    A file whose footer cannot be read is recorded in `errors` and left out of
    the total, so a partial answer is visibly partial rather than a small one.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    count_rows = count_rows if count_rows is not None else _parquet_row_count
    paths = [path for path in api.list_repo_files(dataset, repo_type="dataset",
                                                  revision=revision)
             if path.endswith(".parquet")]
    counts: Dict[str, dict] = {}
    for config in configs:
        prefix = f"{config}/"
        wanted = sorted(path for path in paths if path.startswith(prefix))
        entry: dict = {"files": len(wanted), "files_read": 0, "rows": 0,
                       "errors": {}}
        for path in wanted:
            url = f"hf://datasets/{dataset}@{revision}/{path}"
            try:
                entry["rows"] += int(count_rows(url))
            except Exception as exc:            # noqa: BLE001 - reported per file
                entry["errors"][path] = repr(exc)
                continue
            entry["files_read"] += 1
        if not entry["files_read"]:
            # Zero readable files is not a directory of zero rows, and a supply
            # built on it would silently be zero. Say which it was.
            entry["rows"] = None
        counts[config] = entry
    return counts


def bucket_supply(*, plan: dict, row_counts: Dict[str, dict],
                  probes: Dict[str, dict],
                  bytes_per_token: float = CODE_BYTES_PER_TOKEN,
                  ) -> Dict[str, dict]:
    """Unique tokens each planned bucket can actually deliver.

    Two measured factors and no assumed ones: the rate a directory admits bytes
    at, from a probe of its real rows, times the rows it holds, from its parquet
    footers -- then over the tokenizer's measured bytes per token of code.

    This is the check `source_plan` cannot make on itself. Redistributing the
    unreachable buckets' 14 points onto the four with directories raises Python
    from 55% to 64%, and that is only sound if `Python-all` holds 64% of the
    budget. Otherwise the shortfall has not been fixed, only moved from a bucket
    that announces it to one that does not.

    A bucket missing either factor is reported with `unique_tokens` 0 and a
    basis that names what was missing, which reads downstream as infinite epochs
    -- the safe direction, and distinguishable from a measured zero.

    `per_config` carries the same arithmetic before it is summed, because the
    build has to divide a bucket's budget between its directories and the sum
    cannot answer that. `javascript-typescript` is one bucket over two of them,
    and the two are not the same size: splitting its 14% evenly would ask
    `TypeScript-all` for tokens it does not hold and stop short, while
    `JavaScript-all` sat under its own budget with rows to spare. That
    shortfall would be reported per directory and invisible per bucket, which is
    the shape of miscount this whole measurement exists to prevent.
    """
    supply: Dict[str, dict] = {}
    for bucket, entry in sorted((plan.get("buckets") or {}).items()):
        source = entry.get("source")
        if source == "directories":
            draws = [(config, _admitted_bytes(probes.get(config)))
                     for config in entry.get("configs") or []]
        elif source == "interleaved":
            config = entry.get("source_config") or INTERLEAVED_CONFIG
            draws = [(config, _admitted_language_bytes(
                probes.get(config), entry.get("languages") or []))]
        else:
            supply[bucket] = {"share": entry.get("share", 0.0),
                              "unique_bytes": 0, "unique_tokens": 0,
                              "configs": [], "missing": [], "partial": [],
                              "per_config": {},
                              "lower_bound": False,
                              "basis": f"{bucket} is not drawn from this "
                                       f"revision: {entry.get('reason', '')}"}
            continue

        unique_bytes = 0.0
        missing: List[str] = []
        partial: List[str] = []
        parts: List[str] = []
        per_config: Dict[str, int] = {}
        for config, admitted in draws:
            probe = probes.get(config) or {}
            rows_read = probe.get("rows_read") or 0
            counted = row_counts.get(config) or {}
            total = counted.get("rows")
            if admitted is None or not rows_read or not total:
                missing.append(config)
                continue
            per_row = admitted / rows_read
            unique_bytes += per_row * total
            per_config[config] = (int(per_row * total / bytes_per_token)
                                  if bytes_per_token > 0 else 0)
            files, files_read = counted.get("files") or 0, counted.get("files_read")
            of_files = ""
            if files and files_read is not None and files_read < files:
                # A directory counted from some of its files is a floor, not a
                # measurement -- one 429 that outlasts its retries takes a tenth
                # of `all-all` with it. It matters in one direction: a floor that
                # clears the epoch cap has cleared it, but a floor that fails it
                # may be failing on the files nobody read.
                partial.append(config)
                of_files = f" ({files_read} of {files} files read)"
            parts.append(f"{total:,} rows of {config}{of_files} at "
                         f"{per_row:,.0f} admitted bytes/row read")
        basis = " + ".join(parts)
        if partial:
            basis = f"lower bound -- {basis}"
        if missing:
            basis = ((basis + "; " if basis else "")
                     + "unmeasured: " + ", ".join(missing))
        supply[bucket] = {
            "share": entry.get("share", 0.0),
            "unique_bytes": int(unique_bytes),
            "unique_tokens": int(unique_bytes / bytes_per_token)
            if bytes_per_token > 0 else 0,
            "configs": [config for config, _ in draws],
            "missing": missing,
            "partial": partial,
            "per_config": per_config,
            "lower_bound": bool(partial),
            "basis": basis or "no directory measured for this bucket",
        }
    return supply


def _admitted_bytes(probe: Optional[dict]) -> Optional[int]:
    """Bytes one probe admitted, over both splits."""
    if not probe or probe.get("admitted_bytes") is None:
        return None
    return sum((probe.get("admitted_bytes") or {}).values())


def _admitted_language_bytes(probe: Optional[dict],
                             languages: Sequence[str]) -> Optional[int]:
    """Bytes one probe admitted in the named languages."""
    if not probe or probe.get("admitted_languages") is None:
        return None
    admitted = probe["admitted_languages"]
    return sum(admitted.get(normalize_language(name), {}).get("bytes", 0)
               for name in languages)


def plan_problems(plan: dict) -> List[str]:
    """Everything in a plan that a corpus build must not be run on.

    A dropped or capped bucket is not one of them: those are the decision this
    plan exists to make, recorded per bucket with the budget that would reverse
    it. What is a problem is a plan that cannot be built *as written* -- the
    fallback directory missing, a fallback language the directory carries no
    admitted row of (the spelling failure that already cost four directories,
    and which looks exactly like absence), or shares that do not add up.
    """
    problems: List[str] = []
    interleaved = plan.get("interleaved") or {}
    buckets = plan.get("buckets") or {}
    fallbacks = {bucket: entry for bucket, entry in buckets.items()
                 if entry.get("source") in ("interleaved", "dropped")}
    if fallbacks and not interleaved.get("available"):
        problems.append(
            f"{len(fallbacks)} bucket(s) fall back to {INTERLEAVED_CONFIG!r} and "
            f"the revision does not carry that directory either")
    for bucket, entry in sorted(fallbacks.items()):
        if not entry.get("interleaved_rows"):
            problems.append(
                f"{bucket} has no directory of its own and the interleaved one "
                f"admitted no row of {', '.join(entry.get('languages') or [])}; "
                f"check the spelling against the probe's language histogram "
                f"before reading this as absence")
    total = sum(entry["share"] for entry in buckets.values())
    if buckets and abs(total - 1.0) > 1e-6:
        problems.append(
            f"the planned shares sum to {total:.6f}, not 1.0; the shortfall from "
            f"{plan.get('redistributed')} was not fully redistributed")
    return problems


def split_is_disjoint(repositories: Sequence[str],
                      *, holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                      salt: str = SPLIT_SALT) -> dict:
    """Partition a repository list the way the build's two passes will.

    The audit that a shard tree cannot perform on itself: given the repository
    names a build recorded, this re-derives each side from the parameters in the
    manifest and reports the two sets and their intersection. Used by the corpus
    check rather than by the build -- the build's disjointness is guaranteed by
    `repository_split` being a function, and this is how that guarantee is
    *shown* rather than asserted.
    """
    sides: Dict[str, Set[str]] = {name: set() for name in SPLITS}
    for repository in repositories:
        sides[repository_split(repository, holdout_frac=holdout_frac,
                               salt=salt)].add(repository)
    return {
        "train": sorted(sides["train"]),
        "holdout": sorted(sides["holdout"]),
        "overlap": sorted(sides["train"] & sides["holdout"]),
        "holdout_frac": holdout_frac,
        "split_salt": salt,
    }


# ------------------------------------------------------------- the build ---
#
# What turns the measured plan into shards. Everything above answers a question
# about the source; this answers "what exactly does the build run", and it is
# deliberately a pure function of the plan rather than a second reading of it.
# The build is the expensive step -- hours of streaming -- so the arithmetic
# that decides which directory is asked for how many tokens is worth being able
# to print, test and diff before a row is read.
#
# The writer is `dataprep.run_source`, unchanged: the shard format, the resume
# position, the per-source manifest and the memory discipline are the general
# corpus's and there is no version of "the code corpus writes its own shards"
# that is not a second implementation of all four.

#: The row field carrying a file's text in `codeparrot/github-code`.
CODE_TEXT_FIELD = "code"

#: Ceiling on one bucket's holdout, in tokens. A holdout exists to measure code
#: BPB on repositories the model never trained on, and a few million tokens per
#: language measures that as precisely as fifty would. The cost of not capping
#: it is real: the holdout pass streams the whole directory and keeps the 2% of
#: repositories on its side, so every holdout token costs ~50 tokens of
#: streaming. At 3B total the uncapped holdout would be ~39M tokens -- and
#: ~2 billion tokens of streaming to collect them.
DEFAULT_HOLDOUT_CAP_TOKENS = 2_000_000


#: Tokens per admitted code document, measured on the live smoke: 62,354 tokens
#: over 25 `Python-all` documents. Used only to turn a token budget into a
#: document count for the cadences below, never into a corpus size -- a rough
#: constant is fine for "how often should this checkpoint" and would not be for
#: "how much does this hold", which `bucket_supply` measures per directory.
CODE_TOKENS_PER_DOCUMENT = 2_500

#: Durable checkpoints a source should take over its budget.
CHECKPOINTS_PER_SOURCE = 20

#: Never coarser than this, in documents. A holdout pass yields few documents
#: however long it streams, and a floor is what makes it checkpoint at all.
MIN_CHECKPOINT_DOCUMENTS = 200


def checkpoint_every_for(token_budget: int, *,
                         checkpoints: int = CHECKPOINTS_PER_SOURCE,
                         tokens_per_document: int = CODE_TOKENS_PER_DOCUMENT,
                         minimum: int = MIN_CHECKPOINT_DOCUMENTS) -> int:
    """How often one source should write a durable checkpoint, in documents.

    `run_source` counts this cadence in *yielded* documents -- the ones that
    passed the gate -- and a holdout pass yields about 2% of what it streams.
    At the shared 50,000-document default the whole 2M-token Python holdout
    yields ~800 documents and therefore never checkpoints once, so the passes
    with the most streaming to lose are exactly the ones a fixed cadence never
    protects. A crash at 90% of one restarts it at row zero.

    It cannot simply be made small either: every checkpoint calls
    `ShardWriter.flush_partial`, which closes the buffer as a *new shard file*,
    so a 500-document cadence over the 418M-token Python source would leave
    ~840 shards behind. Deriving it from the budget keeps the count bounded at
    both ends -- about `checkpoints` per source, and never coarser than
    `minimum` for a source that yields very few documents.
    """
    if token_budget <= 0 or checkpoints <= 0 or tokens_per_document <= 0:
        return minimum
    documents = token_budget / tokens_per_document
    return max(minimum, int(documents // checkpoints))


def code_text(row: dict) -> str:
    """`SourceSpec.text_fn` for a GitHub code row.

    A named module-level function rather than the lambda `dataprep.MIXTURE`
    uses for its text fields, because this one is rebuilt rather than written
    once: a build that resumes re-derives its specs in a fresh process, and the
    two have to agree about which column the text came from. A name is also what
    makes that answerable from a traceback.
    """
    return row.get(CODE_TEXT_FIELD) or ""


def config_slug(config: str) -> str:
    """A parquet directory name as a shard-key fragment.

    `C-all` and `C++-all` are two directories of one bucket whose names differ
    only in the characters a naive slug throws away -- so a naive slug gives
    both the same key, the second source's shards land in the first's directory
    under the first's prefix, and `_recover_source_stats` reads the pair as one
    interrupted source and resumes it. The corpus would contain C++ files
    manifested as C, and nothing would have failed.
    """
    value = config.replace("+", "p").replace("#", "sharp").lower()
    return "-".join(part for part in re.split(r"[^a-z0-9]+", value) if part)


def source_key(bucket: str, config: str, *, configs: Sequence[str]) -> str:
    """The shard directory one (bucket, directory) pair fills.

    The bucket alone when the bucket has one directory, because the bucket is
    the unit a mixture weight, a BPB holdout and a model card are written in,
    and `code-python-python-all` says the same thing twice. Qualified by the
    directory only when there is more than one to tell apart.
    """
    return (f"code-{bucket}" if len(tuple(configs)) <= 1
            else f"code-{bucket}-{config_slug(config)}")


def code_token_budget(total_tokens: int) -> int:
    """The code portion of a total training budget.

    The other 35% is technical prose and general replay, neither of which any
    code directory supplies -- so this is the number every share below is a
    share *of*, and taking the total by mistake would over-ask every directory
    by half again.
    """
    return int(round(total_tokens * CORPUS_SHARES["code"]))


def _largest_remainder(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Split `total` across `weights` so the parts sum to exactly `total`."""
    positive = {key: value for key, value in weights.items() if value > 0}
    if not positive:
        return {key: 0 for key in weights}
    scale = sum(positive.values())
    exact = {key: total * value / scale for key, value in positive.items()}
    parts = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(parts.values())
    # Ties broken by name, so the same plan produces the same budgets on every
    # machine and a resumed build asks for what the interrupted one did.
    for key in sorted(positive, key=lambda k: (-(exact[k] - parts[k]), k))[:remainder]:
        parts[key] += 1
    return {key: parts.get(key, 0) for key in weights}


def config_budgets(bucket_tokens: int, configs: Sequence[str],
                   per_config_supply: Optional[Dict[str, int]] = None,
                   ) -> Tuple[Dict[str, int], str]:
    """`({config: tokens}, basis)` -- one bucket's budget over its directories.

    Proportional to each directory's measured unique-token supply when that
    measurement is available, which is the only division that does not create a
    shortfall out of nothing: asking two directories for equal halves of a
    bucket only works if they hold equal halves of it.

    Falls back to an even split, and *says so* in the basis, when the supply was
    never measured. That is a real state -- a plan can be built before a
    headroom pass -- and the difference between "divided on a measurement" and
    "divided evenly because there was none" is exactly what a manifest has to
    carry for a later shortfall to be readable.
    """
    configs = list(configs)
    supply = {config: int((per_config_supply or {}).get(config) or 0)
              for config in configs}
    if len(configs) == 1:
        return {configs[0]: int(bucket_tokens)}, "the bucket's only directory"
    if all(value > 0 for value in supply.values()):
        parts = _largest_remainder(int(bucket_tokens), supply)
        measured = ", ".join(f"{config} {supply[config] / 1e6:,.0f}M"
                             for config in configs)
        return parts, f"by measured unique tokens ({measured})"
    parts = _largest_remainder(int(bucket_tokens),
                               {config: 1.0 for config in configs})
    unmeasured = ", ".join(config for config in configs if supply[config] <= 0)
    return parts, (f"evenly across {len(configs)} directories; no measured "
                   f"supply for {unmeasured}")


@dataclass
class CodeSource:
    """One shard directory the build will fill, and everything it takes to.

    The `spec` and the `gate` travel together because they are two halves of one
    decision: the spec says which rows are streamed and the gate says which of
    them are admitted, and a manifest that records one without the other cannot
    say what the shards contain.
    """

    key: str
    bucket: str
    config: str
    split: str
    token_budget: int
    basis: str
    gate: "RepositoryGate"
    spec: object                       # dataprep.SourceSpec, imported lazily

    def out_dir(self, out_root: str) -> str:
        """Where this source's shards live under a build root.

        Split first, then key, so that `<root>/train` and `<root>/holdout` are
        each an ordinary mixture root -- the shape `--data-dir` and
        `--holdout-root` already take, and the shape `daedalus.data`'s
        `resolve_mixture` already reads, with the same key naming both sides of
        one source the way `make_mixture_holdout_split` does.
        """
        import os

        return os.path.join(out_root, self.split, self.key)


def corpus_specs(*, plan: dict, split: str, total_tokens: int,
                 supply: Optional[dict] = None,
                 holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                 salt: str = SPLIT_SALT,
                 holdout_cap_tokens: int = DEFAULT_HOLDOUT_CAP_TOKENS,
                 max_repositories: int = DEFAULT_MAX_REPOSITORIES,
                 dataset: str = GITHUB_CODE_DATASET,
                 revision: str = GITHUB_CODE_REVISION,
                 ) -> List[CodeSource]:
    """The sources one side of the build runs, derived from the measured plan.

    Called once per side. The two calls produce the same keys over the same
    directories with the same language filters and differ only in which side of
    `repository_split` their gates want, which is what makes the holdout a
    second independent stream over the same rows rather than a slice of the
    first: no document can reach both, because no repository can.

    The holdout's budget is not its share of the corpus. It is `holdout_frac` of
    each bucket, capped -- see `DEFAULT_HOLDOUT_CAP_TOKENS`. A holdout sized as
    a share of a 3B-token corpus is 50x more measurement than any BPB gate
    needs, at ~50x its own size in streaming.

    Refuses a plan `plan_problems` rejects. The build is the expensive step and
    every problem that function reports is one that produces shards -- an empty
    directory, a bucket of the wrong language -- rather than an error.
    """
    from daedalus.dataprep import SourceSpec

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    problems = plan_problems(plan)
    if problems:
        raise ValueError("this plan cannot be built as written: "
                         + "; ".join(problems))
    code_tokens = code_token_budget(total_tokens)
    supply = supply or {}

    sources: List[CodeSource] = []
    for bucket, entry in sorted((plan.get("buckets") or {}).items(),
                                key=lambda kv: (-kv[1].get("share", 0.0), kv[0])):
        share = float(entry.get("share") or 0.0)
        if share <= 0:
            # A dropped bucket. `source_plan` already recorded why and at what
            # budget it would come back; building it at 0 tokens would put an
            # empty directory in the corpus that reads as a failed source.
            continue
        source = entry.get("source")
        if source == "directories":
            configs = list(entry.get("configs") or [])
            languages = None
        elif source == "interleaved":
            configs = [entry.get("source_config") or INTERLEAVED_CONFIG]
            languages = frozenset(entry.get("languages") or [])
        else:
            raise ValueError(
                f"{bucket} has share {share} but source {source!r}, which names "
                f"no directory to read it from")
        if not configs:
            raise ValueError(f"{bucket} has share {share} and no directory")

        bucket_tokens = int(round(code_tokens * share))
        if split == "holdout":
            bucket_tokens = min(int(holdout_cap_tokens),
                                max(1, int(round(bucket_tokens * holdout_frac))))
        budgets, basis = config_budgets(
            bucket_tokens, configs,
            (supply.get(bucket) or {}).get("per_config"))
        for config in configs:
            gate = RepositoryGate(want=split, holdout_frac=holdout_frac,
                                  salt=salt, max_repositories=max_repositories,
                                  languages=languages)
            key = source_key(bucket, config, configs=configs)
            sources.append(CodeSource(
                key=key, bucket=bucket, config=config, split=split,
                token_budget=budgets[config], basis=basis, gate=gate,
                spec=SourceSpec(
                    key=key, dataset=dataset, revision=revision,
                    # A fraction of the code budget rather than of the whole
                    # one, recorded for the reader; the build passes
                    # `token_budget` to `run_source` and never multiplies this.
                    share=(budgets[config] / code_tokens) if code_tokens else 0.0,
                    text_fn=code_text, filter_fn=gate,
                    load_kwargs={"data_files": github_code_data_files(config)},
                    # One group per bucket, both sides of the split inside it,
                    # so a file vendored into two repositories of the same
                    # language is caught once -- including across the split,
                    # where a near-duplicate is a leak rather than waste. The
                    # catch is windowed by `DedupState.near_dup_reset_every`,
                    # so it supplements the repository split and does not
                    # replace it.
                    near_dup_group=f"code-{bucket}",
                    note=f"phase 8 code corpus, {bucket} {split} from "
                         f"{config}; {basis}")))
    return sources


#: What two gate manifests must agree on before their counters may be added.
#: Each one changes which rows the gate admits, so a merge across a difference
#: would produce a manifest describing a corpus neither pass built.
GATE_IDENTITY_FIELDS = ("split", "holdout_frac", "split_salt", "languages")


def merge_gate_manifest(prior: Optional[dict], current: dict,
                        max_repositories: int = DEFAULT_MAX_REPOSITORIES,
                        ) -> dict:
    """One gate manifest covering a resumed source's attempts, not just its last.

    A gate lives in the process that streams the source. The shards, the stream
    position and the counters in the shard manifest all survive that process
    dying; the gate's licence histogram and repository list do not. Without this
    the manifest beside a resumed source would describe only the final attempt
    -- reporting, for a source that stopped at 90% and finished on a second run,
    the licences of the last 10% as if they were the corpus's.

    Refuses to merge across a difference in what the gate *was*, because those
    counters are not addable: the same directory under a different `holdout_frac`
    or a different language filter is a different corpus.
    """
    current = dict(current)
    if not prior:
        return {**current, "attempts": int(current.get("attempts") or 1)}
    mismatched = [name for name in GATE_IDENTITY_FIELDS
                  if prior.get(name) != current.get(name)]
    if mismatched:
        raise ValueError(
            "refusing to merge gate manifests that disagree about "
            + ", ".join(f"{name} ({prior.get(name)!r} vs {current.get(name)!r})"
                        for name in mismatched))

    def _sum(name: str) -> dict:
        merged = dict(prior.get(name) or {})
        for key, value in (current.get(name) or {}).items():
            merged[key] = merged.get(key, 0) + value
        return dict(sorted(merged.items()))

    names = sorted(set(prior.get("repository_names") or [])
                   | set(current.get("repository_names") or []))
    truncated = bool(prior.get("repositories_truncated")
                     or current.get("repositories_truncated"))
    if len(names) > max_repositories:
        names = names[:max_repositories]
        truncated = True
    return {
        **current,
        "rows_seen": int(prior.get("rows_seen") or 0) + int(current.get("rows_seen") or 0),
        "rows_admitted": (int(prior.get("rows_admitted") or 0)
                          + int(current.get("rows_admitted") or 0)),
        "refused": _sum("refused"),
        "licenses": _sum("licenses"),
        "repository_fields": _sum("repository_fields"),
        "repository_names": names,
        # The count of *distinct* names, which is not the sum of two counts:
        # a repository met by both attempts is one repository.
        "repositories": len(names) if not truncated
                        else max(int(prior.get("repositories") or 0),
                                 int(current.get("repositories") or 0)),
        "repositories_truncated": truncated,
        "attempts": int(prior.get("attempts") or 1) + int(current.get("attempts") or 1),
    }
