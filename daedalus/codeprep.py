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
REFUSAL_REASONS = ("no_repository", "non_permissive", "unknown_license",
                   "other_split")


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
    """

    want: str
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC
    salt: str = SPLIT_SALT
    max_repositories: int = DEFAULT_MAX_REPOSITORIES
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

    def __call__(self, row: dict) -> bool:
        self.seen += 1
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
                 salt: str = SPLIT_SALT) -> dict:
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
    """
    record: dict = {"dataset": GITHUB_CODE_DATASET, "config": config,
                    "data_files": github_code_data_files(config),
                    "rows_requested": rows}
    gate = RepositoryGate(want="train", holdout_frac=holdout_frac, salt=salt,
                          max_repositories=rows)
    holdout = RepositoryGate(want="holdout", holdout_frac=holdout_frac, salt=salt,
                             max_repositories=rows)
    columns: Dict[str, int] = {}
    samples: List[dict] = []
    try:
        for index, row in enumerate(stream(config) if stream is not None
                                    else _stream_github_code(config)):
            if index >= rows:
                break
            for key in row:
                columns[key] = columns.get(key, 0) + 1
            train_ok, holdout_ok = gate(row), holdout(row)
            if (train_ok or holdout_ok) and len(samples) < 3:
                samples.append({"repository": repository_of(row),
                                "license": normalize_license(row.get("license")),
                                "split": "train" if train_ok else "holdout",
                                "chars": len(row.get("code") or "")})
    except Exception as exc:                    # noqa: BLE001 - reported per config
        record.update({"resolved": False, "error": repr(exc)})
        return record

    train_manifest = gate.manifest()
    record.update({
        "resolved": gate.seen > 0,
        "rows_read": gate.seen,
        "columns": dict(sorted(columns.items())),
        "repository_fields": train_manifest["repository_fields"],
        "licenses": train_manifest["licenses"],
        "admitted": {"train": gate.admitted, "holdout": holdout.admitted},
        "repositories": {"train": gate.repositories_count,
                         "holdout": holdout.repositories_count},
        # The two gates refuse each other's rows under `other_split`, so the
        # licence and repository refusals are the ones that describe the source.
        "refused": {reason: train_manifest["refused"][reason]
                    for reason in REFUSAL_REASONS if reason != "other_split"},
        "samples": samples,
    })
    if gate.seen and not (gate.admitted or holdout.admitted):
        # The silent failure, named. Every one of these has the same shape --
        # rows arrived and none survived -- and none of them raises.
        record["problem"] = (
            "read {:,} rows and admitted none: the repository field, the "
            "licence vocabulary or the split is not what this module assumes"
            .format(gate.seen))
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
                    stream: Optional[Callable[[str], object]] = None,
                    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                    salt: str = SPLIT_SALT) -> dict:
    """`probe_source` over every config of every requested bucket."""
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
