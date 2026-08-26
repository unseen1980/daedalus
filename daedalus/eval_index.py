"""The frozen decontamination index: every scored item, on every scored split.

Why this exists
---------------
The released corpus was filtered with `_build_eval_index(limit=2000)`, and that
one default opened two separate holes.

**Coverage.** HellaSwag has 10,042 items and 2,000 of them were indexed, so
19.9% of the task we report a score on was never filtered against; ARC-Easy
84.2%. `scripts/contam_scan.py` measured what came through that gap. This
module is the other half: the index that does not have it.

**Provenance, which is the worse of the two.** Nothing recorded *which* index a
source was filtered against. `334c86c` (2026-08-09 18:09Z) moved ARC-Easy and
OpenBookQA onto their `test` splits to match lm-evaluation-harness, so sources
built before it were filtered against `validation` -- the splits we do *not*
score -- and the corpus does not say which side of that line any source falls
on. Establishing it afterwards took rebuilding the index at the old splits and
matching the 183,359-gram count in a build log. That is archaeology, and it
only worked because a count happened to have been logged.

Building the index live at the start of every build is what makes that
unavoidable: the index is a function of whatever `datasets` returned that day
and of whatever `TASK_SPLITS` said that week, and neither is written down. So
this module builds it **once**, writes it to disk beside the item counts,
splits and revisions it was built from, and content-addresses it.
`run_dataprep` loads a frozen index by path and records its digest in the
manifest, so "which eval items was this source filtered against" is answered by
the artifacts rather than by git.

What "complete" means, and why it is refused rather than warned
---------------------------------------------------------------
`eval.load_all_tasks` skips a task it cannot load, with a warning, because one
briefly-unavailable benchmark must not crash a scoring run. For an index that
bias points exactly the wrong way: a HellaSwag outage would yield an index that
looks fine, filters nothing against HellaSwag, and leaves no trace in the
corpus -- the same silent failure as the split gap, arriving by a different
road. So `build_index` *refuses* an index that is missing a task, that loaded a
task at zero items, that resolved a split other than the one that task is
scored on, or that was truncated by a limit. The refusals are the deliverable
here; the union of n-grams is the easy part.

`allow_partial=True` still permits a deliberately limited index -- reproducing
what the build actually used is how the exposure was measured -- but it is
marked `complete: false` in its own provenance and `write_index` will not write
it to the default path without being told again.

On-disk format
--------------
gzip'd text, one n-gram per line, sorted, plus a JSON sidecar. n-grams cannot
contain a newline (`ngram_set` splits on whitespace and rejoins with single
spaces, pinned by `test_ngrams_never_contain_a_newline`), so the format is
lossless. Sorting makes the bytes a function of the set alone rather than of
the run that wrote it, which is what lets the digest name the index: two builds
of the same items produce the same digest, and a manifest quoting that digest
identifies the filter exactly.

`load_index` recomputes the digest and refuses a file that disagrees with its
sidecar. A truncated write -- a full disk during the rebuild -- would otherwise
load as a smaller index that filters less and reports no error at all, which is
the same class of reassuring-for-the-wrong-reason failure this file exists to
close.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

SCHEMA = 1
DEFAULT_N = 13
DEFAULT_INDEX_PATH = "data/decontam/eval-index-13gram.txt.gz"


class IncompleteIndex(RuntimeError):
    """Raised when an index would not cover every scored item and split.

    Carries `problems` so a caller reports every gap at once rather than one
    per re-run; finding out about ARC-Easy only after fixing HellaSwag is how a
    twenty-minute rebuild becomes five.
    """

    def __init__(self, problems: List[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class IndexDigestMismatch(RuntimeError):
    """Raised when an index file's contents disagree with its sidecar digest."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sidecar_path(path: str) -> str:
    """The provenance file beside an index. `<path>` -> `<path>.json`.

    Appended rather than substituted for the extension so the pair sorts
    together and neither name can collide with the other's.
    """
    return path + ".json"


def index_digest(ngrams: Iterable[str]) -> str:
    """`sha256:<hex>` over the sorted n-grams, newline-terminated.

    Sorted, so the digest identifies the *set* and not the order a particular
    build happened to emit it in -- a digest that changed run to run could not
    be quoted in a manifest as the identity of the filter, which is the whole
    reason it is written down. Hashed incrementally because the complete index
    is millions of n-grams and materialising them as one string to hash would
    cost several hundred megabytes for no benefit.
    """
    digest = hashlib.sha256()
    for gram in sorted(ngrams):
        digest.update(gram.encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _default_loader():
    from eval import load_all_tasks  # repo-root module, see eval.py

    return load_all_tasks


def _default_tasks() -> Tuple[List[str], Dict[str, str]]:
    import eval as E

    return list(E.TASK_LOADERS), dict(E.TASK_SPLITS)


def build_index(n: int = DEFAULT_N, limit: Optional[int] = None,
                loader: Optional[Callable] = None,
                expected_tasks: Optional[List[str]] = None,
                expected_splits: Optional[Dict[str, str]] = None,
                allow_partial: bool = False,
                now: Callable[[], str] = _utcnow,
                ) -> Tuple[Set[str], dict]:
    """`(ngrams, provenance)` for every scored item of every scored split.

    Raises `IncompleteIndex` rather than returning a gap. See the module
    docstring for why a warning is the wrong instrument here.
    """
    from daedalus.data import build_eval_ngram_index

    loader = loader or _default_loader()
    if expected_tasks is None or expected_splits is None:
        tasks, splits = _default_tasks()
        expected_tasks = expected_tasks if expected_tasks is not None else tasks
        expected_splits = expected_splits if expected_splits is not None else splits

    sources: Dict[str, dict] = {}
    loaded = loader(limit=limit, sources=sources)

    problems: List[str] = []
    if limit is not None and not allow_partial:
        problems.append(
            f"limit={limit} truncates every task; a complete index is built "
            f"with limit=None (pass allow_partial=True for a deliberate "
            f"reconstruction of a historical index)")

    tasks_meta: Dict[str, dict] = {}
    texts: List[str] = []
    for name in expected_tasks:
        examples = loaded.get(name)
        if not examples:
            # `load_all_tasks` returns no entry for a task that failed to load
            # and an empty list for one that loaded nothing; neither may pass.
            problems.append(f"task {name!r} contributed no items")
            continue
        resolved = dict(sources.get(name) or {})
        want = expected_splits.get(name)
        got = resolved.get("split")
        if want is not None and got is not None and got != want:
            problems.append(
                f"task {name!r} was indexed on split {got!r} but is scored on "
                f"{want!r}")
        candidates = [ctx + " " + cont for ex in examples
                      for ctx, cont in ex.candidates]
        texts += candidates
        tasks_meta[name] = {
            "items": len(examples),
            "candidates": len(candidates),
            "split": got if got is not None else want,
            "repo": resolved.get("repo"),
            "config": resolved.get("config"),
            "revision": resolved.get("revision"),
        }

    unexpected = sorted(set(loaded) - set(expected_tasks))
    if unexpected:
        # A task added to `TASK_LOADERS` without being added here would be
        # scored and never filtered, which is the split gap wearing a hat.
        problems.append(f"loaded task(s) not in the expected set: {unexpected}")

    if problems:
        raise IncompleteIndex(problems)

    ngrams = build_eval_ngram_index(texts, n=n)
    provenance = {
        "schema": SCHEMA,
        "n": n,
        "limit": limit,
        "complete": limit is None,
        "ngrams": len(ngrams),
        "digest": index_digest(ngrams),
        "built_at": now(),
        "tasks": tasks_meta,
    }
    return ngrams, provenance


def coverage_problems(provenance: dict,
                      expected_tasks: Optional[List[str]] = None,
                      expected_splits: Optional[Dict[str, str]] = None,
                      ) -> List[str]:
    """Everything wrong with an index *as described by its own sidecar*.

    Separate from `build_index`'s checks on purpose: those run against live
    task loads at build time, this runs against a file months later, when the
    question is whether the index a manifest names still covers what the model
    is scored on today. A task added to `TASK_LOADERS` after the index was
    frozen is invisible to the build that used it and visible here.
    """
    if expected_tasks is None or expected_splits is None:
        tasks, splits = _default_tasks()
        expected_tasks = expected_tasks if expected_tasks is not None else tasks
        expected_splits = expected_splits if expected_splits is not None else splits

    problems: List[str] = []
    if provenance.get("schema") != SCHEMA:
        problems.append(f"unknown schema {provenance.get('schema')!r}")
    if not provenance.get("complete"):
        problems.append(f"index is partial (limit={provenance.get('limit')!r})")
    tasks_meta = provenance.get("tasks") or {}
    for name in expected_tasks:
        meta = tasks_meta.get(name)
        if not meta:
            problems.append(f"task {name!r} is not in the index")
            continue
        if not meta.get("items"):
            problems.append(f"task {name!r} contributed no items")
        want = expected_splits.get(name)
        got = meta.get("split")
        if want is not None and got != want:
            problems.append(
                f"task {name!r} was indexed on split {got!r} but is scored on "
                f"{want!r}")
    return problems


def write_index(path: str, ngrams: Iterable[str], provenance: dict,
                allow_partial: bool = False) -> str:
    """Write the index and its sidecar atomically; return the index path.

    Both files go down via tmp+rename. A half-written index is the failure that
    matters most here: it decompresses to fewer n-grams, filters less, and the
    build that used it says nothing -- so the window in which a reader can
    observe a partial file has to be closed rather than documented.
    """
    if not provenance.get("complete") and not allow_partial:
        raise IncompleteIndex([
            f"refusing to write a partial index (limit="
            f"{provenance.get('limit')!r}) without allow_partial"])

    grams = sorted(ngrams)
    digest = index_digest(grams)
    if provenance.get("digest") not in (None, digest):
        raise IndexDigestMismatch(
            f"provenance digest {provenance['digest']} does not describe the "
            f"n-grams being written ({digest})")
    provenance = {**provenance, "digest": digest, "ngrams": len(grams)}

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # `mtime=0` and `filename=""` between them keep the run out of the bytes.
    # gzip stamps both the wall clock and the source filename into its header
    # by default, and the filename here is the *temporary* one -- so the same
    # index written to two paths produced two different files, and the digest
    # would have identified the write rather than the filter. Caught by
    # `test_the_file_on_disk_is_a_function_of_the_set_alone`.
    with open(tmp, "wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as raw:
            for gram in grams:
                raw.write(gram.encode("utf-8"))
                raw.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

    side = sidecar_path(path)
    side_tmp = side + ".tmp"
    with open(side_tmp, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(side_tmp, side)
    return path


def read_provenance(path: str) -> dict:
    """The sidecar for an index path."""
    with open(sidecar_path(path)) as f:
        return json.load(f)


def load_index(path: str, expect_digest: Optional[str] = None,
               ) -> Tuple[Set[str], dict]:
    """`(ngrams, provenance)`, refusing a file that is not what it claims.

    `expect_digest` is what a manifest carries: passing it turns "filtered
    against some index" into "filtered against *this* index", which is the
    question the released corpus could not answer.
    """
    provenance = read_provenance(path)
    ngrams: Set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            gram = line.rstrip("\n")
            if gram:
                ngrams.add(gram)

    digest = index_digest(ngrams)
    recorded = provenance.get("digest")
    if recorded and digest != recorded:
        raise IndexDigestMismatch(
            f"{path} hashes to {digest} but its sidecar records {recorded}")
    if expect_digest and digest != expect_digest:
        raise IndexDigestMismatch(
            f"{path} hashes to {digest}, not the expected {expect_digest}")
    return ngrams, provenance


def manifest_record(provenance: dict, path: Optional[str] = None) -> dict:
    """The compact form a corpus manifest carries for the index it used.

    Deliberately not the whole sidecar: the manifest needs enough to identify
    the filter and to show its coverage without a second file, and the digest
    makes the full sidecar recoverable anyway.
    """
    tasks = provenance.get("tasks") or {}
    return {
        "digest": provenance.get("digest"),
        "path": path,
        "n": provenance.get("n"),
        "ngrams": provenance.get("ngrams"),
        "complete": bool(provenance.get("complete")),
        "built_at": provenance.get("built_at"),
        "items": {name: meta.get("items") for name, meta in sorted(tasks.items())},
        "splits": {name: meta.get("split") for name, meta in sorted(tasks.items())},
    }
