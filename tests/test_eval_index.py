"""Tests for daedalus/eval_index.py.

The frozen index exists to close two silent failures at once -- an index that
covers a fifth of a task, and a corpus that cannot say which index filtered it
-- so the tests are mostly about the ways a *reassuring* index can be produced:
a task that quietly failed to load, a split that quietly differs from the one
we score, a limit that quietly truncates, and a file that quietly decompresses
to fewer n-grams than it was written with.
"""
import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus import eval_index as EI
from daedalus.data import is_contaminated, ngram_set


# ------------------------------------------------------------------- fakes ---

class _Example:
    def __init__(self, candidates):
        self.candidates = candidates


TASKS = ["hellaswag", "arc_easy"]
SPLITS = {"hellaswag": "validation", "arc_easy": "test"}


def _example(word: str, k: int = 20):
    """One example whose candidate text is long enough to yield 13-grams."""
    text = " ".join(f"{word}{i}" for i in range(k))
    return _Example([(text, " tail")])


def _loader(items=None, splits=None, drop=(), fail=()):
    """A stand-in for `eval.load_all_tasks` with its two documented behaviours:
    a task that fails contributes no entry at all, and `sources` is filled only
    for the tasks that loaded."""
    items = items or {name: 3 for name in TASKS}
    splits = {**SPLITS, **(splits or {})}

    def load_all_tasks(limit=None, sources=None):
        out = {}
        for name in TASKS:
            if name in fail:
                continue
            n = items.get(name, 0)
            if limit is not None:
                n = min(n, limit)
            out[name] = [] if name in drop else [
                _example(f"{name}w{i}") for i in range(n)]
            if sources is not None:
                sources[name] = {"repo": f"org/{name}", "config": None,
                                 "split": splits[name], "revision": None,
                                 "n": len(out[name]), "limit": limit}
        return out

    return load_all_tasks


def _build(**kw):
    kw.setdefault("loader", _loader())
    kw.setdefault("expected_tasks", TASKS)
    kw.setdefault("expected_splits", SPLITS)
    kw.setdefault("expected_items", {})      # sized per test, not per benchmark
    kw.setdefault("now", lambda: "2026-08-26T00:00:00Z")
    return EI.build_index(n=13, **kw)


# ------------------------------------------------------------ completeness ---

def test_a_complete_index_covers_every_task_and_records_its_splits():
    ngrams, prov = _build()
    assert prov["complete"] is True and prov["limit"] is None
    assert set(prov["tasks"]) == set(TASKS)
    assert {n: m["split"] for n, m in prov["tasks"].items()} == SPLITS
    assert prov["tasks"]["hellaswag"]["items"] == 3
    assert prov["ngrams"] == len(ngrams) > 0
    assert prov["digest"] == EI.index_digest(ngrams)


def test_a_task_that_failed_to_load_is_refused_not_warned():
    """`load_all_tasks` skips an unavailable benchmark with a warning, which is
    right for scoring and wrong for an index: the resulting corpus would filter
    nothing against that task and say nothing about it."""
    with pytest.raises(EI.IncompleteIndex) as exc:
        _build(loader=_loader(fail=("hellaswag",)))
    assert "hellaswag" in str(exc.value)


def test_a_task_that_loaded_zero_items_is_refused():
    with pytest.raises(EI.IncompleteIndex) as exc:
        _build(loader=_loader(drop=("arc_easy",)))
    assert "arc_easy" in str(exc.value)


def test_indexing_a_split_we_do_not_score_is_refused():
    """The 334c86c failure, stated as a rule instead of reconstructed from a
    build log: sources built before it indexed ARC-Easy `validation` while the
    model is scored on `test`, and nothing recorded the difference."""
    with pytest.raises(EI.IncompleteIndex) as exc:
        _build(loader=_loader(splits={"arc_easy": "validation"}))
    message = str(exc.value)
    assert "arc_easy" in message and "validation" in message and "test" in message


def test_a_limit_is_refused_unless_it_is_asked_for_explicitly():
    with pytest.raises(EI.IncompleteIndex) as exc:
        _build(limit=2000)
    assert "2000" in str(exc.value)

    _, prov = _build(limit=2, allow_partial=True)
    assert prov["complete"] is False and prov["limit"] == 2


def test_a_truncated_task_is_refused_even_though_it_loaded():
    """The failure none of the other guards catch. The Hub is read
    unauthenticated here, and a rate-limited split comes back short rather than
    failing -- so a smaller index would be built, marked complete, digested and
    used, with its own provenance asserting the opposite."""
    with pytest.raises(EI.IncompleteIndex) as exc:
        _build(expected_items={"hellaswag": 10_042})
    message = str(exc.value)
    assert "hellaswag" in message and "10,042" in message and "3" in message


def test_the_expected_counts_describe_the_real_scored_splits():
    """Pinned so a change to a benchmark's size is a decision someone makes
    here rather than a number a rebuild quietly adopts."""
    assert EI.EXPECTED_ITEMS == {"hellaswag": 10_042, "arc_easy": 2_376,
                                 "piqa": 1_838, "openbookqa": 500,
                                 "winogrande": 1_267}
    import eval as E
    assert set(EI.EXPECTED_ITEMS) == set(E.TASK_LOADERS)


def test_a_limit_does_not_trip_the_item_count_guard():
    """A partial index is short in every task by construction; refusing it for
    being short would make `allow_partial` unusable."""
    _, prov = _build(limit=2, allow_partial=True,
                     expected_items={"hellaswag": 10_042})
    assert prov["tasks"]["hellaswag"]["items"] == 2


def test_every_gap_is_reported_at_once():
    """One gap per re-run turns a twenty-minute rebuild into an afternoon."""
    with pytest.raises(EI.IncompleteIndex) as exc:
        _build(loader=_loader(fail=("hellaswag",),
                              splits={"arc_easy": "validation"}))
    assert len(exc.value.problems) == 2


def test_a_task_scored_but_not_expected_here_is_refused():
    """A task added to TASK_LOADERS without reaching this module would be
    scored and never filtered -- the split gap wearing a hat."""
    with pytest.raises(EI.IncompleteIndex) as exc:
        _build(expected_tasks=["hellaswag"],
               expected_splits={"hellaswag": "validation"})
    assert "arc_easy" in str(exc.value)


# ------------------------------------------------------------------ digest ---

def test_the_digest_names_the_set_and_not_the_order_it_was_built_in():
    grams = ["b b b", "a a a", "c c c"]
    assert EI.index_digest(grams) == EI.index_digest(reversed(grams))
    assert EI.index_digest(grams) != EI.index_digest(grams[:2])


def test_ngrams_never_contain_a_newline():
    """What makes the line-delimited on-disk format lossless. `ngram_set`
    splits on whitespace and rejoins with single spaces, so a document full of
    newlines still yields newline-free n-grams."""
    text = "\n".join(f"word{i}" for i in range(40)) + "\r\nlast\ttab"
    assert all("\n" not in g and "\r" not in g for g in ngram_set(text, 13))


# ----------------------------------------------------------- round tripping ---

def test_write_then_load_round_trips_the_index_and_its_provenance(tmp_path):
    ngrams, prov = _build()
    path = str(tmp_path / "idx.txt.gz")
    EI.write_index(path, ngrams, prov)

    loaded, loaded_prov = EI.load_index(path)
    assert loaded == ngrams
    assert loaded_prov["digest"] == prov["digest"]
    assert loaded_prov["tasks"] == prov["tasks"]
    assert os.path.exists(EI.sidecar_path(path))


def test_the_file_on_disk_is_a_function_of_the_set_alone(tmp_path):
    """Two writes of the same index must be byte-identical, or the digest in a
    manifest identifies the run rather than the filter."""
    ngrams, prov = _build()
    a, b = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    EI.write_index(a, ngrams, prov)
    EI.write_index(b, sorted(ngrams, reverse=True), prov)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()


def test_loading_a_truncated_index_refuses_instead_of_filtering_less(tmp_path):
    """A short write -- a full disk during the rebuild -- decompresses to fewer
    n-grams, filters less, and reports nothing. The digest is what turns that
    into an error."""
    ngrams, prov = _build()
    path = str(tmp_path / "idx.txt.gz")
    EI.write_index(path, ngrams, prov)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(sorted(ngrams)[:5]) + "\n")
    with pytest.raises(EI.IndexDigestMismatch):
        EI.load_index(path)


def test_loading_refuses_an_index_that_is_not_the_one_a_manifest_names(tmp_path):
    ngrams, prov = _build()
    path = str(tmp_path / "idx.txt.gz")
    EI.write_index(path, ngrams, prov)
    with pytest.raises(EI.IndexDigestMismatch):
        EI.load_index(path, expect_digest="sha256:" + "0" * 64)
    assert EI.load_index(path, expect_digest=prov["digest"])[1]["ngrams"]


def test_writing_a_partial_index_needs_saying_so_twice(tmp_path):
    ngrams, prov = _build(limit=2, allow_partial=True)
    path = str(tmp_path / "partial.gz")
    with pytest.raises(EI.IncompleteIndex):
        EI.write_index(path, ngrams, prov)
    EI.write_index(path, ngrams, prov, allow_partial=True)
    assert EI.read_provenance(path)["complete"] is False


def test_write_refuses_a_provenance_that_describes_different_ngrams(tmp_path):
    ngrams, prov = _build()
    with pytest.raises(EI.IndexDigestMismatch):
        EI.write_index(str(tmp_path / "x.gz"), sorted(ngrams)[:3], prov)


def test_a_written_index_still_matches_the_documents_it_indexed(tmp_path):
    """The predicate has to survive the round trip, not just the set: the
    corpus filter is `is_contaminated`, and an index that loads but no longer
    matches would be the quietest failure of all."""
    ngrams, prov = _build()
    path = str(tmp_path / "idx.txt.gz")
    EI.write_index(path, ngrams, prov)
    loaded, _ = EI.load_index(path)

    contaminated = _example("hellaswagw0").candidates[0]
    assert is_contaminated(contaminated[0] + " " + contaminated[1], loaded)
    assert not is_contaminated(" ".join(f"clean{i}" for i in range(30)), loaded)


# ---------------------------------------------------------------- coverage ---

def test_coverage_problems_reads_a_frozen_index_against_todays_tasks():
    _, prov = _build()
    assert EI.coverage_problems(prov, TASKS, SPLITS, {}) == []

    grown = TASKS + ["winogrande"]
    problems = EI.coverage_problems(prov, grown,
                                    {**SPLITS, "winogrande": "validation"}, {})
    assert any("winogrande" in p for p in problems)

    moved = EI.coverage_problems(prov, TASKS,
                                 {**SPLITS, "arc_easy": "validation"}, {})
    assert any("arc_easy" in p for p in moved)


def test_coverage_problems_flags_an_index_a_split_has_since_outgrown():
    """A frozen index goes stale silently: the corpus it filtered is unchanged,
    but the benchmark it was built from is not the one being scored."""
    _, prov = _build()
    problems = EI.coverage_problems(prov, TASKS, SPLITS, {"hellaswag": 10_042})
    assert any("hellaswag" in p and "10,042" in p for p in problems)


def test_coverage_problems_flags_a_partial_index():
    _, prov = _build(limit=2, allow_partial=True)
    assert any("partial" in p
               for p in EI.coverage_problems(prov, TASKS, SPLITS, {}))


def test_manifest_record_carries_the_identity_and_the_coverage():
    _, prov = _build()
    record = EI.manifest_record(prov, path="data/decontam/eval-index-13gram.txt.gz")
    assert record["digest"] == prov["digest"]
    assert record["complete"] is True
    assert record["items"] == {"arc_easy": 3, "hellaswag": 3}
    assert record["splits"] == SPLITS
    assert json.dumps(record)      # must survive a manifest write
