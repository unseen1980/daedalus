"""Tests for scripts/contam_scan.py.

The scan exists to turn `STATUS.md`'s decontamination caveat into a number that
can be quoted next to the eval scores, so the tests are mostly about the ways a
scan can report a *reassuring* number for the wrong reason: sampling that
silently reads nothing, document splitting that mislabels a partial document, a
"disjoint" pair of indices that is not, and a zero-hit result that comes from
scanning zero documents rather than from a clean corpus.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import contam_scan as CS
from daedalus.data import is_contaminated, ngram_set


# ------------------------------------------------------------------ offsets ---

def test_window_offsets_are_token_uniform_and_inside_the_source():
    offs = CS.window_offsets(1_000_000, 1_000, 10)
    assert len(offs) == 10
    assert offs[0] == 0 and offs[-1] == 1_000_000 - 1_000
    gaps = {b - a for a, b in zip(offs, offs[1:])}
    assert max(gaps) - min(gaps) <= 1        # evenly spaced up to rounding
    assert all(0 <= o <= 1_000_000 - 1_000 for o in offs)


def test_window_offsets_never_oversample_a_small_source():
    """A source too small for `k` disjoint windows must yield fewer windows,
    not the same `k` overlapping ones -- otherwise a 0.4M-token source
    contributes as many sampled tokens as a 5B one and the corpus-weighted rate
    is quietly dominated by the smallest source in the mixture."""
    assert len(CS.window_offsets(10_000, 1_000, 120)) == 10
    assert CS.window_offsets(500, 1_000, 5) == [0]
    assert CS.window_offsets(0, 1_000, 5) == []


# ------------------------------------------------------------------- locate ---

def test_locate_maps_a_global_offset_into_the_right_shard():
    shards = [{"file": "a.bin", "tokens": 100}, {"file": "b.bin", "tokens": 50}]
    assert CS.locate(shards, 0) == (0, 0, 100)
    assert CS.locate(shards, 99) == (0, 99, 1)
    assert CS.locate(shards, 100) == (1, 0, 50)
    assert CS.locate(shards, 149) == (1, 49, 1)
    assert CS.locate(shards, 150) is None


def test_locate_reports_the_real_remaining_tokens_of_a_short_final_shard():
    """The last shard of a source is short. A caller that assumed every shard
    holds `shard_tokens` would read past the end -- memmap returns a truncated
    slice rather than raising, so the accounting would claim tokens that were
    never scanned."""
    shards = [{"file": "a.bin", "tokens": 100_000_000},
              {"file": "b.bin", "tokens": 5_019}]
    idx, _, left = CS.locate(shards, 100_000_000 + 19)
    assert (idx, left) == (1, 5_000)


# --------------------------------------------------------------- doc splits ---

def test_read_window_crosses_shard_boundaries_and_returns_a_full_window(tmp_path):
    """The regression guard for the first version of this scan: it read only to
    the end of the containing shard, so a window landing one token before a
    boundary returned one token. The scan then reported zero hits over almost
    no data, which is indistinguishable in the output from a clean corpus."""
    words = [f"w{i}" for i in range(40)]
    vocab = {w: i + 1 for i, w in enumerate(words)}
    d = _write_source(tmp_path, [words[:20] for _ in range(30)], vocab,
                      shard_tokens=63)
    shards = json.load(open(d / "manifest.json"))["shards"]
    assert len(shards) > 5

    win = CS.read_window(str(d), shards, 62, 100)     # 1 token before a boundary
    assert win.size == 100
    tail = CS.read_window(str(d), shards, 630 - 10, 100)   # runs off the end
    assert tail.size == 10


def test_split_documents_returns_only_whole_documents():
    eos = 0
    toks = [7, 7, eos, 1, 2, 3, eos, 4, 5, eos, 9, 9]
    docs, excluded, partial = CS.split_documents(toks, eos)
    assert docs == [[1, 2, 3], [4, 5]]
    assert excluded == 2 + 2          # the [7,7] head and the [9,9] tail
    assert partial == 2
    # Every token is accounted for: whole-document content, the eos separators,
    # and the two edge fragments. An `excluded` that quietly swallowed the eos
    # tokens would overstate the mass this scan cannot classify.
    assert sum(len(d) for d in docs) + toks.count(eos) + excluded == len(toks)


def test_split_documents_excludes_everything_when_no_document_is_whole():
    """A window landing inside one very long document yields no classifiable
    document. It must contribute zero *docs* and its tokens must show up as
    excluded, not be silently counted as a clean document -- that is exactly
    how a scan reports 0% by measuring nothing."""
    docs, excluded, partial = CS.split_documents([5] * 100, eos_id=0)
    assert docs == []
    assert excluded == 100
    assert partial == 1


def test_split_documents_drops_empty_documents_between_adjacent_eos():
    docs, _, _ = CS.split_documents([0, 0, 1, 2, 0], eos_id=0)
    assert docs == [[1, 2]]


# ------------------------------------------------------------- classification --

def test_classify_document_agrees_with_is_contaminated():
    """The scan uses one `ngram_set` and two `isdisjoint` calls for speed. That
    is only legitimate if it decides exactly what the pipeline's own predicate
    decides, so pin it against `is_contaminated` directly."""
    a = ngram_set("the quick brown fox jumps over the lazy dog and then "
                  "some more words here", 13)
    b = ngram_set("an entirely different sentence with enough words in "
                  "it to make one gram", 13)
    for text in ["the quick brown fox jumps over the lazy dog and then some "
                 "more words here",
                 "an entirely different sentence with enough words in it to "
                 "make one gram",
                 "nothing at all like either of the two indexed strings above "
                 "not even close"]:
        hit = CS.classify_document(text, {"a": a, "b": b})
        assert hit["a"] == is_contaminated(text, a)
        assert hit["b"] == is_contaminated(text, b)


def test_a_short_document_can_never_hit_either_index():
    """Fewer than n words means no n-grams. Worth pinning because a predicate
    that returned True on the empty set would mark every short document
    contaminated and inflate the exposure rate."""
    hit = CS.classify_document("too short", {"a": set(), "b": set()})
    assert hit == {"a": False, "b": False}


# ------------------------------------------------------------------- wilson ---

def test_wilson_upper_bounds_zero_hits_away_from_certainty():
    """0/1000 is not "0%". The whole point of reporting an upper bound is that
    the interesting result is an absence of hits, where the normal
    approximation collapses to [0, 0] and claims certainty."""
    assert CS.wilson_upper(0, 1000) > 0.0
    assert CS.wilson_upper(0, 1000) < 0.01
    assert CS.wilson_upper(0, 10) > CS.wilson_upper(0, 1000)
    assert CS.wilson_upper(0, 0) == 1.0


def test_wilson_upper_is_above_the_point_estimate():
    for hits, n in [(1, 100), (50, 100), (99, 100)]:
        assert CS.wilson_upper(hits, n) >= hits / n


# ------------------------------------------------------- corpus-level rates ---

def _src(name, source_tokens, doc_tokens, tok_unc, docs=10, docs_unc=1):
    d = {"source": name, "source_tokens": source_tokens, "docs": docs,
         "doc_tokens": doc_tokens, "tokens_scanned": doc_tokens,
         "tokens_excluded": 0, "stopped_early": None}
    for n in CS.INDEX_NAMES:
        d[f"docs_{n}"] = 0
        d[f"tokens_{n}"] = 0
    d["docs_limit_gap"], d["tokens_limit_gap"] = docs_unc, tok_unc
    return d


def test_corpus_rate_is_weighted_by_real_source_size_not_by_sample_size():
    """Both sources contribute the same sampled tokens, but one is 1000x the
    other in the corpus. A plain pooled rate would report ~50%; the quantity
    the reader wants is the share of the *training stream*, which is ~0.1%."""
    per_source = [_src("big", 1_000_000_000, 10_000, 0, docs_unc=0),
                  _src("tiny", 1_000_000, 10_000, 10_000, docs_unc=10)]
    totals = CS.corpus_rates(per_source)
    assert totals["token_rate_limit_gap"] == pytest.approx(
        1_000_000 / 1_001_000_000, rel=1e-6)
    assert totals["token_rate_limit_gap"] < 0.002


def test_corpus_rates_survive_a_source_that_yielded_no_documents():
    """A source whose windows all landed mid-document contributes no
    denominator. Dividing by it would raise and lose the whole scan."""
    empty = _src("empty", 1_000, 0, 0, docs=0, docs_unc=0)
    empty["tokens_scanned"], empty["tokens_excluded"] = 100, 100
    totals = CS.corpus_rates([_src("ok", 1_000, 500, 0, docs=5, docs_unc=0), empty])
    assert totals["docs"] == 5
    assert totals["token_rate_limit_gap"] == 0.0


# --------------------------------------------------------- end-to-end scan ----

class _FakeTokenizer:
    """Decodes ids back to the words they were built from. The real tokenizer
    is a 200 MB download and this test is about the scan's bookkeeping, not
    about SmolLM2's vocabulary."""

    def __init__(self, vocab):
        self.itos = {i: w for w, i in vocab.items()}

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)


def _write_source(tmp_path, docs, vocab, eos_id=0, shard_tokens=64):
    """Write `docs` (lists of words) as a real multi-shard source directory."""
    ids = []
    for d in docs:
        ids.extend(vocab[w] for w in d)
        ids.append(eos_id)
    d = tmp_path / "src"
    d.mkdir()
    shards, i = [], 0
    while i < len(ids):
        chunk = ids[i:i + shard_tokens]
        name = f"src_{len(shards):05d}.bin"
        np.asarray(chunk, dtype=np.uint16).tofile(d / name)
        shards.append({"file": name, "tokens": len(chunk)})
        i += shard_tokens
    with open(d / "manifest.json", "w") as f:
        json.dump({"total_tokens": len(ids), "eos_id": eos_id,
                   "shards": shards}, f)
    return d


def test_scan_source_finds_a_planted_document_and_leaves_clean_ones_alone(tmp_path):
    words = [f"w{i}" for i in range(40)]
    marker = [f"m{i}" for i in range(20)]
    vocab = {w: i + 1 for i, w in enumerate(words + marker)}   # 0 is eos

    clean = [words[:30] for _ in range(6)]
    planted = marker[:20]
    docs = clean[:3] + [planted] + clean[3:]
    d = _write_source(tmp_path, docs, vocab, shard_tokens=4096)

    idx = {"filtered": set(), "split_gap": set(),
           "limit_gap": ngram_set(" ".join(marker[:20]), 13)}
    res = CS.scan_source(str(d), _FakeTokenizer(vocab), idx, window=4096, k=1)
    # The first document has no eos in front of it, so it is a head fragment
    # and is excluded -- the same rule that protects against classifying a
    # document the window only partly contains.
    assert res["docs"] == len(docs) - 1
    assert res["docs_limit_gap"] == 1
    assert res["docs_filtered"] == 0 and res["docs_split_gap"] == 0
    assert res["tokens_limit_gap"] == len(planted)


def test_scan_source_reads_across_several_shards(tmp_path):
    """Systematic offsets have to resolve into whichever shard file holds them.
    An off-by-one in `locate` would keep re-reading shard 0 and report a rate
    for the first 100M tokens of every source."""
    words = [f"w{i}" for i in range(40)]
    marker = [f"m{i}" for i in range(20)]
    vocab = {w: i + 1 for i, w in enumerate(words + marker)}
    docs = [words[:20] for _ in range(30)]
    docs[-2] = marker[:20]                     # near the end -> a late shard
    d = _write_source(tmp_path, docs, vocab, shard_tokens=63)
    assert len(json.load(open(d / "manifest.json"))["shards"]) > 5

    idx = {"filtered": set(), "split_gap": set(),
           "limit_gap": ngram_set(" ".join(marker[:20]), 13)}
    res = CS.scan_source(str(d), _FakeTokenizer(vocab), idx, window=64, k=200)
    assert res["docs_limit_gap"] == 1, "the planted late document was never read"


def test_scan_source_counts_excluded_edge_tokens(tmp_path):
    words = [f"w{i}" for i in range(40)]
    vocab = {w: i + 1 for i, w in enumerate(words)}
    d = _write_source(tmp_path, [words[:20] for _ in range(20)], vocab)
    res = CS.scan_source(str(d), _FakeTokenizer(vocab), {"filtered": set()},
                         window=64, k=3)
    assert res["tokens_excluded"] > 0
    assert res["tokens_excluded"] < res["tokens_scanned"]
    assert res["doc_tokens"] + res["tokens_excluded"] <= res["tokens_scanned"]


def test_scan_source_stops_cleanly_when_memory_runs_low(tmp_path, monkeypatch):
    """ADDENDUM 2 rule 4: a clean stop with a reason beats a wedge. The result
    must still be well-formed so the report can say the scan was truncated
    rather than silently reporting a rate over three windows."""
    words = [f"w{i}" for i in range(40)]
    vocab = {w: i + 1 for i, w in enumerate(words)}
    d = _write_source(tmp_path, [words[:20] for _ in range(60)], vocab)
    monkeypatch.setattr(CS, "_available_gb", lambda: 0.5)
    res = CS.scan_source(str(d), _FakeTokenizer(vocab), {"filtered": set()},
                         window=64, k=50, min_available_gb=6.0)
    assert res["stopped_early"]
    assert res["windows"] == 0


# ------------------------------------------------------------------ indices ---

def test_the_two_indices_are_disjoint_by_construction():
    """A window hitting both indices must mean two different eval items. If
    `uncovered` still contained the covered n-grams, the control row would
    inherit every uncovered hit and always look non-zero -- destroying the one
    check that makes this scan falsifiable."""
    a = ngram_set("one two three four five six seven eight nine ten eleven "
                  "twelve thirteen fourteen", 13)
    b = ngram_set("one two three four five six seven eight nine ten eleven "
                  "twelve thirteen fourteen fifteen", 13)
    assert a & (b - a) == set()
    assert (b - a)


def test_report_names_the_control_row_and_the_excluded_mass(tmp_path):
    """The report is read from a phone by someone deciding whether to trust an
    eval number. Both caveats have to survive into the rendered markdown."""
    per_source = [_src("s", 1_000_000, 10_000, 0, docs=100, docs_unc=0)]
    totals = CS.corpus_rates(per_source)
    md = CS.format_report(per_source, totals,
                          {"hellaswag": {"indexed": 2000, "total": 10042,
                                         "build_split": None}},
                          {"filtered": 1, "split_gap": 2, "limit_gap": 3},
                          {"window": 32768, "k": 120, "n": 13})
    assert "negative control" in md
    assert "19.9%" in md
    assert "excluded" in md
    assert "upper bound" in md.lower()


# ------------------------------------------------- the three-index decomposition --

class _Ex:
    def __init__(self, candidates):
        self.candidates = candidates


def _fake_loaders(monkeypatch, per_split):
    """Install one task whose text differs per split, so the decomposition can
    be tested without downloading five benchmarks."""
    import eval as E

    def loader(split=None, limit=None):
        items = per_split[split]
        return [_Ex([(t, "")]) for t in (items[:limit] if limit else items)]

    monkeypatch.setattr(E, "TASK_LOADERS", {"t": loader})
    monkeypatch.setattr(E, "TASK_SPLITS", {"t": "test"})


def test_build_indices_returns_three_pairwise_disjoint_sets(monkeypatch):
    """Disjointness is what lets the report add the rows up and what makes
    `filtered` a control: if `split_gap` still contained the filtered n-grams,
    every `filtered` hit would also be a `split_gap` hit and neither number
    would mean anything on its own."""
    words = lambda p, k: " ".join(f"{p}{i}" for i in range(k))
    _fake_loaders(monkeypatch, {
        "validation": [words("v", 30)],
        "test": [words("t", 30), words("u", 30)],
    })
    idx, _ = CS.build_indices(used_limit=1, build_splits={"t": "validation"})
    a, b, c = idx["filtered"], idx["split_gap"], idx["limit_gap"]
    assert a and b and c
    assert not (a & b) and not (a & c) and not (b & c)


def test_the_split_the_build_indexed_lands_in_split_gap_not_in_filtered(monkeypatch):
    """The finding this scan exists to quantify: the corpus indexed ARC-Easy
    and OpenBookQA `validation`, but is scored on `test` (334c86c). The scored
    split's n-grams must therefore show up as *exposure*, not as control."""
    words = lambda p, k: " ".join(f"{p}{i}" for i in range(k))
    scored, built = words("t", 30), words("v", 30)
    _fake_loaders(monkeypatch, {"validation": [built], "test": [scored]})

    idx, _ = CS.build_indices(used_limit=1, build_splits={"t": "validation"})
    assert CS.classify_document(built, idx)["filtered"] is True
    hit = CS.classify_document(scored, idx)
    assert hit["split_gap"] is True, "the scored split must count as exposure"
    assert hit["filtered"] is False, "the build never saw it; it is not control"


def test_without_a_build_split_override_there_is_no_split_gap(monkeypatch):
    """`--build-splits ''` must collapse to the naive two-index view rather
    than silently keeping the ARC/OBQA assumption on a corpus built after the
    fix -- the next dataprep run will not have this gap."""
    words = lambda p, k: " ".join(f"{p}{i}" for i in range(k))
    _fake_loaders(monkeypatch, {"validation": [words("v", 30)],
                                "test": [words("t", 30), words("u", 30)]})
    idx, _ = CS.build_indices(used_limit=1, build_splits=None)
    assert idx["split_gap"] == set()
    assert idx["filtered"] and idx["limit_gap"]


def test_a_task_that_fails_to_load_is_reported_as_zero_not_silently_dropped(monkeypatch):
    """`load_all_tasks` swallows a failing task with a warning, which is how a
    decontamination index can come out short without anything in the manifest
    saying so. The coverage table has to show the zero."""
    import eval as E

    def ok(split=None, limit=None):
        return [_Ex([("a b c d e f g h i j k l m n", "")])]

    def broken(split=None, limit=None):
        raise RuntimeError("hub down")

    monkeypatch.setattr(E, "TASK_LOADERS", {"good": ok, "bad": broken})
    monkeypatch.setattr(E, "TASK_SPLITS", {})
    _, coverage = CS.build_indices(used_limit=1, build_splits=None)
    assert coverage["bad"]["total"] == 0
    assert coverage["good"]["total"] == 1
