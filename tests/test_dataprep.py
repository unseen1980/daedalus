import multiprocessing
"""Tests for daedalus/dataprep.py. Fully offline: dataset streaming, the
tokenizer, and the eval n-gram index are all monkeypatched/faked -- no
network calls, no real HF datasets.

Run: python -m pytest tests/test_dataprep.py -v
"""
import json
import os
import resource

import pytest

import daedalus.dataprep as dp
from daedalus.dataprep import (
    GATED_SUBSTITUTION_NOTES,
    MIXTURE,
    DedupState,
    SourceSpec,
    _set_worker_memory_limit,
    exact_hash,
    run_dataprep,
    run_source,
)


class _NoopWandb:
    """Stand-in for daedalus.wandb_logger.WandbLogger. run_dataprep enables
    W&B by default and this environment often has WANDB_API_KEY exported, so
    without this stub the offline test module below would create real W&B
    runs."""

    def __init__(self, *a, **k):
        self.run = None

    def log(self, *a, **k):
        pass

    def run_url(self, *a, **k):
        return None

    def finish(self):
        pass


@pytest.fixture(autouse=True)
def _no_real_wandb(monkeypatch):
    """run_dataprep logs through `WandbSidecar`, which spawns a real child
    process -- stub it, or every test in this module forks a publisher."""
    import daedalus.wandb_logger as wandb_logger
    import daedalus.wandb_sidecar as wandb_sidecar
    monkeypatch.setattr(wandb_logger, "WandbLogger", _NoopWandb)
    monkeypatch.setattr(wandb_sidecar, "WandbSidecar", _NoopWandb)


class FakeTokenizer:
    def encode(self, text):
        return [abs(hash(w)) % 1000 + 10 for w in text.split()]


def fake_rows(n, prefix="doc", words_per_doc=60):
    """`n` distinct fake documents, each long enough to clear min_chars."""
    for i in range(n):
        yield {"text": " ".join([f"{prefix}{i}_w{j}" for j in range(words_per_doc)])}


# ------------------------------------------------------------------ mixture ---

def test_dataprep_never_imports_train_module():
    """Regression guard: dataprep.py's parent process forks worker processes
    via multiprocessing's fork start method, so it must never import train.py
    -- train.py's TrainArgs dataclass calls torch.cuda.is_available() at
    class-definition (i.e. import) time, which was measured to inflate this
    process's virtual address space from ~4 GB to ~17 GB on a real GPU box
    (CUDA context/driver reservations). Forked workers inherit that bloated
    address space via copy-on-write and then trip their own (deliberately
    tight) per-worker RLIMIT_AS backstop on the very next shared-library
    mmap -- this exact bug crashed a live sweep-scale dataprep run when
    train.py's WandbLogger was first (wrongly) reused here; see STATUS.md.
    daedalus/wandb_logger.py exists specifically to avoid this."""
    import inspect

    source = inspect.getsource(dp)
    assert "from train import" not in source
    assert "import train" not in source


_NEEDS_FORK = pytest.mark.skipif(
    multiprocessing.get_start_method(allow_none=False) != "fork",
    reason="run_dataprep relies on the fork start method (see dataprep.py); "
           "macOS and Windows default to spawn",
)

def test_mixture_shares_sum_to_one():
    assert sum(s.share for s in MIXTURE) == pytest.approx(1.0, abs=1e-9)


def test_mixture_keys_unique():
    keys = [s.key for s in MIXTURE]
    assert len(keys) == len(set(keys))


def test_gated_substitution_notes_cover_known_gated_sources():
    assert "nvidia/Nemotron-CC-v2" in GATED_SUBSTITUTION_NOTES
    assert "nvidia/Nemotron-CC-Math-v1" in GATED_SUBSTITUTION_NOTES


def test_stack_edu_python_uses_per_language_data_files_not_a_row_filter():
    """Regression check: codeparrot/github-code's 'default' config interleaves
    all languages with Python vanishingly rare in stream order (a live check
    scanned 646k consecutive rows with zero Python matches) -- filtering rows
    from it would take unbounded time. Must instead select the parquet-converted
    revision's Python-all/ data_files directly, which are already 100% Python."""
    spec = next(s for s in MIXTURE if s.key == "stack-edu-python")
    assert "data_files" in spec.load_kwargs
    assert "Python" in spec.load_kwargs["data_files"]
    assert spec.filter_fn is None


# -------------------------------------------------------------------- dedup ---

def test_exact_hash_stable_and_normalizes_whitespace_case():
    a = exact_hash("Hello   World")
    b = exact_hash("hello world")
    c = exact_hash("hello there")
    assert a == b
    assert a != c


def test_dedup_state_rejects_exact_duplicate():
    d = DedupState()
    text = "the quick brown fox jumps over the lazy dog " * 5
    assert d.keep(text, "g", None) is True
    assert d.keep(text, "g", None) is False
    assert d.counters["exact_dup"] == 1
    assert d.counters["kept"] == 1


def test_dedup_state_rejects_contaminated_text():
    d = DedupState()
    eval_index = {"a b c d e f g h i j k l m"}
    text = "prefix words here a b c d e f g h i j k l m suffix words too"
    assert d.keep(text, "g", eval_index) is False
    assert d.counters["contaminated"] == 1


def test_dedup_state_rejects_near_duplicate():
    d = DedupState(num_perm=32)
    base = " ".join(f"word{i}" for i in range(200))
    near_dup = base + " onemoreword"  # tiny edit, Jaccard well above 0.85
    assert d.keep(base, "g", None) is True
    assert d.keep(near_dup, "g", None) is False
    assert d.counters["near_dup"] == 1


def test_dedup_state_near_dup_filter_resets_bound_memory():
    """With reset_every=1, the near-dup filter is rebuilt empty before every
    doc, so a near-duplicate is NOT caught across the reset boundary -- the
    documented, deliberate bounded-memory tradeoff."""
    d = DedupState(num_perm=32, near_dup_reset_every=1)
    base = " ".join(f"word{i}" for i in range(200))
    near_dup = base + " onemoreword"
    assert d.keep(base, "g", None) is True
    assert d.keep(near_dup, "g", None) is True  # not caught: filter was reset
    assert d.counters["near_dup"] == 0


def test_dedup_state_exact_dup_set_resets_bound_memory():
    """Exact-dup detection is bounded the same way as near-dup (ADDENDUM 2
    rule 2: never accumulate an unbounded signature table). With
    reset_every=1, the exact-seen set for a group is cleared before every
    doc alongside the near-dup filter, so an exact duplicate is NOT caught
    across the reset boundary -- the same documented, deliberate
    bounded-memory tradeoff as near-dup."""
    d = DedupState(near_dup_reset_every=1)
    text = "the quick brown fox jumps over the lazy dog " * 5
    assert d.keep(text, "g", None) is True
    assert d.keep(text, "g", None) is True  # not caught: exact_seen was reset with the filter
    assert d.counters["exact_dup"] == 0


def test_dedup_state_near_dup_groups_independent():
    d = DedupState(num_perm=32)
    base = " ".join(f"word{i}" for i in range(200))
    near_dup = base + " onemoreword"
    assert d.keep(base, "group-a", None) is True
    assert d.keep(near_dup, "group-b", None) is True  # different group, own filter


# --------------------------------------------------------------- run_source ---

def test_run_source_writes_shards_and_respects_token_budget(tmp_path, monkeypatch):
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(1000))

    stats = run_source(spec, FakeTokenizer(), token_budget=500, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, shard_tokens=1_000_000,
                       log=lambda *a, **k: None)

    assert stats["tokens"] >= 500
    assert stats["n_kept"] > 0
    assert stats["n_seen"] < 1000  # stopped early once budget was hit
    with open(stats["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["total_tokens"] == stats["tokens"]


def test_run_source_drops_too_short_and_duplicate_docs(tmp_path, monkeypatch):
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    rows = [{"text": "short"}] + list(fake_rows(3)) + [next(fake_rows(1))]  # last is an exact dup
    monkeypatch.setattr(dp, "_stream_rows", lambda s: iter(rows))

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, min_chars=20,
                       log=lambda *a, **k: None)

    assert stats["n_too_short"] == 1
    assert stats["n_kept"] == 3  # 3 unique docs; the 4th is an exact dup of doc 0


def test_a_source_manifest_records_what_it_took_to_build_it(tmp_path, monkeypatch):
    """Phase 7 step 3. These shard directories do not stay beside the corpus
    manifest -- they are uploaded per source, hardlinked into holdouts, and read
    through `--data-dir` by four later phases -- so a source that is only
    reproducible next to that other file is not reproducible."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0,
                      text_fn=lambda r: r["text"], split="train[:1%]",
                      revision="refs/convert/parquet",
                      filter_fn=lambda r: True,
                      near_dup_group="web-overlap",
                      load_kwargs={"data_files": "Python-all/*.parquet"})
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(20))

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9,
                       out_dir=str(tmp_path), dedup=DedupState(),
                       eval_ngram_index=None, min_chars=20,
                       log=lambda *a, **k: None)

    with open(stats["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["source_revision"] == "refs/convert/parquet"
    assert manifest["source_split"] == "train[:1%]"
    assert manifest["source_load_kwargs"] == {"data_files": "Python-all/*.parquet"}
    filters = manifest["filters"]
    assert filters["min_chars"] == 20
    assert filters["row_filter"] is True
    assert filters["near_dup_group"] == "web-overlap"
    assert filters["near_dup_threshold"] == DedupState.threshold
    assert filters["near_dup_reset_every"] == DedupState.near_dup_reset_every
    # No field can record a lambda, so what identifies the filters is the tree
    # that ran them.
    assert manifest["builder_git_sha"]
    assert set(manifest["drops"]) == set(dp.DROP_REASONS)


def test_a_source_that_pins_nothing_records_the_null_rather_than_omitting_it(
        tmp_path, monkeypatch):
    """`source_revision: null` is a real answer: the build took whatever the
    dataset's default branch pointed at that day. An absent key reads as an
    older manifest; an explicit null reads as an unpinned source."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0,
                      text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(5))

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9,
                       out_dir=str(tmp_path), dedup=DedupState(),
                       eval_ngram_index=None, log=lambda *a, **k: None)

    with open(stats["manifest_path"]) as f:
        manifest = json.load(f)
    assert "source_revision" in manifest and manifest["source_revision"] is None
    assert manifest["filters"]["row_filter"] is False


def test_the_drop_counts_are_this_sources_and_not_the_whole_workers(
        tmp_path, monkeypatch):
    """`DedupState` is shared by every source a worker handles, so its counters
    are the worker's. Written straight through, source two would inherit source
    one's duplicates and the last source would report the whole group's."""
    dedup = DedupState()
    first = SourceSpec("src-a", "fake/dataset", share=1.0,
                       text_fn=lambda r: r["text"])
    rows = list(fake_rows(3)) + [next(fake_rows(1))]   # the last is an exact dup
    monkeypatch.setattr(dp, "_stream_rows", lambda s: iter(rows))
    a = run_source(first, FakeTokenizer(), token_budget=10**9,
                   out_dir=str(tmp_path / "a"), dedup=dedup,
                   eval_ngram_index=None, log=lambda *a, **k: None)

    second = SourceSpec("src-b", "fake/dataset", share=1.0,
                        text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(3, prefix="other"))
    b = run_source(second, FakeTokenizer(), token_budget=10**9,
                   out_dir=str(tmp_path / "b"), dedup=dedup,
                   eval_ngram_index=None, log=lambda *a, **k: None)

    assert a["drops"]["exact_dup"] == 1
    assert b["drops"]["exact_dup"] == 0
    assert dedup.counters["exact_dup"] == 1     # the worker's total, unchanged
    with open(b["manifest_path"]) as f:
        assert json.load(f)["drops"]["exact_dup"] == 0


def test_the_drop_counts_survive_a_resume(tmp_path, monkeypatch):
    """Seeded from the resume state for the reason `n_kept` is: a source that
    stopped and continued dropped both attempts' documents, and counting only
    the last one puts the drops and the keeps on different denominators."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0,
                      text_fn=lambda r: r["text"])
    rows = list(fake_rows(2)) + [next(fake_rows(1))]
    monkeypatch.setattr(dp, "_stream_rows", lambda s: iter(rows))

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9,
                       out_dir=str(tmp_path), dedup=DedupState(),
                       eval_ngram_index=None, min_chars=20,
                       resume_seed={"n_seen": 9, "n_kept": 6, "n_too_short": 2,
                                    "tokens": 0,
                                    "drops": {"exact_dup": 4, "near_dup": 1,
                                              "contaminated": 3}},
                       log=lambda *a, **k: None)

    assert stats["drops"] == {"exact_dup": 5, "near_dup": 1, "contaminated": 3}
    with open(stats["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["drops"]["exact_dup"] == 5
    assert manifest["drops"]["too_short"] == manifest["n_too_short"] == 2


def test_a_recovered_manifest_seeds_the_dedup_drops_and_not_too_short(
        tmp_path, monkeypatch):
    """`too_short` is recovered one field up as `n_too_short`; folding it back
    in here would seed the next attempt with a count it goes on to make again."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0,
                      text_fn=lambda r: r["text"])
    rows = [{"text": "short"}] + list(fake_rows(2)) + [next(fake_rows(1))]
    monkeypatch.setattr(dp, "_stream_rows", lambda s: iter(rows))
    run_source(spec, FakeTokenizer(), token_budget=10**9,
               out_dir=str(tmp_path / spec.key), dedup=DedupState(),
               eval_ngram_index=None, min_chars=20, log=lambda *a, **k: None)

    recovered = dp._recover_source_stats(spec.key, str(tmp_path))
    assert recovered["n_too_short"] == 1
    assert recovered["drops"] == {"exact_dup": 1, "near_dup": 0,
                                  "contaminated": 0}


def test_run_source_continues_after_dataset_row_error_is_isolated_by_caller(tmp_path, monkeypatch):
    # run_source itself doesn't need to swallow per-row errors -- confirms a
    # clean empty source (0 rows) just yields an empty, valid manifest.
    spec = SourceSpec("empty-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_stream_rows", lambda s: iter([]))

    stats = run_source(spec, FakeTokenizer(), token_budget=1000, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, log=lambda *a, **k: None)

    assert stats["tokens"] == 0
    assert stats["shards"] == []


def test_documents_skip_fast_forwards_without_yielding(monkeypatch):
    """`skip` (issue #3's within-source respawn: resuming a source mid-stream
    after an RSS-respawn chunk boundary) must silently discard the first
    `skip` would-be-yielded documents -- not yield them, not count them
    toward `max_docs` -- then behave exactly as an unskipped call for the
    rest of the stream."""
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(10))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    all_docs = list(dp._documents(spec, max_docs=None))
    skipped_docs = list(dp._documents(spec, max_docs=None, skip=4))

    assert skipped_docs == all_docs[4:]


def test_documents_skip_zero_is_original_behavior(monkeypatch):
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(5))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    assert list(dp._documents(spec, max_docs=None, skip=0)) == list(dp._documents(spec, max_docs=None))


def test_run_source_soft_rss_stop_returns_incomplete_not_error(tmp_path, monkeypatch):
    """issue #3, round two: `finemath-3plus`/`infiwebmath-3plus`/`finephrase`
    all hit the *hard* RSS cap late (95%/88%/44% through their token budget)
    despite the group-boundary trim fix (3527c9f) -- proof the growth is
    substantially within a single long-running source, not just carryover
    between groups. Crossing the lower `rss_soft_limit_gb` threshold must
    stop the source cleanly (no exception) and mark it `incomplete` with a
    `resume_skip`, not `error` -- an error would make --resume redo the
    whole source from scratch, exactly what this fix is meant to avoid."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(3.5))  # over soft, under hard

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, rss_limit_gb=4.0,
                       rss_soft_limit_gb=3.0, rss_check_every=10, log=lambda *a, **k: None)

    assert stats.get("incomplete") is True
    assert "error" not in stats
    assert stats["resume_skip"] == 10  # stopped at the first RSS checkpoint
    assert stats["tokens"] > 0  # partial progress was flushed, not discarded
    with open(stats["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["total_tokens"] == stats["tokens"]


def test_run_source_hard_cap_still_raises_even_with_soft_limit_set(tmp_path, monkeypatch):
    """The hard `rss_limit_gb` cap must still fire (as `error`, same as
    before) when RSS is already over it -- `rss_soft_limit_gb` only adds an
    earlier, non-fatal off-ramp, it doesn't replace the existing backstop."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(5.0))  # over both

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, rss_limit_gb=4.0,
                       rss_soft_limit_gb=3.0, rss_check_every=10, log=lambda *a, **k: None)

    assert "WorkerMemoryExceeded" in stats["error"]
    assert "incomplete" not in stats


def test_run_source_resume_reproduces_unchunked_result(tmp_path, monkeypatch):
    """A source split into two RSS-respawn chunks (soft-stop, then a
    resume_skip/resume_seed continuation) must end up with exactly the same
    n_seen/n_kept/tokens as running it start-to-finish in one process -- the
    chunk boundary must not lose or duplicate a single document, and shards
    must accumulate (not clobber) across the boundary."""
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    baseline = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path / "baseline"),
                          dedup=DedupState(), eval_ngram_index=None, log=lambda *a, **k: None)

    chunked_dir = str(tmp_path / "chunked")
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(3.5))
    chunk1 = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=chunked_dir,
                        dedup=DedupState(), eval_ngram_index=None, rss_soft_limit_gb=3.0,
                        rss_check_every=37, log=lambda *a, **k: None)
    assert chunk1.get("incomplete") is True
    assert 0 < chunk1["resume_skip"] < 200

    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(0.1))
    chunk2 = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=chunked_dir,
                        dedup=DedupState(), eval_ngram_index=None,
                        resume_skip=chunk1["resume_skip"], resume_seed=chunk1,
                        log=lambda *a, **k: None)

    assert "incomplete" not in chunk2
    assert chunk2["n_seen"] == baseline["n_seen"] == 200
    assert chunk2["n_kept"] == baseline["n_kept"]
    assert chunk2["tokens"] == baseline["tokens"]
    with open(chunk2["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["total_tokens"] == baseline["tokens"]
    assert len(chunk2["shards"]) >= len(chunk1["shards"])  # appended, not replaced


# ------------------------------------------- O(1) stream-position resume ---
#
# Issue #3's durable fix. The old contract resumed a soft-stopped source by
# replaying every already-processed row, which is O(n) per respawn and so
# O(n^2/chunk) over a source -- ~8.5e8 rows of pure re-reading for
# fineweb-edu alone -- and left ~0.6 GB of extra RSS in the resumed worker,
# so each respawn did less work than the last. datasets 5.x
# `state_dict`/`load_state_dict` restore the position directly instead.


class _FakeStatefulRows:
    """Stands in for `_RowStream` over a real HF `IterableDataset`: an
    iterable of rows that reports its exact position (`state()`) and can be
    constructed at one (`stream_state`), like datasets 5.x does.

    The two capabilities are separable, because real streams separate them:
    `resumable=False` models a stream that cannot report a position at all
    (an old `datasets`, or a monkeypatched stub), while `restorable=False`
    models one that reports positions fine but rejected the state it was
    handed -- `_RowStream`'s `load_state_dict` failure path. Both must fall
    back to the replay rather than silently restarting from row 0."""

    def __init__(self, n, stream_state=None, resumable=True, restorable=True, prefix="doc"):
        self.n = n
        self.prefix = prefix
        self.pos = 0
        self.resumed = False
        self._resumable = resumable
        if stream_state and resumable and restorable:
            self.pos = stream_state["row"]
            self.resumed = True

    def __iter__(self):
        while self.pos < self.n:
            row = next(fake_rows(1, prefix=f"{self.prefix}{self.pos}_"))
            self.pos += 1
            yield row

    def state(self):
        return {"row": self.pos} if self._resumable else None


def _stateful_stream(n=200, resumable=True, restorable=True):
    def _stream_rows(spec, stream_state=None, log=print):
        return _FakeStatefulRows(n, stream_state, resumable=resumable, restorable=restorable)
    return _stream_rows


def test_document_stream_reports_position_after_last_yielded_doc(monkeypatch):
    monkeypatch.setattr(dp, "_stream_rows", _stateful_stream(10))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    stream = dp._documents(spec, max_docs=None)
    assert stream.state() is None  # nothing consumed yet
    it = iter(stream)
    for _ in range(4):
        next(it)
    assert stream.state() == {"epoch": 0, "hf_state": {"row": 4}}


def test_document_stream_resumes_from_position_without_replaying(monkeypatch):
    """The whole point of the fix: a resumed stream must yield exactly the
    documents an unresumed one would yield next, without re-reading the
    prefix."""
    monkeypatch.setattr(dp, "_stream_rows", _stateful_stream(10))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    all_docs = list(dp._documents(spec, max_docs=None))
    stream = dp._documents(spec, max_docs=None)
    it = iter(stream)
    first_four = [next(it) for _ in range(4)]
    state = stream.state()

    resumed = list(dp._documents(spec, max_docs=None, stream_state=state))
    assert first_four == all_docs[:4]
    assert resumed == all_docs[4:]


def test_document_stream_prefers_position_over_skip_and_never_double_skips(monkeypatch):
    """`resume_skip` and `stream_state` describe the same boundary, so a
    caller passing both (which `run_dataprep` does, keeping the replay path
    as a fallback) must not skip twice."""
    monkeypatch.setattr(dp, "_stream_rows", _stateful_stream(10))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    all_docs = list(dp._documents(spec, max_docs=None))
    resumed = list(dp._documents(spec, max_docs=None, skip=4,
                                  stream_state={"epoch": 0, "hf_state": {"row": 4}}))
    assert resumed == all_docs[4:]


def test_document_stream_falls_back_to_replay_when_position_unavailable(monkeypatch):
    """A stream that can't restore a position must still resume correctly via
    the O(n) replay rather than duplicating an entire chunk of documents."""
    monkeypatch.setattr(dp, "_stream_rows", _stateful_stream(10, resumable=False))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    all_docs = list(dp._documents(spec, max_docs=None))
    resumed = list(dp._documents(spec, max_docs=None, skip=4,
                                  stream_state={"epoch": 0, "hf_state": {"row": 4}}))
    assert resumed == all_docs[4:]


def test_document_stream_reports_the_prior_position_until_replay_finishes(monkeypatch):
    """Mid-replay the live position runs ahead of the caller's counters --
    the skipped documents were already processed and counted by an earlier
    chunk. Saving the live position then (e.g. a network error interrupts a
    long replay) would resume past already-flushed work with `skip` no longer
    applied, duplicating it. So `state()` reports the prior position until
    the replay is complete."""
    # Reports positions, but rejects the one it is handed -- so it replays.
    monkeypatch.setattr(dp, "_stream_rows", _stateful_stream(10, restorable=False))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    prior = {"epoch": 0, "hf_state": {"row": 6}}

    stream = dp._documents(spec, max_docs=None, skip=6, stream_state=prior,
                            log=lambda *a, **k: None)
    it = iter(stream)
    next(it)  # replays 6 documents, then yields the 7th
    assert stream.state() == {"epoch": 0, "hf_state": {"row": 7}}  # real position once replayed

    # Interrupted *during* the replay: must still report the prior position,
    # not the partial one that has overrun the caller's counters.
    mid = dp._documents(spec, max_docs=None, skip=6, stream_state=prior,
                         log=lambda *a, **k: None)
    mid._rows = _FakeStatefulRows(10, {"row": 3})
    mid._replay_done = False
    assert mid.state() == prior


def test_document_stream_state_is_none_for_plain_iterator_streams(monkeypatch):
    """Every other test in this module monkeypatches `_stream_rows` with a
    one-argument stub returning a plain iterator; `state()` must degrade to
    None for those rather than raising."""
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(5))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    stream = dp._documents(spec, max_docs=None)
    list(stream)
    assert stream.state() is None


def test_row_stream_falls_back_when_load_state_dict_rejects_the_state(monkeypatch):
    """A stale or unsupported state must not abort the source -- `_RowStream`
    reports `resumed=False` so `_DocumentStream` replays instead."""
    class _Rejecting:
        def load_state_dict(self, state):
            raise ValueError("stale state")

        def state_dict(self):
            return {"row": 0}

        def __iter__(self):
            return iter([])

    monkeypatch.setattr(dp, "load_dataset", None, raising=False)
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _Rejecting())
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    warnings = []
    rows = dp._RowStream(spec, {"row": 5}, log=warnings.append)
    assert rows.resumed is False
    assert any("could not restore stream position" in w for w in warnings)


def test_run_source_returns_and_persists_the_stream_position(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "_stream_rows", _stateful_stream(200))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(3.5))

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, rss_soft_limit_gb=3.0,
                       rss_check_every=37, log=lambda *a, **k: None)

    assert stats["incomplete"] is True
    assert stats["stream_state"] == {"epoch": 0, "hf_state": {"row": stats["n_seen"]}}
    with open(stats["manifest_path"]) as f:
        manifest = json.load(f)
    # Persisted, so a hard worker crash (which loses the returned stats dict
    # entirely) still leaves a resumable position on disk.
    assert manifest["stream_state"] == stats["stream_state"]
    assert manifest["n_seen"] == stats["n_seen"]


def test_run_source_resume_via_stream_position_reproduces_unchunked_result(tmp_path, monkeypatch):
    """The `resume_skip` equivalence test above, but through the O(1) path:
    two chunks joined by a stream position must produce exactly the same
    n_seen/n_kept/tokens as one unchunked run -- no document lost, none
    duplicated, shards appended not clobbered."""
    monkeypatch.setattr(dp, "_stream_rows", _stateful_stream(200))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    baseline = run_source(spec, FakeTokenizer(), token_budget=10**9,
                          out_dir=str(tmp_path / "baseline"), dedup=DedupState(),
                          eval_ngram_index=None, log=lambda *a, **k: None)

    chunked_dir = str(tmp_path / "chunked")
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(3.5))
    chunk1 = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=chunked_dir,
                        dedup=DedupState(), eval_ngram_index=None, rss_soft_limit_gb=3.0,
                        rss_check_every=37, log=lambda *a, **k: None)
    assert chunk1.get("incomplete") is True
    assert 0 < chunk1["n_seen"] < 200

    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(0.1))
    chunk2 = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=chunked_dir,
                        dedup=DedupState(), eval_ngram_index=None,
                        resume_seed=chunk1, resume_stream_state=chunk1["stream_state"],
                        log=lambda *a, **k: None)

    assert "incomplete" not in chunk2
    assert chunk2["n_seen"] == baseline["n_seen"] == 200
    assert chunk2["n_kept"] == baseline["n_kept"]
    assert chunk2["tokens"] == baseline["tokens"]
    assert len(chunk2["shards"]) >= len(chunk1["shards"])


def test_run_group_worker_threads_stream_position_into_run_source(tmp_path, monkeypatch):
    seen = {}

    def fake_run_source(spec, tokenizer, token_budget, out_dir, dedup, idx, **kw):
        seen[spec.key] = kw.get("resume_stream_state")
        return {"key": spec.key, "tokens": 1, "shards": []}

    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "run_source", fake_run_source)
    # Required whenever `_run_group_worker` is called directly rather than in
    # a forked child -- otherwise it permanently caps THIS pytest process's
    # RLIMIT_AS at 12 GB and every later test in the session starves. See the
    # same note in test_run_group_worker_stops_group_after_rss_cap_trip.
    monkeypatch.setattr(dp, "_set_worker_memory_limit", lambda *a, **k: None)
    spec = SourceSpec("srcA", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_ACTIVE_MIXTURE_BY_KEY", {"srcA": spec})
    state = {"epoch": 0, "hf_state": {"row": 99}}

    dp._run_group_worker("g", ["srcA"], 1000, str(tmp_path), 1_000_000, None, None,
                          resume_state={"srcA": {"resume_skip": 99, "resume_seed": {},
                                                  "stream_state": state}})

    assert seen["srcA"] == state


def _echo_resume_state_group_worker(group_key, spec_keys, target_tokens, out_root, shard_tokens,
                                     max_docs_per_source, eval_ngram_index, rss_limit_gb=None,
                                     rss_soft_limit_gb=None, resume_state=None,
                                     rss_check_every=5_000, checkpoint_every=50_000):
    """Fake `_run_group_worker` that echoes whatever resume position it was
    handed back through its *returned stats*. The return value is the only
    channel that survives the fork -- `_run_group_worker` really runs in a
    child process, so a parent-side list the child appends to would never be
    observed (an earlier version of these tests failed exactly that way).
    A key with no resume state soft-stops with a position; a key with one
    completes and reports what it received."""
    resume_state = resume_state or {}
    results = []
    for key in spec_keys:
        handed = resume_state.get(key)
        common = {"key": key, "dataset": "fake/dataset", "n_too_short": 0,
                  "manifest_path": os.path.join(out_root, key, "manifest.json"),
                  "token_budget": 1000}
        if handed is None:
            results.append({**common, "n_seen": 10, "n_kept": 10, "tokens": 10,
                            "shards": [{"file": f"{key}_00000.bin", "tokens": 10}],
                            "elapsed_s": 1.0, "achieved_fraction": 0.5,
                            "incomplete": True, "resume_skip": 10,
                            "stream_state": {"epoch": 0, "hf_state": {"row": 10}}})
        else:
            results.append({**common, "n_seen": 20, "n_kept": 20, "tokens": 20,
                            "shards": [{"file": f"{key}_00000.bin", "tokens": 20}],
                            "elapsed_s": 2.0, "achieved_fraction": 1.0,
                            "received_stream_state": handed.get("stream_state")})
    return group_key, results, {}


def _observing_group_worker(group_key, spec_keys, target_tokens, out_root, *a, **kw):
    """Reads the run-level manifest as it stands *during* the run -- after
    resume bookkeeping, before any result has come back -- and returns it in
    the stats. Module-level, not a closure: `_run_group_worker` is pickled by
    qualified name for the child, so a nested function fails to pickle and
    the run silently takes the crash-recovery path instead."""
    during = json.load(open(os.path.join(os.path.dirname(out_root), "manifest.json")))
    return group_key, [{"key": spec_keys[0], "tokens": 99, "n_seen": 5,
                        "shards": [{"file": f"{spec_keys[0]}_00000.bin", "tokens": 99}],
                        "observed_manifest": during}], {}


@_NEEDS_FORK
def test_run_dataprep_hands_the_stream_position_to_the_respawned_worker(tmp_path, monkeypatch):
    """A soft-stopped source's continuation must carry its position, not just
    its `resume_skip` -- otherwise the respawned worker replays the whole
    prefix and the run makes less progress with every respawn."""
    monkeypatch.setattr(dp, "_run_group_worker", _echo_resume_state_group_worker)

    manifest = run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(1),
        skip_decontam=True, max_workers=1, rss_soft_limit_gb=3.0,
        log=lambda *a, **k: None)

    assert manifest["sources"][0]["received_stream_state"] == {"epoch": 0, "hf_state": {"row": 10}}


@_NEEDS_FORK
def test_run_dataprep_recovers_the_stream_position_across_runs(tmp_path, monkeypatch):
    """Cross-process --resume: an errored source with real progress must be
    handed its saved position, so the next attempt continues rather than
    re-streaming from row 0 (four attempts in a row carried zero tokens
    forward before this -- see issue #3)."""
    monkeypatch.setattr(dp, "_run_group_worker", _echo_resume_state_group_worker)
    state = {"epoch": 0, "hf_state": {"row": 4242}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "target_tokens": 1000,
        "sources": [{"key": "src0", "tokens": 10, "n_seen": 4242,
                     "shards": [{"file": "src0_00000.bin", "tokens": 10}], "stream_state": state,
                     "error": "WorkerMemoryExceeded('worker RSS 4.17 GB exceeded the 4.00 GB cap')"}],
    }))

    manifest = run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(manifest_path), mixture=_fake_mixture(1),
        skip_decontam=True, max_workers=1, resume=True, log=lambda *a, **k: None)

    assert manifest["sources"][0]["received_stream_state"] == state


@_NEEDS_FORK
def test_run_dataprep_recovers_a_source_that_was_in_flight_when_the_run_died(tmp_path, monkeypatch):
    """A source still running when the run ended (killed, out of credit, box
    rebooted) has no run-level manifest entry at all -- one is only appended
    when a source *finishes*. Its shards and saved position sit on disk with
    nothing looking at them, so the next --resume restarted it from row 0 and
    overwrote those shards from _00000. Found by killing a live validation
    run mid-source: 5 shards and a valid position on disk, `sources: []` in
    the run-level manifest."""
    monkeypatch.setattr(dp, "_run_group_worker", _echo_resume_state_group_worker)
    state = {"epoch": 0, "hf_state": {"row": 35500}}
    source_dir = tmp_path / "shards" / "src0"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text(json.dumps({
        "total_tokens": 37_365_481,
        "shards": [{"file": f"src0_{i:05d}.bin", "tokens": 7_473_096} for i in range(5)],
        "stream_state": state, "n_seen": 35_500, "n_kept": 35_000,
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"target_tokens": 1000, "sources": []}))

    manifest = run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(manifest_path), mixture=_fake_mixture(1),
        skip_decontam=True, max_workers=1, resume=True, log=lambda *a, **k: None)

    assert manifest["sources"][0]["received_stream_state"] == state


@_NEEDS_FORK
def test_run_dataprep_redoes_an_in_flight_source_that_has_no_resume_point(tmp_path, monkeypatch):
    """Shards on disk with no recorded position at all (e.g. the per-source
    manifest was truncated mid-write and only .bin files survived) must be
    redone from scratch, not appended to: appending at an unknown stream
    position would silently duplicate documents into the corpus."""
    monkeypatch.setattr(dp, "_run_group_worker", _echo_resume_state_group_worker)
    source_dir = tmp_path / "shards" / "src0"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text("")           # truncated by the crash
    (source_dir / "src0_00000.bin").write_bytes(b"\x00\x00" * 50)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"target_tokens": 1000, "sources": []}))

    manifest = run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(manifest_path), mixture=_fake_mixture(1),
        skip_decontam=True, max_workers=1, resume=True, log=lambda *a, **k: None)

    # The initial dispatch got no seed, so the fake worker took its "fresh"
    # branch and the only position that ever reached it is the one it
    # generated itself at its own soft stop -- not anything read off disk.
    assert manifest["sources"][0]["received_stream_state"] == {"epoch": 0, "hf_state": {"row": 10}}


@_NEEDS_FORK
def test_run_dataprep_keeps_an_errored_entry_on_disk_until_it_is_superseded(tmp_path, monkeypatch):
    """A failed source's entry used to be deleted from `manifest["sources"]`
    the moment a resume started, so the on-disk manifest immediately lost
    that source's recorded progress. A run that then died before the source
    finished left nothing behind. That bit for real: killing attempt 5
    mid-replay cost `finephrase` and `infiwebmath-3plus` their `n_seen` --
    2.57B tokens' worth of resume position -- recoverable only from this
    file's git history. The entry now survives on disk until a new result
    replaces it."""
    monkeypatch.setattr(dp, "_run_group_worker", _observing_group_worker)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "target_tokens": 1000,
        "sources": [{"key": "src0", "tokens": 10, "n_seen": 4242,
                     "shards": [{"file": "src0_00000.bin", "tokens": 10}],
                     "error": "WorkerMemoryExceeded('...')"}],
    }))

    manifest = run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(manifest_path), mixture=_fake_mixture(1),
        skip_decontam=True, max_workers=1, resume=True, log=lambda *a, **k: None)

    # Still on disk, with its resume position, while the retry was in flight.
    during = manifest["sources"][0]["observed_manifest"]
    assert [s["key"] for s in during["sources"]] == ["src0"]
    assert during["sources"][0]["n_seen"] == 4242
    # Replaced, not duplicated, once the retry produced a result.
    assert len(manifest["sources"]) == 1
    assert manifest["sources"][0]["tokens"] == 99
    assert "error" not in manifest["sources"][0]


@_NEEDS_FORK
def test_run_dataprep_recovers_stream_position_from_the_on_disk_source_manifest(tmp_path, monkeypatch):
    """When the run-level manifest entry predates stream positions (e.g. it
    was written by the crash-recovery path, which reconstructs an entry from
    disk), fall back to the per-source manifest, which `run_source` updates
    at every chunk boundary."""
    monkeypatch.setattr(dp, "_run_group_worker", _echo_resume_state_group_worker)
    state = {"epoch": 0, "hf_state": {"row": 777}}
    source_dir = tmp_path / "shards" / "src0"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text(json.dumps({
        "total_tokens": 10, "shards": [{"file": "src0_00000.bin", "tokens": 10}],
        "stream_state": state, "n_seen": 777,
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "target_tokens": 1000,
        "sources": [{"key": "src0", "tokens": 10, "n_seen": 777,
                     "shards": [{"file": "src0_00000.bin", "tokens": 10}],
                     "error": "BrokenProcessPool()"}],
    }))

    manifest = run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(manifest_path), mixture=_fake_mixture(1),
        skip_decontam=True, max_workers=1, resume=True, log=lambda *a, **k: None)

    assert manifest["sources"][0]["received_stream_state"] == state


# ------------------------------------------------------------ run_dataprep ---

def _fake_mixture(n_sources=2):
    return [
        SourceSpec(f"src{i}", f"fake/dataset{i}", share=1.0 / n_sources, text_fn=lambda r: r["text"])
        for i in range(n_sources)
    ]


@_NEEDS_FORK
def test_run_dataprep_end_to_end_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))

    manifest = run_dataprep(
        target_tokens=2000, out_root=str(tmp_path / "shards"),
        manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(2),
        skip_decontam=True, log=lambda *a, **k: None,
    )

    assert len(manifest["sources"]) == 2
    assert manifest["total_tokens"] > 0
    assert manifest["substitutions"] == GATED_SUBSTITUTION_NOTES
    assert "achieved_fraction" in manifest


@_NEEDS_FORK
def test_run_dataprep_resume_skips_completed_sources(tmp_path, monkeypatch):
    # Sources now stream inside forked worker processes, so the call count must
    # be tracked via a real file (visible across the fork boundary), not an
    # in-memory dict (each fork gets its own private copy-on-write copy, and
    # mutations inside a worker would never be seen back in this test process).
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    calls_file = tmp_path / "calls.log"

    def counting_stream(spec):
        with open(calls_file, "a") as f:
            f.write(spec.key + "\n")
        return fake_rows(200)

    monkeypatch.setattr(dp, "_stream_rows", counting_stream)
    manifest_path = str(tmp_path / "manifest.json")
    out_root = str(tmp_path / "shards")

    run_dataprep(target_tokens=2000, out_root=out_root, manifest_path=manifest_path,
                mixture=_fake_mixture(2), skip_decontam=True, log=lambda *a, **k: None)
    assert calls_file.read_text().count("\n") == 2

    # second run, same manifest: both sources already recorded, must not re-stream
    manifest2 = run_dataprep(target_tokens=2000, out_root=out_root, manifest_path=manifest_path,
                             mixture=_fake_mixture(2), skip_decontam=True, resume=True,
                             log=lambda *a, **k: None)
    assert calls_file.read_text().count("\n") == 2  # unchanged
    assert len(manifest2["sources"]) == 2


@_NEEDS_FORK
def test_run_dataprep_resume_retries_errored_sources(tmp_path, monkeypatch):
    """A source that hit an error (e.g. an RSS-cap trip, see STATUS.md's
    malloc_trim incident) only got a partial share of its token budget --
    finepdfs-edu and finemath-3plus stopped at 11% and 2.8% in the real run.
    Its manifest entry must not be treated as "done" on --resume, or it is
    skipped forever. A clean completion still must not be re-streamed."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    calls_file = tmp_path / "calls.log"

    def counting_stream(spec):
        with open(calls_file, "a") as f:
            f.write(spec.key + "\n")
        return fake_rows(200)

    monkeypatch.setattr(dp, "_stream_rows", counting_stream)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "target_tokens": 2000,
        "sources": [
            {"key": "src0", "dataset": "fake/0", "tokens": 500, "n_seen": 200, "n_kept": 200},
            {"key": "src1", "dataset": "fake/1", "tokens": 10, "n_seen": 5,
             "error": "WorkerMemoryExceeded('worker RSS 4.22 GB exceeded the 4.00 GB cap')"},
        ],
        "dedup_counters": {"kept": 200, "exact_dup": 0, "near_dup": 0, "contaminated": 0},
    }))

    manifest = run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                             manifest_path=str(manifest_path), mixture=_fake_mixture(2),
                             skip_decontam=True, resume=True, log=lambda *a, **k: None)

    calls = calls_file.read_text().splitlines()
    # src0 is re-streamed too, and that is the point of _demote_short_sources:
    # this fixture gives it 500 tokens against a 1,000-token budget (share 0.5
    # of target 2,000), so it never finished. The old contract called it done
    # because its entry carried no error, which is the bug that made the 60B
    # top-up a twelve-second no-op. A source at or over its budget is still
    # skipped -- test_resume_does_not_restream_a_source_that_met_its_budget.
    assert sorted(calls) == ["src0", "src1"]
    by_key = {s["key"]: s for s in manifest["sources"]}
    assert "error" not in by_key["src1"]  # replaced by the clean retry's result
    assert by_key["src1"]["tokens"] > 10  # got a real share this time, not the stub 10
    assert by_key["src0"]["tokens"] >= 500  # continued from its 500, never truncated


def test_resume_does_not_restream_a_source_that_met_its_budget(tmp_path, monkeypatch):
    """The other half of the contract: a finished source stays finished.

    Guards the obvious regression from _demote_short_sources -- a rule that
    reopened everything would re-stream the whole corpus on every --resume.
    """
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    calls_file = tmp_path / "calls.log"

    def counting_stream(spec):
        with open(calls_file, "a") as f:
            f.write(spec.key + "\n")
        return fake_rows(200)

    monkeypatch.setattr(dp, "_stream_rows", counting_stream)
    manifest_path = tmp_path / "manifest.json"
    # Both sources at exactly their 1,000-token budget (share 0.5 of 2,000).
    manifest_path.write_text(json.dumps({
        "target_tokens": 2000,
        "sources": [
            {"key": "src0", "dataset": "fake/0", "tokens": 1000, "n_seen": 200, "n_kept": 200},
            {"key": "src1", "dataset": "fake/1", "tokens": 1000, "n_seen": 200, "n_kept": 200},
        ],
        "dedup_counters": {"kept": 400, "exact_dup": 0, "near_dup": 0, "contaminated": 0},
    }))

    run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                 manifest_path=str(manifest_path), mixture=_fake_mixture(2),
                 skip_decontam=True, resume=True, log=lambda *a, **k: None)

    assert not calls_file.exists(), "re-streamed a source that had met its budget"


def test_an_exhausted_source_is_not_retried_forever(tmp_path, monkeypatch):
    """A stream with no more documents stays below budget permanently.

    Without the `exhausted` marker, _demote_short_sources would re-dispatch it
    on every single --resume to discover the same emptiness again.
    """
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    calls_file = tmp_path / "calls.log"

    def counting_stream(spec):
        with open(calls_file, "a") as f:
            f.write(spec.key + "\n")
        return fake_rows(200)

    monkeypatch.setattr(dp, "_stream_rows", counting_stream)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "target_tokens": 2000,
        "sources": [
            {"key": "src0", "dataset": "fake/0", "tokens": 500, "n_seen": 200,
             "n_kept": 200, "exhausted": True},
            {"key": "src1", "dataset": "fake/1", "tokens": 1000, "n_seen": 200, "n_kept": 200},
        ],
        "dedup_counters": {"kept": 400, "exact_dup": 0, "near_dup": 0, "contaminated": 0},
    }))

    run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                 manifest_path=str(manifest_path), mixture=_fake_mixture(2),
                 skip_decontam=True, resume=True, log=lambda *a, **k: None)

    assert not calls_file.exists(), "re-streamed a source whose stream is exhausted"


def test_run_source_marks_an_exhausted_stream(tmp_path, monkeypatch):
    """The marker is set where it is observed: the stream ran out, no break."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(3))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    stats = run_source(spec, FakeTokenizer(), token_budget=10_000_000,
                       out_dir=str(tmp_path), dedup=DedupState(), eval_ngram_index=None,
                       shard_tokens=1_000_000, log=lambda *a, **k: None)
    assert stats.get("exhausted") is True
    assert stats["tokens"] < 10_000_000


def test_run_source_does_not_mark_exhausted_when_the_budget_is_met(tmp_path, monkeypatch):
    """Breaking on the budget is the normal path and must not look exhausted."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(500))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    stats = run_source(spec, FakeTokenizer(), token_budget=50, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None,
                       shard_tokens=1_000_000, log=lambda *a, **k: None)
    assert "exhausted" not in stats
    assert stats["tokens"] >= 50


@_NEEDS_FORK
def test_run_dataprep_shares_dedup_within_group_across_sources(tmp_path, monkeypatch):
    """Sources sharing a near_dup_group must run in the same worker process
    sharing one DedupState, so cross-source exact/near-dup catching (e.g. the
    real fineweb-edu+dclm-baseline overlap) survives the parallel-group split."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    shared_doc = "same shared document text repeated many times for length here " * 5

    def stream(spec):
        return iter([{"text": shared_doc}])

    monkeypatch.setattr(dp, "_stream_rows", stream)
    mixture = [
        SourceSpec("srcA", "fake/a", share=0.5, text_fn=lambda r: r["text"], near_dup_group="shared"),
        SourceSpec("srcB", "fake/b", share=0.5, text_fn=lambda r: r["text"], near_dup_group="shared"),
    ]
    manifest = run_dataprep(target_tokens=1000, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"), mixture=mixture,
                            skip_decontam=True, log=lambda *a, **k: None)

    by_key = {s["key"]: s for s in manifest["sources"]}
    assert by_key["srcA"]["n_kept"] == 1
    assert by_key["srcB"]["n_kept"] == 0  # exact dup of srcA's doc, caught via the shared DedupState
    assert manifest["dedup_counters"]["exact_dup"] == 1


@_NEEDS_FORK
def test_run_dataprep_no_resume_redoes_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))
    manifest_path = str(tmp_path / "manifest.json")
    out_root = str(tmp_path / "shards")

    run_dataprep(target_tokens=2000, out_root=out_root, manifest_path=manifest_path,
                mixture=_fake_mixture(2), skip_decontam=True, log=lambda *a, **k: None)
    manifest2 = run_dataprep(target_tokens=2000, out_root=out_root, manifest_path=manifest_path,
                             mixture=_fake_mixture(2), skip_decontam=True, resume=False,
                             log=lambda *a, **k: None)
    assert len(manifest2["sources"]) == 2  # fresh manifest, not appended to the old one


@_NEEDS_FORK
def test_run_dataprep_source_failure_is_recorded_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())

    def boom(spec):
        if spec.key == "src0":
            raise RuntimeError("simulated dataset failure")
        return fake_rows(200)

    monkeypatch.setattr(dp, "_stream_rows", boom)
    manifest = run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"),
                            mixture=_fake_mixture(2), skip_decontam=True,
                            log=lambda *a, **k: None)

    by_key = {s["key"]: s for s in manifest["sources"]}
    assert "error" in by_key["src0"]
    assert by_key["src0"]["tokens"] == 0
    assert by_key["src1"]["tokens"] > 0


@_NEEDS_FORK
def test_run_dataprep_group_setup_failure_is_recorded_not_fatal(tmp_path, monkeypatch):
    """Regression test: if a worker fails during setup (tokenizer load, mem
    limit) rather than mid-source, that must be recorded per-source too, not
    raised -- a bare `get_tokenizer()` call outside try/except previously let
    this crash the whole worker's future, which (before the fut.result()
    hardening below) would have taken down every other in-flight group too."""
    def boom_tokenizer():
        raise RuntimeError("simulated tokenizer load failure")

    monkeypatch.setattr(dp, "get_tokenizer", boom_tokenizer)
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))

    manifest = run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"),
                            mixture=_fake_mixture(2), skip_decontam=True,
                            log=lambda *a, **k: None)

    assert len(manifest["sources"]) == 2
    for stats in manifest["sources"]:
        assert "error" in stats
        assert stats["tokens"] == 0


def _crashing_group_worker(group_key, spec_keys, *a, **k):
    """Module-level (picklable, forkable) stand-in for _run_group_worker that
    crashes unconditionally -- simulates a worker dying in a way that
    entirely escapes _run_group_worker's own internal try/except (e.g. a
    segfault or an OS kill), to exercise run_dataprep's outer fut.result()
    guard."""
    raise RuntimeError("simulated total worker crash")


def test_recover_source_stats_prefers_valid_manifest(tmp_path):
    source_dir = tmp_path / "srcA"
    source_dir.mkdir()
    (source_dir / "manifest.json").write_text(json.dumps({
        "total_tokens": 42, "shards": [{"file": "srcA_00000.bin", "tokens": 42}],
        "source_dataset": "fake/dataset", "source_config": "cfg",
    }))
    stats = dp._recover_source_stats("srcA", str(tmp_path))
    assert stats["key"] == "srcA"
    assert stats["dataset"] == "fake/dataset"
    assert stats["config"] == "cfg"
    assert stats["tokens"] == 42
    assert stats["shards"] == [{"file": "srcA_00000.bin", "tokens": 42}]
    # A manifest written before stream positions were recorded (or by the
    # .bin-scanning fallback) simply has none -- recovery must still work,
    # falling back to the O(n) replay path rather than raising.
    assert stats["stream_state"] is None


def test_recover_source_stats_reads_back_persisted_stream_position(tmp_path):
    """The per-source manifest is the only record that survives a worker
    dying hard, so it carries the resume position and counters -- otherwise
    a crashed source restarts from row 0 on the next --resume, which is the
    failure that produced zero carried-forward tokens across four dataprep
    attempts (issue #3)."""
    source_dir = tmp_path / "srcD"
    source_dir.mkdir()
    state = {"epoch": 0, "hf_state": {"shard_idx": 7, "shard_example_idx": 1234}}
    (source_dir / "manifest.json").write_text(json.dumps({
        "total_tokens": 42, "shards": [{"file": "srcD_00000.bin", "tokens": 42}],
        "source_dataset": "fake/dataset", "source_config": "cfg",
        "stream_state": state, "n_seen": 900, "n_kept": 850, "n_too_short": 5,
    }))
    stats = dp._recover_source_stats("srcD", str(tmp_path))
    assert stats["stream_state"] == state
    assert stats["n_seen"] == 900
    assert stats["n_kept"] == 850
    assert stats["n_too_short"] == 5


def test_recover_source_stats_falls_back_to_bin_files_when_manifest_corrupt(tmp_path):
    source_dir = tmp_path / "srcB"
    source_dir.mkdir()
    (source_dir / "manifest.json").write_text("")  # truncated mid-write by a crash
    (source_dir / "srcB_00000.bin").write_bytes(b"\x00\x00" * 50)
    (source_dir / "srcB_00001.bin").write_bytes(b"\x00\x00" * 25)
    stats = dp._recover_source_stats("srcB", str(tmp_path))
    assert stats["tokens"] == 75
    assert stats["shards"] == [
        {"file": "srcB_00000.bin", "tokens": 50},
        {"file": "srcB_00001.bin", "tokens": 25},
    ]


def test_recover_source_stats_returns_none_when_nothing_flushed(tmp_path):
    assert dp._recover_source_stats("srcC", str(tmp_path)) is None
    (tmp_path / "srcC").mkdir()
    assert dp._recover_source_stats("srcC", str(tmp_path)) is None  # empty dir, no shards


@_NEEDS_FORK
def test_run_dataprep_survives_worker_crash_that_escapes_internal_handling(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))
    monkeypatch.setattr(dp, "_run_group_worker", _crashing_group_worker)

    manifest = run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"),
                            mixture=_fake_mixture(2), skip_decontam=True,
                            log=lambda *a, **k: None)

    assert len(manifest["sources"]) == 2  # both groups' sources recorded despite the crash
    for stats in manifest["sources"]:
        assert "error" in stats
        assert stats["tokens"] == 0


def test_run_group_worker_skips_remaining_sources_after_rss_cap_trip(tmp_path, monkeypatch):
    """Regression test for the eighth incident (STATUS.md/COSTS.md): live,
    starting a second source in the same worker process right after the
    first one had already tripped its RSS cap is what produced a raw
    C-level `malloc` failure moments later, taking the whole worker process
    down hard enough to poison every OTHER group's in-flight progress too
    (see _recover_source_stats's docstring). Once one source in a group
    exceeds its RSS cap, the rest of that group's sources must be skipped
    in this worker -- not attempted -- so a fresh worker on the next
    --resume attempt gets a clean memory budget instead."""
    def always_over_cap(limit_gb):
        raise dp.WorkerMemoryExceeded("boom, always over cap")

    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_check_worker_rss", always_over_cap)
    # _run_group_worker is normally only ever run inside a short-lived forked
    # child (see _set_worker_memory_limit's docstring: "soft==hard means this
    # process can never raise it again, which is fine: workers are
    # short-lived"). Calling it directly here, in the long-lived pytest
    # process, would otherwise permanently cap THIS process's RLIMIT_AS at
    # 12 GB for the rest of the test session -- caught live: it starved every
    # later CUDA/torch test in the same session of virtual address space.
    monkeypatch.setattr(dp, "_set_worker_memory_limit", lambda *a, **k: None)
    calls = []

    def tracking_stream_rows(spec):
        calls.append(spec.key)
        return fake_rows(5000)

    monkeypatch.setattr(dp, "_stream_rows", tracking_stream_rows)
    old_mixture = dp._ACTIVE_MIXTURE_BY_KEY
    dp._ACTIVE_MIXTURE_BY_KEY = {s.key: s for s in _fake_mixture(2)}
    try:
        group_key, results, _ = dp._run_group_worker(
            "group", ["src0", "src1"], target_tokens=10_000_000, out_root=str(tmp_path / "shards"),
            shard_tokens=100, max_docs_per_source=None, eval_ngram_index=None, rss_limit_gb=2.0)
    finally:
        dp._ACTIVE_MIXTURE_BY_KEY = old_mixture

    by_key = {s["key"]: s for s in results}
    assert "WorkerMemoryExceeded" in by_key["src0"]["error"]
    assert "skipped" in by_key["src1"]["error"]
    assert calls == ["src0"]  # src1's stream was never touched


def test_run_group_worker_trims_allocator_before_returning(tmp_path, monkeypatch):
    """Regression test for issue #3: with more groups than workers,
    `ProcessPoolExecutor` reuses each OS process for multiple group tasks, so
    a group's `DedupState`/tokenizer memory that isn't trimmed before
    returning becomes the *next* group's inherited RSS baseline on that same
    process. Caught live: `cosmopedia-v2` and `finewiki-en` -- both small,
    light sources -- tripped the RSS cap within their first ~5-25K documents
    after landing on a worker that had just finished the heavy
    `stack-edu-python` group. `_run_group_worker` must call `_malloc_trim`
    (after `gc.collect()`) before returning, on both the normal-completion
    and the setup-failure path."""
    calls = []
    monkeypatch.setattr(dp, "_malloc_trim", lambda: calls.append(True))
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))
    monkeypatch.setattr(dp, "_set_worker_memory_limit", lambda *a, **k: None)
    old_mixture = dp._ACTIVE_MIXTURE_BY_KEY
    dp._ACTIVE_MIXTURE_BY_KEY = {s.key: s for s in _fake_mixture(1)}
    try:
        dp._run_group_worker(
            "group", ["src0"], target_tokens=10_000_000, out_root=str(tmp_path / "shards"),
            shard_tokens=100, max_docs_per_source=None, eval_ngram_index=None, rss_limit_gb=None)
    finally:
        dp._ACTIVE_MIXTURE_BY_KEY = old_mixture
    assert calls == [True]


def test_run_group_worker_trims_allocator_after_setup_failure(tmp_path, monkeypatch):
    """Same as above, but for the setup-failure path (tokenizer load dies
    before any source runs) -- that return must also trim, not just the
    normal-completion one, since it's reached just as easily by a reused
    worker."""
    calls = []
    monkeypatch.setattr(dp, "_malloc_trim", lambda: calls.append(True))

    def boom_tokenizer():
        raise RuntimeError("simulated tokenizer load failure")

    monkeypatch.setattr(dp, "get_tokenizer", boom_tokenizer)
    monkeypatch.setattr(dp, "_set_worker_memory_limit", lambda *a, **k: None)
    old_mixture = dp._ACTIVE_MIXTURE_BY_KEY
    dp._ACTIVE_MIXTURE_BY_KEY = {s.key: s for s in _fake_mixture(1)}
    try:
        dp._run_group_worker(
            "group", ["src0"], target_tokens=10_000_000, out_root=str(tmp_path / "shards"),
            shard_tokens=100, max_docs_per_source=None, eval_ngram_index=None, rss_limit_gb=None)
    finally:
        dp._ACTIVE_MIXTURE_BY_KEY = old_mixture
    assert calls == [True]


def test_run_group_worker_propagates_incomplete_result_on_soft_rss_stop(tmp_path, monkeypatch):
    """issue #3, round two: a source that crosses `rss_soft_limit_gb` inside
    `run_source` must come back through `_run_group_worker`'s `results` list
    still marked `incomplete`/`resume_skip` -- not swallowed, not converted
    into an `error` -- so `run_dataprep`'s orchestration loop can resubmit
    it. Unlike the hard-cap case (see
    test_run_group_worker_skips_remaining_sources_after_rss_cap_trip), a
    soft-stopped source does NOT poison the rest of this group's sources: the
    soft limit leaves real headroom below the hard cap by design, so it's
    safe to keep going in the same process."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_set_worker_memory_limit", lambda *a, **k: None)
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(6000))
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(3.5))  # over soft(3.0), under hard(4.0)
    old_mixture = dp._ACTIVE_MIXTURE_BY_KEY
    dp._ACTIVE_MIXTURE_BY_KEY = {s.key: s for s in _fake_mixture(1)}
    try:
        group_key, results, _ = dp._run_group_worker(
            "group", ["src0"], target_tokens=10_000_000, out_root=str(tmp_path / "shards"),
            shard_tokens=100, max_docs_per_source=None, eval_ngram_index=None,
            rss_limit_gb=4.0, rss_soft_limit_gb=3.0)
    finally:
        dp._ACTIVE_MIXTURE_BY_KEY = old_mixture

    assert group_key == "group"
    stats = results[0]
    assert stats.get("incomplete") is True
    assert "error" not in stats
    assert stats["resume_skip"] > 0
    assert stats["tokens"] > 0


def test_run_group_worker_passes_rss_check_every_through_to_run_source(tmp_path, monkeypatch):
    """`rss_check_every` (issue #3's follow-up fix: bursty growth of up to
    ~0.7 GB was measured live within a single default 5,000-doc check window,
    close enough to jump straight past the soft *and* hard threshold between
    two checks) must reach `run_source` through `_run_group_worker`, not just
    be accepted and silently dropped. Verified by setting it far tighter than
    the default and confirming the soft-stop trips at that exact, much
    earlier checkpoint instead of the default 5,000-doc one."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_set_worker_memory_limit", lambda *a, **k: None)
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(500))
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(3.5))  # over soft(3.0), under hard(4.0)
    old_mixture = dp._ACTIVE_MIXTURE_BY_KEY
    dp._ACTIVE_MIXTURE_BY_KEY = {s.key: s for s in _fake_mixture(1)}
    try:
        group_key, results, _ = dp._run_group_worker(
            "group", ["src0"], target_tokens=10_000_000, out_root=str(tmp_path / "shards"),
            shard_tokens=100, max_docs_per_source=None, eval_ngram_index=None,
            rss_limit_gb=4.0, rss_soft_limit_gb=3.0, rss_check_every=17)
    finally:
        dp._ACTIVE_MIXTURE_BY_KEY = old_mixture

    stats = results[0]
    assert stats.get("incomplete") is True
    assert stats["resume_skip"] == 17  # first check boundary at the passed-through interval, not 5,000


def _incomplete_then_done_group_worker(group_key, spec_keys, target_tokens, out_root,
                                        shard_tokens, max_docs_per_source, eval_ngram_index,
                                        rss_limit_gb=None, rss_soft_limit_gb=None, resume_state=None,
                                        rss_check_every=5_000, checkpoint_every=50_000):
    """Module-level (picklable, forkable) stand-in for `_run_group_worker`:
    the first time it sees a key it reports `incomplete` with a fixed
    partial result; once called again with that key present in
    `resume_state` (i.e. `run_dataprep` resubmitted it as a continuation),
    it reports the source done. Lets
    test_run_dataprep_resubmits_incomplete_source_to_a_fresh_process exercise
    the real orchestration loop (continuation_keys/resume_state/
    _submit_respawn, a real extra fork) without having to fight simulating
    real cross-process RSS growth."""
    resume_state = resume_state or {}
    results = []
    for key in spec_keys:
        if key not in resume_state:
            results.append({
                "key": key, "dataset": "fake/dataset", "n_seen": 100, "n_kept": 100,
                "n_too_short": 0, "tokens": 500, "shards": [{"file": f"{key}_00000.bin", "tokens": 500}],
                "manifest_path": os.path.join(out_root, key, "manifest.json"), "elapsed_s": 1.0,
                "token_budget": 1000, "achieved_fraction": 0.5,
                "incomplete": True, "resume_skip": 100,
            })
        else:
            seed = resume_state[key]["resume_seed"]
            results.append({
                "key": key, "dataset": "fake/dataset", "n_seen": 200, "n_kept": 200,
                "n_too_short": 0, "tokens": 1000,
                "shards": seed["shards"] + [{"file": f"{key}_00001.bin", "tokens": 500}],
                "manifest_path": os.path.join(out_root, key, "manifest.json"),
                "elapsed_s": seed["elapsed_s"] + 1.0, "token_budget": 1000, "achieved_fraction": 1.0,
            })
    return group_key, results, {"kept": 1}


@_NEEDS_FORK
def test_run_dataprep_resubmits_incomplete_source_to_a_fresh_process(tmp_path, monkeypatch):
    """issue #3, round two: a source that comes back `incomplete` (soft RSS
    stop, not an error) must be resubmitted -- with its resume_skip/
    resume_seed -- to a brand-new throwaway process, not recorded as done
    and not left for the operator to notice and rerun --resume by hand. This
    is the real orchestration loop end-to-end (continuation_keys/
    resume_state/_submit_respawn), with a real extra fork -- only
    `_run_group_worker` itself is faked, to avoid simulating real
    cross-process RSS growth."""
    monkeypatch.setattr(dp, "_run_group_worker", _incomplete_then_done_group_worker)

    manifest = run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(1),
        skip_decontam=True, rss_soft_limit_gb=3.0, log=lambda *a, **k: None,
    )

    assert len(manifest["sources"]) == 1
    stats = manifest["sources"][0]
    assert "incomplete" not in stats
    assert "error" not in stats
    assert stats["tokens"] == 1000
    assert stats["n_seen"] == 200
    assert manifest["dedup_counters"]["kept"] == 2  # accumulated across both chunks


@_NEEDS_FORK
def test_run_dataprep_shuts_down_main_pool_after_initial_dispatch(tmp_path, monkeypatch):
    """Found live via a 1-group/1-worker validation of the issue #3 respawn
    fix: the main pool `ex` was only ever `shutdown()` in the outer `finally`,
    at the very end of the whole run. Every continuation goes through its own
    throwaway `tw_ex` (see `_submit_respawn`), so `ex` never receives another
    submit() after its initial dispatch -- a worker that finishes its last
    queued group (by completing OR soft-stopping) was sitting idle, holding
    its trimmed-but-nonzero RSS (~2 GB observed live) for the rest of the
    run. With `max_workers=4` in production, that is up to ~8 GB permanently
    stranded on top of the actively-cycling throwaway workers' own budget --
    enough to blow the ADDENDUM 2 RAM ceiling. `ex.shutdown(wait=False)`
    right after the initial dispatch fixes it: it does not cancel the
    already-submitted futures (cancel_futures defaults to False), it only
    tells the pool to retire each worker as soon as its queue drains instead
    of leaving it idle. Verified here by asserting `ex`'s `shutdown()` is
    called before the throwaway pool for the soft-stopped source's
    continuation is even constructed."""
    events = []
    instance_order = []  # first-constructed instance is always `ex`, the main pool

    class RecordingExecutor(dp.ProcessPoolExecutor):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            instance_order.append(self)
            label = "main" if len(instance_order) == 1 else "throwaway"
            events.append((label, "init"))

        def shutdown(self, *a, **k):
            label = "main" if instance_order[0] is self else "throwaway"
            events.append((label, "shutdown"))
            return super().shutdown(*a, **k)

    monkeypatch.setattr(dp, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(dp, "_run_group_worker", _incomplete_then_done_group_worker)

    run_dataprep(
        target_tokens=1000, out_root=str(tmp_path / "shards"),
        manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(1),
        skip_decontam=True, rss_soft_limit_gb=3.0, max_workers=1, log=lambda *a, **k: None,
    )

    main_shutdown_idx = events.index(("main", "shutdown"))
    throwaway_init_idx = events.index(("throwaway", "init"))
    assert main_shutdown_idx < throwaway_init_idx, (
        f"main pool must shut down right after initial dispatch, before the "
        f"respawn's throwaway pool is even created: {events}"
    )


def _resume_state_capturing_group_worker(group_key, spec_keys, target_tokens, out_root,
                                          shard_tokens, max_docs_per_source, eval_ngram_index,
                                          rss_limit_gb=None, rss_soft_limit_gb=None, resume_state=None,
                                          rss_check_every=5_000, checkpoint_every=50_000):
    """Module-level (picklable, forkable) stand-in for `_run_group_worker`
    used to assert what `resume_state` a fresh `--resume` process actually
    receives for a previously-errored, on-disk-recoverable source, without
    having to fork a real worker that streams real data. Reports every key
    "done", folding in whatever `resume_seed` it was handed (if any) so the
    caller can assert progress was carried forward rather than restarted."""
    resume_state = resume_state or {}
    results = []
    for key in spec_keys:
        seed = (resume_state.get(key) or {}).get("resume_seed")
        skip = (resume_state.get(key) or {}).get("resume_skip", 0)
        base_tokens = seed["tokens"] if seed else 0
        results.append({
            "key": key, "dataset": "fake/dataset", "n_seen": skip + 50, "n_kept": skip + 50,
            "n_too_short": 0, "tokens": base_tokens + 500,
            "shards": (seed["shards"] if seed else []) + [{"file": f"{key}_new.bin", "tokens": 500}],
            "manifest_path": os.path.join(out_root, key, "manifest.json"), "elapsed_s": 1.0,
            "token_budget": 1000, "achieved_fraction": 1.0,
        })
    return group_key, results, {"kept": 1}


@_NEEDS_FORK
def test_run_dataprep_resume_recovers_partial_progress_from_errored_manifest_entry(tmp_path, monkeypatch):
    """issue #3, round two, the cross-process half of the fix: four
    dataprep-full attempts in a row dropped every `error`-recorded manifest
    entry on `--resume` and re-streamed those sources from document 0,
    discarding real tokens already flushed to disk as complete shards (see
    COSTS.md). An errored entry with real progress (`tokens>0` and `shards`
    on disk) must instead be seeded into `resume_state_by_key` and handed to
    the fresh run's *initial* group dispatch via the same resume_skip/
    resume_seed shape a live soft-stop uses -- verified here by asserting
    the final manifest's tokens/shards include the pre-existing progress,
    not just what this call itself produced. A sibling errored entry with
    zero progress (nothing ever flushed) must NOT be seeded and is simply
    redone from scratch, same as before this fix."""
    out_root = str(tmp_path / "shards")
    manifest_path = str(tmp_path / "manifest.json")
    os.makedirs(out_root)
    preexisting_manifest = {
        "target_tokens": 2000, "sources": [
            {"key": "src0", "dataset": "fake/dataset0", "n_seen": 40, "n_kept": 40,
             "n_too_short": 0, "tokens": 4000, "elapsed_s": 12.0, "token_budget": 1000,
             "achieved_fraction": 4.0, "shards": [{"file": "src0_00000.bin", "tokens": 4000}],
             "manifest_path": os.path.join(out_root, "src0", "manifest.json"),
             "error": "WorkerMemoryExceeded('worker RSS 4.02 GB exceeded the 4.00 GB cap')"},
            {"key": "src1", "dataset": "fake/dataset1", "n_seen": 0, "n_kept": 0,
             "n_too_short": 0, "tokens": 0, "elapsed_s": 0.1, "token_budget": 1000,
             "achieved_fraction": 0.0, "shards": [],
             "manifest_path": os.path.join(out_root, "src1", "manifest.json"),
             "error": "SomeTransientError()"},
        ],
        "dedup_counters": {"kept": 40, "exact_dup": 0, "near_dup": 0, "contaminated": 0},
    }
    with open(manifest_path, "w") as f:
        json.dump(preexisting_manifest, f)

    monkeypatch.setattr(dp, "_run_group_worker", _resume_state_capturing_group_worker)

    manifest = run_dataprep(
        target_tokens=2000, out_root=out_root, manifest_path=manifest_path,
        mixture=_fake_mixture(2), skip_decontam=True, log=lambda *a, **k: None,
    )

    by_key = {s["key"]: s for s in manifest["sources"]}
    assert "error" not in by_key["src0"]
    assert by_key["src0"]["tokens"] == 4500  # 4000 recovered + 500 new, not just 500 from scratch
    assert by_key["src0"]["n_seen"] == 90  # skip=40 (recovered) + 50 new
    assert {s["file"] for s in by_key["src0"]["shards"]} == {"src0_00000.bin", "src0_new.bin"}

    assert "error" not in by_key["src1"]
    assert by_key["src1"]["tokens"] == 500  # no progress to recover -- ran fresh, skip=0
    assert by_key["src1"]["n_seen"] == 50


@_NEEDS_FORK
def test_run_dataprep_recovers_flushed_shards_when_worker_crash_escapes_internal_handling(
        tmp_path, monkeypatch):
    """Regression test for the eighth incident (see STATUS.md/COSTS.md):
    concurrent.futures marks EVERY still-pending future across the whole
    ProcessPoolExecutor as BrokenProcessPool when any one worker hard-crashes
    (a raw malloc failure, not a catchable Python exception) -- not just the
    future belonging to the process that actually died. A live run had
    stack-edu-python/finephrase/finepdfs-edu reported at 0 tokens despite
    700M/300M/400M real tokens already flushed to disk as complete .bin
    shards. Simulate that: pre-write a real shard file for src0 as if
    run_source had already flushed it, then crash the whole worker (as
    _crashing_group_worker does) and assert the flushed tokens survive in
    the manifest instead of being reported as 0."""
    out_root = str(tmp_path / "shards")
    src0_dir = os.path.join(out_root, "src0")
    os.makedirs(src0_dir)
    with open(os.path.join(src0_dir, "src0_00000.bin"), "wb") as f:
        f.write(b"\x00\x00" * 100)  # 100 uint16 tokens, matching TOKEN_DTYPE

    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))
    monkeypatch.setattr(dp, "_run_group_worker", _crashing_group_worker)

    manifest = run_dataprep(target_tokens=2000, out_root=out_root,
                            manifest_path=str(tmp_path / "manifest.json"),
                            mixture=_fake_mixture(2), skip_decontam=True,
                            log=lambda *a, **k: None)

    by_key = {s["key"]: s for s in manifest["sources"]}
    assert "error" in by_key["src0"]
    assert by_key["src0"]["tokens"] == 100  # recovered from the pre-flushed .bin file
    assert by_key["src0"]["shards"] == [{"file": "src0_00000.bin", "tokens": 100}]
    assert "error" in by_key["src1"]
    assert by_key["src1"]["tokens"] == 0  # nothing was ever flushed for src1


@_NEEDS_FORK
def test_run_dataprep_never_initialises_wandb_in_the_forking_process(tmp_path, monkeypatch):
    """The parent forks workers for the whole life of the run -- once per
    group at dispatch, and again for every within-source RSS respawn. wandb
    starts an asyncio manager on init, and any process forked afterwards
    inherits one belonging to the parent, so touching wandb there raises
    ForkedError. `dataprep-full-attempt5` lost finepdfs-edu, cosmopedia-v2
    and finewiki-en exactly that way: each soft-stopped correctly, and each
    respawn then died in `_run_group_worker`'s setup.

    The old mitigation -- submit the main pool before `wandb.init()` so the
    initial workers fork while the parent is clean -- could only ever cover
    the initial dispatch; respawns fork later by construction. So the parent
    must never construct a real `WandbLogger` at all: it writes to a file and
    a `subprocess`-launched child owns the W&B run. This asserts that
    directly, rather than relying on this offline suite reproducing a real
    fork-safety failure."""
    import daedalus.wandb_logger as wandb_logger
    import daedalus.wandb_sidecar as wandb_sidecar

    inits = []

    class ExplodingLogger:
        def __init__(self, *a, **k):
            inits.append(k)
            raise AssertionError("run_dataprep must not initialise wandb in the forking process")

    monkeypatch.setattr(wandb_logger, "WandbLogger", ExplodingLogger)
    monkeypatch.setattr(wandb_sidecar, "WandbSidecar", _NoopWandb)
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))

    manifest = run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                             manifest_path=str(tmp_path / "manifest.json"),
                             mixture=_fake_mixture(2), skip_decontam=True,
                             log=lambda *a, **k: None)

    assert len(manifest["sources"]) == 2
    assert inits == []


@_NEEDS_FORK
def test_run_dataprep_skip_decontam_avoids_eval_import(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))

    def explode(**kwargs):
        raise AssertionError("_build_eval_index must not be called when skip_decontam=True")

    monkeypatch.setattr(dp, "_build_eval_index", explode)
    manifest = run_dataprep(target_tokens=500, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"),
                            mixture=_fake_mixture(1), skip_decontam=True,
                            log=lambda *a, **k: None)
    assert manifest["decontam"] == "skipped (skip_decontam=True)"
    assert manifest["decontam_index"] is None


# ------------------------------------------------- the frozen decontam index ---

def _frozen_index(tmp_path, texts=("alpha " * 20,), complete=True):
    """A real written index, so these tests exercise the loader rather than a
    stub of it -- the digest check is the thing under test."""
    from daedalus.data import build_eval_ngram_index
    from daedalus.eval_index import EXPECTED_ITEMS, index_digest, write_index

    ngrams = build_eval_ngram_index(list(texts), n=13)
    provenance = {
        "schema": 1, "n": 13, "limit": None if complete else 2000,
        "complete": complete, "ngrams": len(ngrams),
        "digest": index_digest(ngrams), "built_at": "2026-08-26T00:00:00Z",
        # Real item counts and real splits: `run_dataprep` runs the coverage
        # check against today's tasks, so a stub with round numbers in it would
        # be refused for the right reason and prove nothing about the wiring.
        "tasks": {name: {"items": EXPECTED_ITEMS[name], "candidates": 4 * EXPECTED_ITEMS[name],
                         "split": split, "repo": f"org/{name}", "config": None,
                         "revision": None}
                  for name, split in [("hellaswag", "validation"),
                                      ("arc_easy", "test"),
                                      ("piqa", "validation"),
                                      ("openbookqa", "test"),
                                      ("winogrande", "validation")]},
    }
    path = str(tmp_path / "eval-index.txt.gz")
    write_index(path, ngrams, provenance, allow_partial=not complete)
    return path, provenance


@_NEEDS_FORK
def test_a_frozen_index_is_used_and_recorded_in_the_manifest(tmp_path, monkeypatch):
    """The point of the whole exercise: after the run, the manifest says which
    eval items the corpus was filtered against. The released corpus cannot."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))
    monkeypatch.setattr(dp, "_build_eval_index", lambda **kw: (_ for _ in ()).throw(
        AssertionError("must not build an index when a frozen one was given")))
    path, provenance = _frozen_index(tmp_path)

    manifest = run_dataprep(target_tokens=500, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"),
                            mixture=_fake_mixture(1), eval_index_path=path,
                            eval_index_digest=provenance["digest"],
                            log=lambda *a, **k: None)

    record = manifest["decontam_index"]
    assert record["digest"] == provenance["digest"]
    assert record["path"] == path
    assert record["complete"] is True
    assert record["splits"]["arc_easy"] == "test"
    assert record["items"]["hellaswag"] == 10_042


@_NEEDS_FORK
def test_an_index_that_is_not_the_one_named_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))
    path, _ = _frozen_index(tmp_path)
    from daedalus.eval_index import IndexDigestMismatch

    with pytest.raises(IndexDigestMismatch):
        run_dataprep(target_tokens=500, out_root=str(tmp_path / "shards"),
                     manifest_path=str(tmp_path / "manifest.json"),
                     mixture=_fake_mixture(1), eval_index_path=path,
                     eval_index_digest="sha256:" + "0" * 64,
                     log=lambda *a, **k: None)


@_NEEDS_FORK
def test_a_partial_frozen_index_stops_the_run(tmp_path, monkeypatch):
    """Freezing an index does not make it complete. An index that covers a
    fifth of HellaSwag is exactly what this phase is removing, so handing one
    to the rebuild has to fail rather than be recorded and used."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))
    path, _ = _frozen_index(tmp_path, complete=False)

    with pytest.raises(ValueError, match="scored on"):
        run_dataprep(target_tokens=500, out_root=str(tmp_path / "shards"),
                     manifest_path=str(tmp_path / "manifest.json"),
                     mixture=_fake_mixture(1), eval_index_path=path,
                     log=lambda *a, **k: None)


@_NEEDS_FORK
def test_the_in_process_index_is_recorded_as_partial(tmp_path, monkeypatch):
    """The unchanged default still runs, and still writes a manifest -- but one
    that cannot later be mistaken for a complete-index build."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))
    monkeypatch.setattr(dp, "_build_eval_index", lambda **kw: {"a b c"})

    manifest = run_dataprep(target_tokens=500, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"),
                            mixture=_fake_mixture(1), log=lambda *a, **k: None)

    record = manifest["decontam_index"]
    assert record["complete"] is False and record["limit"] == 2000
    assert record["digest"] is None


@_NEEDS_FORK
def test_resuming_records_the_index_this_run_used_not_the_previous_one(tmp_path, monkeypatch):
    """A resume that switches to the frozen index is how the phase-7 rebuild
    continues a corpus started under the old default. The manifest has to
    describe the filter in force, or one file claims two different ones."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))
    monkeypatch.setattr(dp, "_build_eval_index", lambda **kw: {"a b c"})
    manifest_path = str(tmp_path / "manifest.json")

    run_dataprep(target_tokens=500, out_root=str(tmp_path / "shards"),
                 manifest_path=manifest_path, mixture=_fake_mixture(1),
                 log=lambda *a, **k: None)
    path, provenance = _frozen_index(tmp_path)
    manifest = run_dataprep(target_tokens=500, out_root=str(tmp_path / "shards"),
                            manifest_path=manifest_path, mixture=_fake_mixture(1),
                            resume=True, eval_index_path=path,
                            log=lambda *a, **k: None)

    assert manifest["decontam_index"]["digest"] == provenance["digest"]


# ------------------------------------------------------- RAM discipline (ADDENDUM 2) ---

def test_set_worker_memory_limit_applies_rlimit_as(monkeypatch):
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda which, limits: calls.append((which, limits)))

    _set_worker_memory_limit(2.5)

    assert len(calls) == 1
    which, (soft, hard) = calls[0]
    assert which == resource.RLIMIT_AS
    assert soft == hard == int(2.5 * (1024 ** 3))


def test_set_worker_memory_limit_noop_when_falsy(monkeypatch):
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda which, limits: calls.append((which, limits)))

    _set_worker_memory_limit(None)
    _set_worker_memory_limit(0)

    assert calls == []


def test_set_worker_memory_limit_swallows_sandbox_rejection(monkeypatch):
    """Some sandboxes forbid lowering RLIMIT_AS -- must not crash the worker."""
    def boom(which, limits):
        raise OSError("not permitted")
    monkeypatch.setattr(resource, "setrlimit", boom)

    _set_worker_memory_limit(2.5)  # must not raise


@_NEEDS_FORK
def test_run_dataprep_aborts_cleanly_on_low_memory(tmp_path, monkeypatch):
    """ADDENDUM 2 rule 4: if available memory drops below the floor, abort
    in-flight groups cleanly and record why, instead of continuing toward a
    hang the operator would have to reboot."""
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))

    class FakeVM:
        available = 1 * (1024 ** 3)  # 1 GB: under any sane min_available_gb floor

    monkeypatch.setattr(dp.psutil, "virtual_memory", lambda: FakeVM())
    terminated = []
    monkeypatch.setattr(dp, "_terminate_children", lambda log: terminated.append(True))

    manifest = run_dataprep(
        target_tokens=2000, out_root=str(tmp_path / "shards"),
        manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(3),
        skip_decontam=True, log=lambda *a, **k: None,
        min_available_gb=6.0, mem_poll_interval_s=0.01,
    )

    assert manifest["aborted_low_memory"] is True
    assert "aborted_reason" in manifest
    assert terminated == [True]


class _FakeProc:
    def __init__(self, rss_gb):
        self._rss = rss_gb * (1024 ** 3)

    def memory_info(self):
        class M:
            pass
        m = M()
        m.rss = self._rss
        return m


def test_check_worker_rss_raises_when_over_limit(monkeypatch):
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(5.0))
    with pytest.raises(dp.WorkerMemoryExceeded):
        dp._check_worker_rss(2.0)


def test_check_worker_rss_noop_under_limit_or_disabled(monkeypatch):
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(1.0))
    dp._check_worker_rss(2.0)  # under limit: no raise
    dp._check_worker_rss(None)  # disabled: no raise


def test_check_worker_rss_trims_allocator_before_measuring(monkeypatch):
    """See STATUS.md's fineweb-edu/finepdfs-edu/finemath-3plus incident:
    `_check_worker_rss` must trim glibc's arena before measuring, since a
    reclaimable fragmentation spike (not real usage) was tripping the cap and
    abandoning healthy sources at 2-11% of their token budget."""
    calls = []
    monkeypatch.setattr(dp, "_malloc_trim", lambda: calls.append(True))
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(1.0))
    dp._check_worker_rss(2.0)
    assert calls == [True]


def test_malloc_trim_swallows_missing_libc_symbol(monkeypatch):
    """Must not raise on a libc without `malloc_trim` (e.g. musl) -- this
    runs on every RSS check, so a crash here would take down every worker."""
    import ctypes

    class _NoTrimLibc:
        pass  # no malloc_trim attribute

    monkeypatch.setattr(ctypes, "CDLL", lambda name: _NoTrimLibc())
    dp._malloc_trim()  # must not raise


def test_documents_truncates_pathologically_long_text(monkeypatch):
    """A live probe against real finepdfs-edu data (STATUS.md) found a single
    huge document costing hundreds of MB of permanent worker RSS through
    tokenizer/minhash processing. `_documents` must cap text length before
    yielding so no single document can dominate memory or a training window."""
    huge = "word " * (dp._MAX_DOC_CHARS)  # far longer than the cap in chars
    normal = "short document text"
    monkeypatch.setattr(dp, "_stream_rows", lambda s: iter([{"text": huge}, {"text": normal}]))
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])

    docs = list(dp._documents(spec, max_docs=None))

    assert len(docs[0]) == dp._MAX_DOC_CHARS
    assert docs[1] == normal  # untouched: well under the cap


def test_run_source_manifests_partial_shards_on_worker_memory_exceeded(tmp_path, monkeypatch):
    """A worker RSS-cap trip mid-stream must not orphan shards already
    flushed to disk: run_source catches it internally, still closes the
    writer and writes a manifest for whatever's there, and records the error
    in the returned stats instead of raising. Caught live: a real dataprep
    run tripped this on the critical-path fineweb-edu source after 3 full
    shards (300M tokens) were already flushed, and the old
    raise-and-discard behavior would have orphaned them (no manifest.json ->
    unusable by MixtureBatchSource/make_holdout_split). See STATUS.md."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(50))
    monkeypatch.setattr(dp.psutil, "Process", lambda: _FakeProc(99.0))

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, rss_limit_gb=2.0,
                       rss_check_every=1, log=lambda *a, **k: None)

    assert "WorkerMemoryExceeded" in stats["error"]
    assert stats["tokens"] > 0  # the first row was flushed before the cap tripped
    with open(stats["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["total_tokens"] == stats["tokens"]


@_NEEDS_FORK
def test_run_dataprep_does_not_abort_when_memory_is_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))

    class FakeVM:
        available = 20 * (1024 ** 3)  # comfortably above the floor

    monkeypatch.setattr(dp.psutil, "virtual_memory", lambda: FakeVM())

    manifest = run_dataprep(
        target_tokens=2000, out_root=str(tmp_path / "shards"),
        manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(2),
        skip_decontam=True, log=lambda *a, **k: None,
        min_available_gb=6.0, mem_poll_interval_s=0.01,
    )

    assert "aborted_low_memory" not in manifest
    assert len(manifest["sources"]) == 2


# ------------------------------------------------------------------- wandb ---

class _RecordingWandb:
    """Overrides the module's autouse `_no_real_wandb` stub for tests that
    need to inspect what run_dataprep actually logs."""

    instances = []

    def __init__(self, *a, **k):
        self.init_kwargs = k
        self.logs = []
        self.finished = False
        _RecordingWandb.instances.append(self)

    def log(self, record, step=None):
        self.logs.append(record)

    def run_url(self, *a, **k):
        return None

    def finish(self):
        self.finished = True


@_NEEDS_FORK
def test_run_dataprep_logs_progress_to_wandb(tmp_path, monkeypatch):
    import daedalus.wandb_sidecar as wandb_sidecar

    _RecordingWandb.instances = []
    monkeypatch.setattr(wandb_sidecar, "WandbSidecar", _RecordingWandb)
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))

    run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(2),
                skip_decontam=True, log=lambda *a, **k: None,
                mem_poll_interval_s=0.01)

    assert len(_RecordingWandb.instances) == 1
    wb = _RecordingWandb.instances[0]
    assert wb.init_kwargs["tags"] == ["dataprep"]
    assert wb.finished is True
    assert wb.logs  # at least one progress tick was logged
    last = wb.logs[-1]
    assert last["total_tokens"] > 0
    assert last["groups_remaining"] == 0
    assert any("tree_rss_gb" in rec for rec in wb.logs)


@_NEEDS_FORK
def test_run_dataprep_wandb_disabled_still_completes(tmp_path, monkeypatch):
    import daedalus.wandb_sidecar as wandb_sidecar

    _RecordingWandb.instances = []
    monkeypatch.setattr(wandb_sidecar, "WandbSidecar", _RecordingWandb)
    monkeypatch.setattr(dp, "get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(200))

    manifest = run_dataprep(target_tokens=2000, out_root=str(tmp_path / "shards"),
                            manifest_path=str(tmp_path / "manifest.json"), mixture=_fake_mixture(2),
                            skip_decontam=True, log=lambda *a, **k: None,
                            wandb_enabled=False, mem_poll_interval_s=0.01)

    assert len(manifest["sources"]) == 2
    # The sidecar is still constructed (enabled=False is handled inside it,
    # same as train.py's pattern) -- but real code takes the enabled=False
    # no-op path internally when it's the real class, not this recorder.
    assert _RecordingWandb.instances[0].init_kwargs["enabled"] is False


# --------------------------------------------------- absolute budgets ---
#
# The corpus this project actually built is not proportional to the
# blueprint's shares: the math and finephrase sources overshot while
# fineweb-edu/dclm-baseline starved (issue #4 section 4.2). Finishing it
# means topping specific sources up to specific *absolute* token counts,
# which `share * target_tokens` alone cannot express.


def test_build_budget_mixture_reproduces_absolute_budgets_exactly():
    budgets = {"fineweb-edu": 3_750_000_000, "dclm-baseline": 2_050_000_000,
               "finewiki-en": 410_000_000}
    mixture, target = dp.build_budget_mixture(budgets)
    assert target == sum(budgets.values())
    # This is the identity every downstream consumer uses (_run_group_worker).
    got = {s.key: int(round(s.share * target)) for s in mixture}
    assert got == budgets


def test_build_budget_mixture_shares_sum_to_one():
    mixture, _ = dp.build_budget_mixture(
        {"fineweb-edu": 1, "dclm-baseline": 2, "cosmopedia-v2": 3})
    assert abs(sum(s.share for s in mixture) - 1.0) < 1e-9


def test_build_budget_mixture_drops_unlisted_sources_and_keeps_mixture_order():
    mixture, _ = dp.build_budget_mixture(
        {"finewiki-en": 10, "fineweb-edu": 20})
    # Order follows MIXTURE, not the dict, so grouping/dispatch is stable.
    assert [s.key for s in mixture] == ["fineweb-edu", "finewiki-en"]


def test_build_budget_mixture_preserves_spec_fields_that_matter():
    """`share` is the only field that may change. In particular
    `near_dup_group` must survive, or fineweb-edu and dclm-baseline would
    stop cross-deduplicating against each other (~32% overlap)."""
    mixture, _ = dp.build_budget_mixture(
        {"fineweb-edu": 1, "dclm-baseline": 1, "everyday-conversations": 1})
    by_key = {s.key: s for s in mixture}
    orig = {s.key: s for s in dp.MIXTURE}
    assert by_key["fineweb-edu"].near_dup_group == by_key["dclm-baseline"].near_dup_group
    for k, s in by_key.items():
        assert s.dataset == orig[k].dataset
        assert s.config == orig[k].config
        assert s.filter_fn is orig[k].filter_fn
        assert s.text_fn is orig[k].text_fn
        assert s.max_epochs == orig[k].max_epochs
        assert s.load_kwargs == orig[k].load_kwargs


def test_build_budget_mixture_rejects_unknown_and_empty_and_negative():
    with pytest.raises(ValueError, match="unknown source key"):
        dp.build_budget_mixture({"not-a-source": 1})
    with pytest.raises(ValueError, match="at least one source"):
        dp.build_budget_mixture({})
    with pytest.raises(ValueError, match="non-negative"):
        dp.build_budget_mixture({"fineweb-edu": -1})
    with pytest.raises(ValueError, match="positive number of tokens"):
        dp.build_budget_mixture({"fineweb-edu": 0})


def test_build_budget_mixture_allows_zero_budget_for_an_already_complete_source():
    """A source already at/over target is passed with its on-disk count so it
    no-ops on --resume while still being recovered into the run manifest.
    Zero is legal for one source as long as the total is positive."""
    mixture, target = dp.build_budget_mixture(
        {"fineweb-edu": 1_000, "everyday-conversations": 0})
    assert target == 1_000
    got = {s.key: int(round(s.share * target)) for s in mixture}
    assert got["everyday-conversations"] == 0


@_NEEDS_FORK
def test_run_dataprep_resume_retargets_the_manifest(tmp_path):
    """A resumed manifest carries the previous run's `target_tokens`. A run
    that deliberately retargets the corpus (45B -> 13B balanced) must not keep
    reporting the abandoned goal as its denominator."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "target_tokens": 45_000_000_000,
        "sources": [{"key": "fineweb-edu", "dataset": "x", "tokens": 5, "shards": ["a"]}],
    }))
    # The entry carries no `error`, so it lands in `done_keys` and no group is
    # dispatched -- this exercises the resume/retarget path with no streaming.
    out = dp.run_dataprep(
        target_tokens=13_000_000_000, out_root=str(tmp_path / "shards"),
        manifest_path=str(manifest_path), skip_decontam=True, resume=True,
        mixture=[s for s in dp.MIXTURE if s.key == "fineweb-edu"],
        max_workers=1, wandb_enabled=False, log=lambda *a, **k: None,
    )
    assert out["target_tokens"] == 13_000_000_000
    assert json.loads(manifest_path.read_text())["target_tokens"] == 13_000_000_000


@_NEEDS_FORK
def test_run_dataprep_resume_clears_a_previous_runs_abort_flag(tmp_path):
    """`aborted_low_memory`/`aborted_reason` describe the run that wrote the
    manifest, and only the abort path sets them -- but resume loaded them back
    and the success path re-serialized them untouched, so a single low-memory
    abort marked the manifest aborted permanently.

    Real consequence: attempt 10 finished all ten sources
    ("=== done fineweb-edu: tokens=3,750,002,609 ===") and still wrote
    `aborted_low_memory: true`, with a reason naming "5 group(s) still
    running" when it had run with --max-workers 2. Nothing reads the field in
    code, so the whole cost is a false signal to the next reader."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "target_tokens": 13_000_000_000,
        "aborted_low_memory": True,
        "aborted_reason": "available memory 4.6 GB dropped below the 6.0 GB floor",
        "sources": [{"key": "fineweb-edu", "dataset": "x", "tokens": 5, "shards": ["a"]}],
    }))
    out = dp.run_dataprep(
        target_tokens=13_000_000_000, out_root=str(tmp_path / "shards"),
        manifest_path=str(manifest_path), skip_decontam=True, resume=True,
        mixture=[s for s in dp.MIXTURE if s.key == "fineweb-edu"],
        max_workers=1, wandb_enabled=False, log=lambda *a, **k: None,
    )
    assert "aborted_low_memory" not in out
    assert "aborted_reason" not in out
    on_disk = json.loads(manifest_path.read_text())
    assert "aborted_low_memory" not in on_disk
    assert "aborted_reason" not in on_disk


# ------------------------------------------- address space vs resident ---
#
# Attempt 8 raised the per-worker RSS cap 4.0 -> 8.0 GB while the RLIMIT_AS
# backstop stayed pinned at 12.0 GB. Workers hit the address-space cap long
# before their resident budget and died on a raw Rust
# `memory allocation of 1048576 bytes failed` -- an uncatchable crash that
# poisoned the whole pool and marked all six in-flight groups
# `BrokenProcessPool` at once. The two guards must not contradict each other.


def test_vmem_cap_scales_with_the_resident_cap():
    assert dp.worker_vmem_cap_gb(8.0) >= 8.0 + 12.0
    assert dp.worker_vmem_cap_gb(16.0) >= 16.0 + 12.0
    # Always strictly above the resident budget, with real headroom for the
    # ~7.5 GB of non-resident address space torch/tokenizers/pyarrow reserve.
    for rss in (1.0, 2.5, 4.0, 8.0, 12.0, 32.0):
        assert dp.worker_vmem_cap_gb(rss) > rss + 7.5, rss


def test_vmem_cap_never_drops_below_the_import_floor():
    """`import torch; import transformers` alone reserves ~4.3 GB of address
    space, so a small RSS budget must not shrink the cap below what merely
    starting up costs."""
    assert dp.worker_vmem_cap_gb(0.5) >= dp._WORKER_VMEM_HARD_CAP_GB
    # Disabled/absent resident cap falls back to the bare floor.
    assert dp.worker_vmem_cap_gb(None) == dp._WORKER_VMEM_HARD_CAP_GB
    assert dp.worker_vmem_cap_gb(0) == dp._WORKER_VMEM_HARD_CAP_GB


def test_group_worker_sets_a_vmem_cap_matched_to_its_rss_cap(monkeypatch):
    """The regression that actually bit: `_run_group_worker` called
    `_set_worker_memory_limit()` with no argument, so the cap ignored
    `rss_limit_gb` entirely."""
    seen = {}
    monkeypatch.setattr(dp, "_set_worker_memory_limit",
                        lambda limit_gb=None: seen.setdefault("limit", limit_gb))
    monkeypatch.setattr(dp, "get_tokenizer", lambda: (_ for _ in ()).throw(
        RuntimeError("stop after the limit is set")))
    monkeypatch.setattr(dp, "_ACTIVE_MIXTURE_BY_KEY",
                        {s.key: s for s in dp.MIXTURE})
    dp._run_group_worker("g", [], 1, "/tmp/nonexistent-out", 1, None, None,
                         rss_limit_gb=8.0)
    assert seen["limit"] == dp.worker_vmem_cap_gb(8.0)
    assert seen["limit"] >= 20.0


def test_tokens_on_disk_counts_uint16_shard_bytes(tmp_path):
    """`total_tokens` only moves when a source finishes, so a 10 h build shows
    a flat line on the operator's phone -- indistinguishable from a hang."""
    root = tmp_path / "shards"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "a_00000.bin").write_bytes(b"\x00" * 200)   # 100 tokens
    (root / "a" / "a_00001.bin").write_bytes(b"\x00" * 100)   # 50 tokens
    (root / "b" / "b_00000.bin").write_bytes(b"\x00" * 40)    # 20 tokens
    # Non-shard files and loose files at the root must not be counted.
    (root / "a" / "manifest.json").write_text("{}")
    (root / "uploaded.json").write_text("{}")
    assert dp._tokens_on_disk(str(root)) == 170


def test_tokens_on_disk_never_raises_on_a_missing_or_racing_tree(tmp_path):
    """It only reports; it must never take down the run it is reporting on."""
    assert dp._tokens_on_disk(str(tmp_path / "does-not-exist")) == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert dp._tokens_on_disk(str(empty)) == 0


# ------------------------------------------- mid-source durable progress ---
#
# The per-source manifest is the only record that survives a hard worker death
# or a clean low-memory abort, and it used to be written exactly once -- when
# the source *stopped*. That was tolerable while a 3.0 GB soft RSS limit made
# stops frequent; raising it to 7.0 GB to cure the respawn thrashing silently
# removed the checkpoint cadence along with it. A clean abort ~50 min into
# attempt 9 threw away every token fineweb-edu had written since it started:
# 8 shards on disk, manifest still recording 7.


def test_run_source_records_a_resume_position_before_the_source_ends(tmp_path, monkeypatch):
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    seen_manifests = []

    real_write = dp._write_source_manifest

    def spy(writer, spec_, state, stats, provenance=None):
        path = real_write(writer, spec_, state, stats, provenance)
        with open(path) as f:
            seen_manifests.append(json.load(f))
        return path

    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(400))
    monkeypatch.setattr(dp, "_write_source_manifest", spy)

    stats = run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
                       dedup=DedupState(), eval_ngram_index=None, shard_tokens=10**9,
                       checkpoint_every=100, log=lambda *a, **k: None)

    # 400 documents at a 100-doc interval -> several mid-source checkpoints
    # plus the final one, not just the final one.
    assert len(seen_manifests) > 1, "no mid-source checkpoint was written"
    mid = seen_manifests[0]
    assert mid["n_seen"] == 100
    assert mid["n_seen"] < stats["n_seen"]


def test_mid_source_checkpoint_describes_exactly_what_is_on_disk(tmp_path, monkeypatch):
    """The buffer must be flushed before the manifest is written, or the
    recorded token count would include tokens no reader can find -- and a
    resume from that position would silently overstate the corpus."""
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    snapshots = []
    real_write = dp._write_source_manifest

    def spy(writer, spec_, state, stats, provenance=None):
        path = real_write(writer, spec_, state, stats, provenance)
        with open(path) as f:
            m = json.load(f)
        on_disk = sum(os.path.getsize(os.path.join(str(tmp_path), s["file"])) // 2
                      for s in m["shards"])
        snapshots.append((m["total_tokens"], on_disk, stats["tokens"]))
        return path

    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(300))
    monkeypatch.setattr(dp, "_write_source_manifest", spy)
    run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
               dedup=DedupState(), eval_ngram_index=None, shard_tokens=10**9,
               checkpoint_every=100, log=lambda *a, **k: None)

    assert snapshots
    for manifest_tokens, bytes_on_disk, stats_tokens in snapshots:
        assert manifest_tokens == bytes_on_disk, "manifest counts unflushed tokens"
        assert manifest_tokens == stats_tokens, "manifest and stats disagree"


def test_checkpoint_every_zero_restores_the_old_write_once_behaviour(tmp_path, monkeypatch):
    spec = SourceSpec("fake-src", "fake/dataset", share=1.0, text_fn=lambda r: r["text"])
    calls = []
    real_write = dp._write_source_manifest
    monkeypatch.setattr(dp, "_stream_rows", lambda s: fake_rows(300))
    monkeypatch.setattr(dp, "_write_source_manifest",
                        lambda *a: (calls.append(1), real_write(*a))[1])
    run_source(spec, FakeTokenizer(), token_budget=10**9, out_dir=str(tmp_path),
               dedup=DedupState(), eval_ngram_index=None, shard_tokens=10**9,
               checkpoint_every=0, log=lambda *a, **k: None)
    assert len(calls) == 1


def test_shard_writer_flush_partial_makes_the_buffer_durable(tmp_path):
    from daedalus.data import ShardWriter
    w = ShardWriter(out_dir=str(tmp_path), shard_tokens=10**9, prefix="s")
    w.write([1, 2, 3, 4])
    assert w.shards == [] and w.total_tokens == 0     # still buffered
    w.flush_partial()
    assert len(w.shards) == 1 and w.total_tokens == 4
    assert os.path.getsize(os.path.join(str(tmp_path), w.shards[0]["file"])) == 8
    w.flush_partial()                                  # idempotent when empty
    assert len(w.shards) == 1
    w.close()
    assert len(w.shards) == 1


def test_group_worker_passes_checkpoint_every_through_to_run_source(monkeypatch):
    seen = {}
    monkeypatch.setattr(dp, "_set_worker_memory_limit", lambda *a, **k: None)
    monkeypatch.setattr(dp, "get_tokenizer", lambda: object())
    monkeypatch.setattr(dp, "_ACTIVE_MIXTURE_BY_KEY",
                        {s.key: s for s in dp.MIXTURE})
    monkeypatch.setattr(dp, "run_source",
                        lambda *a, **k: (seen.update(k), {"key": "fineweb-edu",
                                                          "tokens": 0})[1])
    dp._run_group_worker("g", ["fineweb-edu"], 1000, "/tmp/x", 1, None, None,
                         rss_limit_gb=4.0, checkpoint_every=1234)
    assert seen["checkpoint_every"] == 1234
