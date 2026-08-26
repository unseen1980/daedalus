"""Tests for daedalus/data.py. Fully offline: no HF network calls -- tokenizer
loading, dataset streaming, and Hub upload/download are exercised only via
their local (non-network) building blocks (ShardWriter, PackedTokenDataset,
dedup/decontam) using a fake tokenizer.

Run: python -m pytest tests/test_data.py -v
"""
import json
import os

import numpy as np
import pytest
import torch

from daedalus.data import (
    DEFAULT_EOS_ID,
    NearDupFilter,
    PackedTokenDataset,
    ShardWriter,
    build_eval_ngram_index,
    is_contaminated,
    minhash_signature,
    ngram_set,
    shingles,
    tokenize_and_pack,
    tokenize_document,
)


class FakeTokenizer:
    """Deterministic word->id tokenizer, no network, no vocab file."""

    def encode(self, text):
        return [abs(hash(w)) % 1000 + 10 for w in text.split()]


# ------------------------------------------------------------- ShardWriter ---

def test_shard_writer_flushes_at_shard_size(tmp_path):
    w = ShardWriter(str(tmp_path), shard_tokens=10)
    w.write(list(range(25)))
    w.close()
    assert len(w.shards) == 3  # 10, 10, 5
    assert w.total_tokens == 25
    assert [s["tokens"] for s in w.shards] == [10, 10, 5]
    for s in w.shards:
        assert os.path.exists(os.path.join(tmp_path, s["file"]))


def test_shard_writer_empty_buffer_no_extra_shard(tmp_path):
    w = ShardWriter(str(tmp_path), shard_tokens=5)
    w.write(list(range(10)))  # exactly 2 shards, nothing left over
    w.close()
    assert len(w.shards) == 2


def test_shard_writer_manifest_roundtrip(tmp_path):
    w = ShardWriter(str(tmp_path), shard_tokens=4)
    w.write(list(range(9)))
    w.close()
    path = w.write_manifest({"note": "x"})
    with open(path) as f:
        manifest = json.load(f)
    assert manifest["total_tokens"] == 9
    assert manifest["note"] == "x"
    assert manifest["dtype"] == "uint16"


def test_shard_writer_resume_from_continues_shard_numbering(tmp_path):
    """dataprep.py's within-source RSS-respawn continuation seeds a new
    ShardWriter from a prior chunk's already-flushed shards -- it must
    append (`_00002`, `_00003`, ...), not restart at `_00000` and clobber
    the earlier chunk's files."""
    w1 = ShardWriter(str(tmp_path), shard_tokens=10, prefix="src")
    w1.write(list(range(20)))  # 2 full shards: src_00000.bin, src_00001.bin
    w1.close()
    prior_shard_files = {s["file"] for s in w1.shards}

    w2 = ShardWriter(str(tmp_path), shard_tokens=10, prefix="src",
                      resume_from={"shards": w1.shards, "total_tokens": w1.total_tokens})
    w2.write(list(range(20, 35)))  # one more full shard + a 5-token partial
    w2.close()

    assert [s["file"] for s in w2.shards] == [
        "src_00000.bin", "src_00001.bin", "src_00002.bin", "src_00003.bin"]
    assert w2.total_tokens == 35
    for name in prior_shard_files:  # original chunk's files untouched
        assert os.path.exists(os.path.join(tmp_path, name))
    arr = np.fromfile(os.path.join(tmp_path, "src_00003.bin"), dtype=np.uint16)
    assert len(arr) == 5


def test_shard_writer_resume_from_none_is_original_behavior(tmp_path):
    w = ShardWriter(str(tmp_path), shard_tokens=10, prefix="src", resume_from=None)
    assert w.shards == []
    assert w.total_tokens == 0
    assert w._shard_idx == 0


def test_shard_dtype_is_uint16_on_disk(tmp_path):
    w = ShardWriter(str(tmp_path), shard_tokens=100)
    w.write([1, 2, 49151])
    w.close()
    arr = np.fromfile(os.path.join(tmp_path, w.shards[0]["file"]), dtype=np.uint16)
    np.testing.assert_array_equal(arr, [1, 2, 49151])


# ---------------------------------------------------------- tokenize_and_pack ---

def test_tokenize_document_appends_eos():
    ids = tokenize_document(FakeTokenizer(), "hello world", eos_id=DEFAULT_EOS_ID)
    assert ids[-1] == DEFAULT_EOS_ID
    assert len(ids) == 3


def test_tokenize_and_pack_writes_manifest_and_shards(tmp_path):
    docs = ["the quick brown fox", "jumps over", "the lazy dog and then some more words here"]
    result = tokenize_and_pack(FakeTokenizer(), docs, str(tmp_path),
                               eos_id=DEFAULT_EOS_ID, shard_tokens=6)
    assert result["n_documents"] == 3
    expected_tokens = sum(len(d.split()) + 1 for d in docs)  # +1 EOS each
    assert result["total_tokens"] == expected_tokens
    with open(os.path.join(tmp_path, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["eos_id"] == DEFAULT_EOS_ID
    assert sum(s["tokens"] for s in manifest["shards"]) == expected_tokens


def test_tokenize_and_pack_is_dense_no_padding(tmp_path):
    """Concatenating all shard bytes reproduces the flat tokenized+EOS stream
    exactly -- no padding tokens inserted between documents."""
    docs = ["a b c", "d e"]
    tok = FakeTokenizer()
    expected = []
    for d in docs:
        expected += tokenize_document(tok, d, DEFAULT_EOS_ID)
    tokenize_and_pack(tok, docs, str(tmp_path), eos_id=DEFAULT_EOS_ID, shard_tokens=1000)
    with open(os.path.join(tmp_path, "manifest.json")) as f:
        manifest = json.load(f)
    all_tokens = []
    for s in manifest["shards"]:
        all_tokens += list(np.fromfile(os.path.join(tmp_path, s["file"]), dtype=np.uint16))
    assert all_tokens == expected


# -------------------------------------------------------- PackedTokenDataset ---

def _write_shards(tmp_path, tokens, shard_tokens, eos_id=DEFAULT_EOS_ID):
    w = ShardWriter(str(tmp_path), shard_tokens=shard_tokens)
    w.write(tokens)
    w.close()
    w.write_manifest({"eos_id": eos_id})
    return str(tmp_path)


def test_packed_dataset_length_and_window_shape(tmp_path):
    tokens = list(range(100))
    d = _write_shards(tmp_path, tokens, shard_tokens=100)
    ds = PackedTokenDataset(d, seq_len=10)
    assert len(ds) == 100 - 10 + 1
    x = ds[0]
    assert x.shape == (10,)
    assert x.dtype == torch.int64


def test_packed_dataset_windows_are_correct(tmp_path):
    """A window is a single seq_len-length sequence -- Daedalus.forward takes
    one array as both input_ids and targets and shifts internally (model.py),
    so the dataset must NOT pre-split into an (input, target) pair."""
    tokens = list(range(50))
    d = _write_shards(tmp_path, tokens, shard_tokens=50)
    ds = PackedTokenDataset(d, seq_len=8)
    x = ds[3]
    assert x.tolist() == tokens[3:11]


def test_packed_dataset_spans_multiple_shards_contiguously(tmp_path):
    """Windows must be able to read across a shard boundary transparently."""
    tokens = list(range(30))
    d = _write_shards(tmp_path, tokens, shard_tokens=20)  # shard0: 0-19, shard1: 20-29
    ds = PackedTokenDataset(d, seq_len=5)
    # last valid window in shard0 starts at offset 15 (15..19 inclusive, 5 tokens)
    total_shard0_windows = 20 - 5 + 1
    x = ds[total_shard0_windows - 1]
    assert x.tolist() == tokens[15:20]


def test_packed_dataset_index_out_of_range_raises(tmp_path):
    tokens = list(range(20))
    d = _write_shards(tmp_path, tokens, shard_tokens=20)
    ds = PackedTokenDataset(d, seq_len=5)
    with pytest.raises(IndexError):
        ds[len(ds)]
    with pytest.raises(IndexError):
        ds[-1]


def test_packed_dataset_doc_ids_increment_after_eos(tmp_path):
    # doc0 = [10, 11], EOS(2), doc1 = [12, 13, 14], EOS(2)
    tokens = [10, 11, 2, 12, 13, 14, 2]
    d = _write_shards(tmp_path, tokens, shard_tokens=100, eos_id=2)
    ds = PackedTokenDataset(d, seq_len=7, return_doc_ids=True)
    x, doc_ids = ds[0]
    assert x.tolist() == [10, 11, 2, 12, 13, 14, 2]
    # doc id increments the token *after* each EOS, not the EOS itself
    assert doc_ids.tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_packed_dataset_stride_seq_len_gives_nonoverlapping_windows(tmp_path):
    """eval.py's full-holdout bpb pass needs stride=seq_len so __len__ covers
    the shard once instead of exploding by ~seq_len x (every sliding offset)."""
    tokens = list(range(100))
    d = _write_shards(tmp_path, tokens, shard_tokens=100)
    ds = PackedTokenDataset(d, seq_len=10, stride=10)
    assert len(ds) == 10  # 100 tokens / 10 per window, no overlap
    assert ds[0].tolist() == tokens[0:10]
    assert ds[1].tolist() == tokens[10:20]
    assert ds[9].tolist() == tokens[90:100]


def test_packed_dataset_stride_drops_remainder(tmp_path):
    tokens = list(range(25))
    d = _write_shards(tmp_path, tokens, shard_tokens=25)
    ds = PackedTokenDataset(d, seq_len=10, stride=10)
    assert len(ds) == 2  # windows at 0 and 10; the trailing 5 tokens don't fill a window
    assert ds[1].tolist() == tokens[10:20]


def test_make_loader_batches_correctly(tmp_path):
    from daedalus.data import make_loader
    tokens = list(range(200))
    d = _write_shards(tmp_path, tokens, shard_tokens=200)
    loader = make_loader(d, seq_len=8, batch_size=4, shuffle=False, num_workers=0)
    xb = next(iter(loader))
    assert xb.shape == (4, 8)


def test_packed_dataset_batch_feeds_model_directly(tmp_path):
    """The [B, seq_len] batch this dataset produces must be usable unmodified
    as both `input_ids` and `targets` to Daedalus.forward (model.py)."""
    from daedalus.config import PRESETS
    from daedalus.data import make_loader
    from daedalus.model import Daedalus
    import copy

    cfg = copy.deepcopy(PRESETS["tiny"])
    tokens = list(np.random.randint(0, cfg.vocab_size, size=200))
    d = _write_shards(tmp_path, tokens, shard_tokens=200)
    loader = make_loader(d, seq_len=16, batch_size=2, shuffle=False, num_workers=0)
    xb = next(iter(loader))

    model = Daedalus(cfg)
    model.train()
    _, loss, _ = model(xb, targets=xb)
    assert loss.item() > 0
    assert torch.isfinite(loss)


# -------------------------------------------------------------- make_loader ---

def test_shuffled_loader_never_materializes_a_full_permutation(tmp_path):
    """The blocker that would have made `hero` and `abl-arch` impossible.

    DataLoader(shuffle=True) installs a RandomSampler that runs
    `torch.randperm(len(ds)).tolist()` before yielding anything. With stride=1
    -- what training uses -- len(ds) is one window per *token*, so that
    permutation is sized by the corpus rather than by the run: measured at
    ~32 bytes/window, 109 GB for fineweb-edu at 3.75B and 418 GB for the full
    14B mixture, on a 30 GB box. It would have thrashed rather than raised.

    Asserting `randperm` is never called pins the mechanism directly, which a
    peak-RSS threshold could only do approximately.
    """
    from daedalus.data import make_loader
    d = _write_shards(tmp_path, list(range(4000)), shard_tokens=4000)

    called = []
    real_randperm = torch.randperm

    def spy(*a, **k):
        called.append(a[0] if a else k.get("n"))
        return real_randperm(*a, **k)

    torch.randperm = spy
    try:
        loader = make_loader(d, seq_len=16, batch_size=2, shuffle=True, num_workers=0)
        for i, batch in enumerate(loader):
            assert batch.shape == (2, 16)
            if i >= 4:
                break
    finally:
        torch.randperm = real_randperm
    assert called == [], f"torch.randperm called with n={called}; permutation materialized"


def test_shuffled_loader_samples_with_replacement_lazily(tmp_path):
    from daedalus.data import make_loader
    d = _write_shards(tmp_path, list(range(4000)), shard_tokens=4000)
    loader = make_loader(d, seq_len=16, batch_size=2, shuffle=True, num_workers=0)
    from torch.utils.data import RandomSampler
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.replacement is True


def test_shuffled_loader_is_deterministic_for_a_fixed_generator(tmp_path):
    """ShardBatchSource seeds a generator so a run is reproducible; that has to
    survive the move off DataLoader's own shuffling."""
    from daedalus.data import make_loader
    d = _write_shards(tmp_path, list(range(4000)), shard_tokens=4000)

    def first_batches(seed):
        g = torch.Generator().manual_seed(seed)
        loader = make_loader(d, seq_len=16, batch_size=2, shuffle=True,
                             num_workers=0, generator=g)
        return [b.clone() for _, b in zip(range(3), loader)]

    a, b, c = first_batches(0), first_batches(0), first_batches(1)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "same seed diverged"
    assert not all(torch.equal(x, y) for x, y in zip(a, c)), "different seeds identical"


def test_shuffled_loader_draws_varied_windows(tmp_path):
    """Sampling with replacement must still cover the shard, not sit on one
    window -- a sampler that returned a constant index would pass the
    no-randperm test above while destroying training."""
    from daedalus.data import make_loader
    d = _write_shards(tmp_path, list(range(4000)), shard_tokens=4000)
    g = torch.Generator().manual_seed(0)
    loader = make_loader(d, seq_len=16, batch_size=4, shuffle=True,
                         num_workers=0, generator=g)
    starts = set()
    for i, batch in enumerate(loader):
        starts.update(batch[:, 0].tolist())
        if i >= 20:
            break
    assert len(starts) > 20, f"only {len(starts)} distinct window starts drawn"


def test_unshuffled_loader_still_reads_in_order(tmp_path):
    """eval.py's bounded val_bpb pass depends on sequential order; the shuffle
    branch must not have changed it."""
    from daedalus.data import make_loader
    d = _write_shards(tmp_path, list(range(4000)), shard_tokens=4000)
    loader = make_loader(d, seq_len=16, batch_size=2, shuffle=False,
                         num_workers=0, stride=16)
    first = next(iter(loader))
    assert first[0, 0].item() == 0
    assert first[1, 0].item() == 16


# --------------------------------------------------------- make_holdout_split ---

def test_make_holdout_split_reserves_whole_shards_by_default_frac(tmp_path):
    from daedalus.data import make_holdout_split
    tokens = list(range(500))  # 5 shards of 100 tokens each
    d = _write_shards(tmp_path / "src", tokens, shard_tokens=100)
    train_dir = str(tmp_path / "train")
    holdout_dir = str(tmp_path / "holdout")
    result = make_holdout_split(d, train_dir, holdout_dir, holdout_frac=0.02)
    # target = 500*0.02=10 tokens, so exactly one (100-token) shard is reserved
    assert len(result["holdout"]["shards"]) == 1
    assert len(result["train"]["shards"]) == 4
    assert result["train"]["total_tokens"] + result["holdout"]["total_tokens"] == 500


def test_make_holdout_split_reserves_more_shards_for_larger_frac(tmp_path):
    from daedalus.data import make_holdout_split
    tokens = list(range(500))
    d = _write_shards(tmp_path / "src", tokens, shard_tokens=100)
    result = make_holdout_split(d, str(tmp_path / "train"), str(tmp_path / "holdout"),
                                holdout_frac=0.25)
    assert len(result["holdout"]["shards"]) == 2  # 100 tokens < 125 target < 200
    assert len(result["train"]["shards"]) == 3


def test_make_holdout_split_never_empties_train_split(tmp_path):
    from daedalus.data import make_holdout_split
    tokens = list(range(300))
    d = _write_shards(tmp_path / "src", tokens, shard_tokens=100)
    result = make_holdout_split(d, str(tmp_path / "train"), str(tmp_path / "holdout"),
                                holdout_frac=0.99)
    assert len(result["train"]["shards"]) == 1
    assert len(result["holdout"]["shards"]) == 2


def test_make_holdout_split_raises_on_single_shard_source(tmp_path):
    from daedalus.data import make_holdout_split
    tokens = list(range(50))
    d = _write_shards(tmp_path / "src", tokens, shard_tokens=100)
    with pytest.raises(ValueError):
        make_holdout_split(d, str(tmp_path / "train"), str(tmp_path / "holdout"))


def test_make_holdout_split_shards_are_disjoint_and_windows_load_correctly(tmp_path):
    from daedalus.data import make_holdout_split, PackedTokenDataset
    tokens = list(range(500))
    d = _write_shards(tmp_path / "src", tokens, shard_tokens=100)
    train_dir = str(tmp_path / "train")
    holdout_dir = str(tmp_path / "holdout")
    make_holdout_split(d, train_dir, holdout_dir, holdout_frac=0.02)

    train_files = {s["file"] for s in json.load(open(os.path.join(train_dir, "manifest.json")))["shards"]}
    holdout_files = {s["file"] for s in json.load(open(os.path.join(holdout_dir, "manifest.json")))["shards"]}
    assert train_files.isdisjoint(holdout_files)

    train_ds = PackedTokenDataset(train_dir, seq_len=10)
    holdout_ds = PackedTokenDataset(holdout_dir, seq_len=10)
    assert len(train_ds) > 0 and len(holdout_ds) > 0
    # holdout is the last shard (tokens 400-499)
    assert holdout_ds[0][0].item() == 400


def test_make_holdout_split_hardlinks_not_copies(tmp_path):
    from daedalus.data import make_holdout_split
    tokens = list(range(500))
    d = _write_shards(tmp_path / "src", tokens, shard_tokens=100)
    train_dir = str(tmp_path / "train")
    holdout_dir = str(tmp_path / "holdout")
    make_holdout_split(d, train_dir, holdout_dir, holdout_frac=0.02)
    src_stat = os.stat(os.path.join(d, "shard_00000.bin"))
    train_stat = os.stat(os.path.join(train_dir, "shard_00000.bin"))
    assert src_stat.st_ino == train_stat.st_ino  # same inode -- hardlinked, not copied


def test_make_holdout_split_relinks_when_split_dir_holds_a_different_corpus(tmp_path):
    """Reusing a split root across two corpora must not keep the old bytes.

    Shard names restart at _00000 for every source, so two different corpora
    have same-named, different-content shards -- exactly the state this box is
    in (data/shards/fineweb-edu/fineweb-edu_00000.bin and
    data/shards-sweep/fineweb-edu/fineweb-edu_00000.bin are distinct inodes).
    The old name-only existence check left the stale file in place and wrote a
    manifest describing the new one, so training read tokens that did not match
    its own manifest with nothing visible in any log.
    """
    from daedalus.data import make_holdout_split, PackedTokenDataset
    train_dir = str(tmp_path / "train")
    holdout_dir = str(tmp_path / "holdout")

    old = _write_shards(tmp_path / "corpus-old", list(range(500)), shard_tokens=100)
    make_holdout_split(old, train_dir, holdout_dir, holdout_frac=0.02)
    assert PackedTokenDataset(train_dir, seq_len=10)[0][0].item() == 0

    # A different corpus, same shard filenames, disjoint token values.
    new = _write_shards(tmp_path / "corpus-new", list(range(1000, 1500)), shard_tokens=100)
    make_holdout_split(new, train_dir, holdout_dir, holdout_frac=0.02)

    for split_dir, src_dir in ((train_dir, new), (holdout_dir, new)):
        manifest = json.load(open(os.path.join(split_dir, "manifest.json")))
        for s in manifest["shards"]:
            dst_stat = os.stat(os.path.join(split_dir, s["file"]))
            src_stat = os.stat(os.path.join(src_dir, s["file"]))
            assert (dst_stat.st_dev, dst_stat.st_ino) == (src_stat.st_dev, src_stat.st_ino), (
                f"{s['file']} in {split_dir} still points at the previous corpus")
    # The bytes a reader actually gets are the new corpus's, not the old one's.
    assert PackedTokenDataset(train_dir, seq_len=10)[0][0].item() == 1000


def test_make_holdout_split_reuses_identical_links_without_relinking(tmp_path):
    """The inode check must not turn every rerun into a relink storm: calling
    it twice on the same source is a no-op, and the inode is preserved."""
    from daedalus.data import make_holdout_split
    d = _write_shards(tmp_path / "src", list(range(500)), shard_tokens=100)
    train_dir = str(tmp_path / "train")
    holdout_dir = str(tmp_path / "holdout")
    make_holdout_split(d, train_dir, holdout_dir, holdout_frac=0.02)
    first = os.stat(os.path.join(train_dir, "shard_00000.bin")).st_ino
    make_holdout_split(d, train_dir, holdout_dir, holdout_frac=0.02)
    assert os.stat(os.path.join(train_dir, "shard_00000.bin")).st_ino == first


# ------------------------------------------------------- make_mixture_holdout_split ---

def test_make_mixture_holdout_split_splits_every_source(tmp_path):
    from daedalus.data import make_mixture_holdout_split
    mixture = tmp_path / "mixture"
    _write_shards(mixture / "source-a", list(range(500)), shard_tokens=100)
    _write_shards(mixture / "source-b", list(range(300)), shard_tokens=100)
    train_root = str(tmp_path / "train")
    holdout_root = str(tmp_path / "holdout")

    result = make_mixture_holdout_split(str(mixture), train_root, holdout_root,
                                        holdout_frac=0.02)

    assert set(result) == {"source-a", "source-b"}
    assert os.path.exists(os.path.join(train_root, "source-a", "manifest.json"))
    assert os.path.exists(os.path.join(holdout_root, "source-b", "manifest.json"))


def test_make_mixture_holdout_split_skips_single_shard_source(tmp_path):
    """A source with only one shard can't be split (make_holdout_split raises)
    -- must be skipped, not crash the whole mixture split."""
    from daedalus.data import make_mixture_holdout_split
    mixture = tmp_path / "mixture"
    _write_shards(mixture / "big", list(range(500)), shard_tokens=100)
    _write_shards(mixture / "tiny", list(range(50)), shard_tokens=100)  # 1 shard
    train_root = str(tmp_path / "train")
    holdout_root = str(tmp_path / "holdout")

    result = make_mixture_holdout_split(str(mixture), train_root, holdout_root)

    assert set(result) == {"big"}
    assert not os.path.exists(os.path.join(train_root, "tiny"))


def test_make_mixture_holdout_split_raises_when_all_sources_too_small(tmp_path):
    from daedalus.data import make_mixture_holdout_split
    mixture = tmp_path / "mixture"
    _write_shards(mixture / "tiny", list(range(50)), shard_tokens=100)
    with pytest.raises(ValueError):
        make_mixture_holdout_split(str(mixture), str(tmp_path / "train"),
                                   str(tmp_path / "holdout"))


def test_make_mixture_holdout_split_raises_on_empty_mixture_root(tmp_path):
    from daedalus.data import make_mixture_holdout_split
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        make_mixture_holdout_split(str(empty), str(tmp_path / "train"),
                                   str(tmp_path / "holdout"))


# ---------------------------------------------------------------- dedup -----

def test_shingles_basic():
    s = shingles("a b c d e f", n=5)
    assert s == ["a b c d e", "b c d e f"]


def test_shingles_short_text_returns_whole_text():
    assert shingles("a b", n=5) == ["a b"]
    assert shingles("", n=5) == []


def test_minhash_signature_near_duplicate_detected():
    a = "the quick brown fox jumps over the lazy dog in the park today"
    b = "the quick brown fox jumps over the lazy dog in the park yesterday"
    c = "completely unrelated text about something else entirely different"
    ma, mb, mc = (minhash_signature(t) for t in (a, b, c))
    assert ma.jaccard(mb) > 0.5
    assert ma.jaccard(mc) < 0.3


def test_near_dup_filter_flags_repeated_text():
    f = NearDupFilter(threshold=0.8)
    text = "the quick brown fox jumps over the lazy dog many times over"
    assert f.is_duplicate(text) is False
    assert f.is_duplicate(text) is True  # exact repeat -> duplicate
    assert f.is_duplicate("something totally different about cooking recipes") is False


# -------------------------------------------------------------- decontam ----

def test_ngram_set_size():
    g = ngram_set("a b c d e", n=3)
    assert g == {"a b c", "b c d", "c d e"}


def test_build_eval_ngram_index_and_contamination():
    eval_texts = ["the mitochondria is the powerhouse of the cell and more words"]
    idx = build_eval_ngram_index(eval_texts, n=8)
    contaminated = "the mitochondria is the powerhouse of the cell and more words here"
    clean = "an entirely different sentence about cooking pasta for dinner tonight"
    assert is_contaminated(contaminated, idx, n=8) is True
    assert is_contaminated(clean, idx, n=8) is False


# ------------------------------------------------- replacement-sampling cost --
# `make_loader(shuffle=True)` samples with replacement (see its docstring for
# why a real permutation is not affordable at stride=1). The docstring used to
# justify that purely by collision rate -- 0.069% of draws for `hero` -- but
# "the same window twice" and "how much of the corpus is seen at all" are
# different questions and only the second one reaches the model. These pin the
# second one so the claim in the docstring cannot rot.

def _poisson_unseen_frac(n_draws: int, seq_len: int, corpus_tokens: int) -> float:
    """A token is covered by any of the `seq_len` windows starting within
    `seq_len` of it, so its coverage count is Poisson with this mean."""
    import math
    lam = n_draws * seq_len / corpus_tokens
    return math.exp(-lam)



def test_the_worst_covered_sources_are_the_ones_with_data_to_spare(tmp_path):
    """The corpus-wide average understates the cost, which is why the docstring
    now leads with the per-source spread. A source's Poisson mean *is* its
    epoch count, and `runs/preflight/epochs-at-60b.md` measures those at 1.71
    to 4.00 -- so the sources the 4-epoch cap never binds on (the two maths
    ones, which have the most disk left over) are covered worst, not best."""
    at = lambda epochs: _poisson_unseen_frac(int(epochs * 1_000_000), 1, 1_000_000)
    assert round(at(4.00) * 100, 2) == 1.83      # 7 of 10 sources, incl. fineweb-edu
    assert round(at(2.60) * 100, 2) == 7.43      # finephrase
    assert round(at(1.71) * 100, 2) == 18.09     # finemath-3plus, infiwebmath-3plus
    assert at(1.71) > 9 * at(4.00), (
        "the least-repeated sources must still read as the worst-covered ones; "
        "if this inverts, the docstring's argument has stopped being true")

    # Coverage improves fast with budget -- what makes this acceptable rather
    # than alarming, and the reason 60B was preferred to a smaller budget.
    assert round(_poisson_unseen_frac(2 * 1000, 1, 1000) * 100, 2) == 13.53
    assert round(_poisson_unseen_frac(4 * 1000, 1, 1000) * 100, 2) == 1.83


def test_replacement_sampling_coverage_matches_the_poisson_prediction(tmp_path):
    """The prediction above is arithmetic; this checks the loader actually
    behaves that way. Token values are `range(N)`, so each drawn window's first
    element *is* its start index -- which lets coverage be reconstructed from
    the batches themselves rather than by reaching into the sampler.
    """
    from daedalus.data import make_loader

    N, SEQ = 20_000, 64
    n_draws = 879                      # -> lam = 879*64/20000 = 2.813, hero's
    d = _write_shards(tmp_path, list(range(N)), shard_tokens=N)

    g = torch.Generator().manual_seed(1234)
    loader = make_loader(d, seq_len=SEQ, batch_size=1, shuffle=True,
                         num_workers=0, num_samples=n_draws, generator=g)

    seen = bytearray(N)
    drawn = 0
    for xb in loader:
        start = int(xb[0, 0].item())   # token value == index, by construction
        seen[start:start + SEQ] = b"\x01" * SEQ
        drawn += 1
    assert drawn == n_draws

    unseen = 1.0 - sum(seen) / N
    predicted = _poisson_unseen_frac(n_draws, SEQ, N)
    assert abs(unseen - predicted) < 0.02, (
        f"observed {unseen:.4f} unseen vs Poisson {predicted:.4f} -- if this "
        f"drifted, make_loader is no longer sampling with replacement")
    # And the point of the whole exercise: it is emphatically not ~0.
    assert unseen > 0.03


# ------------------------------------------------- tokenizer provenance ----
# Added in Phase 4, where four shard directories differing only in vocabulary
# sit side by side. Reading the wrong one is not an error anything raises: the
# dtype, the shape and the manifest schema are identical, and a 32,768-token
# id served to a 49,152-row embedding is simply a different word.

class _FingerprintTokenizer:
    """Enough of the `PreTrainedTokenizer` surface to be fingerprinted."""

    def __init__(self, vocab_size: int, salt: int = 0):
        self.vocab_size = vocab_size
        self.name_or_path = f"fake-v{vocab_size}-s{salt}"
        self._salt = salt

    def encode(self, text, add_special_tokens=True):
        return [(ord(c) + self._salt) % self.vocab_size for c in text]

    def convert_tokens_to_ids(self, token):
        return 0


def test_fingerprint_records_vocab_size_and_a_tokenization_digest():
    from daedalus.data import tokenizer_fingerprint

    fingerprint = tokenizer_fingerprint(_FingerprintTokenizer(32768))
    assert fingerprint["vocab_size"] == 32768
    assert len(fingerprint["probe_digest"]) == 32
    assert "partial" not in fingerprint


def test_two_vocabularies_of_the_same_size_fingerprint_differently():
    """Vocabulary size alone cannot identify a tokenizer -- Phase 4 trains
    candidates a future retrain could easily match the size of."""
    from daedalus.data import tokenizer_fingerprint

    a = tokenizer_fingerprint(_FingerprintTokenizer(32768, salt=0))
    b = tokenizer_fingerprint(_FingerprintTokenizer(32768, salt=1))
    assert a["vocab_size"] == b["vocab_size"]
    assert a["probe_digest"] != b["probe_digest"]


def test_a_minimal_tokenizer_yields_a_partial_fingerprint_not_a_crash():
    """Packing shards must not start requiring a full tokenizer
    implementation; the fingerprint records what it could read and says so."""
    from daedalus.data import tokenizer_fingerprint

    assert tokenizer_fingerprint(FakeTokenizer())["partial"] is True


def test_tokenize_and_pack_records_the_tokenizer_in_the_manifest(tmp_path):
    out = tmp_path / "shards"
    tokenize_and_pack(_FingerprintTokenizer(32768), ["hello world"] * 4,
                      str(out), shard_tokens=64)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tokenizer"]["vocab_size"] == 32768


def test_batched_packing_produces_identical_ids(tmp_path):
    """The batched path exists for speed and must be indistinguishable in
    output, or Phase 4's corpora differ from every other corpus in this
    repository for a reason nobody would find."""

    class _BatchTokenizer(_FingerprintTokenizer):
        def __call__(self, texts):
            return {"input_ids": [self.encode(t) for t in texts]}

    documents = [f"document number {i} with some text" for i in range(37)]
    one_at_a_time = tmp_path / "single"
    batched = tmp_path / "batched"
    tokenize_and_pack(_BatchTokenizer(4096), documents, str(one_at_a_time),
                      shard_tokens=1000)
    tokenize_and_pack(_BatchTokenizer(4096), documents, str(batched),
                      shard_tokens=1000, batch_documents=8)

    for name in ("total_tokens", "n_documents"):
        assert (json.loads((one_at_a_time / "manifest.json").read_text())[name]
                == json.loads((batched / "manifest.json").read_text())[name])
    left = np.fromfile(one_at_a_time / "shard_00000.bin", dtype=np.uint16)
    right = np.fromfile(batched / "shard_00000.bin", dtype=np.uint16)
    assert np.array_equal(left, right)


def test_reading_shards_under_a_different_tokenizer_is_refused():
    from daedalus.data import assert_manifest_tokenizer, tokenizer_fingerprint

    manifest = {"tokenizer": tokenizer_fingerprint(_FingerprintTokenizer(32768))}
    assert_manifest_tokenizer(manifest, _FingerprintTokenizer(32768))   # same
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        assert_manifest_tokenizer(manifest, _FingerprintTokenizer(49152))


def test_a_manifest_without_a_fingerprint_still_loads():
    """Every corpus already on disk was packed before the field existed."""
    from daedalus.data import assert_manifest_tokenizer

    assert_manifest_tokenizer({"total_tokens": 10}, _FingerprintTokenizer(32768))


def _packed_tree(root, name, vocab_size=None, tokens=64):
    """One shard directory, with or without a recorded vocabulary."""
    from daedalus.data import ShardWriter

    out_dir = root / name
    writer = ShardWriter(str(out_dir), shard_tokens=tokens)
    writer.write(list(range(tokens)))
    writer.close()
    extra = {"eos_id": 0, "source_key": name}
    if vocab_size is not None:
        extra["tokenizer"] = {"name": f"fake/tok-{vocab_size}",
                              "vocab_size": vocab_size,
                              "probe_digest": "0" * 32}
    writer.write_manifest(extra)
    return out_dir


def test_a_tree_packed_under_another_vocabulary_is_refused_before_it_trains(
        tmp_path):
    """The reader with no tokenizer to compare against. A 49,152-row model
    reading 32,768-vocabulary shards indexes every id successfully and trains
    on text that means something else -- nothing raises, and the loss curve
    looks like a run that is working."""
    from daedalus.data import assert_shards_vocab_size

    tree = _packed_tree(tmp_path, "code", vocab_size=32_768)
    assert_shards_vocab_size(str(tree), 32_768)
    with pytest.raises(ValueError, match="shard vocabulary mismatch") as excinfo:
        assert_shards_vocab_size(str(tree), 49_152)
    # Both numbers and the directory, because "a mismatch" is not actionable.
    assert "32,768" in str(excinfo.value) and "49,152" in str(excinfo.value)
    assert "'code'" in str(excinfo.value)


def test_one_wrong_source_in_a_mixture_root_is_named(tmp_path):
    """A mixture is wrong per source: two of three can be right, and the run
    would sample the third at its blueprint share without a word."""
    from daedalus.data import assert_shards_vocab_size

    root = tmp_path / "mixture"
    _packed_tree(root, "fineweb-edu", vocab_size=49_152)
    _packed_tree(root, "dclm-baseline", vocab_size=49_152)
    _packed_tree(root, "stack-edu-python", vocab_size=32_768)

    with pytest.raises(ValueError, match="stack-edu-python") as excinfo:
        assert_shards_vocab_size(str(root), 49_152)
    assert "fineweb-edu" not in str(excinfo.value)


def test_a_tree_that_records_no_vocabulary_still_loads(tmp_path):
    """Every corpus on this box predates the fingerprint, including the one
    phase 8 continues from. Unknown must not read as mismatched."""
    from daedalus.data import assert_shards_vocab_size, manifest_vocab_size

    tree = _packed_tree(tmp_path, "legacy")
    assert manifest_vocab_size(json.loads(
        (tree / "manifest.json").read_text())) is None
    assert_shards_vocab_size(str(tree), 49_152)

    root = tmp_path / "mixed"
    _packed_tree(root, "legacy-source")
    _packed_tree(root, "rebuilt-source", vocab_size=49_152)
    assert_shards_vocab_size(str(root), 49_152)


def test_get_tokenizer_default_is_still_smollm2():
    """The explicit-path form must not change what any existing caller gets."""
    import inspect

    from daedalus.data import SMOLLM2_TOKENIZER, get_tokenizer

    assert inspect.signature(get_tokenizer).parameters["name"].default is None
    assert "name or SMOLLM2_TOKENIZER" in inspect.getsource(get_tokenizer)
    assert SMOLLM2_TOKENIZER == "HuggingFaceTB/SmolLM2-135M"
