"""Tests for the provenance eval.py records beside every score.

A five-task mean is compared against 47.31 with a half-point gate. Whether that
comparison is valid depends entirely on facts the number itself does not carry:
which checkpoint bytes were loaded, which tokenizer scored them, whether
`val_bpb` was a full pass or the default 100-batch sample, which dataset
revision and split each task came from, and how many items each task actually
contributed. Every one of those has silently changed a Daedalus number before --
the ARC-Easy/OpenBookQA split bug is written up in eval.py's own TASK_SPLITS
comment -- so they are recorded, not assumed.
"""

import json

import pytest

import eval as eval_module


class _FakeTokenizer:
    def __init__(self, vocab=None, specials=("<|endoftext|>",)):
        self._vocab = vocab or {"a": 0, "b": 1}
        self.all_special_tokens = list(specials)
        self.vocab_size = len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)


# ------------------------------------------------------------------ digests ---

def test_tokenizer_digest_is_stable_and_order_independent():
    left = _FakeTokenizer({"a": 0, "b": 1})
    right = _FakeTokenizer({"b": 1, "a": 0})

    assert eval_module.tokenizer_digest(left) == eval_module.tokenizer_digest(right)
    assert len(eval_module.tokenizer_digest(left)) == 64


def test_tokenizer_digest_changes_when_the_vocabulary_changes():
    baseline = eval_module.tokenizer_digest(_FakeTokenizer({"a": 0, "b": 1}))
    altered = eval_module.tokenizer_digest(_FakeTokenizer({"a": 0, "b": 2}))

    assert baseline != altered


def test_tokenizer_digest_changes_when_special_tokens_change():
    baseline = eval_module.tokenizer_digest(_FakeTokenizer())
    altered = eval_module.tokenizer_digest(
        _FakeTokenizer(specials=("<|endoftext|>", "<|im_start|>")))

    assert baseline != altered


def test_file_digest_hashes_bytes(tmp_path):
    target = tmp_path / "ckpt.pt"
    target.write_bytes(b"weights")

    assert len(eval_module.file_digest(target)) == 64
    assert eval_module.file_digest(target) == eval_module.file_digest(target)


def test_file_digest_of_a_missing_file_is_recorded_as_unavailable(tmp_path):
    assert eval_module.file_digest(tmp_path / "absent.pt") is None


# -------------------------------------------------------- task provenance ---

def _tracing_fallback(rows, repo="ybisk/piqa", revision=None):
    """A stand-in that honours `_load_with_fallback`'s recording contract.

    The real function appends the repo/config/split/revision it resolved to
    `_LOAD_TRACE` at the moment it succeeds; a double that returns rows without
    recording would let `load_all_tasks` look correct here while producing empty
    provenance in production.
    """
    def fallback(candidates, split):
        eval_module._LOAD_TRACE.append(
            {"repo": repo, "config": None, "split": split, "revision": revision})
        return rows, repo
    return fallback


def test_real_load_with_fallback_records_what_it_resolved(monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset",
                        lambda repo, *args, **kwargs: ["row"])
    eval_module._LOAD_TRACE.clear()

    eval_module._load_with_fallback([("ybisk/piqa", None)], "validation")

    assert eval_module._LOAD_TRACE[-1] == {
        "repo": "ybisk/piqa", "config": None, "split": "validation",
        "revision": None}


def test_load_all_tasks_records_the_repo_split_and_item_count(monkeypatch):
    rows = [{"goal": "g", "sol1": "a", "sol2": "b", "label": 0}]
    monkeypatch.setattr(eval_module, "TASK_LOADERS",
                        {"piqa": eval_module.load_piqa})
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        _tracing_fallback(rows))

    sources = {}
    tasks = eval_module.load_all_tasks(limit=None, sources=sources)

    assert len(tasks["piqa"]) == 1
    assert sources["piqa"]["repo"] == "ybisk/piqa"
    assert sources["piqa"]["split"] == "validation"
    assert sources["piqa"]["n"] == 1
    assert sources["piqa"]["limit"] is None
    # Also published module-side, which is how `_cli` reads it without
    # changing `load_all_tasks`'s call signature.
    assert eval_module.LAST_TASK_SOURCES["piqa"] == sources["piqa"]


def test_task_provenance_records_the_limit_that_was_applied(monkeypatch):
    rows = [{"goal": f"g{i}", "sol1": "a", "sol2": "b", "label": 0}
            for i in range(5)]
    monkeypatch.setattr(eval_module, "TASK_LOADERS",
                        {"piqa": eval_module.load_piqa})
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        _tracing_fallback(rows))

    sources = {}
    eval_module.load_all_tasks(limit=2, sources=sources)

    # A limited run and a full-split run are not comparable, and the item count
    # is the only thing that says which happened.
    assert sources["piqa"]["limit"] == 2
    assert sources["piqa"]["n"] == 2


def test_load_all_tasks_clears_stale_sources_between_runs(monkeypatch):
    rows = [{"goal": "g", "sol1": "a", "sol2": "b", "label": 0}]
    monkeypatch.setattr(eval_module, "TASK_LOADERS",
                        {"piqa": eval_module.load_piqa})
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        _tracing_fallback(rows))
    eval_module.load_all_tasks()

    monkeypatch.setattr(eval_module, "TASK_LOADERS", {})
    eval_module.load_all_tasks()

    # A second run must not inherit the first run's task list, or a results
    # file would claim provenance for benchmarks it never scored.
    assert eval_module.LAST_TASK_SOURCES == {}


def test_load_all_tasks_leaves_a_failing_task_out_of_provenance(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr(eval_module, "TASK_LOADERS", {"broken": broken})

    sources = {}
    tasks = eval_module.load_all_tasks(sources=sources)

    assert tasks == {}
    assert sources == {}


# ------------------------------------------------------ provenance record ---

def test_provenance_record_distinguishes_a_bounded_sample_from_a_full_pass():
    sampled = eval_module.provenance_record(
        checkpoints=["a.pt"], config="daedalus-150m", tokenizer=_FakeTokenizer(),
        task_sources={}, shard_dir="data/holdout", bpb_max_batches=100,
        bpb_batch_size=8, seq_len=2048, seed=7)
    full = eval_module.provenance_record(
        checkpoints=["a.pt"], config="daedalus-150m", tokenizer=_FakeTokenizer(),
        task_sources={}, shard_dir="data/holdout", bpb_max_batches=None,
        bpb_batch_size=8, seq_len=2048, seed=7)

    assert sampled["bpb"]["mode"] == "sample"
    assert sampled["bpb"]["max_batches"] == 100
    assert full["bpb"]["mode"] == "full"
    assert full["bpb"]["max_batches"] is None


def test_provenance_record_marks_bpb_not_applicable_without_a_shard_dir():
    record = eval_module.provenance_record(
        checkpoints=["a.pt"], config="daedalus-150m", tokenizer=_FakeTokenizer(),
        task_sources={}, shard_dir=None, bpb_max_batches=100,
        bpb_batch_size=8, seq_len=2048, seed=7)

    assert record["bpb"]["mode"] == "not-applicable"


def test_provenance_record_hashes_every_scored_checkpoint(tmp_path):
    first = tmp_path / "one.pt"
    second = tmp_path / "two.pt"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    record = eval_module.provenance_record(
        checkpoints=[str(first), str(second)], config="daedalus-150m",
        tokenizer=_FakeTokenizer(), task_sources={}, shard_dir=None,
        bpb_max_batches=None, bpb_batch_size=8, seq_len=2048, seed=7)

    digests = {entry["path"]: entry["sha256"] for entry in record["checkpoints"]}
    assert digests[str(first)] != digests[str(second)]
    assert all(len(value) == 64 for value in digests.values())


def test_provenance_record_carries_seed_config_and_tokenizer_digest():
    tokenizer = _FakeTokenizer()

    record = eval_module.provenance_record(
        checkpoints=[], config="daedalus-150m", tokenizer=tokenizer,
        task_sources={"piqa": {"repo": "ybisk/piqa", "split": "validation",
                               "n": 1838, "limit": None}},
        shard_dir=None, bpb_max_batches=None, bpb_batch_size=8, seq_len=2048,
        seed=20260824)

    assert record["seed"] == 20260824
    assert record["config"] == "daedalus-150m"
    assert record["tokenizer"]["sha256"] == eval_module.tokenizer_digest(tokenizer)
    assert record["tasks"]["piqa"]["n"] == 1838
    assert record["schema"] == eval_module.PROVENANCE_SCHEMA
    assert "created_at" in record and record["created_at"].endswith("Z")


def test_provenance_record_names_a_peer_model_instead_of_a_checkpoint():
    record = eval_module.provenance_record(
        checkpoints=[], config="daedalus-150m", tokenizer=_FakeTokenizer(),
        task_sources={}, shard_dir=None, bpb_max_batches=None,
        bpb_batch_size=8, seq_len=2048, seed=1, hf_model="EleutherAI/pythia-160m")

    assert record["hf_model"] == "EleutherAI/pythia-160m"
    assert record["checkpoints"] == []


# ------------------------------------------------------------------ writing ---

def test_write_results_embeds_provenance_in_both_files(tmp_path):
    provenance = {"schema": 1, "seed": 3}
    results = tmp_path / "results.json"
    items = tmp_path / "results.items.json"

    eval_module.write_results(results, items,
                              per_checkpoint=[{"checkpoint": "a.pt"}],
                              mean={"piqa": 0.5}, per_item={"a.pt": {}},
                              provenance=provenance, task_limit=None)

    payload = json.loads(results.read_text())
    sidecar = json.loads(items.read_text())
    assert payload["provenance"] == provenance
    # The sidecar is what a paired comparison reads; it must be able to prove
    # on its own which run produced it.
    assert sidecar["provenance"] == provenance


def test_write_results_creates_missing_directories(tmp_path):
    target = tmp_path / "nested" / "deep" / "results.json"

    eval_module.write_results(target, None, per_checkpoint=[], mean={},
                              per_item={}, provenance={"schema": 1},
                              task_limit=None)

    assert target.exists()


def test_write_results_skips_the_sidecar_when_there_are_no_items(tmp_path):
    results = tmp_path / "results.json"
    items = tmp_path / "results.items.json"

    eval_module.write_results(results, items, per_checkpoint=[], mean={},
                              per_item={}, provenance={"schema": 1},
                              task_limit=None)

    assert results.exists()
    assert not items.exists()
