"""Tests for daedalus/shard_uploader.py. Fully offline -- the Hub API is
faked, so no network calls and no real uploads.

Run: python -m pytest tests/test_shard_uploader.py -v
"""
import json
import os

import pytest

from daedalus import shard_uploader as su


class FakeApi:
    """Stands in for huggingface_hub.HfApi. Records what was uploaded, and
    can be told to fail specific paths so the retry contract is testable."""

    def __init__(self, token=None, fail_paths=(), fail_create=False):
        self.token = token
        self.uploaded = []
        self.created = []
        self._fail_paths = set(fail_paths)
        self._fail_create = fail_create

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
        if self._fail_create:
            raise ConnectionError("hub unreachable")
        self.created.append((repo_id, repo_type, private))

    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None,
                     repo_type=None):
        if path_in_repo in self._fail_paths:
            raise OSError(f"upload failed for {path_in_repo}")
        self.uploaded.append(path_in_repo)


@pytest.fixture
def fake_api(monkeypatch):
    """Patches the HfApi symbol the module imports lazily inside upload_once."""
    holder = {}

    def install(**kwargs):
        api = FakeApi(**kwargs)
        holder["api"] = api
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: api)
        return api

    return install


def _make_source(root, key, shard_sizes, sealed=True):
    """A source directory with `.bin` files and (optionally) the per-source
    manifest that seals them."""
    source_dir = os.path.join(root, key)
    os.makedirs(source_dir, exist_ok=True)
    shards = []
    for i, n_tokens in enumerate(shard_sizes):
        name = f"{key}_{i:05d}.bin"
        with open(os.path.join(source_dir, name), "wb") as f:
            f.write(b"\x00\x00" * n_tokens)
        shards.append({"file": name, "tokens": n_tokens})
    if sealed:
        with open(os.path.join(source_dir, "manifest.json"), "w") as f:
            json.dump({"shards": shards, "total_tokens": sum(shard_sizes)}, f)
    return source_dir


# ------------------------------------------------------------ discovery ---

def test_pending_lists_every_sealed_shard(tmp_path):
    _make_source(str(tmp_path), "srcA", [10, 20])
    _make_source(str(tmp_path), "srcB", [5])
    pending = su.pending_shards(str(tmp_path), {"files": {}})
    assert sorted((k, n) for k, n, _ in pending) == [
        ("srcA", "srcA_00000.bin"), ("srcA", "srcA_00001.bin"), ("srcB", "srcB_00000.bin"),
    ]


def test_pending_ignores_shards_no_manifest_has_sealed_yet(tmp_path):
    """A `.bin` still being written by ShardWriter is not in any manifest
    yet. Uploading it would push a truncated shard."""
    _make_source(str(tmp_path), "srcA", [10], sealed=False)
    assert su.pending_shards(str(tmp_path), {"files": {}}) == []


def test_pending_ignores_a_manifest_being_rewritten(tmp_path):
    """run_source rewrites the per-source manifest at every chunk boundary,
    so a pass can catch it mid-write. That must be skipped, not fatal."""
    source_dir = _make_source(str(tmp_path), "srcA", [10])
    with open(os.path.join(source_dir, "manifest.json"), "w") as f:
        f.write('{"shards": [')  # truncated mid-write
    assert su.pending_shards(str(tmp_path), {"files": {}}) == []


def test_pending_skips_already_uploaded_shards(tmp_path):
    _make_source(str(tmp_path), "srcA", [10, 20])
    state = {"files": {"srcA/srcA_00000.bin": 20}}  # 10 tokens = 20 bytes
    pending = su.pending_shards(str(tmp_path), state)
    assert [n for _, n, _ in pending] == ["srcA_00001.bin"]


def test_pending_reuploads_a_shard_whose_size_changed(tmp_path):
    """A source redone from scratch reuses shard filenames with different
    content. Identity is (name, size), so a changed shard is re-uploaded
    rather than assumed current."""
    _make_source(str(tmp_path), "srcA", [10])
    state = {"files": {"srcA/srcA_00000.bin": 999}}
    pending = su.pending_shards(str(tmp_path), state)
    assert [n for _, n, _ in pending] == ["srcA_00000.bin"]


def test_pending_on_a_missing_root_is_empty_not_an_error(tmp_path):
    assert su.pending_shards(str(tmp_path / "nope"), {"files": {}}) == []


# -------------------------------------------------------------- uploads ---

def test_upload_once_uploads_shards_and_their_manifests(tmp_path, fake_api):
    api = fake_api()
    _make_source(str(tmp_path), "srcA", [10, 20])

    summary = su.upload_once(str(tmp_path), "me/corpus", token="t", log=lambda *a: None)

    assert summary["uploaded"] == 2
    assert summary["failed"] == 0
    assert summary["pending_after"] == 0
    assert api.uploaded == ["srcA/srcA_00000.bin", "srcA/srcA_00001.bin", "srcA/manifest.json"]
    assert api.created == [("me/corpus", "dataset", True)]


def test_upload_once_is_incremental_across_passes(tmp_path, fake_api):
    api = fake_api()
    _make_source(str(tmp_path), "srcA", [10])
    su.upload_once(str(tmp_path), "me/corpus", token="t", log=lambda *a: None)
    api.uploaded.clear()

    # A second pass with nothing new must upload nothing at all -- the whole
    # point of tracking files rather than calling upload_folder each time.
    summary = su.upload_once(str(tmp_path), "me/corpus", token="t", log=lambda *a: None)
    assert summary["uploaded"] == 0
    assert api.uploaded == []

    _make_source(str(tmp_path), "srcA", [10, 20])  # a new shard was sealed
    summary = su.upload_once(str(tmp_path), "me/corpus", token="t", log=lambda *a: None)
    assert summary["uploaded"] == 1
    assert "srcA/srcA_00001.bin" in api.uploaded


def test_upload_once_leaves_a_failed_file_pending_and_keeps_going(tmp_path, fake_api):
    api = fake_api(fail_paths=["srcA/srcA_00000.bin"])
    _make_source(str(tmp_path), "srcA", [10, 20])

    summary = su.upload_once(str(tmp_path), "me/corpus", token="t", log=lambda *a: None)

    assert summary["uploaded"] == 1
    assert summary["failed"] == 1
    assert summary["pending_after"] == 1  # the failed one, retried next pass
    assert "srcA/srcA_00001.bin" in api.uploaded


def test_upload_once_survives_an_unreachable_hub(tmp_path, fake_api):
    fake_api(fail_create=True)
    _make_source(str(tmp_path), "srcA", [10])

    summary = su.upload_once(str(tmp_path), "me/corpus", token="t", log=lambda *a: None)

    assert summary["uploaded"] == 0
    assert summary["failed"] == 1  # nothing raised out


def test_upload_state_survives_an_interrupted_pass(tmp_path, fake_api):
    """State is persisted after every file, so an interrupted pass does not
    re-push the gigabytes it already sent."""
    api = fake_api(fail_paths=["srcA/srcA_00002.bin"])
    _make_source(str(tmp_path), "srcA", [10, 20, 30])
    su.upload_once(str(tmp_path), "me/corpus", token="t", log=lambda *a: None)

    state = su._load_state(os.path.join(str(tmp_path), su.STATE_FILENAME))
    assert set(state["files"]) == {"srcA/srcA_00000.bin", "srcA/srcA_00001.bin"}


def test_upload_state_file_corrupt_is_treated_as_empty(tmp_path):
    (tmp_path / su.STATE_FILENAME).write_text("{not json")
    assert su._load_state(str(tmp_path / su.STATE_FILENAME)) == {"files": {}}


def test_max_files_throttles_a_single_pass(tmp_path, fake_api):
    """Bandwidth is shared with dataprep's own streaming downloads, so a pass
    is bounded rather than draining everything at once."""
    api = fake_api()
    _make_source(str(tmp_path), "srcA", [10, 20, 30, 40])

    summary = su.upload_once(str(tmp_path), "me/corpus", token="t", max_files=2,
                              log=lambda *a: None)

    assert summary["uploaded"] == 2
    assert summary["pending_after"] == 2


# ---------------------------------------------------------------- watch ---

def test_watch_polls_until_max_passes_without_sleeping_after_the_last(tmp_path, fake_api):
    fake_api()
    _make_source(str(tmp_path), "srcA", [10, 20, 30])
    sleeps = []

    totals = su.watch(str(tmp_path), "me/corpus", token="t", interval_s=123,
                       max_files_per_pass=1, max_passes=3, log=lambda *a: None,
                       sleep=sleeps.append)

    assert totals["uploaded"] == 3
    assert totals["passes"] == 3
    assert sleeps == [123, 123]  # slept between passes, not after the final one
