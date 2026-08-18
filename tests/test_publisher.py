"""Tests for daedalus/publisher.py -- the final-model publish path.

Fully offline: the Hub API is faked, so no network and no real uploads. The
point of this file is that the deliverable (weights + tokenizer + model card +
GGUFs) has a tested route off this box, and that the route refuses to ship an
artifact nobody could identify later.

Run: python -m pytest tests/test_publisher.py -v
"""
import json
import os

import pytest

from daedalus import publisher


class FakeApi:
    """Stands in for huggingface_hub.HfApi, in the style of
    tests/test_ckpt_uploader.py's fake."""

    def __init__(self, token=None, fail_folder=False):
        self.token = token
        self.created = []
        self.folders = []
        self.files = []
        self.commit_messages = []
        self._fail_folder = fail_folder

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
        self.created.append({"repo_id": repo_id, "repo_type": repo_type,
                             "private": private, "exist_ok": exist_ok})

    def upload_folder(self, folder_path=None, repo_id=None, repo_type=None,
                      commit_message=None):
        if self._fail_folder:
            raise ConnectionError("hub unreachable")
        self.folders.append((folder_path, repo_id, repo_type))
        self.commit_messages.append(commit_message)

    def upload_file(self, path_or_fileobj=None, path_in_repo=None,
                    repo_id=None, repo_type=None, commit_message=None):
        self.files.append((path_in_repo, repo_id, repo_type))
        self.commit_messages.append(commit_message)


def make_model_dir(tmp_path, name="hf", weights="model.safetensors",
                   omit=()):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, body in (("config.json", '{"model_type": "lfm2"}'),
                           ("README.md", "# card\n"),
                           ("tokenizer.json", "{}")):
        if filename in omit:
            continue
        (d / filename).write_text(body)
    if weights and weights not in omit:
        (d / weights).write_bytes(b"\x00" * 16)
    return str(d)


# ------------------------------------------------------------ the guard rail ---

def test_a_complete_directory_is_publishable(tmp_path):
    assert publisher.check_publishable(make_model_dir(tmp_path)) == []


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json"])
def test_missing_essentials_block_the_publish(tmp_path, missing):
    model_dir = make_model_dir(tmp_path, omit=(missing,))
    problems = publisher.check_publishable(model_dir)
    assert any(missing in p for p in problems)
    with pytest.raises(ValueError):
        publisher.publish_model(model_dir, "me/x", token="t", api=FakeApi())


def test_a_model_with_no_card_is_refused(tmp_path):
    """The card is required, not nice-to-have. An artifact with no provenance,
    no stated success bar and no way to tell a 13B-token run from a 40B one is
    what the model-card work existed to stop shipping."""
    model_dir = make_model_dir(tmp_path, omit=("README.md",))
    assert publisher.check_publishable(model_dir) == ["missing README.md"]
    with pytest.raises(ValueError, match="README.md"):
        publisher.publish_model(model_dir, "me/x", token="t", api=FakeApi())


def test_weights_are_required_and_either_format_counts(tmp_path):
    empty = make_model_dir(tmp_path, name="empty", weights=None)
    assert any("no weights" in p for p in publisher.check_publishable(empty))
    for weights in ("model.safetensors", "pytorch_model.bin"):
        d = make_model_dir(tmp_path, name=f"d-{weights}", weights=weights)
        assert publisher.check_publishable(d) == []


def test_a_missing_directory_is_reported_not_crashed(tmp_path):
    problems = publisher.check_publishable(str(tmp_path / "nope"))
    assert len(problems) == 1 and "not a directory" in problems[0]


# ---------------------------------------------------------------- publishing ---

def test_publish_uploads_the_folder_then_the_ggufs(tmp_path):
    model_dir = make_model_dir(tmp_path)
    gguf = tmp_path / "model-q4_0.gguf"
    gguf.write_bytes(b"\x00" * 32)
    api = FakeApi()

    result = publisher.publish_model(model_dir, "me/daedalus-150m",
                                     gguf_paths=[str(gguf)], token="t", api=api)

    assert api.created[0]["repo_type"] == "model"
    assert api.folders == [(model_dir, "me/daedalus-150m", "model")]
    # GGUFs land under gguf/ rather than the repo root, so they cannot collide
    # with the HF weights the same repo serves to transformers.
    assert api.files == [("gguf/model-q4_0.gguf", "me/daedalus-150m", "model")]
    assert result["ggufs"] == [{"name": "model-q4_0.gguf", "size": 32}]
    assert result["url"] == "https://huggingface.co/me/daedalus-150m"


def test_repos_are_created_private_unless_asked_otherwise(tmp_path):
    """Publishing is outward-facing and hard to walk back."""
    model_dir = make_model_dir(tmp_path)
    api = FakeApi()
    publisher.publish_model(model_dir, "me/x", token="t", api=api)
    assert api.created[0]["private"] is True

    api2 = FakeApi()
    publisher.publish_model(model_dir, "me/x", token="t", private=False, api=api2)
    assert api2.created[0]["private"] is False


def test_publishing_without_a_token_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN_WRITE", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN_WRITE"):
        publisher.publish_model(make_model_dir(tmp_path), "me/x", api=FakeApi())


def test_a_failed_folder_upload_propagates(tmp_path):
    """Unlike ckpt_uploader, which must never raise inside a training loop,
    this is the last thing between a finished model and nobody having it."""
    with pytest.raises(ConnectionError):
        publisher.publish_model(make_model_dir(tmp_path), "me/x", token="t",
                                api=FakeApi(fail_folder=True))


def test_a_missing_gguf_is_skipped_not_fatal(tmp_path):
    model_dir = make_model_dir(tmp_path)
    api = FakeApi()
    result = publisher.publish_model(
        model_dir, "me/x", gguf_paths=[str(tmp_path / "absent.gguf")],
        token="t", api=api)
    assert result["ggufs"] == []
    assert api.folders  # the model still published


# ------------------------------------------------------------------ find_ggufs ---

def test_find_ggufs_collects_across_dirs_and_dedupes_by_name(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    (a / "model-f16.gguf").write_bytes(b"1")
    (a / "model-q4_0.gguf").write_bytes(b"1")
    (b / "model-q4_0.gguf").write_bytes(b"1")   # same name, later dir loses
    (a / "notes.txt").write_bytes(b"1")

    found = publisher.find_ggufs(str(a), str(b), str(tmp_path / "missing"), "")
    assert [os.path.basename(p) for p in found] == ["model-f16.gguf",
                                                    "model-q4_0.gguf"]
    assert all(p.startswith(str(a)) for p in found)


def test_find_ggufs_does_not_recurse(tmp_path):
    """A run directory holds checkpoints and intermediates; walking it would
    sweep up whatever else the job wrote."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.gguf").write_bytes(b"1")
    assert publisher.find_ggufs(str(tmp_path)) == []


# ------------------------------------------------------------------------ cli ---

def test_check_only_reports_without_uploading(tmp_path, capsys):
    model_dir = make_model_dir(tmp_path)
    assert publisher._cli(["--model-dir", model_dir, "--check-only"]) == 0
    assert json.loads(capsys.readouterr().out)["publishable"] is True


def test_check_only_exits_nonzero_on_an_unpublishable_dir(tmp_path, capsys):
    model_dir = make_model_dir(tmp_path, omit=("README.md",))
    assert publisher._cli(["--model-dir", model_dir, "--check-only"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["publishable"] is False and out["problems"] == ["missing README.md"]


def test_cli_defaults_to_private_and_honours_public(tmp_path, monkeypatch):
    model_dir = make_model_dir(tmp_path)
    seen = {}

    def fake_publish(md, repo, **kw):
        seen.update(kw)
        return {"repo_id": repo}

    monkeypatch.setattr(publisher, "publish_model", fake_publish)
    publisher._cli(["--model-dir", model_dir])
    assert seen["private"] is True
    publisher._cli(["--model-dir", model_dir, "--public"])
    assert seen["private"] is False
