"""Tests for daedalus/wandb_sidecar.py. Fully offline -- the W&B client is
faked and no child process is spawned except where the test says so.

Run: python -m pytest tests/test_wandb_sidecar.py -v
"""
import json
import os

import pytest

from daedalus import wandb_sidecar as ws


class FakeRun:
    url = "https://wandb.ai/e/p/runs/abc123"


class FakeWandbLogger:
    """Stands in for daedalus.wandb_logger.WandbLogger inside `publish`."""
    instances = []

    def __init__(self, project=None, entity=None, name=None, config=None, tags=None,
                  enabled=True):
        self.project, self.name, self.config, self.tags = project, name, config, tags
        self.run = FakeRun() if enabled else None
        self.logged = []
        self.finished = False
        FakeWandbLogger.instances.append(self)

    def log(self, record, step=None):
        self.logged.append((record, step))

    def finish(self):
        self.finished = True


@pytest.fixture(autouse=True)
def _fake_logger(monkeypatch):
    FakeWandbLogger.instances = []
    import daedalus.wandb_logger as wl
    monkeypatch.setattr(wl, "WandbLogger", FakeWandbLogger)


# ------------------------------------------------------------- the parent ---

def test_disabled_sidecar_starts_no_process_and_swallows_logs(tmp_path):
    wb = ws.WandbSidecar("p", None, "n", {}, enabled=False,
                          progress_path=str(tmp_path / "progress.jsonl"))
    wb.log({"total_tokens": 1})
    wb.finish()
    assert wb.proc is None
    assert not (tmp_path / "progress.jsonl").exists()


def test_parent_writes_jsonl_and_never_imports_wandb(tmp_path, monkeypatch):
    """The whole point: the parent must not initialise wandb, because every
    process it forks afterwards would inherit a poisoned asyncio manager and
    raise ForkedError. It only ever appends to a file."""
    spawned = {}

    def fake_popen(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(ws.subprocess, "Popen", fake_popen)
    path = str(tmp_path / "progress.jsonl")
    wb = ws.WandbSidecar("proj", "ent", "run", {"a": 1}, tags=["dataprep"],
                          progress_path=path)
    wb.log({"total_tokens": 10})
    wb.log({"total_tokens": 20}, step=3)

    lines = [json.loads(l) for l in open(path)]
    assert lines == [{"total_tokens": 10}, {"total_tokens": 20, "_step": 3}]
    assert spawned["argv"][1:3] == ["-m", "daedalus.wandb_sidecar"]
    meta = json.loads(spawned["argv"][4])
    assert meta["project"] == "proj" and meta["entity"] == "ent" and meta["name"] == "run"
    # fork+exec, not a bare fork, so the child gets a fresh interpreter.
    assert spawned["kwargs"]["start_new_session"] is True


def test_parent_survives_a_child_that_cannot_start(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("no fork for you")

    monkeypatch.setattr(ws.subprocess, "Popen", boom)
    warnings = []
    wb = ws.WandbSidecar("p", None, "n", {}, progress_path=str(tmp_path / "p.jsonl"),
                          log=warnings.append)
    wb.log({"x": 1})
    wb.finish()
    assert wb.proc is None
    assert any("sidecar failed to start" in w for w in warnings)


def test_parent_truncates_a_previous_runs_progress_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ws.subprocess, "Popen", lambda *a, **k: object())
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps({"total_tokens": 999}) + "\n")   # a previous run's line
    wb = ws.WandbSidecar("p", None, "n", {}, progress_path=str(path))
    wb.log({"total_tokens": 1})
    assert [json.loads(l) for l in open(path)] == [{"total_tokens": 1}]


def test_parent_clears_a_previous_runs_url_file(tmp_path, monkeypatch):
    """A relaunch must not announce the run it replaced.

    `run_url` returns the first non-empty value it sees, and the child needs
    ~30-60 s for `wandb.init` to return, so a leftover URL is readable long
    before the real one exists. On 2026-08-10 the dclm top-up was killed and
    relaunched two minutes later and the live run announced itself as the dead
    one -- a 7-byte run that never logged a step. That URL is what reaches the
    log, STATUS.md and the operator's phone, so a healthy job presents as a dead
    dashboard, which is exactly the hang this logging exists to rule out.
    """
    monkeypatch.setattr(ws.subprocess, "Popen", lambda *a, **k: object())
    path = tmp_path / "progress.jsonl"
    stale = tmp_path / "progress.jsonl.url"
    stale.write_text("https://wandb.ai/e/p/runs/DEAD\n")

    wb = ws.WandbSidecar("p", None, "n", {}, progress_path=str(path))

    # Nothing to report yet is the correct answer; the previous run's URL is not.
    assert wb.run_url(timeout_s=0.05, poll_s=0.01) is None
    # And once this run's child publishes, that is what comes back.
    stale.write_text("https://wandb.ai/e/p/runs/LIVE\n")
    assert wb.run_url(timeout_s=1, poll_s=0.01).endswith("/LIVE")


def test_run_url_returns_none_rather_than_blocking(tmp_path, monkeypatch):
    monkeypatch.setattr(ws.subprocess, "Popen", lambda *a, **k: object())
    wb = ws.WandbSidecar("p", None, "n", {}, progress_path=str(tmp_path / "p.jsonl"))
    assert wb.run_url(timeout_s=0.05, poll_s=0.01) is None   # child never wrote one


def test_run_url_reads_what_the_child_published(tmp_path, monkeypatch):
    monkeypatch.setattr(ws.subprocess, "Popen", lambda *a, **k: object())
    path = tmp_path / "p.jsonl"
    wb = ws.WandbSidecar("p", None, "n", {}, progress_path=str(path))
    (tmp_path / "p.jsonl.url").write_text("https://wandb.ai/e/p/runs/xyz\n")
    assert wb.run_url(timeout_s=1, poll_s=0.01) == "https://wandb.ai/e/p/runs/xyz"


# -------------------------------------------------------------- the child ---

def test_publish_forwards_records_and_stops_at_the_sentinel(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        {"total_tokens": 1}, {"total_tokens": 2, "_step": 7}, {ws._SENTINEL: True},
        {"total_tokens": 3},  # after the sentinel: must not be published
    ]))

    n = publish = ws.publish(str(path), {"project": "p", "name": "n"}, poll_s=0)

    logger = FakeWandbLogger.instances[-1]
    assert n == 2
    assert logger.logged == [({"total_tokens": 1}, None), ({"total_tokens": 2}, 7)]
    assert logger.finished is True


def test_publish_writes_the_run_url_for_the_parent(tmp_path):
    path = tmp_path / "progress.jsonl"
    url_path = tmp_path / "progress.jsonl.url"
    path.write_text(json.dumps({ws._SENTINEL: True}) + "\n")

    ws.publish(str(path), {"project": "p", "name": "n", "url_path": str(url_path)}, poll_s=0)

    assert url_path.read_text() == FakeRun.url


def test_publish_skips_a_torn_line_instead_of_dying(tmp_path):
    """The parent appends while the child reads, so a partial final line is
    normal. It must be skipped, not fatal."""
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps({"total_tokens": 1}) + "\n"
                    + '{"total_tokens": \n'
                    + json.dumps({ws._SENTINEL: True}) + "\n")

    n = ws.publish(str(path), {"project": "p", "name": "n"}, poll_s=0)

    assert n == 1
    assert FakeWandbLogger.instances[-1].finished is True


def test_publish_gives_up_when_idle_rather_than_tailing_forever(tmp_path):
    """An orphaned publisher (parent killed without a sentinel) must not
    outlive the build."""
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps({"total_tokens": 1}) + "\n")
    slept = []

    n = ws.publish(str(path), {"project": "p", "name": "n"}, poll_s=1.0,
                    max_idle_s=3.0, sleep=slept.append)

    assert n == 1
    assert len(slept) == 3  # bounded, then exited
    assert FakeWandbLogger.instances[-1].finished is True


def test_cli_rejects_missing_arguments(capsys):
    assert ws._main(["daedalus.wandb_sidecar"]) == 2
    assert "usage:" in capsys.readouterr().out
