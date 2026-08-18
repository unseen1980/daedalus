"""Tests for daedalus/ckpt_uploader.py. Fully offline -- the Hub API is faked,
so no network calls and no real uploads.

The live end-to-end restore against the real Hub is a separate, network-using
check (`runs/preflight/hub-restore.md`); this file covers the contract.

Run: python -m pytest tests/test_ckpt_uploader.py -v
"""
import json
import os
import sys
import time
from unittest import mock

import pytest

from daedalus import ckpt_uploader as cu


_NEEDS_PROC = pytest.mark.skipif(
    not os.path.exists("/proc"),
    reason="reads /proc/<pid>/io, which only exists on Linux",
)


class FakeApi:
    """Stands in for huggingface_hub.HfApi. Records what was uploaded and to
    which revision, and can be told to fail specific paths so the retry
    contract is testable."""

    def __init__(self, token=None, fail_paths=(), fail_create=False,
                 fail_branches=()):
        self.token = token
        self.uploaded = []          # (path_in_repo, revision)
        self.created = []
        self.branches = []
        self.commit_messages = []
        self._fail_paths = set(fail_paths)
        self._fail_create = fail_create
        self._fail_branches = set(fail_branches)

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
        if self._fail_create:
            raise ConnectionError("hub unreachable")
        self.created.append((repo_id, repo_type, private))

    def create_branch(self, repo_id=None, branch=None, repo_type=None,
                      exist_ok=None):
        if branch in self._fail_branches:
            raise OSError(f"cannot create {branch}")
        self.branches.append(branch)

    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None,
                    repo_type=None, revision=None, commit_message=None):
        if path_in_repo in self._fail_paths:
            raise OSError(f"upload failed for {path_in_repo}")
        self.uploaded.append((path_in_repo, revision))
        self.commit_messages.append(commit_message)


@pytest.fixture
def fake_api(monkeypatch):
    holder = {}

    def install(**kwargs):
        api = FakeApi(**kwargs)
        holder["api"] = api
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: api)
        return api

    return install


def _stage_payload(outbox, name, path_in_repo, revision="main", body=b"weights",
                   seal=True, **meta):
    os.makedirs(outbox, exist_ok=True)
    path = os.path.join(outbox, name)
    with open(path, "wb") as f:
        f.write(body)
    if seal:
        cu.stage(outbox, path, path_in_repo, revision=revision, meta=meta)
    return path


# ------------------------------------------------------------- discovery ---

def test_only_sealed_payloads_are_pending(tmp_path):
    """An unsealed payload is one a torch.save may still be writing. Uploading
    it would push a truncated checkpoint that looks fine until it is needed."""
    outbox = str(tmp_path / "outbox")
    _stage_payload(outbox, "sealed.pt", "rolling/weights.pt")
    _stage_payload(outbox, "unsealed.pt", "rolling/weights.pt", seal=False)
    pending = cu.pending_uploads(outbox)
    assert [r["payload"] for _, r in pending] == ["sealed.pt"]


def test_payload_shorter_than_its_sidecar_is_not_pending(tmp_path):
    """Guards the window between rename and seal: a payload that does not
    match the size its sidecar recorded is not yet safe to send."""
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "w.pt", "rolling/weights.pt", body=b"0123456789")
    with open(path, "wb") as f:
        f.write(b"012")  # truncated after sealing
    assert cu.pending_uploads(outbox) == []


def test_sidecar_without_payload_is_cleaned_up(tmp_path):
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "w.pt", "rolling/weights.pt")
    os.remove(path)
    assert cu.pending_uploads(outbox) == []
    assert not os.path.exists(path + cu.SIDECAR_SUFFIX)


def test_missing_outbox_is_not_an_error(tmp_path):
    assert cu.pending_uploads(str(tmp_path / "nope")) == []


# ------------------------------------------------------------ supersede ---

def test_newer_rolling_payload_supersedes_older(tmp_path, fake_api):
    """If uploads fall behind the trainer, a stale rolling checkpoint has no
    value -- sending it would consume the bandwidth the current one needs."""
    api = fake_api()
    outbox = str(tmp_path / "outbox")
    old = _stage_payload(outbox, "weights-step1.pt", "rolling/weights.pt",
                         revision="rolling", step=1, kind="rolling")
    new = _stage_payload(outbox, "weights-step2.pt", "rolling/weights.pt",
                         revision="rolling", step=2, kind="rolling")
    summary = cu.upload_once(outbox, "me/repo")

    assert summary["uploaded"] == 1 and summary["superseded"] == 1
    assert ("rolling/weights.pt", "rolling") in api.uploaded
    assert api.commit_messages[0] == "rolling step 2"
    assert not os.path.exists(old) and not os.path.exists(new)


def test_different_repo_paths_do_not_supersede_each_other(tmp_path, fake_api):
    """The milestone and the rolling copy are different slots -- the milestone
    must never be dropped because a rolling checkpoint is newer."""
    api = fake_api()
    outbox = str(tmp_path / "outbox")
    _stage_payload(outbox, "milestone.pt", "milestone/checkpoint.pt",
                   revision="hero-stable-end-step10", step=10, kind="milestone")
    _stage_payload(outbox, "weights-step11.pt", "rolling/weights.pt",
                   revision="rolling", step=11, kind="rolling")
    summary = cu.upload_once(outbox, "me/repo")

    assert summary["uploaded"] == 2 and summary["superseded"] == 0
    assert ("milestone/checkpoint.pt", "hero-stable-end-step10") in api.uploaded


# --------------------------------------------------------------- upload ---

def test_upload_creates_repo_and_branch_and_removes_payload(tmp_path, fake_api):
    api = fake_api()
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "milestone.pt", "milestone/checkpoint.pt",
                          revision="hero-stable-end-step7", step=7,
                          kind="milestone")
    cu.upload_once(outbox, "me/repo")

    assert api.created == [("me/repo", "model", True)]
    assert api.branches == ["hero-stable-end-step7"]
    # Disk is bounded: an uploaded payload does not linger for four days.
    assert not os.path.exists(path)
    assert not os.path.exists(path + cu.SIDECAR_SUFFIX)


def test_main_revision_needs_no_branch(tmp_path, fake_api):
    api = fake_api()
    outbox = str(tmp_path / "outbox")
    _stage_payload(outbox, "w.pt", "weights.pt", revision="main", step=1)
    cu.upload_once(outbox, "me/repo")
    assert api.branches == []


def test_pointer_written_to_main(tmp_path, fake_api):
    """A phone can read step/tokens without pulling 321 MB."""
    api = fake_api()
    outbox = str(tmp_path / "outbox")
    _stage_payload(outbox, "w.pt", "rolling/r1/weights.pt", revision="rolling",
                   step=42, tokens_seen=1234, kind="rolling", run_name="r1")
    cu.upload_once(outbox, "me/repo")
    # Run-scoped: hero's pointer must not be replaced by abl-arch's.
    assert ("latest-rolling-r1.json", "main") in api.uploaded


def test_failed_upload_leaves_payload_pending(tmp_path, fake_api):
    api = fake_api(fail_paths={"rolling/weights.pt"})
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "w.pt", "rolling/weights.pt",
                          revision="rolling", step=1)
    summary = cu.upload_once(outbox, "me/repo")

    assert summary["failed"] == 1 and summary["uploaded"] == 0
    assert os.path.exists(path), "a failed upload must not delete the payload"
    assert len(cu.pending_uploads(outbox)) == 1  # retried next pass


def test_unreachable_repo_is_retried_not_lost(tmp_path, fake_api):
    fake_api(fail_create=True)
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "w.pt", "rolling/weights.pt", step=1)
    summary = cu.upload_once(outbox, "me/repo")
    assert summary["failed"] == 1 and os.path.exists(path)


def test_branch_failure_leaves_milestone_pending(tmp_path, fake_api):
    """The milestone is the branch point for all future work -- a branch that
    could not be created must not silently drop it."""
    fake_api(fail_branches={"hero-stable-end-step7"})
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "m.pt", "milestone/checkpoint.pt",
                          revision="hero-stable-end-step7", step=7)
    summary = cu.upload_once(outbox, "me/repo")
    assert summary["failed"] == 1 and os.path.exists(path)


def test_pointer_failure_does_not_lose_the_checkpoint(tmp_path, fake_api):
    """The pointer is cosmetic; the checkpoint is not."""
    fake_api(fail_paths={"latest-rolling-r1.json"})
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "w.pt", "rolling/r1/weights.pt",
                          revision="rolling", step=1, kind="rolling",
                          run_name="r1")
    summary = cu.upload_once(outbox, "me/repo")
    assert summary["uploaded"] == 1 and not os.path.exists(path)


def test_empty_outbox_makes_no_api_calls(tmp_path, fake_api):
    api = fake_api()
    summary = cu.upload_once(str(tmp_path / "outbox"), "me/repo")
    assert summary["uploaded"] == 0 and api.created == []


# ------------------------------------------------ closed-client reconnect ---
# huggingface_hub's own retry loop (_http_backoff_base, measured on 1.18.0)
# calls close_session() on a ConnectError but keeps reusing the now-closed
# local client for its own remaining retries, so one transient DNS blip
# guarantees that whole call fails with RuntimeError("... client has been
# closed."). Reproduced live against the real Hub on 2026-08-12 -- see
# ckpt_uploader._call_with_reconnect's docstring. These tests cover our
# workaround, not huggingface_hub's internals.

def test_call_with_reconnect_retries_the_closed_client_error():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Cannot send a request, as the client has "
                              "been closed.")
        return "ok"

    assert cu._call_with_reconnect(flaky, attempts=3) == "ok"
    assert calls["n"] == 3


def test_call_with_reconnect_finds_the_marker_in_a_wrapped_cause():
    """huggingface_hub's real chunked-LFS path wraps the closed-client error
    as `RuntimeError("Error while uploading '<path>' to the Hub.") from exc`
    -- the marker is on `__cause__`, not on the outer message."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            try:
                raise RuntimeError("Cannot send a request, as the client "
                                  "has been closed.")
            except RuntimeError as inner:
                raise RuntimeError(
                    "Error while uploading 'rolling/hero/weights.pt' to "
                    "the Hub.") from inner
        return "ok"

    assert cu._call_with_reconnect(flaky, attempts=3) == "ok"
    assert calls["n"] == 2


def test_call_with_reconnect_gives_up_after_its_attempt_budget():
    def always_closed():
        raise RuntimeError("Cannot send a request, as the client has been "
                          "closed.")

    with pytest.raises(RuntimeError, match="client has been closed"):
        cu._call_with_reconnect(always_closed, attempts=3)


def test_call_with_reconnect_does_not_swallow_unrelated_runtime_errors():
    """Only the specific closed-client message is a known-transient
    self-heal; any other RuntimeError must surface immediately."""
    def boom():
        raise RuntimeError("something else entirely")

    with pytest.raises(RuntimeError, match="something else entirely"):
        cu._call_with_reconnect(boom, attempts=3)


def test_upload_survives_one_closed_client_blip(tmp_path, fake_api):
    """End to end: a payload whose upload_file hits the closed-client error
    once must still land -- not be left pending for a whole extra pass."""
    api = fake_api()
    calls = {"n": 0}
    real_upload_file = api.upload_file

    def flaky_upload_file(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Cannot send a request, as the client has "
                              "been closed.")
        return real_upload_file(**kwargs)

    api.upload_file = flaky_upload_file
    outbox = str(tmp_path / "outbox")
    path = _stage_payload(outbox, "w.pt", "rolling/weights.pt",
                          revision="rolling", step=1)
    summary = cu.upload_once(outbox, "me/repo")

    assert summary["uploaded"] == 1 and summary["failed"] == 0
    assert not os.path.exists(path), "should not be left pending on a blip"


def test_state_records_what_landed(tmp_path, fake_api):
    fake_api()
    outbox = str(tmp_path / "outbox")
    _stage_payload(outbox, "w.pt", "rolling/weights.pt", revision="rolling",
                   step=99, tokens_seen=5, kind="rolling")
    cu.upload_once(outbox, "me/repo")
    with open(os.path.join(outbox, cu.STATE_FILENAME)) as f:
        state = json.load(f)
    assert state["uploads"]["rolling:rolling/weights.pt"]["step"] == 99


# ---------------------------------------------------------------- watch ---

def test_watch_survives_an_exploding_pass(tmp_path, monkeypatch):
    """The uploader is unattended for four days; a bad pass must not kill it."""
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("hub exploded")

    monkeypatch.setattr(cu, "upload_once_bounded", boom)
    totals = cu.watch(str(tmp_path), "me/repo", max_passes=3,
                      log=lambda *a: None, sleep=lambda s: None)
    assert calls["n"] == 3 and totals["passes"] == 3 and totals["failed"] == 3


def test_watch_sleeps_between_passes_but_not_after_the_last(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(cu, "upload_once_bounded",
                        lambda *a, **k: {"uploaded": 0, "failed": 0,
                                         "superseded": 0, "bytes": 0})
    cu.watch(str(tmp_path), "me/repo", max_passes=3, interval_s=7,
             log=lambda *a: None, sleep=slept.append)
    assert slept == [7, 7]


# ------------------------------------------------- the pass deadline ---
#
# On 2026-08-10 the live uploader wedged mid-transfer of the 1.4 GB milestone,
# socket in CLOSE-WAIT, 0% CPU, no bytes on the wire. Every failure path in
# this module assumes failure *raises*; a hang does not, so nothing retried and
# nothing was logged. It reproduced on restart, and it ignored SIGTERM.

def _substitute_child(script):
    """Run `python -c script` in place of the uploader child.

    The real `Popen` is kept deliberately, so the process, its
    `/proc/<pid>/io` counters and the `kill()` are all real. Mocking the
    process out would test the poll loop against the one thing it exists to
    measure -- see the 2026-08-12 note below.
    """
    import subprocess
    real_popen = subprocess.Popen

    def spy(cmd, **kw):
        spy.seen = {"cmd": cmd, "env": kw.get("env") or {}}
        return real_popen([sys.executable, "-c", script], **kw)

    spy.seen = {}
    return spy


# On 2026-08-12 the bound above became the bug. The box's uplink fell to
# ~105 KB/s, which puts the 321 MB rolling checkpoint at ~51 min -- past the
# 900 s ceiling. Every pass was killed at 900 s having sent ~90 MB and the next
# restarted from zero, so `hero` ran 9 h uninsured while the uplink was
# saturated the whole time. A fixed wall-clock bound cannot tell "wedged" from
# "slow", so the bound is now progress, with wall-clock demoted to a backstop.

def _stage_sized_payload(outbox, nbytes, step=1):
    """A real staged payload of `nbytes`, as the trainer would leave it.

    Real bytes because `pending_uploads` refuses a payload whose on-disk size
    disagrees with its sidecar -- the guard against pushing a half-written
    checkpoint -- so a sparse stub would simply be skipped and the ceiling
    would not scale at all.
    """
    payload = os.path.join(str(outbox), f"weights-step{step:09d}.pt")
    with open(payload, "wb") as f:
        f.write(b"\0" * nbytes)
    cu.stage(str(outbox), payload, "rolling/hero/weights.pt",
             revision="rolling", meta={"kind": "rolling", "step": step})
    return payload


# 100 KB at the 20 KB/s floor rate buys a ~4.9 s ceiling -- the same arithmetic
# that gives hero's 321 MB checkpoint ~4.4 h, at a size a test can afford.
CEILING_TEST_BYTES = 100 * 1024


def test_a_slow_but_moving_transfer_is_not_killed(tmp_path):
    """The 2026-08-12 regression: progress must beat the wall clock.

    The child runs well past `deadline_s` and would have been killed outright
    by the old bound. It does real IO throughout, so the pass must let it
    finish -- otherwise the retry is the same doomed attempt, forever.
    """
    import subprocess
    import time as _time
    _stage_sized_payload(tmp_path, CEILING_TEST_BYTES)
    # ~320 KB/s for ~3 s: slow, but comfortably carrying the real 64 KB quota in
    # every window. The stall timeout is kept short on purpose so the quota is
    # what decides this, rather than a window too long to ever expire.
    script = ("import sys, time\n"
              "for _ in range(30):\n"
              "    sys.stderr.write('x' * 32768); sys.stderr.flush()\n"
              "    time.sleep(0.1)\n"
              "print('{\"uploaded\": 1, \"failed\": 0, \"superseded\": 0, \"bytes\": 7}')\n")
    logged = []
    t0 = _time.time()
    with mock.patch.object(subprocess, "Popen", _substitute_child(script)):
        summary = cu.upload_once_bounded(
            str(tmp_path), "me/repo", token="t",
            deadline_s=0.5,          # far below the child's ~3 s runtime
            stall_timeout_s=1.0, poll_s=0.05, log=logged.append)
    assert _time.time() - t0 > 2.0, "the child did not actually outlive deadline_s"
    assert summary["uploaded"] == 1, f"a moving transfer was killed: {logged}"
    assert not any("killed" in m for m in logged), logged


@_NEEDS_PROC
def test_a_stalled_child_is_killed_within_the_stall_timeout(tmp_path):
    """The original wedge, still caught -- and now faster than the old 900 s.

    Not a mock: a child that sleeps forever does no IO at all. SIGKILL is what
    returns the socket and the RSS; SIGTERM was not enough against the live
    wedge, because CPython runs signal handlers between bytecodes and the main
    thread was blocked in a C-level socket call.
    """
    import subprocess
    import time as _time
    logged = []
    t0 = _time.time()
    with mock.patch.object(subprocess, "Popen",
                           _substitute_child("import time; time.sleep(120)")):
        summary = cu.upload_once_bounded(
            str(tmp_path), "me/repo", token="t",
            deadline_s=600.0,        # the ceiling must NOT be what stops this
            stall_timeout_s=1.0, poll_s=0.05, log=logged.append)
    elapsed = _time.time() - t0
    assert summary["failed"] == 1 and summary["uploaded"] == 0
    assert elapsed < 20, f"the stall bound did not fire ({elapsed:.0f}s)"
    assert any("moved only" in m and "process io" in m and "retries" in m for m in logged), logged


@_NEEDS_PROC
def test_a_trickling_wedge_is_killed_too(tmp_path):
    """A wedge is not silent, which a first cut of this bound assumed.

    Caught live on 2026-08-12: a `hero` upload sat in CLOSE-WAIT -- peer gone,
    25 bytes stuck in the recv queue, no possibility of completing -- and still
    moved ~32 B/s. Against a "no progress at all" test that transfer would have
    been nursed forever, which is the very livelock this bound replaced.

    So the child here dribbles steadily and must still be killed: progress is
    measured against the window's start, not the previous sample.
    """
    import subprocess
    import time as _time
    # ~320 B/s -- always "moving", never carrying its window's 64 KB quota.
    trickle = ("import sys, time\n"
               "while True:\n"
               "    sys.stderr.write('x' * 32); sys.stderr.flush()\n"
               "    time.sleep(0.1)\n")
    logged = []
    t0 = _time.time()
    with mock.patch.object(subprocess, "Popen", _substitute_child(trickle)):
        summary = cu.upload_once_bounded(
            str(tmp_path), "me/repo", token="t",
            deadline_s=600.0,        # the ceiling must NOT be what stops this
            stall_timeout_s=2.0, poll_s=0.05, log=logged.append)
    elapsed = _time.time() - t0
    assert summary["failed"] == 1, "a trickling wedge was treated as healthy"
    assert elapsed < 30, f"the stall bound did not fire ({elapsed:.0f}s)"
    assert any("moved only" in m for m in logged), logged


def test_the_backstop_ceiling_scales_with_the_bytes_pending(tmp_path):
    """A crawling-but-not-stalled pass is still bounded, by payload size.

    With nothing pending the ceiling collapses to `deadline_s`, which is what
    stops a small pass from crawling for hours.
    """
    import subprocess
    import time as _time
    # Writes fast and forever, so it always clears its stall quota and *only*
    # the ceiling can stop it. That combination -- /proc readable, progress
    # healthy, pass never ending -- is what a first cut of this got wrong: the
    # ceiling was chained onto the stall check as an `elif`, which made it
    # unreachable in production and left the pass unbounded.
    crawler = _substitute_child("import sys, time\n"
                                "while True:\n"
                                "    sys.stderr.write('z' * 65536); sys.stderr.flush()\n"
                                "    time.sleep(0.05)\n")

    # Nothing staged: the ceiling collapses to `deadline_s`, which is what stops
    # a pass with no work to do from crawling for hours.
    logged = []
    t0 = _time.time()
    with mock.patch.object(subprocess, "Popen", crawler):
        summary = cu.upload_once_bounded(
            str(tmp_path), "me/repo", token="t",
            deadline_s=0.5,
            stall_timeout_s=600.0,   # the stall bound must NOT be what stops this
            poll_s=0.05, log=logged.append)
    bare = _time.time() - t0
    assert summary["failed"] == 1
    assert bare < 3.0, f"the bare ceiling did not fire ({bare:.1f}s)"
    assert any("ceiling" in m for m in logged), logged

    # With a payload staged the same pass is bounded by the payload instead, so
    # the identical `deadline_s` no longer decides it.
    _stage_sized_payload(tmp_path, CEILING_TEST_BYTES)
    logged = []
    t0 = _time.time()
    with mock.patch.object(subprocess, "Popen", crawler):
        summary = cu.upload_once_bounded(
            str(tmp_path), "me/repo", token="t",
            deadline_s=0.5, stall_timeout_s=600.0,
            poll_s=0.05, log=logged.append)
    scaled = _time.time() - t0
    assert summary["failed"] == 1
    # The elapsed time is the proof that it scaled: same `deadline_s`, same
    # child, ~10x the bound, and the only thing that changed is the payload.
    assert scaled > 4.0, f"the ceiling did not scale with the payload ({scaled:.1f}s)"
    assert scaled > bare * 2
    assert any("ceiling" in m and "pending" in m for m in logged), logged


def test_a_good_pass_returns_the_childs_summary_and_relays_its_log(tmp_path):
    import subprocess
    logged = []
    script = ("print('ckpt-uploader: rolling/x/weights.pt@rolling step 7 (321 MB)')\n"
              "print('{\"uploaded\": 1, \"failed\": 0, \"superseded\": 0, \"bytes\": 321}')\n")
    with mock.patch.object(subprocess, "Popen", _substitute_child(script)):
        summary = cu.upload_once_bounded(str(tmp_path), "me/repo", token="t",
                                         poll_s=0.05, log=logged.append)
    assert summary == {"uploaded": 1, "failed": 0, "superseded": 0, "bytes": 321}
    assert any("step 7 (321 MB)" in m for m in logged), logged


def test_a_child_that_dies_is_a_failure_not_a_silent_success(tmp_path):
    import subprocess
    logged = []
    with mock.patch.object(
            subprocess, "Popen",
            _substitute_child("import sys; sys.stderr.write('boom\\n'); "
                              "sys.exit(1)")):
        summary = cu.upload_once_bounded(str(tmp_path), "me/repo", token="t",
                                         poll_s=0.05, log=logged.append)
    assert summary["failed"] == 1
    assert any("exited 1" in m and "boom" in m for m in logged), logged


def test_the_token_reaches_the_child_and_is_not_in_the_command(tmp_path):
    """The child re-reads HF_TOKEN_WRITE from its env. A token on the command
    line would land in `ps` output and in any log that echoes the pass."""
    import subprocess
    spy = _substitute_child("print('{}')")
    with mock.patch.object(subprocess, "Popen", spy):
        cu.upload_once_bounded(str(tmp_path), "me/repo", token="sekrit",
                               poll_s=0.05, log=lambda *a: None)
    assert spy.seen["env"]["HF_TOKEN_WRITE"] == "sekrit"
    assert "sekrit" not in " ".join(spy.seen["cmd"])
    assert "--once" in spy.seen["cmd"]


def test_watch_keeps_polling_after_a_wedged_pass(tmp_path, monkeypatch):
    """What the live bug cost: one wedge stopped every later upload too."""
    calls = {"n": 0}

    def wedged(*a, **kw):
        calls["n"] += 1
        return {"uploaded": 0, "failed": 1, "superseded": 0, "bytes": 0}

    monkeypatch.setattr(cu, "upload_once_bounded", wedged)
    totals = cu.watch(str(tmp_path), "me/repo", token="t", max_passes=3,
                      deadline_s=1.0, log=lambda *a: None, sleep=lambda s: None)
    assert calls["n"] == 3 and totals["failed"] == 3


def test_watch_threads_the_stall_bound_through_to_the_pass(tmp_path, monkeypatch):
    """A bound the watcher does not pass on is a bound that does not exist."""
    seen = {}

    def spy(outbox, repo, **kw):
        seen.update(kw)
        return {"uploaded": 0, "failed": 0, "superseded": 0, "bytes": 0}

    monkeypatch.setattr(cu, "upload_once_bounded", spy)
    cu.watch(str(tmp_path), "me/repo", token="t", max_passes=1,
             deadline_s=11.0, stall_timeout_s=7.0,
             log=lambda *a: None, sleep=lambda s: None)
    assert seen["deadline_s"] == 11.0 and seen["stall_timeout_s"] == 7.0


@_NEEDS_PROC
def test_progress_is_read_from_a_real_process(tmp_path):
    """`_proc_io_bytes` is the whole basis of the new bound, so it is measured
    against a real process rather than assumed."""
    import subprocess
    proc = subprocess.Popen([sys.executable, "-c",
                             "import sys, time\n"
                             "for _ in range(40):\n"
                             "    sys.stderr.write('y' * 4096); sys.stderr.flush()\n"
                             "    time.sleep(0.05)\n"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        first = cu._proc_io_bytes(proc.pid)
        assert first is not None and first > 0
        time.sleep(0.5)
        assert cu._proc_io_bytes(proc.pid) > first, "IO counters did not advance"
    finally:
        proc.kill()
        proc.wait()
    assert cu._proc_io_bytes(2 ** 22 + 1) is None  # a pid that cannot exist


# ------------------------------------------------------------- hub URIs ---

@pytest.mark.parametrize("uri,expected", [
    ("hub://me/daedalus-ckpt/rolling/weights.pt?rev=rolling",
     ("me/daedalus-ckpt", "rolling/weights.pt", "rolling")),
    ("hub://me/daedalus-ckpt/milestone/checkpoint.pt?rev=hero-stable-end-step10",
     ("me/daedalus-ckpt", "milestone/checkpoint.pt", "hero-stable-end-step10")),
    ("hub://me/repo/weights.pt", ("me/repo", "weights.pt", "main")),
])
def test_parse_hub_uri(uri, expected):
    assert cu.parse_hub_uri(uri) == expected


@pytest.mark.parametrize("uri", ["hub://me/repo", "hub://me", "s3://me/repo/w.pt"])
def test_parse_hub_uri_rejects_junk(uri):
    with pytest.raises(ValueError):
        cu.parse_hub_uri(uri)


def test_resolve_resume_passes_local_paths_through(tmp_path):
    """The common case must not touch the network at all."""
    local = str(tmp_path / "checkpoint.pt")
    assert cu.resolve_resume(local, str(tmp_path / "dl")) == local
    assert cu.resolve_resume(None, str(tmp_path / "dl")) is None


def test_resolve_resume_downloads_a_hub_uri(tmp_path, monkeypatch):
    seen = {}

    def fake_download(repo_id, path_in_repo, dest_dir, revision="main",
                      token=None, log=print):
        seen.update(repo_id=repo_id, path=path_in_repo, revision=revision,
                    dest=dest_dir)
        return os.path.join(dest_dir, "checkpoint.pt")

    monkeypatch.setattr(cu, "download_checkpoint", fake_download)
    out = cu.resolve_resume("hub://me/repo/milestone/checkpoint.pt?rev=br",
                            str(tmp_path / "dl"))
    assert seen == {"repo_id": "me/repo", "path": "milestone/checkpoint.pt",
                    "revision": "br", "dest": str(tmp_path / "dl")}
    assert out.endswith("checkpoint.pt")


# ----------------------------------------------------------------- lock ---
# hero.py restarts train.py after a crash, and a SIGKILLed trainer cannot reap
# the uploader it spawned. Without a lock, a four-day run accumulates one
# orphaned watcher per restart, all racing to delete each other's payloads.

def test_lock_is_acquired_on_a_fresh_outbox(tmp_path):
    outbox = str(tmp_path / "outbox")
    assert cu.acquire_lock(outbox) is True
    with open(os.path.join(outbox, cu.LOCK_FILENAME)) as f:
        assert int(f.read()) == os.getpid()


def test_lock_is_refused_while_a_live_uploader_holds_it(tmp_path, monkeypatch):
    outbox = str(tmp_path / "outbox")
    os.makedirs(outbox)
    with open(os.path.join(outbox, cu.LOCK_FILENAME), "w") as f:
        f.write("4242")
    monkeypatch.setattr(cu, "_process_is_an_uploader", lambda pid: pid == 4242)
    assert cu.acquire_lock(outbox, log=lambda *a: None) is False


def test_lock_is_taken_over_from_a_dead_holder(tmp_path, monkeypatch):
    outbox = str(tmp_path / "outbox")
    os.makedirs(outbox)
    with open(os.path.join(outbox, cu.LOCK_FILENAME), "w") as f:
        f.write("4242")
    monkeypatch.setattr(cu, "_process_is_an_uploader", lambda pid: False)
    assert cu.acquire_lock(outbox) is True


def test_lock_survives_a_corrupt_lockfile(tmp_path):
    """A truncated write must not wedge every future uploader."""
    outbox = str(tmp_path / "outbox")
    os.makedirs(outbox)
    with open(os.path.join(outbox, cu.LOCK_FILENAME), "w") as f:
        f.write("not-a-pid")
    assert cu.acquire_lock(outbox) is True


def test_reacquiring_our_own_lock_is_fine(tmp_path):
    outbox = str(tmp_path / "outbox")
    assert cu.acquire_lock(outbox) is True
    assert cu.acquire_lock(outbox) is True


def test_pid_reuse_does_not_look_like_a_live_uploader():
    """os.kill(pid, 0) alone would call any recycled PID a live holder, and the
    lock would never be released. The cmdline match also has to be the dotted
    module path -- a bare "ckpt_uploader" matches this very test process."""
    assert cu._process_is_an_uploader(os.getpid()) is False  # pytest, not us
    assert cu._process_is_an_uploader(999_999) is False      # not alive


# ------------------------------------------------------- liveness query ---
# The trainer needs to ask "is anything serving this outbox?" without taking
# the lock -- it is not an uploader, and taking the lock would lock out the
# real one.

def test_uploader_is_live_reports_a_live_holder(tmp_path, monkeypatch):
    outbox = str(tmp_path / "outbox")
    os.makedirs(outbox)
    with open(os.path.join(outbox, cu.LOCK_FILENAME), "w") as f:
        f.write("4242")
    monkeypatch.setattr(cu, "_process_is_an_uploader", lambda pid: pid == 4242)
    assert cu.uploader_is_live(outbox) is True


def test_uploader_is_live_reports_a_dead_holder(tmp_path, monkeypatch):
    outbox = str(tmp_path / "outbox")
    os.makedirs(outbox)
    with open(os.path.join(outbox, cu.LOCK_FILENAME), "w") as f:
        f.write("4242")
    monkeypatch.setattr(cu, "_process_is_an_uploader", lambda pid: False)
    assert cu.uploader_is_live(outbox) is False


def test_uploader_is_live_is_false_with_no_lockfile(tmp_path):
    assert cu.uploader_is_live(str(tmp_path / "nothing-here")) is False


def test_uploader_is_live_survives_a_corrupt_lockfile(tmp_path):
    outbox = str(tmp_path / "outbox")
    os.makedirs(outbox)
    with open(os.path.join(outbox, cu.LOCK_FILENAME), "w") as f:
        f.write("not-a-pid")
    assert cu.uploader_is_live(outbox) is False


def test_the_liveness_query_does_not_take_the_lock(tmp_path):
    """It must be read-only: a trainer that stamped its own pid into the
    lockfile would lock out every real uploader for the rest of the run."""
    outbox = str(tmp_path / "outbox")
    os.makedirs(outbox)
    cu.uploader_is_live(outbox)
    assert not os.path.exists(os.path.join(outbox, cu.LOCK_FILENAME))
