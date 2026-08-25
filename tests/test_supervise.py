"""Tests for daedalus/supervise.py -- the retry-from-checkpoint primitive.

Everything is injected (runner, sleeper, log), so no process is spawned and
no wall-clock is spent.
"""
import json
import os

import pytest

from daedalus.supervise import (
    INFLIGHT_SCHEMA,
    TrainingFailed,
    proc_start_ticks,
    run_with_resume,
)


def _recorder(returncodes):
    """A runner that plays back a fixed sequence of exit codes."""
    calls = []
    seq = iter(returncodes)

    def runner(cmd):
        calls.append(list(cmd))
        return next(seq)

    return runner, calls


def test_success_first_time_never_resumes(tmp_path):
    runner, calls = _recorder([0])
    report = run_with_resume(["train.py"], str(tmp_path / "ckpt.pt"),
                             runner=runner, sleeper=lambda s: None,
                             log=lambda m: None)
    assert report == {"attempts": 1, "resumed": False, "returncodes": [0]}
    assert "--resume" not in calls[0]


def test_retry_resumes_from_an_existing_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    runner, calls = _recorder([1, 0])
    report = run_with_resume(["train.py"], str(ckpt), runner=runner,
                             sleeper=lambda s: None, log=lambda m: None)
    assert report["attempts"] == 2
    assert report["resumed"] is True
    assert "--resume" not in calls[0]
    assert calls[1][-2:] == ["--resume", str(ckpt)]


def test_no_resume_flag_when_no_checkpoint_exists(tmp_path):
    """A crash before the first 30-minute checkpoint leaves nothing to resume
    from; claiming otherwise would corrupt the record."""
    runner, calls = _recorder([1, 0])
    report = run_with_resume(["train.py"], str(tmp_path / "missing.pt"),
                             runner=runner, sleeper=lambda s: None,
                             log=lambda m: None)
    assert report["resumed"] is False
    assert "--resume" not in calls[1]


def test_gives_up_and_reports_every_returncode(tmp_path):
    runner, calls = _recorder([1, 2, 3])
    with pytest.raises(TrainingFailed) as exc:
        run_with_resume(["train.py"], str(tmp_path / "ckpt.pt"),
                        max_attempts=3, runner=runner,
                        sleeper=lambda s: None, log=lambda m: None)
    assert exc.value.attempts == 3
    assert exc.value.returncodes == [1, 2, 3]
    assert len(calls) == 3


def test_backoff_is_exponential_and_capped(tmp_path):
    """A crash that recurs instantly would otherwise burn every attempt in
    seconds and leave the box idle for the rest of the night."""
    slept = []
    runner, _ = _recorder([1, 1, 1, 1, 0])
    run_with_resume(["train.py"], str(tmp_path / "ckpt.pt"), max_attempts=5,
                    backoff_sec=60, max_backoff_sec=200, runner=runner,
                    sleeper=slept.append, log=lambda m: None)
    assert slept == [60, 120, 200, 200]  # doubling, then clamped


def test_no_sleep_after_the_final_attempt(tmp_path):
    slept = []
    runner, _ = _recorder([1, 1])
    with pytest.raises(TrainingFailed):
        run_with_resume(["train.py"], str(tmp_path / "ckpt.pt"),
                        max_attempts=2, runner=runner, sleeper=slept.append,
                        log=lambda m: None)
    assert len(slept) == 1  # between the two attempts only


def test_the_original_command_is_not_mutated(tmp_path):
    """Appending --resume to the caller's list would compound it across
    attempts: --resume a --resume a --resume a."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("w")
    cmd = ["train.py", "--config", "x"]
    runner, calls = _recorder([1, 1, 0])
    run_with_resume(cmd, str(ckpt), runner=runner, sleeper=lambda s: None,
                    log=lambda m: None)
    assert cmd == ["train.py", "--config", "x"]
    assert calls[2].count("--resume") == 1


# ------------------------------------------------------- watchdog halts ---
#
# The bug these pin: watchdog.py SIGTERMs the trainer and *exits*, so the
# supervisor sees a non-zero exit code that is indistinguishable from a crash.
# Retrying is right for a crash and catastrophic for a divergence -- it resumes
# the diverged checkpoint with no watchdog left running, trains a broken model
# for the rest of the run, and exits 0.

def _marker(tmp_path, kind="divergence", reason="loss diverged at step 12"):
    import json
    p = tmp_path / "watchdog-halt.json"
    p.write_text(json.dumps({"kind": kind, "reason": reason, "at": 1.0}))
    return str(p)


def test_a_watchdog_halt_is_not_resumed(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    runner, calls = _recorder([143, 0])      # SIGTERM, then a would-be resume
    with pytest.raises(TrainingFailed) as exc:
        run_with_resume(["train.py"], str(ckpt), max_attempts=5, runner=runner,
                        sleeper=lambda s: None, log=lambda m: None,
                        halt_marker=_marker(tmp_path))
    assert len(calls) == 1, "resumed a run the watchdog deliberately halted"
    assert exc.value.halt["kind"] == "divergence"
    assert "diverged" in str(exc.value)


def test_a_halt_stops_before_the_backoff_sleep(tmp_path):
    """15 minutes of a rented box on the way to a conclusion already known."""
    slept = []
    runner, _ = _recorder([143])
    with pytest.raises(TrainingFailed):
        run_with_resume(["train.py"], str(tmp_path / "ckpt.pt"), max_attempts=5,
                        runner=runner, sleeper=slept.append, log=lambda m: None,
                        halt_marker=_marker(tmp_path, "stall", "no progress"))
    assert slept == []


def test_a_crash_without_a_marker_still_resumes(tmp_path):
    """The marker must not turn every crash terminal -- restarting those is
    the entire reason this module exists."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    runner, calls = _recorder([1, 0])
    report = run_with_resume(["train.py"], str(ckpt), runner=runner,
                             sleeper=lambda s: None, log=lambda m: None,
                             halt_marker=str(tmp_path / "watchdog-halt.json"))
    assert report["attempts"] == 2 and report["resumed"] is True
    assert "--resume" in calls[1]


def test_an_unreadable_marker_is_treated_as_absent(tmp_path):
    """Failing towards "keep training" on a corrupt marker: a half-written
    file must not end a four-day run that is otherwise healthy."""
    p = tmp_path / "watchdog-halt.json"
    p.write_text("{not json")
    runner, _ = _recorder([1, 0])
    report = run_with_resume(["train.py"], str(tmp_path / "ckpt.pt"),
                             runner=runner, sleeper=lambda s: None,
                             log=lambda m: None, halt_marker=str(p))
    assert report["attempts"] == 2


def test_no_marker_path_keeps_the_old_behaviour(tmp_path):
    runner, _ = _recorder([1, 1, 0])
    report = run_with_resume(["train.py"], str(tmp_path / "ckpt.pt"),
                             runner=runner, sleeper=lambda s: None,
                             log=lambda m: None)
    assert report["attempts"] == 3


# --------------------------------------- relaunch after an interruption ---
#
# The bug these pin: `attempt > 1` is the wrong test for "has this run already
# started". A supervisor that is *killed* -- by a session ending, by a reboot,
# by an OOM -- leaves an open in-flight marker beside a checkpoint, and the next
# launch of the same command is attempt 1 of a *fresh* supervisor, so it carried
# no `--resume` and restarted at step zero. Phase 4 lost 60.3M tokens and the
# 673MB checkpoint sitting next to the run that way, and phases 5-8 have longer
# arms. The marker is the evidence that a launch already happened; the resume
# decision now reads it instead of counting attempts.

def _open_marker(run_dir, cmd, ckpt, **overrides):
    """An in-flight marker for a launch that was killed, not finished."""
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "schema": INFLIGHT_SCHEMA,
        "run_dir": str(run_dir),
        "cmd": list(cmd),
        "ckpt_path": str(ckpt),
        "completed": False,
        "outcome": None,
        "started_at": "2026-08-24T21:00:00Z",
        # A pid that cannot be alive, so `supervisor_is_live` is a definite
        # False rather than the unknown an absent field would give.
        "supervisor_pid": 2 ** 30,
        "supervisor_start_ticks": 1,
    }
    payload.update(overrides)
    with open(os.path.join(str(run_dir), "inflight.json"), "w") as handle:
        json.dump(payload, handle)
    return payload


def test_relaunching_an_interrupted_run_resumes_on_the_first_attempt(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("60.3M tokens of training")
    cmd = ["train.py", "--run-name", "tok-probe-32768-equal-bytes"]
    _open_marker(tmp_path, cmd, ckpt)
    runner, calls = _recorder([0])
    report = run_with_resume(list(cmd), str(ckpt), runner=runner,
                             sleeper=lambda s: None, log=lambda m: None)
    assert calls[0][-2:] == ["--resume", str(ckpt)], "restarted from step zero"
    assert report["resumed"] is True


def test_a_relaunch_with_a_different_command_starts_fresh(tmp_path):
    """The checkpoint beside a marker belongs to the command that wrote it.
    Handing it to a different one resumes the wrong weights."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    _open_marker(tmp_path, ["train.py", "--total-tokens", "200000000"], ckpt)
    runner, calls = _recorder([0])
    report = run_with_resume(["train.py", "--total-tokens", "500000000"],
                             str(ckpt), runner=runner, sleeper=lambda s: None,
                             log=lambda m: None)
    assert "--resume" not in calls[0]
    assert report["resumed"] is False


def test_a_closed_marker_starts_fresh(tmp_path):
    """`mark_inflight_done` closes the marker for both endings. A completed run
    is not interrupted, and an `attempts_exhausted` one already spent its
    retries -- neither is a relaunch to continue."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    cmd = ["train.py"]
    _open_marker(tmp_path, cmd, ckpt, completed=True, outcome="completed")
    runner, calls = _recorder([0])
    report = run_with_resume(list(cmd), str(ckpt), runner=runner,
                             sleeper=lambda s: None, log=lambda m: None)
    assert "--resume" not in calls[0]
    assert report["resumed"] is False


def test_a_live_supervisor_marker_does_not_hand_over_the_checkpoint(tmp_path):
    """An open marker whose launcher is still alive is not an interruption --
    it is a run in progress, possibly asleep in its inter-attempt backoff. Two
    trainers resuming one checkpoint is worse than the restart this replaces,
    so the resume needs the previous owner to be provably gone."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    cmd = ["train.py"]
    _open_marker(tmp_path, cmd, ckpt, supervisor_pid=os.getpid(),
                 supervisor_start_ticks=proc_start_ticks(os.getpid()))
    runner, calls = _recorder([0])
    report = run_with_resume(list(cmd), str(ckpt), runner=runner,
                             sleeper=lambda s: None, log=lambda m: None)
    assert "--resume" not in calls[0]
    assert report["resumed"] is False


def test_an_interrupted_run_the_watchdog_halted_is_not_resumed(tmp_path):
    """A SIGKILLed supervisor never reaches `mark_inflight_done`, so a diverged
    run can leave an *open* marker next to its halt marker. Resuming that is the
    exact failure `halt_marker` exists to prevent."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    cmd = ["train.py"]
    _open_marker(tmp_path, cmd, ckpt)
    runner, calls = _recorder([0])
    report = run_with_resume(list(cmd), str(ckpt), runner=runner,
                             sleeper=lambda s: None, log=lambda m: None,
                             halt_marker=_marker(tmp_path))
    assert "--resume" not in calls[0]
    assert report["resumed"] is False


def test_an_interrupted_run_with_no_checkpoint_claims_no_resume(tmp_path):
    """Interrupted before the first checkpoint: there is nothing to continue
    from, and the report must not say there was."""
    cmd = ["train.py"]
    _open_marker(tmp_path, cmd, tmp_path / "ckpt.pt")
    runner, calls = _recorder([0])
    report = run_with_resume(list(cmd), str(tmp_path / "ckpt.pt"),
                             runner=runner, sleeper=lambda s: None,
                             log=lambda m: None)
    assert "--resume" not in calls[0]
    assert report["resumed"] is False


def test_resume_interrupted_can_be_turned_off(tmp_path):
    """The escape hatch for a caller that means "start this arm over"."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("weights")
    cmd = ["train.py"]
    _open_marker(tmp_path, cmd, ckpt)
    runner, calls = _recorder([0])
    report = run_with_resume(list(cmd), str(ckpt), runner=runner,
                             sleeper=lambda s: None, log=lambda m: None,
                             resume_interrupted=False)
    assert "--resume" not in calls[0]
    assert report["resumed"] is False
