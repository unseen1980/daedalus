"""Tests for daedalus/supervise.py -- the retry-from-checkpoint primitive.

Everything is injected (runner, sleeper, log), so no process is spawned and
no wall-clock is spent.
"""
import pytest

from daedalus.supervise import TrainingFailed, run_with_resume


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
