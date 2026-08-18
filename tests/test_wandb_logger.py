"""W&B is the operator's primary live view of a ~92 h `hero` run, watched from a
phone (AGENT.md SS5.1). These tests pin the two properties that matter at that
duration: a log failure never reaches the training loop, and a *transient* one
does not cost the dashboard for the rest of the run.

The failure mode this file exists for is not a crash -- `log()` has always
swallowed exceptions -- but silence. A frozen dashboard looks exactly like a dead
run, which is the alarm it is there to raise.
"""
from daedalus.wandb_logger import (MAX_CONSECUTIVE_FAILURES, RETRY_AFTER_SEC,
                                   WandbLogger)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeRun:
    """A wandb run whose `log` fails for the first `fail_times` calls."""

    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = 0
        self.logged = []

    def log(self, record, step=None):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("connection reset")
        self.logged.append((record, step))


def make(fail_times=0, clock=None):
    clock = clock or FakeClock()
    lg = WandbLogger("p", None, "n", {}, enabled=False)
    lg.clock = clock
    lg.run = FakeRun(fail_times)
    return lg, lg.run, clock


def test_a_disabled_logger_never_touches_wandb():
    lg = WandbLogger("p", None, "n", {}, enabled=False)
    assert lg.run is None
    lg.log({"loss": 1.0}, step=1)      # must not raise
    lg.finish()


def test_a_log_failure_never_reaches_the_caller():
    lg, run, _ = make(fail_times=1)
    lg.log({"loss": 1.0}, step=1)      # must not raise
    assert run.calls == 1


def test_a_transient_failure_mutes_but_does_not_disable():
    """The regression this file was written for. One blip used to set
    `self.run = None` permanently -- at hero scale that is 89 h of blank
    dashboard bought by one dropped packet."""
    lg, run, clock = make(fail_times=1)
    lg.log({"loss": 1.0}, step=1)
    assert lg.run is not None, "a single failure must not disable W&B"

    # Muted: calls in the retry window do not reach wandb at all.
    clock.advance(RETRY_AFTER_SEC - 1)
    lg.log({"loss": 2.0}, step=2)
    assert run.calls == 1, "muted window should not call through"

    # Past the window it retries, succeeds, and resumes.
    clock.advance(2)
    lg.log({"loss": 3.0}, step=3)
    assert run.calls == 2
    assert run.logged == [({"loss": 3.0}, 3)]


def test_a_success_resets_the_failure_counter():
    """Six failures spread across a long run must not accumulate into a
    permanent disable -- the cap is for a dead W&B, not a flaky link."""
    lg, run, clock = make(fail_times=0)
    for i in range(MAX_CONSECUTIVE_FAILURES * 3):
        run.fail_times = 1                     # fail once...
        lg.log({"i": i}, step=i)
        assert lg.run is not None
        clock.advance(RETRY_AFTER_SEC + 1)
        lg.log({"i": i}, step=i)               # ...then succeed
        assert lg._consecutive_failures == 0
    assert lg.run is not None


def test_a_genuinely_dead_wandb_is_disabled_after_the_cap():
    lg, run, clock = make(fail_times=10_000)
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        lg.log({"loss": 1.0})
        clock.advance(RETRY_AFTER_SEC + 1)
    assert lg.run is None, "a dead W&B must stop being retried forever"
    before = run.calls
    lg.log({"loss": 1.0})
    assert run.calls == before, "disabled logger must not call through"


def test_the_cap_costs_at_most_one_hour_of_retries():
    """Bounds the noise a dead W&B produces in a 92 h run's log."""
    assert MAX_CONSECUTIVE_FAILURES * RETRY_AFTER_SEC <= 3600.0


def test_finish_is_safe_when_the_run_is_gone():
    lg, _, _ = make(fail_times=10_000)
    lg.run = None
    lg.finish()                                 # must not raise
