"""Standalone W&B wrapper (AGENT.md SS5.1), deliberately dependency-light.

Split out of train.py so callers that don't otherwise need torch --
dataprep.py's orchestrator, in particular -- can log to W&B without
importing it. That's not just tidiness: dataprep.py forks worker processes
from this orchestrator, and `import torch` plus `torch.cuda.is_available()`
inflates this process's virtual address space from ~4 GB to ~17 GB on a real
GPU box (CUDA context/driver reservations) merely by touching torch, *before*
any of that memory is actually used. Forked children inherit that bloated
address space via copy-on-write, which then trips their own (deliberately
tight) per-worker RLIMIT_AS backstop on the very next shared-library mmap --
this exact failure mode broke a live dataprep run (see STATUS.md) after
train.py's WandbLogger was first reused here.
"""
import os
import time
from typing import Optional

# How long to stay muted after a failed log before trying again, and how many
# consecutive mutes to tolerate before giving up for good. 10 min x 6 means a
# genuinely dead W&B costs six warning lines over an hour and then goes quiet,
# while a transient blip costs at most one 10-minute hole in the dashboard.
RETRY_AFTER_SEC = 600.0
MAX_CONSECUTIVE_FAILURES = 6


class WandbLogger:
    """Thin W&B wrapper that never crashes or blocks a run (AGENT.md SS5.1):
    missing API key -> offline mode; init failure -> warn and continue without.

    A *log* failure mutes W&B for `RETRY_AFTER_SEC` and then retries, rather
    than disabling it permanently. That distinction matters at `hero` scale: the
    run is ~92 h and W&B is the operator's primary live view, monitored from a
    phone (AGENT.md SS5.1). Under the old behaviour a single transient network
    blip at hour 3 left the dashboard frozen for the remaining 89 h behind one
    WARNING line in a log nobody reads -- and a frozen dashboard is
    indistinguishable from a dead run, which is exactly the alarm it exists to
    raise. Training is never affected either way: every path here swallows its
    exception, and `metrics.jsonl` is written separately.
    """

    def __init__(self, project: str, entity: Optional[str], name: str,
                config: dict, tags=None, enabled: bool = True, clock=time.time,
                run_id: Optional[str] = None, resume: Optional[str] = None):
        """`run_id`/`resume` control *run identity across processes*.

        W&B mints a fresh random run id on every `init()` -- the run *name* is
        only a display label, so two processes launched with the same name are
        two unrelated runs at two different URLs. That is the wrong default
        under `supervise.run_with_resume`, which restarts `train.py` after a
        crash: the URL published in STATUS.md and the hero gate issue would
        freeze at the crash point while training continued, unseen, somewhere
        else. A frozen dashboard is indistinguishable from a dead run, which is
        the alarm W&B exists to raise (AGENT.md SS5.1).

        Passing a stable `run_id` with `resume="allow"` re-attaches instead.
        The caller decides when that is right -- see `resolve_wandb_run_id` in
        train.py, which re-attaches only when this process actually resumed a
        checkpoint, so re-running a job under a recycled name still gets a
        clean run rather than appending to a discarded curve.
        """
        self.run = None
        self.clock = clock
        self._muted_until = 0.0
        self._consecutive_failures = 0
        if not enabled:
            return
        try:
            import wandb
            if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
                print("WARNING: WANDB_API_KEY not set; forcing WANDB_MODE=offline")
                os.environ["WANDB_MODE"] = "offline"
            kwargs = dict(project=project, entity=entity, name=name,
                          config=config, tags=tags or [])
            if run_id is not None:
                kwargs["id"] = run_id
            if resume is not None:
                kwargs["resume"] = resume
            self.run = wandb.init(**kwargs)
        except Exception as e:
            print(f"WARNING: W&B init failed ({e}); continuing without W&B")
            self.run = None

    def log(self, record: dict, step: Optional[int] = None) -> None:
        if self.run is None or self.clock() < self._muted_until:
            return
        try:
            self.run.log(record, step=step)
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"WARNING: W&B log failed ({e}) "
                      f"{self._consecutive_failures}x; disabling W&B for this run")
                self.run = None
                return
            self._muted_until = self.clock() + RETRY_AFTER_SEC
            print(f"WARNING: W&B log failed ({e}); muting W&B for "
                  f"{RETRY_AFTER_SEC / 60:.0f} min then retrying "
                  f"(failure {self._consecutive_failures}/"
                  f"{MAX_CONSECUTIVE_FAILURES})")
            return
        # Only a *successful* log clears the counter, so six failures spread
        # across a 92 h run never accumulate into a permanent disable -- the
        # cap is for W&B being genuinely dead, not for a flaky link.
        if self._consecutive_failures:
            print(f"W&B recovered after {self._consecutive_failures} "
                  f"failed attempt(s)")
            self._consecutive_failures = 0

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
