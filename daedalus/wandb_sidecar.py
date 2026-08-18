"""A W&B run owned by a separate process, fed by a file.

`run_dataprep` cannot call `wandb.init()` itself. wandb starts an asyncio
manager thread on init, and any process forked *after* that inherits a
manager belonging to the parent; touching wandb there raises
`ForkedError("This operation is not valid in a forked process. Original
PID=..., current PID=...")`. dataprep forks workers constantly -- once per
group at dispatch, and again for every within-source RSS respawn -- so an
initialised wandb in the parent poisons every respawn for the rest of the
run.

That is not hypothetical. `dataprep-full-attempt5` (13:46 UTC) died exactly
this way: `finepdfs-edu`, `cosmopedia-v2` and `finewiki-en` each soft-stopped
correctly, and each respawn then failed in `_run_group_worker`'s setup with
`ForkedError` raised out of `get_tokenizer()` (transformers touches wandb on
import). Every source was recorded failed at its first chunk boundary. It did
not show up in any of the offline validations because those pass
`wandb_enabled=False`, so no wandb session ever existed to inherit.

The old mitigation -- submit the main pool *before* `wandb.init()`, so the
initial workers fork while the parent is still clean -- only ever covered the
initial dispatch. Respawns fork later, by construction. Ordering cannot fix
that; the parent has to stay clean permanently.

So: the parent appends JSON lines to a file and never imports wandb. A child
launched through `subprocess` (fork + **exec**, so a genuinely fresh
interpreter, not an inherited one) owns the W&B run and tails that file. The
parent's fork-safety is then independent of W&B entirely, and the fragile
"submit before init" ordering is no longer load-bearing.

`WandbSidecar` is a drop-in for `WandbLogger` -- same `log`/`finish` -- so
callers don't care which they hold.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional

_SENTINEL = "__finish__"


class WandbSidecar:
    """Same surface as `WandbLogger`, but the W&B run lives in a child
    process. Never raises, never blocks: if the child can't start, logging
    silently becomes a no-op and the caller carries on (AGENT.md §5.1)."""

    def __init__(self, project: str, entity: Optional[str], name: str,
                  config: dict, tags=None, enabled: bool = True,
                  progress_path: Optional[str] = None, log=print):
        self.path = progress_path
        self.proc = None
        self._file = None
        self._log = log
        self.url_path = None
        if not enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            # Truncate: a previous run's leftover lines would otherwise be
            # replayed into this run's W&B history.
            self._file = open(self.path, "w")
            self.url_path = f"{self.path}.url"
            # Clear it for the same reason, and it is not cosmetic. `run_url`
            # polls this file and returns the first non-empty value it sees, so
            # a relaunch under the same --run-name reports the *previous* run's
            # URL: the child needs ~30-60 s for `wandb.init` to return, and the
            # stale file is readable instantly. Measured 2026-08-10 -- the dclm
            # top-up was killed and relaunched 2 min later, and the live run
            # (p0px5fyu) announced itself as the dead one (x2i51sis, 7 bytes,
            # never logged a step). That URL is what reaches the log line,
            # STATUS.md and the operator's phone, so the dashboard looks dead
            # while the job is fine -- indistinguishable from the hang this
            # logging exists to rule out.
            try:
                os.remove(self.url_path)
            except OSError:
                pass
            meta = {"project": project, "entity": entity, "name": name,
                    "config": config, "tags": list(tags or []),
                    "url_path": self.url_path}
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "daedalus.wandb_sidecar", self.path, json.dumps(meta)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                # start_new_session so a Ctrl-C aimed at the build doesn't
                # also kill the publisher mid-flush.
                start_new_session=True,
            )
        except Exception as e:
            self._log(f"WARNING: W&B sidecar failed to start ({e}); continuing without W&B")
            self.proc = None

    def log(self, record: dict, step: Optional[int] = None) -> None:
        if self._file is None:
            return
        try:
            payload = dict(record)
            if step is not None:
                payload["_step"] = step
            self._file.write(json.dumps(payload) + "\n")
            self._file.flush()
        except Exception as e:
            self._log(f"WARNING: W&B sidecar write failed ({e}); disabling W&B for this run")
            self._file = None

    def run_url(self, timeout_s: float = 30.0, poll_s: float = 0.5) -> Optional[str]:
        """The child writes the run URL out as soon as `wandb.init` returns.
        Polled rather than awaited so a slow or offline W&B never delays the
        build -- it just returns None."""
        if not self.url_path:
            return None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with open(self.url_path) as f:
                    url = f.read().strip()
                if url:
                    return url
            except OSError:
                pass
            time.sleep(poll_s)
        return None

    def finish(self) -> None:
        if self._file is not None:
            try:
                self._file.write(json.dumps({_SENTINEL: True}) + "\n")
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=30)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            self.proc = None


def publish(path: str, meta: dict, poll_s: float = 1.0,
             max_idle_s: Optional[float] = None, sleep=time.sleep) -> int:
    """Child entry point: tail `path`, forwarding each JSON line to W&B until
    the sentinel arrives (or the parent dies). Returns the number of records
    published.

    `max_idle_s` bounds how long it waits with no new line *and* no live
    parent, so an orphaned publisher can't outlive the build forever."""
    from daedalus.wandb_logger import WandbLogger

    wb = WandbLogger(project=meta.get("project", "daedalus"), entity=meta.get("entity"),
                      name=meta.get("name", "dataprep"), config=meta.get("config", {}),
                      tags=meta.get("tags"), enabled=True)
    url_path = meta.get("url_path")
    if url_path and wb.run is not None:
        try:
            with open(url_path, "w") as f:
                f.write(getattr(wb.run, "url", "") or "")
        except OSError:
            pass

    published = 0
    idle_for = 0.0
    parent_pid = os.getppid()
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                if max_idle_s is not None and idle_for >= max_idle_s:
                    break
                # The parent exiting without a sentinel (killed, OOM) must
                # still end the publisher rather than leave it tailing a file
                # nobody writes to.
                if parent_pid != 1 and os.getppid() != parent_pid:
                    break
                sleep(poll_s)
                idle_for += poll_s
                continue
            idle_for = 0.0
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line; the next read gets the rest
            if record.get(_SENTINEL):
                break
            step = record.pop("_step", None)
            wb.log(record, step=step)
            published += 1
    wb.finish()
    return published


def _main(argv) -> int:
    if len(argv) < 3:
        print("usage: python -m daedalus.wandb_sidecar <progress.jsonl> <meta-json>")
        return 2
    publish(argv[1], json.loads(argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
