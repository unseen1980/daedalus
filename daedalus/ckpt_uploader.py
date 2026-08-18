"""Out-of-band model-checkpoint uploader: pushes training checkpoints to a
private HF **model** repo while training is still running.

Why this exists
---------------
`train.py` writes a single rolling `runs/<RUN_NAME>/checkpoint.pt` every 30 min.
That is enough to survive a *crash*, and nothing else. `workspace_is_volume` is
false on this box, so a recycle or a reclaim takes the disk with it -- and
AGENT.md SS0.2 is explicit: *"Never store state only on this box... If it isn't
pushed, it doesn't exist."* `hero` is ~95 GPU-hours and ~$43.70 of a budget with
no slack; losing it at hour 90 with nothing recoverable was the single largest
uninsured risk in the plan.

Why a separate process rather than an upload call in the training loop
----------------------------------------------------------------------
Exactly the reason `shard_uploader` exists, and this module deliberately mirrors
it. A multi-hundred-MB `upload_file` inline in `fit()` stalls the training loop
for the whole transfer -- on a slow or flaky link that is minutes of dead GPU per
checkpoint, and a hung connection is an indefinite hang with no watchdog signal
(the loop is alive, just blocked). Here the transfer happens in a process that
can be paused, throttled or killed without touching training.

The design constraints it is written to
---------------------------------------
* **Torch-free.** Imported by, but never importing, `train.py`'s stack. The
  watcher process holds ~150 MB RSS instead of the ~2 GB a torch import would
  add, which matters under ADDENDUM 2's 20 GB ceiling.
* **Sealed payloads only.** A payload is uploaded only once its `.upload.json`
  sidecar exists, which the writer emits *after* the payload is fully written
  and renamed into place. Uploading a file still being `torch.save`d would push
  a truncated checkpoint that looks fine until the day it is needed.
* **Unique local names, fixed repo paths.** The rolling checkpoint is staged as
  `weights-step<N>.pt` but always uploaded to the same `path_in_repo`. If the
  local name were fixed, the trainer's next write could overwrite the file
  mid-upload; unique names make that race structurally impossible.
* **Supersede rather than queue.** If uploads fall behind the trainer, only the
  newest pending payload for a given repo path is sent and the older ones are
  deleted unsent. A superseded rolling checkpoint has no value, and uploading it
  would consume the very bandwidth the current one needs.
* **Bounded disk.** Payloads are deleted after a confirmed upload, so the outbox
  holds at most a cadence's worth rather than growing across a four-day run.
* **Never interferes.** Every failure is caught and retried next pass. Missing
  token or repo is a no-op, not an error, so it can be left running unattended.

The two kinds of checkpoint
---------------------------
**rolling** -- weights only, bf16, ~321 MB, every ~2 h, to a fixed path on its
own branch. Pure disaster recovery: the answer to "the box vanished at hour 90".
AGENT.md SS0.4 asks for weights-only at intervals and full optimizer state only
at milestones, which is what the bf16 cast buys -- a bf16 round-trip of the fp32
master weights costs ~0.4% relative error and a few hundred steps of recovery,
against losing four days.

**milestone** -- full fp32 weights *and* both optimizer states, written once at
the step where WSD's decay phase begins, under its own revision that the rolling
upload cannot reach. This is the reusable branch point: WSD's practical edge over
cosine is that the pre-decay checkpoint can be continued on more or different
data and then re-decayed, whereas continuing from an annealed lr~0 model needs a
re-warmup from a converged state and is measurably worse. Optimizer state is
included because a resume without Muon's momentum buffers and AdamW's moments
has to rebuild them and loses ground.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional, Tuple

# In code rather than only in `.env` on purpose. The overnight chain sources
# `.env` once at launch, so a variable added later never reaches the jobs it
# starts -- and a repo id is not a secret. Overridable with
# DAEDALUS_HF_MODEL_REPO or `--hub-repo`.
DEFAULT_MODEL_REPO = "Unseen1980/daedalus-checkpoints"

SIDECAR_SUFFIX = ".upload.json"
STATE_FILENAME = "uploaded.json"
OUTBOX_DIRNAME = "hub_outbox"
HUB_URI_PREFIX = "hub://"

# Floor on the wall-clock ceiling for a single upload pass. It exists only to
# convert an indefinite hang into a retry. See `upload_once_bounded` for why a
# *process* is what enforces it, and why the real guard is progress rather than
# this number.
#
# It used to be the *whole* bound, and on 2026-08-12 that cost `hero` nine hours
# of insurance. The box's uplink fell from ~360 KB/s to ~105 KB/s, which puts a
# 321 MB rolling checkpoint at ~51 min -- over this ceiling. Every pass was then
# SIGKILLed at 900 s having sent ~90 MB, and the next pass restarted from zero,
# forever. Not a hang: a livelock, and one that consumed the entire uplink while
# never delivering a byte. A fixed wall-clock bound cannot tell "wedged" from
# "slow", and on a rented box the link speed is not ours to assume.
DEFAULT_PASS_DEADLINE_S = 900.0

# Failing to move STALL_MIN_PROGRESS_BYTES within this window is what defines
# "wedged". Still below the old 900 s ceiling, so the CLOSE-WAIT wedge this
# bound was written for is caught sooner than before, while a slow-but-moving
# transfer is left alone.
#
# Not tighter than this, deliberately. A killed pass restarts from zero, so a
# false positive costs a whole transfer while a slow true positive costs
# minutes. (Only *mostly* from zero: the Hub dedupes LFS blobs by hash, so if
# the blob had already landed the retry's preupload check skips it. That is
# what makes a rare false positive survivable rather than ruinous.)
DEFAULT_STALL_TIMEOUT_S = 600.0

# A window must carry at least this much or the pass is wedged. A floor rather
# than "no progress at all" because a wedge is not silent: caught live on
# 2026-08-12, a `hero` upload sat in CLOSE-WAIT -- peer gone, 25 bytes stuck in
# the recv queue, nothing left to send to -- while still dribbling ~32 B/s.
# A strict zero-progress test would have watched that forever.
#
# 64 KB per 600 s is ~109 B/s: ~3x the observed trickle, and ~1000x below the
# slowest healthy transfer yet measured here (105 KB/s, which carries 63 MB in
# the same window). The two regimes are three orders of magnitude apart, so the
# threshold does not need to be precise -- only to sit between them.
STALL_MIN_PROGRESS_BYTES = 64 * 1024

# The backstop ceiling is derived from what is pending at this floor rate, so it
# scales with the payload instead of assuming a link speed. 20 KB/s is well
# under the worst rate yet measured here (105 KB/s), so a healthy transfer never
# meets it; it only stops a pass that crawls without ever technically stalling.
MIN_PROGRESS_RATE_BPS = 20 * 1024


def _proc_io_bytes(pid: int) -> Optional[int]:
    """Bytes this process has read+written so far, or None if unreadable.

    Both counters, not just `wchar`: a pass spends its first minutes hashing the
    payload (`rchar` climbs, `wchar` flat) and the rest sending it (`wchar`
    climbs). Watching either alone reads the other phase as a stall.

    `rchar`/`wchar` rather than `read_bytes`/`write_bytes` because those count
    only actual block-device traffic -- a socket send never touches the disk, so
    the transfer we care most about would be invisible.
    """
    try:
        total = 0
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key in ("rchar", "wchar"):
                    total += int(value.strip())
        return total
    except (OSError, ValueError):
        return None  # no /proc, or the child already exited


def outbox_dir(run_dir: str) -> str:
    return os.path.join(run_dir, OUTBOX_DIRNAME)


# ------------------------------------------------------------------ state ---

def _load_state(path: str) -> dict:
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"uploads": {}}
    state.setdefault("uploads", {})
    return state


def _save_state(path: str, state: dict) -> None:
    # Write-then-rename, as in shard_uploader: a crash mid-write must not leave
    # a truncated state file behind.
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------- staging ---

def stage(outbox: str, payload_path: str, path_in_repo: str,
          revision: str = "main", meta: Optional[dict] = None) -> str:
    """Seal an already-written payload for upload by emitting its sidecar.

    The caller must have written `payload_path` completely (and atomically)
    *before* calling this -- the sidecar's existence is the only signal the
    uploader has that a file is safe to read.
    """
    os.makedirs(outbox, exist_ok=True)
    record = {
        "payload": os.path.basename(payload_path),
        "path_in_repo": path_in_repo,
        "revision": revision,
        "size": os.path.getsize(payload_path),
        "staged_at": time.time(),
    }
    record.update(meta or {})
    sidecar = payload_path + SIDECAR_SUFFIX
    tmp = sidecar + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, sidecar)
    return sidecar


def pending_uploads(outbox: str) -> List[Tuple[str, dict]]:
    """`(sidecar_path, record)` for every sealed payload still on disk, oldest
    first. A sidecar whose payload is missing (already uploaded and deleted, or
    removed by hand) is cleaned up rather than retried forever."""
    pending = []
    if not os.path.isdir(outbox):
        return pending
    for name in sorted(os.listdir(outbox)):
        if not name.endswith(SIDECAR_SUFFIX):
            continue
        sidecar = os.path.join(outbox, name)
        try:
            with open(sidecar) as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # being written right now; next pass picks it up
        payload = os.path.join(outbox, record.get("payload", ""))
        if not os.path.isfile(payload):
            try:
                os.remove(sidecar)
            except OSError:
                pass
            continue
        # A payload smaller than the sidecar claims means the writer was
        # interrupted between rename and seal; wait rather than push a stub.
        if os.path.getsize(payload) != record.get("size"):
            continue
        pending.append((sidecar, record))
    pending.sort(key=lambda sr: sr[1].get("staged_at", 0))
    return pending


def _partition_superseded(pending):
    """Split pending into `(to_send, superseded)`. Two payloads bound for the
    same `revision:path_in_repo` are the same slot; only the newest is worth
    the bandwidth."""
    newest = {}
    for sidecar, record in pending:
        key = f"{record.get('revision')}:{record.get('path_in_repo')}"
        prev = newest.get(key)
        if prev is None or record.get("staged_at", 0) >= prev[1].get("staged_at", 0):
            newest[key] = (sidecar, record)
    keep = {id(sr[0]) for sr in newest.values()}
    to_send, superseded = [], []
    for item in pending:
        (to_send if id(item[0]) in keep else superseded).append(item)
    return to_send, superseded


def _discard(sidecar: str, outbox: str, record: dict) -> None:
    for path in (os.path.join(outbox, record.get("payload", "")), sidecar):
        try:
            os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------ connectivity ---

_ipv4_client_installed = False


def _install_ipv4_client_factory() -> None:
    """Force huggingface_hub's shared httpx client onto IPv4, and give it a
    connect timeout. Idempotent; safe to call before every `HfApi()`.

    Root cause found live on 2026-08-12: this container has no IPv6 route at
    all (`ip -6 route show` is empty -- only loopback has an IPv6 address),
    but `huggingface.co` resolves to AAAA (IPv6) records preferentially
    (`getent hosts huggingface.co` returns only IPv6 addresses), and
    huggingface_hub's default client factory builds `httpx.Client(timeout=
    None)` -- no timeout at all. Every connection attempt tried IPv6 first,
    on a route that does not exist, and had nothing to time it out: measured
    hangs of 90s+ moving 0 bytes, `curl -6` hanging past a 5s cap outright,
    while `curl -4` to the same host completed in ~10s and a plain IPv4-
    forced `httpx.Client` request took 0.7s. This is almost certainly also
    the "CLOSE-WAIT wedge" and "0 B/s trickle" failures logged earlier
    (2026-08-10, 2026-08-12) -- both consistent with a connect that hung
    instead of failing.

    `local_address="0.0.0.0"` on `httpx.HTTPTransport` is the documented way
    to pin the outbound socket family to IPv4 (httpcore binds the local
    endpoint, which forces `AF_INET`). Read/write/pool timeouts are left at
    `None` deliberately -- a multi-hundred-MB checkpoint legitimately takes
    minutes on this box's uplink, and `upload_once_bounded` already bounds
    the whole pass from outside via `/proc` IO progress; only *connect* ever
    hung here, so only *connect* gets a timeout.
    """
    global _ipv4_client_installed
    if _ipv4_client_installed:
        return
    import httpx
    from huggingface_hub.utils import _http

    def _ipv4_client_factory() -> "httpx.Client":
        transport = httpx.HTTPTransport(local_address="0.0.0.0")
        return httpx.Client(
            transport=transport,
            event_hooks={"request": [_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=httpx.Timeout(connect=15.0, read=None, write=None, pool=None),
        )

    # `set_client_factory` is a private huggingface_hub API and is not present
    # in every release. Forcing IPv4 is an optimisation for one hosting
    # provider whose IPv6 path black-holes large uploads, not a requirement --
    # so when the hook is missing, carry on with the default client rather than
    # making the uploader unimportable.
    setter = getattr(_http, "set_client_factory", None)
    if setter is None:
        print("note: this huggingface_hub has no set_client_factory; using the "
              "default HTTP client (IPv4 pinning unavailable)")
        _ipv4_client_installed = True
        return
    setter(_ipv4_client_factory)
    _ipv4_client_installed = True


# --------------------------------------------------------------- uploading ---

_CLOSED_CLIENT_MARKER = "client has been closed"


def _call_with_reconnect(fn, attempts: int = 3, log=print):
    """Call `fn()`, retrying immediately if huggingface_hub's shared httpx
    client was left closed by a transient `ConnectError` (e.g. a DNS blip).

    `_http_backoff_base` (huggingface_hub.utils._http, measured on 1.18.0)
    calls `close_session()` when it catches a `ConnectError` mid-retry, but
    keeps calling `.request()` on the *local* `client` variable it captured
    before the loop started -- it never re-fetches `get_session()`. Every
    remaining retry in that same call then fails with `RuntimeError("Cannot
    send a request, as the client has been closed.")`, even though the
    network recovered within the same retry window. Caught live on
    2026-08-12: a single isolated DNS failure (measured ~1 in 10-30 lookups
    on this box, not sustained) turned into hours of "leaving pending" on
    every pass, because each pass's first hiccup doomed the whole pass.

    `close_session()` sets the module-level client to `None` *before*
    raising back out, so by the time this except sees the RuntimeError, the
    next top-level call (this retry) calls `get_session()` fresh and just
    works -- no backoff needed, since the underlying network was never
    actually down for more than the original blip.

    The marker can show up wrapped rather than bare: `_commit_api.py`'s
    `_wrapped_lfs_upload` catches *any* exception from the real chunked LFS
    transfer and re-raises `RuntimeError("Error while uploading '<path>' to
    the Hub.") from exc` -- the closed-client error becomes `__cause__`, not
    the message on `str(e)`. Caught live: the actual failing log line during
    `hero` was exactly that wrapped form, which a bare `str(e)` check misses
    entirely. So the marker is searched down the whole `__cause__` chain.
    """
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return fn()
        except RuntimeError as e:
            chain, cur = [e], e
            while cur.__cause__ is not None:
                cur = cur.__cause__
                chain.append(cur)
            if not any(_CLOSED_CLIENT_MARKER in str(exc) for exc in chain) \
                    or attempt == attempts - 1:
                raise
            last = e
            log(f"ckpt-uploader: hub client was left closed by a transient "
                f"connect error ({e!r}); retrying immediately "
                f"({attempt + 1}/{attempts})")
    raise last  # pragma: no cover -- unreachable, loop always returns or raises


def upload_once(outbox: str, repo_id: str, token: Optional[str] = None,
                state_path: Optional[str] = None, log=print) -> dict:
    """One pass: upload every sealed payload, newest-per-slot only. Returns a
    summary. Never raises -- a failed payload stays pending for the next pass."""
    from huggingface_hub import HfApi

    _install_ipv4_client_factory()
    state_path = state_path or os.path.join(outbox, STATE_FILENAME)
    summary = {"uploaded": 0, "failed": 0, "superseded": 0, "bytes": 0}
    pending = pending_uploads(outbox)
    if not pending:
        return summary

    to_send, superseded = _partition_superseded(pending)
    for sidecar, record in superseded:
        _discard(sidecar, outbox, record)
        summary["superseded"] += 1
        log(f"ckpt-uploader: dropping superseded {record.get('payload')} "
            f"(a newer checkpoint for {record.get('path_in_repo')} is pending)")

    api = HfApi(token=token)
    try:
        _call_with_reconnect(
            lambda: api.create_repo(repo_id, repo_type="model", private=True,
                                    exist_ok=True), log=log)
    except Exception as e:
        log(f"ckpt-uploader: cannot reach {repo_id} ({e!r}); will retry next pass")
        summary["failed"] = len(to_send)
        return summary

    state = _load_state(state_path)
    for sidecar, record in to_send:
        payload = os.path.join(outbox, record["payload"])
        revision = record.get("revision") or "main"
        path_in_repo = record["path_in_repo"]
        if revision != "main":
            # Idempotent: the milestone revision is created on first use and
            # reused on later passes.
            try:
                _call_with_reconnect(
                    lambda: api.create_branch(repo_id=repo_id, branch=revision,
                                              repo_type="model", exist_ok=True),
                    log=log)
            except Exception as e:
                log(f"ckpt-uploader: cannot create branch {revision} ({e!r}); "
                    f"leaving {record['payload']} pending")
                summary["failed"] += 1
                continue
        try:
            _call_with_reconnect(
                lambda: api.upload_file(
                    path_or_fileobj=payload, path_in_repo=path_in_repo,
                    repo_id=repo_id, repo_type="model", revision=revision,
                    commit_message=f"{record.get('kind', 'checkpoint')} "
                                   f"step {record.get('step')}"), log=log)
        except Exception as e:
            log(f"ckpt-uploader: {path_in_repo}@{revision} failed ({e!r}); "
                f"leaving pending")
            summary["failed"] += 1
            continue

        state["uploads"][f"{revision}:{path_in_repo}"] = {
            "step": record.get("step"), "tokens_seen": record.get("tokens_seen"),
            "kind": record.get("kind"), "size": record.get("size"),
            "uploaded_at": time.time(),
        }
        _save_state(state_path, state)
        summary["uploaded"] += 1
        summary["bytes"] += record.get("size", 0)
        log(f"ckpt-uploader: {path_in_repo}@{revision} step {record.get('step')} "
            f"({record.get('size', 0) / 1e6:.0f} MB)")

        # A small pointer on `main` so the checkpoint that exists can be read
        # without downloading it -- what STATUS.md and a phone can check.
        _upload_pointer(api, repo_id, revision, path_in_repo, record, log)
        _discard(sidecar, outbox, record)

    return summary


def upload_once_bounded(outbox: str, repo_id: str, token: Optional[str] = None,
                        deadline_s: float = DEFAULT_PASS_DEADLINE_S,
                        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
                        log=print, poll_s: float = 10.0) -> dict:
    """One pass, in a child process that can be killed. Never raises.

    Why a subprocess rather than a timeout argument
    -----------------------------------------------
    `upload_once` catches every exception and leaves a failed payload pending
    for the next pass, which is correct and was also unreachable: on
    2026-08-10 the uploader wedged mid-transfer of the 1.4 GB milestone with
    its socket in CLOSE-WAIT -- the server had sent FIN and the client sat in
    a blocking send forever. A hang is not an exception, so nothing retried,
    nothing logged, and no checkpoint reached the Hub again for the rest of
    the run. `watch`'s belt-and-braces `except` could not see it either. It
    reproduced immediately on restart.

    There is no supported knob to bound it in-process:

    * `HfApi.upload_file` takes no timeout (huggingface_hub 1.18.0), and the
      `HF_HUB_*_TIMEOUT` constants cover downloads and ETag probes only.
    * `socket.setdefaulttimeout()` does **not** reach it -- measured against
      urllib3 2.7.0 with a black-hole listener, a GET still hung with the
      default set to 3 s.
    * A worker *thread* cannot be killed, so a wedged one would leak its
      socket and ~1 GB of RSS on every pass -- and RAM, not VRAM, is the
      scarce resource here (ADDENDUM 2).

    A child process has none of those problems: `subprocess.run(timeout=...)`
    SIGKILLs it, which returns the socket, the fd and the memory. SIGTERM is
    not enough on its own -- the wedged uploader ignored it, because CPython
    can only run a signal handler between bytecodes and the main thread was
    blocked in a C-level socket call.

    Nothing is lost when a pass is killed: payloads and their sidecars stay in
    the outbox, `uploaded.json` is only written after a confirmed upload, and
    `_discard` runs only on success. The next pass repeats the attempt. That
    matters because the wedge is transient rather than a size limit -- a
    1435.3 MB milestone had already uploaded successfully on an earlier run.

    Why the bound is progress and not wall-clock
    --------------------------------------------
    "Nothing is lost" holds for *one* kill. It stops holding when the deadline
    is shorter than the transfer, because then every pass is killed and the
    retry is the same doomed attempt -- a livelock that saturates the uplink and
    delivers nothing. That is what happened on 2026-08-12 (see
    `DEFAULT_PASS_DEADLINE_S`), and no choice of constant prevents it: the link
    is rented and its speed changes under us.

    So the pass is killed when it stops *moving*, measured from outside via
    `/proc/<pid>/io`, with the wall-clock ceiling demoted to a backstop derived
    from the pending bytes. This is strictly stronger than the old bound in both
    directions -- a genuine wedge dies in 5 min instead of 15, and a healthy
    transfer is no longer capped at an arbitrary size-blind number.

    Where `/proc` is unreadable the ceiling alone still applies, so the
    behaviour degrades to exactly what it was before.
    """
    import subprocess
    import tempfile

    cmd = [sys.executable, "-m", "daedalus.ckpt_uploader",
           "--outbox", outbox, "--repo", repo_id, "--once"]
    env = dict(os.environ)
    if token:
        env["HF_TOKEN_WRITE"] = token
    # The package root, so the child can import `daedalus` wherever the parent
    # was started from.
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    failed = {"uploaded": 0, "failed": 1, "superseded": 0, "bytes": 0}

    # Scale the backstop to the work in hand. `pending_uploads` is a directory
    # listing and cannot raise past its own try, but a bad outbox must not take
    # the pass down with it.
    try:
        pending_bytes = sum(int(rec.get("size") or 0)
                            for _, rec in pending_uploads(outbox))
    except Exception:
        pending_bytes = 0
    ceiling_s = max(deadline_s, pending_bytes / MIN_PROGRESS_RATE_BPS)

    # Temp files rather than pipes: this waits in a poll loop instead of
    # `communicate()`, and a child that filled a 64 KB pipe buffer would block
    # forever with nothing draining it -- reintroducing the very hang the
    # subprocess exists to bound.
    try:
        out_f = tempfile.TemporaryFile(mode="w+")
        err_f = tempfile.TemporaryFile(mode="w+")
    except Exception as e:
        log(f"ckpt-uploader: could not run a pass ({e!r}); will retry next pass")
        return failed

    try:
        with out_f, err_f:
            try:
                proc = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                                        stdout=out_f, stderr=err_f)
            except Exception as e:  # could not spawn at all
                log(f"ckpt-uploader: could not run a pass ({e!r}); "
                    f"will retry next pass")
                return failed

            started = time.time()
            # `window_base` is the counter when the current window opened; a
            # window that carries its quota resets it. Comparing against the
            # window start rather than the previous sample is what makes a
            # trickle count as a stall -- sample-to-sample it always "moved".
            window_base = _proc_io_bytes(proc.pid)
            last_io = window_base
            window_start = started
            killed = None
            while proc.poll() is None:
                time.sleep(poll_s)
                now = time.time()
                io_now = _proc_io_bytes(proc.pid)
                if io_now is not None:
                    last_io = io_now
                    if window_base is None:
                        window_base, window_start = io_now, now
                    elif io_now - window_base >= STALL_MIN_PROGRESS_BYTES:
                        window_base, window_start = io_now, now
                    elif now - window_start > stall_timeout_s:
                        moved = io_now - window_base
                        # "io" rather than "sent": this counter is read+write
                        # combined, so it includes hashing the payload and is
                        # much larger than the bytes that reached the network.
                        # Calling it "sent" made the first live kill read as if
                        # 997 MB of a 321 MB file had gone out.
                        killed = (f"moved only {moved / 1024:.0f} KB in "
                                  f"{now - window_start:.0f}s "
                                  f"({last_io / 1e6:.0f} MB of process io)")
                # Independent of the stall check, not an `elif` on it: chaining
                # the two made the backstop unreachable whenever /proc *was*
                # readable -- i.e. always, in production. The ceiling has to
                # hold for a pass that keeps clearing its quota and still never
                # ends.
                if not killed and now - started > ceiling_s:
                    killed = (f"exceeded its {ceiling_s / 60:.1f} min ceiling "
                              f"for {pending_bytes / 1e6:.1f} MB pending")
                if killed:
                    proc.kill()
                    proc.wait()
                    break

            if killed:
                log(f"ckpt-uploader: pass killed -- it {killed}; payloads stay "
                    f"pending and the next pass retries")
                return failed

            out_f.seek(0)
            stdout = out_f.read()
            err_f.seek(0)
            stderr = err_f.read()
    except Exception as e:  # the poll loop itself must never take the watcher down
        log(f"ckpt-uploader: pass failed while supervising it ({e!r}); "
            f"will retry next pass")
        return failed

    # Re-emit the child's own lines so the training log keeps saying which
    # checkpoint landed and how big it was.
    for line in (stdout or "").splitlines():
        if line.startswith("ckpt-uploader:"):
            log(line)
    if proc.returncode != 0:
        tail = (stderr or "").strip().splitlines()[-3:]
        log(f"ckpt-uploader: pass exited {proc.returncode}; " + " | ".join(tail))
        return failed
    try:
        start = stdout.index("{")
        return json.loads(stdout[start:])
    except (ValueError, AttributeError, json.JSONDecodeError):
        # It ran and said nothing parseable -- treat as a no-op rather than a
        # failure, since the pass may legitimately have had nothing to send.
        return {"uploaded": 0, "failed": 0, "superseded": 0, "bytes": 0}


def _upload_pointer(api, repo_id: str, revision: str, path_in_repo: str,
                    record: dict, log=print) -> None:
    pointer = {
        "revision": revision, "path_in_repo": path_in_repo,
        "step": record.get("step"), "tokens_seen": record.get("tokens_seen"),
        "kind": record.get("kind"), "run_name": record.get("run_name"),
        "updated_at": time.time(),
    }
    # Run-scoped, like the checkpoint paths themselves: one pointer per job, so
    # `hero`'s does not get replaced by `abl-arch`'s.
    run = record.get("run_name")
    name = f"latest-{record.get('kind', 'checkpoint')}"
    name = f"{name}-{run}.json" if run else f"{name}.json"
    try:
        _call_with_reconnect(
            lambda: api.upload_file(
                path_or_fileobj=json.dumps(pointer, indent=2).encode(),
                path_in_repo=name, repo_id=repo_id, repo_type="model",
                revision="main", commit_message=f"pointer: {name}"), log=log)
    except Exception as e:
        # Cosmetic only -- the checkpoint itself is already safely uploaded.
        log(f"ckpt-uploader: pointer {name} failed ({e!r}); checkpoint is fine")


LOCK_FILENAME = "uploader.pid"


def _process_is_an_uploader(pid: int) -> bool:
    """True if `pid` is alive *and* is a ckpt_uploader. The cmdline check is
    what stops PID reuse from making a stale lock look live forever."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            # The dotted module path, not the bare name: a bare "ckpt_uploader"
            # also matches `pytest tests/test_ckpt_uploader.py`, so a recycled
            # PID running the suite would look like a live holder and block
            # uploads for the rest of the run.
            return b"daedalus.ckpt_uploader" in f.read()
    except OSError:
        return False  # no /proc (or gone): treat as not ours


def uploader_is_live(outbox: str) -> bool:
    """Is *some* uploader currently serving `outbox`?

    Read-only companion to `acquire_lock`, for the trainer rather than for an
    uploader: it must not take the lock, only report on it.

    Needed because the process serving a resumed run is usually not the
    trainer's own child. A SIGKILLed trainer cannot reap its uploader, so the
    orphan keeps the lock and keeps delivering, and every later attempt's
    uploader exits on that lock by design. `self.uploader_proc.poll()` therefore
    says "dead" for the rest of the run while uploads are perfectly healthy --
    so liveness cannot be read off the child handle alone.
    """
    path = os.path.join(outbox, LOCK_FILENAME)
    try:
        with open(path) as f:
            holder = int(f.read().strip())
    except (OSError, ValueError):
        return False
    return _process_is_an_uploader(holder)


def acquire_lock(outbox: str, log=print) -> bool:
    """One uploader per outbox.

    `hero.py` restarts `train.py` after a crash, and a SIGKILLed trainer cannot
    reap the uploader it spawned. Without this, a four-day run accumulates one
    orphaned watcher per restart, all polling the same directory and racing to
    delete each other's payloads. The incumbent keeps the lock -- it is already
    draining the outbox, so the new process is the redundant one.
    """
    os.makedirs(outbox, exist_ok=True)
    path = os.path.join(outbox, LOCK_FILENAME)
    try:
        with open(path) as f:
            holder = int(f.read().strip())
    except (OSError, ValueError):
        holder = None
    if holder is not None and holder != os.getpid() and _process_is_an_uploader(holder):
        log(f"ckpt-uploader: pid {holder} already owns {outbox}; exiting")
        return False
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return True


def watch(outbox: str, repo_id: str, token: Optional[str] = None,
          interval_s: float = 300.0, max_passes: Optional[int] = None,
          log=print, sleep=time.sleep,
          deadline_s: float = DEFAULT_PASS_DEADLINE_S,
          stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S) -> dict:
    """Poll `outbox` forever, uploading whatever the trainer has sealed.

    The poll interval is much shorter than the ~2 h staging cadence on purpose:
    polling is a directory listing, and a short interval means a checkpoint is
    off the box within minutes of being written rather than hours.

    Each pass runs under a hard deadline in its own process
    (`upload_once_bounded`) because a wedged transfer is a hang, not an
    exception, and the `except` below cannot see one.
    """
    totals = {"uploaded": 0, "failed": 0, "superseded": 0, "bytes": 0, "passes": 0}
    while max_passes is None or totals["passes"] < max_passes:
        try:
            summary = upload_once_bounded(outbox, repo_id, token=token,
                                          deadline_s=deadline_s,
                                          stall_timeout_s=stall_timeout_s,
                                          log=log)
        except Exception as e:  # belt and braces: this process must never die
            log(f"ckpt-uploader: pass failed unexpectedly ({e!r}); continuing")
            summary = {"uploaded": 0, "failed": 1, "superseded": 0, "bytes": 0}
        for k in ("uploaded", "failed", "superseded", "bytes"):
            totals[k] += summary.get(k, 0)
        totals["passes"] += 1
        if max_passes is None or totals["passes"] < max_passes:
            sleep(interval_s)
    return totals


# --------------------------------------------------------------- restoring ---

def parse_hub_uri(uri: str) -> Tuple[str, str, str]:
    """`hub://<owner>/<name>/<path...>[?rev=<revision>]` -> (repo_id, path, revision).

    A URI rather than three more CLI flags because `--resume` is already
    threaded through `hero.py` and `abl_arch.py`; this keeps one knob and lets
    a Hub checkpoint be dropped in anywhere a local path works.
    """
    if not uri.startswith(HUB_URI_PREFIX):
        raise ValueError(f"not a hub URI: {uri!r}")
    body = uri[len(HUB_URI_PREFIX):]
    revision = "main"
    if "?" in body:
        body, query = body.split("?", 1)
        for part in query.split("&"):
            if part.startswith("rev="):
                revision = part[len("rev="):]
    segments = [s for s in body.split("/") if s]
    if len(segments) < 3:
        raise ValueError(
            f"hub URI needs owner/name/path, got {uri!r} "
            f"(e.g. hub://me/daedalus-ckpt/rolling/weights.pt?rev=rolling)")
    repo_id = "/".join(segments[:2])
    return repo_id, "/".join(segments[2:]), revision


def download_checkpoint(repo_id: str, path_in_repo: str, dest_dir: str,
                        revision: str = "main", token: Optional[str] = None,
                        log=print) -> str:
    """Fetch a checkpoint from the Hub into `dest_dir` and return its local
    path. Deliberately separate from `load_checkpoint` so the restore path can
    be exercised without a GPU, and so this module stays torch-free."""
    from huggingface_hub import hf_hub_download

    _install_ipv4_client_factory()
    os.makedirs(dest_dir, exist_ok=True)
    log(f"ckpt-uploader: downloading {repo_id}:{path_in_repo}@{revision} -> {dest_dir}")
    local = hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                            repo_type="model", revision=revision,
                            local_dir=dest_dir, token=token)
    log(f"ckpt-uploader: got {local} ({os.path.getsize(local) / 1e6:.0f} MB)")
    return local


def resolve_resume(spec: Optional[str], dest_dir: str,
                   token: Optional[str] = None, log=print) -> Optional[str]:
    """Pass a local path straight through; materialise a `hub://` URI first.
    This is the whole of "resume from the Hub" as far as callers are
    concerned."""
    if not spec or not spec.startswith(HUB_URI_PREFIX):
        return spec
    repo_id, path_in_repo, revision = parse_hub_uri(spec)
    return download_checkpoint(repo_id, path_in_repo, dest_dir,
                               revision=revision, token=token, log=log)


# --------------------------------------------------------------------- cli ---

def _cli(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Upload training checkpoints to the Hub, out of band from "
                    "training itself.")
    p.add_argument("--outbox", required=True,
                   help=f"runs/<name>/{OUTBOX_DIRNAME}")
    p.add_argument("--repo", required=True, help="private HF model repo id")
    p.add_argument("--interval-s", type=float, default=300.0)
    p.add_argument("--pass-deadline-s", type=float, default=DEFAULT_PASS_DEADLINE_S,
                   help="floor on the backstop ceiling for a pass; the real "
                        "ceiling scales with the bytes pending")
    p.add_argument("--stall-timeout-s", type=float, default=DEFAULT_STALL_TIMEOUT_S,
                   help="kill and retry a pass that makes no progress at all "
                        "for this long")
    p.add_argument("--once", action="store_true", help="single pass, then exit")
    p.add_argument("--download", default=None,
                   help="hub:// URI to fetch instead of uploading")
    p.add_argument("--dest", default=None, help="destination dir for --download")
    args = p.parse_args(argv)

    token = os.environ.get("HF_TOKEN_WRITE")
    if args.download:
        repo_id, path_in_repo, revision = parse_hub_uri(args.download)
        print(download_checkpoint(repo_id, path_in_repo, args.dest or ".",
                                  revision=revision, token=token))
        return
    if not token:
        print("ckpt-uploader: HF_TOKEN_WRITE not set; nothing to do")
        return
    if args.once:
        print(json.dumps(upload_once(args.outbox, args.repo, token=token), indent=2))
        return
    # ~48 uploads over hero's four days, each a multi-line tqdm bar, would bury
    # the training log in redraw noise. The per-file line this module logs
    # itself carries the same information.
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    if not acquire_lock(args.outbox):
        return
    watch(args.outbox, args.repo, token=token, interval_s=args.interval_s,
          deadline_s=args.pass_deadline_s,
          stall_timeout_s=args.stall_timeout_s)


if __name__ == "__main__":
    _cli()
