"""Out-of-band shard uploader: pushes finished dataprep shards to the Hub
while `dataprep` is still building the rest.

Why this exists as its own process rather than `run_dataprep`'s `--hf-repo`:
that path calls `upload_shards` *inline in the parent's event loop*, between
`wait()` calls. A multi-GB `upload_folder` there blocks the memory poll
(ADDENDUM 2 rule 4) and the respawn dispatch for its whole duration, which is
precisely the class of stall that has already cost this project hours. Here
the upload runs in a separate process that can be paused, throttled or killed
without touching the build.

Why it is needed at all: `workspace_is_volume` is false on this box, so
nothing on disk survives a recycle or destroy, and AGENT.md §0.2 is explicit
that state must not live only here. A ~60 h corpus build is far too much to
hold in one place.

Design constraints it is written to:

* **Shard-at-a-time, never folder-at-a-time.** `upload_folder` would re-walk
  and re-hash the whole tree on every pass; the corpus is ~90 GB. Each `.bin`
  is uploaded individually and recorded, so a pass over an already-uploaded
  corpus costs a JSON read and nothing else.
* **Only sealed shards.** A `.bin` is only uploaded once a per-source
  manifest lists it, which `ShardWriter` writes after `_flush` completes.
  Uploading a file still being written would push a truncated shard.
* **Bounded memory.** State is a flat dict of filename -> size; uploads
  stream from disk. Nothing proportional to corpus size is held.
* **Never interferes.** Every failure is caught and retried on the next pass.
  It is a no-op if the token is missing, rather than an error, so it can be
  left running unattended.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

STATE_FILENAME = "uploaded.json"


def _load_state(path: str) -> dict:
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"files": {}}
    state.setdefault("files", {})
    return state


def _save_state(path: str, state: dict) -> None:
    # Write-then-rename: a crash mid-write must not leave a truncated state
    # file, which would re-upload the entire corpus on the next pass.
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def pending_shards(out_root: str, state: dict) -> List[Tuple[str, str, int]]:
    """`(source_key, filename, size_bytes)` for every shard that a per-source
    manifest lists, exists on disk at its full size, and has not been
    uploaded at that size yet.

    Size is part of the identity on purpose: a shard that was re-flushed by a
    from-scratch retry of its source has the same name but different content,
    and must be re-uploaded rather than assumed current."""
    pending = []
    files = state.get("files", {})
    if not os.path.isdir(out_root):
        return pending
    for key in sorted(os.listdir(out_root)):
        source_dir = os.path.join(out_root, key)
        manifest_path = os.path.join(source_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue  # nothing sealed yet for this source
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # being rewritten right now; next pass picks it up
        for shard in manifest.get("shards", []):
            name = shard.get("file")
            if not name:
                continue
            path = os.path.join(source_dir, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if files.get(f"{key}/{name}") == size:
                continue
            pending.append((key, name, size))
    return pending


def upload_once(out_root: str, repo_id: str, token: Optional[str] = None,
                 state_path: Optional[str] = None, max_files: Optional[int] = None,
                 log=print) -> dict:
    """One pass: upload every pending shard, plus each source's manifest.
    Returns a summary dict. Never raises -- a failed file is simply left
    pending for the next pass."""
    from huggingface_hub import HfApi

    # Same IPv6-route/no-timeout hang ckpt_uploader hit live on 2026-08-12
    # (no IPv6 route on this container, huggingface.co resolves AAAA-first,
    # default httpx client has no connect timeout) -- shares the fix rather
    # than duplicating it, since both modules use the same shared client.
    from daedalus.ckpt_uploader import _install_ipv4_client_factory
    _install_ipv4_client_factory()

    state_path = state_path or os.path.join(out_root, STATE_FILENAME)
    state = _load_state(state_path)
    pending = pending_shards(out_root, state)
    if max_files is not None:
        pending = pending[:max_files]
    summary = {"uploaded": 0, "failed": 0, "bytes": 0, "pending_after": 0}
    if not pending:
        return summary

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        log(f"shard-uploader: cannot reach {repo_id} ({e!r}); will retry next pass")
        summary["failed"] = len(pending)
        return summary

    touched_sources = set()
    for key, name, size in pending:
        local = os.path.join(out_root, key, name)
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=f"{key}/{name}",
                             repo_id=repo_id, repo_type="dataset")
        except Exception as e:
            log(f"shard-uploader: {key}/{name} failed ({e!r}); leaving pending")
            summary["failed"] += 1
            continue
        state["files"][f"{key}/{name}"] = size
        summary["uploaded"] += 1
        summary["bytes"] += size
        touched_sources.add(key)
        # Persist after every file: an interrupted pass must not re-upload
        # gigabytes it already pushed.
        _save_state(state_path, state)
        log(f"shard-uploader: {key}/{name} ({size / 1e9:.2f} GB)")

    # The manifest is small and changes every chunk boundary, so it is
    # re-uploaded whenever any of its shards moved rather than tracked.
    for key in sorted(touched_sources):
        manifest_path = os.path.join(out_root, key, "manifest.json")
        try:
            api.upload_file(path_or_fileobj=manifest_path,
                             path_in_repo=f"{key}/manifest.json",
                             repo_id=repo_id, repo_type="dataset")
        except Exception as e:
            log(f"shard-uploader: {key}/manifest.json failed ({e!r})")

    summary["pending_after"] = len(pending_shards(out_root, state))
    return summary


def watch(out_root: str, repo_id: str, token: Optional[str] = None,
           interval_s: float = 300.0, max_files_per_pass: Optional[int] = 4,
           max_passes: Optional[int] = None, log=print,
           sleep=time.sleep) -> dict:
    """Poll `out_root` forever, uploading whatever is newly sealed.

    `max_files_per_pass` throttles deliberately: this shares ~660 Mbit/s with
    dataprep's own streaming downloads, and starving the build to push data
    faster would be a bad trade. Four ~200 MB shards per 5 minutes is well
    under the link and still drains faster than the build produces.
    """
    totals = {"uploaded": 0, "failed": 0, "bytes": 0, "passes": 0}
    while max_passes is None or totals["passes"] < max_passes:
        summary = upload_once(out_root, repo_id, token=token,
                               max_files=max_files_per_pass, log=log)
        totals["uploaded"] += summary["uploaded"]
        totals["failed"] += summary["failed"]
        totals["bytes"] += summary["bytes"]
        totals["passes"] += 1
        if summary["uploaded"]:
            log(f"shard-uploader: pass {totals['passes']}: "
                f"+{summary['uploaded']} file(s), {summary['pending_after']} still pending, "
                f"{totals['bytes'] / 1e9:.1f} GB total this session")
        if max_passes is None or totals["passes"] < max_passes:
            sleep(interval_s)
    return totals


def _cli():
    import argparse

    p = argparse.ArgumentParser(description="Upload finished dataprep shards to the Hub, "
                                            "out of band from the build itself.")
    p.add_argument("--out", default="data/shards")
    p.add_argument("--repo", required=True, help="private HF dataset repo id")
    p.add_argument("--interval-s", type=float, default=300.0)
    p.add_argument("--max-files-per-pass", type=int, default=4,
                   help="throttle: shares bandwidth with dataprep's own streaming (0 = unlimited)")
    p.add_argument("--once", action="store_true", help="single pass, then exit")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN_WRITE")
    if not token:
        print("shard-uploader: HF_TOKEN_WRITE not set; nothing to do")
        return
    max_files = args.max_files_per_pass or None
    if args.once:
        print(json.dumps(upload_once(args.out, args.repo, token=token, max_files=max_files),
                          indent=2))
    else:
        watch(args.out, args.repo, token=token, interval_s=args.interval_s,
              max_files_per_pass=max_files)


if __name__ == "__main__":
    _cli()
