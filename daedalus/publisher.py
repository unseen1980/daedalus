"""Publish the *finished* model to the Hub: HF-format weights, the SmolLM2
tokenizer, the model card, and the Q4_0/Q8_0 GGUFs the operator actually runs.

This is the last gap in AGENT.md SS0.2 ("Never store state only on this box...
If it isn't pushed, it doesn't exist"). Three upload paths existed and none of
them covered the deliverable:

* `daedalus/shard_uploader.py` + `data.py` -> the corpus, to a *dataset* repo.
* `daedalus/ckpt_uploader.py` -> rolling and milestone `.pt` checkpoints, to a
  *model* repo. Those are resume artifacts: a `torch.save` of our own module
  tree, useless to anyone without this repo checked out.
* ...and nothing at all for the thing the project is for. A GGUF sitting in
  `runs/<run>/export/` is one lost instance away from never having existed,
  and cannot be downloaded and run by the operator.

Deliberately *not* out-of-band, unlike the other two. Those upload while
training runs, so a multi-GB transfer must never block the loop. This one runs
after everything is finished, when blocking is exactly what we want -- the
process must not exit until the artifact is safe.

Publishing is outward-facing and hard to walk back, so the defaults are the
careful ones: repos are created **private**, and going public takes an explicit
`--public`. Nothing here publishes on its own; it is called at the end of a
job, or by hand.
"""
import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple

# Same reasoning as ckpt_uploader.DEFAULT_MODEL_REPO: a repo id is not a secret,
# and putting it in code rather than only in `.env` means a chain that sourced
# `.env` hours ago still agrees with what the code does.
DEFAULT_RELEASE_REPO = "Unseen1980/daedalus-150m"

# Repositories holding artifacts that are *already released* or that a released
# model depends on. Uploading an experiment into either is unrecoverable: the
# Hub keeps superseded blobs but the repo's head is what anyone downloading gets,
# and a research probe wearing the release's name is worse than no probe at all.
#
# This is not a hypothetical. `publish_model`'s `repo_id` **defaults** to
# DEFAULT_RELEASE_REPO, so publishing a Phase 3 recovery arm correctly -- with
# private=True, a good card, and full provenance -- and simply forgetting to
# pass `--repo-id` would overwrite the released model. The research program's
# plan requires refusing a target that resolves to a released-model path, and
# the refusal belongs here rather than in each caller, because the dangerous
# case is precisely the caller that did not think about it.
PROTECTED_REPOS = (
    DEFAULT_RELEASE_REPO,             # the published 150M model
    "Unseen1980/daedalus-checkpoints",  # the released run's resume artifacts
)


class ProtectedRepoError(ValueError):
    """Raised when a publish would land on a released-model repository."""


def _normalize_repo(repo_id: str) -> str:
    """Hub repo ids compare case-insensitively and ignore surrounding space."""
    return (repo_id or "").strip().strip("/").lower()


def assert_not_released(repo_id: str,
                        protected: Iterable[str] = PROTECTED_REPOS) -> None:
    """Refuse a target that is a released-model repository.

    Callers that genuinely mean to publish the release pass
    `allow_released=True`, which is a deliberate, greppable act rather than the
    default behaviour of an argument nobody supplied.
    """
    target = _normalize_repo(repo_id)
    for candidate in protected:
        if target == _normalize_repo(candidate):
            raise ProtectedRepoError(
                f"refusing to publish into {repo_id!r}: it is a released-model "
                f"repository. Experiment artifacts belong in a separate private "
                f"repo. Pass allow_released=True (CLI: --allow-released) only "
                f"if you really are republishing the release.")

# What a publishable model directory must contain. The card is on this list on
# purpose: an artifact with no provenance, no stated success bar and no way to
# tell a 13B-token run from a 40B one is exactly what the model-card work
# existed to stop shipping, and "we'll add the README later" is how it would
# come back.
REQUIRED_FILES = ("config.json", "README.md", "tokenizer.json")

WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _weights_present(model_dir: str) -> bool:
    try:
        names = os.listdir(model_dir)
    except OSError:
        return False
    return any(n.endswith(WEIGHT_SUFFIXES) for n in names)


def check_publishable(model_dir: str) -> List[str]:
    """Reasons `model_dir` is not fit to publish, or []. Separate from the
    upload so a caller can fail the check without touching the network."""
    problems = []
    if not os.path.isdir(model_dir):
        return [f"{model_dir} is not a directory"]
    for name in REQUIRED_FILES:
        if not os.path.exists(os.path.join(model_dir, name)):
            problems.append(f"missing {name}")
    if not _weights_present(model_dir):
        problems.append(f"no weights (*.safetensors / *.bin) in {model_dir}")
    return problems


def find_ggufs(*dirs: str) -> List[str]:
    """Every .gguf under the given directories, deduplicated, sorted.

    Non-recursive on purpose. llama.cpp's conversion leaves intermediates
    around (an f16 next to the Q4_0), and both are worth publishing -- but
    walking a whole run directory would sweep up whatever else the job wrote.
    """
    found = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".gguf"):
                found.setdefault(name, os.path.join(d, name))
    return [found[k] for k in sorted(found)]


def publish_model(model_dir: str, repo_id: str = DEFAULT_RELEASE_REPO, *,
                  gguf_paths: Iterable[str] = (), token: Optional[str] = None,
                  private: bool = True, commit_message: Optional[str] = None,
                  allow_released: bool = False,
                  extra_files: Iterable[Tuple[str, str]] = (),
                  api=None, log=print) -> dict:
    """Upload `model_dir` plus any GGUFs to `repo_id` as a Hub *model* repo.

    Raises ValueError if the directory is not publishable -- unlike the
    checkpoint uploader, which must never raise because it runs inside a
    training loop. Here a failure is the last thing standing between a finished
    model and nobody being able to use it, so it should be loud.

    Refuses a released-model `repo_id` unless `allow_released` is set; see
    `assert_not_released`. The check runs *before* the publishability check so
    the wrong-target error is the one reported, rather than a missing-file
    complaint about a directory that was never going there anyway.

    `extra_files` is a sequence of `(local_path, path_in_repo)` for scorecards
    and verdicts that are not part of the HF model directory. Provenance for a
    research artifact is not optional decoration -- a recovery probe whose
    gate verdict lives only on a box that gets recycled is a set of weights
    nobody can interpret.
    """
    if not allow_released:
        assert_not_released(repo_id)
    problems = check_publishable(model_dir)
    if problems:
        raise ValueError(f"refusing to publish {model_dir}: "
                         f"{'; '.join(problems)}")

    token = token or os.environ.get("HF_TOKEN_WRITE")
    if not token:
        raise ValueError("HF_TOKEN_WRITE is not set; cannot publish")

    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    message = commit_message or f"publish {os.path.basename(model_dir)}"
    # The model directory first, GGUFs second. If the transfer dies partway the
    # repo is then a valid HF model missing its quantizations, rather than a
    # bare GGUF with no config or card to explain it.
    api.upload_folder(folder_path=model_dir, repo_id=repo_id,
                      repo_type="model", commit_message=message)
    uploaded = [os.path.basename(p) for p in sorted(os.listdir(model_dir))
                if os.path.isfile(os.path.join(model_dir, p))]
    log(f"publisher: uploaded {len(uploaded)} files from {model_dir} "
        f"to {repo_id} (private={private})")

    gguf_done: List[Tuple[str, int]] = []
    for path in gguf_paths:
        if not os.path.exists(path):
            log(f"publisher: skipping {path} (does not exist)")
            continue
        name = os.path.basename(path)
        api.upload_file(path_or_fileobj=path, path_in_repo=f"gguf/{name}",
                        repo_id=repo_id, repo_type="model",
                        commit_message=f"{message}: {name}")
        size = os.path.getsize(path)
        gguf_done.append((name, size))
        log(f"publisher: uploaded gguf/{name} ({size / 1e6:.1f} MB)")

    extra_done: List[str] = []
    for local_path, path_in_repo in extra_files:
        if not os.path.exists(local_path):
            log(f"publisher: skipping {local_path} (does not exist)")
            continue
        api.upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="model",
                        commit_message=f"{message}: {path_in_repo}")
        extra_done.append(path_in_repo)
        log(f"publisher: uploaded {path_in_repo}")

    return {"repo_id": repo_id, "private": private, "files": uploaded,
            "ggufs": [{"name": n, "size": s} for n, s in gguf_done],
            "extra_files": extra_done,
            "url": f"https://huggingface.co/{repo_id}"}


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_published(repo_id: str, paths_in_repo: Iterable[str],
                     local_paths: Iterable[str], *, token: Optional[str] = None,
                     dest: Optional[str] = None, downloader=None,
                     log=print) -> dict:
    """Re-download published files and check they hash-match what was sent.

    A successful `upload_file` means the request was accepted, not that the
    bytes on the Hub are the bytes on disk. The failure this catches is quiet
    by construction: a truncated or empty artifact downloads fine, loads in
    llama.cpp far enough to produce numbers, and those numbers are wrong. For
    a repository whose entire purpose is letting someone re-decide a gate
    against real artifacts, "it uploaded" is not the claim that matters.

    Downloads into a fresh directory rather than reading the local cache, so a
    cached copy of the file we just uploaded cannot stand in for the remote one.
    """
    token = token or os.environ.get("HF_TOKEN_WRITE")
    if downloader is None:
        from huggingface_hub import hf_hub_download

        def downloader(repo_id, filename, local_dir):
            return hf_hub_download(repo_id=repo_id, filename=filename,
                                   repo_type="model", token=token,
                                   local_dir=local_dir, force_download=True)

    import tempfile
    checked, mismatched = [], []
    with tempfile.TemporaryDirectory(dir=dest) as scratch:
        for path_in_repo, local in zip(paths_in_repo, local_paths):
            expected = sha256_file(local)
            fetched = downloader(repo_id, path_in_repo, scratch)
            actual = sha256_file(fetched)
            entry = {"path_in_repo": path_in_repo, "sha256": expected,
                     "matched": actual == expected}
            if not entry["matched"]:
                entry["downloaded_sha256"] = actual
                mismatched.append(path_in_repo)
            checked.append(entry)
            log(f"publisher: verified {path_in_repo} "
                f"{'OK' if entry['matched'] else 'MISMATCH'}")
    return {"repo_id": repo_id, "checked": checked,
            "mismatched": mismatched, "passed": not mismatched}


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-dir", required=True,
                   help="HF-format export directory (config.json, weights, "
                        "tokenizer, README.md)")
    p.add_argument("--repo-id", default=os.environ.get(
        "DAEDALUS_HF_RELEASE_REPO", DEFAULT_RELEASE_REPO))
    p.add_argument("--gguf-dir", action="append", default=[],
                   help="directory to collect .gguf files from; repeatable")
    p.add_argument("--public", action="store_true",
                   help="create the repo public. Off by default: publishing is "
                        "outward-facing and hard to walk back.")
    p.add_argument("--check-only", action="store_true",
                   help="report whether the directory is publishable, upload "
                        "nothing")
    p.add_argument("--allow-released", action="store_true",
                   help="permit a released-model repo id. Off by default so an "
                        "experiment cannot land on the release by inheriting "
                        "--repo-id's default.")
    p.add_argument("--extra-file", action="append", default=[],
                   metavar="LOCAL:IN_REPO",
                   help="additional file to upload, e.g. a gate verdict or "
                        "scorecard; repeatable")
    p.add_argument("--verify", action="store_true",
                   help="after uploading, re-download every --extra-file into a "
                        "fresh directory and check it hash-matches the local "
                        "copy. Exits non-zero on any mismatch.")
    a = p.parse_args(argv)

    extra_files = []
    for spec in a.extra_file:
        local, _, in_repo = spec.partition(":")
        if not local or not in_repo:
            raise SystemExit(f"--extra-file expects LOCAL:IN_REPO, got {spec!r}")
        extra_files.append((local, in_repo))

    problems = check_publishable(a.model_dir)
    if a.check_only:
        print(json.dumps({"model_dir": a.model_dir, "publishable": not problems,
                          "problems": problems}, indent=2))
        return 0 if not problems else 1

    result = publish_model(a.model_dir, a.repo_id,
                           gguf_paths=find_ggufs(*a.gguf_dir),
                           private=not a.public,
                           allow_released=a.allow_released,
                           extra_files=extra_files)
    print(json.dumps(result, indent=2))

    if a.verify and extra_files:
        verified = verify_published(
            a.repo_id, [in_repo for _, in_repo in extra_files],
            [local for local, _ in extra_files])
        print(json.dumps(verified, indent=2))
        if not verified["passed"]:
            print(f"VERIFICATION FAILED: {verified['mismatched']}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
