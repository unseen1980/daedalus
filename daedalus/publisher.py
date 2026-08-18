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
                  api=None, log=print) -> dict:
    """Upload `model_dir` plus any GGUFs to `repo_id` as a Hub *model* repo.

    Raises ValueError if the directory is not publishable -- unlike the
    checkpoint uploader, which must never raise because it runs inside a
    training loop. Here a failure is the last thing standing between a finished
    model and nobody being able to use it, so it should be loud.
    """
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

    return {"repo_id": repo_id, "private": private, "files": uploaded,
            "ggufs": [{"name": n, "size": s} for n, s in gguf_done],
            "url": f"https://huggingface.co/{repo_id}"}


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
    a = p.parse_args(argv)

    problems = check_publishable(a.model_dir)
    if a.check_only:
        print(json.dumps({"model_dir": a.model_dir, "publishable": not problems,
                          "problems": problems}, indent=2))
        return 0 if not problems else 1

    result = publish_model(a.model_dir, a.repo_id,
                           gguf_paths=find_ggufs(*a.gguf_dir),
                           private=not a.public)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
