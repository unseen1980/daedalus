"""Every shard `hero` will read must be byte-exact against its manifest.

Why this is a test and not a one-off: a truncated shard is the likely corruption
from an interrupted dataprep, and this project's dataprep was interrupted many
times (issue #3, four failed full attempts, several per-worker RSS trips). It
does not surface until the loader memmaps the file, which for `hero` is
*startup* -- so `run_with_resume` would burn its ten attempts over ~1 h of
backoff and the operator would wake to a dead $59.85 run.

`scripts/validate_split.py` is the runnable version with the per-source table.
These are the assertions, so they run in the suite before every push.
"""
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERO_SPLIT = os.path.join(REPO, "data", "shards-hero-split", "train")
ITEMSIZE = {"uint16": 2, "uint32": 4, "int32": 4}


def _sources(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "manifest.json")))


def _manifests(root):
    for src in _sources(root):
        d = os.path.join(root, src)
        with open(os.path.join(d, "manifest.json")) as f:
            yield src, d, json.load(f)


requires_split = pytest.mark.skipif(
    not _sources(HERO_SPLIT), reason="the hero split is not carved on this box")


@requires_split
def test_every_shard_is_byte_exact_against_its_manifest():
    """size on disk == tokens * itemsize, for all 329 shards."""
    bad = []
    for src, d, man in _manifests(HERO_SPLIT):
        itemsize = ITEMSIZE.get(man.get("dtype", "uint16"))
        assert itemsize, f"{src}: unknown dtype {man.get('dtype')!r}"
        for entry in man["shards"]:
            path = os.path.join(d, entry["file"])
            want = int(entry["tokens"]) * itemsize
            if not os.path.exists(path):
                bad.append(f"{src}/{entry['file']}: missing")
                continue
            got = os.path.getsize(path)
            if got != want:
                bad.append(f"{src}/{entry['file']}: {got:,} bytes, "
                           f"manifest implies {want:,} ({got - want:+,})")
    assert not bad, "truncated or missing shards:\n  " + "\n  ".join(bad[:20])


@requires_split
def test_each_manifest_total_agrees_with_its_own_shards():
    for src, _, man in _manifests(HERO_SPLIT):
        summed = sum(int(e["tokens"]) for e in man["shards"])
        assert int(man["total_tokens"]) == summed, (
            f"{src}: total_tokens {man['total_tokens']:,} != "
            f"sum of shards {summed:,}")


@requires_split
def test_the_split_still_holds_the_token_count_the_gate_was_approved_on():
    """16,932,674,383 is the number in `STATUS.md`, the gate, and every epoch
    and mixture-skew figure derived from them. If the split is ever re-carved,
    this fails rather than letting the approved numbers quietly describe a
    different corpus."""
    total = sum(int(man["total_tokens"]) for _, _, man in _manifests(HERO_SPLIT))
    assert total == 16_932_674_383, f"train split is now {total:,} tokens"


@requires_split
def test_the_nine_sources_the_gate_names_are_the_nine_on_disk():
    assert _sources(HERO_SPLIT) == [
        "cosmopedia-v2", "dclm-baseline", "finemath-3plus", "finepdfs-edu",
        "finephrase", "fineweb-edu", "finewiki-en", "infiwebmath-3plus",
        "stack-edu-python",
    ], "everyday-conversations is deliberately absent (one shard, cannot split)"
