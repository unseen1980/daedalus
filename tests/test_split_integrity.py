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


# ------------------------------------------------- the phase 5-8 proxy split ---
# The corpus this box carries is not the hero split: it is `data/shards-train`
# with `data/holdout` carved off it, and every phase 5, 6 and 7 number is a
# training run over the first and a held-out BPB over the second. That makes
# their disjointness the load-bearing assumption of three phases' evidence, and
# it is one an ordinary mistake breaks quietly: `make_mixture_holdout_split`
# reserves whole *tail shards* and hardlinks them, so a re-carve, a re-fetch
# that renumbers shards, or a source with too few shards to split can put the
# same tokens on both sides. Nothing downstream would notice -- the BPB would
# simply come out better, uniformly, and read as a model that had learned.

TRAIN_ROOT = os.path.join(REPO, "data", "shards-train")
HOLDOUT_ROOT = os.path.join(REPO, "data", "holdout")

requires_proxy_split = pytest.mark.skipif(
    not (_sources(TRAIN_ROOT) and _sources(HOLDOUT_ROOT)),
    reason="the phase 5-8 train/holdout pair is not carved on this box")


def _shard_identity(root, src, man):
    """`{file name}` and `{(device, inode)}` for one source's shards.

    Both, because either alone misses a real case: two hardlinks of one shard
    under different names share an inode, and two genuinely different shards can
    only be told apart by name once they are in separate directories.
    """
    names, inodes = set(), set()
    for entry in man["shards"]:
        path = os.path.join(root, src, entry["file"])
        names.add(entry["file"])
        if os.path.exists(path):
            stat = os.stat(path)
            inodes.add((stat.st_dev, stat.st_ino))
    return names, inodes


@requires_proxy_split
def test_no_source_puts_the_same_shard_in_both_the_train_split_and_the_holdout():
    shared, compared = {}, []
    for src, _, holdout in _manifests(HOLDOUT_ROOT):
        train_dir = os.path.join(TRAIN_ROOT, src, "manifest.json")
        if not os.path.isfile(train_dir):
            continue
        with open(train_dir) as f:
            train = json.load(f)
        compared.append(src)
        h_names, h_inodes = _shard_identity(HOLDOUT_ROOT, src, holdout)
        t_names, t_inodes = _shard_identity(TRAIN_ROOT, src, train)
        overlap = (h_names & t_names) | {f"inode {i}" for i in h_inodes & t_inodes}
        if overlap:
            shared[src] = sorted(str(item) for item in overlap)
    assert not shared, (
        "these sources are scored on tokens they trained on, so every phase 5-8 "
        f"held-out BPB over them is a memorization measurement: {shared}")
    # A disjointness check that examined nothing passes for the wrong reason,
    # which is the same silent-vacuum failure the rest of this file guards.
    assert compared == _sources(HOLDOUT_ROOT), (
        f"only compared {compared} of the holdout's {_sources(HOLDOUT_ROOT)}; a "
        f"source scored without a matching train manifest was not checked")


@requires_proxy_split
def test_every_shard_of_the_proxy_split_is_byte_exact_against_its_manifest():
    """The same check the hero split gets, on the corpus that is actually being
    trained over here. A truncated shard surfaces at loader startup, which for a
    detached multi-hour sweep means the arm dies and the box sits idle."""
    bad = []
    for root in (TRAIN_ROOT, HOLDOUT_ROOT):
        for src, d, man in _manifests(root):
            itemsize = ITEMSIZE.get(man.get("dtype", "uint16"))
            assert itemsize, f"{src}: unknown dtype {man.get('dtype')!r}"
            summed = sum(int(e["tokens"]) for e in man["shards"])
            assert int(man["total_tokens"]) == summed, (
                f"{os.path.basename(root)}/{src}: total_tokens "
                f"{man['total_tokens']:,} != sum of shards {summed:,}")
            for entry in man["shards"]:
                path = os.path.join(d, entry["file"])
                want = int(entry["tokens"]) * itemsize
                if not os.path.exists(path):
                    bad.append(f"{os.path.basename(root)}/{src}/{entry['file']}: missing")
                elif os.path.getsize(path) != want:
                    bad.append(
                        f"{os.path.basename(root)}/{src}/{entry['file']}: "
                        f"{os.path.getsize(path):,} bytes, manifest implies {want:,}")
    assert not bad, "truncated or missing shards:\n  " + "\n  ".join(bad[:20])
