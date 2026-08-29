"""The corpus slice has to keep the corpus's shape.

`MixtureBatchSource` renormalizes over whatever sources it finds on disk, so a
lopsided download does not raise -- it trains the recovery probes on a
different distribution from the one the released model was pretrained on and
reports success. Everything here targets that silent case.
"""

import json

import pytest

from scripts import fetch_corpus_subset as fetcher


REPO_FILES = [
    "README.md",
    "fineweb-edu/manifest.json",
    "fineweb-edu/shard-00000.bin",
    "fineweb-edu/shard-00001.bin",
    "dclm-baseline/manifest.json",
    "dclm-baseline/shard-00000.bin",
    "finewiki-en/manifest.json",
    "finewiki-en/shard-00000.bin",
]


def _manifest(n_shards, tokens_each=100_000_000, eos_id=0):
    return {
        "eos_id": eos_id,
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
        "shards": [{"file": f"shard-{i:05d}.bin", "tokens": tokens_each}
                   for i in range(n_shards)],
        "total_tokens": n_shards * tokens_each,
    }


def test_files_group_by_source_directory():
    grouped = fetcher.group_by_source(REPO_FILES)
    assert set(grouped) == {"fineweb-edu", "dclm-baseline", "finewiki-en"}
    assert grouped["fineweb-edu"] == ["fineweb-edu/manifest.json",
                                      "fineweb-edu/shard-00000.bin",
                                      "fineweb-edu/shard-00001.bin"]
    # A top-level README is not a source.
    assert "README.md" not in grouped


def test_budgets_follow_the_corpus_mixture_shares():
    """The shares come from `dataprep.MIXTURE`, the same definition that built
    the corpus -- not from a second copy that could drift from it."""
    shares = fetcher.mixture_shares()
    assert shares["fineweb-edu"] == pytest.approx(0.375)
    targets = fetcher.per_source_targets(
        ["fineweb-edu", "dclm-baseline"], 1_000_000_000)
    # 0.375 : 0.225 renormalized over the two present sources.
    assert targets["fineweb-edu"] == pytest.approx(625_000_000, rel=1e-6)
    assert targets["dclm-baseline"] == pytest.approx(375_000_000, rel=1e-6)
    assert sum(targets.values()) == pytest.approx(1_000_000_000, rel=1e-6)


def test_a_missing_source_redistributes_rather_than_underfilling():
    """Asking for a budget and silently returning two thirds of it would leave
    a probe short of tokens and reaching for a second epoch without saying so."""
    full = fetcher.per_source_targets(list(fetcher.mixture_shares()), 1_000_000)
    partial = fetcher.per_source_targets(["fineweb-edu", "dclm-baseline"],
                                         1_000_000)
    assert sum(partial.values()) == pytest.approx(1_000_000, rel=1e-3)
    assert partial["fineweb-edu"] > full["fineweb-edu"]


def test_a_source_the_mixture_never_mentions_gets_nothing():
    """It is not part of the distribution the released model was trained on."""
    targets = fetcher.per_source_targets(["fineweb-edu", "some-other-corpus"],
                                         1_000_000)
    assert "some-other-corpus" not in targets


def test_a_repo_with_no_recognised_source_refuses_to_guess():
    with pytest.raises(ValueError, match="refusing to guess"):
        fetcher.per_source_targets(["mystery"], 1_000_000)


def test_shards_are_taken_whole_and_from_the_front():
    """From the front because `select_holdout_shards` reserves the holdout from
    the end; whole because a partial .bin has no manifest entry describing its
    real length."""
    manifest = _manifest(5, tokens_each=10)
    chosen = fetcher.select_shards(manifest, target_tokens=25)
    assert [s["file"] for s in chosen] == ["shard-00000.bin", "shard-00001.bin",
                                           "shard-00002.bin"]


def test_a_budget_smaller_than_one_shard_still_takes_one():
    """A source that rounded to zero would drop out of the mixture entirely,
    which is the lopsided download this module exists to prevent."""
    chosen = fetcher.select_shards(_manifest(3, tokens_each=10), target_tokens=1)
    assert len(chosen) == 1


def test_asking_for_more_than_the_source_holds_takes_all_of_it():
    chosen = fetcher.select_shards(_manifest(2, tokens_each=10),
                                   target_tokens=10_000)
    assert len(chosen) == 2


def test_a_source_with_no_shards_yields_nothing_rather_than_raising():
    assert fetcher.select_shards({"shards": []}, 100) == []


def test_the_local_manifest_describes_only_what_was_downloaded():
    """A manifest naming a file that was not fetched is not a smaller corpus,
    it is a FileNotFoundError inside np.memmap at the first batch."""
    manifest = _manifest(5, tokens_each=10)
    chosen = fetcher.select_shards(manifest, 25)
    local = fetcher.local_manifest(manifest, chosen)
    assert [s["file"] for s in local["shards"]] == [s["file"] for s in chosen]
    assert local["total_tokens"] == 30
    # Provenance of what was left behind, so the slice is recognisable as one.
    assert local["subset_of"] == {"shards": 5, "total_tokens": 50}


def test_the_local_manifest_keeps_the_fields_the_loader_reads():
    """`ShardDataset` reads `eos_id` off the manifest; dropping it would
    silently fall back to the default and mis-segment documents."""
    manifest = _manifest(3, eos_id=7)
    local = fetcher.local_manifest(manifest, manifest["shards"][:1])
    assert local["eos_id"] == 7
    assert local["tokenizer"] == "HuggingFaceTB/SmolLM2-135M"


def test_the_plan_is_computable_without_downloading_anything():
    manifests = {"fineweb-edu": _manifest(4, tokens_each=100),
                 "dclm-baseline": _manifest(4, tokens_each=100),
                 "finewiki-en": _manifest(4, tokens_each=100)}
    plan = fetcher.plan_fetch(REPO_FILES, manifests, target_tokens=600)
    assert set(plan["sources"]) == set(manifests)
    assert plan["planned_tokens"] > 0
    # Every source keeps at least one shard, so none silently leaves the
    # mixture.
    assert all(entry["shards"] for entry in plan["sources"].values())
    # The bigger share really does get more.
    assert (len(plan["sources"]["fineweb-edu"]["shards"])
            >= len(plan["sources"]["finewiki-en"]["shards"]))


def test_a_source_without_a_manifest_is_skipped_not_half_fetched():
    """Shards whose manifest never arrived cannot be loaded, and a directory
    of orphaned .bin files would make the mixture loader's source count wrong."""
    manifests = {"fineweb-edu": _manifest(2, tokens_each=100)}
    plan = fetcher.plan_fetch(REPO_FILES, manifests, target_tokens=100)
    assert set(plan["sources"]) == {"fineweb-edu"}


def test_the_plan_round_trips_through_json(tmp_path):
    """It is written next to the shards as the record of what the probes
    trained on, so it has to survive serialization."""
    manifests = {"fineweb-edu": _manifest(2, tokens_each=100)}
    plan = fetcher.plan_fetch(REPO_FILES, manifests, target_tokens=150)
    path = tmp_path / "fetch-plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True))
    assert json.loads(path.read_text()) == plan
