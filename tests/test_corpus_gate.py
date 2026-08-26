"""Tests for the mechanical Phase 7 corpus gate.

Every criterion is exercised twice, once against evidence that should pass it and
once against evidence that must fail it. The failing halves are the point: this
gate exists because "the corpus is clean" asserted by the session that built it
is not evidence, and a check that passes everything is that assertion with more
steps.

The allocation criteria are driven through `train.summarize_mixture` rather than
handwritten dicts, so a change to the shape the trainer reports breaks these
tests instead of silently making the gate read fields that no longer exist.
"""

import json

import pytest

import eval as E
from scripts.corpus_gate import (
    contamination_verdict,
    decontam_index_verdict,
    epoch_cap_verdict,
    main,
    manifest_provenance_verdict,
    mixture_allocation,
    mixture_skew_verdict,
    run_gate,
)
from train import summarize_mixture


#: Taken from the evaluator rather than retyped, so the fixture cannot drift
#: away from the registry the gate reads. A test that pinned its own copy would
#: keep passing after a task moved split -- which is the exact defect the
#: coverage criterion exists to catch.
SCORED_SPLITS = dict(E.TASK_SPLITS)


def _index(tmp_path, *, tasks=None, limit=None, complete=True, problems=()):
    tasks = tasks if tasks is not None else {
        name: {"split": split, "items": 100, "repo": f"org/{name}"}
        for name, split in SCORED_SPLITS.items()}
    path = tmp_path / "decontam-index.json"
    path.write_text(json.dumps({
        "path": "data/decontam/eval-index-13gram.txt.gz",
        "problems": list(problems),
        "provenance": {"built_at": "2026-08-26T07:19:08Z", "complete": complete,
                       "digest": "sha256:abc", "limit": limit, "n": 13,
                       "ngrams": 1371773, "schema": 1, "tasks": tasks},
    }))
    return path


def _scan(tmp_path, *, filtered=0, split_gap=0, limit_gap=0, totals=True,
          per_source=None, name="contam-exposure"):
    payload = {"per_source": [{"source": name, "source_tokens": tokens}
                              for name, tokens in (per_source or {}).items()]}
    if totals:
        payload["totals"] = {
            "corpus_tokens": 16988128870, "docs": 174932,
            "sampled_frac": 0.0132,
            "docs_filtered": filtered, "doc_rate_filtered_upper95": 2.2e-05,
            "docs_split_gap": split_gap, "doc_rate_split_gap_upper95": 3.2e-05,
            "docs_limit_gap": limit_gap, "doc_rate_limit_gap_upper95": 3.2e-05,
        }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def _manifest(root, name, *, tokens=1_000_000, revision=None, resolved="c0ffee",
              filters=True, git_sha="abc1234", subset_of=None):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_key": name, "source_dataset": f"org/{name}", "eos_id": 0,
        "total_tokens": tokens,
        "shards": [{"file": f"{name}_00000.bin", "tokens": tokens}],
        "source_revision": revision,
    }
    if subset_of is not None:
        payload["subset_of"] = {"shards": 100, "total_tokens": subset_of}
    if resolved is not None:
        payload["source_release"] = {"repo_id": f"org/{name}",
                                     "resolved_commit": resolved,
                                     "license": "odc-by"}
    if filters:
        payload["filters"] = {"min_chars": 200, "row_filter": False}
    if git_sha is not None:
        payload["builder_git_sha"] = git_sha
    (directory / "manifest.json").write_text(json.dumps(payload))
    return directory / "manifest.json"


def _summary(on_disk, *, budget, max_epochs=4.0, probs=None, target=None):
    names = sorted(on_disk)
    target = target or {name: 1.0 / len(names) for name in names}
    return summarize_mixture(names, target, probs or dict(target), on_disk,
                             budget, max_epochs)


# ------------------------------------------------- the frozen decontam index ---

def test_index_passes_when_every_scored_split_is_covered_in_full(tmp_path):
    verdict = decontam_index_verdict(_index(tmp_path))

    assert verdict["passed"], verdict["detail"]
    assert verdict["observed"]["arc_easy"]["split"] == "test"
    assert verdict["observed"]["limit"] is None


def test_index_fails_when_a_task_was_indexed_on_the_wrong_split(tmp_path):
    """The released build's actual defect: ARC-Easy and OpenBookQA were indexed
    on `validation` while the evaluator scores `test`, so the scored items were
    never filtered and nothing anywhere reported an error."""

    assert SCORED_SPLITS["arc_easy"] == "test"
    tasks = {name: {"split": split, "items": 100}
             for name, split in SCORED_SPLITS.items()}
    tasks["arc_easy"] = {"split": "validation", "items": 570}

    verdict = decontam_index_verdict(_index(tmp_path, tasks=tasks))

    assert not verdict["passed"]
    assert "arc_easy" in verdict["detail"]
    assert "'test'" in verdict["detail"]


def test_index_fails_for_a_task_added_to_the_evaluator_since_it_was_built(
        tmp_path, monkeypatch):
    """Coverage is read against the evaluator's registry, not a list in the
    gate. Scoring a sixth task makes an index built for five incomplete, and
    the gate has to say so rather than keep passing on the five it knows."""

    monkeypatch.setitem(E.TASK_SPLITS, "boolq", "validation")

    verdict = decontam_index_verdict(_index(tmp_path))

    assert not verdict["passed"]
    assert "boolq is not in the index" in verdict["detail"]


def test_index_fails_when_a_per_task_limit_was_applied(tmp_path):
    verdict = decontam_index_verdict(_index(tmp_path, limit=2000))

    assert not verdict["passed"]
    assert "2000" in verdict["detail"]


def test_index_fails_when_a_scored_task_is_absent(tmp_path):
    tasks = {name: {"split": split, "items": 100}
             for name, split in SCORED_SPLITS.items() if name != "winogrande"}

    verdict = decontam_index_verdict(_index(tmp_path, tasks=tasks))

    assert not verdict["passed"]
    assert "winogrande is not in the index" in verdict["detail"]
    assert verdict["observed"]["winogrande"]["indexed"] is False


def test_index_fails_when_it_records_itself_incomplete(tmp_path):
    verdict = decontam_index_verdict(_index(tmp_path, complete=False))

    assert not verdict["passed"]
    assert "complete" in verdict["detail"]


def test_index_fails_rather_than_raises_when_the_artifact_is_missing(tmp_path):
    verdict = decontam_index_verdict(tmp_path / "nope.json")

    assert not verdict["passed"]
    assert verdict["detail"]


# ----------------------------------------------------- the corpus-wide scan ---

def test_contamination_passes_on_zero_hits_and_says_what_that_bounds(tmp_path):
    verdict = contamination_verdict(_scan(tmp_path))

    assert verdict["passed"], verdict["detail"]
    # A zero over a sample bounds the rate; it does not prove absence, and the
    # detail has to say so or the verdict will be read as the stronger claim.
    assert "1.32" in verdict["detail"]
    assert "rather than proving it zero" in verdict["detail"]


@pytest.mark.parametrize("index", ["filtered", "split_gap", "limit_gap"])
def test_contamination_fails_on_a_hit_in_any_index(tmp_path, index):
    verdict = contamination_verdict(_scan(tmp_path, **{index: 1}))

    assert not verdict["passed"]
    assert index in verdict["detail"]


def test_contamination_fails_when_the_scan_recorded_no_totals(tmp_path):
    verdict = contamination_verdict(_scan(tmp_path, totals=False))

    assert not verdict["passed"]
    assert "no totals" in verdict["detail"]


def test_contamination_passes_when_the_scan_covered_this_corpus(tmp_path):
    supply = {"a": 5_000_000_000, "b": 1_000_000}
    path = _scan(tmp_path, per_source=supply)

    verdict = contamination_verdict(path, supply)

    assert verdict["passed"], verdict["detail"]


def test_contamination_fails_when_the_scan_saw_less_than_the_corpus_holds(
        tmp_path):
    """A scan artifact names no shard tree, so nothing links it to the corpus
    being gated except the extent it recorded per source. `fineweb-edu` grew by
    158M tokens after this program's scan ran, and without this check those
    tokens are covered by a clean verdict that never saw them."""

    path = _scan(tmp_path, per_source={"a": 5_035_572_292})

    verdict = contamination_verdict(path, {"a": 5_193_493_853})

    assert not verdict["passed"]
    assert "157,921,561 were never scanned" in verdict["detail"]


def test_contamination_fails_when_a_source_was_never_scanned(tmp_path):
    path = _scan(tmp_path, per_source={"a": 1_000})

    verdict = contamination_verdict(path, {"a": 1_000, "b": 2_000})

    assert not verdict["passed"]
    assert "did not cover 'b'" in verdict["detail"]


def test_contamination_fails_when_the_scan_read_a_different_corpus(tmp_path):
    path = _scan(tmp_path, per_source={"a": 1_000, "elsewhere": 7})

    verdict = contamination_verdict(path, {"a": 1_000})

    assert not verdict["passed"]
    assert "which this corpus does not hold" in verdict["detail"]


# ------------------------------------------------------------- the epoch cap ---

def test_epoch_cap_passes_when_every_source_stays_under_the_limit():
    summary = _summary({"a": 1_000_000_000, "b": 1_000_000_000},
                       budget=1_000_000_000)

    verdict = epoch_cap_verdict(summary)

    assert verdict["passed"], verdict["detail"]
    assert verdict["observed"]["epochs"]["a"] == pytest.approx(0.5)


def test_epoch_cap_passes_at_exactly_the_limit():
    """A capped source is pinned at exactly `max_epochs`, and the acceptance
    says no source may *exceed* four -- so the boundary has to pass, or every
    corpus whose cap worked would be refused for the cap working."""

    summary = _summary({"a": 250_000_000}, budget=1_000_000_000)

    verdict = epoch_cap_verdict(summary)

    assert verdict["observed"]["epochs"]["a"] == pytest.approx(4.0)
    assert verdict["passed"], verdict["detail"]


def test_epoch_cap_fails_and_names_the_source_that_ran_out():
    summary = _summary({"a": 1_000_000_000, "b": 403_573},
                       budget=1_000_000_000)

    verdict = epoch_cap_verdict(summary)

    assert not verdict["passed"]
    assert verdict["detail"].startswith("b would be read")
    assert verdict["observed"]["most_repeated_source"] == "b"


# ---------------------------------------------------------- the sampled mix ---

def test_skew_passes_when_the_blueprint_is_delivered():
    summary = _summary({"a": 1_000_000_000, "b": 1_000_000_000},
                       budget=1_000_000_000)

    verdict = mixture_skew_verdict(summary)

    assert verdict["passed"], verdict["detail"]
    assert verdict["observed"]["l1_skew_pts"] == pytest.approx(0.0)
    assert verdict["observed"]["all_capped_fallback"] is False


def test_skew_fails_when_the_cap_reweighted_the_mixture_too_far():
    """One short source water-fills onto the rest: the cap worked, repetition is
    bounded, and the corpus still is not delivering the blueprint."""

    on_disk = {"a": 1_000, "b": 10_000_000}
    summary = _summary(on_disk, budget=100_000,
                       probs={"a": 0.04, "b": 0.96})

    verdict = mixture_skew_verdict(summary)

    assert not verdict["passed"]
    assert verdict["observed"]["l1_skew_pts"] == pytest.approx(92.0)
    assert verdict["observed"]["all_capped_fallback"] is False
    assert "over the 5-pt bound" in verdict["detail"]


def test_skew_fails_on_the_all_capped_fallback_despite_a_perfect_skew():
    """The trap this gate exists for.

    Every source is over the limit, so `cap_weights_by_epochs` returns the
    target shares unchanged and accepts unbounded repetition. The skew is then
    0.00 -- its best possible value -- at the one budget where nothing bounds
    how often a document is re-read. A gate reading the skew alone would hand
    this corpus its cleanest verdict.
    """

    summary = _summary({"a": 1_000, "b": 1_000}, budget=100_000)

    assert summary["l1_skew_pts"] == pytest.approx(0.0)

    verdict = mixture_skew_verdict(summary)

    assert not verdict["passed"]
    assert verdict["observed"]["all_capped_fallback"] is True
    assert "by construction rather than by measurement" in verdict["detail"]


# ------------------------------------------------------ manifest provenance ---

def test_provenance_passes_on_a_hub_resolved_commit(tmp_path):
    _manifest(tmp_path, "fineweb-edu")

    verdict = manifest_provenance_verdict(tmp_path, ["fineweb-edu"])

    assert verdict["passed"], verdict["detail"]
    assert verdict["observed"]["fineweb-edu"]["license"] == "odc-by"


def test_provenance_passes_on_an_explicit_revision_without_a_lookup(tmp_path):
    """`source_revision` pins the bytes by itself; the Hub lookup is what closes
    the gap for the eight sources that declare none, not a second requirement."""

    _manifest(tmp_path, "stack-edu-python", revision="refs/convert/parquet",
              resolved=None)

    verdict = manifest_provenance_verdict(tmp_path, ["stack-edu-python"])

    assert verdict["passed"], verdict["detail"]


def test_provenance_fails_on_a_manifest_pinned_to_nothing(tmp_path):
    """The released build's manifests: a dataset name, a stream position, and no
    statement at all about which revision of that dataset was read."""

    _manifest(tmp_path, "dclm-baseline", resolved=None, filters=False,
              git_sha=None)

    verdict = manifest_provenance_verdict(tmp_path, ["dclm-baseline"])

    assert not verdict["passed"]
    assert "no source_revision and no resolved commit" in verdict["detail"]
    assert "no filters block" in verdict["detail"]
    assert "no builder_git_sha" in verdict["detail"]


def test_provenance_fails_rather_than_passes_vacuously(tmp_path):
    verdict = manifest_provenance_verdict(tmp_path, [])

    assert not verdict["passed"]


# ---------------------------------------------------------------- the whole ---

#: What `_corpus` holds, and therefore what a scan of it must say it read.
CORPUS_SUPPLY = {"a": 1_000_000_000, "b": 1_000_000_000}


def _corpus(tmp_path, **kwargs):
    root = tmp_path / "shards"
    for name, tokens in CORPUS_SUPPLY.items():
        _manifest(root, name, tokens=tokens, **kwargs)
    return root


def test_allocation_reads_the_same_manifests_the_trainer_would(tmp_path):
    root = _corpus(tmp_path)

    summary = mixture_allocation(str(root), budget_tokens=1_000_000_000,
                                 weights={"a": 0.5, "b": 0.5})

    assert sorted(summary["per_source"]) == ["a", "b"]
    assert summary["total_tokens_on_disk"] == 2_000_000_000


def test_allocation_counts_the_built_source_not_the_shards_fetched_here(
        tmp_path):
    """A work box holds a fetch of a source, and its manifest says so. Counting
    the local shards answers "what can this box train on"; the epoch cap asks
    how often a successor would re-read the *source*, and on this program's box
    the two differ by 10x-30x -- enough to turn a corpus inside the cap into one
    that appears to blow through it."""

    root = tmp_path / "shards"
    _manifest(root, "a", tokens=100_000_000, subset_of=5_000_000_000)

    summary = mixture_allocation(str(root), budget_tokens=5_000_000_000,
                                 weights={"a": 1.0})

    assert summary["per_source"]["a"]["epochs"] == pytest.approx(1.0)
    assert summary["supply_basis"]["a"] == "subset_of.total_tokens"
    assert summary["tokens_present_locally"]["a"] == 100_000_000


def test_allocation_can_be_asked_about_only_what_is_on_this_box(tmp_path):
    root = tmp_path / "shards"
    _manifest(root, "a", tokens=100_000_000, subset_of=5_000_000_000)

    summary = mixture_allocation(str(root), budget_tokens=5_000_000_000,
                                 weights={"a": 1.0}, local_supply=True)

    assert summary["per_source"]["a"]["epochs"] == pytest.approx(50.0)
    assert summary["supply_basis"]["a"] == "total_tokens"


def test_allocation_uses_total_tokens_when_nothing_says_it_is_a_subset(
        tmp_path):
    root = tmp_path / "shards"
    _manifest(root, "a", tokens=5_000_000_000)

    summary = mixture_allocation(str(root), budget_tokens=5_000_000_000,
                                 weights={"a": 1.0})

    assert summary["supply_basis"]["a"] == "total_tokens"
    assert summary["per_source"]["a"]["epochs"] == pytest.approx(1.0)


def test_dropping_a_source_renormalizes_the_rest(tmp_path):
    """The exhausted source takes its share to zero and the remainder absorbs
    it, so the skew the drop buys is measured rather than argued."""

    root = tmp_path / "shards"
    _manifest(root, "a", tokens=1_000_000_000)
    _manifest(root, "tiny", tokens=400_000)

    summary = mixture_allocation(str(root), budget_tokens=1_000_000_000,
                                 weights={"a": 0.98, "tiny": 0.02},
                                 drop_sources=["tiny"])

    assert sorted(summary["per_source"]) == ["a"]
    assert summary["per_source"]["a"]["target_share"] == pytest.approx(1.0)
    assert summary["l1_skew_pts"] == pytest.approx(0.0)


def test_allocation_keeps_the_cap_narration_off_the_verdict_stream(
        tmp_path, capsys):
    """stdout is the verdict document. `cap_weights_by_epochs` warns there when
    every source is over the limit, which would leave a caller parsing JSON with
    a prose line in front of it."""

    root = tmp_path / "shards"
    _manifest(root, "a", tokens=1_000, subset_of=1_000)
    _manifest(root, "b", tokens=1_000, subset_of=1_000)

    summary = mixture_allocation(str(root), budget_tokens=100_000,
                                 weights={"a": 0.5, "b": 0.5})

    assert summary["max_epochs_seen"] == pytest.approx(50.0)
    assert capsys.readouterr().out == ""


def test_gate_passes_when_every_criterion_holds(tmp_path):
    root = _corpus(tmp_path)

    verdict = run_gate(shards_root=str(root), budget_tokens=1_000_000_000,
                       decontam_index=str(_index(tmp_path)),
                       contam_scan=str(_scan(tmp_path,
                                             per_source=CORPUS_SUPPLY)),
                       weights={"a": 0.5, "b": 0.5})

    assert verdict["passed"], [item for item in verdict["criteria"]
                               if not item["passed"]]
    assert verdict["sources"] == ["a", "b"]
    assert len(verdict["criteria"]) == 5


def test_gate_fails_as_a_whole_when_one_criterion_fails(tmp_path):
    root = _corpus(tmp_path)

    verdict = run_gate(shards_root=str(root), budget_tokens=1_000_000_000,
                       decontam_index=str(_index(tmp_path)),
                       contam_scan=str(_scan(tmp_path, split_gap=1,
                                             per_source=CORPUS_SUPPLY)),
                       weights={"a": 0.5, "b": 0.5})

    assert not verdict["passed"]
    failed = [item["criterion"] for item in verdict["criteria"]
              if not item["passed"]]
    assert failed == ["corpus-contamination"]


def test_cli_exit_status_is_the_verdict(tmp_path, capsys):
    root = _corpus(tmp_path)
    out = tmp_path / "gate.json"
    argv = ["--shards-root", str(root), "--budget-tokens", "1000000000",
            "--decontam-index", str(_index(tmp_path)),
            "--contam-scan", str(_scan(tmp_path, per_source=CORPUS_SUPPLY)),
            "--out", str(out)]

    # The blueprint mixture does not know sources 'a' and 'b', so the weights
    # have to come from a file -- which is also the path the phase 7 verdict
    # will be read through.
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"a": 0.5, "b": 0.5}}))

    assert main(argv + ["--weights-from", str(weights)]) == 0
    written = json.loads(out.read_text())
    assert written["passed"]
    assert written["weights_source"] == str(weights)

    dirty = _scan(tmp_path, filtered=3, per_source=CORPUS_SUPPLY, name="dirty")
    assert main(argv + ["--weights-from", str(weights),
                        "--contam-scan", str(dirty)]) == 1
