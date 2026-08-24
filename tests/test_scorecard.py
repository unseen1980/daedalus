"""Tests for the single scorecard schema every later gate consumes.

The point of this schema is that a gate never has to trust a number's
provenance by convention. Phase 3 compares a QAT recovery against the released
baseline; Phase 8 compares Daedalus-Code against an untouched base. Both are
decided by ~1-point margins, so a scorecard that cannot prove *which* items
were scored, under which artifact hash, at which BPB mode, is not evidence.
"""

import json
import math

import pytest

from daedalus.scorecard import (
    SCORECARD_SCHEMA,
    ArtifactRef,
    GateCheck,
    Provenance,
    Scorecard,
    ScorecardError,
    evaluate_gates,
    item_digest,
    load_scorecard,
    paired_outcomes,
    sha256_file,
    write_scorecard,
)


ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64


def _provenance(**overrides) -> Provenance:
    fields = {
        "artifact": ArtifactRef(path="gguf/hero-base-f16.gguf", sha256=ZERO_SHA,
                                kind="gguf-f16"),
        "tokenizer": ArtifactRef(path="hf/base/tokenizer.json", sha256=ONE_SHA,
                                 kind="tokenizer"),
        "seed": 20260824,
        "git_sha": "abc1234",
    }
    fields.update(overrides)
    return Provenance(**fields)


def _scorecard(**overrides) -> Scorecard:
    fields = {
        "kind": "retrieval",
        "name": "passkey",
        "provenance": _provenance(),
        "metrics": {"exact_match": 0.75},
        "items": [
            {"id": "d256-0", "correct": 1},
            {"id": "d256-1", "correct": 0},
            {"id": "d512-0", "correct": 1},
            {"id": "d512-1", "correct": 1},
        ],
        "created_at": "2026-08-24T12:00:00Z",
    }
    fields.update(overrides)
    return Scorecard(**fields)


# ------------------------------------------------------------------ schema ---

def test_scorecard_records_schema_version_and_item_accounting():
    card = _scorecard()

    payload = card.to_dict()

    assert payload["schema"] == SCORECARD_SCHEMA
    assert payload["kind"] == "retrieval"
    assert payload["name"] == "passkey"
    assert payload["item_count"] == 4
    assert payload["item_digest"] == item_digest(card.items)
    assert payload["metrics"]["exact_match"] == 0.75


def test_scorecard_round_trips_through_json():
    card = _scorecard()

    restored = Scorecard.from_dict(json.loads(json.dumps(card.to_dict())))

    assert restored.to_dict() == card.to_dict()


def test_unknown_kind_is_refused():
    with pytest.raises(ScorecardError, match="kind"):
        _scorecard(kind="vibes").to_dict()


def test_non_finite_metric_is_refused():
    with pytest.raises(ScorecardError, match="finite"):
        _scorecard(metrics={"exact_match": float("nan")}).to_dict()


def test_non_numeric_metric_is_refused():
    with pytest.raises(ScorecardError, match="numeric"):
        _scorecard(metrics={"exact_match": "0.75"}).to_dict()


def test_short_hash_is_refused():
    with pytest.raises(ScorecardError, match="sha256"):
        _scorecard(provenance=_provenance(
            artifact=ArtifactRef(path="x.gguf", sha256="deadbeef", kind="gguf-f16"),
        )).to_dict()


def test_sample_bpb_mode_requires_a_batch_bound():
    with pytest.raises(ScorecardError, match="bpb_sample_batches"):
        _scorecard(kind="bpb", provenance=_provenance(bpb_mode="sample")).to_dict()


def test_full_bpb_mode_refuses_a_batch_bound():
    with pytest.raises(ScorecardError, match="full"):
        _scorecard(
            kind="bpb",
            provenance=_provenance(bpb_mode="full", bpb_sample_batches=100),
        ).to_dict()


def test_full_and_sample_bpb_modes_are_distinguishable_in_the_payload():
    full = _scorecard(kind="bpb", provenance=_provenance(bpb_mode="full")).to_dict()
    sample = _scorecard(
        kind="bpb",
        provenance=_provenance(bpb_mode="sample", bpb_sample_batches=100),
    ).to_dict()

    assert full["provenance"]["bpb_mode"] == "full"
    assert full["provenance"]["bpb_sample_batches"] is None
    assert sample["provenance"]["bpb_mode"] == "sample"
    assert sample["provenance"]["bpb_sample_batches"] == 100


def test_provenance_carries_seed_task_revisions_and_runtime():
    card = _scorecard(provenance=_provenance(
        task_revisions={"humaneval-plus": "v0.1.10"},
        runtime={"llama_cpp_commit": "7584430", "device": "cpu", "threads": 24},
    ))

    payload = card.to_dict()["provenance"]

    assert payload["seed"] == 20260824
    assert payload["task_revisions"] == {"humaneval-plus": "v0.1.10"}
    assert payload["runtime"]["llama_cpp_commit"] == "7584430"


def test_item_digest_is_order_sensitive_and_stable():
    items = [{"id": "a", "correct": 1}, {"id": "b", "correct": 0}]

    assert item_digest(items) == item_digest(list(items))
    assert item_digest(items) != item_digest(list(reversed(items)))


def test_item_digest_ignores_key_order_within_an_item():
    assert item_digest([{"id": "a", "correct": 1}]) == \
        item_digest([{"correct": 1, "id": "a"}])


# ------------------------------------------------------------------ pairing ---

def test_paired_outcomes_refuse_mismatched_item_digests():
    left = _scorecard()
    right = _scorecard(items=[
        {"id": "d256-0", "correct": 1},
        {"id": "d256-9", "correct": 0},
        {"id": "d512-0", "correct": 1},
        {"id": "d512-1", "correct": 1},
    ])

    with pytest.raises(ScorecardError, match="digest"):
        paired_outcomes(left, right)


def test_paired_outcomes_refuse_mismatched_item_counts():
    left = _scorecard()
    right = _scorecard(items=left.items[:3])

    with pytest.raises(ScorecardError, match="item_count"):
        paired_outcomes(left, right)


def test_paired_outcomes_report_mcnemar_discordance():
    left = _scorecard()
    right = _scorecard(items=[
        {"id": "d256-0", "correct": 1},
        {"id": "d256-1", "correct": 1},
        {"id": "d512-0", "correct": 0},
        {"id": "d512-1", "correct": 1},
    ])

    paired = paired_outcomes(left, right, field="correct")

    assert paired["n"] == 4
    assert paired["both"] == 2
    assert paired["left_only"] == 1
    assert paired["right_only"] == 1
    assert paired["neither"] == 0
    assert paired["delta"] == pytest.approx(0.0)


def test_paired_outcomes_compare_continuous_fields():
    left = _scorecard(items=[{"id": "a", "nll": 2.0}, {"id": "b", "nll": 4.0}])
    right = _scorecard(items=[{"id": "a", "nll": 2.5}, {"id": "b", "nll": 3.5}])

    paired = paired_outcomes(left, right, field="nll")

    assert paired["n"] == 2
    assert paired["mean_left"] == pytest.approx(3.0)
    assert paired["mean_right"] == pytest.approx(3.0)
    assert paired["per_item_delta"] == pytest.approx([0.5, -0.5])


def test_paired_outcomes_refuse_a_field_absent_from_the_items():
    left = _scorecard()
    right = _scorecard()

    with pytest.raises(ScorecardError, match="field"):
        paired_outcomes(left, right, field="nll")


# -------------------------------------------------------------------- gates ---

def test_gate_checks_record_measured_values_and_verdicts():
    card = _scorecard(metrics={"exact_match": 0.75, "q4_penalty_pct": 2.4})

    verdict = evaluate_gates(card, [
        GateCheck(metric="exact_match", comparator=">=", threshold=0.5),
        GateCheck(metric="q4_penalty_pct", comparator="<=", threshold=1.0),
    ])

    assert verdict["passed"] is False
    assert verdict["checks"][0] == {
        "metric": "exact_match", "comparator": ">=", "threshold": 0.5,
        "measured": 0.75, "passed": True,
    }
    assert verdict["checks"][1]["passed"] is False


def test_gate_passes_only_when_every_check_passes():
    card = _scorecard(metrics={"exact_match": 0.75, "q4_penalty_pct": 0.4})

    verdict = evaluate_gates(card, [
        GateCheck(metric="exact_match", comparator=">=", threshold=0.5),
        GateCheck(metric="q4_penalty_pct", comparator="<=", threshold=1.0),
    ])

    assert verdict["passed"] is True


def test_gate_on_a_missing_metric_fails_loudly():
    card = _scorecard()

    with pytest.raises(ScorecardError, match="missing metric"):
        evaluate_gates(card, [GateCheck(metric="absent", comparator=">=", threshold=0.0)])


def test_unknown_comparator_is_refused():
    with pytest.raises(ScorecardError, match="comparator"):
        evaluate_gates(_scorecard(),
                       [GateCheck(metric="exact_match", comparator="~=", threshold=0.5)])


# --------------------------------------------------------------------- io ----

def test_write_scorecard_splits_items_into_a_sidecar(tmp_path):
    card = _scorecard()
    out = tmp_path / "nested" / "passkey.json"

    paths = write_scorecard(out, card)

    payload = json.loads(out.read_text())
    assert "items" not in payload
    assert payload["item_count"] == 4
    assert payload["item_digest"] == item_digest(card.items)

    sidecar = json.loads(paths["items"].read_text())
    assert paths["items"] == tmp_path / "nested" / "passkey.items.json"
    assert sidecar["item_digest"] == payload["item_digest"]
    assert sidecar["items"] == card.items


def test_write_scorecard_leaves_no_temporary_files(tmp_path):
    write_scorecard(tmp_path / "card.json", _scorecard())

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "card.items.json", "card.json"]


def test_write_scorecard_without_items_writes_no_sidecar(tmp_path):
    card = _scorecard(kind="bpb", items=None, metrics={"bpb": 1.23},
                      provenance=_provenance(bpb_mode="full"), item_count=17)

    paths = write_scorecard(tmp_path / "bpb.json", card)

    assert "items" not in paths
    assert [path.name for path in tmp_path.iterdir()] == ["bpb.json"]


def test_load_scorecard_reattaches_sidecar_items(tmp_path):
    card = _scorecard()
    out = tmp_path / "passkey.json"
    write_scorecard(out, card)

    restored = load_scorecard(out)

    assert restored.items == card.items
    assert restored.to_dict() == card.to_dict()


def test_load_scorecard_refuses_a_tampered_sidecar(tmp_path):
    out = tmp_path / "passkey.json"
    paths = write_scorecard(out, _scorecard())
    sidecar = json.loads(paths["items"].read_text())
    sidecar["items"][0]["correct"] = 0
    paths["items"].write_text(json.dumps(sidecar))

    with pytest.raises(ScorecardError, match="digest"):
        load_scorecard(out)


def test_load_scorecard_refuses_a_missing_sidecar(tmp_path):
    out = tmp_path / "passkey.json"
    paths = write_scorecard(out, _scorecard())
    paths["items"].unlink()

    with pytest.raises(ScorecardError, match="sidecar"):
        load_scorecard(out)


def test_sha256_file_hashes_bytes_not_paths(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    third = tmp_path / "c.bin"
    first.write_bytes(b"daedalus")
    second.write_bytes(b"daedalus")
    third.write_bytes(b"daedalus-code")

    digest = sha256_file(first)

    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert digest == sha256_file(second)
    assert digest != sha256_file(third)


def test_scorecard_without_items_still_reports_counts():
    card = _scorecard(kind="bpb", items=None, metrics={"bpb": 1.23},
                      provenance=_provenance(bpb_mode="full"), item_count=17)

    payload = card.to_dict()

    assert payload["item_count"] == 17
    assert payload["item_digest"] is None
    assert not math.isnan(payload["metrics"]["bpb"])
