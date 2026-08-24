"""Tests for per-language code BPB and the general replay holdout.

Phase 8 trades general ability for code ability and is gated on both sides at
once: code BPB must improve by >=5% while general BPB regresses by <=1.5%. A
single aggregate cannot referee that, and an aggregate whose weighting is
implicit is worse than none -- token-weighting a code holdout lets Python's
volume hide a collapse in Rust. Every aggregate here is therefore emitted under
all three weightings, named.
"""

import json

import pytest

from daedalus.scorecard import ScorecardError, load_scorecard


def _make_holdout(root, sources):
    for name, tokens in sources.items():
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"total_tokens": tokens}))
    return root


# ---------------------------------------------------------------- discovery ---

def test_discover_sources_finds_every_manifest_backed_directory(tmp_path):
    from scripts.bpb_eval import discover_sources

    _make_holdout(tmp_path, {"python": 100, "rust": 50, "go": 25})
    (tmp_path / "not-a-source").mkdir()

    assert discover_sources(tmp_path) == {"go": 25, "python": 100, "rust": 50}


def test_discover_sources_refuses_a_root_with_no_sources(tmp_path):
    from scripts.bpb_eval import discover_sources

    with pytest.raises(ValueError, match="no source"):
        discover_sources(tmp_path)


def test_discover_sources_refuses_a_manifest_without_a_token_count(tmp_path):
    from scripts.bpb_eval import discover_sources

    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "manifest.json").write_text(json.dumps({}))

    with pytest.raises(ValueError, match="total_tokens"):
        discover_sources(tmp_path)


# --------------------------------------------------------------- evaluation ---

def test_evaluate_sources_records_one_item_per_language(tmp_path):
    from scripts.bpb_eval import evaluate_sources

    _make_holdout(tmp_path, {"python": 100, "rust": 50})
    measured = {"python": 1.20, "rust": 1.60}

    records = evaluate_sources(tmp_path, bpb_fn=lambda path: measured[path.name])

    assert [record["id"] for record in records] == ["python", "rust"]
    assert records[0]["bpb"] == pytest.approx(1.20)
    assert records[0]["tokens"] == 100


def test_evaluate_sources_refuses_a_non_finite_measurement(tmp_path):
    from scripts.bpb_eval import evaluate_sources

    _make_holdout(tmp_path, {"python": 100})

    with pytest.raises(ValueError, match="finite"):
        evaluate_sources(tmp_path, bpb_fn=lambda path: float("nan"))


# -------------------------------------------------------------- aggregation ---

def test_summarize_bpb_emits_every_weighting_by_name():
    from scripts.bpb_eval import summarize_bpb

    records = [
        {"id": "python", "bpb": 1.0, "tokens": 900},
        {"id": "rust", "bpb": 2.0, "tokens": 100},
    ]

    metrics = summarize_bpb(records)

    assert metrics["bpb_python"] == pytest.approx(1.0)
    assert metrics["bpb_rust"] == pytest.approx(2.0)
    assert metrics["bpb_equal_weight"] == pytest.approx(1.5)
    assert metrics["bpb_token_weighted"] == pytest.approx(1.1)
    # With no explicit mixture the headline is the equal-weighted number, so a
    # small language cannot be drowned out by a large one.
    assert metrics["bpb"] == pytest.approx(1.5)
    assert metrics["n_sources"] == 2


def test_summarize_bpb_uses_explicit_mixture_weights_when_given():
    from scripts.bpb_eval import summarize_bpb

    records = [
        {"id": "python", "bpb": 1.0, "tokens": 900},
        {"id": "rust", "bpb": 2.0, "tokens": 100},
    ]

    metrics = summarize_bpb(records, weights={"python": 0.55, "rust": 0.45})

    assert metrics["bpb"] == pytest.approx(1.45)
    assert metrics["bpb_equal_weight"] == pytest.approx(1.5)


def test_summarize_bpb_refuses_weights_that_name_an_absent_source():
    from scripts.bpb_eval import summarize_bpb

    records = [{"id": "python", "bpb": 1.0, "tokens": 900}]

    with pytest.raises(ValueError, match="haskell"):
        summarize_bpb(records, weights={"python": 0.5, "haskell": 0.5})


def test_parse_weights_reads_name_equals_fraction_pairs():
    from scripts.bpb_eval import parse_weights

    assert parse_weights(["python=0.55", "rust=0.08"]) == {"python": 0.55,
                                                           "rust": 0.08}
    with pytest.raises(ValueError):
        parse_weights(["python"])


# --------------------------------------------------------------- scorecards ---

def _run(tmp_path, out_dir, **kwargs):
    from scripts.bpb_eval import run_bpb_eval
    from daedalus.scorecard import ArtifactRef

    defaults = dict(
        name="code-bpb",
        holdout_root=tmp_path,
        out_dir=out_dir,
        artifact=ArtifactRef(path="ckpt.pt", sha256="a" * 64, kind="checkpoint",
                             config="daedalus-150m"),
        tokenizer_ref=ArtifactRef(path="t.json", sha256="b" * 64,
                                  kind="tokenizer"),
        seed=3, git_sha="deadbee", max_batches=None,
        bpb_fn=lambda path: {"python": 1.2, "rust": 1.6}[path.name],
    )
    defaults.update(kwargs)
    return run_bpb_eval(**defaults)


def test_run_bpb_eval_records_a_full_pass_as_full(tmp_path):
    _make_holdout(tmp_path, {"python": 100, "rust": 50})
    out = tmp_path / "out"

    paths = _run(tmp_path, out)
    card = load_scorecard(paths["scorecard"])

    assert card.kind == "bpb"
    assert card.name == "code-bpb"
    assert card.provenance.bpb_mode == "full"
    assert card.provenance.bpb_sample_batches is None
    assert card.item_count == 2
    assert card.metrics["bpb_python"] == pytest.approx(1.2)


def test_run_bpb_eval_records_a_bounded_pass_as_sample(tmp_path):
    _make_holdout(tmp_path, {"python": 100, "rust": 50})
    out = tmp_path / "out"

    paths = _run(tmp_path, out, max_batches=100)
    card = load_scorecard(paths["scorecard"])

    # The distinction the plan requires in every result file: a 100-batch
    # sample and a full pass are different measurements and must never be
    # compared as if they were the same.
    assert card.provenance.bpb_mode == "sample"
    assert card.provenance.bpb_sample_batches == 100


def test_run_bpb_eval_carries_the_weighting_into_the_record(tmp_path):
    _make_holdout(tmp_path, {"python": 100, "rust": 50})
    out = tmp_path / "out"

    paths = _run(tmp_path, out, weights={"python": 0.7, "rust": 0.3})
    card = load_scorecard(paths["scorecard"])

    assert card.details["weighting"] == "explicit"
    assert card.details["weights"] == {"python": 0.7, "rust": 0.3}
    assert card.metrics["bpb"] == pytest.approx(1.2 * 0.7 + 1.6 * 0.3)


def test_run_bpb_eval_defaults_to_equal_weighting_in_the_record(tmp_path):
    _make_holdout(tmp_path, {"python": 100, "rust": 50})
    out = tmp_path / "out"

    card = load_scorecard(_run(tmp_path, out)["scorecard"])

    assert card.details["weighting"] == "equal"


def test_two_checkpoints_are_pairable_at_source_granularity(tmp_path):
    from daedalus.scorecard import paired_outcomes

    _make_holdout(tmp_path, {"python": 100, "rust": 50})
    before = load_scorecard(_run(tmp_path, tmp_path / "before")["scorecard"])
    after = load_scorecard(_run(
        tmp_path, tmp_path / "after",
        bpb_fn=lambda path: {"python": 1.1, "rust": 1.7}[path.name])["scorecard"])

    paired = paired_outcomes(before, after, field="bpb")

    assert paired["n"] == 2
    assert paired["per_item_delta"] == pytest.approx([-0.1, 0.1])


def test_a_holdout_with_different_languages_is_not_pairable(tmp_path):
    from daedalus.scorecard import paired_outcomes

    _make_holdout(tmp_path, {"python": 100, "rust": 50})
    before = load_scorecard(_run(tmp_path, tmp_path / "before")["scorecard"])

    other = tmp_path / "other"
    _make_holdout(other, {"python": 100, "go": 50})
    after = load_scorecard(_run(
        other, tmp_path / "after",
        bpb_fn=lambda path: {"python": 1.1, "go": 1.7}[path.name])["scorecard"])

    with pytest.raises(ScorecardError, match="digest"):
        paired_outcomes(before, after, field="bpb")
