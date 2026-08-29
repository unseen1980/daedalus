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


def test_discover_sources_can_be_narrowed_to_a_named_subset(tmp_path):
    """Phase 6's arms trained on one source; the box's holdout root has three.

    Scoring an arm over sources it never trained on answers a different
    question than the screen asks, and costs three times the GPU hours to get
    the wrong answer.
    """
    from scripts.bpb_eval import discover_sources

    _make_holdout(tmp_path, {"fineweb-edu": 100, "dclm-baseline": 50,
                             "stack-edu-python": 25})

    assert discover_sources(tmp_path, ["fineweb-edu"]) == {"fineweb-edu": 100}


def test_discover_sources_refuses_a_subset_naming_an_absent_source(tmp_path):
    """A scorecard that quietly measured two of the three sources it was asked
    for describes a holdout that does not exist."""
    from scripts.bpb_eval import discover_sources

    _make_holdout(tmp_path, {"python": 100})

    with pytest.raises(ValueError, match="rust"):
        discover_sources(tmp_path, ["python", "rust"])


def test_evaluate_sources_only_measures_the_requested_subset(tmp_path):
    """The filter must reach the measurement, not just the reported table --
    a filter applied after `bpb_fn` ran would have already spent the hours."""
    from scripts.bpb_eval import evaluate_sources

    _make_holdout(tmp_path, {"fineweb-edu": 100, "dclm-baseline": 50})
    measured = []

    records = evaluate_sources(
        tmp_path, sources=["fineweb-edu"],
        bpb_fn=lambda path: (measured.append(path.name), 1.25)[1])

    assert measured == ["fineweb-edu"]
    assert [record["id"] for record in records] == ["fineweb-edu"]


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


# ----------------------------------------------------------- holdout build ---
#
# `--holdout-root` is required, and Phase 3 found the obvious consequence the
# hard way: the corpus fetched onto the box is train shards only, so there was
# no split to point it at, and full-pass BPB -- third in the preregistered
# selection order -- went unmeasured. Building the split is part of scoring,
# not a separate errand someone has to remember first.

def _write_source(root, name, tokens, shard_tokens):
    from daedalus.data import ShardWriter

    directory = root / name
    directory.mkdir(parents=True)
    writer = ShardWriter(str(directory), shard_tokens=shard_tokens)
    writer.write(list(range(tokens)))
    writer.close()
    writer.write_manifest({"eos_id": 0})
    return directory


def test_build_holdout_materializes_one_split_per_source(tmp_path):
    from scripts.bpb_eval import build_holdout

    mixture = tmp_path / "shards"
    _write_source(mixture, "python", tokens=1000, shard_tokens=100)
    _write_source(mixture, "rust", tokens=1000, shard_tokens=100)

    record = build_holdout(mixture, tmp_path / "holdout", tmp_path / "train",
                           holdout_frac=0.1)

    assert (tmp_path / "holdout" / "python" / "manifest.json").exists()
    assert (tmp_path / "holdout" / "rust" / "manifest.json").exists()
    assert record["sources"]["python"]["holdout_tokens"] == 100
    assert record["sources"]["python"]["train_tokens"] == 900


def test_build_holdout_reserves_shards_the_train_split_does_not_keep(tmp_path):
    """Disjointness is the whole point of a holdout, so assert it directly
    rather than trusting the frac arithmetic to imply it."""
    from scripts.bpb_eval import build_holdout

    mixture = tmp_path / "shards"
    _write_source(mixture, "python", tokens=1000, shard_tokens=100)

    build_holdout(mixture, tmp_path / "holdout", tmp_path / "train",
                  holdout_frac=0.2)

    held = {path.name for path in (tmp_path / "holdout" / "python").glob("*.bin")}
    trained = {path.name for path in (tmp_path / "train" / "python").glob("*.bin")}
    assert held and trained
    assert held.isdisjoint(trained)


def test_build_holdout_is_idempotent(tmp_path):
    """Re-scoring an arm must not depend on whether the split already exists."""
    from scripts.bpb_eval import build_holdout

    mixture = tmp_path / "shards"
    _write_source(mixture, "python", tokens=1000, shard_tokens=100)

    first = build_holdout(mixture, tmp_path / "holdout", tmp_path / "train",
                          holdout_frac=0.1)
    second = build_holdout(mixture, tmp_path / "holdout", tmp_path / "train",
                           holdout_frac=0.1)

    assert first["sources"] == second["sources"]


def test_build_holdout_records_a_source_too_small_to_split(tmp_path):
    """`make_mixture_holdout_split` skips a single-shard source rather than
    raising. Skipping quietly would leave the scorecard describing a mixture it
    did not measure, so the skip is part of the record."""
    from scripts.bpb_eval import build_holdout

    mixture = tmp_path / "shards"
    _write_source(mixture, "python", tokens=1000, shard_tokens=100)
    _write_source(mixture, "tiny", tokens=50, shard_tokens=1000)

    record = build_holdout(mixture, tmp_path / "holdout", tmp_path / "train",
                           holdout_frac=0.1)

    assert record["skipped"] == ["tiny"]
    assert "tiny" not in record["sources"]
    assert not (tmp_path / "holdout" / "tiny").exists()


def test_build_holdout_reports_how_much_of_the_mixture_it_covers(tmp_path):
    """Naming the skipped sources is not enough: a reader counting names cannot
    tell whether the gap is 5% of the training mixture or half of it. On this
    box seven of ten sources are single-shard, and the answer is 31%."""
    from scripts.bpb_eval import build_holdout

    mixture = tmp_path / "shards"
    _write_source(mixture, "python", tokens=1000, shard_tokens=100)
    _write_source(mixture, "tiny", tokens=50, shard_tokens=1000)

    record = build_holdout(mixture, tmp_path / "holdout", tmp_path / "train",
                           holdout_frac=0.1,
                           weights={"python": 0.75, "tiny": 0.25})

    assert record["mixture_share_covered"] == pytest.approx(0.75)


# ---------------------------------------------------------------- exposure ---
#
# A holdout carved out of the training corpus *after* a run has trained on it is
# not held out from that run. `MixtureBatchSource` samples windows with
# replacement across every shard of a source, holdout shards included, so the
# expected number of times the run covered each of a source's tokens is exactly
# the epoch count the mixture resolver already computes. Reporting it turns "the
# holdout is a bit contaminated" into a number per source.

def test_exposure_reports_epochs_per_source(tmp_path):
    from scripts.bpb_eval import recovery_exposure

    # Both sources are large enough that the 4.0-epoch cap does not bind, so
    # the shares are the blueprint ones and the arithmetic is the plain one.
    # (A capped source is the subject of its own test below.)
    mixture = tmp_path / "shards"
    _write_source(mixture, "big", tokens=1000, shard_tokens=100)
    _write_source(mixture, "small", tokens=1000, shard_tokens=100)

    record = recovery_exposure(mixture, run_tokens=1000,
                               weights={"big": 0.5, "small": 0.5})

    # 1000 tokens x a 0.5 share = 500 drawn from a 1000-token source.
    assert record["sources"]["big"]["epochs"] == pytest.approx(0.5)
    assert record["sources"]["big"]["tokens_drawn"] == pytest.approx(500)


def test_exposure_names_the_sources_a_run_covered_more_than_once(tmp_path):
    """The most contaminated holdouts are the small sources -- exactly the ones
    an equal-weighted aggregate over-counts."""
    from scripts.bpb_eval import recovery_exposure

    mixture = tmp_path / "shards"
    _write_source(mixture, "big", tokens=10000, shard_tokens=1000)
    _write_source(mixture, "small", tokens=100, shard_tokens=10)

    record = recovery_exposure(mixture, run_tokens=1000,
                               weights={"big": 0.5, "small": 0.5},
                               max_epochs=100.0)

    assert record["repeated_sources"] == ["small"]
    assert record["max_epochs_seen"] == pytest.approx(5.0)


def test_exposure_rides_the_same_epoch_cap_training_used(tmp_path):
    """`resolve_mixture` caps a short source's share at `max_epochs`, and the
    run sampled the post-cap shares. Reporting pre-cap shares would overstate
    the contamination of precisely the sources the cap protects."""
    from scripts.bpb_eval import recovery_exposure

    mixture = tmp_path / "shards"
    _write_source(mixture, "big", tokens=10000, shard_tokens=1000)
    _write_source(mixture, "small", tokens=100, shard_tokens=10)

    record = recovery_exposure(mixture, run_tokens=1000,
                               weights={"big": 0.5, "small": 0.5},
                               max_epochs=2.0)

    assert record["sources"]["small"]["epochs"] <= 2.0 + 1e-9


def test_run_bpb_eval_carries_extra_details_into_the_record(tmp_path):
    _make_holdout(tmp_path, {"python": 100, "rust": 50})

    paths = _run(tmp_path, tmp_path / "out",
                 details_extra={"exposure": {"max_epochs_seen": 0.5}})

    payload = json.loads(paths["scorecard"].read_text())
    assert payload["details"]["exposure"]["max_epochs_seen"] == 0.5


# ------------------------------------------------------- which tokenizer ----

def test_the_tokenizer_flag_decodes_the_holdout_not_just_labels_it():
    """Bits per byte is nats-per-token converted through *the bytes those
    tokens stand for*, and the byte count comes from decoding them. `--tokenizer`
    used to reach only the scorecard's provenance while `get_tokenizer()`
    decoded with SmolLM2 regardless -- invisible while every artifact shared one
    vocabulary, and wrong from Phase 4 on, which is exactly when BPB became the
    metric that decides between vocabularies."""
    import inspect

    from scripts import bpb_eval

    source = inspect.getsource(bpb_eval.main)
    assert "get_tokenizer(args.tokenizer)" in source
    assert "get_tokenizer()" not in source
    # And, being part of the measurement, checked against the shards rather
    # than trusted -- a mismatch here is a finite, plausible, wrong BPB.
    assert "assert_shards_tokenizer(args.holdout_root" in source


def test_a_tokenizer_directory_is_hashed_by_its_vocabulary_file(tmp_path):
    """A saved HF tokenizer is a directory, and `sha256_file` on one raises.
    `tokenizer.json` is the right digest anyway: it carries the vocabulary and
    the merges, which is what "which tokenizer scored this" means."""
    from scripts.bpb_eval import tokenizer_artifact

    directory = tmp_path / "v32768"
    directory.mkdir()
    (directory / "tokenizer.json").write_text('{"model": {}}')
    reference = tokenizer_artifact(directory)
    assert reference.kind == "tokenizer"
    assert reference.path == str(directory)
    assert len(reference.sha256) == 64

    with pytest.raises(FileNotFoundError):
        tokenizer_artifact(tmp_path / "not-a-tokenizer")
