"""The final report's refusals, which are the only part of it that can be tested.

Whether the report *says the right thing* is a judgement. Whether it can be made
to say a forbidden thing is not, and that is what these cover: a proxy dressed
as a model gain, a number with no file behind it, and a headline read from bytes
that have since changed.
"""

from __future__ import annotations

import json

import pytest

from daedalus.final_report import (
    ArtifactRecord,
    Claim,
    FinalReportError,
    Section,
    build_report,
    render_markdown,
    require,
    validate_report,
    verify_artifacts,
    write_report,
)


def _claim(**overrides) -> Claim:
    payload = {
        "key": "example",
        "scope": "released-model",
        "statement": "an example finding",
        "sources": ["runs/eval/example.json"],
    }
    payload.update(overrides)
    return Claim(**payload)


# ------------------------------------------------------------------- scopes ---

def test_proxy_claim_cannot_be_marked_as_applying_to_the_released_model():
    """The mistake the plan warns about three times, refused structurally."""

    with pytest.raises(FinalReportError, match="not a measurement of the shipped"):
        _claim(key="tokenizer-32k", scope="proxy", applies_to_released_model=True)


def test_projection_cannot_be_marked_as_applying_to_the_released_model():
    with pytest.raises(FinalReportError, match="not a measurement of the shipped"):
        _claim(key="v2-guess", scope="projection", applies_to_released_model=True)


@pytest.mark.parametrize("scope", ["released-model", "code-model"])
def test_measured_scopes_may_apply_to_a_model(scope):
    claim = _claim(scope=scope, applies_to_released_model=True)
    assert claim.to_dict()["applies_to_released_model"] is True


def test_unknown_scope_is_refused():
    with pytest.raises(FinalReportError, match="unknown scope"):
        _claim(scope="probably-fine")


def test_section_refuses_a_claim_of_another_measurement_scope():
    """A section heading is what a skimming reader takes the scope from."""

    mixed = [_claim(key="a", scope="proxy"), _claim(key="b", scope="released-model")]
    with pytest.raises(FinalReportError, match="contains claims of another scope"):
        Section(key="s", title="S", scope="proxy", summary="x", claims=mixed)


def test_proxy_claim_cannot_hide_in_a_released_model_section():
    """The direction that matters: a proxy under a shipped-model heading."""

    mixed = [_claim(key="a", scope="released-model"), _claim(key="b", scope="proxy")]
    with pytest.raises(FinalReportError, match=r"contains claims of another scope"):
        Section(key="s", title="S", scope="released-model", summary="x",
                claims=mixed)


def test_process_claims_may_sit_beside_the_numbers_they_qualify():
    """`process` is not a model measurement, so it cannot be misattributed.

    A gate outcome belongs next to the metric it stopped, not in a section of
    its own where a reader will not connect the two.
    """

    section = Section(key="s", title="S", scope="released-model", summary="x",
                      claims=[_claim(key="a", scope="released-model"),
                              _claim(key="b", scope="process")])

    assert [claim["scope"] for claim in section.to_dict()["claims"]] == \
        ["released-model", "process"]


def test_a_process_claim_still_cannot_apply_to_the_released_model():
    """The exemption above must not become a way in for a model claim."""

    with pytest.raises(FinalReportError, match="not a measurement of the shipped"):
        _claim(scope="process", applies_to_released_model=True)


# ------------------------------------------------------------------ sources ---

def test_claim_without_a_source_is_refused():
    with pytest.raises(FinalReportError, match="names no source"):
        _claim(sources=[])


def test_empty_statement_is_refused():
    with pytest.raises(FinalReportError, match="empty statement"):
        _claim(statement="   ")


def test_require_names_the_file_when_a_key_is_missing():
    with pytest.raises(FinalReportError, match="verdict.json has no 'gate.passed'"):
        require({"gate": {}}, "gate.passed", "verdict.json")


def test_require_reads_a_nested_value():
    assert require({"gate": {"passed": False}}, "gate.passed", "v.json") is False


# ---------------------------------------------------------------- artifacts ---

def test_verify_artifacts_matches_a_recorded_digest(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"weights")
    from daedalus.scorecard import sha256_file
    record = ArtifactRecord(name="m", role="final", path=str(target),
                            kind="gguf-q4_0", expected_sha256=sha256_file(target))

    summary = verify_artifacts([record])

    assert summary["all_recorded_digests_match"] is True
    assert summary["matched"] == ["m"]


def test_verify_artifacts_catches_bytes_that_moved(tmp_path):
    """A headline measured on bytes that no longer exist is not a headline."""

    target = tmp_path / "model.gguf"
    target.write_bytes(b"weights")
    record = ArtifactRecord(name="m", role="final", path=str(target),
                            kind="gguf-q4_0", expected_sha256="0" * 64)

    summary = verify_artifacts([record])

    assert summary["all_recorded_digests_match"] is False
    assert summary["mismatched"] == ["m"]
    assert "sha256 mismatch" in summary["records"][0]["sha256"]["detail"]


def test_missing_artifact_is_a_failure_not_a_blank(tmp_path):
    record = ArtifactRecord(name="gone", role="final",
                            path=str(tmp_path / "absent.gguf"), kind="gguf-f16",
                            expected_sha256="0" * 64)

    summary = verify_artifacts([record])

    assert summary["mismatched"] == ["gone"]
    assert "missing" in summary["records"][0]["sha256"]["detail"]


def test_artifact_without_a_recorded_digest_is_fingerprinted_not_passed(tmp_path):
    """Hashing a file now is a weaker claim than matching what measured it."""

    target = tmp_path / "model.gguf"
    target.write_bytes(b"weights")
    record = ArtifactRecord(name="m", role="final", path=str(target),
                            kind="gguf-q4_0", expected_sha256=None)

    summary = verify_artifacts([record])

    assert summary["matched"] == []
    assert summary["fingerprinted_only"] == ["m"]
    # Absence of a mismatch is not a pass, and the summary must not imply it is.
    assert summary["all_recorded_digests_match"] is True
    assert summary["records"][0]["sha256"]["verified"] is None


def test_artifact_record_carries_every_field_the_plan_names(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"w")
    record = ArtifactRecord(
        name="m", role="final", path=str(target), kind="gguf-q4_0",
        config="daedalus-150m", tokenizer="smollm2-49152",
        data_manifest="runs/codeprep/train-mixture.json", seed=20260824,
        producing_commit="abc1234", hub_repo="Unseen1980/x",
        hub_revision="main", hub_path="gguf/model.gguf", hub_private=True)

    payload = record.verify().to_dict()

    assert payload["config"] == "daedalus-150m"
    assert payload["data_manifest"] == "runs/codeprep/train-mixture.json"
    assert payload["seed"] == 20260824
    assert payload["producing_commit"] == "abc1234"
    assert payload["hub"] == {"repo": "Unseen1980/x", "revision": "main",
                              "path": "gguf/model.gguf", "private": True}


# -------------------------------------------------------------- the report ---

def _report(tmp_path, claims=None):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"weights")
    from daedalus.scorecard import sha256_file
    return build_report(
        program={"name": "test"},
        sections=[Section(key="s", title="Section", scope="released-model",
                          summary="a summary",
                          claims=claims if claims is not None else [_claim()])],
        artifacts=[ArtifactRecord(name="m", role="final", path=str(target),
                                  kind="gguf-q4_0",
                                  expected_sha256=sha256_file(target))],
        validation={"full_suite": "passed"})


def test_build_report_validates_and_round_trips(tmp_path):
    payload = _report(tmp_path)

    assert validate_report(payload) == {"claims": 1, "sections": 1}
    written = write_report(tmp_path / "out" / "report.json", payload)
    assert json.loads(written.read_text())["schema"] == payload["schema"]


def test_validate_report_catches_a_hand_edited_scope(tmp_path):
    """The serialised form is what a later reader holds, so re-check it there."""

    payload = _report(tmp_path)
    payload["sections"][0]["claims"][0]["scope"] = "proxy"
    payload["sections"][0]["claims"][0]["applies_to_released_model"] = True

    with pytest.raises(FinalReportError, match="marked as applying"):
        validate_report(payload)


def test_validate_report_catches_a_duplicate_claim_key(tmp_path):
    payload = _report(tmp_path, claims=[_claim(key="one"), _claim(key="two")])

    payload["sections"][0]["claims"][1]["key"] = "one"
    with pytest.raises(FinalReportError, match="duplicate claim key"):
        validate_report(payload)


def test_markdown_is_rendered_from_the_json(tmp_path):
    """Two documents holding the same number independently can disagree."""

    payload = _report(tmp_path, claims=[
        _claim(key="q4", statement="Q4_0 penalty", value=5.5387, units="%")])

    rendered = render_markdown(payload, title="Report", preamble="Preamble.")

    assert "5.5387 %" in rendered
    assert "runs/eval/example.json" in rendered
    # The scope disclaimer is not optional boilerplate: it is the one sentence
    # that stops phases 4-7 being read as gains on the shipped model.
    assert "No proxy result in this report is a statement about the released" \
        in rendered


def test_markdown_keeps_each_caveat_with_its_finding(tmp_path):
    """A flat caveat list is sentences whose subject the reader has to guess."""

    payload = _report(tmp_path, claims=[
        _claim(key="a", statement="code BPB improvement",
               caveats=["carried by one source"]),
        _claim(key="b", statement="general BPB regression",
               caveats=["still accruing at 1B"])])

    rendered = render_markdown(payload, title="R", preamble="P")

    assert "- **code BPB improvement**\n  - carried by one source" in rendered
    assert "- **general BPB regression**\n  - still accruing at 1B" in rendered


def test_markdown_does_not_end_with_a_blank_line(tmp_path):
    """`git diff --check` rejects one, so the report could not be committed."""

    payload = _report(tmp_path)

    rendered = render_markdown(payload, title="R", preamble="P")

    assert rendered.endswith("|\n")
    assert not rendered.endswith("\n\n")


def test_markdown_flags_a_mismatched_artifact_loudly(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"weights")
    payload = build_report(
        program={}, sections=[],
        artifacts=[ArtifactRecord(name="m", role="final", path=str(target),
                                  kind="gguf-q4_0", expected_sha256="0" * 64)],
        validation={})

    rendered = render_markdown(payload, title="R", preamble="P")

    assert "**MISMATCH**" in rendered
    assert payload["artifacts"]["all_recorded_digests_match"] is False
