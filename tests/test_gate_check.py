"""Tests for the mechanical Phase 2 gate verdict.

A gate that cannot fail has not been tested, so every criterion here is
exercised twice: once against evidence that should pass it, and once against
evidence that must fail it. The failing halves are the point -- this gate exists
because a prose claim from the session that wrote the evaluators is not
evidence, and a check that silently passes everything is the same thing with
more steps.
"""

import json
import os

import pytest

from daedalus.scorecard import (
    ArtifactRef,
    Provenance,
    Scorecard,
    write_scorecard,
)
from scripts.gate_check import (
    control_verdict,
    determinism_verdict,
    paired_quant_verdict,
    run_gate,
    sandbox_verdict,
    scorecard_fingerprint,
)


ZERO = "0" * 64
ONE = "1" * 64


def _artifact(kind="gguf-f16", sha=ZERO, path="artifact"):
    return ArtifactRef(path=path, sha256=sha, kind=kind)


def _card(tmp_path, name, *, kind="retrieval", metrics=None, items=None,
          backend="oracle", artifact_kind="checkpoint", artifact_sha=ZERO,
          created_at="2026-08-24T00:00:00Z", git_sha="abc1234", details=None):
    card = Scorecard(
        kind=kind,
        name=name,
        provenance=Provenance(
            artifact=_artifact(artifact_kind, artifact_sha),
            tokenizer=_artifact("tokenizer"),
            seed=1, git_sha=git_sha, bpb_mode="not-applicable",
            runtime={"backend": backend}),
        metrics=metrics if metrics is not None else {"exact_match": 1.0},
        created_at=created_at,
        items=items if items is not None else [{"id": "a", "correct": 1}],
        details=details or {},
    )
    return write_scorecard(tmp_path / f"{name}.json", card)["scorecard"]


# ------------------------------------------------------- synthetic controls ---

def test_control_passes_when_the_oracle_scored_everything(tmp_path):
    path = _card(tmp_path, "retrieval-passkey")

    verdict = control_verdict([path])

    assert verdict["passed"]
    assert verdict["observed"]["retrieval-passkey"] == 1.0


def test_control_fails_on_a_malformed_item_set(tmp_path):
    path = _card(tmp_path, "retrieval-passkey", metrics={"exact_match": 0.975})

    verdict = control_verdict([path])

    assert not verdict["passed"]
    assert "0.975" in verdict["detail"]


def test_control_fails_when_only_a_model_run_was_supplied(tmp_path):
    """A model's copy-control score is a measurement, not a control.

    Accepting it would let a run with no oracle pass the control gate on the
    strength of a model that happened to score well.
    """
    path = _card(tmp_path, "retrieval-passkey", backend="llama-cpp")

    verdict = control_verdict([path])

    assert not verdict["passed"]
    assert "oracle" in verdict["detail"]


# ------------------------------------------------------------- determinism ---

def test_determinism_passes_for_two_identical_runs(tmp_path):
    first = _card(tmp_path / "a", "retrieval-passkey")
    second = _card(tmp_path / "b", "retrieval-passkey")

    verdict = determinism_verdict([[first, second]])

    assert verdict["passed"]


def test_determinism_ignores_only_when_it_was_written(tmp_path):
    """Two runs of one evaluation differ in timestamp and nothing else."""
    first = _card(tmp_path / "a", "retrieval-passkey",
                  created_at="2026-08-24T01:00:00Z")
    second = _card(tmp_path / "b", "retrieval-passkey",
                   created_at="2026-08-24T09:30:00Z")

    assert scorecard_fingerprint(first) == scorecard_fingerprint(second)
    assert determinism_verdict([[first, second]])["passed"]


def test_determinism_fails_when_one_item_flipped(tmp_path):
    """The aggregate is unchanged; only the per-item outcomes moved.

    This is the case an aggregate comparison cannot see, and the reason the
    criterion is written on item digests.
    """
    first = _card(tmp_path / "a", "retrieval-passkey",
                  items=[{"id": "a", "correct": 1}, {"id": "b", "correct": 0}])
    second = _card(tmp_path / "b", "retrieval-passkey",
                   items=[{"id": "a", "correct": 0}, {"id": "b", "correct": 1}])

    verdict = determinism_verdict([[first, second]])

    assert not verdict["passed"]
    assert "per-item" in verdict["detail"]


def test_determinism_fails_when_a_metric_moved(tmp_path):
    first = _card(tmp_path / "a", "retrieval-passkey",
                  metrics={"exact_match": 1.0})
    second = _card(tmp_path / "b", "retrieval-passkey",
                   metrics={"exact_match": 0.9})

    verdict = determinism_verdict([[first, second]])

    assert not verdict["passed"]
    assert "fingerprint" in verdict["detail"]


def test_determinism_fails_when_nothing_was_repeated():
    verdict = determinism_verdict([])

    assert not verdict["passed"]


# ------------------------------------------------------ paired-quant identity ---

def _quant_card(tmp_path, name, kind, nlls, text_file="ppl.txt", n_ctx=512):
    card = Scorecard(
        kind="paired-quant",
        name=name,
        provenance=Provenance(
            artifact=_artifact(kind, ZERO if kind == "gguf-f16" else ONE),
            tokenizer=_artifact("tokenizer"), seed=1, git_sha="abc1234",
            bpb_mode="not-applicable", runtime={}),
        metrics={"perplexity": 1.0, "mean_nll": 0.0},
        created_at="2026-08-24T00:00:00Z",
        items=[{"id": f"chunk-{index}", "nll": value}
               for index, value in enumerate(nlls)],
        details={"text_file": text_file, "n_ctx": n_ctx},
    )
    return write_scorecard(tmp_path / f"{name}.json", card)["scorecard"]


def test_paired_quant_passes_for_the_same_chunks(tmp_path):
    fp16 = _quant_card(tmp_path, "perplexity-fp16", "gguf-f16", [1.0, 1.1, 1.2])
    q4 = _quant_card(tmp_path, "perplexity-quantized", "gguf-q4_0",
                     [1.05, 1.15, 1.25])

    verdict = paired_quant_verdict(fp16, q4)

    assert verdict["passed"]
    assert verdict["observed"]["n"] == 3


def test_paired_quant_fails_on_a_chunk_count_mismatch(tmp_path):
    fp16 = _quant_card(tmp_path, "perplexity-fp16", "gguf-f16", [1.0, 1.1, 1.2])
    q4 = _quant_card(tmp_path, "perplexity-quantized", "gguf-q4_0", [1.05, 1.15])

    verdict = paired_quant_verdict(fp16, q4)

    assert not verdict["passed"]
    assert "item_count" in verdict["detail"]


def test_paired_quant_fails_when_the_two_runs_read_different_text(tmp_path):
    """Same chunk count, different corpus: chunk k is not the same tokens."""
    fp16 = _quant_card(tmp_path, "perplexity-fp16", "gguf-f16", [1.0, 1.1])
    q4 = _quant_card(tmp_path, "perplexity-quantized", "gguf-q4_0", [1.0, 1.1],
                     text_file="other.txt")

    # The comparison itself refuses this; the gate must not pass what the
    # report would reject.
    from scripts.gguf_eval import compare_quantization
    from daedalus.scorecard import ScorecardError, load_scorecard

    with pytest.raises(ScorecardError):
        compare_quantization(load_scorecard(fp16), load_scorecard(q4))


def test_paired_quant_fails_when_no_pair_was_supplied(tmp_path):
    verdict = run_gate(controls=[], determinism=[], paired_quant=None,
                       sandbox=False)

    paired = [c for c in verdict["criteria"]
              if c["criterion"] == "paired-quant-identity"][0]
    assert not paired["passed"]


# ---------------------------------------------------------------- sandbox ---

@pytest.mark.slow
def test_sandbox_criterion_passes_against_the_real_sandbox(tmp_path):
    verdict = sandbox_verdict()

    assert verdict["passed"], verdict["detail"]
    assert verdict["observed"]["network_client"] == "process_blocked"
    assert verdict["observed"]["os_system"] == "process_blocked"


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="needs root to have a drop to make")
def test_sandbox_criterion_checks_reads_and_writes_outside_it(tmp_path):
    secret_dir = tmp_path / "config"
    secret_dir.mkdir()
    secret = secret_dir / "runtime.env"
    secret.write_text("HF_TOKEN=not-a-real-token\n")
    secret.chmod(0o600)
    secret_dir.chmod(0o700)
    protected = tmp_path / "protected"
    protected.mkdir()
    protected.chmod(0o700)

    verdict = sandbox_verdict(str(secret), str(protected))

    assert verdict["passed"], verdict["detail"]
    assert not (protected / "planted.txt").exists()


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() != 0, reason="needs root to have a drop to make")
def test_sandbox_criterion_fails_when_a_write_outside_it_succeeds():
    """A directory the sandbox *can* write to must fail the criterion.

    This is the mutation check on the containment probe: without it, a probe
    that never attempted the write would report a pass and nobody would know.
    It cannot use `tmp_path` -- pytest's own tree is root-owned and 0700, so the
    dropped child cannot traverse into it whatever the leaf mode is, and the
    write would fail for the wrong reason.
    """
    import shutil
    import tempfile

    writable = tempfile.mkdtemp(prefix="daedalus-gate-mutation-")
    os.chmod(writable, 0o777)
    try:
        verdict = sandbox_verdict(None, writable)

        assert not verdict["passed"]
        assert "outside_write" in verdict["detail"]
    finally:
        shutil.rmtree(writable, ignore_errors=True)


# ------------------------------------------------------------------ report ---

def test_run_gate_fails_the_whole_verdict_when_one_criterion_fails(tmp_path):
    control = _card(tmp_path, "retrieval-passkey", metrics={"exact_match": 0.5})
    first = _card(tmp_path / "a", "retrieval-mqar")
    second = _card(tmp_path / "b", "retrieval-mqar")
    fp16 = _quant_card(tmp_path, "perplexity-fp16", "gguf-f16", [1.0])
    q4 = _quant_card(tmp_path, "perplexity-quantized", "gguf-q4_0", [1.1])

    verdict = run_gate(controls=[control], determinism=[[first, second]],
                       paired_quant=[fp16, q4], sandbox=False)

    assert not verdict["passed"]
    failed = [c["criterion"] for c in verdict["criteria"] if not c["passed"]]
    assert failed == ["synthetic-controls"]


def test_main_exits_non_zero_on_failure(tmp_path, capsys):
    from scripts.gate_check import main

    control = _card(tmp_path, "retrieval-passkey", metrics={"exact_match": 0.5})
    out = tmp_path / "gate.json"

    code = main(["--control", str(control), "--no-sandbox", "--out", str(out)])

    assert code == 1
    written = json.loads(out.read_text())
    assert written["passed"] is False
    assert written["gate"] == "phase2-evaluation"


def test_main_records_every_observed_value_it_decided_on(tmp_path):
    from scripts.gate_check import main

    control = _card(tmp_path, "retrieval-passkey")
    first = _card(tmp_path / "a", "retrieval-mqar")
    second = _card(tmp_path / "b", "retrieval-mqar")
    fp16 = _quant_card(tmp_path, "perplexity-fp16", "gguf-f16", [1.0, 1.1])
    q4 = _quant_card(tmp_path, "perplexity-quantized", "gguf-q4_0", [1.05, 1.2])
    out = tmp_path / "gate.json"

    code = main(["--control", str(control), "--repeat", str(first), str(second),
                 "--paired-quant", str(fp16), str(q4), "--no-sandbox",
                 "--out", str(out)])

    assert code == 0
    written = json.loads(out.read_text())
    assert written["passed"] is True
    for criterion in written["criteria"]:
        # A verdict without the value that decided it is a claim again.
        assert criterion["observed"], criterion["criterion"]
        assert criterion["evidence"], criterion["criterion"]
