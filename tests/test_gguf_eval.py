"""Tests for paired FP16 vs Q4_0 evaluation over identical items.

The quantization gates in this program are written in percents of perplexity
(<=3%, stretch <=1%) against a baseline penalty of about 6%. Two independently
computed aggregate perplexities cannot resolve that: the difference of two
unpaired estimates carries both estimates' error. Both artifacts score the
*same* chunks of the *same* text, so the damage is measured as a paired
per-chunk delta, and the pairing is enforced rather than assumed.
"""

import json
import math
import subprocess

import pytest

from daedalus.scorecard import ScorecardError, load_scorecard


CHUNK_STDOUT = """\
perplexity: tokenizing the input ..
perplexity: calculating perplexity over 4 chunks, n_ctx=512
[1]4.0000,[2]4.5000,[3]4.2500,
[4]4.3000,
Final estimate: PPL = 4.3000 +/- 0.05
"""


def _fake_runner(record, stdout=CHUNK_STDOUT, returncode=0, stderr=""):
    def runner(command, **kwargs):
        record.append({"command": list(command), "kwargs": kwargs})
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    return runner


# ------------------------------------------------------------------ parsing ---

def test_parse_chunks_reads_every_running_estimate():
    from scripts.gguf_eval import parse_perplexity_chunks

    assert parse_perplexity_chunks(CHUNK_STDOUT) == [4.0, 4.5, 4.25, 4.3]


def test_parse_chunks_refuses_output_with_no_chunks():
    from scripts.gguf_eval import parse_perplexity_chunks

    with pytest.raises(ValueError, match="chunk"):
        parse_perplexity_chunks("Final estimate: PPL = 4.3\n")


def test_chunk_nlls_invert_the_running_mean_exactly():
    from scripts.gguf_eval import chunk_nlls, running_perplexities

    truth = [1.0, 2.0, 1.5, 0.5]
    recovered = chunk_nlls(running_perplexities(truth))

    assert recovered == pytest.approx(truth)


def test_chunk_nlls_reproduce_the_final_perplexity():
    from scripts.gguf_eval import chunk_nlls

    nlls = chunk_nlls([4.0, 4.5, 4.25, 4.3])

    assert math.exp(sum(nlls) / len(nlls)) == pytest.approx(4.3)


def test_chunk_nlls_refuse_a_non_positive_estimate():
    from scripts.gguf_eval import chunk_nlls

    with pytest.raises(ValueError, match="positive"):
        chunk_nlls([4.0, 0.0])


# ------------------------------------------------------------------ running ---

def test_run_perplexity_invokes_stock_llama_perplexity_with_a_bounded_timeout(
        tmp_path):
    from scripts.gguf_eval import run_perplexity

    calls = []
    text = tmp_path / "ppl.txt"
    text.write_text("hello world\n")

    result = run_perplexity(tmp_path / "m.gguf", text,
                            binary=tmp_path / "llama-perplexity",
                            n_ctx=512, threads=6, timeout_s=99.0,
                            runner=_fake_runner(calls))

    command = calls[0]["command"]
    assert command[0] == str(tmp_path / "llama-perplexity")
    assert command[command.index("-m") + 1] == str(tmp_path / "m.gguf")
    assert command[command.index("-f") + 1] == str(text)
    assert command[command.index("-c") + 1] == "512"
    assert command[command.index("-t") + 1] == "6"
    assert command[command.index("-ngl") + 1] == "0"
    assert "--no-warmup" in command
    assert calls[0]["kwargs"]["timeout"] == 99.0
    assert result["perplexity"] == pytest.approx(4.3)
    assert len(result["chunk_nll"]) == 4


def test_run_perplexity_raises_on_a_failed_run(tmp_path):
    from scripts.gguf_eval import run_perplexity

    text = tmp_path / "ppl.txt"
    text.write_text("hello\n")

    with pytest.raises(RuntimeError, match="llama-perplexity"):
        run_perplexity(tmp_path / "m.gguf", text,
                       binary=tmp_path / "llama-perplexity",
                       runner=_fake_runner([], returncode=2, stderr="boom"))


# --------------------------------------------------------------- scorecards ---

def _write_cards(tmp_path, left_nlls, right_nlls):
    from scripts.gguf_eval import perplexity_scorecard
    from daedalus.scorecard import ArtifactRef, write_scorecard

    paths = {}
    for name, nlls, kind in (("fp16", left_nlls, "gguf-f16"),
                             ("q4_0", right_nlls, "gguf-q4_0")):
        card = perplexity_scorecard(
            name=f"perplexity-{name}",
            artifact=ArtifactRef(path=f"{name}.gguf", sha256="a" * 64, kind=kind),
            tokenizer_ref=ArtifactRef(path="t.json", sha256="b" * 64,
                                      kind="tokenizer"),
            chunk_nll=nlls, seed=1, git_sha="deadbee",
            text_file="ppl.txt", n_ctx=512, runtime={"llama_cpp_commit": "7584430"})
        paths[name] = write_scorecard(tmp_path / f"{name}.json", card)["scorecard"]
    return paths


def test_perplexity_scorecard_records_chunks_as_pairable_items(tmp_path):
    paths = _write_cards(tmp_path, [1.0, 2.0, 1.5], [1.1, 2.1, 1.4])

    card = load_scorecard(paths["fp16"])

    assert card.kind == "paired-quant"
    assert card.item_count == 3
    assert [item["id"] for item in card.items] == ["chunk-0", "chunk-1", "chunk-2"]
    assert card.metrics["perplexity"] == pytest.approx(math.exp(4.5 / 3))
    assert card.provenance.bpb_mode == "not-applicable"
    assert card.details["text_file"] == "ppl.txt"
    assert card.details["n_ctx"] == 512


def test_compare_quantization_reports_a_paired_penalty(tmp_path):
    from scripts.gguf_eval import compare_quantization

    paths = _write_cards(tmp_path, [1.0, 2.0, 1.5], [1.1, 2.1, 1.4])

    comparison = compare_quantization(load_scorecard(paths["fp16"]),
                                      load_scorecard(paths["q4_0"]))

    assert comparison["n_chunks"] == 3
    assert comparison["mean_nll_delta"] == pytest.approx(0.1 / 3, abs=1e-9)
    assert comparison["perplexity_fp16"] == pytest.approx(math.exp(4.5 / 3))
    assert comparison["perplexity_q4_0"] == pytest.approx(math.exp(4.6 / 3))
    assert comparison["q4_penalty_pct"] == pytest.approx(
        (math.exp(4.6 / 3) / math.exp(4.5 / 3) - 1) * 100)
    assert comparison["chunks_worse"] == 2
    assert comparison["chunks_better"] == 1


def test_compare_quantization_refuses_a_chunk_count_mismatch(tmp_path):
    from scripts.gguf_eval import compare_quantization

    paths = _write_cards(tmp_path, [1.0, 2.0, 1.5], [1.1, 2.1])

    with pytest.raises(ScorecardError, match="item_count"):
        compare_quantization(load_scorecard(paths["fp16"]),
                             load_scorecard(paths["q4_0"]))


def test_compare_quantization_refuses_artifacts_of_the_same_precision(tmp_path):
    from scripts.gguf_eval import compare_quantization
    from daedalus.scorecard import ArtifactRef

    paths = _write_cards(tmp_path, [1.0, 2.0], [1.1, 2.1])
    left = load_scorecard(paths["fp16"])
    right = load_scorecard(paths["q4_0"])
    right.provenance.artifact = ArtifactRef(path="also-f16.gguf", sha256="c" * 64,
                                            kind="gguf-f16")

    with pytest.raises(ScorecardError, match="precision"):
        compare_quantization(left, right)


def test_compare_quantization_refuses_a_different_evaluation_text(tmp_path):
    from scripts.gguf_eval import compare_quantization

    paths = _write_cards(tmp_path, [1.0, 2.0], [1.1, 2.1])
    left = load_scorecard(paths["fp16"])
    right = load_scorecard(paths["q4_0"])
    right.details["text_file"] = "some-other-corpus.txt"

    with pytest.raises(ScorecardError, match="text"):
        compare_quantization(left, right)


def test_write_comparison_emits_a_readable_record(tmp_path):
    from scripts.gguf_eval import compare_quantization, write_comparison

    paths = _write_cards(tmp_path, [1.0, 2.0, 1.5], [1.1, 2.1, 1.4])
    comparison = compare_quantization(load_scorecard(paths["fp16"]),
                                      load_scorecard(paths["q4_0"]))

    out = write_comparison(tmp_path / "quant-comparison.json", comparison)

    payload = json.loads(out.read_text())
    assert payload["q4_penalty_pct"] == pytest.approx(
        comparison["q4_penalty_pct"])
    assert payload["fp16_sha256"] == "a" * 64
