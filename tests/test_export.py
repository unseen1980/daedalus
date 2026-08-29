"""Tests for export.py. Offline / CPU-only for the HF conversion path (the
core correctness claim: our weights load into the real Lfm2ForCausalLM and
produce bit-identical logits). llama.cpp subprocess calls are tested via
monkeypatched subprocess.run -- actually building/running llama.cpp is a
one-time manual smoke check, not part of the automated suite.

Run: python -m pytest tests/test_export.py -v
"""
import json
import os
import re
import subprocess

import pytest
import torch

import export as export_module
from daedalus.config import PRESETS
from daedalus.model import Daedalus
from export import (
    MAX_PPL_DELTA_PCT,
    convert_to_gguf,
    export_hf_model,
    export_tokenizer,
    is_all_attention,
    measure_decode_speed,
    measure_perplexity,
    parse_perplexity,
    perplexity_delta_pct,
    quantize_gguf,
    setup_llama_cpp,
    to_hf_config,
    to_hf_state_dict,
    to_qwen3_config,
    to_qwen3_state_dict,
    verify_quantization,
)


# ------------------------------------------------------------- dense twin ---

def test_is_all_attention_distinguishes_the_twins():
    assert is_all_attention(PRESETS["dense-150m"])
    assert not is_all_attention(PRESETS["daedalus-150m"])
    assert not is_all_attention(PRESETS["tiny"])


def test_to_qwen3_config_mirrors_our_shape():
    cfg = PRESETS["dense-150m"]
    q = to_qwen3_config(cfg)
    assert q.hidden_size == cfg.hidden_size
    assert q.intermediate_size == cfg.block_ff_dim  # literal, never auto-adjusted
    assert q.num_hidden_layers == cfg.num_hidden_layers
    assert q.num_attention_heads == cfg.num_attention_heads
    assert q.num_key_value_heads == cfg.num_key_value_heads
    assert q.head_dim == cfg.head_dim
    # transformers stores rope theta under rope_parameters on Qwen3Config (it
    # is a per-layer attribute there), not as a plain `rope_theta` field --
    # read it the way the config actually serialises it, which is also what
    # convert_hf_to_gguf.py reads.
    assert q.to_dict()["rope_parameters"]["rope_theta"] == cfg.rope_theta
    assert q.tie_word_embeddings is True
    assert (q.bos_token_id, q.eos_token_id, q.pad_token_id) == (0, 0, 0)


def test_to_qwen3_state_dict_raises_on_a_hybrid_config():
    """A conv block has no q_proj -- catch the mistake at mapping time rather
    than shipping a silently half-populated model."""
    cfg = PRESETS["tiny"]
    ours = Daedalus(cfg)
    with pytest.raises(ValueError, match="unmapped Daedalus state_dict key"):
        to_qwen3_state_dict(ours.state_dict(), cfg.num_hidden_layers,
                            cfg.tie_word_embeddings)


def _dense_tiny():
    """A tiny all-attention config -- the dense twin's shape, laptop-sized."""
    import dataclasses
    return dataclasses.replace(PRESETS["tiny"], layer_types=None,
                               num_attention_blocks=PRESETS["tiny"].num_hidden_layers)


def test_dense_twin_exports_as_qwen3_with_identical_logits(tmp_path, monkeypatch):
    """The dense twin must NOT ship as lfm2: llama.cpp's lfm2 graph is hybrid
    (recurrent conv state + KV cache) and aborts in llama_decode on a conv-free
    model. Verified live -- convert and quantize succeed, then llama-bench dies
    with GGML_ASSERT(buffer). So it exports as Qwen3ForCausalLM, and that
    export has to be exact.
    """
    from transformers import Qwen3ForCausalLM
    from daedalus.muon import build_optimizers
    from train import save_checkpoint

    cfg = _dense_tiny()
    assert is_all_attention(cfg)
    monkeypatch.setitem(PRESETS, "dense-tiny", cfg)

    torch.manual_seed(0)
    ours = Daedalus(cfg)
    ours.eval()
    muon, adamw, _ = build_optimizers(ours)
    ckpt = save_checkpoint(str(tmp_path / "ckpt.pt"), ours, muon, adamw,
                           step=0, tokens_seen=0, cfg=cfg)

    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt, "dense-tiny", out_dir, dtype=torch.float32)

    with open(os.path.join(out_dir, "config.json")) as f:
        assert json.load(f)["model_type"] == "qwen3"

    hf_model = Qwen3ForCausalLM.from_pretrained(out_dir)
    hf_model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        our_logits, _, _ = ours(x, targets=None, return_logits=True)
        hf_logits = hf_model(input_ids=x).logits
    assert torch.allclose(our_logits.float(), hf_logits.float(), atol=1e-4)


def test_hybrid_still_exports_as_lfm2(tmp_path):
    """The dense-twin dispatch must not touch the hybrid path."""
    from daedalus.muon import build_optimizers
    from train import save_checkpoint

    cfg = PRESETS["tiny"]
    torch.manual_seed(0)
    ours = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(ours)
    ckpt = save_checkpoint(str(tmp_path / "ckpt.pt"), ours, muon, adamw,
                           step=0, tokens_seen=0, cfg=cfg)
    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out_dir, dtype=torch.float32)
    with open(os.path.join(out_dir, "config.json")) as f:
        assert json.load(f)["model_type"] == "lfm2"


# --------------------------------------------------------------- to_hf_config ---

def test_to_hf_config_forces_literal_ff_dim():
    cfg = PRESETS["daedalus-150m"]
    hf_cfg = to_hf_config(cfg)
    assert hf_cfg.intermediate_size == cfg.block_ff_dim
    assert hf_cfg.block_auto_adjust_ff_dim is False
    assert hf_cfg.layer_types == cfg.layer_types
    assert hf_cfg.vocab_size == cfg.vocab_size


def test_to_hf_config_carries_token_ids():
    cfg = PRESETS["daedalus-150m"]
    hf_cfg = to_hf_config(cfg)
    d = cfg.to_hf_dict()
    assert hf_cfg.bos_token_id == d["bos_token_id"]
    assert hf_cfg.eos_token_id == d["eos_token_id"]
    assert hf_cfg.pad_token_id == d["pad_token_id"]


# ----------------------------------------------------------- to_hf_state_dict ---

def test_to_hf_state_dict_maps_known_keys_and_ties_embeddings():
    sd = {
        "embed_tokens.weight": torch.randn(10, 4),
        "norm.weight": torch.randn(4),
        "layers.0.conv.conv.weight": torch.randn(4, 1, 3),
    }
    hf_sd = to_hf_state_dict(sd, tie_word_embeddings=True)
    assert set(hf_sd.keys()) == {
        "model.embed_tokens.weight", "model.embedding_norm.weight",
        "model.layers.0.conv.conv.weight", "lm_head.weight",
    }
    assert torch.equal(hf_sd["lm_head.weight"], hf_sd["model.embed_tokens.weight"])


def test_to_hf_state_dict_respects_explicit_lm_head_when_not_tied():
    sd = {
        "embed_tokens.weight": torch.randn(10, 4),
        "norm.weight": torch.randn(4),
        "lm_head.weight": torch.randn(10, 4),
    }
    hf_sd = to_hf_state_dict(sd, tie_word_embeddings=False)
    assert hf_sd["lm_head.weight"].shape == (10, 4)
    assert not torch.equal(hf_sd["lm_head.weight"], hf_sd["model.embed_tokens.weight"])


def test_to_hf_state_dict_raises_on_unmapped_key():
    with pytest.raises(ValueError):
        to_hf_state_dict({"totally_unknown_key": torch.zeros(1)}, tie_word_embeddings=True)


# ---------------------------------------------------------------- full export ---

def test_export_hf_model_roundtrip_matches_original_logits(tmp_path):
    """The core correctness claim: our checkpoint, loaded into the real HF
    Lfm2ForCausalLM via export_hf_model + save_pretrained/from_pretrained,
    must reproduce our own model's forward pass -- this is what makes the
    downstream GGUF conversion trustworthy."""
    from transformers import Lfm2ForCausalLM
    from daedalus.muon import build_optimizers
    from train import save_checkpoint

    cfg = PRESETS["tiny"]
    torch.manual_seed(0)
    ours = Daedalus(cfg)
    ours.eval()
    muon, adamw, _ = build_optimizers(ours)
    ckpt_path = save_checkpoint(str(tmp_path / "ckpt.pt"), ours, muon, adamw,
                                step=0, tokens_seen=0, cfg=cfg)

    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt_path, "tiny", out_dir, dtype=torch.float32)

    assert os.path.exists(os.path.join(out_dir, "config.json"))
    hf_model = Lfm2ForCausalLM.from_pretrained(out_dir)
    hf_model.eval()

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        our_logits, _, _ = ours(x, targets=None, return_logits=True)
        hf_logits = hf_model(input_ids=x).logits

    assert torch.allclose(our_logits.float(), hf_logits.float(), atol=1e-4)


def test_export_carries_a_qat_runs_master_weights_through_unchanged(tmp_path):
    """The composition Phase 3 depends on, through the real export function.

    `test_qat.py` proves `q4_0_qdq` is `llama-quantize`'s grid, and that fp16
    storage does not move an already-on-grid weight. Neither says what
    `export_hf_model` does with a checkpoint written *while QAT was active* --
    and that is where the weights could be swapped for something else, because
    under `parametrize` the module's `weight` attribute is the quantized view
    and only `parametrizations.weight.original` is the trainable master.

    Exporting the quantized view would not look wrong: it is on the grid, it
    loads, and it converts. It would simply have thrown away the full-precision
    master that the next recovery stage is supposed to continue from, and
    silently re-quantized an already-quantized tensor.
    """
    from daedalus import qat
    from daedalus.muon import build_optimizers
    from train import save_checkpoint

    cfg = PRESETS["tiny"]
    torch.manual_seed(0)
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model)

    qat.enable_qat(model)
    masters = {name: qat.master_weight(mod).detach().clone()
               for name, mod, _ in qat.plan_qat(model)}
    quantized_view = {name: dict(model.named_modules())[name].weight.detach().clone()
                      for name in masters}
    # The premise: the two really are different tensors here.
    assert any(not torch.equal(masters[n], quantized_view[n]) for n in masters)

    ckpt_path = save_checkpoint(str(tmp_path / "qat.pt"), model, muon, adamw,
                                step=7, tokens_seen=1024, cfg=cfg)
    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt_path, "tiny", out_dir, dtype=torch.float32)

    from safetensors.torch import load_file
    exported = load_file(os.path.join(out_dir, "model.safetensors"))
    hf_names = {"embed_tokens": "model.embed_tokens.weight"}

    checked = 0
    for name, master in masters.items():
        hf_name = hf_names.get(name, f"model.{name}.weight")
        if hf_name not in exported:
            continue
        assert torch.equal(exported[hf_name].float(), master.float()), hf_name
        checked += 1
    assert checked, "no QAT tensor was matched to an exported one"


def test_export_hf_model_untied_head(tmp_path):
    """Same equivalence check, but for an untied-embeddings config -- a
    separate code path in to_hf_state_dict (no synthetic lm_head copy)."""
    import copy
    from transformers import Lfm2ForCausalLM
    from daedalus.muon import build_optimizers
    from train import save_checkpoint

    cfg = copy.deepcopy(PRESETS["tiny"])
    cfg.tie_word_embeddings = False
    torch.manual_seed(1)
    ours = Daedalus(cfg)
    ours.eval()
    muon, adamw, _ = build_optimizers(ours)
    ckpt_path = save_checkpoint(str(tmp_path / "ckpt.pt"), ours, muon, adamw,
                                step=0, tokens_seen=0, cfg=cfg)

    out_dir = str(tmp_path / "hf")
    # PRESETS["tiny"] is a module-level singleton with tie_word_embeddings=True;
    # export_hf_model looks it up by name, so patch the module's copy in place
    # for the duration of this test rather than mutating global state permanently.
    original = PRESETS["tiny"]
    PRESETS["tiny"] = cfg
    try:
        export_hf_model(ckpt_path, "tiny", out_dir, dtype=torch.float32)
    finally:
        PRESETS["tiny"] = original

    hf_model = Lfm2ForCausalLM.from_pretrained(out_dir)
    hf_model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        our_logits, _, _ = ours(x, targets=None, return_logits=True)
        hf_logits = hf_model(input_ids=x).logits
    assert torch.allclose(our_logits.float(), hf_logits.float(), atol=1e-4)


# -------------------------------------------------------------- perplexity ---

def test_parse_perplexity_extracts_value():
    output = "some setup lines\nFinal estimate: PPL = 12.3456 +/- 0.0789\ndone\n"
    assert parse_perplexity(output) == pytest.approx(12.3456)


def test_parse_perplexity_raises_when_missing():
    with pytest.raises(ValueError):
        parse_perplexity("no perplexity line here")


def test_perplexity_delta_pct():
    assert perplexity_delta_pct(10.0, 10.1) == pytest.approx(1.0)
    assert perplexity_delta_pct(10.0, 10.0) == 0.0
    assert perplexity_delta_pct(10.0, 9.9) == pytest.approx(1.0)


# -------------------------------------------------------------- llama.cpp ---

def test_quantize_gguf_never_passes_imatrix_flag(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    out = quantize_gguf("in.gguf", str(tmp_path / "out.gguf"), "/fake/llama.cpp",
                        qtype="Q4_0")
    assert "--imatrix" not in captured["cmd"]
    assert captured["cmd"][-1] == "Q4_0"
    assert captured["cmd"][0].endswith(os.path.join("build", "bin", "llama-quantize"))
    assert out == str(tmp_path / "out.gguf")


def test_quantize_gguf_leaves_token_embd_at_llama_cpp_default(monkeypatch, tmp_path):
    """`token_embd` must ship Q6_K, NOT the blueprint's Q8_0.

    DAEDALUS-BLUEPRINT-v6.md:59 says "Keep `token_embd`/`output` at Q8_0", so a
    future reader greping for Q8_0 has every reason to add
    `--token-embedding-type q8_0` here and think they are fixing a bug. They
    would be giving back 10.2% of CPU decode -- the project's headline metric --
    to buy +0.045% perplexity, which is inside its own error bars.

    Measured on a real 500M-token checkpoint, alternating back to back:
    Q6_K embd = 1030.5 tok/s / 95.56 MiB / PPL 35.9114; Q8_0 embd = 924.9 tok/s
    / 104.28 MiB / PPL 35.8954. Full writeup and the tensor-type dump of a real
    .gguf: runs/preflight/token-embd-quant-grid.md.

    So this test pins a deliberate deviation, not an implementation detail.
    """
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    quantize_gguf("in.gguf", str(tmp_path / "out.gguf"), "/fake/llama.cpp",
                  qtype="Q4_0")
    assert "--token-embedding-type" not in captured["cmd"]
    assert "--output-tensor-type" not in captured["cmd"]
    # Exactly [bin, in, out, qtype] -- no type overrides of any kind.
    assert len(captured["cmd"]) == 4, captured["cmd"]


def test_convert_to_gguf_builds_expected_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    out_path = str(tmp_path / "model.gguf")
    convert_to_gguf("/fake/hf_dir", out_path, "/fake/llama.cpp", outtype="f16")
    # must invoke this venv's interpreter (has torch), not a bare "python3"
    # which can resolve to a torch-less system Python -- found live.
    assert captured["cmd"][0] == export_module.sys.executable
    assert captured["cmd"][0] != "python3"
    assert captured["cmd"][1].endswith("convert_hf_to_gguf.py")
    assert "/fake/hf_dir" in captured["cmd"]
    assert "--outtype" in captured["cmd"] and "f16" in captured["cmd"]
    assert out_path in captured["cmd"]


def test_measure_perplexity_parses_subprocess_output(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Final estimate: PPL = 7.7000 +/- 0.01\n", stderr="")

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    ppl = measure_perplexity("model.gguf", "/fake/llama.cpp", "text.txt")
    assert ppl == pytest.approx(7.7)


def test_setup_llama_cpp_skips_clone_and_build_when_already_present(monkeypatch, tmp_path):
    build_bin = tmp_path / "build" / "bin"
    build_bin.mkdir(parents=True)
    (tmp_path / "convert_hf_to_gguf.py").write_text("# stub")
    (build_bin / "llama-quantize").write_text("stub")
    (build_bin / "llama-perplexity").write_text("stub")
    (build_bin / "llama-bench").write_text("stub")

    def fail_run(cmd, **kwargs):
        raise AssertionError(f"subprocess.run should not be called, got {cmd}")

    monkeypatch.setattr(export_module.subprocess, "run", fail_run)
    result = setup_llama_cpp(str(tmp_path))
    assert result == str(tmp_path)


def test_setup_llama_cpp_rebuilds_when_only_bench_missing(monkeypatch, tmp_path):
    """A box that ran the pre-llama-bench version of setup_llama_cpp has
    llama-quantize/llama-perplexity but not llama-bench -- must not be
    treated as already-set-up."""
    build_bin = tmp_path / "build" / "bin"
    build_bin.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / "convert_hf_to_gguf.py").write_text("# stub")
    (build_bin / "llama-quantize").write_text("stub")
    (build_bin / "llama-perplexity").write_text("stub")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    setup_llama_cpp(str(tmp_path))
    assert any("llama-bench" in c for cmd in calls for c in cmd)


def _fake_bench(monkeypatch, per_depth=None, default=(174.9, 3.2, 16),
                fail_depths=(), hang_depths=()):
    """Stub llama-bench, keyed on the `-d` value in the command line."""
    calls = []

    def fake_run(cmd, **kwargs):
        depth = int(cmd[cmd.index("-d") + 1]) if "-d" in cmd else 0
        calls.append({"cmd": cmd, "depth": depth, "kwargs": kwargs})
        if depth in hang_depths:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        if depth in fail_depths:
            raise subprocess.CalledProcessError(1, cmd, stderr="boom")
        avg, sd, nt = (per_depth or {}).get(depth, default)
        n_gen = int(cmd[cmd.index("-n") + 1])
        stdout = json.dumps([{"avg_ts": avg, "stddev_ts": sd,
                              "n_threads": nt, "n_gen": n_gen}])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    return calls


def test_measure_decode_speed_parses_llama_bench_json(monkeypatch):
    calls = _fake_bench(monkeypatch)
    result = measure_decode_speed("model.gguf", "/fake/llama.cpp", n_gen=64,
                                  depths=(0,))

    # The depth-0 keys are the contract every existing consumer reads
    # (`scripts/abl_table.py`, the model card, `runs/eval/*.json`); growing
    # depths must not move them.
    assert result["tok_per_sec"] == 174.9
    assert result["tok_per_sec_stddev"] == 3.2
    assert result["n_gen"] == 64
    assert result["n_threads"] == 16

    cmd = calls[0]["cmd"]
    assert "-p" in cmd and "0" in cmd
    assert "-n" in cmd and "64" in cmd
    assert "-o" in cmd and "json" in cmd
    # Depth 0 is llama-bench's own default, so the command stays byte-identical
    # to the one that produced every number already on record.
    assert "-d" not in cmd


def test_measure_decode_speed_passes_thread_count(monkeypatch):
    calls = _fake_bench(monkeypatch, default=(1.0, 0.0, 4))
    measure_decode_speed("model.gguf", "/fake/llama.cpp", n_gen=32, threads=4,
                         depths=(0,))
    assert "-t" in calls[0]["cmd"] and "4" in calls[0]["cmd"]


def test_measure_decode_speed_reports_the_trained_context_by_default():
    """2048 must be in the default sweep, not an opt-in.

    The project's headline Pareto claim is hybrid-vs-dense CPU decode, and
    `abl_arch.py` calls this function with no `depths` argument. A default of
    depth 0 alone measures a conv hybrid exactly where it has least to gain --
    1.06x against SmolLM2 at depth 0 versus 2.08x at 2048 on this box.
    """
    assert 0 == export_module.DECODE_DEPTHS[0], "depth 0 stays the headline key"
    assert 2048 in export_module.DECODE_DEPTHS


def test_measure_decode_speed_sweeps_depths_and_keys_them(monkeypatch):
    calls = _fake_bench(monkeypatch, per_depth={
        0: (960.9, 3.4, 8), 512: (933.7, 28.5, 8), 2048: (648.6, 12.6, 8)})
    result = measure_decode_speed("model.gguf", "/fake/llama.cpp",
                                  depths=(0, 512, 2048))

    assert [c["depth"] for c in calls] == [0, 512, 2048]
    assert result["by_depth"]["2048"]["tok_per_sec"] == 648.6
    assert result["by_depth"]["512"]["tok_per_sec"] == 933.7
    assert result["by_depth"]["0"]["tok_per_sec"] == 960.9
    # Top level still reads depth 0, not the last depth measured.
    assert result["tok_per_sec"] == 960.9


def test_measure_decode_speed_deep_failure_does_not_lose_the_export(monkeypatch):
    """A failed deep measurement is recorded, not raised.

    This runs inside `abl_arch.export_and_bench`, whose caller catches
    exceptions per arm -- so a raise here would cost a 12-hour arm its GGUF
    and its perplexity check to gain nothing. The measurement that already
    worked before depths existed (depth 0) must still be returned.
    """
    _fake_bench(monkeypatch, default=(960.9, 3.4, 8), fail_depths=(2048,))
    result = measure_decode_speed("model.gguf", "/fake/llama.cpp",
                                  depths=(0, 2048))

    assert result["tok_per_sec"] == 960.9
    assert result["by_depth"]["2048"]["tok_per_sec"] is None
    assert "CalledProcessError" in result["by_depth"]["2048"]["error"]


def test_every_bench_invocation_is_time_bounded(monkeypatch):
    """A wedged llama-bench must not stall the export it runs inside.

    The best-effort contract above is written against *exceptions*, and a hang
    is not one: with no timeout `subprocess.run` waits forever, so the export
    step never returns and `abl_arch.py` waits on it with the GPU idle. The
    Hub uploader wedged in exactly this shape twice on 2026-08-10 and no
    `except` could see it.
    """
    calls = _fake_bench(monkeypatch, default=(960.9, 3.4, 8))
    measure_decode_speed("model.gguf", "/fake/llama.cpp", depths=(0, 512, 2048))

    assert len(calls) == 3
    for call in calls:
        timeout = call["kwargs"].get("timeout")
        assert timeout, f"depth {call['depth']} bench has no timeout"
        # The live three-depth sweep on the real Q4_0 GGUF measures 14.8 s
        # (runs/preflight/export-depth-sweep-live.md). A bound anywhere near
        # that would turn a slow box into a lost measurement, which is the
        # opposite of the point.
        assert timeout >= 300.0, "too tight to be safe on a loaded box"


def test_a_wedged_deep_bench_is_recorded_like_any_other_failure(monkeypatch):
    """Timing out at depth 2048 costs the deep number, never the export."""
    _fake_bench(monkeypatch, default=(960.9, 3.4, 8), hang_depths=(2048,))
    result = measure_decode_speed("model.gguf", "/fake/llama.cpp",
                                  depths=(0, 2048))

    assert result["tok_per_sec"] == 960.9, "depth 0 still returned"
    assert result["by_depth"]["2048"]["tok_per_sec"] is None
    assert "TimeoutExpired" in result["by_depth"]["2048"]["error"]


def test_measure_decode_speed_depth_zero_failure_still_raises(monkeypatch):
    """Depth 0 keeps the old strict contract -- no benchmark at all is a bug."""
    _fake_bench(monkeypatch, fail_depths=(0,))
    with pytest.raises(subprocess.CalledProcessError):
        measure_decode_speed("model.gguf", "/fake/llama.cpp", depths=(0, 2048))


def test_measure_decode_speed_always_measures_depth_zero(monkeypatch):
    """Depth 0 backs the top-level keys, so it is added if a caller omits it."""
    calls = _fake_bench(monkeypatch)
    result = measure_decode_speed("model.gguf", "/fake/llama.cpp",
                                  depths=(2048,))
    assert [c["depth"] for c in calls] == [0, 2048]
    assert result["tok_per_sec"] == 174.9


def test_measure_decode_speed_selects_the_decode_row(monkeypatch):
    """A prefill row must never be mistaken for the decode measurement."""
    def fake_run(cmd, **kwargs):
        stdout = json.dumps([
            {"avg_ts": 9999.0, "stddev_ts": 1.0, "n_threads": 8, "n_gen": 0},
            {"avg_ts": 648.6, "stddev_ts": 12.6, "n_threads": 8, "n_gen": 128},
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    result = measure_decode_speed("model.gguf", "/fake/llama.cpp", n_gen=128,
                                  depths=(0,))
    assert result["tok_per_sec"] == 648.6


# --------------------------------------------------------- verify_quantization ---

def _stub_verify_quantization(monkeypatch, fp16_ppl, q4_0_ppl):
    calls = []
    monkeypatch.setattr(export_module, "convert_to_gguf",
                         lambda *a, **k: calls.append(("convert", a, k)))
    monkeypatch.setattr(export_module, "quantize_gguf",
                         lambda *a, **k: calls.append(("quantize", a, k)))
    ppls = iter([fp16_ppl, q4_0_ppl])
    monkeypatch.setattr(export_module, "measure_perplexity",
                         lambda *a, **k: (calls.append(("ppl", a, k)), next(ppls))[1])
    return calls


def test_verify_quantization_passes_within_threshold(monkeypatch, tmp_path):
    calls = _stub_verify_quantization(monkeypatch, fp16_ppl=7.0, q4_0_ppl=7.0 * (1 + MAX_PPL_DELTA_PCT / 200))
    result = verify_quantization("/fake/hf_dir", "/fake/llama.cpp", "text.txt", str(tmp_path))

    assert result["passes_threshold"] is True
    assert result["delta_pct"] == pytest.approx(perplexity_delta_pct(result["fp16_ppl"], result["q4_0_ppl"]))
    assert result["fp16_gguf"] == os.path.join(str(tmp_path), "model-f16.gguf")
    assert result["q4_0_gguf"] == os.path.join(str(tmp_path), "model-q4_0.gguf")
    # conversion must run before quantization, and quantization before either perplexity measurement
    kinds = [c[0] for c in calls]
    assert kinds.index("convert") < kinds.index("quantize") < kinds.index("ppl")
    # fp16 gguf is measured before the quantized one, matching fp16_ppl/q4_0_ppl order
    ppl_paths = [c[1][0] for c in calls if c[0] == "ppl"]
    assert ppl_paths == [result["fp16_gguf"], result["q4_0_gguf"]]


def test_verify_quantization_fails_over_threshold(monkeypatch, tmp_path):
    _stub_verify_quantization(monkeypatch, fp16_ppl=7.0, q4_0_ppl=7.0 * (1 + (MAX_PPL_DELTA_PCT + 5) / 100))
    result = verify_quantization("/fake/hf_dir", "/fake/llama.cpp", "text.txt", str(tmp_path))
    assert result["passes_threshold"] is False
    assert result["delta_pct"] > MAX_PPL_DELTA_PCT


# --------------------------------------------------------------- model card ---
# The `hero` hard precondition requires the Hub checkpoint repo and the exact
# branch-from-milestone command to be recorded in the model card. Before this,
# export.py wrote no card at all, so these tests are the only thing standing
# between "documented" and "was documented in STATUS.md once".

def _tiny_checkpoint(tmp_path, cfg_name="tiny", tokens_seen=0, run_name="somerun"):
    from daedalus.muon import build_optimizers
    from train import save_checkpoint

    cfg = PRESETS[cfg_name]
    torch.manual_seed(0)
    ours = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(ours)
    run_dir = tmp_path / "runs" / run_name
    run_dir.mkdir(parents=True)
    return save_checkpoint(str(run_dir / "checkpoint.pt"), ours, muon, adamw,
                           step=7, tokens_seen=tokens_seen, cfg=cfg)


MILESTONE = {
    "revision": "hero-stable-end-step1234",
    "step": 1234,
    "tokens_seen": 22_000_000_000,
    "decay_frac": 0.45,
    "lr_mult_at_branch": 1.0,
    "muon_lr": 0.02,
    "adam_lr": 3e-4,
    "config": "daedalus-150m",
    "repo": "Unseen1980/daedalus-checkpoints",
    "path_in_repo": "milestone/hero/checkpoint.pt",
}


def test_export_writes_a_model_card_stating_the_success_bar(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path)
    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out_dir, dtype=torch.float32)

    with open(os.path.join(out_dir, "README.md")) as f:
        card = f.read()
    # Frontmatter first, or the Hub renders the card as plain text.
    assert card.startswith("---\n")
    assert export_module.SUCCESS_BAR in card
    assert "Deviations from the blueprint" in card


def test_model_card_publishes_the_branch_command_from_the_milestone(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path)
    with open(os.path.join(os.path.dirname(ckpt), "milestone.json"), "w") as f:
        json.dump(MILESTONE, f)
    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out_dir, dtype=torch.float32)

    with open(os.path.join(out_dir, "README.md")) as f:
        card = f.read()
    assert ("--resume 'hub://Unseen1980/daedalus-checkpoints/"
            "milestone/hero/checkpoint.pt?rev=hero-stable-end-step1234'") in card
    # The run name is not a field of the milestone record; it has to be read
    # out of path_in_repo, and a wrong one makes the command unrunnable.
    assert "--run-name hero-ext" in card


def test_the_branch_command_carries_every_flag_the_run_needs():
    """The published command was `--run-name` + `--resume` and nothing else,
    and **both** omissions fail without an error. Executed rather than reasoned
    about (`runs/preflight/branch-command.md`), against a real 150M milestone
    checkpoint whose step/tokens_seen were set to hero's:

    | command                    | result                                |
    |----------------------------|---------------------------------------|
    | as published               | 0 steps, rc 0, no metrics.jsonl       |
    | + `--total-tokens`         | trains on random tokens, loss 13.04   |
    | + `--data-dir` as well     | loss 4.74 -> 3.94 on real text        |

    `--total-tokens` defaults to 5e9 while hero's milestone carries 30.5e9
    tokens_seen, so `fit()` breaks at the top of its first iteration
    (`train.py:1579`); `--data-dir` defaults to None, which selects
    `SyntheticBatchSource`.
    """
    command = export_module._branch_command(MILESTONE)

    assert "--data-dir" in command, \
        "without it, branching trains the model on random tokens"
    assert "--total-tokens" in command, \
        "without it, the 5e9 default is below the milestone and nothing trains"
    # The bound has to come from the record: a fixed number would go stale the
    # moment the budget changes, which is how 40B became 60B mid-project.
    assert str(MILESTONE["tokens_seen"]) in command, \
        "the command must name the budget the new one has to exceed"


def test_the_branch_command_names_the_arms_own_config():
    """`--config` defaults to `daedalus-150m`, so the dense twin's own card
    published a command that would load its weights into the hybrid class."""
    dense = {**MILESTONE, "config": "dense-150m",
             "path_in_repo": "milestone/abl-arch-dense-150m/checkpoint.pt"}
    assert "--config dense-150m" in export_module._branch_command(dense)
    assert "--config daedalus-150m" in export_module._branch_command(MILESTONE)


def test_the_branch_command_parses_against_train_pys_real_parser():
    """Anchors the command to the parser rather than to a literal, so a renamed
    flag fails here instead of at the end of a six-day run.

    The two placeholders are substituted with real values first -- they are
    placeholders precisely because the card cannot know them -- and what is
    asserted is that everything else parses and lands where it should."""
    import shlex
    import train

    command = export_module._branch_command(MILESTONE)
    filled = (command.replace("\\\n", " ")
                     .replace("<YOUR_SHARD_DIR>", "/tmp/shards")
                     .replace(f"<NEW_BUDGET_GREATER_THAN_{MILESTONE['tokens_seen']}>",
                              str(MILESTONE["tokens_seen"] * 2)))
    argv = shlex.split(filled)
    assert argv[:2] == ["python", "train.py"], filled
    args = train.parse_args(argv[2:])       # raises SystemExit on any bad flag

    assert args.data_dir == "/tmp/shards"
    assert args.config == MILESTONE["config"]
    assert args.resume.startswith("hub://")
    assert MILESTONE["revision"] in args.resume
    # The one that decides whether the run does anything at all.
    assert args.total_tokens > MILESTONE["tokens_seen"], \
        "the command's own budget must clear the milestone it resumes"


def test_branch_command_survives_a_milestone_without_a_run_segment():
    command = export_module._branch_command(
        {**MILESTONE, "path_in_repo": "checkpoint.pt"})
    assert command is None or "--run-name" in command


def test_model_card_quotes_the_checkpoints_tokens_not_the_milestones(tmp_path):
    """The milestone is written at 55% of a run. Quoting its `tokens_seen` as
    the training total would understate every finished run by 45%."""
    ckpt = _tiny_checkpoint(tmp_path, tokens_seen=40_000_000_000)
    with open(os.path.join(os.path.dirname(ckpt), "milestone.json"), "w") as f:
        json.dump(MILESTONE, f)
    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out_dir, dtype=torch.float32)

    with open(os.path.join(out_dir, "README.md")) as f:
        card = f.read()
    assert "tokens seen: 40,000,000,000" in card
    assert "tokens seen: 22,000,000,000" not in card


def test_model_card_says_not_measured_rather_than_omitting_the_eval_table():
    """A missing section reads as 'not applicable'. For an eval table that is
    the one thing it must never mean."""
    card = export_module.render_model_card("daedalus-150m")
    assert "## Evaluation" in card
    assert "_Not yet measured for this export._" in card


def test_model_card_renders_measured_eval_and_quantization():
    card = export_module.render_model_card(
        "daedalus-150m",
        metrics={"val_bpb": 0.9123, "hellaswag_acc_norm": 0.311},
        quantization={"fp16_ppl": 21.5, "q4_0_ppl": 21.68, "delta_pct": 0.837,
                      "passes_threshold": True,
                      "decode_speed": {"tok_per_sec": 958.7,
                                       "tok_per_sec_stddev": 29.9,
                                       "n_threads": 6}})
    assert "| val_bpb | 0.9123 |" in card
    assert "| hellaswag_acc_norm | 0.3110 |" in card
    assert "0.837%" in card
    assert "958.7 tok/s" in card
    # An export predating the depth sweep has only the top-level number. It is
    # still depth 0, and the card must say so rather than leave a bare tok/s
    # figure that cannot be reproduced.
    assert "context depth 0" in card
    assert "_Not yet measured for this export._" not in card


def test_model_card_reports_decode_speed_by_depth():
    """The card is what the operator reads on the Hub, and an unlabelled tok/s
    number is the depth-0 one -- the least interesting number this model
    produces, since its whole design is that decode barely slows with context.
    """
    card = export_module.render_model_card(
        "daedalus-150m",
        quantization={"delta_pct": 0.9, "passes_threshold": True,
                      "decode_speed": {
                          "tok_per_sec": 951.9, "tok_per_sec_stddev": 37.8,
                          "n_threads": 8,
                          "by_depth": {
                              "0": {"depth": 0, "tok_per_sec": 951.9,
                                    "tok_per_sec_stddev": 37.8},
                              "2048": {"depth": 2048, "tok_per_sec": 648.5,
                                       "tok_per_sec_stddev": 7.0}}}})
    assert "648.5 tok/s" in card
    assert "951.9 tok/s" in card
    assert "the trained context" in card


def test_model_card_omits_a_depth_that_failed_to_measure():
    """A best-effort depth that failed records tok_per_sec None. It must not
    reach the card as a 0.0 tok/s claim."""
    card = export_module.render_model_card(
        "daedalus-150m",
        quantization={"delta_pct": 0.9, "passes_threshold": True,
                      "decode_speed": {
                          "tok_per_sec": 951.9, "tok_per_sec_stddev": 37.8,
                          "n_threads": 8,
                          "by_depth": {
                              "0": {"depth": 0, "tok_per_sec": 951.9,
                                    "tok_per_sec_stddev": 37.8},
                              "2048": {"depth": 2048, "tok_per_sec": None,
                                       "error": "CalledProcessError"}}}})
    assert "951.9 tok/s" in card
    assert "0.0 tok/s" not in card
    assert "depth **2048**" not in card


def test_model_card_falls_back_to_the_default_repo_when_a_run_opted_out(monkeypatch):
    """`sweep` passes --hub-repo "", so its milestone records repo=None. The
    card must still name where checkpoints go, not print 'None'."""
    monkeypatch.delenv("DAEDALUS_HF_MODEL_REPO", raising=False)
    card = export_module.render_model_card(
        "daedalus-150m", milestone={**MILESTONE, "repo": None})
    assert export_module.ckpt_uploader.DEFAULT_MODEL_REPO in card
    assert "**`None`**" not in card
    # No repo means no publishable branch point -- say so rather than emit a
    # hub:// command that resolves nowhere.
    assert "hub://None" not in card


def test_model_card_reports_the_dense_twins_real_export_class():
    card = export_module.render_model_card("dense-150m")
    assert "`Qwen3ForCausalLM`" in card
    assert "`Lfm2ForCausalLM`" not in card
    assert export_module.render_model_card("daedalus-150m").count(
        "`Lfm2ForCausalLM`") == 1


def test_model_card_records_the_blueprint_interleave():
    card = export_module.render_model_card("daedalus-150m")
    assert "ccccAccAcAcAcAccAc" in card  # AGENT.md SS2, conv-first


def test_write_model_card_never_raises_and_never_kills_an_export(tmp_path, monkeypatch):
    """A card is documentation. An export that already produced correct
    weights must not fail because a string could not be formatted."""
    monkeypatch.setattr(export_module, "render_model_card",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert export_module.write_model_card(str(tmp_path), "tiny") is None

    # ...and the same failure inside a real export leaves the weights intact.
    ckpt = _tiny_checkpoint(tmp_path)
    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out_dir, dtype=torch.float32)
    assert os.path.exists(os.path.join(out_dir, "config.json"))
    assert not os.path.exists(os.path.join(out_dir, "README.md"))


def test_a_malformed_milestone_costs_a_section_not_the_export(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path)
    with open(os.path.join(os.path.dirname(ckpt), "milestone.json"), "w") as f:
        f.write("{not json")
    out_dir = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out_dir, dtype=torch.float32)
    with open(os.path.join(out_dir, "README.md")) as f:
        card = f.read()
    assert "no branch point is published" in card


def test_the_perplexity_text_holds_no_literal_special_token_strings():
    """`llama-perplexity` does not parse special tokens, so a literal
    `<|endoftext|>` in the eval text is spelled out as ~7 junk tokens.

    Measured on 2026-08-10: the file carried 153 of them as document
    separators. HF maps each to the single id 0 (the ids the corpus was built
    with); `common_tokenize(ctx, prompt, true)` in
    `tools/perplexity/perplexity.cpp:475` passes `parse_special=false`, so
    llama.cpp instead scored ~1,000 tokens of mangled separator text and its
    chunking drifted out of alignment with the corpus tokenizer -- 294 chunks
    against HF's 292. Both effects on the same real GGUF and text span:

        with separators   fp16 9.6791  Q4_0 9.8413  delta +1.676%
        without           fp16 8.8776  Q4_0 9.0357  delta +1.781%

    So absolute perplexity read ~9% high and the Q4_0 damage the QAT decision
    rests on read ~6% *low*. The fp16-vs-Q4_0 gate survived it because both
    sides see the same junk, which is exactly why nothing caught it.

    See runs/preflight/gguf-vs-pytorch-fidelity.md.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "eval", "ppl-finewiki-150k.txt")
    if not os.path.exists(path):
        pytest.skip("perplexity text not present")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    found = re.findall(r"<\|[a-z_]+\|>", text)
    assert not found, (
        f"{len(found)} literal special-token strings in the perplexity text "
        f"({sorted(set(found))}); llama-perplexity will spell them out")


def test_the_exported_chat_template_matches_what_post_trains_on(tmp_path):
    """The shipped artifact must declare the prompt format `post.py` trained.

    `daedalus/chatml.py` warns that a train/inference formatting mismatch "is
    invisible at training time and shows up only as a model that ignores its
    prompt", and requires post.py and export to share one renderer. SmolLM2
    ships no `chat_template`, so before this the exported tokenizer had none
    and neither did the GGUF -- inference fell through to whatever the consumer
    defaulted to. llama.cpp happens to default to chatml, which happens to be
    ours; this pins it as a contract instead of a coincidence.
    """
    from transformers import AutoTokenizer
    from daedalus.chatml import render_messages, render_prompt

    out = str(tmp_path / "tok")
    export_tokenizer(out)
    tok = AutoTokenizer.from_pretrained(out)
    assert tok.chat_template, "no chat template reached the exported tokenizer"

    convos = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "system", "content": "Be concise."},
         {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello."},
         {"role": "user", "content": "And now?"}],
    ]
    for messages in convos:
        assert tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) == render_prompt(messages)
        assert tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ) == render_messages(messages)


def test_the_exported_vocabulary_is_still_byte_identical_to_smollm2(tmp_path):
    """Adding a chat template must not disturb a single token id.

    AGENT.md SS2 locks the tokenizer, and llama.cpp's converter keys its
    pre-tokenizer on a hash of the vocabulary -- so this is the assertion that
    makes the template safe rather than merely convenient.
    """
    from transformers import AutoTokenizer

    out = str(tmp_path / "tok")
    export_tokenizer(out)
    ours = AutoTokenizer.from_pretrained(out)
    upstream = AutoTokenizer.from_pretrained(export_module.SMOLLM2_TOKENIZER)

    assert ours.get_vocab() == upstream.get_vocab()
    assert (ours.bos_token_id, ours.eos_token_id) == \
           (upstream.bos_token_id, upstream.eos_token_id)
    probe = "The capital of France is Paris.\n\nWater boils at 100°C."
    assert ours(probe).input_ids == upstream(probe).input_ids


# --------------------------------------------------- unattended timeouts ---
#
# `hero`'s export runs at the end of a ~6-day run with nothing watching it.
# `measure_decode_speed` was bounded on 2026-08-10; the other four call sites
# were left unbounded on purpose, because picking a value before their real
# durations were known risked killing a legitimately slow `llama-perplexity`.
# The durations were measured off the two real `abl-arch` exports (see the
# comment above CONVERT_TIMEOUT_S), so these now close the gap.


def _capture_run(monkeypatch, raise_timeout=False, stdout=""):
    """Record every subprocess.run call export.py makes."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    return calls


def test_every_subprocess_call_site_in_export_is_time_bounded():
    """Structural: no `subprocess.run` in export.py may omit `timeout=`.

    Written as an AST audit rather than one test per function because the
    defect is a *class* -- a new call site added later is unbounded by default,
    and the failure mode is silence, not an error. This is the assertion that
    catches the next one.
    """
    import ast

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "export.py")
    tree = ast.parse(open(src).read())
    unbounded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "run"
                and isinstance(f.value, ast.Name) and f.value.id == "subprocess"):
            continue
        if not any(kw.arg == "timeout" for kw in node.keywords):
            unbounded.append(node.lineno)

    assert not unbounded, (
        f"export.py:{unbounded} call subprocess.run with no timeout. An "
        f"unattended export that hangs there never returns and never raises; "
        f"the box bills $10.78/day with the GPU idle until a human notices.")


@pytest.mark.parametrize("call,bound,measured", [
    ("convert", export_module.CONVERT_TIMEOUT_S, 3.29),
    ("quantize", export_module.QUANTIZE_TIMEOUT_S, 0.72),
    ("perplexity", export_module.PPL_TIMEOUT_S, 50.0),
])
def test_the_bounds_are_wide_multiples_of_the_measured_durations(call, bound,
                                                                 measured):
    """A bound that can fail a healthy run is worse than none at all.

    `measured` is the slower of the two real `abl-arch` arms on this box
    (llama-perplexity is the upper bound implied by arm 2's 114.5 s tail).
    20x is the floor: it absorbs a fully loaded box, a cold page cache and a
    larger perplexity corpus, and still turns a wedge into a traceback the
    same day.
    """
    assert bound >= 20 * measured, (
        f"{call} bound {bound}s is under 20x its measured {measured}s")


def test_convert_and_quantize_pass_their_bounds_through(monkeypatch, tmp_path):
    calls = _capture_run(monkeypatch)
    convert_to_gguf(str(tmp_path / "hf"), str(tmp_path / "m.gguf"), "/fake")
    quantize_gguf(str(tmp_path / "m.gguf"), str(tmp_path / "q.gguf"), "/fake")

    assert calls[0]["kwargs"]["timeout"] == export_module.CONVERT_TIMEOUT_S
    assert calls[1]["kwargs"]["timeout"] == export_module.QUANTIZE_TIMEOUT_S


def test_measure_perplexity_passes_its_bound_through(monkeypatch, tmp_path):
    calls = _capture_run(monkeypatch, stdout="Final estimate: PPL = 7.6266\n")
    ppl = measure_perplexity("m.gguf", "/fake", str(tmp_path / "t.txt"))

    assert ppl == 7.6266
    assert calls[0]["kwargs"]["timeout"] == export_module.PPL_TIMEOUT_S


def test_the_build_path_is_bounded_too(monkeypatch, tmp_path):
    """Skipped on this box (the artifacts exist), so it is never exercised --
    which is exactly why it needs an assertion rather than a look."""
    calls = _capture_run(monkeypatch)
    setup_llama_cpp(str(tmp_path))

    assert len(calls) == 3, "clone + configure + build"
    for c in calls:
        assert c["kwargs"].get("timeout"), f"unbounded: {c['cmd'][0]}"


@pytest.mark.parametrize("fn,args", [
    (convert_to_gguf, ("hf", "out.gguf", "/fake")),
    (quantize_gguf, ("in.gguf", "out.gguf", "/fake")),
    (measure_perplexity, ("m.gguf", "/fake", "text.txt")),
])
def test_a_wedged_step_raises_instead_of_hanging(monkeypatch, fn, args, tmp_path):
    """The chain gates on `quantization_check.json` and logs the exit code.

    So a timeout must propagate: a non-zero rc plus a missing sentinel is a
    diagnosable failure, whereas swallowing it would ship an export with a
    silently missing artifact -- the shape of five of the six chain bugs found
    on 2026-08-11.
    """
    monkeypatch.chdir(tmp_path)
    _capture_run(monkeypatch, raise_timeout=True)
    with pytest.raises(subprocess.TimeoutExpired):
        fn(*args)


# ------------------------------- the dtype the artifact declares vs holds ---

def test_the_written_config_declares_the_dtype_the_weights_are_in(tmp_path):
    """A published model whose config.json disagrees with its own safetensors
    makes `from_pretrained` cast against the wrong hint.

    Checked against the **written files**, not against either the config object
    or the source, because both mislead: `to_qwen3_config` carries a literal
    that `save_pretrained` then overwrites, and `to_hf_config` sets none at all
    while transformers 5.14.1 still writes the real dtype under `dtype`. The
    only reliable statement is about the artifact on disk.

    This also pins fp16 as the exported dtype, which is load-bearing for QAT --
    see `export_hf_model`'s docstring and `runs/preflight/qat-survives-export.md`.
    """
    import json

    import safetensors.torch as st

    ckpt = _tiny_checkpoint(tmp_path)
    out = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out)

    cfg = json.load(open(os.path.join(out, "config.json")))
    declared = cfg.get("dtype") or cfg.get("torch_dtype")
    tensors = st.load_file(os.path.join(out, "model.safetensors"))
    actual = str(next(iter(tensors.values())).dtype).replace("torch.", "")

    assert actual == "float16", (
        f"weights written as {actual}; fp16 is what keeps QAT's weights on the "
        f"Q4_0 grid through export")
    assert declared == actual, (
        f"config.json declares {declared!r} beside {actual!r} weights")


# ------------------------------------------------ explicit tokenizer path ----

def test_export_tokenizer_still_defaults_to_smollm2():
    """Phase 4 added an override. Every shipped export must be unaffected by
    it, because SmolLM2's pre-tokenizer hash is the one llama.cpp's converter
    recognises and any other vocabulary makes conversion raise."""
    import inspect

    signature = inspect.signature(export_module.export_tokenizer)
    assert signature.parameters["tokenizer"].default is None
    assert signature.parameters["expected_vocab_size"].default is None
    assert "tokenizer or SMOLLM2_TOKENIZER" in inspect.getsource(
        export_module.export_tokenizer)


def test_export_tokenizer_refuses_a_vocabulary_the_model_cannot_index(tmp_path):
    """A 49,152-token tokenizer beside a 32,768-row embedding produces a
    directory that converts, quantizes and loads -- and decodes the wrong token
    for every id. The guard is the only thing between the override and that."""
    with pytest.raises(ValueError, match="rows"):
        export_module.export_tokenizer(str(tmp_path / "hf"),
                                       expected_vocab_size=32768)


def test_config_json_never_carries_the_tokenizer_field(tmp_path):
    """`tokenizer` is a training-only knob. llama.cpp's converter reads
    config.json by name and fingerprints the tokenizer from the files beside
    it; an extra key there is a gratuitous difference from every `Lfm2Config`
    the converter has seen."""
    import json

    from daedalus.config import PRESETS, tokenizer_probe_preset_name

    probe = PRESETS[tokenizer_probe_preset_name(32768)]
    assert "tokenizer" in vars(probe)
    assert "tokenizer" not in probe.to_hf_dict()

    ckpt = _tiny_checkpoint(tmp_path)
    out = str(tmp_path / "hf")
    export_hf_model(ckpt, "tiny", out)
    assert "tokenizer" not in json.load(open(os.path.join(out, "config.json")))


def test_the_model_card_names_a_non_default_vocabulary(tmp_path):
    """A card claiming a byte-identical SmolLM2 tokenizer beside a 32,768-row
    embedding would be false, and the card travels with the weights."""
    from daedalus.config import PRESETS, tokenizer_probe_preset_name

    name = tokenizer_probe_preset_name(32768)
    PRESETS[name].tokenizer = "data/tokenizer-lab/tokenizers/v32768"
    try:
        out = tmp_path / "card"
        out.mkdir()
        export_module.write_model_card(str(out), name)
        card = (out / "README.md").read_text()
    finally:
        PRESETS[name].tokenizer = None
    assert "v32768" in card
    assert "reused byte-identical" not in card.split("| tokenizer |")[1].split("\n")[0]
