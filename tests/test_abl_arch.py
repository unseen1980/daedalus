"""Tests for abl_arch.py: the daedalus-150m vs. dense-150m driver (AGENT.md
SS4 `abl-arch`).

Training/eval/export are mocked out at the run_training/eval_val_bpb/
export_and_bench boundary -- these tests exercise the orchestration (mixture
holdout-split wiring, per-config reporting, partial-failure handling), not
train.py/eval.py/export.py themselves (already covered by their own test
files).
"""
import json
import os
import subprocess

import pytest

import abl_arch
from daedalus.data import ShardWriter, DEFAULT_EOS_ID


@pytest.fixture(autouse=True)
def _no_real_watchdog(monkeypatch):
    """Every arm now starts a watchdog subprocess. Left real, these tests would
    each spawn a `python watchdog.py` into a tmp cwd where that file does not
    exist -- a process per test, doing nothing, purely to die. Tests that care
    about the watchdog install their own recorder over these.
    """
    monkeypatch.setattr(abl_arch, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(abl_arch, "stop_watchdog", lambda *a, **k: None)


def _write_mixture(root, sources):
    """sources: {name: n_tokens}. Writes a two-shard-minimum source per name
    so make_mixture_holdout_split can actually split each one."""
    for name, n_tokens in sources.items():
        w = ShardWriter(str(root / name), shard_tokens=n_tokens // 2)
        w.write(list(range(n_tokens)))
        w.close()
        w.write_manifest({"eos_id": DEFAULT_EOS_ID})
    return str(root)


def test_run_abl_arch_trains_and_reports_both_configs(tmp_path, monkeypatch):
    mixture_dir = _write_mixture(tmp_path / "mixture", {"src-a": 500, "src-b": 400})

    def fake_run_training(data_dir, config, total_tokens, run_name, wandb_enabled,
                          extra_train_args=None, **kwargs):
        assert os.path.isdir(data_dir)
        return f"runs/{run_name}/checkpoint.pt"

    def fake_eval_val_bpb(ckpt_path, holdout_root, config, device, **kwargs):
        assert os.path.isdir(holdout_root)
        return {"val_bpb": 1.5 if "dense" in config else 1.7, "per_source_val_bpb": {}}

    def fake_export_and_bench(ckpt_path, config, out_dir, llama_cpp_dir, **kwargs):
        return {"decode_speed": {"tok_per_sec": 200.0 if "dense" not in config else 150.0}}

    monkeypatch.setattr(abl_arch, "run_training", fake_run_training)
    monkeypatch.setattr(abl_arch, "eval_val_bpb", fake_eval_val_bpb)
    monkeypatch.setattr(abl_arch, "export_and_bench", fake_export_and_bench)

    out_path = str(tmp_path / "results.json")
    result = abl_arch.run_abl_arch(
        mixture_dir, ["daedalus-150m", "dense-150m"], total_tokens=1000,
        holdout_frac=0.2, split_root=str(tmp_path / "split"), out_path=out_path,
        llama_cpp_dir="/fake/llama.cpp", wandb_enabled=False)

    assert set(result["runs"]) == {"daedalus-150m", "dense-150m"}
    assert result["runs"]["daedalus-150m"]["val_bpb"] == 1.7
    assert result["runs"]["dense-150m"]["val_bpb"] == 1.5
    assert result["runs"]["daedalus-150m"]["export"]["decode_speed"]["tok_per_sec"] == 200.0
    with open(out_path) as f:
        assert json.load(f)["runs"]["dense-150m"]["val_bpb"] == 1.5


def test_run_abl_arch_survives_one_config_failure(tmp_path, monkeypatch):
    mixture_dir = _write_mixture(tmp_path / "mixture", {"src-a": 500, "src-b": 400})

    def fake_run_training(data_dir, config, total_tokens, run_name, wandb_enabled,
                          extra_train_args=None, **kwargs):
        if config == "dense-150m":
            raise subprocess.CalledProcessError(1, ["train.py"])
        return f"runs/{run_name}/checkpoint.pt"

    monkeypatch.setattr(abl_arch, "run_training", fake_run_training)
    monkeypatch.setattr(abl_arch, "eval_val_bpb",
                        lambda *a, **k: {"val_bpb": 1.5, "per_source_val_bpb": {}})
    monkeypatch.setattr(abl_arch, "export_and_bench", lambda *a, **k: {})

    result = abl_arch.run_abl_arch(
        mixture_dir, ["daedalus-150m", "dense-150m"], total_tokens=1000,
        holdout_frac=0.2, split_root=str(tmp_path / "split"),
        out_path=str(tmp_path / "results.json"), llama_cpp_dir="/fake/llama.cpp",
        wandb_enabled=False)

    assert "error" in result["runs"]["dense-150m"]
    assert result["runs"]["daedalus-150m"]["val_bpb"] == 1.5


def test_run_abl_arch_survives_one_eval_failure(tmp_path, monkeypatch):
    """A config that trains fine but whose eval_val_bpb raises (e.g. a CUDA
    error, not a subprocess failure) must not crash the whole driver -- the
    other config's results still get reported and results.json still gets
    written."""
    mixture_dir = _write_mixture(tmp_path / "mixture", {"src-a": 500, "src-b": 400})

    def fake_run_training(data_dir, config, total_tokens, run_name, wandb_enabled,
                          extra_train_args=None, **kwargs):
        return f"runs/{run_name}/checkpoint.pt"

    def fake_eval_val_bpb(ckpt_path, holdout_root, config, device, **kwargs):
        if config == "dense-150m":
            raise RuntimeError("CUDA out of memory")
        return {"val_bpb": 1.7, "per_source_val_bpb": {}}

    monkeypatch.setattr(abl_arch, "run_training", fake_run_training)
    monkeypatch.setattr(abl_arch, "eval_val_bpb", fake_eval_val_bpb)
    monkeypatch.setattr(abl_arch, "export_and_bench", lambda *a, **k: {})

    out_path = str(tmp_path / "results.json")
    result = abl_arch.run_abl_arch(
        mixture_dir, ["daedalus-150m", "dense-150m"], total_tokens=1000,
        holdout_frac=0.2, split_root=str(tmp_path / "split"), out_path=out_path,
        llama_cpp_dir="/fake/llama.cpp", wandb_enabled=False)

    assert "error" in result["runs"]["dense-150m"]
    assert "CUDA out of memory" in result["runs"]["dense-150m"]["error"]
    assert result["runs"]["daedalus-150m"]["val_bpb"] == 1.7
    with open(out_path) as f:
        assert json.load(f)["runs"]["daedalus-150m"]["val_bpb"] == 1.7


def test_run_training_builds_expected_train_cmd(monkeypatch):
    captured = {}

    def fake_subprocess_run(cmd, check):
        captured["cmd"] = cmd
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    ckpt = abl_arch.run_training("data/train-split", "dense-150m", 5_000_000_000,
                                 "abl-arch-dense-150m", wandb_enabled=False)

    assert ckpt == os.path.join("runs", "abl-arch-dense-150m", "checkpoint.pt")
    cmd = captured["cmd"]
    assert "--data-dir" in cmd and cmd[cmd.index("--data-dir") + 1] == "data/train-split"
    assert "--config" in cmd and cmd[cmd.index("--config") + 1] == "dense-150m"
    assert "--total-tokens" in cmd and cmd[cmd.index("--total-tokens") + 1] == "5000000000"
    assert "--no-wandb" in cmd
    assert "--tags" in cmd and cmd[cmd.index("--tags") + 1] == "abl-arch"
    assert "--muon-lr" not in cmd  # both configs use the same default lr


def test_run_training_omits_no_wandb_when_enabled(monkeypatch):
    captured = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: captured.setdefault("cmd", cmd))
    abl_arch.run_training("data/x", "tiny", 1000, "r", wandb_enabled=True)
    assert "--no-wandb" not in captured["cmd"]


def test_eval_val_bpb_token_weights_across_sources(tmp_path, monkeypatch):
    holdout_root = tmp_path / "holdout"
    for name, n_tokens in {"source-a": 100, "source-b": 300}.items():
        d = holdout_root / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"total_tokens": n_tokens}))

    bpb_by_source = {"source-a": 2.0, "source-b": 4.0}

    def fake_evaluate_bpb(model, shard_dir, seq_len, tokenizer, device, batch_size,
                          max_batches):
        assert max_batches is None  # final report: full holdout pass
        return bpb_by_source[os.path.basename(shard_dir)]

    monkeypatch.setattr("eval.evaluate_bpb", fake_evaluate_bpb)
    monkeypatch.setattr("daedalus.data.get_tokenizer", lambda: "fake-tok")
    monkeypatch.setattr("train.load_checkpoint", lambda *a, **k: None)

    result = abl_arch.eval_val_bpb("ckpt.pt", str(holdout_root), "tiny", "cpu")

    # No training weights given -> fall back to holdout tokens.
    # weighted mean: (2.0*100 + 4.0*300) / 400 = 3.5
    assert result["val_bpb"] == pytest.approx(3.5)
    per_source = result["per_source_val_bpb"]
    assert per_source["source-a"]["val_bpb"] == 2.0
    assert per_source["source-a"]["tokens"] == 100
    assert per_source["source-b"]["val_bpb"] == 4.0
    assert per_source["source-b"]["tokens"] == 300


def test_eval_val_bpb_weights_by_the_training_mixture_not_holdout_size(
        tmp_path, monkeypatch):
    """The headline number must describe the blend the arms trained on.

    `make_holdout_split` reserves whole shard *files*, so a source's holdout
    share is set by the arbitrary size of its trailing partial shard. Measured
    on the real 9-source corpus that weighted stack-edu-python at 1.71x its
    training share and fineweb-edu at 0.65x. Here source-b holds 75% of the
    holdout tokens but is only 10% of the training mixture."""
    holdout_root = tmp_path / "holdout"
    for name, n_tokens in {"source-a": 100, "source-b": 300}.items():
        d = holdout_root / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"total_tokens": n_tokens}))

    bpb_by_source = {"source-a": 2.0, "source-b": 4.0}
    monkeypatch.setattr(
        "eval.evaluate_bpb",
        lambda model, shard_dir, *a, **k: bpb_by_source[os.path.basename(shard_dir)])
    monkeypatch.setattr("daedalus.data.get_tokenizer", lambda: "fake-tok")
    monkeypatch.setattr("train.load_checkpoint", lambda *a, **k: None)

    result = abl_arch.eval_val_bpb("ckpt.pt", str(holdout_root), "tiny", "cpu",
                                   weights={"source-a": 0.9, "source-b": 0.1})

    # 2.0*0.9 + 4.0*0.1 = 2.2, not the holdout-weighted 3.5.
    assert result["val_bpb"] == pytest.approx(2.2)
    assert result["per_source_val_bpb"]["source-a"]["weight"] == pytest.approx(0.9)
    assert result["per_source_val_bpb"]["source-b"]["weight"] == pytest.approx(0.1)


def test_mixture_sampling_weights_mirrors_the_sampler(tmp_path, monkeypatch):
    """Weights must match what MixtureBatchSource will actually draw with:
    blueprint shares, present-only, renormalized, epoch-capped."""
    import train as train_mod
    from daedalus.dataprep import MIXTURE

    shares = {s.key: s.share for s in MIXTURE}
    keys = [s.key for s in MIXTURE][:3]
    big, small = keys[0], keys[1]              # third source absent on disk

    train_root = tmp_path / "train"
    on_disk = {big: 400_000_000, small: 50_000_000}
    for name, n in on_disk.items():
        d = train_root / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"total_tokens": n}))

    # Uncapped: blueprint shares renormalized over the two present sources.
    w = abl_arch.mixture_sampling_weights(str(train_root), total_run_tokens=None)
    assert set(w) == {big, small}
    assert sum(w.values()) == pytest.approx(1.0)
    assert w[big] == pytest.approx(shares[big] / (shares[big] + shares[small]))

    # Capped: over a 1B-token run, `small` alone would exceed 4 epochs, so it
    # is clamped and the freed mass water-fills to `big`.
    run_tokens = 1_000_000_000
    capped = abl_arch.mixture_sampling_weights(str(train_root), run_tokens)
    assert sum(capped.values()) == pytest.approx(1.0)
    assert w[small] * run_tokens > 4.0 * on_disk[small]          # would have
    assert capped[small] * run_tokens <= 4.0 * on_disk[small] + 1  # now does not
    assert capped[big] > w[big]


def test_eval_val_bpb_raises_when_holdout_root_empty(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr("daedalus.data.get_tokenizer", lambda: "fake-tok")
    monkeypatch.setattr("train.load_checkpoint", lambda *a, **k: None)
    with pytest.raises(ValueError):
        abl_arch.eval_val_bpb("ckpt.pt", str(empty), "tiny", "cpu")


# NOTE: two tests here previously asserted the opposite contract --
# `test_export_and_bench_skips_gguf_without_ppl_text_file` required that no
# GGUF and no decode speed be produced without `--ppl-text-file`. That encoded
# the bug as intended behaviour: `--ppl-text-file` defaults to None, so a
# default abl-arch run would train both models for ~$11 and report no decode
# numbers, which is the measured-CPU-decode half of the Pareto claim this
# experiment exists to produce. They are replaced by the two tests at the end
# of this file, which pin the new contract: GGUF and decode speed always,
# perplexity only when a text file is supplied.


# ------------------------------------------------- export_and_bench wiring ---

def _patch_export(monkeypatch, calls):
    """Stub out export.py at its module boundary so export_and_bench's own
    control flow is what gets tested, not llama.cpp."""
    import export

    monkeypatch.setattr(export, "export_hf_model",
                        lambda ckpt, cfg, out, **k: calls.append("hf") or out)
    monkeypatch.setattr(export, "export_tokenizer", lambda out: out)
    monkeypatch.setattr(export, "convert_to_gguf",
                        lambda hf, out, d, **k: calls.append("convert"))
    monkeypatch.setattr(export, "quantize_gguf",
                        lambda i, o, d, **k: calls.append("quantize"))
    monkeypatch.setattr(export, "measure_perplexity",
                        lambda g, d, t, n: calls.append("ppl") or 10.0)
    monkeypatch.setattr(export, "measure_decode_speed",
                        lambda g, d, n_gen=128, **k: calls.append("bench") or
                        {"tok_per_sec": 756.0, "n_gen": n_gen})


def test_export_and_bench_measures_decode_speed_without_a_ppl_text_file(tmp_path, monkeypatch):
    """The regression that would have cost ~$11 of training and produced no
    headline number: decode speed used to live inside `if ppl_text_file:`,
    which defaults to None, so a default abl-arch run reported no decode
    numbers at all -- silently, with no error to notice."""
    calls = []
    _patch_export(monkeypatch, calls)
    result = abl_arch.export_and_bench("ckpt.pt", "daedalus-150m", str(tmp_path),
                                       "/fake/llama.cpp", ppl_text_file=None)
    assert result["decode_speed"]["tok_per_sec"] == 756.0
    assert "bench" in calls
    assert "convert" in calls and "quantize" in calls, "GGUF must still be built"
    assert "ppl" not in calls, "perplexity needs a text file; must be skipped"
    assert "fp16_ppl" not in result


def test_export_and_bench_adds_the_perplexity_check_when_given_a_text_file(tmp_path, monkeypatch):
    calls = []
    _patch_export(monkeypatch, calls)
    result = abl_arch.export_and_bench("ckpt.pt", "daedalus-150m", str(tmp_path),
                                       "/fake/llama.cpp",
                                       ppl_text_file=str(tmp_path / "ppl.txt"))
    assert calls.count("ppl") == 2, "fp16 and Q4_0 must both be measured"
    assert result["fp16_ppl"] == 10.0 and result["q4_0_ppl"] == 10.0
    assert result["delta_pct"] == 0.0 and result["passes_threshold"] is True
    assert result["decode_speed"]["tok_per_sec"] == 756.0


# --- muon_lr provenance (sweep -> abl-arch) --------------------------------
#
# `_cli()` accepted no lr and never populated run_abl_arch's extra_train_args,
# so both arms trained at train.py's default 0.02 regardless of what `sweep`
# had just concluded. The first sweep's winner was 0.01, so the two disagreed
# in practice. Nothing raised; results.json simply did not record which lr
# produced the headline comparison.

def test_resolve_muon_lr_defaults_to_train_py(tmp_path):
    extra, prov = abl_arch.resolve_muon_lr(None, None)
    assert extra == []
    assert prov["muon_lr"] is None
    # Must NOT re-encode 0.02 here -- two defaults that can drift apart is the
    # bug this helper exists to prevent.
    assert "default" in prov["source"]


def test_resolve_muon_lr_explicit_value():
    extra, prov = abl_arch.resolve_muon_lr(0.015, None)
    assert extra == ["--muon-lr", "0.015"]
    assert prov == {"muon_lr": 0.015, "source": "--muon-lr"}


def test_resolve_muon_lr_reads_sweep_winner(tmp_path):
    best = tmp_path / "best-wsdfix.json"
    best.write_text(json.dumps({"best_lr": 0.01, "best_val_bpb": 1.117,
                                "git_commit": "deadbee"}))
    extra, prov = abl_arch.resolve_muon_lr(None, str(best))
    assert extra == ["--muon-lr", "0.01"]
    assert prov["muon_lr"] == 0.01
    assert prov["source"] == str(best)
    # Provenance ties the ablation back to the sweep that chose its lr.
    assert prov["sweep_git_commit"] == "deadbee"
    assert prov["sweep_best_val_bpb"] == 1.117


def _best_json(tmp_path, probes, best_lr):
    best = tmp_path / "best-wsdfix.json"
    best.write_text(json.dumps({
        "best_lr": best_lr, "git_commit": "deadbee",
        "best_val_bpb": min(p["val_bpb"] for p in probes if p.get("val_bpb")),
        "probes": probes}))
    return str(best)


def test_a_sweep_tie_keeps_the_blueprints_lr_not_the_noise_winner(tmp_path):
    """The likeliest outcome of the re-sweep, and it had no branch. The two
    finished probes differ by ~0.08% val_bpb -- an order of magnitude inside
    the 0.5% noise floor -- and `check_sweep.py` only *warns* about that and
    exits 0, so the chain would hand a noise-chosen lr to a ~$11 ablation and
    then to a ~$44 hero. DAEDALUS-BLUEPRINT-v6.md locks Muon lr 0.02; a tie is
    not evidence for deviating from a locked decision."""
    best = _best_json(tmp_path, [{"muon_lr": 0.01, "val_bpb": 1.09178},
                                 {"muon_lr": 0.02, "val_bpb": 1.09265},
                                 {"muon_lr": 0.04, "val_bpb": 1.09300}], 0.01)
    extra, prov = abl_arch.resolve_muon_lr(None, best)

    # No --muon-lr at all: train.py's own default governs, so 0.02 is defined
    # in exactly one place.
    assert extra == []
    assert prov["muon_lr"] is None
    assert prov["sweep_best_lr"] == 0.01          # what the sweep said...
    assert prov["sweep_winner_margin"] < abl_arch.NOISE_FRAC   # ...and why not used
    assert "did not discriminate" in prov["why"]
    # The blueprint winning must not erase which sweep it beat.
    assert prov["sweep_file"] == best
    assert prov["sweep_git_commit"] == "deadbee"


def test_a_discriminating_sweep_still_overrides_the_blueprint(tmp_path):
    """The other half of the rule, and the reason the sweep is run at all: when
    the grid does measure something, the measurement wins."""
    best = _best_json(tmp_path, [{"muon_lr": 0.01, "val_bpb": 1.05},
                                 {"muon_lr": 0.02, "val_bpb": 1.12},
                                 {"muon_lr": 0.04, "val_bpb": 1.30}], 0.01)
    extra, prov = abl_arch.resolve_muon_lr(None, best)

    assert extra == ["--muon-lr", "0.01"]
    assert prov["muon_lr"] == 0.01
    assert prov["sweep_winner_margin"] > abl_arch.NOISE_FRAC


def test_a_bad_third_probe_does_not_unlock_a_tie_between_the_other_two(tmp_path):
    """Why the decision is the winner-vs-runner-up margin and not the
    full-grid spread `check_sweep.py` reports. Tonight's real grid: 0.01 and
    0.02 landed 0.43% apart -- a tie -- so if the 0.04 probe comes in 2% worse
    the *range* looks decisive while the choice actually being made, 0.02 over
    0.01, is still noise. Using the range would have let a bad third probe
    unlock a deviation from a locked blueprint decision that nothing
    measured."""
    best = _best_json(tmp_path, [{"muon_lr": 0.01, "val_bpb": 1.09178},
                                 {"muon_lr": 0.02, "val_bpb": 1.087067},
                                 {"muon_lr": 0.04, "val_bpb": 1.11000}], 0.02)
    spread = (1.11000 - 1.087067) / 1.087067
    assert spread > abl_arch.NOISE_FRAC, "the range alone would look decisive"

    extra, prov = abl_arch.resolve_muon_lr(None, best)
    assert extra == []
    assert prov["sweep_winner_margin"] < abl_arch.NOISE_FRAC


def test_blueprint_default_lr_is_train_py_default():
    """The tie path returns no --muon-lr and relies on train.py carrying the
    blueprint's 0.02. If that default is ever changed, a sweep tie would
    silently train at the new value instead."""
    import train
    assert train.TrainArgs(run_name="x").muon_lr == 0.02
    # ...and that the CLI default agrees with it, since that is the one the
    # arms actually inherit when no --muon-lr is passed.
    assert train.parse_args(["--run-name", "x"]).muon_lr == 0.02



def test_a_malformed_probes_list_falls_back_to_the_winner(tmp_path):
    """This runs at the launch of a ~24 h unattended job. A best.json from an
    older sweep (no 'probes'), or one with junk in it, must fall through to the
    previous behaviour rather than raise on the launch path."""
    for payload in ({"best_lr": 0.04},                       # pre-'probes' file
                    {"best_lr": 0.04, "probes": "nonsense"},
                    {"best_lr": 0.04, "probes": [{"muon_lr": 0.04,
                                                  "val_bpb": None}]},
                    {"best_lr": 0.04, "probes": [{"muon_lr": 0.04,
                                                  "val_bpb": 0.0}]}):
        best = tmp_path / "b.json"
        best.write_text(json.dumps(payload))
        extra, prov = abl_arch.resolve_muon_lr(None, str(best))
        assert extra == ["--muon-lr", "0.04"], payload
        assert prov["muon_lr"] == 0.04, payload


def test_an_explicit_muon_lr_still_beats_the_tie_rule(tmp_path):
    """--muon-lr is the operator overriding both the sweep and the blueprint;
    the tie rule must not quietly reinstate 0.02 underneath it."""
    extra, prov = abl_arch.resolve_muon_lr(0.04, None)
    assert extra == ["--muon-lr", "0.04"]
    assert prov == {"muon_lr": 0.04, "source": "--muon-lr"}


def test_resolve_muon_lr_rejects_both_sources(tmp_path):
    best = tmp_path / "best.json"
    best.write_text(json.dumps({"best_lr": 0.01}))
    with pytest.raises(ValueError, match="not both"):
        abl_arch.resolve_muon_lr(0.02, str(best))


def test_resolve_muon_lr_rejects_sweep_with_no_winner(tmp_path):
    """Every probe failing leaves a best.json with a 'probes' list and no
    'best_lr'. Training $11 of ablation against a silently-absent winner is
    worse than refusing to start."""
    best = tmp_path / "best.json"
    best.write_text(json.dumps({"probes": [{"muon_lr": 0.01, "error": "boom"}]}))
    with pytest.raises(ValueError, match="no 'best_lr'"):
        abl_arch.resolve_muon_lr(None, str(best))


def test_run_abl_arch_passes_lr_to_both_arms_and_records_it(tmp_path, monkeypatch):
    """The end-to-end property that matters: whatever lr is resolved reaches
    *every* arm's train.py argv, and lands in results.json."""
    mixture_dir = _write_mixture(tmp_path / "mixture", {"src-a": 500})
    seen = {}

    def fake_run_training(data_dir, config, total_tokens, run_name, wandb_enabled,
                          extra_train_args=None, **kwargs):
        seen[config] = extra_train_args
        return f"runs/{run_name}/checkpoint.pt"

    monkeypatch.setattr(abl_arch, "run_training", fake_run_training)
    monkeypatch.setattr(abl_arch, "eval_val_bpb",
                        lambda *a, **k: {"val_bpb": 1.0, "per_source_val_bpb": {}})
    monkeypatch.setattr(abl_arch, "export_and_bench", lambda *a, **k: {})

    out = tmp_path / "results.json"
    extra, prov = abl_arch.resolve_muon_lr(0.01, None)
    result = abl_arch.run_abl_arch(
        mixture_dir, ["daedalus-150m", "dense-150m"], 1000, 0.2,
        str(tmp_path / "split"), str(out), "vendor/llama.cpp",
        wandb_enabled=False, extra_train_args=extra, lr_provenance=prov)

    assert seen["daedalus-150m"][:2] == ["--muon-lr", "0.01"]
    # Same lr for both twins, or it stops being an architecture comparison.
    assert seen["dense-150m"] == seen["daedalus-150m"]
    assert result["lr"] == {"muon_lr": 0.01, "source": "--muon-lr"}
    assert json.loads(out.read_text())["lr"]["muon_lr"] == 0.01


def _argv_of_each_arm(tmp_path, monkeypatch, extra_train_args=None):
    """Run run_abl_arch with everything below train.py faked out, and return
    {config: argv-tail} as it would have reached each arm's train.py."""
    mixture_dir = _write_mixture(tmp_path / "mixture", {"src-a": 500, "src-b": 500})
    seen = {}

    def fake_run_training(data_dir, config, total_tokens, run_name, wandb_enabled,
                          extra_train_args=None, **kwargs):
        seen[config] = (list(extra_train_args or []), data_dir)
        return f"runs/{run_name}/checkpoint.pt"

    monkeypatch.setattr(abl_arch, "run_training", fake_run_training)
    monkeypatch.setattr(abl_arch, "eval_val_bpb",
                        lambda *a, **k: {"val_bpb": 1.0, "per_source_val_bpb": {}})
    monkeypatch.setattr(abl_arch, "export_and_bench", lambda *a, **k: {})

    abl_arch.run_abl_arch(
        mixture_dir, ["daedalus-150m", "dense-150m"], 1000, 0.2,
        str(tmp_path / "split"), str(tmp_path / "results.json"),
        "vendor/llama.cpp", wandb_enabled=False,
        extra_train_args=extra_train_args)
    return seen


def test_both_arms_get_the_holdout_as_val_dir(tmp_path, monkeypatch):
    """abl-arch decides on val_bpb, and until this landed neither arm computed
    one until its ~12 h of training were already paid for: `run_abl_arch`
    carved a holdout, used it for the *final* eval, and never passed it to
    train.py. `hero.py` has passed --val-dir since 298c059, so this is also the
    first GPU exercise of a path a four-day run depends on."""
    seen = _argv_of_each_arm(tmp_path, monkeypatch)

    for config, (argv, _) in seen.items():
        assert "--val-dir" in argv, config
    # Identical for both twins, like every other train flag here.
    assert seen["dense-150m"][0] == seen["daedalus-150m"][0]


def test_val_dir_is_the_holdout_and_not_the_training_root(tmp_path, monkeypatch):
    """The silent failure this guards: validating on the data being trained on
    reports a flattering, monotonic curve that is not a generalization signal
    at all -- and `Trainer._val_bpb` swallows exceptions, so nothing would say
    so. The directory must also exist by the time an arm starts, since
    train.py's only response to a bad holdout is `val_bpb: null`."""
    seen = _argv_of_each_arm(tmp_path, monkeypatch)

    argv, data_dir = seen["daedalus-150m"]
    val_dir = argv[argv.index("--val-dir") + 1]
    assert os.path.isdir(val_dir)
    assert os.path.realpath(val_dir) != os.path.realpath(data_dir)
    assert os.path.basename(val_dir) == "holdout"
    # A mixture root, which is why evaluate_bpb_mixture (not evaluate_bpb) is
    # what train.py calls: no manifest.json at the top, one per source below.
    assert not os.path.exists(os.path.join(val_dir, "manifest.json"))
    assert os.path.exists(os.path.join(val_dir, "src-a", "manifest.json"))


def test_val_dir_does_not_accumulate_across_arms(tmp_path, monkeypatch):
    """`train_args` is built once, before the loop. Appending to the caller's
    list inside it would give arm 2 a second --val-dir (harmless) and mutate
    the caller's list (not), and the same bug shape would double any flag
    added here later."""
    caller_args = ["--muon-lr", "0.02"]
    seen = _argv_of_each_arm(tmp_path, monkeypatch, extra_train_args=caller_args)

    for config, (argv, _) in seen.items():
        assert argv.count("--val-dir") == 1, config
        assert argv[:2] == ["--muon-lr", "0.02"], config
    assert caller_args == ["--muon-lr", "0.02"]


def test_cli_forwards_sweep_winner_to_run_abl_arch(tmp_path, monkeypatch):
    """The actual regression. The bug lived in `_cli()`, not in any helper:
    it parsed no lr and passed no extra_train_args, so run_abl_arch always
    received None. A test that only exercises resolve_muon_lr would have
    passed against the broken code."""
    best = tmp_path / "best-wsdfix.json"
    best.write_text(json.dumps({"best_lr": 0.04, "best_val_bpb": 1.1}))
    captured = {}

    def fake_run_abl_arch(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(abl_arch, "run_abl_arch", fake_run_abl_arch)
    monkeypatch.setattr("sys.argv", ["abl_arch.py", "--best-json", str(best)])
    abl_arch._cli()

    assert captured["extra_train_args"] == ["--muon-lr", "0.04"]
    assert captured["lr_provenance"]["muon_lr"] == 0.04


def test_cli_without_lr_flags_passes_no_lr(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(abl_arch, "run_abl_arch",
                        lambda *a, **k: captured.update(k) or {})
    monkeypatch.setattr("sys.argv", ["abl_arch.py"])
    abl_arch._cli()
    assert captured["extra_train_args"] == []
    assert captured["lr_provenance"]["muon_lr"] is None


def test_cli_micro_batch_reaches_both_arms(tmp_path, monkeypatch):
    """dense-150m needs ~33% more activation memory than the hybrid, so
    abl-arch may have to run at a smaller micro-batch than the hybrid alone
    would need. It must be the SAME for both arms: micro_batch sets `accum`,
    which sets tokens/step, which sets the step count, and 'same steps' is the
    property that makes the head-to-head publishable."""
    best = tmp_path / "best.json"
    best.write_text(json.dumps({"best_lr": 0.01}))
    captured = {}
    monkeypatch.setattr(abl_arch, "run_abl_arch",
                        lambda *a, **k: captured.update(k) or {})
    monkeypatch.setattr("sys.argv", ["abl_arch.py", "--best-json", str(best),
                                     "--micro-batch", "12"])
    abl_arch._cli()
    assert captured["extra_train_args"] == ["--muon-lr", "0.01",
                                            "--micro-batch", "12"]
    assert captured["lr_provenance"]["micro_batch"] == 12


def test_cli_micro_batch_without_lr(monkeypatch):
    captured = {}
    monkeypatch.setattr(abl_arch, "run_abl_arch",
                        lambda *a, **k: captured.update(k) or {})
    monkeypatch.setattr("sys.argv", ["abl_arch.py", "--micro-batch", "8"])
    abl_arch._cli()
    assert captured["extra_train_args"] == ["--micro-batch", "8"]


# --- crash recovery within an arm ------------------------------------------
#
# Each arm is ~12 h and the run is unattended. A transient failure used to
# propagate out of subprocess.run(check=True) into run_abl_arch's handler,
# which recorded the error and moved to the next config -- discarding up to
# 12 h and ~$5.4 for a blip. train.py checkpoints every 30 min.

def test_run_training_retries_with_resume_after_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # real cwd, so the checkpoint path is real
    ckpt = tmp_path / "runs" / "r" / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("weights")

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = {}
    abl_arch.run_training("data/x", "tiny", 1000, "r", wandb_enabled=False,
                          report=report)

    assert len(calls) == 2
    assert "--resume" not in calls[0]
    assert "--resume" in calls[1]
    assert calls[1][calls[1].index("--resume") + 1] == os.path.join(
        "runs", "r", "checkpoint.pt")
    assert report["attempts"] == 2
    assert report["resumed"] is True
    # The deviation must be stated, not swallowed: a resumed arm no longer
    # sees byte-identical data to one that ran straight through.
    assert "byte-identical" in report["identical_data_caveat"]


def test_run_training_does_not_resume_without_a_checkpoint(tmp_path, monkeypatch):
    """A first attempt that dies before the 30-min checkpoint leaves nothing
    to resume from; passing --resume at a path that does not exist would be
    a lie in the recorded provenance even though train.py tolerates it."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = {}
    abl_arch.run_training("data/x", "tiny", 1000, "r", wandb_enabled=False,
                          report=report)

    assert len(calls) == 2
    assert "--resume" not in calls[1]
    assert report["resumed"] is False
    assert "identical_data_caveat" not in report


def test_run_training_gives_up_after_max_attempts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = {}
    with pytest.raises(subprocess.CalledProcessError):
        abl_arch.run_training("data/x", "tiny", 1000, "r", wandb_enabled=False,
                              max_attempts=3, report=report)
    assert len(calls) == 3
    assert report["attempts"] == 3


def test_run_abl_arch_records_a_resumed_arm_in_results(tmp_path, monkeypatch):
    """The caveat has to reach results.json, or the writeup cannot confront
    it."""
    mixture_dir = _write_mixture(tmp_path / "mixture", {"src-a": 500})

    def fake_run_training(data_dir, config, total_tokens, run_name, wandb_enabled,
                          extra_train_args=None, report=None, **kwargs):
        if report is not None and config == "dense-150m":
            report["resumed"] = True
            report["attempts"] = 2
        return f"runs/{run_name}/checkpoint.pt"

    monkeypatch.setattr(abl_arch, "run_training", fake_run_training)
    monkeypatch.setattr(abl_arch, "eval_val_bpb",
                        lambda *a, **k: {"val_bpb": 1.5, "per_source_val_bpb": {}})
    monkeypatch.setattr(abl_arch, "export_and_bench", lambda *a, **k: {})

    out_path = str(tmp_path / "results.json")
    result = abl_arch.run_abl_arch(
        mixture_dir, ["daedalus-150m", "dense-150m"], 1000, 0.2,
        str(tmp_path / "split"), out_path, "/fake/llama.cpp", wandb_enabled=False)

    assert result["runs"]["dense-150m"]["resumed"] is True
    assert json.loads(open(out_path).read())["runs"]["dense-150m"]["attempts"] == 2


# ------------------------------------------------------------- run prefix ---
# run_name was hardcoded to "abl-arch-<config>", so a throwaway run wrote to
# the same run dir as the real one. run_training's retry path resumes from
# runs/<run_name>/checkpoint.pt when it exists, so a leftover smoke checkpoint
# would be silently picked up by a real arm's second attempt -- poisoning an
# ~$11, ~24 h head-to-head with four steps of garbage and reporting only
# `resumed: True`. sweep.py has --run-prefix for exactly this reason.

def test_run_prefix_defaults_to_abl_arch(tmp_path, monkeypatch):
    names = []
    monkeypatch.setattr(abl_arch, "run_training",
                        lambda *a, **k: names.append(a[3]) or "ckpt.pt")
    monkeypatch.setattr(abl_arch, "eval_val_bpb", lambda *a, **k: {"val_bpb": 1.0})
    monkeypatch.setattr(abl_arch, "export_and_bench", lambda *a, **k: {})
    monkeypatch.setattr("daedalus.data.make_mixture_holdout_split",
                        lambda *a, **k: None)
    monkeypatch.setattr(abl_arch, "mixture_sampling_weights", lambda *a, **k: {})

    abl_arch.run_abl_arch(str(tmp_path / "mix"), ["daedalus-150m", "dense-150m"],
                          1000, 0.02, str(tmp_path / "split"),
                          str(tmp_path / "out.json"), "/nonexistent",
                          wandb_enabled=False)
    assert names == ["abl-arch-daedalus-150m", "abl-arch-dense-150m"]


def test_run_prefix_isolates_a_smoke_run(tmp_path, monkeypatch):
    names = []
    monkeypatch.setattr(abl_arch, "run_training",
                        lambda *a, **k: names.append(a[3]) or "ckpt.pt")
    monkeypatch.setattr(abl_arch, "eval_val_bpb", lambda *a, **k: {"val_bpb": 1.0})
    monkeypatch.setattr(abl_arch, "export_and_bench", lambda *a, **k: {})
    monkeypatch.setattr("daedalus.data.make_mixture_holdout_split",
                        lambda *a, **k: None)
    monkeypatch.setattr(abl_arch, "mixture_sampling_weights", lambda *a, **k: {})

    abl_arch.run_abl_arch(str(tmp_path / "mix"), ["daedalus-150m"], 1000, 0.02,
                          str(tmp_path / "split"), str(tmp_path / "out.json"),
                          "/nonexistent", wandb_enabled=False,
                          run_prefix="abl-smoke")
    assert names == ["abl-smoke-daedalus-150m"]
    assert not any(n.startswith("abl-arch-") for n in names), \
        "a smoke run must not write into the real arm's run dir"


def test_cli_forwards_run_prefix(monkeypatch):
    captured = {}
    monkeypatch.setattr(abl_arch, "run_abl_arch",
                        lambda *a, **k: captured.update(k) or {})
    monkeypatch.setattr("sys.argv", ["abl_arch.py", "--run-prefix", "abl-smoke"])
    abl_arch._cli()
    assert captured["run_prefix"] == "abl-smoke"


def test_cli_run_prefix_defaults_so_the_live_chain_is_unaffected(monkeypatch):
    """The chain script is already running and does not pass --run-prefix."""
    captured = {}
    monkeypatch.setattr(abl_arch, "run_abl_arch",
                        lambda *a, **k: captured.update(k) or {})
    monkeypatch.setattr("sys.argv", ["abl_arch.py"])
    abl_arch._cli()
    assert captured["run_prefix"] == "abl-arch"


# ------------------------------------------------------------- exit status ---

def _cli_with(monkeypatch, runs, argv=("abl_arch.py",)):
    monkeypatch.setattr(abl_arch, "run_abl_arch", lambda *a, **k: {"runs": runs})
    monkeypatch.setattr("sys.argv", list(argv))
    return abl_arch._cli


def test_cli_exits_nonzero_when_an_arm_failed(monkeypatch):
    """Exiting 0 with an errored arm makes a failed ~25 h unattended job look
    finished to the chain log and the heartbeat."""
    cli = _cli_with(monkeypatch, {"daedalus-150m": {"val_bpb": 1.0},
                                  "dense-150m": {"error": "boom"}})
    with pytest.raises(SystemExit) as e:
        cli()
    assert e.value.code == 1


def test_cli_exits_nonzero_when_every_arm_failed(monkeypatch):
    cli = _cli_with(monkeypatch, {"daedalus-150m": {"error": "boom"},
                                  "dense-150m": {"error": "boom"}})
    with pytest.raises(SystemExit) as e:
        cli()
    assert e.value.code == 1


def test_cli_exits_zero_when_both_arms_succeeded(monkeypatch):
    cli = _cli_with(monkeypatch, {"daedalus-150m": {"val_bpb": 1.0},
                                  "dense-150m": {"val_bpb": 1.1}})
    cli()      # must not raise SystemExit


# ------------------------------------------------- watchdog on each arm ---
#
# abl-arch is ~24 h and ~$11.4, unattended, and had no divergence or stall
# detection at all: an arm that went to NaN at hour 3 trained on for nine more
# hours and reported a val_bpb that meant nothing, with the head-to-head then
# drawn against it. hero.py has had a watchdog since it was written; this is
# the same wiring on the same shared launcher.

def test_each_arm_is_watched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    started = []
    monkeypatch.setattr(abl_arch, "start_watchdog",
                        lambda *a, **k: started.append((a, k)))
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: None)

    abl_arch.run_training("data/x", "tiny", 5_000_000_000, "abl-arch-tiny",
                          wandb_enabled=False)

    (run_name, run_dir, target, stall), _ = started[0]
    assert run_name == "abl-arch-tiny"
    assert run_dir == os.path.join("runs", "abl-arch-tiny")
    assert target == 5_000_000_000        # so completion is detectable
    assert stall == 30.0


def test_the_watchdog_is_stopped_even_when_the_arm_fails(tmp_path, monkeypatch):
    """Otherwise the second arm trains for 12 h with the first arm's watchdog
    still polling a finished directory."""
    monkeypatch.chdir(tmp_path)
    stopped = []
    monkeypatch.setattr(abl_arch, "start_watchdog", lambda *a, **k: "wd")
    monkeypatch.setattr(abl_arch, "stop_watchdog", stopped.append)

    def always_fails(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", always_fails)
    with pytest.raises(subprocess.CalledProcessError):
        abl_arch.run_training("data/x", "tiny", 1000, "r", wandb_enabled=False,
                              max_attempts=2)
    assert stopped == ["wd"]


def test_a_halted_arm_is_not_resumed(tmp_path, monkeypatch):
    """The bug this closes: the watchdog SIGTERMs a diverged trainer and exits,
    the retry loop reads that as a crash, resumes the diverged checkpoint, and
    runs the remaining hours unwatched to produce a meaningless number."""
    import watchdog as watchdog_mod
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    ckpt = run_dir / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("weights")

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        watchdog_mod.write_halt_marker(str(run_dir), "divergence",
                                       "loss diverged at step 900")
        raise subprocess.CalledProcessError(-15, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = {}
    with pytest.raises(RuntimeError, match="halted by watchdog"):
        abl_arch.run_training("data/x", "tiny", 1000, "r", wandb_enabled=False,
                              max_attempts=3, report=report)

    assert len(calls) == 1, "resumed an arm the watchdog deliberately halted"
    assert report["halt"]["kind"] == "divergence"
    assert report["attempts"] == 1


def test_a_stale_marker_does_not_block_a_new_arm(tmp_path, monkeypatch):
    """A halt from a previous launch must not make every later run refuse to
    retry -- cleared at the start of the arm, never inside the retry loop."""
    import watchdog as watchdog_mod
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_text("weights")
    watchdog_mod.write_halt_marker(str(run_dir), "stall", "yesterday's news")

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    abl_arch.run_training("data/x", "tiny", 1000, "r", wandb_enabled=False)
    assert len(calls) == 2 and "--resume" in calls[1]
