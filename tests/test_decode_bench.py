"""Tests for scripts/decode_bench.py.

`llama-bench` is replaced throughout: these check the parts that decide whether
a *number* is trustworthy -- that rounds really alternate, that the depth and
thread flags reach the binary, and that a failed round is dropped rather than
silently counted as a zero -- not llama.cpp itself.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import decode_bench  # noqa: E402


def _bench_json(ts, n_gen=128):
    return json.dumps([{"n_gen": n_gen, "n_prompt": 0, "avg_ts": ts}])


class _Result:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_parses_the_decode_row(monkeypatch):
    monkeypatch.setattr(decode_bench.subprocess, "run",
                        lambda *a, **k: _Result(_bench_json(912.5)))
    assert decode_bench.run_once("bench", "m.gguf", 8, 128) == 912.5


def test_picks_the_row_matching_n_gen_not_the_first(monkeypatch):
    """With -p 0 there is one row, but selecting positionally would silently
    report prompt-processing speed the day a prefill row comes back."""
    rows = json.dumps([{"n_gen": 0, "n_prompt": 512, "avg_ts": 5000.0},
                       {"n_gen": 128, "n_prompt": 0, "avg_ts": 900.0}])
    monkeypatch.setattr(decode_bench.subprocess, "run",
                        lambda *a, **k: _Result(rows))
    assert decode_bench.run_once("bench", "m.gguf", 8, 128) == 900.0


def test_threads_and_depth_reach_the_binary(monkeypatch):
    """A depth that never arrives would measure every model at depth 0, which
    is exactly where a conv hybrid has least to gain."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _Result(_bench_json(1.0))

    monkeypatch.setattr(decode_bench.subprocess, "run", fake_run)
    decode_bench.run_once("bench", "m.gguf", threads=6, n_gen=64, depth=2048)
    cmd = seen["cmd"]
    assert cmd[cmd.index("-t") + 1] == "6"
    assert cmd[cmd.index("-d") + 1] == "2048"
    assert cmd[cmd.index("-n") + 1] == "64"
    assert cmd[cmd.index("-p") + 1] == "0"


@pytest.mark.parametrize("outcome", [
    _Result("", returncode=1, stderr="boom"),
    _Result("not json"),
    _Result("[]"),
])
def test_a_failed_round_is_none_not_zero(monkeypatch, outcome):
    """Counting a failure as 0 tok/s would drag a mean down and look like a
    slow model rather than a broken run."""
    monkeypatch.setattr(decode_bench.subprocess, "run", lambda *a, **k: outcome)
    assert decode_bench.run_once("bench", "m.gguf", 8, 128) is None


def test_a_timeout_is_none(monkeypatch):
    def boom(*a, **k):
        raise decode_bench.subprocess.TimeoutExpired("bench", 1)
    monkeypatch.setattr(decode_bench.subprocess, "run", boom)
    assert decode_bench.run_once("bench", "m.gguf", 8, 128) is None


def test_rounds_alternate_between_models(monkeypatch, tmp_path):
    """The reason this script exists. A non-alternating comparison on this box
    reported 1.29x where alternating rounds put the truth at 1.15x -- the
    machine's own load drifted underneath the measurement."""
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    a.write_bytes(b"a"), b.write_bytes(b"bb")
    order = []

    def fake_once(bench_bin, model, threads, n_gen, depth=0, **kw):
        order.append(os.path.basename(model))
        return 100.0 if model.endswith("a.gguf") else 50.0

    monkeypatch.setattr(decode_bench, "run_once", fake_once)
    out = decode_bench.bench({"ours": str(a), "peer": str(b)}, threads=8,
                             rounds=3, n_gen=128)

    assert order == ["a.gguf", "b.gguf"] * 3
    assert out["models"]["ours"]["mean"] == 100.0
    assert out["models"]["peer"]["mean"] == 50.0
    assert out["models"]["ours"]["samples"] == [100.0, 100.0, 100.0]
    assert out["models"]["ours"]["file_mb"] is not None


def test_failed_rounds_are_dropped_from_the_mean(monkeypatch, tmp_path):
    m = tmp_path / "m.gguf"
    m.write_bytes(b"x")
    seq = iter([900.0, None, 800.0])
    monkeypatch.setattr(decode_bench, "run_once",
                        lambda *a, **k: next(seq))
    out = decode_bench.bench({"ours": str(m)}, threads=8, rounds=3, n_gen=128)
    assert out["models"]["ours"]["samples"] == [900.0, 800.0]
    assert out["models"]["ours"]["mean"] == 850.0


def test_a_single_sample_reports_no_spread(monkeypatch, tmp_path):
    """statistics.stdev raises on n=1; reporting 0.0 would claim a precision
    the run does not have."""
    m = tmp_path / "m.gguf"
    m.write_bytes(b"x")
    monkeypatch.setattr(decode_bench, "run_once", lambda *a, **k: 900.0)
    out = decode_bench.bench({"ours": str(m)}, threads=8, rounds=1, n_gen=128)
    assert out["models"]["ours"]["mean"] == 900.0
    assert out["models"]["ours"]["stdev"] is None


def test_every_model_failing_leaves_none_rather_than_a_crash(monkeypatch, tmp_path):
    m = tmp_path / "m.gguf"
    m.write_bytes(b"x")
    monkeypatch.setattr(decode_bench, "run_once", lambda *a, **k: None)
    out = decode_bench.bench({"ours": str(m)}, threads=8, rounds=2, n_gen=128)
    assert out["models"]["ours"]["mean"] is None
    assert out["models"]["ours"]["samples"] == []


def test_cli_runs_every_thread_count_and_depth(monkeypatch, tmp_path, capsys):
    a = tmp_path / "a.gguf"
    a.write_bytes(b"a")
    calls = []

    def fake_bench(models, threads, rounds, n_gen, bench_bin, depth=0):
        calls.append((threads, depth))
        return {"threads": threads, "depth": depth, "models": {}}

    monkeypatch.setattr(decode_bench, "bench", fake_bench)
    out = tmp_path / "out.json"
    rc = decode_bench.main(["--models", f"ours={a}", "--threads", "6", "8",
                            "--depths", "0", "2048", "--out", str(out)])
    assert rc == 0
    assert calls == [(6, 0), (6, 2048), (8, 0), (8, 2048)]
    report = json.loads(out.read_text())
    assert len(report["passes"]) == 4
    # The caveat travels with the numbers rather than living in a doc nobody
    # opens next to the JSON.
    assert "ratio" in report["metric"]


def test_cli_rejects_a_missing_model(tmp_path, capsys):
    assert decode_bench.main(["--models", f"ours={tmp_path}/nope.gguf"]) == 2


def test_cli_rejects_a_spec_without_a_name(tmp_path):
    a = tmp_path / "a.gguf"
    a.write_bytes(b"a")
    assert decode_bench.main(["--models", str(a)]) == 2
