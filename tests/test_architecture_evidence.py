"""Tests for scripts/architecture_evidence.py -- the export column's artifacts.

What can go wrong here does not look like a failure. A sweep that raises on the
first shape stock llama.cpp refuses reports one arm instead of fifteen; a skip
keyed on a file rather than on the checkpoint that produced it leaves a retrained
arm wearing the old shape's GGUF; and a cleanup that runs on the failure path
deletes the intermediates somebody needs to read to find out why the conversion
was refused. Each produces an artifact tree that looks finished.

No torch and no llama.cpp: the conversion and quantization are injected, so what
is pinned here is the layout, the skip rule and the failure handling rather than
llama.cpp's behaviour, which is the thing this module exists to *measure* and
must not assume.

Run: python -m pytest tests/test_architecture_evidence.py -v
"""
import json
from pathlib import Path

import pytest

from scripts import architecture_evidence as evidence
from scripts.architecture_evidence import (exported_from, export_arm,
                                           export_arms, gguf_paths,
                                           manifest_path, summarize)
from scripts.architecture_evidence import main as evidence_main
from scripts.architecture_sweep import ARMS_BY_NAME, CONTROL


# --------------------------------------------------------------- fixtures ----

ARM = ARMS_BY_NAME["a2-kv1"]


def _llama_cpp(tmp_path: Path) -> Path:
    """A stock tree with the two entry points this module invokes."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# stock converter\n")
    (root / "build" / "bin" / "llama-quantize").write_text("#!/bin/sh\n")
    return root


def _checkpoint(run_root: Path, arm, tag: str = "stagea",
                payload: bytes = b"weights") -> Path:
    path = run_root / f"arch-{tag}-{arm.name}" / "checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class FakeToolchain:
    """Stands in for `convert_hf_to_gguf.py` and `llama-quantize`.

    Writes the output file its real counterpart would, so the module's "did this
    actually produce a GGUF" checks are exercised rather than bypassed.
    """

    def __init__(self, *, convert_rc: int = 0, quantize_rc: int = 0,
                 convert_writes: bool = True, quantize_writes: bool = True):
        self.convert_rc = convert_rc
        self.quantize_rc = quantize_rc
        self.convert_writes = convert_writes
        self.quantize_writes = quantize_writes
        self.calls = []
        self.exports = []

    def export_fn(self, checkpoint, config, hf_dir):
        self.exports.append((checkpoint, config, hf_dir))
        Path(hf_dir).mkdir(parents=True, exist_ok=True)
        (Path(hf_dir) / "config.json").write_text("{}")

    def runner(self, command, timeout):
        command = [str(part) for part in command]
        self.calls.append(command)
        if command[-1] == "Q4_0":
            if self.quantize_writes and self.quantize_rc == 0:
                Path(command[2]).write_bytes(b"q4" * 8)
            return self.quantize_rc, "quantize output"
        out_index = command.index("--outfile") + 1
        if self.convert_writes and self.convert_rc == 0:
            Path(command[out_index]).write_bytes(b"f16" * 8)
        return self.convert_rc, "line one\nconverter output"


# ------------------------------------------------------- failures are data ----

def test_a_refused_conversion_is_recorded_rather_than_raised(tmp_path):
    """The export column's whole question is whether stock llama.cpp takes this
    shape. A refusal is that answer, so it lands in the record with its return
    code and its output, and the arm is simply not marked converted."""
    _checkpoint(tmp_path / "runs", ARM)
    tools = FakeToolchain(convert_rc=1)

    record = export_arm(ARM, run_root=str(tmp_path / "runs"),
                        root=str(tmp_path / "evidence"),
                        llama_cpp_dir=str(_llama_cpp(tmp_path)),
                        export_fn=tools.export_fn, runner=tools.runner)

    assert record["converted"] is False
    assert record["quantized"] is False
    assert record["convert_returncode"] == 1
    assert record["convert_tail"][-1] == "converter output"
    # Refused at the conversion, so the quantizer was never asked.
    assert [call for call in tools.calls if call[-1] == "Q4_0"] == []


def test_a_refused_arm_does_not_stop_the_sweep(tmp_path):
    """Fourteen answers must not be abandoned to report one refusal."""
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    for arm in (CONTROL, ARM):
        _checkpoint(run_root, arm)

    class OnlyControlConverts(FakeToolchain):
        def runner(self, command, timeout):
            command = [str(part) for part in command]
            self.convert_rc = 0 if CONTROL.name in " ".join(command) else 1
            return super().runner(command, timeout)

    tools = OnlyControlConverts()
    report = export_arms([CONTROL, ARM], run_root=str(run_root), root=str(root),
                         llama_cpp_dir=str(_llama_cpp(tmp_path)),
                         export_fn=tools.export_fn, runner=tools.runner)

    assert report["arms"][CONTROL.name]["quantized"] is True
    assert report["arms"][ARM.name]["converted"] is False
    assert summarize(report["arms"]) == {
        "quantized": [CONTROL.name], "refused": [ARM.name], "skipped": [],
        "n_quantized": 1, "n_arms": 2}


def test_an_arm_that_never_trained_is_recorded_not_omitted(tmp_path):
    """A dropped arm and an arm nobody ran read identically in a manifest."""
    record = export_arm(ARM, run_root=str(tmp_path / "runs"),
                        root=str(tmp_path / "evidence"),
                        llama_cpp_dir=str(_llama_cpp(tmp_path)),
                        export_fn=lambda *a: pytest.fail("exported anyway"),
                        runner=lambda *a: pytest.fail("ran anyway"))

    assert record["skipped"] == "no-checkpoint"
    assert record["converted"] is False
    assert "checkpoint" in record["reason"]


def test_an_incomplete_llama_cpp_is_refused_before_the_hf_export(tmp_path):
    """A missing toolchain is an environment fault, and a minute of export
    would prove nothing about the arm."""
    _checkpoint(tmp_path / "runs", ARM)
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-quantize").unlink()

    record = export_arm(ARM, run_root=str(tmp_path / "runs"),
                        root=str(tmp_path / "evidence"),
                        llama_cpp_dir=str(llama_cpp),
                        export_fn=lambda *a: pytest.fail("exported anyway"),
                        runner=lambda *a: pytest.fail("ran anyway"))

    assert record["skipped"] == "no-llama-cpp"
    assert "llama-quantize" in record["reason"]


def test_a_conversion_that_exits_zero_without_a_file_is_not_converted(tmp_path):
    """The record says a GGUF exists only when one does."""
    _checkpoint(tmp_path / "runs", ARM)
    tools = FakeToolchain(convert_writes=False)

    record = export_arm(ARM, run_root=str(tmp_path / "runs"),
                        root=str(tmp_path / "evidence"),
                        llama_cpp_dir=str(_llama_cpp(tmp_path)),
                        export_fn=tools.export_fn, runner=tools.runner)

    assert record["convert_returncode"] == 0
    assert record["converted"] is False


# ----------------------------------------------------------- the skip rule ----

def test_a_finished_export_is_skipped_rather_than_rebuilt(tmp_path):
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    _checkpoint(run_root, ARM)
    llama_cpp = str(_llama_cpp(tmp_path))
    first = FakeToolchain()
    export_arms([ARM], run_root=str(run_root), root=str(root),
                llama_cpp_dir=llama_cpp, export_fn=first.export_fn,
                runner=first.runner)

    second = FakeToolchain()
    report = export_arms([ARM], run_root=str(run_root), root=str(root),
                         llama_cpp_dir=llama_cpp, export_fn=second.export_fn,
                         runner=second.runner)

    assert report["arms"][ARM.name]["skipped"] == "already-exported"
    assert second.calls == [] and second.exports == []
    # The skip keeps the artifact record, not just the word "skipped".
    assert report["arms"][ARM.name]["gguf_q4_0"]["sha256"]


def test_a_retrained_checkpoint_is_re_exported(tmp_path):
    """Keyed on the digest: new weights in the run directory mean the artifact
    beside them describes a model that no longer exists."""
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    _checkpoint(run_root, ARM, payload=b"first")
    llama_cpp = str(_llama_cpp(tmp_path))
    first = FakeToolchain()
    before = export_arms([ARM], run_root=str(run_root), root=str(root),
                         llama_cpp_dir=llama_cpp, export_fn=first.export_fn,
                         runner=first.runner)

    _checkpoint(run_root, ARM, payload=b"retrained")
    second = FakeToolchain()
    after = export_arms([ARM], run_root=str(run_root), root=str(root),
                        llama_cpp_dir=llama_cpp, export_fn=second.export_fn,
                        runner=second.runner)

    assert "skipped" not in after["arms"][ARM.name]
    assert second.exports, "a changed checkpoint must be exported again"
    assert (after["arms"][ARM.name]["checkpoint_sha256"]
            != before["arms"][ARM.name]["checkpoint_sha256"])


def test_a_deleted_gguf_is_rebuilt_even_though_the_digest_matches(tmp_path):
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    _checkpoint(run_root, ARM)
    llama_cpp = str(_llama_cpp(tmp_path))
    first = FakeToolchain()
    export_arms([ARM], run_root=str(run_root), root=str(root),
                llama_cpp_dir=llama_cpp, export_fn=first.export_fn,
                runner=first.runner)
    gguf_paths(ARM, root=str(root))[1].unlink()

    second = FakeToolchain()
    report = export_arms([ARM], run_root=str(run_root), root=str(root),
                         llama_cpp_dir=llama_cpp, export_fn=second.export_fn,
                         runner=second.runner)

    assert "skipped" not in report["arms"][ARM.name]
    assert report["arms"][ARM.name]["quantized"] is True


def test_refresh_rebuilds_an_export_that_still_matches(tmp_path):
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    _checkpoint(run_root, ARM)
    llama_cpp = str(_llama_cpp(tmp_path))
    first = FakeToolchain()
    export_arms([ARM], run_root=str(run_root), root=str(root),
                llama_cpp_dir=llama_cpp, export_fn=first.export_fn,
                runner=first.runner)

    second = FakeToolchain()
    report = export_arms([ARM], run_root=str(run_root), root=str(root),
                         llama_cpp_dir=llama_cpp, export_fn=second.export_fn,
                         runner=second.runner, refresh=True)

    assert "skipped" not in report["arms"][ARM.name]
    assert second.exports


@pytest.mark.parametrize("record, matches", [
    (None, False),
    ({"checkpoint_sha256": "a" * 64, "quantized": False}, False),
    ({"checkpoint_sha256": "b" * 64, "quantized": True,
      "gguf_q4_0": {"path": "/nowhere/model-q4_0.gguf"}}, False),
])
def test_exported_from_refuses_every_incomplete_record(record, matches):
    assert exported_from(record, "a" * 64) is matches


# ------------------------------------------------------------- the cleanup ----

def test_the_hf_intermediate_survives_a_failed_quantize(tmp_path):
    """The failure path is exactly when somebody needs to read it."""
    _checkpoint(tmp_path / "runs", ARM)
    tools = FakeToolchain(quantize_rc=1)

    record = export_arm(ARM, run_root=str(tmp_path / "runs"),
                        root=str(tmp_path / "evidence"),
                        llama_cpp_dir=str(_llama_cpp(tmp_path)),
                        export_fn=tools.export_fn, runner=tools.runner)

    assert record["quantized"] is False
    assert Path(record["hf_dir"]).is_dir()
    assert record["quantize_tail"] == ["quantize output"]


def test_the_hf_intermediate_is_removed_once_the_q4_0_exists(tmp_path):
    _checkpoint(tmp_path / "runs", ARM)
    tools = FakeToolchain()

    record = export_arm(ARM, run_root=str(tmp_path / "runs"),
                        root=str(tmp_path / "evidence"),
                        llama_cpp_dir=str(_llama_cpp(tmp_path)),
                        export_fn=tools.export_fn, runner=tools.runner)

    assert record["quantized"] is True
    assert record["hf_dir_removed"] is True
    assert not Path(record["hf_dir"]).exists()
    # The artifacts the later columns read are still there.
    assert Path(record["gguf_q4_0"]["path"]).exists()
    assert Path(record["gguf_f16"]["path"]).exists()


def test_keep_hf_leaves_the_intermediate_in_place(tmp_path):
    _checkpoint(tmp_path / "runs", ARM)
    tools = FakeToolchain()

    record = export_arm(ARM, run_root=str(tmp_path / "runs"),
                        root=str(tmp_path / "evidence"),
                        llama_cpp_dir=str(_llama_cpp(tmp_path)),
                        keep_hf=True, export_fn=tools.export_fn,
                        runner=tools.runner)

    assert Path(record["hf_dir"]).is_dir()
    assert "hf_dir_removed" not in record


# -------------------------------------------------------------- the manifest ---

def test_the_manifest_is_written_after_every_arm(tmp_path):
    """A sweep the deadline or a dead session cuts short keeps what it got."""
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    for arm in (CONTROL, ARM):
        _checkpoint(run_root, arm)
    tools = FakeToolchain()

    def export_fn(checkpoint, config, hf_dir):
        if ARM.name in str(hf_dir):
            raise RuntimeError("the session ended here")
        tools.export_fn(checkpoint, config, hf_dir)

    with pytest.raises(RuntimeError):
        export_arms([CONTROL, ARM], run_root=str(run_root), root=str(root),
                    llama_cpp_dir=str(_llama_cpp(tmp_path)),
                    export_fn=export_fn, runner=tools.runner)

    written = json.loads(manifest_path(root=str(root)).read_text())
    assert written["arms"][CONTROL.name]["quantized"] is True
    assert ARM.name not in written["arms"]


def test_a_second_sweep_merges_into_the_existing_manifest(tmp_path):
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    for arm in (CONTROL, ARM):
        _checkpoint(run_root, arm)
    llama_cpp = str(_llama_cpp(tmp_path))
    tools = FakeToolchain()
    export_arms([CONTROL], run_root=str(run_root), root=str(root),
                llama_cpp_dir=llama_cpp, export_fn=tools.export_fn,
                runner=tools.runner)
    export_arms([ARM], run_root=str(run_root), root=str(root),
                llama_cpp_dir=llama_cpp, export_fn=tools.export_fn,
                runner=tools.runner)

    written = json.loads(manifest_path(root=str(root)).read_text())
    assert sorted(written["arms"]) == sorted([CONTROL.name, ARM.name])


def test_the_two_stages_do_not_share_an_artifact_directory(tmp_path):
    """One arm name, two run directories: a mistyped tag must not overwrite the
    other stage's GGUF."""
    stage_a, stage_b = gguf_paths(ARM, "stagea"), gguf_paths(ARM, "stageb")
    assert stage_a[1] != stage_b[1]
    assert manifest_path("stagea") != manifest_path("stageb")


def test_cli_export_writes_the_manifest_and_summarizes(tmp_path, capsys,
                                                       monkeypatch):
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    _checkpoint(run_root, CONTROL)
    tools = FakeToolchain()
    monkeypatch.setattr(evidence, "_export_hf", tools.export_fn)
    monkeypatch.setattr(evidence, "_run", tools.runner)

    code = evidence_main(["--run-root", str(run_root),
                          "--evidence-root", str(root),
                          "export", "--arms", CONTROL.name,
                          "--llama-cpp-dir", str(_llama_cpp(tmp_path))])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["quantized"] == [CONTROL.name]
    assert json.loads(Path(printed["manifest"]).read_text())["tag"] == "stagea"
