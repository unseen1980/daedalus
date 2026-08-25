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
from typing import Sequence

import pytest

from daedalus.scorecard import (ArtifactRef, Provenance, Scorecard,
                                write_scorecard)
from scripts import architecture_evidence as evidence
from scripts.architecture_evidence import (RETRIEVAL_PER_DEPTH, dropped_models,
                                           exported_from, export_arm,
                                           export_arms, gguf_paths,
                                           manifest_path, retrieval_command,
                                           retrieval_out_dir,
                                           retrieval_scored_from, run_decode,
                                           score_retrieval_arm,
                                           score_retrieval_arms,
                                           summarize_export,
                                           summarize_retrieval)
from scripts.architecture_evidence import main as evidence_main
from scripts.architecture_report import (RETRIEVAL_GATE_TASKS,
                                         RETRIEVAL_MIN_ITEMS_PER_DEPTH,
                                         TRAINED_CONTEXT, decode_entry,
                                         export_check, read_decode_passes,
                                         read_retrieval,
                                         retrieval_scorecard_path)
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
    assert summarize_export(report["arms"]) == {
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


# -------------------------------------------------------------- retrieval ----

def _retrieval_card(path: Path, *, task: str, artifact: str, sha: str,
                    per_depth: int = RETRIEVAL_PER_DEPTH) -> None:
    """A scorecard shaped as `retrieval_eval.py` writes one."""
    metrics = {"exact_match": 0.5, "n": float(per_depth * 4)}
    for depth in (256, 512, 1024, 2048):
        metrics[f"exact_match_d{depth}"] = 0.5
        metrics[f"n_d{depth}"] = float(per_depth)
    write_scorecard(path, Scorecard(
        kind="retrieval", name=f"retrieval-{task}",
        provenance=Provenance(
            artifact=ArtifactRef(path=artifact, sha256=sha, kind="gguf-q4_0"),
            tokenizer=ArtifactRef(path="tok", sha256="0" * 64, kind="tokenizer"),
            seed=1, git_sha="deadbee", bpb_mode="not-applicable"),
        metrics=metrics, created_at="2026-08-25T00:00:00Z", item_count=1))


class FakeRetrieval:
    """Stands in for `retrieval_eval.py`, writing the scorecards it would."""

    def __init__(self, *, returncode: int = 0,
                 tasks: Sequence = RETRIEVAL_GATE_TASKS, sha: str = "c" * 64):
        self.returncode = returncode
        self.tasks = tuple(tasks)
        self.sha = sha
        self.calls = []

    def runner(self, command, timeout):
        command = [str(part) for part in command]
        self.calls.append(command)
        out_dir = Path(command[command.index("--out-dir") + 1])
        gguf = command[command.index("--gguf") + 1]
        out_dir.mkdir(parents=True, exist_ok=True)
        for task in self.tasks:
            _retrieval_card(out_dir / f"retrieval-{task}.json", task=task,
                            artifact=gguf, sha=self.sha)
        return self.returncode, "retrieval output"


def _exported(tmp_path: Path, arm=None, *, sha: str = "c" * 64) -> dict:
    """The export-manifest record a scored arm is read from."""
    arm = arm or ARM
    gguf = tmp_path / "evidence" / f"arch-stagea-{arm.name}" / "model-q4_0.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"q4")
    return {"arm": arm.name, "quantized": True,
            "gguf_q4_0": {"path": str(gguf), "sha256": sha}}


def test_the_retrieval_pass_defaults_to_the_items_the_gate_needs():
    """Ten items per depth -- the evaluator's default -- makes one item worth
    ten points against a two-point threshold, so every cell would read no-power.
    The default here is the gate's own requirement, imported, not a number
    picked to look thorough."""
    assert RETRIEVAL_PER_DEPTH == RETRIEVAL_MIN_ITEMS_PER_DEPTH
    command = retrieval_command("m.gguf", "out", llama_cli="llama-cli")
    assert command[command.index("--per-depth") + 1] == str(RETRIEVAL_PER_DEPTH)
    # Flags the evaluator owns are left to it rather than restated here.
    assert "--seed" not in command and "--depths" not in command


def test_the_arm_is_scored_on_its_q4_0_through_stock_llama_cpp(tmp_path):
    """The artifact kind is the export column's only evidence, so the pass has
    to run on the GGUF and through the stock binary."""
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    fake = FakeRetrieval()

    record = score_retrieval_arm(ARM, _exported(tmp_path),
                                 root=str(tmp_path / "retrieval"),
                                 llama_cpp_dir=str(llama_cpp),
                                 runner=fake.runner)

    assert record["scored"] is True
    assert record["tasks"] == sorted(RETRIEVAL_GATE_TASKS)
    command = fake.calls[0]
    assert command[command.index("--gguf") + 1].endswith("model-q4_0.gguf")
    assert command[command.index("--llama-cli") + 1].endswith("llama-cli")
    # And what it wrote is what `export_check` reads as a stock-llama.cpp load.
    scored = read_retrieval(ARM, root=str(tmp_path / "retrieval"))
    assert export_check(scored)["status"] == "pass"


def test_an_arm_the_converter_refused_is_not_a_retrieval_failure(tmp_path):
    """One cause, recorded once: the export column already says this shape has
    no artifact."""
    record = score_retrieval_arm(ARM, {"arm": ARM.name, "quantized": False},
                                 root=str(tmp_path / "retrieval"),
                                 llama_cpp_dir=str(_llama_cpp(tmp_path)),
                                 runner=lambda *a: pytest.fail("ran anyway"))

    assert record["skipped"] == "no-gguf"
    assert record["scored"] is False


def test_a_missing_llama_cli_is_recorded_rather_than_run(tmp_path):
    record = score_retrieval_arm(ARM, _exported(tmp_path),
                                 root=str(tmp_path / "retrieval"),
                                 llama_cpp_dir=str(_llama_cpp(tmp_path)),
                                 runner=lambda *a: pytest.fail("ran anyway"))

    assert record["skipped"] == "no-llama-cli"
    assert record["scored"] is False


def test_a_partial_pass_is_not_recorded_as_scored(tmp_path):
    """A pass that scored passkey and died before mqar leaves a directory that
    looks scored; the gate would read one task and call the other unmeasured."""
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    fake = FakeRetrieval(returncode=1, tasks=("passkey",))

    record = score_retrieval_arm(ARM, _exported(tmp_path),
                                 root=str(tmp_path / "retrieval"),
                                 llama_cpp_dir=str(llama_cpp),
                                 runner=fake.runner)

    assert record["scored"] is False
    assert record["tasks"] == ["passkey"]
    assert record["tail"] == ["retrieval output"]


def test_a_scored_arm_is_skipped_but_a_re_exported_one_is_not(tmp_path):
    """Keyed on the artifact digest: a re-exported arm writes a new GGUF to the
    same path, and the old curve must not survive under the new artifact."""
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    root = str(tmp_path / "retrieval")
    fake = FakeRetrieval()
    score_retrieval_arm(ARM, _exported(tmp_path), root=root,
                        llama_cpp_dir=str(llama_cpp), runner=fake.runner)

    again = score_retrieval_arm(ARM, _exported(tmp_path), root=root,
                                llama_cpp_dir=str(llama_cpp),
                                runner=lambda *a: pytest.fail("re-scored"))
    assert again["skipped"] == "already-scored"

    rebuilt = FakeRetrieval(sha="d" * 64)
    after = score_retrieval_arm(ARM, _exported(tmp_path, sha="d" * 64),
                                root=root, llama_cpp_dir=str(llama_cpp),
                                runner=rebuilt.runner)
    assert "skipped" not in after and after["scored"] is True
    assert retrieval_scored_from(ARM, tag="stagea", root=root,
                                 artifact_sha="d" * 64)
    assert not retrieval_scored_from(ARM, tag="stagea", root=root,
                                     artifact_sha="c" * 64)


def test_refresh_rescores_an_arm_already_scored_from_this_gguf(tmp_path):
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    root = str(tmp_path / "retrieval")
    fake = FakeRetrieval()
    score_retrieval_arm(ARM, _exported(tmp_path), root=root,
                        llama_cpp_dir=str(llama_cpp), runner=fake.runner)

    again = score_retrieval_arm(ARM, _exported(tmp_path), root=root,
                                llama_cpp_dir=str(llama_cpp),
                                runner=fake.runner, refresh=True)

    assert "skipped" not in again
    assert len(fake.calls) == 2


def test_the_retrieval_sweep_reads_the_export_manifest_and_summarizes(tmp_path):
    """The GGUF an arm is scored on is the one the manifest says came from its
    checkpoint, so a curve can be traced back to the weights behind it."""
    run_root, evidence_root = tmp_path / "runs", tmp_path / "evidence"
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    for arm in (CONTROL, ARM):
        _checkpoint(run_root, arm)
    tools = FakeToolchain()
    export_arms([CONTROL, ARM], run_root=str(run_root), root=str(evidence_root),
                llama_cpp_dir=str(llama_cpp), export_fn=tools.export_fn,
                runner=tools.runner)

    fake = FakeRetrieval()
    report = score_retrieval_arms([CONTROL, ARM], evidence_root=str(evidence_root),
                                  root=str(tmp_path / "retrieval"),
                                  llama_cpp_dir=str(llama_cpp),
                                  runner=fake.runner)

    assert summarize_retrieval(report["arms"])["scored"] == sorted(
        [CONTROL.name, ARM.name])
    scored_paths = {call[call.index("--gguf") + 1] for call in fake.calls}
    assert scored_paths == {
        str(gguf_paths(arm, root=str(evidence_root))[1])
        for arm in (CONTROL, ARM)}
    written = json.loads(Path(report["summary"]).read_text())
    assert sorted(written["arms"]) == sorted([CONTROL.name, ARM.name])


def test_the_two_stages_do_not_share_a_retrieval_directory():
    assert retrieval_out_dir(ARM, "stagea") != retrieval_out_dir(ARM, "stageb")


def test_the_scorecards_land_where_the_gate_looks_for_them(tmp_path):
    """The join between this producer and its reader, pinned.

    A scorecard written one directory away from `retrieval_scorecard_path` is
    the most expensive kind of mistake available here: every pass succeeds,
    every file is on disk, and the gate reports the column unmeasured anyway --
    an arm's hour of llama-cli spent to change nothing.
    """
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    root = str(tmp_path / "retrieval")

    score_retrieval_arm(ARM, _exported(tmp_path), root=root,
                        llama_cpp_dir=str(llama_cpp),
                        runner=FakeRetrieval().runner)

    for task in RETRIEVAL_GATE_TASKS:
        assert retrieval_scorecard_path(ARM, task, tag="stagea",
                                        root=root).exists()


def test_cli_retrieval_scores_the_arms_the_export_manifest_names(
        tmp_path, capsys, monkeypatch):
    run_root, evidence_root = tmp_path / "runs", tmp_path / "evidence"
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    _checkpoint(run_root, CONTROL)
    tools = FakeToolchain()
    export_arms([CONTROL], run_root=str(run_root), root=str(evidence_root),
                llama_cpp_dir=str(llama_cpp), export_fn=tools.export_fn,
                runner=tools.runner)
    fake = FakeRetrieval()
    monkeypatch.setattr(evidence, "_run", fake.runner)

    code = evidence_main(["--run-root", str(run_root),
                          "--evidence-root", str(evidence_root),
                          "retrieval", "--arms", CONTROL.name,
                          "--retrieval-root", str(tmp_path / "retrieval"),
                          "--llama-cpp-dir", str(llama_cpp)])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["scored"] == [CONTROL.name]
    assert printed["n_scored"] == 1


# ----------------------------------------------------------------- decode ----

class FakeBench:
    """Stands in for `decode_bench.py`, writing the report it would."""

    def __init__(self, *, returncode: int = 0, depths=(0, 2048)):
        self.returncode = returncode
        self.depths = tuple(depths)
        self.calls = []

    def runner(self, command, timeout):
        command = [str(part) for part in command]
        self.calls.append(command)
        names = [spec.split("=", 1)[0] for spec in command[
            command.index("--models") + 1:] if "=" in spec]
        out = Path(command[command.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"passes": [
            {"threads": 8, "depth": depth,
             "models": {name: {"mean": 100.0, "file_mb": 60.0}
                        for name in names}}
            for depth in self.depths]}))
        return self.returncode, "bench output"


def _exported_manifest(tmp_path: Path, arms) -> Path:
    """An export manifest with a Q4_0 on disk for each arm."""
    evidence_root = tmp_path / "evidence"
    records = {}
    for arm in arms:
        gguf = evidence_root / f"arch-stagea-{arm.name}" / "model-q4_0.gguf"
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"q4")
        records[arm.name] = {"arm": arm.name, "quantized": True,
                             "gguf_q4_0": {"path": str(gguf), "sha256": "c" * 64}}
    path = manifest_path(root=str(evidence_root))
    path.write_text(json.dumps({"tag": "stagea", "arms": records}))
    return evidence_root


def test_every_arm_and_the_control_land_in_one_invocation(tmp_path):
    """`decode_check` refuses to read an arm and the control out of separate
    invocations, so one pass has to contain both."""
    evidence_root = _exported_manifest(tmp_path, (CONTROL, ARM))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")
    bench = FakeBench()

    report = run_decode([CONTROL, ARM], evidence_root=str(evidence_root),
                        out=str(tmp_path / "decode.json"),
                        llama_cpp_dir=str(llama_cpp), runner=bench.runner)

    assert report["measured"] is True
    assert len(bench.calls) == 1, "one invocation, or the numbers are not paired"
    specs = [part for part in bench.calls[0] if "=" in part]
    assert sorted(spec.split("=")[0] for spec in specs) == sorted(
        [f"arch-stagea-{CONTROL.name}", f"arch-stagea-{ARM.name}"])
    # And the report reads back as a pass containing both, at both depths.
    passes = read_decode_passes(tmp_path / "decode.json")
    assert sorted(depth for _, depth in passes) == [0, TRAINED_CONTEXT]
    for entry in passes.values():
        assert decode_entry(entry, [f"arch-stagea-{ARM.name}"])["mean"] == 100.0


def test_a_report_missing_the_trained_context_is_not_measured(tmp_path):
    """Depth 0 alone measures this architecture where its argument does not
    apply."""
    evidence_root = _exported_manifest(tmp_path, (CONTROL,))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")
    bench = FakeBench(depths=(0,))

    report = run_decode([CONTROL], evidence_root=str(evidence_root),
                        out=str(tmp_path / "decode.json"),
                        llama_cpp_dir=str(llama_cpp), runner=bench.runner)

    assert report["measured"] is False
    assert report["measured_depths"] == [0]
    assert report["tail"] == ["bench output"]


def test_a_narrower_rerun_refuses_to_replace_a_wider_report(tmp_path):
    """One file holds the passes, so re-running for one arm after a full sweep
    would delete the others' decode numbers and still look complete."""
    evidence_root = _exported_manifest(tmp_path, (CONTROL, ARM))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")
    out = tmp_path / "decode.json"
    run_decode([CONTROL, ARM], evidence_root=str(evidence_root), out=str(out),
               llama_cpp_dir=str(llama_cpp), runner=FakeBench().runner)

    refused = run_decode([CONTROL], evidence_root=str(evidence_root),
                         out=str(out), llama_cpp_dir=str(llama_cpp),
                         runner=lambda *a: pytest.fail("overwrote the report"))

    assert refused["skipped"] == "would-drop-models"
    assert refused["dropped"] == [f"arch-stagea-{ARM.name}"]
    assert refused["measured"] is False
    # The wider report is still on disk, untouched.
    assert dropped_models(out, {}) == [f"arch-stagea-{name}"
                                       for name in sorted([CONTROL.name, ARM.name])]


def test_refresh_replaces_a_report_deliberately(tmp_path):
    evidence_root = _exported_manifest(tmp_path, (CONTROL, ARM))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")
    out = tmp_path / "decode.json"
    run_decode([CONTROL, ARM], evidence_root=str(evidence_root), out=str(out),
               llama_cpp_dir=str(llama_cpp), runner=FakeBench().runner)

    replaced = run_decode([CONTROL], evidence_root=str(evidence_root),
                          out=str(out), llama_cpp_dir=str(llama_cpp),
                          runner=FakeBench().runner, refresh=True)

    assert replaced["measured"] is True
    assert dropped_models(out, {}) == [f"arch-stagea-{CONTROL.name}"]


def test_an_arm_without_an_artifact_is_left_out_of_the_pass(tmp_path):
    """A refused conversion has no file to benchmark; the rest still are."""
    evidence_root = _exported_manifest(tmp_path, (CONTROL,))
    records = json.loads(manifest_path(root=str(evidence_root)).read_text())
    records["arms"][ARM.name] = {"arm": ARM.name, "converted": False,
                                 "quantized": False}
    manifest_path(root=str(evidence_root)).write_text(json.dumps(records))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")

    report = run_decode([CONTROL, ARM], evidence_root=str(evidence_root),
                        out=str(tmp_path / "decode.json"),
                        llama_cpp_dir=str(llama_cpp), runner=FakeBench().runner)

    assert sorted(report["models"]) == [f"arch-stagea-{CONTROL.name}"]
    assert report["measured"] is True


def test_nothing_exported_yet_is_a_skip_not_a_benchmark(tmp_path):
    report = run_decode([CONTROL], evidence_root=str(tmp_path / "evidence"),
                        out=str(tmp_path / "decode.json"),
                        llama_cpp_dir=str(_llama_cpp(tmp_path)),
                        runner=lambda *a: pytest.fail("benchmarked nothing"))

    assert report["skipped"] == "no-gguf"
    assert report["measured"] is False


def test_cli_decode_exits_non_zero_when_the_column_is_unmeasured(tmp_path,
                                                                 capsys,
                                                                 monkeypatch):
    """A phase that exits 0 without measuring is how an unmeasured column
    reaches a report looking finished."""
    evidence_root = _exported_manifest(tmp_path, (CONTROL,))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")
    monkeypatch.setattr(evidence, "_run", FakeBench(depths=(0,)).runner)

    code = evidence_main(["--evidence-root", str(evidence_root),
                          "decode", "--arms", CONTROL.name,
                          "--out", str(tmp_path / "decode.json"),
                          "--llama-cpp-dir", str(llama_cpp)])

    assert code == 1
    assert json.loads(capsys.readouterr().out)["measured"] is False


def test_cli_decode_measures_and_exits_zero(tmp_path, capsys, monkeypatch):
    evidence_root = _exported_manifest(tmp_path, (CONTROL,))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")
    monkeypatch.setattr(evidence, "_run", FakeBench().runner)

    code = evidence_main(["--evidence-root", str(evidence_root),
                          "decode", "--arms", CONTROL.name,
                          "--out", str(tmp_path / "decode.json"),
                          "--llama-cpp-dir", str(llama_cpp),
                          "--note", "scoring pass running on the GPU"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["measured"] is True
    assert printed["depths"] == [0, TRAINED_CONTEXT]


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


# ------------------------------------------ the decision reaches the columns ----
# The retrieval column is one `llama-cli` process per item -- two tasks, four
# depths, RETRIEVAL_PER_DEPTH items -- so measuring an arm the screen already put
# outside its BPB floor spends about half an hour of a shared CPU on a cell the
# gate will never read: `bpb_check` blocks that arm whatever its retention is.
# The arm list therefore comes from the same committed report stage B trains
# from, and both refusals that protects are exercised here.


def _commit_stage_a_report(root, *, selected, verdict="advance", scored=None):
    """The half of the stage-A report a launcher reads, where it reads it."""
    from scripts.architecture_report import report_path
    from scripts.architecture_sweep import ARMS, STAGE_A

    path = report_path(STAGE_A.tag, str(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "tag": STAGE_A.tag, "created_at": "2026-08-25T00:00:00Z",
        "control": CONTROL.name, "shape": {"name": STAGE_A.name},
        "rows": [{"arm": arm.name} for arm in (ARMS if scored is None
                                               else scored)],
        "stage_b": {"verdict": verdict, "selected": list(selected),
                    "frontier": list(selected), "eligible": list(selected),
                    "dropped_from_frontier": [],
                    "rule": {"floor_pct": 0.5, "max_arms": 3}},
    }, indent=2) + "\n")
    return path


def test_the_retrieval_pass_measures_the_arms_the_report_advanced(
        tmp_path, capsys, monkeypatch):
    evidence_root = _exported_manifest(tmp_path, (CONTROL, ARM,
                                                  ARMS_BY_NAME["a3-kv4"]))
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    fake = FakeRetrieval()
    monkeypatch.setattr(evidence, "_run", fake.runner)
    _commit_stage_a_report(tmp_path / "reports", selected=[ARM.name])

    code = evidence_main(["--evidence-root", str(evidence_root),
                          "--report-root", str(tmp_path / "reports"),
                          "retrieval", "--arms-from-report", "stagea",
                          "--retrieval-root", str(tmp_path / "retrieval"),
                          "--llama-cpp-dir", str(llama_cpp)])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    # The control is measured because every column is a delta against it; the
    # arm the screen dropped is not, and that is the hour this buys back.
    assert printed["scored"] == sorted([CONTROL.name, ARM.name])
    assert printed["advanced_from"].endswith("stagea-report.json")
    summary = json.loads(
        (Path(evidence_root) / "retrieval-stagea.json").read_text())
    assert summary["advanced_from"]["selected"] == [ARM.name]
    assert sorted(summary["arms"]) == sorted([CONTROL.name, ARM.name])


def test_export_records_which_report_chose_its_arms(tmp_path, monkeypatch):
    """A manifest that lists three of fifteen arms is either a screen's
    selection or an interrupted sweep, and the difference is not recoverable
    from the arm count."""
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    _checkpoint(run_root, CONTROL)
    _checkpoint(run_root, ARM)
    tools = FakeToolchain()
    monkeypatch.setattr(evidence, "_export_hf", tools.export_fn)
    monkeypatch.setattr(evidence, "_run", tools.runner)
    _commit_stage_a_report(tmp_path / "reports", selected=[ARM.name])

    code = evidence_main(["--run-root", str(run_root),
                          "--evidence-root", str(root),
                          "--report-root", str(tmp_path / "reports"),
                          "export", "--arms-from-report", "stagea",
                          "--llama-cpp-dir", str(_llama_cpp(tmp_path))])

    assert code == 0
    manifest = json.loads(manifest_path(root=str(root)).read_text())
    assert manifest["advanced_from"]["report"].endswith("stagea-report.json")
    assert sorted(manifest["arms"]) == sorted([CONTROL.name, ARM.name])


def test_a_second_pass_keeps_where_the_first_pass_arms_came_from(tmp_path,
                                                                 monkeypatch):
    """A rerun without the flag must not turn a recorded selection into an
    unattributed list; nothing about the manifest would show the loss."""
    run_root, root = tmp_path / "runs", tmp_path / "evidence"
    _checkpoint(run_root, CONTROL)
    tools = FakeToolchain()
    monkeypatch.setattr(evidence, "_export_hf", tools.export_fn)
    monkeypatch.setattr(evidence, "_run", tools.runner)
    _commit_stage_a_report(tmp_path / "reports", selected=[CONTROL.name])
    common = ["--run-root", str(run_root), "--evidence-root", str(root),
              "--report-root", str(tmp_path / "reports"), "export",
              "--llama-cpp-dir", str(_llama_cpp(tmp_path))]

    evidence_main(common + ["--arms-from-report", "stagea"])
    evidence_main(common + ["--arms", CONTROL.name])

    manifest = json.loads(manifest_path(root=str(root)).read_text())
    assert manifest["advanced_from"]["report"].endswith("stagea-report.json")


def test_naming_the_evidence_arms_twice_is_refused(tmp_path, monkeypatch):
    """Two sources of truth for the arm list is the failure this closes, so
    neither silently wins."""
    monkeypatch.setattr(evidence, "_run",
                        lambda *a: pytest.fail("measured an unresolved list"))
    _commit_stage_a_report(tmp_path / "reports", selected=[ARM.name])

    with pytest.raises(SystemExit):
        evidence_main(["--evidence-root", str(tmp_path / "evidence"),
                       "--report-root", str(tmp_path / "reports"),
                       "retrieval", "--arms-from-report", "stagea",
                       "--arms", "a3-kv4",
                       "--retrieval-root", str(tmp_path / "retrieval")])


def test_a_no_advance_report_stops_the_evidence_pass(tmp_path, monkeypatch):
    """`no-advance` means no arm held the control's quality, so every arm is
    already blocked on BPB; measuring their retention is hours spent to confirm
    a verdict the report has recorded."""
    monkeypatch.setattr(evidence, "_run",
                        lambda *a: pytest.fail("measured past a no-advance"))
    _commit_stage_a_report(tmp_path / "reports", selected=[],
                           verdict="no-advance")

    with pytest.raises(SystemExit):
        evidence_main(["--evidence-root", str(tmp_path / "evidence"),
                       "--report-root", str(tmp_path / "reports"),
                       "retrieval", "--arms-from-report", "stagea",
                       "--retrieval-root", str(tmp_path / "retrieval")])


def test_an_unscored_screen_stops_the_evidence_pass(tmp_path, monkeypatch):
    """No report is not an empty arm list: the columns here are measured on a
    conclusion, and there is no default better than stopping."""
    monkeypatch.setattr(evidence, "_run",
                        lambda *a: pytest.fail("measured without a report"))

    with pytest.raises(SystemExit):
        evidence_main(["--evidence-root", str(tmp_path / "evidence"),
                       "--report-root", str(tmp_path / "reports"),
                       "decode", "--arms-from-report", "stagea",
                       "--out", str(tmp_path / "decode.json")])


# ------------------------------------------------------------------ chain ----
# Three passes over one arm list, in a lane that is not the GPU's. Launched
# separately each needs a session to notice the previous one finished, and the
# CPU lane idles until one arrives -- so what is pinned here is the ordering,
# and which failures stop the chain versus which are recorded and stepped past.


class FakeChain:
    """One runner for all three stock passes, dispatched by command shape.

    The chain is a single process, so every pass reaches `_run`. A fake that
    answered only one of them could not exercise the ordering, which is the
    behaviour being added.
    """

    def __init__(self, *, tools=None, retrieval=None, bench=None):
        self.tools = tools if tools is not None else FakeToolchain()
        self.retrieval = retrieval if retrieval is not None else FakeRetrieval()
        self.bench = bench if bench is not None else FakeBench()
        self.order = []

    def runner(self, command, timeout):
        parts = [str(part) for part in command]
        if "--models" in parts:
            self.order.append("decode")
            return self.bench.runner(parts, timeout)
        if "--gguf" in parts:
            self.order.append("retrieval")
            return self.retrieval.runner(parts, timeout)
        self.order.append("export")
        return self.tools.runner(parts, timeout)

    @property
    def passes(self) -> list:
        """The order the three passes first ran in, deduplicated.

        `export` reaches the runner twice per arm -- convert then quantize --
        so the raw log answers "which pass ran first" only after collapsing
        repeats.
        """
        seen = []
        for name in self.order:
            if name not in seen:
                seen.append(name)
        return seen


def _chain_box(tmp_path: Path, arms, monkeypatch, chain=None):
    """A box with a checkpoint per arm and all three stock entry points."""
    chain = chain if chain is not None else FakeChain()
    run_root = tmp_path / "runs"
    for arm in arms:
        _checkpoint(run_root, arm)
    llama_cpp = _llama_cpp(tmp_path)
    (llama_cpp / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    (llama_cpp / "build" / "bin" / "llama-bench").write_text("#!/bin/sh\n")
    monkeypatch.setattr(evidence, "_export_hf", chain.tools.export_fn)
    monkeypatch.setattr(evidence, "_run", chain.runner)
    return chain, [
        "--run-root", str(run_root),
        "--evidence-root", str(tmp_path / "evidence"),
        # Named explicitly rather than left empty: an empty --arms means "the
        # whole grid", which is a fifteen-arm chain rather than a focused test.
        "all", "--arms", ",".join(arm.name for arm in arms) or CONTROL.name,
        "--retrieval-root", str(tmp_path / "retrieval"),
        "--out", str(tmp_path / "decode.json"),
        "--llama-cpp-dir", str(llama_cpp),
    ]


def test_the_chain_runs_the_three_passes_in_order(tmp_path, capsys,
                                                  monkeypatch):
    """Export has to precede both stock passes -- they read the manifest it
    writes -- and one phase running all three is what keeps the CPU lane busy
    while stage B holds the GPU."""
    chain, argv = _chain_box(tmp_path, (CONTROL, ARM), monkeypatch)

    code = evidence_main(argv)

    assert code == 0
    assert chain.passes == ["export", "retrieval", "decode"]
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["stopped_at"] is None
    assert [stage["stage"] for stage in printed["stages"]] == [
        "export", "retrieval", "decode"]
    # And it leaves one artifact recording the whole chain, beside the three
    # each pass already writes.
    written = json.loads(
        (tmp_path / "evidence" / "evidence-stagea.json").read_text())
    assert written["arms"] == [CONTROL.name, ARM.name]
    assert written["ok"] is True


def test_nothing_exported_stops_the_chain_before_the_stock_passes(
        tmp_path, capsys, monkeypatch):
    """Both later passes measure the GGUFs export writes, so with none of them
    the chain would only write two summaries saying `no-gguf` per arm."""
    chain, argv = _chain_box(tmp_path, (CONTROL, ARM), monkeypatch,
                             chain=FakeChain(tools=FakeToolchain(convert_rc=1)))

    code = evidence_main(argv)

    assert code == 1
    assert chain.passes == ["export"], "measured past an empty export"
    printed = json.loads(capsys.readouterr().out)
    assert printed["stopped_at"] == "export" and printed["ok"] is False
    assert len(printed["stages"]) == 1
    assert printed["stages"][0]["ok"] is False
    assert "Q4_0 GGUF" in printed["stages"][0]["reason"]


def test_a_refused_arm_does_not_stop_the_chain(tmp_path, capsys, monkeypatch):
    """A conversion stock llama.cpp declines is that arm's finding, not a
    broken chain: `export_check` already reads it as unmeasured, and the arms
    that did convert still have four columns to answer."""

    class RefusesOneArm(FakeToolchain):
        def runner(self, command, timeout):
            parts = [str(part) for part in command]
            if any(f"arch-stagea-{ARM.name}" in part for part in parts):
                return 1, "converter refused this shape"
            return super().runner(parts, timeout)

    chain, argv = _chain_box(tmp_path, (CONTROL, ARM), monkeypatch,
                             chain=FakeChain(tools=RefusesOneArm()))

    code = evidence_main(argv)

    assert code == 0
    assert chain.passes == ["export", "retrieval", "decode"]
    printed = json.loads(capsys.readouterr().out)
    assert printed["stages"][0]["quantized"] == [CONTROL.name]
    assert printed["stages"][0]["refused"] == [ARM.name]
    assert printed["stages"][0]["ok"] is True
    assert printed["ok"] is True


def test_a_retrieval_failure_does_not_cost_the_decode_column(tmp_path, capsys,
                                                             monkeypatch):
    """They are different columns read off the same GGUFs. `gate_verdict`
    already returns `unproven` for one unmeasured column; dropping decode
    because retrieval failed would only widen that for no reason."""
    chain, argv = _chain_box(
        tmp_path, (CONTROL, ARM), monkeypatch,
        chain=FakeChain(retrieval=FakeRetrieval(returncode=1)))

    code = evidence_main(argv)

    assert code == 1, "a chain that did not measure must not exit 0"
    assert chain.passes == ["export", "retrieval", "decode"]
    printed = json.loads(capsys.readouterr().out)
    stages = {stage["stage"]: stage for stage in printed["stages"]}
    assert stages["retrieval"]["ok"] is False
    assert stages["decode"]["ok"] is True
    assert printed["stopped_at"] is None


def test_the_chain_measures_the_arms_the_report_advanced(tmp_path, capsys,
                                                         monkeypatch):
    """Same handoff the retrieval pass has: the list is read, not retyped, so
    the arms measured here and the arms stage B trains cannot differ."""
    chain, _ = _chain_box(tmp_path, (CONTROL, ARM, ARMS_BY_NAME["a3-kv4"]),
                          monkeypatch)
    _commit_stage_a_report(tmp_path / "reports", selected=[ARM.name])

    code = evidence_main([
        "--run-root", str(tmp_path / "runs"),
        "--evidence-root", str(tmp_path / "evidence"),
        "--report-root", str(tmp_path / "reports"),
        "all", "--arms-from-report", "stagea",
        "--retrieval-root", str(tmp_path / "retrieval"),
        "--out", str(tmp_path / "decode.json"),
        "--llama-cpp-dir", str(tmp_path / "llama.cpp")])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["arms"] == [CONTROL.name, ARM.name]
    assert printed["advanced_from"]["report"].endswith("stagea-report.json")


def test_the_chain_cannot_narrow_the_depths_the_gate_requires(tmp_path,
                                                              monkeypatch):
    """`decode_check` reads depth 0 and the trained context or it reads
    nothing, so a flag that narrowed either could only buy time by producing a
    column the gate refuses."""
    _chain_box(tmp_path, (CONTROL,), monkeypatch)

    with pytest.raises(SystemExit):
        evidence_main(["--evidence-root", str(tmp_path / "evidence"),
                       "all", "--depths", "0"])
