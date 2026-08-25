"""Phase 6: the shippable artifact for each arm, so the gate's export column
stops reading `unmeasured`.

`architecture_report.py` gates a shape on five columns and `score` measures one
of them. Two of the rest need nothing but the artifacts that already exist --
`kv` is arithmetic, `bpb` is the scorecard -- and two need a file this repository
does not yet produce for an arm: the retrieval column wants a stock llama.cpp run
against the arm, and `export_check` reads its verdict off *that* run's artifact
kind rather than off a flag anybody set. Both start here, with a Q4_0 GGUF per
arm converted and quantized by unmodified llama.cpp.

Four decisions, each of which is a way this step could quietly stop being
evidence.

**A conversion failure is the measurement, not an error.** Whether stock
llama.cpp can convert and load a shape is exactly what the export column asks,
and fifteen arms differ only in how many of their blocks attend. If an attention
placement the grid explores is one `convert_hf_to_gguf.py` refuses, that is the
phase's finding about that shape -- so the failure is recorded with its return
code and its output tail, the arm is marked unconverted, and the sweep continues
to the next arm. Raising instead would abandon fourteen answers to report one,
and `export_check` already knows how to read an arm with no GGUF: `unmeasured`,
never a pass.

**The Q4_0 file is the artifact, and the f16 GGUF is kept beside it.** Q4_0 is
what a deployment ships and what `decode_bench` measures, so it is what the
retrieval pass should score -- a retention number taken on an f16 artifact
answers a question about a file nobody runs. The f16 GGUF is the quantizer's own
input and costs one conversion to keep, so it stays: a later paired FP16-vs-Q4
reading of any arm needs it, and re-deriving it means re-exporting the
checkpoint.

**Keyed on the checkpoint digest, like every other re-entrant step in this
phase.** An arm is skipped only when the bytes that produced its GGUF are the
bytes sitting in its run directory now. A `--refresh` that retrained an arm, or a
stage-B run landing in a stage-A directory, must not leave the old shape's
artifact behind wearing the new arm's name.

**The intermediate HF directory is deleted once its Q4_0 exists.** Fifteen arms
at ~105M parameters is ~3GB of safetensors that no later step reads, on a box
that also holds fifteen checkpoints and a corpus. It is deleted only after a
successful quantize, so a failure leaves every intermediate in place for
diagnosis, and `--keep-hf` turns the cleanup off. Nothing is reachable by this
path except directories this module itself wrote.

Subcommands: `export`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daedalus.scorecard import sha256_file  # noqa: E402
from scripts.architecture_sweep import (ARMS, REPORT_ROOT, RUN_ROOT,  # noqa: E402
                                        SHAPES, STAGE_A, ArchArm, StageShape,
                                        arm_checkpoint_path, arm_run_name,
                                        arms_for, selected_arms)

#: Where an arm's GGUFs land: beside the sweep and scorecard artifacts, one
#: directory per *run* rather than per grid point, because one arm name maps to
#: two run directories across stages and a GGUF is a property of a checkpoint.
EVIDENCE_ROOT = f"{REPORT_ROOT}/evidence"

#: Stock llama.cpp, as installed on this box. Overridable, never vendored: the
#: program's fixed decision is that an unmodified binary must be able to run
#: what this phase recommends.
DEFAULT_LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp")

#: This module's own bounds on the two stock invocations. A 105M-parameter
#: conversion runs in a couple of minutes and a quantize in well under one, so
#: these are generous enough to absorb a loaded box and short enough that a hung
#: converter cannot eat a sweep's night.
CONVERT_TIMEOUT_S = 1800.0
QUANTIZE_TIMEOUT_S = 900.0


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


# =================================================================== layout ====

def arm_evidence_dir(arm: ArchArm, tag: str = "stagea",
                     root: str = EVIDENCE_ROOT) -> Path:
    return Path(root) / arm_run_name(arm, tag)


def gguf_paths(arm: ArchArm, tag: str = "stagea",
               root: str = EVIDENCE_ROOT) -> Tuple[Path, Path]:
    """`(f16, q4_0)` for this arm. Named for the quantization, not the arm, so a
    directory listing says what a file is without a lookup table."""
    directory = arm_evidence_dir(arm, tag, root)
    return directory / "model-f16.gguf", directory / "model-q4_0.gguf"


def manifest_path(tag: str = "stagea", root: str = EVIDENCE_ROOT) -> Path:
    return Path(root) / f"export-{tag}.json"


def _artifact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "MB": round(path.stat().st_size / 1e6, 2), "sha256": sha256_file(path)}


def exported_from(record: Optional[dict], checkpoint_sha: str) -> bool:
    """True when `record` describes a Q4_0 built from exactly these bytes.

    Both halves are required. The digest alone would keep an arm whose GGUF was
    deleted off the disk; the file alone would keep an artifact built from a
    checkpoint that has since been retrained, which is the failure `--refresh`
    exists to cause and must not silently survive.
    """
    if not record or record.get("checkpoint_sha256") != checkpoint_sha:
        return False
    if not record.get("quantized"):
        return False
    q4 = (record.get("gguf_q4_0") or {}).get("path")
    return bool(q4) and Path(q4).exists()


# ================================================================ the steps ====

def _run(command: Sequence[str], timeout: float) -> Tuple[int, str]:
    """One stock invocation, captured. Never raises on a non-zero exit: the exit
    code *is* the finding this module reports."""
    try:
        result = subprocess.run(list(command), capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:g}s: {' '.join(map(str, command))}"
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _export_hf(checkpoint: str, config: str, hf_dir: str) -> None:
    """checkpoint.pt -> the HF directory the stock converter reads.

    Imported inside the call because it pulls torch and transformers, and the
    layout and skip rules above must stay testable without either.
    """
    from export import export_hf_model, export_tokenizer

    export_hf_model(checkpoint, config, hf_dir)
    # No `tokenizer=`: the arms trained on the default SmolLM2 vocabulary, and
    # the tokenizer written here is the one whose ids the GGUF will carry.
    export_tokenizer(hf_dir)


def export_arm(arm: ArchArm, *, tag: str = "stagea", run_root: str = RUN_ROOT,
               root: str = EVIDENCE_ROOT,
               llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR,
               previous: Optional[dict] = None, refresh: bool = False,
               keep_hf: bool = False,
               export_fn: Optional[Callable[[str, str, str], None]] = None,
               runner: Optional[Callable[[Sequence[str], float],
                                         Tuple[int, str]]] = None) -> dict:
    """Convert and quantize one finished arm through unmodified llama.cpp.

    Returns the manifest record for the arm in every outcome, including the ones
    where nothing was produced -- an arm that never trained and an arm whose
    shape the converter refused are different facts, and a record that omitted
    either would read as an arm nobody got to.
    """
    # Resolved here rather than defaulted in the signature so that the CLI path
    # is exercisable without torch or a llama.cpp build: a default bound at
    # definition time cannot be replaced by patching this module.
    export_fn = export_fn or _export_hf
    runner = runner or _run

    checkpoint = arm_checkpoint_path(arm, tag, run_root)
    common = {"arm": arm.name, "preset": arm.config,
              "run": arm_run_name(arm, tag), "is_control": arm.is_control,
              "kv_bytes_per_context_token": arm.kv_bytes_per_context_token,
              "checkpoint": str(checkpoint), "at": _utcnow()}
    if not checkpoint.exists():
        return {**common, "skipped": "no-checkpoint", "converted": False,
                "quantized": False,
                "reason": f"no checkpoint at {checkpoint}; this arm either "
                          "never ran or its run directory moved"}

    digest = sha256_file(checkpoint)
    common["checkpoint_sha256"] = digest
    f16, q4_0 = gguf_paths(arm, tag, root)
    if not refresh and exported_from(previous, digest):
        return {**previous, "skipped": "already-exported"}

    directory = arm_evidence_dir(arm, tag, root)
    directory.mkdir(parents=True, exist_ok=True)
    hf_dir = directory / "hf"

    converter = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    quantizer = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    missing = [str(path) for path in (converter, quantizer) if not path.exists()]
    if missing:
        # Refused before the HF export rather than after it: a missing stock
        # toolchain is an environment fault, and spending a minute of GPU-idle
        # export to discover it would prove nothing about the arm.
        return {**common, "skipped": "no-llama-cpp", "converted": False,
                "quantized": False, "llama_cpp_dir": str(llama_cpp_dir),
                "reason": f"stock llama.cpp is incomplete; missing {missing}"}

    export_fn(str(checkpoint), arm.config, str(hf_dir))

    convert_rc, convert_out = runner(
        [sys.executable, str(converter), str(hf_dir), "--outfile", str(f16),
         "--outtype", "f16"], CONVERT_TIMEOUT_S)
    record = {**common, "llama_cpp_dir": str(llama_cpp_dir),
              "hf_dir": str(hf_dir), "convert_returncode": convert_rc,
              "converted": convert_rc == 0 and f16.exists(),
              "quantized": False}
    if not record["converted"]:
        record["reason"] = ("stock convert_hf_to_gguf.py did not produce a GGUF "
                            "for this shape")
        record["convert_tail"] = convert_out.strip().splitlines()[-12:]
        return record
    record["gguf_f16"] = _artifact(f16)

    quantize_rc, quantize_out = runner(
        [str(quantizer), str(f16), str(q4_0), "Q4_0"], QUANTIZE_TIMEOUT_S)
    record["quantize_returncode"] = quantize_rc
    record["quantized"] = quantize_rc == 0 and q4_0.exists()
    if not record["quantized"]:
        record["reason"] = "stock llama-quantize did not produce a Q4_0 GGUF"
        record["quantize_tail"] = quantize_out.strip().splitlines()[-12:]
        return record
    record["gguf_q4_0"] = _artifact(q4_0)

    if not keep_hf and hf_dir.is_dir():
        shutil.rmtree(hf_dir)
        record["hf_dir_removed"] = True
    return record


def export_arms(arms: Sequence[ArchArm] = ARMS, *, tag: str = "stagea",
                shape: StageShape = STAGE_A, root: str = EVIDENCE_ROOT,
                **kwargs) -> dict:
    """Every arm, control first, with the manifest rewritten after each one.

    Re-entrant like the sweep and the scoring pass: a session that ends partway
    through costs the arm it was on and nothing else. The records are keyed by
    arm name rather than listed, because this file is *merged* on every rerun --
    a list would need a de-duplication rule, and the one it would need is "keep
    the newest record per arm", which is what a dict already is.
    """
    path = manifest_path(tag, root)
    manifest = _read_json(path)
    records = dict(manifest.get("arms") or {})
    for arm in arms:
        records[arm.name] = export_arm(arm, tag=tag, root=root,
                                       previous=records.get(arm.name), **kwargs)
        _write_json(path, {"tag": tag, "shape": shape.name, "at": _utcnow(),
                           "arms": records})
    return {"tag": tag, "shape": shape.name, "manifest": str(path),
            "arms": records}


def summarize(records: dict) -> dict:
    """What the sweep produced, in the terms the gate reads it in."""
    converted = sorted(name for name, record in records.items()
                       if record.get("quantized"))
    refused = sorted(name for name, record in records.items()
                     if record.get("converted") is False
                     and not record.get("skipped"))
    return {"quantized": converted, "refused": refused,
            "skipped": sorted(name for name, record in records.items()
                              if record.get("skipped")
                              and record.get("skipped") != "already-exported"),
            "n_quantized": len(converted), "n_arms": len(records)}


# ====================================================================== cli ====

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--evidence-root", default=EVIDENCE_ROOT)
    parser.add_argument("--shape", default=STAGE_A.name, choices=list(SHAPES),
                        help="which stage's arms and presets to read")
    parser.add_argument("--tag", default=None,
                        help="run-directory prefix; defaults to the one the "
                             "--shape owns, which is what keeps one stage's "
                             "artifacts out of another's directory")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export")
    export.add_argument("--arms", default=None,
                        help="comma-separated subset, control first")
    export.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP_DIR)
    export.add_argument("--keep-hf", action="store_true",
                        help="keep the intermediate HF directory that the "
                             "stock converter reads")
    export.add_argument("--refresh", action="store_true",
                        help="re-export arms whose GGUF already matches their "
                             "checkpoint")

    args = parser.parse_args(argv)
    shape = SHAPES[args.shape]
    tag = shape.tag if args.tag is None else args.tag

    if args.command == "export":
        report = export_arms(selected_arms(args.arms, arms_for(shape)),
                             tag=tag, shape=shape, root=args.evidence_root,
                             run_root=args.run_root,
                             llama_cpp_dir=args.llama_cpp_dir,
                             keep_hf=args.keep_hf, refresh=args.refresh)
        print(json.dumps({**summarize(report["arms"]),
                          "manifest": report["manifest"]}, indent=2))
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
