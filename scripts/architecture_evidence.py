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

**Which arms are measured is read, not retyped.** `--arms-from-report stagea`
takes the list out of that screen's own report, exactly as stage B takes its
training list. It matters most here because the retrieval column is the most
expensive thing this phase runs: one `llama-cli` process per item, two tasks over
four depths at `RETRIEVAL_PER_DEPTH` items, so ~400 stock invocations per arm --
around half an hour of CPU each, or most of a day for the full grid, on a box
that is training at the same time. Spending that on arms the screen has already
put outside its quality floor buys nothing the gate will read: `bpb_check`
blocks them whatever their retention turns out to be. Reading the list rather
than typing it also means the arms measured here and the arms trained at stage B
cannot silently differ.

Unlike stage B, this pass may read the report of the shape it is measuring.
Selecting stage A's arms from stage A's own *BPB* screen and then measuring their
retrieval, export and decode is not circular -- the columns are different
questions, and none of them chose the arms. What would be circular is a training
stage taking its arm list from its own results, which is why `advanced_selection`
keeps that refusal and this caller does not ask for it.

**The intermediate HF directory is deleted once its Q4_0 exists.** Fifteen arms
at ~105M parameters is ~3GB of safetensors that no later step reads, on a box
that also holds fifteen checkpoints and a corpus. It is deleted only after a
successful quantize, so a failure leaves every intermediate in place for
diagnosis, and `--keep-hf` turns the cleanup off. Nothing is reachable by this
path except directories this module itself wrote.

Subcommands: `export`, `retrieval`, `decode`, `all`.
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
from scripts.architecture_report import (RETRIEVAL_GATE_TASKS,  # noqa: E402
                                         RETRIEVAL_MIN_ITEMS_PER_DEPTH,
                                         RETRIEVAL_ROOT, TRAINED_CONTEXT,
                                         advanced_selection,
                                         decode_report_path,
                                         read_decode_passes, read_retrieval,
                                         selection_notes, swept_arms,
                                         templated_cards)
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

#: One arm's whole retrieval pass: two tasks over four depths at
#: `RETRIEVAL_PER_DEPTH` items, each item a separate `llama-cli` invocation over
#: a prompt as long as the trained context. Generous because the box may be
#: training at the same time, and bounded because a wedged pass must not take
#: the arms behind it with it.
RETRIEVAL_TIMEOUT_S = 7200.0

#: Items per depth for this phase's retrieval pass. Not a number chosen here:
#: `RETRIEVAL_MIN_ITEMS_PER_DEPTH` is what the gate's own 2-point threshold needs
#: before a cell can carry it at all, and `retrieval_eval.py`'s default of ten
#: puts every cell in `no-power` -- a whole column measured at a resolution
#: coarser than the thing it screens for. Imported rather than copied so the two
#: cannot drift; raising the gate's threshold lowers this automatically.
RETRIEVAL_PER_DEPTH = RETRIEVAL_MIN_ITEMS_PER_DEPTH

#: The two depths `decode_check` reads: where a conv hybrid has least to gain,
#: and the context the arms were trained at. Both, or the column is unmeasured --
#: a decode number at depth 0 alone measures this architecture exactly where its
#: argument does not apply.
DECODE_DEPTHS = (0, TRAINED_CONTEXT)

#: `decode_bench.py`'s own defaults, restated because this module writes the
#: invocation rather than typing it: three alternating rounds is what its
#: anti-drift argument rests on, and 128 generated tokens is every decode number
#: this project has recorded.
DECODE_ROUNDS = 3
DECODE_N_GEN = 128

#: Every model, at both depths, three rounds each, on a box that may also be
#: training. Bounded so a wedged benchmark cannot hold the phase open.
DECODE_TIMEOUT_S = 10800.0


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

def resolve_arms(args, shape: StageShape, tag: Optional[str] = None
                 ) -> Tuple[Sequence[ArchArm], Optional[dict]]:
    """The arms this pass measures, and the record of where the list came from.

    `--arms` and `--arms-from-report` both name the arm list, and the point of
    the second is that the first is not retyped, so naming both is refused rather
    than resolved by precedence. `selected_arms` puts the control first either
    way: every column here is read as a delta against it, and `decode_check` will
    not read an arm and the control out of separate invocations at all.

    No `for_shape`: see the module docstring. A stage selecting its own *training*
    arms from its own results is circular; measuring a different column on the
    arms a BPB screen advanced is the intended use.

    **With neither flag the default is the arms the stage swept, not the arms
    its shape defines.** Stage A trains its whole grid so the two coincide;
    stage B trains the four arms stage A advanced out of fifteen, and the shape
    has no idea which four. Defaulting to `arms_for(shape)` walks the chain into
    an arm that never trained -- `a8-kv2`, eligible but not selected -- after
    spending the export and retrieval minutes on the arms ahead of it. That is
    the failure the scoring pass hit and fixed; the sweep artifact is the record
    of what ran, so it is what "this stage's arms" means here too.
    """
    if not getattr(args, "arms_from_report", None):
        stage_arms = (swept_arms(tag or shape.tag, arms_for(shape),
                                 args.report_root)
                      or arms_for(shape))
        return selected_arms(args.arms, stage_arms), None
    if args.arms:
        raise SystemExit(
            "--arms and --arms-from-report both name the arm list, and the "
            "point of the second is that the first is not retyped; pass one")
    selection = advanced_selection(from_tag=args.arms_from_report,
                                   report_root=args.report_root)
    for note in selection_notes(selection):
        print(note, file=sys.stderr)
    return (selected_arms(",".join(selection["selected"]), arms_for(shape)),
            selection)


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
                selection: Optional[dict] = None, **kwargs) -> dict:
    """Every arm, control first, with the manifest rewritten after each one.

    Re-entrant like the sweep and the scoring pass: a session that ends partway
    through costs the arm it was on and nothing else. The records are keyed by
    arm name rather than listed, because this file is *merged* on every rerun --
    a list would need a de-duplication rule, and the one it would need is "keep
    the newest record per arm", which is what a dict already is.

    `selection` is the report that chose these arms when one did. Written into
    the manifest rather than left implicit: "these four arms" and "the four arms
    this report advanced, on this rule" are different claims about a partial
    manifest, and only the second can be checked later.
    """
    path = manifest_path(tag, root)
    manifest = _read_json(path)
    records = dict(manifest.get("arms") or {})
    # Preserved across a rerun that did not read a report, for the same reason
    # the records are: a narrower second pass must not erase where the first
    # pass's arm list came from.
    advanced_from = selection or manifest.get("advanced_from")
    for arm in arms:
        records[arm.name] = export_arm(arm, tag=tag, root=root,
                                       previous=records.get(arm.name), **kwargs)
        _write_json(path, {"tag": tag, "shape": shape.name, "at": _utcnow(),
                           "advanced_from": advanced_from, "arms": records})
    return {"tag": tag, "shape": shape.name, "manifest": str(path),
            "advanced_from": advanced_from, "arms": records}


def summarize_export(records: dict) -> dict:
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


# ================================================================ retrieval ====
# The gate's fourth column, and the one that also carries the third: a retrieval
# scorecard whose artifact is a GGUF is the only evidence `export_check` accepts
# that stock llama.cpp loaded this shape, because it exists only if `llama-cli`
# actually generated from that file.
#
# **Scored on the Q4_0 artifact, through the stock binary.** The alternative --
# the PyTorch checkpoint -- is cheaper and measures a model nobody deploys, and
# it would leave the export column unmeasured while looking like a retrieval
# result. Arm and control get identical treatment, so quantization is a constant
# of the comparison rather than a confound in it.
#
# **`retrieval_eval.py` is invoked, not imported.** The scorecard's provenance
# has to be the evaluator's own -- its prompts, its normalization, its artifact
# digest -- and a second in-process path that assembled the same card by hand is
# a second implementation to keep honest.


def retrieval_out_dir(arm: ArchArm, tag: str = "stagea",
                      root: str = RETRIEVAL_ROOT) -> Path:
    return Path(root) / arm_run_name(arm, tag)


def retrieval_command(gguf: str, out_dir: str, *, llama_cli: str,
                      per_depth: int = RETRIEVAL_PER_DEPTH, threads: int = 8,
                      n_ctx: int = 4096, seed: Optional[int] = None,
                      depths: Optional[str] = None) -> list:
    """The exact `retrieval_eval.py` invocation for one arm.

    Optional flags are omitted rather than defaulted here: the evaluator owns
    what a depth set and a seed are for this project, and restating its defaults
    in a second file is how two of them end up disagreeing.
    """
    # Resolved beside this file rather than as a path relative to the working
    # directory: a phase the controller detaches is not guaranteed to be
    # standing in the repository root, and a retrieval pass that dies on
    # "No such file" costs an arm's hour for a cwd.
    evaluator = Path(__file__).resolve().parent / "retrieval_eval.py"
    command = [sys.executable, str(evaluator),
               "--backend", "llama-cpp", "--gguf", str(gguf),
               "--llama-cli", str(llama_cli), "--out-dir", str(out_dir),
               "--per-depth", str(int(per_depth)), "--threads", str(int(threads)),
               "--n-ctx", str(int(n_ctx))]
    if seed is not None:
        command += ["--seed", str(int(seed))]
    if depths:
        command += ["--depths", str(depths)]
    return command


def retrieval_scored_from(arm: ArchArm, *, tag: str, root: str,
                          artifact_sha: str,
                          tasks: Sequence[str] = RETRIEVAL_GATE_TASKS) -> bool:
    """True when every gate task is already scored from exactly this GGUF.

    Both halves again, and the digest half matters more here than at the export
    step: a re-exported arm writes a new GGUF into the same path, and a scorecard
    keyed only on existence would keep the previous artifact's retention curve
    under the new artifact's name.

    A third half, and it is the one that would have wedged this phase. The
    report gate refuses a llama.cpp card that is not recorded raw completion,
    which is correct -- but "scored" here answered on digest alone, so the
    stage-A cards measured through a chat template counted as done. The chain
    would skip re-scoring them, the gate would keep refusing them, and the
    column would stay unmeasured with nothing left to run. `--refresh` is not
    the answer: it discards every card including the valid ones, so the cost of
    fixing one stale arm is remeasuring all of them.

    So the two ends share one judgement, `templated_cards`. A card the gate will
    not read is not a card this arm has been scored on.
    """
    scored = read_retrieval(arm, tag=tag, root=root, tasks=tasks)
    if sorted(scored) != sorted(tasks):
        return False
    if templated_cards(*scored.values()):
        return False
    return all(record.get("artifact_sha256") == artifact_sha
               for record in scored.values())


def score_retrieval_arm(arm: ArchArm, record: Optional[dict], *,
                        tag: str = "stagea", root: str = RETRIEVAL_ROOT,
                        llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR,
                        per_depth: int = RETRIEVAL_PER_DEPTH, threads: int = 8,
                        n_ctx: int = 4096, seed: Optional[int] = None,
                        depths: Optional[str] = None, refresh: bool = False,
                        runner: Optional[Callable[[Sequence[str], float],
                                                  Tuple[int, str]]] = None
                        ) -> dict:
    """Run one arm's retrieval pass through stock llama.cpp on its Q4_0 GGUF."""
    runner = runner or _run
    out_dir = retrieval_out_dir(arm, tag, root)
    common = {"arm": arm.name, "run": arm_run_name(arm, tag),
              "is_control": arm.is_control, "out_dir": str(out_dir),
              "at": _utcnow()}

    artifact = (record or {}).get("gguf_q4_0") or {}
    gguf = Path(artifact["path"]) if artifact.get("path") else None
    if gguf is None or not gguf.exists():
        # An arm the converter refused has no artifact to score, and that is
        # already its export verdict -- recorded here rather than presented as a
        # retrieval failure, which would double-count one cause as two.
        return {**common, "skipped": "no-gguf", "scored": False,
                "reason": "this arm has no Q4_0 GGUF; export it first"}

    llama_cli = Path(llama_cpp_dir) / "build" / "bin" / "llama-cli"
    if not llama_cli.exists():
        return {**common, "skipped": "no-llama-cli", "scored": False,
                "reason": f"stock llama.cpp has no {llama_cli}"}

    digest = artifact.get("sha256") or sha256_file(gguf)
    if not refresh and retrieval_scored_from(arm, tag=tag, root=root,
                                             artifact_sha=digest):
        return {**common, "skipped": "already-scored", "scored": True,
                "artifact_sha256": digest, "gguf": str(gguf)}

    command = retrieval_command(str(gguf), str(out_dir), llama_cli=str(llama_cli),
                                per_depth=per_depth, threads=threads,
                                n_ctx=n_ctx, seed=seed, depths=depths)
    returncode, output = runner(command, RETRIEVAL_TIMEOUT_S)
    scored = read_retrieval(arm, tag=tag, root=root)
    result = {**common, "gguf": str(gguf), "artifact_sha256": digest,
              "per_depth": int(per_depth), "threads": int(threads),
              "returncode": returncode, "command": command,
              "tasks": sorted(scored),
              "scored": returncode == 0
                        and sorted(scored) == sorted(RETRIEVAL_GATE_TASKS)}
    if not result["scored"]:
        result["reason"] = ("the retrieval pass did not leave a scorecard for "
                            f"every gate task {list(RETRIEVAL_GATE_TASKS)}")
        result["tail"] = output.strip().splitlines()[-12:]
    return result


def score_retrieval_arms(arms: Sequence[ArchArm] = ARMS, *, tag: str = "stagea",
                         shape: StageShape = STAGE_A,
                         evidence_root: str = EVIDENCE_ROOT,
                         root: str = RETRIEVAL_ROOT,
                         selection: Optional[dict] = None, **kwargs) -> dict:
    """Every arm with an artifact, control first, summary rewritten as it goes.

    The export manifest is the input: an arm is scored on the GGUF that manifest
    says was built from its checkpoint, so a retrieval curve can always be traced
    back to the weights it came from.
    """
    exported = (_read_json(manifest_path(tag, evidence_root)).get("arms") or {})
    path = Path(evidence_root) / f"retrieval-{tag}.json"
    summary = _read_json(path)
    records = dict(summary.get("arms") or {})
    advanced_from = selection or summary.get("advanced_from")
    for arm in arms:
        records[arm.name] = score_retrieval_arm(arm, exported.get(arm.name),
                                                tag=tag, root=root, **kwargs)
        _write_json(path, {"tag": tag, "shape": shape.name, "at": _utcnow(),
                           "retrieval_root": str(root),
                           "advanced_from": advanced_from, "arms": records})
    return {"tag": tag, "shape": shape.name, "summary": str(path),
            "advanced_from": advanced_from, "arms": records}


def summarize_retrieval(records: dict) -> dict:
    """Which arms now carry a retrieval curve, and which the gate will not read."""
    scored = sorted(name for name, record in records.items()
                    if record.get("scored"))
    return {"scored": scored,
            "failed": sorted(name for name, record in records.items()
                             if not record.get("scored")
                             and not record.get("skipped")),
            "skipped": sorted(name for name, record in records.items()
                              if record.get("skipped")
                              and record.get("skipped") != "already-scored"),
            "n_scored": len(scored), "n_arms": len(records)}


# =================================================================== decode ====
# The gate's fifth column. `decode_check` will only read an arm and the control
# out of the *same* pass, because `decode_bench` alternates models within a pass
# precisely so that concurrent box load hits both -- so this is one invocation
# over every arm, not one invocation per arm.
#
# **Both depths, always.** Depth 0 is where a conv hybrid has least to gain and
# `TRAINED_CONTEXT` is where its whole argument lives; a report with one of them
# leaves the column unmeasured, which is the honest reading and a wasted hour.
#
# **A narrower rerun does not silently replace a wider report.** `--out` is one
# file and `decode_bench` overwrites it, so re-running for two finalists after a
# full pass would delete thirteen arms' decode numbers and leave a report that
# looks complete. Refused unless `--refresh` says so.


def decode_models(records: dict, arms: Sequence[ArchArm], tag: str) -> dict:
    """`{run_name: q4_0 path}` for every arm that has an artifact, in order.

    Keyed by run name rather than grid point: `decode_entry` accepts either, and
    only the run name says which stage's checkpoint a number came from.
    """
    models = {}
    for arm in arms:
        artifact = (records.get(arm.name) or {}).get("gguf_q4_0") or {}
        path = artifact.get("path")
        if path and Path(path).exists():
            models[arm_run_name(arm, tag)] = path
    return models


def decode_command(models: dict, out: str, *, bench_bin: str, threads: int = 8,
                   rounds: int = DECODE_ROUNDS, n_gen: int = DECODE_N_GEN,
                   depths: Sequence[int] = DECODE_DEPTHS,
                   note: Optional[str] = None) -> list:
    """The single `decode_bench.py` invocation that measures every arm."""
    bench = Path(__file__).resolve().parent / "decode_bench.py"
    command = [sys.executable, str(bench), "--models"]
    command += [f"{name}={path}" for name, path in models.items()]
    command += ["--threads", str(int(threads)), "--rounds", str(int(rounds)),
                "--n-gen", str(int(n_gen)), "--depths"]
    command += [str(int(depth)) for depth in depths]
    command += ["--bench-bin", str(bench_bin), "--out", str(out)]
    if note:
        command += ["--note", str(note)]
    return command


def dropped_models(out, models: dict) -> list:
    """Models an existing report measures that this invocation would not.

    Read from the report rather than from a manifest: what is at risk is
    whatever that file already contains, including a peer or a released artifact
    somebody benchmarked alongside the grid.
    """
    measured = set()
    for entry in read_decode_passes(out).values():
        measured.update((entry.get("models") or {}).keys())
    return sorted(measured - set(models))


def run_decode(arms: Sequence[ArchArm] = ARMS, *, tag: str = "stagea",
               shape: StageShape = STAGE_A, evidence_root: str = EVIDENCE_ROOT,
               out: Optional[str] = None, report_root: str = REPORT_ROOT,
               llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR, threads: int = 8,
               rounds: int = DECODE_ROUNDS, n_gen: int = DECODE_N_GEN,
               depths: Sequence[int] = DECODE_DEPTHS, note: Optional[str] = None,
               refresh: bool = False, selection: Optional[dict] = None,
               runner: Optional[Callable[[Sequence[str], float],
                                         Tuple[int, str]]] = None) -> dict:
    """Benchmark every exported arm against the control in one alternating pass."""
    runner = runner or _run
    # Per stage, not per program: see `decode_report_path`. A shared filename
    # made the would-drop-models guard below fire on every stage after the
    # first, which is how stage B finished with an unmeasured decode column.
    out = out or decode_report_path(tag, report_root)
    records = _read_json(manifest_path(tag, evidence_root)).get("arms") or {}
    models = decode_models(records, arms, tag)
    common = {"tag": tag, "shape": shape.name, "out": str(out),
              "models": models, "depths": [int(depth) for depth in depths],
              "threads": int(threads), "advanced_from": selection,
              "at": _utcnow()}
    if not models:
        return {**common, "skipped": "no-gguf", "measured": False,
                "reason": "no arm in the export manifest has a Q4_0 GGUF yet"}

    bench_bin = Path(llama_cpp_dir) / "build" / "bin" / "llama-bench"
    if not bench_bin.exists():
        return {**common, "skipped": "no-llama-bench", "measured": False,
                "reason": f"stock llama.cpp has no {bench_bin}"}

    dropped = dropped_models(out, models)
    if dropped and not refresh:
        return {**common, "skipped": "would-drop-models", "measured": False,
                "dropped": dropped,
                "reason": f"{out} already measures {dropped}, which this "
                          "invocation does not; one file holds the passes and "
                          "rerunning replaces it. Include those arms, write "
                          "elsewhere with --out, or pass --refresh."}

    command = decode_command(models, str(out), bench_bin=str(bench_bin),
                             threads=threads, rounds=rounds, n_gen=n_gen,
                             depths=depths, note=note)
    returncode, output = runner(command, DECODE_TIMEOUT_S)
    passes = read_decode_passes(out)
    covered = sorted({depth for _, depth in passes})
    result = {**common, "returncode": returncode, "command": command,
              "measured_depths": covered,
              "measured": returncode == 0
                          and all(int(depth) in covered for depth in depths)}
    if not result["measured"]:
        result["reason"] = (f"the benchmark did not leave a pass at every depth "
                            f"{list(depths)}")
        result["tail"] = output.strip().splitlines()[-12:]
    return result


# ==================================================================== chain ====
# The three passes above are one pipeline over one input -- the arms a screen
# advanced -- and they run in a lane that is not the GPU's. Launched as three
# separate phases they need three sessions to notice a predecessor finished and
# start the next, and the CPU lane is idle for however long that takes. Under a
# controller whose whole point is that launched work outlives the session that
# launched it, that idle time is the session's schedule leaking into the
# program's: stage B holds the GPU for eight hours, and the evidence columns
# either finish inside that window or the phase ends with a table whose
# `export`, `retrieval` and `decode` cells still read `unmeasured`.
#
# Three decisions about how the passes compose.
#
# **Export gates the chain; retrieval does not gate decode.** Both later passes
# read the export manifest, so nothing quantized means neither has an artifact
# to measure, and the chain stops rather than writing two summaries that say
# `no-gguf` once per arm. Retrieval and decode are *different columns* read off
# the same GGUFs independently, so a retrieval failure must not also cost the
# decode column -- `gate_verdict` already returns `unproven` for one unmeasured
# column, and losing a second to the first one's failure would only widen a
# verdict that is already blocked.
#
# **A refused arm is not a failed chain.** `export_arm` records a conversion
# stock llama.cpp declined as that arm's finding, and `export_check` reads it as
# `unmeasured` rather than as a pass. The chain fails only when *no* arm
# produced an artifact, because that is the case where continuing measures
# nothing at all.
#
# **The depths are deliberately not overridable here.** `decode_check` reads
# depth 0 and the trained context or it reads nothing, and `RETRIEVAL_PER_DEPTH`
# is already the smallest count the gate's own threshold can carry. A chain flag
# that could narrow either would only ever buy time by producing a column the
# gate declines to read, which is the failure this whole module exists to avoid.


def run_all(arms: Sequence[ArchArm] = ARMS, *, tag: str = "stagea",
            shape: StageShape = STAGE_A, run_root: str = RUN_ROOT,
            evidence_root: str = EVIDENCE_ROOT,
            retrieval_root: str = RETRIEVAL_ROOT,
            decode_out: Optional[str] = None, report_root: str = REPORT_ROOT,
            llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR, keep_hf: bool = False,
            per_depth: int = RETRIEVAL_PER_DEPTH, threads: int = 8,
            n_ctx: int = 4096, seed: Optional[int] = None,
            rounds: int = DECODE_ROUNDS, n_gen: int = DECODE_N_GEN,
            note: Optional[str] = None, refresh: bool = False,
            selection: Optional[dict] = None) -> dict:
    """Export, then retrieval, then decode, as one phase.

    Each pass is the same re-entrant one the individual subcommands run, so a
    chain the deadline or a dead controller cut short costs the arm it was on
    and resumes from the manifests the finished passes already wrote.
    """
    stages: list = []

    export = export_arms(arms, tag=tag, shape=shape, root=evidence_root,
                         run_root=run_root, llama_cpp_dir=llama_cpp_dir,
                         keep_hf=keep_hf, refresh=refresh, selection=selection)
    summary = summarize_export(export["arms"])
    stages.append({"stage": "export", **summary,
                   "manifest": export["manifest"],
                   "ok": summary["n_quantized"] > 0,
                   "reason": None if summary["n_quantized"] else
                             "no arm produced a Q4_0 GGUF, so neither the "
                             "retrieval nor the decode column has an artifact "
                             "to measure"})
    advanced_from = export.get("advanced_from")

    if stages[0]["ok"]:
        retrieval = score_retrieval_arms(
            arms, tag=tag, shape=shape, evidence_root=evidence_root,
            root=retrieval_root, llama_cpp_dir=llama_cpp_dir,
            per_depth=per_depth, threads=threads, n_ctx=n_ctx, seed=seed,
            refresh=refresh, selection=selection)
        scored = summarize_retrieval(retrieval["arms"])
        stages.append({"stage": "retrieval", **scored,
                       "summary": retrieval["summary"],
                       "ok": not scored["failed"],
                       "reason": None if not scored["failed"] else
                                 f"{scored['failed']} ran but left no scorecard "
                                 f"for every gate task"})

        decode = run_decode(arms, tag=tag, shape=shape,
                            evidence_root=evidence_root, out=decode_out,
                            report_root=report_root,
                            llama_cpp_dir=llama_cpp_dir, threads=threads,
                            rounds=rounds, n_gen=n_gen, note=note,
                            refresh=refresh, selection=selection)
        stages.append({"stage": "decode", "ok": bool(decode.get("measured")),
                       "out": decode.get("out"),
                       "measured_depths": decode.get("measured_depths"),
                       "skipped": decode.get("skipped"),
                       "reason": decode.get("reason")})

    result = {
        "tag": tag, "shape": shape.name, "at": _utcnow(),
        "advanced_from": advanced_from,
        "arms": [arm.name for arm in arms],
        "stages": stages,
        # Which pass ended the chain, or None. Only export can: the other two
        # are independent columns, so a failure in either is reported without
        # taking the one behind it down.
        "stopped_at": None if stages[0]["ok"] else "export",
        "ok": all(stage["ok"] for stage in stages),
    }
    _write_json(Path(evidence_root) / f"evidence-{tag}.json", result)
    return result


# ====================================================================== cli ====

#: One help string for all three passes: the flag means the same thing in each,
#: and a subcommand whose description of it drifted would read as a different
#: mechanism.
ARMS_FROM_REPORT_HELP = ("take the arms from that tag's committed report "
                         "instead, so the columns are measured on the arms the "
                         "screen advanced rather than on a retyped list")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--evidence-root", default=EVIDENCE_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT,
                        help="where --arms-from-report reads a committed report")
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
    export.add_argument("--arms-from-report", default=None, metavar="TAG",
                        help=ARMS_FROM_REPORT_HELP)
    export.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP_DIR)
    export.add_argument("--keep-hf", action="store_true",
                        help="keep the intermediate HF directory that the "
                             "stock converter reads")
    export.add_argument("--refresh", action="store_true",
                        help="re-export arms whose GGUF already matches their "
                             "checkpoint")

    retrieval = sub.add_parser("retrieval")
    retrieval.add_argument("--arms", default=None,
                           help="comma-separated subset, control first")
    retrieval.add_argument("--arms-from-report", default=None, metavar="TAG",
                           help=ARMS_FROM_REPORT_HELP + "; this is the pass "
                                "where that matters most, at ~400 stock "
                                "llama-cli invocations per arm")
    retrieval.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP_DIR)
    retrieval.add_argument("--retrieval-root", default=RETRIEVAL_ROOT,
                           help="where the gate reads one directory per arm run")
    retrieval.add_argument("--per-depth", type=int, default=RETRIEVAL_PER_DEPTH,
                           help="items per depth; below the default the gate's "
                                "own threshold has no power and every cell "
                                "reads no-power rather than pass")
    retrieval.add_argument("--threads", type=int, default=8)
    retrieval.add_argument("--n-ctx", type=int, default=4096)
    retrieval.add_argument("--seed", type=int, default=None)
    retrieval.add_argument("--depths", default=None)
    retrieval.add_argument("--refresh", action="store_true",
                           help="re-score arms already scored from this GGUF")

    decode = sub.add_parser("decode")
    decode.add_argument("--arms", default=None,
                        help="comma-separated subset, control first; every arm "
                             "named here is measured in one alternating pass")
    decode.add_argument("--arms-from-report", default=None, metavar="TAG",
                        help=ARMS_FROM_REPORT_HELP)
    decode.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP_DIR)
    decode.add_argument("--out", default=None,
                        help="the single report the gate reads; arm and control "
                             "must appear in the same pass inside it. Defaults "
                             "to this stage's decode-<tag>.json")
    decode.add_argument("--threads", type=int, default=8)
    decode.add_argument("--rounds", type=int, default=DECODE_ROUNDS)
    decode.add_argument("--n-gen", type=int, default=DECODE_N_GEN)
    decode.add_argument("--depths", type=int, nargs="+", default=list(DECODE_DEPTHS),
                        help="both the empty context and the trained one, or "
                             "the gate's decode column stays unmeasured")
    decode.add_argument("--note", default=None,
                        help="what else the box was doing; absolutes are only "
                             "comparable within one invocation")
    decode.add_argument("--refresh", action="store_true",
                        help="replace a report that measures models this "
                             "invocation does not")

    chain = sub.add_parser(
        "all", help="export, then retrieval, then decode, as one phase, so the "
                    "CPU lane does not wait for a session between them")
    chain.add_argument("--arms", default=None,
                       help="comma-separated subset, control first")
    chain.add_argument("--arms-from-report", default=None, metavar="TAG",
                       help=ARMS_FROM_REPORT_HELP)
    chain.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP_DIR)
    chain.add_argument("--keep-hf", action="store_true",
                       help="keep the intermediate HF directory the stock "
                            "converter reads")
    chain.add_argument("--retrieval-root", default=RETRIEVAL_ROOT,
                       help="where the gate reads one directory per arm run")
    chain.add_argument("--per-depth", type=int, default=RETRIEVAL_PER_DEPTH,
                       help="retrieval items per depth; below the default the "
                            "gate's threshold has no power")
    chain.add_argument("--n-ctx", type=int, default=4096)
    chain.add_argument("--seed", type=int, default=None)
    chain.add_argument("--out", default=None,
                       help="the single decode report the gate reads; defaults "
                            "to this stage's decode-<tag>.json, which is what "
                            "keeps one stage's pass from refusing the next's")
    chain.add_argument("--threads", type=int, default=8,
                       help="llama.cpp threads, for both stock passes")
    chain.add_argument("--rounds", type=int, default=DECODE_ROUNDS)
    chain.add_argument("--n-gen", type=int, default=DECODE_N_GEN)
    chain.add_argument("--note", default=None,
                       help="what else the box was doing; decode absolutes are "
                            "only comparable within one invocation")
    chain.add_argument("--refresh", action="store_true",
                       help="redo work every pass would otherwise skip as "
                            "already done")
    # No --depths, at either pass. See the chain's comment block: the gate reads
    # depth 0 and the trained context or it reads nothing, so narrowing here
    # could only buy time by producing a column that is refused.

    args = parser.parse_args(argv)
    shape = SHAPES[args.shape]
    tag = shape.tag if args.tag is None else args.tag
    arms, selection = resolve_arms(args, shape, tag)

    if args.command == "export":
        report = export_arms(arms, tag=tag, shape=shape,
                             root=args.evidence_root,
                             run_root=args.run_root,
                             llama_cpp_dir=args.llama_cpp_dir,
                             keep_hf=args.keep_hf, refresh=args.refresh,
                             selection=selection)
        print(json.dumps({**summarize_export(report["arms"]),
                          "advanced_from": (report["advanced_from"] or {}).get("report"),
                          "manifest": report["manifest"]}, indent=2))
        return 0

    if args.command == "retrieval":
        report = score_retrieval_arms(arms,
                                      tag=tag, shape=shape,
                                      evidence_root=args.evidence_root,
                                      root=args.retrieval_root,
                                      llama_cpp_dir=args.llama_cpp_dir,
                                      per_depth=args.per_depth,
                                      threads=args.threads, n_ctx=args.n_ctx,
                                      seed=args.seed, depths=args.depths,
                                      refresh=args.refresh, selection=selection)
        print(json.dumps({**summarize_retrieval(report["arms"]),
                          "advanced_from": (report["advanced_from"] or {}).get("report"),
                          "summary": report["summary"]}, indent=2))
        return 0

    if args.command == "decode":
        report = run_decode(arms,
                            tag=tag, shape=shape,
                            evidence_root=args.evidence_root, out=args.out,
                            report_root=args.report_root,
                            llama_cpp_dir=args.llama_cpp_dir,
                            threads=args.threads, rounds=args.rounds,
                            n_gen=args.n_gen, depths=args.depths,
                            note=args.note, refresh=args.refresh,
                            selection=selection)
        print(json.dumps(report, indent=2))
        # A refused or incomplete benchmark is a non-zero exit: the caller is a
        # phase, and a phase that "passed" without measuring is how an
        # unmeasured column reaches a report as a finished one.
        return 0 if report.get("measured") else 1

    if args.command == "all":
        report = run_all(arms, tag=tag, shape=shape, run_root=args.run_root,
                         evidence_root=args.evidence_root,
                         retrieval_root=args.retrieval_root,
                         decode_out=args.out, report_root=args.report_root,
                         llama_cpp_dir=args.llama_cpp_dir,
                         keep_hf=args.keep_hf, per_depth=args.per_depth,
                         threads=args.threads, n_ctx=args.n_ctx, seed=args.seed,
                         rounds=args.rounds, n_gen=args.n_gen, note=args.note,
                         refresh=args.refresh, selection=selection)
        print(json.dumps(report, indent=2))
        # Same rule as `decode`'s, over three passes: a phase that exits 0
        # without measuring is how an unmeasured column reaches a report
        # looking finished.
        return 0 if report["ok"] else 1

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
