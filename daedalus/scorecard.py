"""One scorecard schema, consumed by every Phase 3-8 gate.

Every evaluator in this program -- retrieval, paired FP16/Q4_0, code
execution, per-language BPB -- writes this shape, and every preregistered gate
reads it. That is deliberate: the decisions this program makes are worth
fractions of a point (a 0.5% FP16 perplexity ceiling, a 1-point retrieval
floor, a 1-point five-task drop), and a number at that resolution is only
evidence if the artifact that produced it, the items it scored, and the mode it
was measured in are all recorded beside it.

Three properties carry that weight:

  - **Item accounting.** `item_count` plus `item_digest` say *which* items were
    scored, in order. `paired_outcomes` refuses to compare two scorecards whose
    digests differ, which is what stops a `--task-limit` on one side, or a
    dataset revision that reorders rows, from pairing item 7 against a
    different question and producing a confident, meaningless delta. `eval.py`
    already learned this lesson once (see its `item_digest`); this generalises
    it to non-cloze evaluations.

  - **BPB mode.** `evaluate_checkpoint` defaults to a bounded 100-batch sample
    because a full pass is prohibitive at every checkpoint. A sampled BPB and a
    full-pass BPB are different measurements, and the plan requires them to be
    distinguishable in every result file, so `bpb_mode` is mandatory and
    `sample` cannot be recorded without the bound that produced it.

  - **Hashes, not paths.** A path says which file was *opened*; a SHA-256 says
    which bytes were scored. The released artifacts under the operator's
    read-only tree ship with a manifest, so a scorecard that records the same
    digest is checkable against it.

Items live in a sidecar rather than the scorecard body, matching `eval.py`'s
existing split: 10k per-item outcomes would bury the handful of numbers a human
reads, but discarding them would leave only unpaired error bars, which overstate
the error of a difference when both models answered the same items.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SCORECARD_SCHEMA = 1

# The evaluation families this program scores. A new family must be added here
# deliberately: an unknown `kind` means a gate would read a scorecard whose
# metric names it has never seen, and silently find none of them.
VALID_KINDS = {
    "cloze",            # the five general tasks (eval.py)
    "bpb",              # held-out bits-per-byte, by source or language
    "retrieval",        # passkey / associative recall, by context depth
    "paired-quant",     # the same items through FP16 and Q4_0 artifacts
    "code-execution",   # HumanEval+ / MBPP+ pass@1 under sandboxed execution
}

VALID_ARTIFACT_KINDS = {
    "checkpoint",       # a Daedalus .pt
    "hf",               # an exported HF directory
    "gguf-f16",
    "gguf-q4_0",
    "gguf-q6_k",
    "hf-peer",          # a published peer model scored through our harness
    "tokenizer",
    # An oracle pass: the benchmark's own reference solutions, scored through
    # the harness that will score models. What is under test there is the
    # harness, so the artifact is the dataset and no model is named.
    "canonical-solutions",
}

# "not-applicable" is not a dodge: a retrieval or code-execution scorecard
# measures no bits-per-byte at all, and forcing it to claim "full" or "sample"
# would put a false mode in the record.
VALID_BPB_MODES = {"full", "sample", "not-applicable"}

_COMPARATORS = {
    ">=": lambda measured, threshold: measured >= threshold,
    "<=": lambda measured, threshold: measured <= threshold,
    ">": lambda measured, threshold: measured > threshold,
    "<": lambda measured, threshold: measured < threshold,
}


class ScorecardError(ValueError):
    """Raised when a scorecard, or a comparison between two, is not evidence."""


# ------------------------------------------------------------------ hashing ---

def sha256_file(path) -> str:
    """SHA-256 of a file's bytes, in the format the artifact manifests use."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def item_digest(items: Optional[Sequence[dict]]) -> Optional[str]:
    """A fingerprint of *which* items were scored, in order.

    Canonical JSON per item (sorted keys), so a dict built in a different key
    order still fingerprints the same -- the identity of an item is its content,
    not the order its recorder happened to populate it. Order *between* items is
    significant, because pairing is positional.
    """

    if items is None:
        return None
    digest = hashlib.sha256()
    for item in items:
        digest.update(json.dumps(item, sort_keys=True, default=str).encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or \
            any(character not in "0123456789abcdef" for character in value.lower()):
        raise ScorecardError(f"{label} must be a 64-character sha256 hex digest, "
                             f"got {value!r}")
    return value.lower()


# ---------------------------------------------------------------- structure ---

@dataclass
class ArtifactRef:
    """What was scored: bytes (sha256) first, path second."""

    path: str
    sha256: str
    kind: str
    revision: Optional[str] = None   # Hub revision, when the artifact came from one
    config: Optional[str] = None     # PRESETS name, for checkpoints

    def to_dict(self) -> dict:
        if self.kind not in VALID_ARTIFACT_KINDS:
            raise ScorecardError(
                f"unknown artifact kind {self.kind!r}; "
                f"expected one of {sorted(VALID_ARTIFACT_KINDS)}")
        return {
            "path": self.path,
            "sha256": _require_sha256(self.sha256, f"{self.kind} sha256"),
            "kind": self.kind,
            "revision": self.revision,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ArtifactRef":
        return cls(
            path=payload["path"],
            sha256=payload["sha256"],
            kind=payload["kind"],
            revision=payload.get("revision"),
            config=payload.get("config"),
        )


@dataclass
class Provenance:
    """Everything a later reader needs to re-run or distrust this measurement."""

    artifact: ArtifactRef
    tokenizer: ArtifactRef
    seed: int
    git_sha: str
    bpb_mode: str = "not-applicable"
    bpb_sample_batches: Optional[int] = None
    task_revisions: Dict[str, str] = field(default_factory=dict)
    runtime: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        if self.bpb_mode not in VALID_BPB_MODES:
            raise ScorecardError(
                f"unknown bpb_mode {self.bpb_mode!r}; "
                f"expected one of {sorted(VALID_BPB_MODES)}")
        # A sampled BPB without its bound is unreproducible, and a "full" BPB
        # carrying a bound is a sampled one mislabelled. Both are refused rather
        # than normalised, because either would put a false claim in the record.
        if self.bpb_mode == "sample" and not (self.bpb_sample_batches or 0) > 0:
            raise ScorecardError(
                "bpb_mode 'sample' requires a positive bpb_sample_batches bound")
        if self.bpb_mode != "sample" and self.bpb_sample_batches is not None:
            raise ScorecardError(
                f"bpb_mode {self.bpb_mode!r} (not 'sample') must not carry "
                "bpb_sample_batches; a full pass has no batch bound")
        return {
            "artifact": self.artifact.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "seed": int(self.seed),
            "git_sha": self.git_sha,
            "bpb_mode": self.bpb_mode,
            "bpb_sample_batches": self.bpb_sample_batches,
            "task_revisions": dict(self.task_revisions),
            "runtime": dict(self.runtime),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Provenance":
        return cls(
            artifact=ArtifactRef.from_dict(payload["artifact"]),
            tokenizer=ArtifactRef.from_dict(payload["tokenizer"]),
            seed=payload["seed"],
            git_sha=payload["git_sha"],
            bpb_mode=payload.get("bpb_mode", "not-applicable"),
            bpb_sample_batches=payload.get("bpb_sample_batches"),
            task_revisions=dict(payload.get("task_revisions", {})),
            runtime=dict(payload.get("runtime", {})),
        )


@dataclass
class Scorecard:
    """One evaluation of one artifact, in the shape every gate reads."""

    kind: str
    name: str
    provenance: Provenance
    metrics: Dict[str, float]
    created_at: str
    items: Optional[List[dict]] = None
    item_count: Optional[int] = None
    details: Dict[str, object] = field(default_factory=dict)
    recorded_item_digest: Optional[str] = None
    """The digest read back from a scorecard whose items were not loaded.

    Set only by `from_dict`. Without it, re-serialising a scorecard read from
    disk without its sidecar would emit `item_digest: null` and quietly destroy
    the one field that lets a later gate prove two evaluations scored the same
    items -- the failure this whole module exists to prevent.
    """

    def resolved_item_digest(self) -> Optional[str]:
        return item_digest(self.items) if self.items is not None \
            else self.recorded_item_digest

    def resolved_item_count(self) -> int:
        if self.items is not None:
            return len(self.items)
        if self.item_count is None:
            raise ScorecardError(
                "a scorecard without items must state item_count explicitly")
        return int(self.item_count)

    def to_dict(self) -> dict:
        if self.kind not in VALID_KINDS:
            raise ScorecardError(
                f"unknown scorecard kind {self.kind!r}; "
                f"expected one of {sorted(VALID_KINDS)}")
        metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ScorecardError(f"metric {key!r} must be numeric, got {value!r}")
            if not math.isfinite(value):
                raise ScorecardError(f"metric {key!r} must be finite, got {value!r}")
            metrics[key] = float(value)
        return {
            "schema": SCORECARD_SCHEMA,
            "kind": self.kind,
            "name": self.name,
            "created_at": self.created_at,
            "provenance": self.provenance.to_dict(),
            "metrics": metrics,
            "item_count": self.resolved_item_count(),
            "item_digest": self.resolved_item_digest(),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: dict, items: Optional[List[dict]] = None) -> "Scorecard":
        return cls(
            kind=payload["kind"],
            name=payload["name"],
            provenance=Provenance.from_dict(payload["provenance"]),
            metrics=dict(payload["metrics"]),
            created_at=payload["created_at"],
            items=items if items is not None else payload.get("items"),
            item_count=payload.get("item_count"),
            details=dict(payload.get("details", {})),
            recorded_item_digest=payload.get("item_digest"),
        )


# ------------------------------------------------------------------ pairing ---

def paired_outcomes(left: Scorecard, right: Scorecard, *,
                    field: str = "correct") -> dict:
    """Compare two scorecards item by item, or refuse to compare them at all.

    The refusals are the point. Two evaluations are only pairable when they
    scored the same items in the same order; otherwise the honest error bar is
    the unpaired one, which is wide enough to swallow every margin this program
    decides on. Rather than silently degrading to that, this raises.

    Binary fields (0/1) yield the McNemar contingency counts a paired
    significance test needs; continuous fields yield per-item deltas.
    """

    if left.items is None or right.items is None:
        raise ScorecardError("paired comparison requires items on both scorecards")
    if left.resolved_item_count() != right.resolved_item_count():
        raise ScorecardError(
            f"item_count mismatch: {left.resolved_item_count()} vs "
            f"{right.resolved_item_count()}; these scorecards are not pairable")
    left_digest, right_digest = item_digest(left.items), item_digest(right.items)

    for card, name in ((left, "left"), (right, "right")):
        missing = [index for index, item in enumerate(card.items) if field not in item]
        if missing:
            raise ScorecardError(
                f"{name} scorecard items are missing field {field!r} "
                f"(first at index {missing[0]})")

    left_values = [item[field] for item in left.items]
    right_values = [item[field] for item in right.items]
    binary = all(value in (0, 1) for value in left_values + right_values)

    paired = {
        "n": len(left_values),
        "field": field,
        "left_item_digest": left_digest,
        "right_item_digest": right_digest,
    }

    if binary:
        # Identity is checked on the *whole* item, so a differing prompt or
        # depth is caught even when both sides happen to score 1.
        identity_left = item_digest([{k: v for k, v in item.items() if k != field}
                                     for item in left.items])
        identity_right = item_digest([{k: v for k, v in item.items() if k != field}
                                      for item in right.items])
        if identity_left != identity_right:
            raise ScorecardError(
                f"item_digest mismatch ({identity_left} vs {identity_right}); "
                "these scorecards scored different items and cannot be paired")
        both = sum(1 for a, b in zip(left_values, right_values) if a and b)
        left_only = sum(1 for a, b in zip(left_values, right_values) if a and not b)
        right_only = sum(1 for a, b in zip(left_values, right_values) if b and not a)
        paired.update({
            "both": both,
            "left_only": left_only,
            "right_only": right_only,
            "neither": len(left_values) - both - left_only - right_only,
            "delta": (sum(right_values) - sum(left_values)) / len(left_values),
        })
        return paired

    if left_digest != right_digest and \
            [item.get("id") for item in left.items] != \
            [item.get("id") for item in right.items]:
        raise ScorecardError(
            f"item_digest mismatch ({left_digest} vs {right_digest}); "
            "these scorecards scored different items and cannot be paired")
    deltas = [float(b) - float(a) for a, b in zip(left_values, right_values)]
    paired.update({
        "mean_left": sum(left_values) / len(left_values),
        "mean_right": sum(right_values) / len(right_values),
        "per_item_delta": deltas,
        "mean_delta": sum(deltas) / len(deltas),
    })
    return paired


# -------------------------------------------------------------------- gates ---

@dataclass(frozen=True)
class GateCheck:
    """One preregistered threshold, named before the measurement is seen."""

    metric: str
    comparator: str
    threshold: float


def evaluate_gates(card: Scorecard, checks: Sequence[GateCheck]) -> dict:
    """Apply preregistered checks, recording every measured value.

    A missing metric raises rather than failing quietly: a gate that "fails"
    because the evaluator never emitted its metric is a bug report, not a
    negative result, and the two must not look alike in the record.
    """

    payload = card.to_dict()
    metrics = payload["metrics"]
    results = []
    for check in checks:
        if check.comparator not in _COMPARATORS:
            raise ScorecardError(
                f"unknown comparator {check.comparator!r}; "
                f"expected one of {sorted(_COMPARATORS)}")
        if check.metric not in metrics:
            raise ScorecardError(
                f"gate references missing metric {check.metric!r}; "
                f"scorecard {card.name!r} has {sorted(metrics)}")
        measured = metrics[check.metric]
        results.append({
            "metric": check.metric,
            "comparator": check.comparator,
            "threshold": check.threshold,
            "measured": measured,
            "passed": bool(_COMPARATORS[check.comparator](measured, check.threshold)),
        })
    return {
        "scorecard": card.name,
        "kind": card.kind,
        "passed": all(result["passed"] for result in results),
        "checks": results,
    }


# ----------------------------------------------------------------------- io ---

def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def items_path_for(path) -> Path:
    """The sidecar path beside a scorecard: `<stem>.items.json`."""

    path = Path(path)
    return path.with_name(path.name[:-len(".json")] + ".items.json"
                          if path.name.endswith(".json") else path.name + ".items.json")


def write_scorecard(path, card: Scorecard) -> Dict[str, Path]:
    """Write the scorecard, and its per-item outcomes to a sidecar beside it."""

    path = Path(path)
    payload = card.to_dict()
    written = {"scorecard": path}
    if card.items is not None:
        sidecar = items_path_for(path)
        _write_json_atomic(sidecar, {
            "schema": SCORECARD_SCHEMA,
            "name": card.name,
            "kind": card.kind,
            "item_digest": payload["item_digest"],
            "item_count": payload["item_count"],
            "items": card.items,
        })
        written["items"] = sidecar
    _write_json_atomic(path, payload)
    return written


def load_scorecard(path) -> Scorecard:
    """Load a scorecard, reattaching and re-verifying its sidecar items."""

    path = Path(path)
    with path.open() as handle:
        payload = json.load(handle)
    items = None
    if payload.get("item_digest") is not None:
        sidecar = items_path_for(path)
        if not sidecar.exists():
            raise ScorecardError(
                f"{path} records item_digest {payload['item_digest']} but its "
                f"items sidecar {sidecar} is missing")
        with sidecar.open() as handle:
            items = json.load(handle)["items"]
        recomputed = item_digest(items)
        if recomputed != payload["item_digest"]:
            raise ScorecardError(
                f"item_digest mismatch for {path}: scorecard records "
                f"{payload['item_digest']}, sidecar hashes to {recomputed}")
    return Scorecard.from_dict(payload, items=items)
