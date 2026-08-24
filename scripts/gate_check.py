"""Decide the Phase 2 gate from what was written, not from what was claimed.

Every later phase reads this gate. Phase 3 spends a 1B-token budget on the
strength of "the evaluators are trustworthy", and Phase 8 compares a code model
against baselines these same evaluators produced. A gate that is asserted in
prose by the session that also wrote the evaluators is not evidence for either.

So each criterion here is decided by reading the scorecards on disk, or by
running the containment it claims, and every verdict carries the observed value
that decided it and the paths it read. The exit status is the verdict: non-zero
when any criterion fails, so a controller can gate on it without parsing prose.

Four criteria, matching the plan's Phase 2 gate:

  - **synthetic controls** scored 100% where the control expects it. The oracle
    backend answers each item from its own prompt, so anything below 1.0 is a
    malformed item, and no model number measured against those items means
    anything.
  - **determinism** -- two runs of the same evaluation at temperature zero
    produced identical per-item outcomes and an identical scorecard fingerprint.
    Aggregate equality is not enough: two runs can agree on a mean while
    disagreeing on which items they got right.
  - **paired-quant identity** -- the FP16 and Q4_0 comparison covered the same
    items in the same order. A quantization penalty is a difference between two
    measurements, and the gates that read it are written at 3% and 1%; pairing
    against different items would produce a confident, meaningless number.
  - **code sandbox** -- executed code reaches neither the network nor files
    outside its working directory. This one is *run*, not read: the claim is
    about behaviour, and the previous version of this sandbox passed a
    read-the-scorecard check while a candidate could still shell out to `curl`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus.scorecard import (  # noqa: E402
    ScorecardError,
    load_scorecard,
    paired_outcomes,
)


GATE_SCHEMA = 1

# Fields that differ between two identical measurements and say nothing about
# whether they measured the same thing.
_VOLATILE = ("created_at",)
_VOLATILE_PROVENANCE = ("git_sha",)


def _verdict(criterion: str, passed: bool, observed: dict,
             evidence: Sequence[str], detail: str = "") -> dict:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "observed": observed,
        "evidence": [str(path) for path in evidence],
        "detail": detail,
    }


def scorecard_fingerprint(path) -> str:
    """A digest of everything about a scorecard except when it was written."""

    with Path(path).open() as handle:
        payload = json.load(handle)
    payload = {key: value for key, value in payload.items() if key not in _VOLATILE}
    provenance = dict(payload.get("provenance", {}))
    for key in _VOLATILE_PROVENANCE:
        provenance.pop(key, None)
    payload["provenance"] = provenance
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


# ------------------------------------------------------------ the criteria ---

def control_verdict(paths: Sequence[str]) -> dict:
    """Oracle-backed scorecards must be perfect; anything else is a broken item."""

    observed: Dict[str, float] = {}
    failures: List[str] = []
    checked: List[str] = []
    for path in paths:
        card = load_scorecard(path)
        backend = card.provenance.runtime.get("backend")
        if backend != "oracle":
            # Not a control. A model's copy-control score is a measurement, and
            # demanding 1.0 of it would turn a weak model into a gate failure.
            continue
        checked.append(str(path))
        measured = float(card.metrics.get("exact_match", float("nan")))
        observed[card.name] = measured
        if measured != 1.0:
            failures.append(f"{card.name} scored {measured}")

    if not checked:
        return _verdict(
            "synthetic-controls", False, observed, paths,
            "no oracle-backed scorecard was supplied; the control gate cannot "
            "be decided by a model-backed run")
    return _verdict("synthetic-controls", not failures, observed, checked,
                    "; ".join(failures))


def determinism_verdict(pairs: Sequence[Sequence[str]]) -> dict:
    """Repeated evaluation must reproduce every item, not just the aggregate."""

    observed: Dict[str, dict] = {}
    failures: List[str] = []
    evidence: List[str] = []
    for first, second in pairs:
        left, right = load_scorecard(first), load_scorecard(second)
        evidence += [str(first), str(second)]
        left_print, right_print = (scorecard_fingerprint(first),
                                   scorecard_fingerprint(second))
        entry = {
            "item_digest": [left.resolved_item_digest(),
                            right.resolved_item_digest()],
            "item_count": [left.resolved_item_count(),
                           right.resolved_item_count()],
            "fingerprint": [left_print, right_print],
        }
        observed[f"{left.name}"] = entry
        if left.resolved_item_digest() != right.resolved_item_digest():
            failures.append(f"{left.name}: per-item outcomes differ")
        if left.resolved_item_count() != right.resolved_item_count():
            failures.append(f"{left.name}: item counts differ")
        if left_print != right_print:
            failures.append(f"{left.name}: scorecard fingerprints differ")

    if not pairs:
        return _verdict("determinism", False, observed, evidence,
                        "no repeated evaluation was supplied")
    return _verdict("determinism", not failures, observed, evidence,
                    "; ".join(failures))


def paired_quant_verdict(fp16_path: str, quantized_path: str) -> dict:
    """The two precisions must have scored the same items in the same order."""

    evidence = [fp16_path, quantized_path]
    left, right = load_scorecard(fp16_path), load_scorecard(quantized_path)
    observed = {
        "item_count": [left.resolved_item_count(), right.resolved_item_count()],
        "item_digest": [left.resolved_item_digest(),
                        right.resolved_item_digest()],
        "artifact_kind": [left.provenance.artifact.kind,
                          right.provenance.artifact.kind],
    }
    try:
        # `paired_outcomes` is the same refusal the comparison itself uses, so
        # the gate cannot pass a pairing the report would have rejected.
        paired = paired_outcomes(left, right, field="nll")
    except ScorecardError as error:
        return _verdict("paired-quant-identity", False, observed, evidence,
                        str(error))
    observed["n"] = paired["n"]
    ids_match = ([item.get("id") for item in left.items]
                 == [item.get("id") for item in right.items])
    observed["ids_in_same_order"] = ids_match
    return _verdict("paired-quant-identity", ids_match, observed, evidence,
                    "" if ids_match else "item ids differ or are reordered")


# Three probes, one per way out of the sandbox that the socket patch missed.
_SANDBOX_PROBES = (
    ("network_client",
     "import subprocess\n"
     "def probe():\n"
     "    subprocess.run(['curl', '-s', 'https://example.com'],\n"
     "                   capture_output=True)\n",
     {"process_blocked"}),
    ("raw_socket",
     "import socket\n"
     "def probe():\n"
     "    socket.create_connection(('example.com', 80), timeout=1)\n",
     {"network_blocked", "exception"}),
    ("os_system",
     "import os\n"
     "def probe():\n"
     "    os.system('curl -s https://example.com')\n",
     {"process_blocked"}),
)


def sandbox_verdict(secret_path: Optional[str] = None,
                    protected_dir: Optional[str] = None) -> dict:
    """Run the containment rather than reading a claim about it."""

    from scripts.code_eval import run_in_sandbox, unprivileged_ids

    observed: Dict[str, object] = {}
    failures: List[str] = []

    ids = unprivileged_ids()
    observed["privilege_drop"] = "not-root" if ids is None else list(ids)

    for name, solution, allowed in _SANDBOX_PROBES:
        verdict = run_in_sandbox(solution, "probe()\n", timeout_s=15.0)
        observed[name] = verdict["category"]
        if verdict["status"] != "failed" or verdict["category"] not in allowed:
            failures.append(
                f"{name}: expected refusal in {sorted(allowed)}, got "
                f"{verdict['status']}/{verdict['category']}")

    # Reading a root-owned secret and writing outside the working directory are
    # only meaningful where the parent is root and there is a privilege to drop.
    if ids is not None and secret_path:
        verdict = run_in_sandbox(
            f"def probe():\n    open({secret_path!r}).read()\n",
            "probe()\n", timeout_s=15.0)
        observed["secret_read"] = verdict["category"]
        if "PermissionError" not in str(verdict["detail"]):
            failures.append(
                f"secret_read: expected PermissionError, got {verdict['detail'][:200]}")
    if ids is not None and protected_dir:
        target = str(Path(protected_dir) / "planted.txt")
        verdict = run_in_sandbox(
            f"def probe():\n    open({target!r}, 'w').write('x')\n",
            "probe()\n", timeout_s=15.0)
        observed["outside_write"] = verdict["category"]
        if "PermissionError" not in str(verdict["detail"]):
            failures.append(
                f"outside_write: expected PermissionError, got {verdict['detail'][:200]}")
        if Path(target).exists():
            failures.append(f"outside_write: {target} was created")

    return _verdict("code-sandbox", not failures, observed,
                    ["scripts/code_eval.py"], "; ".join(failures))


# --------------------------------------------------------------------- run ---

def run_gate(*, controls: Sequence[str], determinism: Sequence[Sequence[str]],
             paired_quant: Optional[Sequence[str]],
             sandbox: bool = True, secret_path: Optional[str] = None,
             protected_dir: Optional[str] = None) -> dict:
    criteria = [control_verdict(controls), determinism_verdict(determinism)]
    if paired_quant:
        criteria.append(paired_quant_verdict(*paired_quant))
    else:
        criteria.append(_verdict("paired-quant-identity", False, {}, [],
                                 "no FP16/Q4_0 pair was supplied"))
    if sandbox:
        criteria.append(sandbox_verdict(secret_path, protected_dir))
    return {
        "schema": GATE_SCHEMA,
        "gate": "phase2-evaluation",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": all(item["passed"] for item in criteria),
        "criteria": criteria,
    }


def write_verdict(path, verdict: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="append", default=[],
                        help="scorecard from an oracle-backed control run; "
                             "repeatable")
    parser.add_argument("--repeat", action="append", nargs=2, default=[],
                        metavar=("FIRST", "SECOND"),
                        help="two scorecards from repeated runs of one "
                             "evaluation; repeatable")
    parser.add_argument("--paired-quant", nargs=2, default=None,
                        metavar=("FP16", "QUANTIZED"))
    parser.add_argument("--no-sandbox", action="store_true",
                        help="skip the containment probes (they execute code)")
    parser.add_argument("--secret-path", default=None,
                        help="a root-owned mode-0600 file the sandbox must not "
                             "be able to read")
    parser.add_argument("--protected-dir", default=None,
                        help="a root-owned directory the sandbox must not be "
                             "able to write into")
    parser.add_argument("--out", default="runs/eval/phase2-gate.json")
    args = parser.parse_args(argv)

    verdict = run_gate(
        controls=args.control, determinism=args.repeat,
        paired_quant=args.paired_quant, sandbox=not args.no_sandbox,
        secret_path=args.secret_path, protected_dir=args.protected_dir)
    write_verdict(args.out, verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    for item in verdict["criteria"]:
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['criterion']}"
              f"{': ' + item['detail'] if item['detail'] else ''}",
              file=sys.stderr)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
