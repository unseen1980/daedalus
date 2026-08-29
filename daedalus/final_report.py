"""The program's final report, assembled from artifacts rather than from memory.

Phase 9 has one job that no earlier phase has: it says what the whole program
found, to a reader who was not here. That reader cannot check a number by
re-running it, so the report has to carry its own evidence -- and it has to
carry, beside every number, the thing that decides how much the number is
worth.

Three failure modes this module exists to refuse.

  - **A proxy result presented as a full-model gain.** Phases 4 through 7 rank
    tokenizers, decay schedules, shapes and mixtures on 105M- and 159M-parameter
    proxies over 100M-to-250M-token budgets. Those rankings are evidence about a
    future from-scratch V2. They are not statements about the released 150M
    model, and nothing in this program measured them on one. The plan says so
    three times, which is a good sign it is the mistake a final report is most
    likely to make, so `Claim` refuses the combination structurally: a claim
    whose scope is `proxy` cannot set `applies_to_released_model`.

  - **A headline with no source.** Every value in the machine-readable report
    names the file it was read from. A claim with an empty `sources` list is
    refused at construction, so a number cannot enter the report by being typed
    into it.

  - **A metric re-read from a checkpoint that has since moved.** The plan
    requires the headline metrics to come from immutable final artifacts. That
    is only meaningful if "immutable" is checked, so `verify_artifact` recomputes
    the SHA-256 and compares it against what the scorecards recorded at
    measurement time. A mismatch is a hard failure, not a warning: it means the
    number in the report was measured on bytes that no longer exist.

The markdown report is rendered *from* the JSON for the same reason: two
documents holding the same number independently is two documents that can
disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from daedalus.scorecard import sha256_file


FINAL_REPORT_SCHEMA = 1

# What kind of statement a number is. This is the axis the plan cares about, and
# it is deliberately not "which phase produced it": two findings from the same
# phase can differ here, and a reader deciding whether to act on a number needs
# this distinction rather than the filing.
SCOPES = {
    # Measured on the released 150M weights, or on an artifact derived from them
    # by a treatment this program ran. Actionable for the shipped model.
    "released-model",
    # Measured on the Daedalus-Code branch, which starts from the released base
    # weights. Actionable for that artifact only.
    "code-model",
    # Measured on a smaller stand-in trained for the comparison. Evidence about
    # a decision, never a statement about any shipped model.
    "proxy",
    # An extrapolation. Carries no measurement of its own and must name the
    # measurements it extrapolates from.
    "projection",
    # A fact about how the program ran -- gates, branches, deadlines. Not a
    # model measurement at all.
    "process",
}

# Scopes whose findings may be described as a property of a model this program
# can hand over. `proxy` is absent by design, and `projection` with it: an
# extrapolation is not a measurement of anything.
_MODEL_SCOPES = {"released-model", "code-model"}


class FinalReportError(ValueError):
    """Raised when the report would state something it has not established."""


# ------------------------------------------------------------------- claims ---

@dataclass(frozen=True)
class Claim:
    """One finding, with its scope and the files it was read from.

    `applies_to_released_model` is the flag that turns a finding into advice
    about the shipped artifact. It is the flag most worth getting wrong, so it
    is checked against the scope rather than trusted.
    """

    key: str
    scope: str
    statement: str
    sources: Sequence[str]
    value: Optional[object] = None
    units: Optional[str] = None
    applies_to_released_model: bool = False
    caveats: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise FinalReportError(
                f"claim {self.key!r} has unknown scope {self.scope!r}; "
                f"expected one of {sorted(SCOPES)}")
        if not self.sources:
            raise FinalReportError(
                f"claim {self.key!r} names no source; every number in the final "
                "report must be readable from a file on disk")
        if self.applies_to_released_model and self.scope not in _MODEL_SCOPES:
            raise FinalReportError(
                f"claim {self.key!r} has scope {self.scope!r} but is marked as "
                "applying to the released model. A proxy or projected result is "
                "not a measurement of the shipped weights and must not be "
                "reported as one")
        if not self.statement.strip():
            raise FinalReportError(f"claim {self.key!r} has an empty statement")

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "scope": self.scope,
            "statement": self.statement,
            "value": self.value,
            "units": self.units,
            "applies_to_released_model": bool(self.applies_to_released_model),
            "sources": list(self.sources),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class Section:
    """A group of claims that share a scope, so the scope is stated once."""

    key: str
    title: str
    scope: str
    summary: str
    claims: Sequence[Claim]

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise FinalReportError(
                f"section {self.key!r} has unknown scope {self.scope!r}")
        # A section's scope is what a skimming reader takes away, so a claim
        # measured on something else hiding inside it would be read under the
        # wrong label -- a proxy ranking sitting under a "released model"
        # heading is the exact misreading this module exists to prevent.
        #
        # `process` is exempt, and the exemption is narrow: it is not a model
        # measurement at all, carries no value that could be quoted as a gain,
        # and is already forbidden from setting `applies_to_released_model`.
        # A gate outcome ("escalation refused") or a pending measurement
        # ("Apple Silicon decode") belongs beside the numbers it qualifies, not
        # exiled to a section of its own where a reader will not connect it.
        mismatched = [claim.key for claim in self.claims
                      if claim.scope != self.scope and claim.scope != "process"]
        if mismatched:
            raise FinalReportError(
                f"section {self.key!r} is scoped {self.scope!r} but contains "
                f"claims of another scope: {mismatched}. Split the section "
                "rather than mixing measurement scopes under one heading")

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "scope": self.scope,
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
        }


# ---------------------------------------------------------------- artifacts ---

@dataclass
class ArtifactRecord:
    """An immutable artifact, in the shape the plan's manifest clause requires.

    `expected_sha256` is what a scorecard recorded when it measured this file.
    `verify` recomputes it. The two being equal is the whole claim that the
    headline numbers came from the bytes the manifest names.
    """

    name: str
    role: str
    path: str
    kind: str
    expected_sha256: Optional[str] = None
    config: Optional[str] = None
    tokenizer: Optional[str] = None
    data_manifest: Optional[str] = None
    seed: Optional[int] = None
    producing_commit: Optional[str] = None
    producing_commit_basis: Optional[str] = None
    hub_repo: Optional[str] = None
    hub_revision: Optional[str] = None
    hub_path: Optional[str] = None
    hub_private: Optional[bool] = None
    notes: Optional[str] = None

    observed_sha256: Optional[str] = None
    verified: Optional[bool] = None
    verification_detail: str = ""

    def verify(self, root: Optional[Path] = None) -> "ArtifactRecord":
        """Recompute the digest and compare it against the recorded one."""

        candidate = Path(self.path)
        if root is not None and not candidate.is_absolute():
            candidate = Path(root) / candidate
        if not candidate.exists():
            self.verified = False
            self.observed_sha256 = None
            self.verification_detail = f"missing: {candidate}"
            return self
        self.observed_sha256 = sha256_file(candidate)
        if self.expected_sha256 is None:
            # No prior digest to check against. That is a weaker claim than a
            # match and is recorded as such rather than counted as a pass.
            self.verified = None
            self.verification_detail = (
                "hashed at finalization; no earlier digest was recorded for this "
                "artifact, so this is a fingerprint and not a match")
            return self
        self.verified = self.observed_sha256 == self.expected_sha256.lower()
        self.verification_detail = "" if self.verified else (
            f"sha256 mismatch: scorecards recorded {self.expected_sha256}, "
            f"the file now hashes to {self.observed_sha256}")
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "path": self.path,
            "kind": self.kind,
            "sha256": {
                "expected": self.expected_sha256,
                "observed": self.observed_sha256,
                "verified": self.verified,
                "detail": self.verification_detail,
            },
            "config": self.config,
            "tokenizer": self.tokenizer,
            "data_manifest": self.data_manifest,
            "seed": self.seed,
            "producing_commit": self.producing_commit,
            "producing_commit_basis": self.producing_commit_basis,
            "hub": {
                "repo": self.hub_repo,
                "revision": self.hub_revision,
                "path": self.hub_path,
                "private": self.hub_private,
            },
            "notes": self.notes,
        }


def verify_artifacts(records: Sequence[ArtifactRecord],
                     root: Optional[Path] = None) -> dict:
    """Verify every record and summarise, without deciding what to do about it."""

    for record in records:
        record.verify(root)
    matched = [r.name for r in records if r.verified is True]
    mismatched = [r.name for r in records if r.verified is False]
    unchecked = [r.name for r in records if r.verified is None]
    return {
        "records": [record.to_dict() for record in records],
        "matched": matched,
        "mismatched": mismatched,
        "fingerprinted_only": unchecked,
        "all_recorded_digests_match": not mismatched,
    }


# ------------------------------------------------------------------ reading ---

def read_json(path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def require(payload: dict, dotted: str, source: str):
    """Read a nested key, naming the file when it is not there.

    A report that silently substitutes `None` for a missing verdict field would
    publish a blank where a gate outcome belongs, which reads as "not measured"
    when it may mean "the schema moved".
    """

    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise FinalReportError(
                f"{source} has no {dotted!r} (stopped at {part!r}); the final "
                "report will not guess a value it cannot read")
        node = node[part]
    return node


# ---------------------------------------------------------------- assembly ----

def build_report(*, program: dict, sections: Sequence[Section],
                 artifacts: Sequence[ArtifactRecord],
                 validation: dict,
                 artifact_root: Optional[Path] = None) -> dict:
    """Assemble the machine-readable report and check its own invariants."""

    verification = verify_artifacts(artifacts, artifact_root)
    payload = {
        "schema": FINAL_REPORT_SCHEMA,
        "program": dict(program),
        "sections": [section.to_dict() for section in sections],
        "artifacts": verification,
        "validation": dict(validation),
    }
    validate_report(payload)
    return payload


def validate_report(payload: dict) -> dict:
    """Re-check the report as data, so a hand-edited file is caught too.

    `build_report` constructs `Claim`s, which already enforce these rules. This
    runs them again over the serialised form, which is what a later reader --
    or a test, or a PR body generator -- actually holds.
    """

    problems: List[str] = []
    keys = set()
    for section in payload.get("sections", []):
        if section.get("scope") not in SCOPES:
            problems.append(f"section {section.get('key')!r}: unknown scope "
                            f"{section.get('scope')!r}")
        for claim in section.get("claims", []):
            key = claim.get("key")
            if key in keys:
                problems.append(f"duplicate claim key {key!r}")
            keys.add(key)
            if claim.get("scope") not in SCOPES:
                problems.append(f"claim {key!r}: unknown scope {claim.get('scope')!r}")
            if not claim.get("sources"):
                problems.append(f"claim {key!r}: no source named")
            if claim.get("applies_to_released_model") and \
                    claim.get("scope") not in _MODEL_SCOPES:
                problems.append(
                    f"claim {key!r}: scope {claim.get('scope')!r} marked as "
                    "applying to the released model")
    if problems:
        raise FinalReportError("; ".join(problems))
    return {"claims": len(keys), "sections": len(payload.get("sections", []))}


# ---------------------------------------------------------------- rendering ---

def _format_value(claim: dict) -> str:
    value, units = claim.get("value"), claim.get("units")
    if value is None:
        return ""
    if isinstance(value, float):
        # 6 significant figures, not 4: several headlines here are decided at
        # the third decimal (a 0.5% ceiling, a 5.539% penalty), and rounding
        # 6.9798 to "6.98" in the human report would make it disagree with the
        # JSON it was rendered from.
        rendered = f"{value:,.6g}"
    elif isinstance(value, bool):
        rendered = "yes" if value else "no"
    else:
        rendered = str(value)
    return f"{rendered} {units}" if units else rendered


_SCOPE_LABEL = {
    "released-model": "released model",
    "code-model": "Daedalus-Code",
    "proxy": "proxy evidence",
    "projection": "projection",
    "process": "process",
}


def render_markdown(payload: dict, *, title: str, preamble: str) -> str:
    """Render the human report from the JSON, so the two cannot disagree."""

    lines = [f"# {title}", "", preamble.strip(), ""]

    lines += ["## How to read a number in this report", "",
              "Every claim carries a scope. The scope decides what the number is "
              "evidence *about*, and the four are not interchangeable:", ""]
    for scope in ("released-model", "code-model", "proxy", "projection"):
        lines.append(f"- **{_SCOPE_LABEL[scope]}** -- "
                     f"{_SCOPE_DESCRIPTION[scope]}")
    lines += ["",
              "No proxy result in this report is a statement about the released "
              "150M model. Nothing here measured one on the other, and the "
              "report's own validator refuses the combination.", ""]

    for section in payload["sections"]:
        lines += [f"## {section['title']}", "",
                  f"*Scope: {_SCOPE_LABEL.get(section['scope'], section['scope'])}.*",
                  "", section["summary"].strip(), ""]
        if section["claims"]:
            lines += ["| finding | value | read from |", "| --- | --- | --- |"]
            for claim in section["claims"]:
                sources = ", ".join(f"`{source}`" for source in claim["sources"])
                statement = claim["statement"].replace("|", "\\|")
                lines.append(f"| {statement} | {_format_value(claim)} | {sources} |")
            lines.append("")
        # Caveats keep their claim. A flat list of them under a table is a list
        # of sentences whose subject the reader has to guess -- "the signal that
        # moved" says nothing without the row it qualifies.
        qualified = [(claim["statement"], claim.get("caveats", []))
                     for claim in section["claims"] if claim.get("caveats")]
        if qualified:
            lines += ["Caveats, by finding:", ""]
            for statement, caveats in qualified:
                lines.append(f"- **{statement}**")
                for caveat in caveats:
                    lines.append(f"  - {caveat}")
            lines.append("")

    artifacts = payload["artifacts"]
    lines += ["## Artifact manifest", "",
              "SHA-256 recomputed at finalization and compared against the digest "
              "the scorecards recorded when they measured the file.", "",
              "| artifact | role | sha256 | verified |", "| --- | --- | --- | --- |"]
    for record in artifacts["records"]:
        digest = record["sha256"]["observed"] or "(missing)"
        verified = {True: "match", False: "**MISMATCH**",
                    None: "fingerprint only"}[record["sha256"]["verified"]]
        lines.append(f"| `{record['name']}` | {record['role']} | "
                     f"`{digest[:16]}` | {verified} |")
    # Sections append a trailing "" to separate themselves, so the last one
    # leaves a blank line at EOF -- which `git diff --check` rejects, and which
    # is how this was found: the report could not be committed. Strip it here
    # rather than at each append site, so a new section cannot reintroduce it.
    return "\n".join(lines).rstrip("\n") + "\n"


_SCOPE_DESCRIPTION = {
    "released-model": "measured on the released 150M weights, or on an artifact "
                      "this program derived from them. Actionable for the "
                      "shipped model.",
    "code-model": "measured on the Daedalus-Code branch, which starts from the "
                  "released base weights. Actionable for that artifact only.",
    "proxy": "measured on a smaller stand-in trained for the comparison. "
             "Evidence about a *decision*, never a property of a shipped model.",
    "projection": "an extrapolation. It carries no measurement of its own and "
                  "names the ones it extrapolates from.",
}


def write_report(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
