"""Phase 4: train and compare three candidate V2 vocabularies.

The deliverable is a migration report, not a checkpoint. A tokenizer cannot be
transplanted into a trained model -- every embedding row and every output logit
is indexed by the vocabulary the model was trained under -- so nothing here
touches the released V1 weights or Daedalus-Code. What it produces is evidence
about whether a future from-scratch V2 should keep SmolLM2's 49,152 tokens or
adopt 24,576, 32,768 or 40,960.

Four decisions are worth stating up front, because each is a way this kind of
comparison usually goes wrong.

**Bits per byte, never per-token perplexity.** A larger vocabulary packs more
bytes into each token, so its per-token likelihood improves by construction
with no improvement in the model. `evaluate_candidate` raises on a perplexity
field rather than converting one, because the failure is silent and flatters
exactly the arm this phase expects to lose.

**Equal bytes and equal compute cannot both hold, so both are run.** Holding
the token budget fixed hands the larger vocabulary more *text*; holding the
byte budget fixed hands the smaller vocabulary more *steps*. Each protocol
biases in the opposite direction, so the pair brackets the answer and a
candidate that only wins under one of them is reported as exactly that.

**The rule is written before the numbers, and its digest is recorded.** The
selection rule comes from the phase brief verbatim; `rule_digest` hashes the
thresholds so a later reader can prove none of them moved after the results
landed. `write_preregistration` refuses to overwrite.

**The sample is source-balanced against the real mixture, not against
intuition.** Shares come from `dataprep.MIXTURE`, so the vocabulary is trained
on the distribution a V2 would actually see, with the one code slot expanded
into the program's planned Python-first multilingual distribution. Sources that
cannot fill their share -- `everyday-conversations` is 403k tokens against a 2%
slot, and the ungated Rust/Go/shell sample is small -- are recorded as
shortfalls rather than force-filled.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus.tokenizer_train import (  # noqa: E402
    CANDIDATE_VOCAB_SIZES,
    INCUMBENT_TOKENIZER,
    INCUMBENT_VOCAB_SIZE,
    RoundTripError,
    bytes_per_token,
    embedding_cost,
    identifier_fragmentation,
    kv_bytes_per_context_token,
    load_tokenizer,
    longest_tokens,
    special_token_isolation,
    throughput,
    train_bpe,
    verify_round_trip,
    whitespace_behaviour,
)


# ============================================================== the rule =====

RULE_TEXT = (
    "A candidate is selected only if no domain regresses bytes per token by "
    "more than 5%, code improves or ties, tiny-model BPB improves or stays "
    "within 0.5%, and projected embedding bytes fall materially."
)

MAX_DOMAIN_FERTILITY_REGRESSION_PCT = 5.0
MAX_CODE_FERTILITY_REGRESSION_PCT = 0.0          # "improves or ties"
MAX_TINY_BPB_REGRESSION_PCT = 0.5
# "Materially" needs a number or it is not preregistered. 5% of the incumbent's
# embedding bytes is the bar: below that the saving is smaller than the 6.5%
# spread between the two embedding quantization grids llama.cpp might pick
# (`runs/preflight/token-embd-quant-grid.md`), so it would not survive a
# packaging decision nobody in this program controls.
MIN_EMBEDDING_BYTE_REDUCTION_PCT = 5.0

# Gates compare decimals against floats produced by arithmetic on measured
# values, and binary floating point does not represent them. Far below any
# measurement's resolution, so it can rescue a boundary case and never move a
# bar. Same reasoning, and the same value, as `qat_recovery.GATE_EPSILON`.
GATE_EPSILON = 1e-9


def _at_most(value: float, limit: float) -> bool:
    return value <= limit + GATE_EPSILON


def rule_digest() -> str:
    """A digest over the rule text and every threshold that decides it."""
    payload = {
        "text": RULE_TEXT,
        "max_domain_fertility_regression_pct": MAX_DOMAIN_FERTILITY_REGRESSION_PCT,
        "max_code_fertility_regression_pct": MAX_CODE_FERTILITY_REGRESSION_PCT,
        "max_tiny_bpb_regression_pct": MAX_TINY_BPB_REGRESSION_PCT,
        "min_embedding_byte_reduction_pct": MIN_EMBEDDING_BYTE_REDUCTION_PCT,
        "candidates": list(CANDIDATE_VOCAB_SIZES),
        "incumbent": INCUMBENT_VOCAB_SIZE,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


class PreregistrationError(RuntimeError):
    """Raised when a plan would be overwritten after the fact."""


# ============================================================== the sample ===

@dataclass(frozen=True)
class SampleSource:
    """One slice of the raw-text sample, with where it comes from."""

    key: str
    domain: str            # general | math | technical | dialogue | code
    share: float           # of the whole raw sample
    kind: str              # "shards" | "parquet" | "stack-smol-xs"
    location: str = ""     # shard dir, HF repo, or language name
    config: dict = field(default_factory=dict)


# The natural-language backbone is `dataprep.MIXTURE`'s shares, so the
# vocabulary is trained on the distribution a V2 model would actually read
# rather than on a hand-picked balance. Restated here as literals rather than
# imported because Phase 7 is going to move MIXTURE, and a Phase 4 result whose
# sample silently changed underneath it would no longer be the result that was
# preregistered.
_NATURAL_SOURCES: Tuple[Tuple[str, str, float], ...] = (
    ("fineweb-edu", "general", 0.375),
    ("dclm-baseline", "general", 0.225),
    ("finepdfs-edu", "technical", 0.08),
    ("finephrase", "general", 0.07),
    ("finemath-3plus", "math", 0.03),
    ("infiwebmath-3plus", "math", 0.03),
    ("cosmopedia-v2", "general", 0.05),
    ("finewiki-en", "general", 0.03),
    ("everyday-conversations", "dialogue", 0.02),
)
CODE_SHARE = 0.09          # `stack-edu-python`'s slot in MIXTURE

# The program's fixed Python-first multilingual distribution, as the shares
# *within* the code slot: Python 55%, JS/TS 12%, C/C++ 10%, Rust 8%, Go 6%,
# Java 5%, shell/SQL/other 4%.
CODE_LANGUAGE_SHARES: Dict[str, float] = {
    "python": 0.55,
    "javascript": 0.08,
    "typescript": 0.04,
    "c": 0.05,
    "cpp": 0.05,
    "rust": 0.08,
    "go": 0.06,
    "java": 0.05,
    "shell": 0.02,
    "sql": 0.02,
}

_GITHUB_CODE = "codeparrot/github-code"
_GITHUB_CODE_REVISION = "refs/convert/parquet"
_STACK_SMOL_XS = "bigcode/the-stack-smol-xs"

# Where each language comes from. The per-language parquet directories are the
# efficient path -- one pass reads only that language -- and the repo already
# uses the same mechanism for Python (`dataprep.MIXTURE`'s note on why a row
# filter over the interleaved `default` config takes unbounded time). The four
# languages with no such directory fall back to an ungated per-language sample,
# topped up from the permissively-licensed `all-mit` slice.
_CODE_LOCATION: Dict[str, dict] = {
    "python": {"kind": "shards", "location": "stack-edu-python"},
    "javascript": {"kind": "parquet", "location": "JavaScript-mit"},
    "typescript": {"kind": "parquet", "location": "TypeScript-all"},
    "c": {"kind": "parquet", "location": "C-all"},
    "cpp": {"kind": "parquet", "location": "C++-all"},
    "java": {"kind": "parquet", "location": "Java-apache-2.0"},
    "rust": {"kind": "stack-smol-xs", "location": "rust", "topup": "Rust"},
    "go": {"kind": "stack-smol-xs", "location": "go", "topup": "GO"},
    "shell": {"kind": "stack-smol-xs", "location": "shell", "topup": "Shell"},
    "sql": {"kind": "stack-smol-xs", "location": "sql", "topup": "SQL"},
}

# Licences whose terms permit redistribution and derivative work without a
# copyleft obligation. Applied to the `-all` directories, which mix licences;
# the `-mit` and `-apache-2.0` directories are already filtered upstream.
PERMISSIVE_LICENSES = frozenset({
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense",
    "cc0-1.0", "0bsd", "zlib", "artistic-2.0",
})


def _build_sources() -> Tuple[SampleSource, ...]:
    sources = [SampleSource(key=key, domain=domain, share=share, kind="shards",
                            location=key)
               for key, domain, share in _NATURAL_SOURCES]
    for language, language_share in CODE_LANGUAGE_SHARES.items():
        spec = _CODE_LOCATION[language]
        sources.append(SampleSource(
            key=f"code-{language}", domain="code",
            share=CODE_SHARE * language_share,
            kind=spec["kind"], location=spec["location"],
            config={k: v for k, v in spec.items()
                    if k not in ("kind", "location")}))
    return tuple(sources)


SAMPLE_SOURCES: Tuple[SampleSource, ...] = _build_sources()
DOMAIN_OF_SOURCE: Dict[str, str] = {s.key: s.domain for s in SAMPLE_SOURCES}
DOMAINS = ("general", "math", "technical", "dialogue", "code")


def source_budgets(target_bytes: int) -> Dict[str, float]:
    """Planned bytes per source, by share."""
    return {s.key: target_bytes * s.share for s in SAMPLE_SOURCES}


def code_language_budgets(code_bytes: int) -> Dict[str, float]:
    """Planned bytes per code language, by the fixed program distribution."""
    return {language: code_bytes * share
            for language, share in CODE_LANGUAGE_SHARES.items()}


# ------------------------------------------------------------------ splits ---

# Disjoint by construction: the split is a pure function of the document's
# bytes, so the same text can never reach two splits even when two sources
# carry it. That is what makes the holdout genuinely held out from tokenizer
# training *and* from LM training, which a random shuffle would not guarantee
# across a cross-source duplicate.
_SPLIT_BUCKETS = (("holdout", 20), ("tokenizer-train", 340))   # of 1000


def split_for(text: str) -> str:
    bucket = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000
    for name, upper in _SPLIT_BUCKETS:
        if bucket < upper:
            return name
    return "lm-train"


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------- readers ----

MIN_DOCUMENT_BYTES = 64


def iter_shard_documents(shard_dir, tokenizer, *, eos_id: int = 0,
                         batch_documents: int = 512) -> Iterator[str]:
    """Recover documents from this box's packed uint16 shards.

    The corpus on disk is already tokenized, and byte-level BPE round-trips
    exactly, so decoding recovers the original bytes rather than an
    approximation of them. That keeps the whole natural-language sample
    offline, deterministic, and hashable against shard files that cannot
    change under it.

    Documents are split on the EOS separator within each shard. The fragment
    after a shard's last EOS is dropped: dense packing runs documents across
    shard boundaries, so that tail is half a document, and one truncated
    document per 100M tokens is not worth the bookkeeping to stitch.
    """
    import numpy as np

    shard_dir = Path(shard_dir)
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    for shard in manifest["shards"]:
        stream = np.memmap(shard_dir / shard["file"], dtype=np.uint16, mode="r")
        boundaries = np.flatnonzero(stream == eos_id)
        start, batch = 0, []
        for end in boundaries:
            piece = stream[start:int(end)]
            start = int(end) + 1
            if piece.size:
                batch.append(piece.tolist())
            if len(batch) >= batch_documents:
                yield from tokenizer.batch_decode(
                    batch, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False)
                batch = []
        if batch:
            yield from tokenizer.batch_decode(
                batch, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)


def iter_parquet_language(directory: str, *, license_filter: bool,
                          token: Optional[str] = None,
                          repo: str = _GITHUB_CODE) -> Iterator[Tuple[str, dict]]:
    """Stream one language directory of the parquet-converted code corpus."""
    from datasets import load_dataset

    dataset = load_dataset(
        repo, split="train", streaming=True, revision=_GITHUB_CODE_REVISION,
        data_files={"train": f"{directory}/partial-train/*.parquet"},
        token=token)
    for row in dataset:
        licence = (row.get("license") or "").lower()
        if license_filter and licence not in PERMISSIVE_LICENSES:
            continue
        code = row.get("code") or ""
        if code:
            yield code, {"license": licence, "repo_name": row.get("repo_name"),
                         "path": row.get("path")}


def iter_stack_smol_xs(language: str, token: Optional[str] = None
                       ) -> Iterator[Tuple[str, dict]]:
    """One ungated per-language sample of The Stack, read as JSON lines."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(_STACK_SMOL_XS, f"data/{language}/data.json",
                            repo_type="dataset", token=token)
    with open(local, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = row.get("content") or ""
            if content:
                yield content, {"license": ",".join(row.get("max_stars_repo_licenses")
                                                    or []) or None,
                                "repo_name": row.get("max_stars_repo_name"),
                                "path": row.get("max_stars_repo_path")}


def iter_all_mit_languages(languages: Sequence[str], *, needed: Dict[str, int],
                           token: Optional[str] = None,
                           max_scanned_bytes: int = 2_000_000_000,
                           max_seconds: float = 900.0
                           ) -> Iterator[Tuple[str, str, dict]]:
    """One bounded pass over the MIT-licensed slice, for languages with no
    per-language directory.

    Rust is 0.23% of that slice's bytes and Go 1.07% (measured on 20,000 rows),
    so filling a Rust quota by scanning is expensive and filling a Go one is
    not. The pass is bounded in both bytes and wall-clock rather than run to
    completion, and whatever it does not reach is recorded as a shortfall --
    the same treatment `everyday-conversations` already gets in the corpus
    notes, and for the same reason: a share that cannot be filled from
    permissively-licensed data is a fact about the data, not a number to
    manufacture.
    """
    from datasets import load_dataset

    wanted = {name: int(count) for name, count in needed.items() if count > 0}
    if not wanted:
        return
    dataset = load_dataset(
        _GITHUB_CODE, split="train", streaming=True,
        revision=_GITHUB_CODE_REVISION,
        data_files={"train": "all-mit/partial-train/*.parquet"}, token=token)

    started, scanned = time.monotonic(), 0
    for row in dataset:
        code = row.get("code") or ""
        scanned += len(code.encode("utf-8"))
        language = row.get("language")
        if language in wanted and code:
            payload = len(code.encode("utf-8"))
            wanted[language] -= payload
            if wanted[language] <= 0:
                del wanted[language]
            yield language, code, {"license": "mit",
                                   "repo_name": row.get("repo_name"),
                                   "path": row.get("path")}
            if not wanted:
                return
        if scanned >= max_scanned_bytes or \
                time.monotonic() - started >= max_seconds:
            return


# ------------------------------------------------------------- the builder ---

class SampleWriter:
    """Writes one source's documents into the three split files, with hashes.

    Every document contributes a row to a gzipped index carrying its SHA-256,
    its byte count and its split. That index *is* the immutable row-hash
    requirement: a later reader can re-derive the split assignment from the
    hash, check any document against it, and detect a sample that changed
    without anyone editing the manifest.
    """

    def __init__(self, root, key: str):
        self.root = Path(root)
        self.key = key
        self.handles = {}
        for split in ("tokenizer-train", "lm-train", "holdout"):
            path = self.root / split / f"{key}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handles[split] = path.open("w", encoding="utf-8")
        index_path = self.root / "rows" / f"{key}.jsonl.gz"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = gzip.open(index_path, "wt", encoding="utf-8")
        self.bytes_by_split = {s: 0 for s in self.handles}
        self.documents_by_split = {s: 0 for s in self.handles}
        self.duplicates = 0
        self.too_short = 0
        self._seen = set()
        self._rolling = hashlib.sha256()

    def add(self, text: str, provenance: Optional[dict] = None) -> int:
        """Returns the bytes accepted (0 when skipped)."""
        payload = text.encode("utf-8")
        if len(payload) < MIN_DOCUMENT_BYTES:
            self.too_short += 1
            return 0
        digest = document_hash(text)
        if digest in self._seen:
            self.duplicates += 1
            return 0
        self._seen.add(digest)
        split = split_for(text)
        self.handles[split].write(json.dumps({"text": text}) + "\n")
        self.bytes_by_split[split] += len(payload)
        self.documents_by_split[split] += 1
        row = {"sha256": digest, "bytes": len(payload), "split": split}
        if provenance:
            row.update({k: v for k, v in provenance.items() if v is not None})
        self.index.write(json.dumps(row, sort_keys=True) + "\n")
        self._rolling.update(digest.encode())
        return len(payload)

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_by_split.values())

    def close(self) -> dict:
        for handle in self.handles.values():
            handle.close()
        self.index.close()
        return {
            "key": self.key,
            "bytes_by_split": dict(self.bytes_by_split),
            "documents_by_split": dict(self.documents_by_split),
            "total_bytes": self.total_bytes,
            "duplicates_dropped": self.duplicates,
            "too_short_dropped": self.too_short,
            "row_digest": self._rolling.hexdigest()[:32],
        }


def _hf_token() -> Optional[str]:
    return (os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _shard_provenance(shard_dir) -> dict:
    from daedalus.scorecard import sha256_file

    shard_dir = Path(shard_dir)
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    return {
        "kind": "local-shards",
        "shard_dir": str(shard_dir),
        "source_dataset": manifest.get("source_dataset"),
        "total_tokens": manifest.get("total_tokens"),
        "shards": [
            {"file": shard["file"], "tokens": shard["tokens"],
             "sha256": sha256_file(shard_dir / shard["file"])}
            for shard in manifest["shards"][:4]
        ],
        "shards_hashed": min(4, len(manifest["shards"])),
        "shards_total": len(manifest["shards"]),
    }


def build_sample(*, shard_root, out_root, target_bytes: int,
                 topup_max_bytes: int = 2_000_000_000,
                 topup_max_seconds: float = 900.0, log=print) -> dict:
    """Materialize the whole raw-text sample and its provenance manifest."""
    from daedalus.data import get_tokenizer

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    token = _hf_token()
    budgets = source_budgets(target_bytes)
    incumbent = get_tokenizer(INCUMBENT_TOKENIZER)

    records: Dict[str, dict] = {}
    shortfall_languages: Dict[str, int] = {}
    writers: Dict[str, SampleWriter] = {}

    for source in SAMPLE_SOURCES:
        planned = int(budgets[source.key])
        writer = SampleWriter(out_root, source.key)
        writers[source.key] = writer
        started = time.monotonic()
        provenance: dict = {}

        if source.kind == "shards":
            shard_dir = Path(shard_root) / source.location
            if not (shard_dir / "manifest.json").exists():
                log(f"  {source.key}: no shards at {shard_dir}; recording a "
                    f"complete shortfall")
                records[source.key] = {**writer.close(), "planned_bytes": planned,
                                       "provenance": {"kind": "missing",
                                                      "shard_dir": str(shard_dir)},
                                       "domain": source.domain}
                continue
            provenance = _shard_provenance(shard_dir)
            for text in iter_shard_documents(shard_dir, incumbent):
                writer.add(text)
                if writer.total_bytes >= planned:
                    break

        elif source.kind == "parquet":
            directory = source.location
            permissive_directory = directory.endswith(("-mit", "-apache-2.0",
                                                       "-bsd-3-clause"))
            provenance = {"kind": "hf-parquet", "repo": _GITHUB_CODE,
                          "revision": _GITHUB_CODE_REVISION,
                          "directory": directory,
                          "license_filter": not permissive_directory}
            for text, row in iter_parquet_language(
                    directory, license_filter=not permissive_directory,
                    token=token):
                writer.add(text, row)
                if writer.total_bytes >= planned:
                    break

        elif source.kind == "stack-smol-xs":
            provenance = {"kind": "hf-stack-smol-xs", "repo": _STACK_SMOL_XS,
                          "language": source.location}
            try:
                for text, row in iter_stack_smol_xs(source.location, token=token):
                    writer.add(text, row)
                    if writer.total_bytes >= planned:
                        break
            except Exception as error:                      # noqa: BLE001
                log(f"  {source.key}: {type(error).__name__}: {error}")
                provenance["error"] = f"{type(error).__name__}: {error}"
            if writer.total_bytes < planned:
                shortfall_languages[source.config["topup"]] = \
                    planned - writer.total_bytes

        records[source.key] = {
            "planned_bytes": planned, "domain": source.domain,
            "provenance": provenance,
            "seconds": round(time.monotonic() - started, 1),
        }
        log(f"  {source.key:26s} {writer.total_bytes/1e6:8.2f} MB of "
            f"{planned/1e6:8.2f} MB planned "
            f"({100 * writer.total_bytes / planned if planned else 0:5.1f}%)")

    # One bounded top-up pass for the languages with no per-language directory.
    if shortfall_languages:
        log(f"  topping up {sorted(shortfall_languages)} from all-mit "
            f"(bounded: {topup_max_bytes/1e9:.1f} GB scanned or "
            f"{topup_max_seconds/60:.0f} min)")
        language_to_key = {source.config.get("topup"): source.key
                           for source in SAMPLE_SOURCES
                           if source.kind == "stack-smol-xs"}
        try:
            for language, text, row in iter_all_mit_languages(
                    list(shortfall_languages), needed=shortfall_languages,
                    token=token, max_scanned_bytes=topup_max_bytes,
                    max_seconds=topup_max_seconds):
                writers[language_to_key[language]].add(text, row)
        except Exception as error:                          # noqa: BLE001
            log(f"  top-up stopped: {type(error).__name__}: {error}")

    summary = {}
    for key, writer in writers.items():
        if key in records and "row_digest" in records[key]:
            summary[key] = records[key]
            continue
        summary[key] = {**writer.close(), **records[key]}
        summary[key]["fill_frac"] = (
            summary[key]["total_bytes"] / summary[key]["planned_bytes"]
            if summary[key]["planned_bytes"] else 0.0)

    manifest = {
        "schema": 1,
        "phase": "phase4-tokenizer-lab",
        "target_bytes": target_bytes,
        "min_document_bytes": MIN_DOCUMENT_BYTES,
        "split_buckets": {name: upper for name, upper in _SPLIT_BUCKETS},
        "code_language_shares": dict(CODE_LANGUAGE_SHARES),
        "code_share_of_sample": CODE_SHARE,
        "sources": summary,
        "totals": {
            "bytes": sum(s["total_bytes"] for s in summary.values()),
            "by_split": {
                split: sum(s["bytes_by_split"][split] for s in summary.values())
                for split in ("tokenizer-train", "lm-train", "holdout")},
            "by_domain": {
                domain: sum(s["total_bytes"] for key, s in summary.items()
                            if DOMAIN_OF_SOURCE[key] == domain)
                for domain in DOMAINS},
        },
        "shortfalls": {
            key: {"planned_bytes": s["planned_bytes"],
                  "achieved_bytes": s["total_bytes"],
                  "fill_frac": s.get("fill_frac", 0.0)}
            for key, s in summary.items()
            if s["total_bytes"] < 0.95 * max(1, s["planned_bytes"])},
    }
    (out_root / "sample-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_split(root, split: str, keys: Optional[Sequence[str]] = None
               ) -> Iterator[Tuple[str, str]]:
    """`(source_key, text)` for one split, in a fixed source order."""
    root = Path(root) / split
    for source in SAMPLE_SOURCES:
        if keys is not None and source.key not in keys:
            continue
        path = root / f"{source.key}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield source.key, json.loads(line)["text"]


def domain_texts(root, split: str) -> Dict[str, List[str]]:
    """The split grouped by domain, which is the unit the rule reads."""
    grouped: Dict[str, List[str]] = {domain: [] for domain in DOMAINS}
    for key, text in read_split(root, split):
        grouped[DOMAIN_OF_SOURCE[key]].append(text)
    return {domain: texts for domain, texts in grouped.items() if texts}


# ============================================================ probe arms =====

LM_PROBE_TOKENS = 200_000_000
PROBE_SEQ_LEN = 1024
PROBE_BATCH_TOKENS = 262_144
PROBE_MICRO_BATCH = 16
PROBE_MUON_LR = 0.02          # the shipped rate; these are from-scratch runs
PROBE_ADAM_LR = 3e-4
PROBE_WARMUP_STEPS = 100
PROBE_DECAY_FRAC = 0.8


def probe_config_name(vocab_size: int) -> str:
    from daedalus.config import tokenizer_probe_preset_name
    return tokenizer_probe_preset_name(vocab_size)


def equal_byte_budget(*, incumbent_bytes_per_token: float,
                      tokens: int = LM_PROBE_TOKENS) -> int:
    """The byte budget every arm reads under the byte-matched protocol.

    Derived from the incumbent because that is the arm every candidate is
    measured against: fixing the text at what SmolLM2 reads in `tokens` tokens
    keeps the incumbent's two protocol runs identical, so the pair of protocols
    costs seven runs rather than eight and the incumbent cannot drift between
    them.
    """
    return int(tokens * incumbent_bytes_per_token)


def tokens_for_byte_budget(byte_budget: int, *, bytes_per_token: float) -> int:
    """How many tokens one vocabulary needs to read a fixed number of bytes."""
    return int(byte_budget / bytes_per_token)


def probe_train_command(*, vocab_size: int, data_dir: str, total_tokens: int,
                        run_name: str, protocol: str,
                        val_dir: Optional[str] = None, device: str = "cuda",
                        python: str = "python",
                        no_compile: bool = True) -> List[str]:
    """One arm's `train.py` argv.

    Built as data so "every arm differs only in vocabulary" is checkable by a
    test rather than by reading seven shell lines. No `--init-from` and no
    `--resume`: these are from-scratch runs, and Phase 3 measured twice what
    `--resume` does to a budget it thinks is already spent.
    """
    if protocol not in ("equal-bytes", "equal-tokens"):
        raise ValueError(f"unknown protocol {protocol!r}")
    command = [
        python, "train.py",
        "--run-name", run_name,
        "--config", probe_config_name(vocab_size),
        "--data-dir", data_dir,
        "--total-tokens", str(total_tokens),
        "--micro-batch", str(PROBE_MICRO_BATCH),
        "--seq-start", str(PROBE_SEQ_LEN), "--seq-end", str(PROBE_SEQ_LEN),
        "--tok-start", str(PROBE_BATCH_TOKENS), "--tok-end", str(PROBE_BATCH_TOKENS),
        "--muon-lr", f"{PROBE_MUON_LR:g}",
        "--adam-lr", f"{PROBE_ADAM_LR:g}",
        "--warmup-steps", str(PROBE_WARMUP_STEPS),
        "--decay-frac", str(PROBE_DECAY_FRAC),
        "--device", device,
        "--hub-repo", "",
    ]
    if val_dir:
        command += ["--val-dir", val_dir]
    if no_compile:
        command += ["--no-compile"]
    return command


# ============================================================== decision =====

def evaluate_candidate(measured: dict) -> dict:
    """Apply the preregistered rule to one candidate, clause by clause.

    Every clause records the value that decided it, so the verdict is readable
    without the measurement files beside it.
    """
    for key in measured:
        if "perplexity" in key.lower():
            raise ValueError(
                f"{key!r}: per-token perplexity is not comparable across "
                f"vocabularies -- a larger vocabulary improves it by packing "
                f"more bytes into each token, with no improvement in the "
                f"model. This rule reads bits per byte only.")

    clauses: List[dict] = []

    round_trip = bool(measured.get("round_trip_passed"))
    clauses.append({
        "clause": "round-trip",
        "passed": round_trip,
        "detail": "arbitrary bytes reproduce exactly" if round_trip
                  else "candidate is rejected before measurement",
    })

    fertility = dict(measured.get("domain_fertility_delta_pct") or {})
    non_code = {domain: delta for domain, delta in fertility.items()
                if domain != "code"}
    worst_domain, worst_delta = (
        max(non_code.items(), key=lambda item: item[1]) if non_code
        else (None, float("nan")))
    clauses.append({
        "clause": "domain-fertility",
        "passed": bool(non_code) and _at_most(
            worst_delta, MAX_DOMAIN_FERTILITY_REGRESSION_PCT),
        "worst_domain": worst_domain,
        "worst_regression_pct": worst_delta,
        "limit_pct": MAX_DOMAIN_FERTILITY_REGRESSION_PCT,
    })

    code_delta = fertility.get("code")
    clauses.append({
        "clause": "code-fertility",
        "passed": code_delta is not None and _at_most(
            code_delta, MAX_CODE_FERTILITY_REGRESSION_PCT),
        "regression_pct": code_delta,
        "limit_pct": MAX_CODE_FERTILITY_REGRESSION_PCT,
    })

    bpb_delta = measured.get("tiny_bpb_delta_pct")
    clauses.append({
        "clause": "tiny-bpb",
        "passed": bpb_delta is not None and _at_most(
            bpb_delta, MAX_TINY_BPB_REGRESSION_PCT),
        "regression_pct": bpb_delta,
        "limit_pct": MAX_TINY_BPB_REGRESSION_PCT,
    })

    candidate_bytes = measured.get("embedding_q6_k_bytes")
    incumbent_bytes = measured.get("incumbent_embedding_q6_k_bytes")
    reduction = (100.0 * (incumbent_bytes - candidate_bytes) / incumbent_bytes
                 if candidate_bytes is not None and incumbent_bytes
                 else None)
    clauses.append({
        "clause": "embedding-bytes",
        "passed": reduction is not None and reduction >= (
            MIN_EMBEDDING_BYTE_REDUCTION_PCT - GATE_EPSILON),
        "reduction_pct": reduction,
        "bar_pct": MIN_EMBEDDING_BYTE_REDUCTION_PCT,
    })

    return {
        "vocab_size": measured.get("vocab_size"),
        "selectable": all(clause["passed"] for clause in clauses),
        "clauses": clauses,
        "measured": measured,
        "rule_digest": rule_digest(),
    }


def decide(verdicts: Sequence[dict]) -> dict:
    """Pick among selectable candidates, or record a negative result.

    Ranked by tiny-model BPB first because that is the only clause measuring
    *model quality* rather than artifact shape; the rest are floors, and a
    floor is satisfied or not, not maximized.
    """
    selectable = [v for v in verdicts if v.get("selectable")]
    if not selectable:
        return {
            "selected": None,
            "negative_result": True,
            "reason": ("no candidate cleared every clause of the preregistered "
                       "rule; recording the negative result rather than "
                       "relaxing a bar after seeing the numbers"),
            "rule_digest": rule_digest(),
            "candidates": [
                {"vocab_size": v.get("vocab_size"), "selectable": False,
                 "failed": [c["clause"] for c in v["clauses"] if not c["passed"]]}
                for v in verdicts],
        }

    def rank(verdict: dict):
        measured = verdict.get("measured") or {}
        return (
            measured.get("tiny_bpb_delta_pct", float("inf")),
            -(measured.get("incumbent_embedding_q6_k_bytes", 0.0)
              - measured.get("embedding_q6_k_bytes", 0.0)),
            (measured.get("domain_fertility_delta_pct") or {}).get(
                "code", float("inf")),
            verdict.get("vocab_size") or 0,
        )

    winner = sorted(selectable, key=rank)[0]
    return {
        "selected": winner.get("vocab_size"),
        "negative_result": False,
        "reason": (f"{winner.get('vocab_size')} cleared every clause and has "
                   f"the best tiny-model BPB among those that did"),
        "rule_digest": rule_digest(),
        "candidates": [
            {"vocab_size": v.get("vocab_size"),
             "selectable": bool(v.get("selectable")),
             "failed": [c["clause"] for c in v["clauses"] if not c["passed"]]}
            for v in verdicts],
    }


# ======================================================== preregistration ====

def build_preregistration(*, sample_target_bytes: int, incumbent: str) -> dict:
    return {
        "schema": 1,
        "phase": "phase4-tokenizer-lab",
        "rule": {
            "text": RULE_TEXT,
            "digest": rule_digest(),
            "thresholds": {
                "max_domain_fertility_regression_pct":
                    MAX_DOMAIN_FERTILITY_REGRESSION_PCT,
                "max_code_fertility_regression_pct":
                    MAX_CODE_FERTILITY_REGRESSION_PCT,
                "max_tiny_bpb_regression_pct": MAX_TINY_BPB_REGRESSION_PCT,
                "min_embedding_byte_reduction_pct":
                    MIN_EMBEDDING_BYTE_REDUCTION_PCT,
            },
        },
        "candidates": list(CANDIDATE_VOCAB_SIZES),
        "incumbent": {"name": incumbent, "vocab_size": INCUMBENT_VOCAB_SIZE},
        "expected_winner": 32768,
        "expected_winner_note": (
            "stated before measuring so a result that matches it is not read "
            "as confirmation of anything; the measured gate decides"),
        "comparison": {
            "metric": "bits-per-byte",
            "refused": ("per-token perplexity, which is not comparable across "
                        "vocabularies and would make the largest vocabulary "
                        "look best by construction"),
            "protocols": {
                "equal-bytes": ("every arm reads the same text; step counts "
                                "differ because a better vocabulary needs "
                                "fewer tokens for it"),
                "equal-tokens": ("every arm takes the same steps on the same "
                                 "batch shape; bytes read differ because a "
                                 "larger vocabulary packs more into a token"),
                "why_both": ("the two cannot hold at once and bias in opposite "
                             "directions, so a candidate that wins under only "
                             "one of them is reported as exactly that"),
            },
        },
        "sample": {
            "target_bytes": sample_target_bytes,
            "shares": {source.key: source.share for source in SAMPLE_SOURCES},
            "domains": DOMAIN_OF_SOURCE,
            "code_language_shares": dict(CODE_LANGUAGE_SHARES),
            "splits": "disjoint by sha256 of the document's bytes",
        },
        "probe": {
            "config": {vocab: probe_config_name(vocab)
                       for vocab in list(CANDIDATE_VOCAB_SIZES)
                       + [INCUMBENT_VOCAB_SIZE]},
            "tokens": LM_PROBE_TOKENS,
            "seq_len": PROBE_SEQ_LEN,
            "batch_tokens": PROBE_BATCH_TOKENS,
            "muon_lr": PROBE_MUON_LR,
            "adam_lr": PROBE_ADAM_LR,
            "seed": 0,
        },
        "scope": (
            "V2 only. A tokenizer cannot be transplanted into a trained model, "
            "so nothing here changes released V1 weights or Daedalus-Code."),
    }


def write_preregistration(path, payload: dict, *, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise PreregistrationError(
            f"{path} already exists; a preregistration is written once, before "
            f"the first measurement. Delete it deliberately if the plan "
            f"genuinely changed before anything was measured.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


# ============================================================ measurement ====

# Identifiers drawn from the sample rather than invented, so fragmentation is
# measured on names people actually write. Collected once and pinned, because a
# fragmentation number over a different word list is not comparable.
def collect_identifiers(root, limit: int = 4000) -> List[str]:
    """Distinct snake_case / camelCase identifiers from the code holdout."""
    import re

    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]{4,40}")
    keywords = {"return", "import", "class", "while", "print", "static",
                "public", "private", "const", "function", "define", "include",
                "struct", "template", "namespace", "package", "extends"}
    seen: Dict[str, None] = {}
    for key, text in read_split(root, "holdout"):
        if DOMAIN_OF_SOURCE[key] != "code":
            continue
        for match in pattern.findall(text):
            if match in keywords or match.isupper():
                continue
            if "_" in match or (match != match.lower() and
                                match != match.capitalize()):
                seen.setdefault(match, None)
                if len(seen) >= limit:
                    return list(seen)
    return list(seen)


def measure_tokenizer(tokenizer, *, name: str, vocab_size: int, root,
                      identifiers: Sequence[str], hidden_size: int = 768
                      ) -> dict:
    """Every intrinsic reading for one vocabulary, on the held-out split."""
    round_trip = verify_round_trip(tokenizer)          # raises on failure
    domains = domain_texts(root, "holdout")
    fertility = bytes_per_token(tokenizer, domains)
    sample = [text for _key, text in read_split(root, "holdout")][:400]
    return {
        "name": name,
        "vocab_size": vocab_size,
        "round_trip": round_trip,
        "fertility": fertility,
        "identifier_fragmentation": identifier_fragmentation(tokenizer,
                                                             identifiers),
        "whitespace": whitespace_behaviour(tokenizer),
        "special_token_isolation": special_token_isolation(tokenizer),
        "longest_tokens": longest_tokens(tokenizer, n=20),
        "throughput": throughput(tokenizer, sample),
        "embedding": embedding_cost(vocab_size, hidden_size),
        "kv": kv_bytes_per_context_token(vocab_size),
    }


def fertility_deltas(candidate: dict, incumbent: dict) -> Dict[str, float]:
    """Per-domain bytes-per-token regression, in percent.

    Positive is worse: a candidate whose bytes-per-token *falls* needs more
    tokens for the same text, which is the regression the rule bounds.
    """
    deltas = {}
    for domain, values in candidate["fertility"].items():
        if domain == "__all__" or domain not in incumbent["fertility"]:
            continue
        base = incumbent["fertility"][domain]["bytes_per_token"]
        deltas[domain] = 100.0 * (base - values["bytes_per_token"]) / base
    return deltas


# --------------------------------------------------------------------- cli ---

def _git_short_sha() -> str:
    import subprocess
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip() or "unknown"
    except Exception:                                       # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/tokenizer-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preregister", help="write the rule before measuring")
    pre.add_argument("--sample-target-bytes", type=int, default=1_240_000_000)
    pre.add_argument("--force", action="store_true")

    sample = sub.add_parser("sample", help="build the raw-text sample")
    sample.add_argument("--shard-root", default="data/shards")
    sample.add_argument("--out", default="data/tokenizer-lab/sample")
    sample.add_argument("--target-bytes", type=int, default=1_240_000_000)
    sample.add_argument("--topup-max-bytes", type=int, default=2_000_000_000)
    sample.add_argument("--topup-max-seconds", type=float, default=900.0)

    train = sub.add_parser("train", help="train the candidate vocabularies")
    train.add_argument("--sample", default="data/tokenizer-lab/sample")
    train.add_argument("--out", default="data/tokenizer-lab/tokenizers")
    train.add_argument("--vocab-size", type=int, action="append", default=[])

    measure = sub.add_parser("measure", help="intrinsic readings on the holdout")
    measure.add_argument("--sample", default="data/tokenizer-lab/sample")
    measure.add_argument("--tokenizers", default="data/tokenizer-lab/tokenizers")

    args = parser.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.command == "preregister":
        payload = build_preregistration(
            sample_target_bytes=args.sample_target_bytes,
            incumbent=INCUMBENT_TOKENIZER)
        payload["git_sha"] = _git_short_sha()
        written = write_preregistration(root / "preregistration.json", payload,
                                        force=args.force)
        print(f"preregistered {len(payload['candidates'])} candidates in {written}")
        print(f"rule digest: {payload['rule']['digest']}")
        return 0

    if args.command == "sample":
        manifest = build_sample(
            shard_root=args.shard_root, out_root=args.out,
            target_bytes=args.target_bytes,
            topup_max_bytes=args.topup_max_bytes,
            topup_max_seconds=args.topup_max_seconds)
        print(json.dumps({"totals": manifest["totals"],
                          "shortfalls": manifest["shortfalls"]},
                         indent=2, sort_keys=True))
        return 0

    if args.command == "train":
        sizes = args.vocab_size or list(CANDIDATE_VOCAB_SIZES)
        from export import CHATML_TEMPLATE
        out_root = Path(args.out)
        written = {}
        for size in sizes:
            destination = out_root / f"v{size}"
            print(f"training {size}-token vocabulary -> {destination}",
                  flush=True)
            started = time.monotonic()
            train_bpe((text for _key, text in read_split(args.sample,
                                                         "tokenizer-train")),
                      vocab_size=size, out_dir=destination,
                      show_progress=False, chat_template=CHATML_TEMPLATE)
            tokenizer = load_tokenizer(destination)
            verify_round_trip(tokenizer)
            written[size] = {"path": str(destination),
                             "seconds": round(time.monotonic() - started, 1)}
            print(f"  done in {written[size]['seconds']}s", flush=True)
        (root / "trained.json").write_text(
            json.dumps(written, indent=2, sort_keys=True) + "\n")
        print(json.dumps(written, indent=2, sort_keys=True))
        return 0

    if args.command == "measure":
        identifiers = collect_identifiers(args.sample)
        print(f"measuring against {len(identifiers)} held-out identifiers",
              flush=True)
        readings = {}
        for size in list(CANDIDATE_VOCAB_SIZES) + [INCUMBENT_VOCAB_SIZE]:
            path = (INCUMBENT_TOKENIZER if size == INCUMBENT_VOCAB_SIZE
                    else str(Path(args.tokenizers) / f"v{size}"))
            print(f"  {size}: {path}", flush=True)
            readings[str(size)] = measure_tokenizer(
                load_tokenizer(path), name=path, vocab_size=size,
                root=args.sample, identifiers=identifiers)
        out = root / "measurements.json"
        out.write_text(json.dumps(readings, indent=2, sort_keys=True) + "\n")
        incumbent = readings[str(INCUMBENT_VOCAB_SIZE)]
        for size in CANDIDATE_VOCAB_SIZES:
            deltas = fertility_deltas(readings[str(size)], incumbent)
            print(f"  {size}: " + "  ".join(
                f"{domain} {delta:+.2f}%" for domain, delta in sorted(deltas.items())))
        print(f"wrote {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
