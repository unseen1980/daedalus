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
    round_trip_report,
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
# 131,072 tokens per step is ~1,526 steps at the probe budget. The shipped
# 512k-token batch would leave 381, of which a 100-step warmup is a quarter --
# a schedule short enough that the comparison would partly be a comparison of
# how each vocabulary handles warmup.
PROBE_BATCH_TOKENS = 131_072
# 16, not 32: measured on this box, micro-batch 32 uncompiled peaked at 20.8 GB
# of 24 and ran at 49.0k tok/s, while 16 compiled peaks at 9.2 GB and runs at
# 91.4k. The step is the same 131,072 tokens either way -- only how many
# accumulation passes it takes changes -- so this is throughput and headroom,
# not a different experiment.
PROBE_MICRO_BATCH = 16
PROBE_MUON_LR = 0.02          # the shipped rate; these are from-scratch runs
PROBE_ADAM_LR = 3e-4
PROBE_WARMUP_STEPS = 100
PROBE_DECAY_FRAC = 0.8


def probe_config_name(vocab_size: int) -> str:
    from daedalus.config import tokenizer_probe_preset_name
    return tokenizer_probe_preset_name(vocab_size)


def equal_byte_budget(*, max_bytes_per_token: float,
                      tokens: int = LM_PROBE_TOKENS,
                      margin: float = 1.01) -> int:
    """The byte budget every arm packs, shared by both protocols.

    Derived from the **least token-efficient** arm, which is the only choice
    that lets one shard set serve both protocols. Under equal-bytes each arm
    trains on all the tokens this many bytes produced for it; under
    equal-tokens each trains on `tokens` of them. The second only works if
    every arm packed at least `tokens`, and a vocabulary that packs more bytes
    per token produces fewer of them -- so the budget has to be sized for the
    worst case.

    Sizing it from the incumbent instead (the obvious choice, since it is the
    comparison arm) leaves the three arms with higher bytes-per-token short of
    the equal-tokens budget. Measured on this sample that is 195.5M, 192.1M and
    189.8M tokens against a 200M budget: not an error, just three arms quietly
    re-reading data the others saw once.

    `margin` covers two ways an exact budget lands just under. Truncation is
    the small one: `int(200e6 * 4.2906) / 4.2906` is 199,999,999. The large one
    is that bytes-per-token is measured on the **holdout** while the budget is
    spent on **LM-train**, and those are different text. Measured here, every
    arm's LM-train fertility ran 2.8-3.4% above its holdout fertility, so a 1%
    margin left two arms short of the token budget -- 198.95M and 196.58M
    against 200M.

    A margin large enough to absorb that would have exceeded the LM-train split
    itself, so `pack` spends the whole split instead and uses this only as the
    floor a split has to clear to be usable at all. The default margin stays
    small deliberately: it is a rounding allowance, not a substitute for
    knowing the fertility of the text being spent.
    """
    return math.ceil(tokens * max_bytes_per_token * margin)


def tokens_for_byte_budget(byte_budget: int, *, bytes_per_token: float) -> int:
    """How many tokens one vocabulary needs to read a fixed number of bytes."""
    return int(byte_budget / bytes_per_token)


def probe_train_command(*, vocab_size: int, data_dir: str, total_tokens: int,
                        run_name: str, protocol: str,
                        val_dir: Optional[str] = None, device: str = "cuda",
                        python: str = "python",
                        no_compile: bool = False) -> List[str]:
    """One arm's `train.py` argv.

    Built as data so "every arm differs only in vocabulary" is checkable by a
    test rather than by reading seven shell lines. No `--init-from` and no
    `--resume`: these are from-scratch runs, and Phase 3 measured twice what
    `--resume` does to a budget it thinks is already spent.

    Compiled, unlike Phase 3's recovery arms. Those disabled compile because
    QAT's fake-quant parametrization interacts with it
    (`runs/preflight/qat-compile-lattice.md`); these probes run no QAT, and
    compile is worth 1.86x here -- 91.4k tok/s against 49.0k, measured on this
    box at this shape, which is the difference between a six-hour arm sweep and
    an eleven-hour one.
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


# ================================================================ packing ====

SHARD_TOKENS = 100_000_000
# Whole batches to the Rust tokenizer instead of one document per call: 4.9
# MB/s single-threaded against 24 cores otherwise sitting idle.
PACK_BATCH_DOCUMENTS = 1000


def pack_for_tokenizer(*, sample_root, tokenizer_path: str, vocab_size: int,
                       label: str, out_root, byte_budget: int,
                       log=print) -> dict:
    """Tokenize the shared LM-train prefix and the whole holdout, one vocabulary.

    Every arm packs *the same documents*: `read_split` walks sources in a fixed
    order and the byte budget is identical, so the prefix is byte-identical
    across arms and the token counts that come out are the whole difference
    between them. That is what makes the byte-matched protocol a controlled
    comparison rather than four runs on four corpora.

    Three holdout layouts are written because three questions are asked of it:
    `holdout-all` is one flat directory whose BPB is a single number directly
    comparable across arms; `holdout/<domain>` gives the per-domain breakdown
    the report needs; and both carry the tokenizer fingerprint, so a later pass
    that points the wrong vocabulary at them is refused rather than silently
    scored.
    """
    from daedalus.data import get_tokenizer, tokenize_and_pack

    tokenizer = get_tokenizer(tokenizer_path)
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(
            f"{tokenizer_path} has {tokenizer.vocab_size} tokens, not "
            f"{vocab_size}; packing under it would produce ids the "
            f"{vocab_size}-row embedding cannot index")

    # Keyed by label, not by vocabulary size: `49152-matched` and
    # `49152-smollm2` share a size and are different tokenizers, so a
    # size-keyed directory would have one silently overwrite the other.
    out_root = Path(out_root) / label
    summary: Dict[str, dict] = {}

    def budgeted(split: str, budget: Optional[int]) -> Iterator[str]:
        spent = 0
        for _key, text in read_split(sample_root, split):
            yield text
            spent += len(text.encode("utf-8"))
            if budget is not None and spent >= budget:
                return

    started = time.monotonic()
    summary["train"] = tokenize_and_pack(
        tokenizer, budgeted("lm-train", byte_budget), str(out_root / "train"),
        shard_tokens=SHARD_TOKENS, tokenizer_name=tokenizer_path,
        batch_documents=PACK_BATCH_DOCUMENTS,
        manifest_extra={"split": "lm-train", "byte_budget": byte_budget})
    log(f"  {label} train: {summary['train']['total_tokens']:,} tokens "
        f"from {byte_budget/1e6:.1f} MB "
        f"({byte_budget / summary['train']['total_tokens']:.4f} bytes/token) "
        f"in {time.monotonic() - started:.0f}s")

    summary["holdout-all"] = tokenize_and_pack(
        tokenizer, (text for _key, text in read_split(sample_root, "holdout")),
        str(out_root / "holdout-all"), shard_tokens=SHARD_TOKENS,
        tokenizer_name=tokenizer_path, batch_documents=PACK_BATCH_DOCUMENTS,
        manifest_extra={"split": "holdout"})

    per_domain = {}
    for domain, texts in domain_texts(sample_root, "holdout").items():
        per_domain[domain] = tokenize_and_pack(
            tokenizer, texts, str(out_root / "holdout" / domain),
            shard_tokens=SHARD_TOKENS, tokenizer_name=tokenizer_path,
            batch_documents=PACK_BATCH_DOCUMENTS,
            manifest_extra={"split": "holdout", "domain": domain})
    summary["holdout-by-domain"] = {
        domain: record["total_tokens"] for domain, record in per_domain.items()}

    summary["vocab_size"] = vocab_size
    summary["label"] = label
    summary["tokenizer"] = tokenizer_path
    summary["byte_budget"] = byte_budget
    summary["measured_bytes_per_token"] = (
        byte_budget / summary["train"]["total_tokens"])
    (out_root / "pack-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


# ================================================================= probing ===

def probe_run_name(label: str, protocol: str) -> str:
    return f"tok-probe-{label}-{protocol}"


def launch_probe(*, vocab_size: int, label: str, protocol: str, shard_root,
                 tokenizer_path: str, device: str = "cuda",
                 run_root: str = "runs", max_attempts: int = 3,
                 stall_min: float = 20.0) -> dict:
    """Run one arm under the existing watchdog + resume supervisor.

    Token budget by protocol, and the asymmetry is the point:

    - `equal-tokens` gives every arm `LM_PROBE_TOKENS`, so every arm takes the
      same steps at the same batch shape and does the same non-embedding work.
      The arms then differ in how much *text* that buys, which favours the
      larger vocabulary.
    - `equal-bytes` gives each arm exactly the tokens its own vocabulary
      produced from the shared byte prefix, so every arm reads the same text.
      The arms then differ in step count, which favours the smaller vocabulary.

    Neither is neutral, so both are run and a candidate that wins under only
    one of them is reported as exactly that.
    """
    from daedalus.supervise import run_with_resume, start_watchdog, stop_watchdog

    shard_dir = Path(shard_root) / label / "train"
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    packed_tokens = int(manifest["total_tokens"])
    total_tokens = (packed_tokens if protocol == "equal-bytes"
                    else LM_PROBE_TOKENS)
    if protocol == "equal-tokens" and packed_tokens < LM_PROBE_TOKENS:
        raise ValueError(
            f"{label} packed only {packed_tokens:,} tokens, below the "
            f"{LM_PROBE_TOKENS:,} the equal-tokens protocol asks for; the arm "
            f"would silently repeat data the others do not")

    name = probe_run_name(label, protocol)
    command = probe_train_command(
        vocab_size=vocab_size, data_dir=str(shard_dir),
        total_tokens=total_tokens, run_name=name, protocol=protocol,
        val_dir=str(Path(shard_root) / label / "holdout-all"),
        device=device)
    command += ["--tokenizer", tokenizer_path]

    run_dir = Path(run_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog = start_watchdog(name, str(run_dir), total_tokens,
                              stall_min=stall_min, supervised=True)
    try:
        report = run_with_resume(
            list(command), str(run_dir / "checkpoint.pt"),
            max_attempts=max_attempts, halt_marker=str(run_dir / "HALTED"),
            inflight_extra={"phase": "phase4-tokenizer-lab", "label": label,
                            "vocab_size": vocab_size, "protocol": protocol,
                            "total_tokens": total_tokens})
    finally:
        stop_watchdog(watchdog)
    return {"run": name, "protocol": protocol, "label": label,
            "vocab_size": vocab_size, "total_tokens": total_tokens,
            "packed_tokens": packed_tokens, "command": list(command), **report}


# ================================================================= scoring ===

def score_probe(*, vocab_size: int, label: str, protocol: str, shard_root,
                tokenizer_path: str, run_root: str = "runs",
                device: str = "cuda", batch_size: int = 8) -> dict:
    """Held-out bits per byte for one arm, pooled and by domain.

    Pooled BPB comes from one flat holdout directory rather than from averaging
    the per-domain numbers: averaging bits-per-byte over domains weights each
    domain by nothing in particular, while the flat pass weights every domain
    by the bytes it actually contributes -- and those bytes are identical
    across arms, which is what makes the arms comparable at all.
    """
    from daedalus.config import PRESETS
    from daedalus.data import get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb
    from train import load_checkpoint

    name = probe_run_name(label, protocol)
    checkpoint = Path(run_root) / name / "checkpoint.pt"
    tokenizer = get_tokenizer(tokenizer_path)
    model = Daedalus(PRESETS[probe_config_name(vocab_size)]).to(device)
    info = load_checkpoint(str(checkpoint), model, map_location=device)
    model.eval()

    root = Path(shard_root) / label

    def bpb(directory: Path) -> Tuple[Optional[float], dict]:
        """BPB over one holdout directory, with the shape that produced it.

        The batch size is capped at the number of non-overlapping windows the
        directory actually holds. `make_loader` drops the last partial batch,
        so a directory with fewer windows than one batch yields *no* batches at
        all and `evaluate_bpb` returns NaN -- which is not a score, serialises
        as invalid JSON, and would reach the report as a number. The dialogue
        holdout is exactly that case: `everyday-conversations` is 403k tokens in
        the whole corpus, so its 2% holdout is ~7k tokens, six windows at
        `seq_len` 1024. Batching is only how the windows are grouped, so
        lowering it changes the arithmetic not at all.
        """
        manifest = json.loads((directory / "manifest.json").read_text())
        tokens = int(manifest["total_tokens"])
        windows = max(0, (tokens - PROBE_SEQ_LEN) // PROBE_SEQ_LEN + 1)
        shape = {"tokens": tokens, "windows": windows,
                 "batch_size": min(batch_size, windows)}
        if windows == 0:
            shape["skipped"] = (
                f"{tokens:,} tokens is under one {PROBE_SEQ_LEN}-token window")
            return None, shape
        return evaluate_bpb(model, str(directory), PROBE_SEQ_LEN, tokenizer,
                            device, batch_size=shape["batch_size"],
                            max_batches=None), shape

    by_domain, shapes = {}, {}
    for directory in sorted((root / "holdout").iterdir()):
        if not (directory / "manifest.json").exists():
            continue
        value, shape = bpb(directory)
        shapes[directory.name] = shape
        if value is not None:
            by_domain[directory.name] = value

    pooled, pooled_shape = bpb(root / "holdout-all")
    return {
        "vocab_size": vocab_size,
        "label": label,
        "protocol": protocol,
        "run": name,
        "checkpoint": str(checkpoint),
        "steps": info.get("step"),
        "tokens_seen": info.get("tokens_seen"),
        "bpb": pooled,
        "bpb_by_domain": by_domain,
        # Every domain's holdout size travels with its score, so a reader can
        # see that dialogue rests on six windows and general on ~4,800.
        "holdout_shape": {**shapes, "__all__": pooled_shape},
    }


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


# Both protocols bias, in opposite directions, so a candidate that clears the
# BPB clause under only the favourable one has not shown an improvement. The
# rule therefore reads the *worse* of the two. This is a decision the original
# preregistration left open ("reported as exactly that"), so it is recorded as
# an addendum -- and `write_addendum` refuses to write once any arm has been
# scored, which is what makes it a decision taken before the numbers rather
# than a reading chosen to suit them.
BPB_PROTOCOL_RULE = (
    "the tiny-BPB clause is evaluated against the worse (larger) regression of "
    "the equal-bytes and equal-tokens protocols; both are recorded")


def build_addendum() -> dict:
    return {
        "schema": 1,
        "phase": "phase4-tokenizer-lab",
        "rule_digest": rule_digest(),
        "bpb_protocol_rule": BPB_PROTOCOL_RULE,
        "matched_control": (
            "a 49,152-token vocabulary trained on the same sample is measured "
            "alongside SmolLM2. The preregistered rule is evaluated against "
            "SmolLM2 exactly as written; the matched control is a diagnostic "
            "column that separates 'what does shrinking the vocabulary cost' "
            "from 'what does retraining on this corpus buy', which the "
            "incumbent comparison conflates."),
        "incumbent_round_trip": (
            "SmolLM2 is missing 21 of the 256 byte-level characters and cannot "
            "round-trip code points U+40000-U+FFFFF. Candidates are still held "
            "to the full round-trip precondition; the incumbent is measured "
            "and its failure recorded, because refusing to measure the "
            "reference would leave the phase with nothing to compare against."),
    }


def write_addendum(root, payload: dict, *, force: bool = False) -> Path:
    """Write the addendum only while no arm has been scored."""
    root = Path(root)
    scored = sorted((root / "scored").glob("*.json"))
    if scored and not force:
        raise PreregistrationError(
            f"{len(scored)} arm(s) already scored ({scored[0].name}, ...); an "
            f"addendum written now could have been chosen to suit them")
    return write_preregistration(root / "addendum.json", payload, force=True)


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

INCUMBENT_KEY = "49152-smollm2"
MATCHED_CONTROL_KEY = "49152-matched"


def measurement_targets(tokenizer_root) -> List[Tuple[str, int, str]]:
    """`(label, vocab_size, path)` for everything measured, in report order.

    Two references, not one, and the second is not decoration.

    The preregistered rule compares candidates to **SmolLM2**, which is what a
    V2 would actually be replacing. But SmolLM2 was trained on a different
    corpus, so any tokenizer retrained on this sample beats it partly for
    reasons that have nothing to do with vocabulary size -- measured here as an
    8-14% bytes-per-token gain on maths, a domain SmolLM2's own mixture
    weighted differently. Read alone, that comparison credits *size* for a
    *corpus-match* effect.

    `49152-matched` is the same 49,152 tokens trained on the same sample with
    the same trainer, so candidate-vs-matched isolates the one variable the
    phase is actually about. It is a diagnostic column: the rule is evaluated
    against the incumbent exactly as written, and adding this changes no
    threshold.
    """
    tokenizer_root = Path(tokenizer_root)
    targets = [(str(size), size, str(tokenizer_root / f"v{size}"))
               for size in CANDIDATE_VOCAB_SIZES]
    matched = tokenizer_root / f"v{INCUMBENT_VOCAB_SIZE}"
    if (matched / "tokenizer.json").exists():
        targets.append((MATCHED_CONTROL_KEY, INCUMBENT_VOCAB_SIZE, str(matched)))
    targets.append((INCUMBENT_KEY, INCUMBENT_VOCAB_SIZE, INCUMBENT_TOKENIZER))
    return targets

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
                      identifiers: Sequence[str], hidden_size: int = 768,
                      require_round_trip: bool = True) -> dict:
    """Every intrinsic reading for one vocabulary, on the held-out split.

    `require_round_trip=False` is used for the *incumbent* only, and it is not
    an exemption granted to make the reference pass. SmolLM2's shipped
    vocabulary is genuinely missing 21 of the 256 byte-level characters, so
    three of the four candidate tokenizers would be held to a bar the artifact
    they are compared against does not clear. Refusing to measure the incumbent
    would leave the phase with no reference at all; measuring it and recording
    the failure is what puts the difference in the report, where it belongs.
    """
    round_trip = (verify_round_trip(tokenizer) if require_round_trip
                  else round_trip_report(tokenizer))
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


# ------------------------------------------------------ stock llama.cpp -----

def check_stock_gguf_conversion(*, label: str, vocab_size: int,
                                tokenizer_path: str, out_dir,
                                llama_cpp_dir: Optional[str] = None) -> dict:
    """Can stock llama.cpp convert a model carrying this vocabulary?

    This is the constraint that decides whether any of this is actionable,
    because "unmodified stock llama.cpp" is a fixed program decision.
    `conversion/base.py::get_vocab_base_pre` identifies a BPE pre-tokenizer by
    hashing the token **ids** a fixed probe string encodes to, compares that
    against a hard-coded list, and raises `NotImplementedError` for anything it
    does not recognise. The hash is over ids, so it moves with the vocabulary
    and the merges -- a newly trained tokenizer cannot match a registered hash
    however faithfully it copies SmolLM2's pre-tokenizer configuration.

    Run rather than argued: the model is randomly initialized because the
    converter reads config.json, the tensor shapes and the tokenizer files, and
    nothing about the weights, so an untrained model answers the question at no
    training cost.
    """
    import subprocess

    from export import export_hf_model, export_tokenizer

    out_dir = Path(out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = out_dir / "hf"

    # A checkpoint in the shape `export_hf_model` reads, weights untrained.
    import torch

    from daedalus.config import PRESETS
    from daedalus.model import Daedalus

    config_name = probe_config_name(vocab_size)
    checkpoint = out_dir / "checkpoint.pt"
    torch.save({"model": Daedalus(PRESETS[config_name]).state_dict(),
                "step": 0, "tokens_seen": 0,
                "config": {"vocab_size": vocab_size}}, checkpoint)

    export_hf_model(str(checkpoint), config_name, str(hf_dir))
    export_tokenizer(str(hf_dir), tokenizer=tokenizer_path,
                     expected_vocab_size=vocab_size)

    llama_cpp_dir = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR",
                                                    "/opt/llama.cpp")
    converter = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not converter.exists():
        return {"label": label, "converter": str(converter),
                "ran": False, "reason": "converter not found"}

    result = subprocess.run(
        [sys.executable, str(converter), str(hf_dir), "--outfile",
         str(out_dir / "model-f16.gguf"), "--outtype", "f16"],
        capture_output=True, text=True)
    combined = result.stdout + result.stderr
    return {
        "label": label,
        "vocab_size": vocab_size,
        "converter": str(converter),
        "ran": True,
        "returncode": result.returncode,
        "converted": result.returncode == 0,
        "pre_tokenizer_unrecognized":
            "BPE pre-tokenizer was not recognized" in combined,
        "chkhsh": next((line.split("chkhsh:")[1].strip()
                        for line in combined.splitlines() if "chkhsh:" in line),
                       None),
        "tail": combined.strip().splitlines()[-12:],
    }


def sweep_order(tokenizer_root, include_matched: bool = False
                ) -> List[Tuple[str, str]]:
    """`(label, protocol)` arms, most decision-relevant first.

    Ordered so an interrupted sweep still answers the question it was run for.
    Every equal-bytes arm comes first because that set alone decides the
    preregistered rule; the equal-tokens set is the promised bracket; the
    size-matched control is diagnostic and last. At ~37 minutes an arm that is
    not an academic distinction.
    """
    rule_labels = [str(size) for size in CANDIDATE_VOCAB_SIZES] + [INCUMBENT_KEY]
    arms = [(label, "equal-bytes") for label in rule_labels]
    arms += [(label, "equal-tokens") for label in rule_labels]
    if include_matched and any(
            label == MATCHED_CONTROL_KEY
            for label, _size, _path in measurement_targets(tokenizer_root)):
        arms += [(MATCHED_CONTROL_KEY, "equal-bytes"),
                 (MATCHED_CONTROL_KEY, "equal-tokens")]
    return arms


def run_sweep(*, root, shard_root, tokenizer_root, device: str = "cuda",
              include_matched: bool = False, skip_existing: bool = True,
              score_only: bool = False, log=print) -> dict:
    """Train and score every arm in order, writing each result as it lands.

    `score_only` re-scores arms that are already trained. Scoring is minutes
    and training is ~37 an arm, so a defect found in the scorer while a sweep
    is in flight is repaired by re-scoring afterwards rather than by killing
    the sweep.

    A killed sweep is now recoverable rather than lost: finished arms are
    skipped by their scored file, and the arm that was mid-flight is picked up
    from its checkpoint, because `run_with_resume` reads the open in-flight
    marker its dead supervisor left instead of counting attempts. That was not
    true when this sweep started, and the difference was 60.3M tokens.
    """
    root = Path(root)
    results = {}
    arms = sweep_order(tokenizer_root, include_matched=include_matched)
    for index, (label, protocol) in enumerate(arms, start=1):
        scored_path = root / "scored" / f"{label}-{protocol}.json"
        if skip_existing and scored_path.exists():
            log(f"[{index}/{len(arms)}] {label} {protocol}: already scored")
            results[f"{label}-{protocol}"] = json.loads(scored_path.read_text())
            continue
        size, path = resolve_label(label, tokenizer_root)
        if score_only:
            checkpoint = Path("runs") / probe_run_name(label, protocol) / "checkpoint.pt"
            if not checkpoint.exists():
                log(f"[{index}/{len(arms)}] {label} {protocol}: no checkpoint, "
                    f"skipping")
                continue
        else:
            log(f"[{index}/{len(arms)}] {label} {protocol}: training", flush=True)
            launched = launch_probe(
                vocab_size=size, label=label, protocol=protocol,
                shard_root=shard_root, tokenizer_path=path, device=device)
            probe_path = root / "probes" / f"{label}-{protocol}.json"
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            probe_path.write_text(json.dumps(launched, indent=2, sort_keys=True,
                                             default=str) + "\n")

        log(f"[{index}/{len(arms)}] {label} {protocol}: scoring", flush=True)
        record = score_probe(vocab_size=size, label=label, protocol=protocol,
                             shard_root=shard_root, tokenizer_path=path,
                             device=device)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        results[f"{label}-{protocol}"] = record
        log(f"[{index}/{len(arms)}] {label} {protocol}: bpb {record['bpb']:.4f} "
            f"after {record.get('tokens_seen') or 0:,} tokens", flush=True)
    return results


def resolve_label(label: str, tokenizer_root) -> Tuple[int, str]:
    for known, size, path in measurement_targets(tokenizer_root):
        if known == label:
            return size, path
    raise SystemExit(
        f"{label!r} is not a measured tokenizer; known: "
        f"{[k for k, _s, _p in measurement_targets(tokenizer_root)]}")


def decide_from_artifacts(root) -> dict:
    """Assemble each candidate's measured numbers and apply the rule.

    Reads the artifacts the earlier stages wrote rather than re-deriving
    anything, so a number in the verdict is the number in the file it cites.
    """
    root = Path(root)
    measurements = json.loads((root / "measurements.json").read_text())
    scored: Dict[str, Dict[str, dict]] = {}
    for path in sorted((root / "scored").glob("*.json")):
        record = json.loads(path.read_text())
        scored.setdefault(record["label"], {})[record["protocol"]] = record

    incumbent = measurements[INCUMBENT_KEY]
    incumbent_bpb = {protocol: record["bpb"]
                     for protocol, record in scored.get(INCUMBENT_KEY, {}).items()}

    verdicts, diagnostics = [], {}
    for label in (str(size) for size in CANDIDATE_VOCAB_SIZES):
        reading = measurements[label]
        deltas = {protocol: 100.0 * (record["bpb"] - incumbent_bpb[protocol])
                  / incumbent_bpb[protocol]
                  for protocol, record in scored.get(label, {}).items()
                  if protocol in incumbent_bpb}
        measured = {
            "vocab_size": reading["vocab_size"],
            "domain_fertility_delta_pct": fertility_deltas(reading, incumbent),
            # The worse of the two protocols; see BPB_PROTOCOL_RULE.
            "tiny_bpb_delta_pct": max(deltas.values()) if deltas else None,
            "tiny_bpb_delta_pct_by_protocol": deltas,
            "embedding_q6_k_bytes": reading["embedding"]["q6_k_bytes"],
            "incumbent_embedding_q6_k_bytes":
                incumbent["embedding"]["q6_k_bytes"],
            "round_trip_passed": reading["round_trip"]["passed"],
        }
        verdicts.append(evaluate_candidate(measured))
        if MATCHED_CONTROL_KEY in measurements:
            diagnostics[label] = {
                "fertility_vs_matched_control_pct": fertility_deltas(
                    reading, measurements[MATCHED_CONTROL_KEY]),
            }

    verdict = decide(verdicts)
    verdict["clauses_by_candidate"] = {
        str(v["vocab_size"]): v["clauses"] for v in verdicts}
    verdict["measured"] = {str(v["vocab_size"]): v["measured"] for v in verdicts}
    verdict["diagnostics"] = diagnostics
    verdict["addendum"] = json.loads((root / "addendum.json").read_text()) \
        if (root / "addendum.json").exists() else None
    return verdict


# ------------------------------------------------------------------ report ---

def _fertility_table(measurements: dict, reference: str) -> List[str]:
    rows = [f"| tokenizer | " + " | ".join(DOMAINS) + " | all |",
            "|---|" + "---|" * (len(DOMAINS) + 1)]
    for label, reading in measurements.items():
        if label == reference:
            continue
        deltas = fertility_deltas(reading, measurements[reference])
        cells = [f"{deltas[d]:+.2f}%" if d in deltas else "n/a"
                 for d in DOMAINS]
        overall = 100.0 * (
            measurements[reference]["fertility"]["__all__"]["bytes_per_token"]
            - reading["fertility"]["__all__"]["bytes_per_token"]) / \
            measurements[reference]["fertility"]["__all__"]["bytes_per_token"]
        rows.append(f"| {label} | " + " | ".join(cells) + f" | {overall:+.2f}% |")
    return rows


def write_report(root, sample_root="data/tokenizer-lab/sample") -> Path:
    """The migration report: what was measured, against what, and what it means."""
    root = Path(root)
    measurements = json.loads((root / "measurements.json").read_text())
    verdict = json.loads((root / "verdict.json").read_text()) \
        if (root / "verdict.json").exists() else {}
    manifest_path = Path(sample_root) / "sample-manifest.json"
    sample = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    gguf = json.loads((root / "gguf-check.json").read_text()) \
        if (root / "gguf-check.json").exists() else {}

    lines = [
        "# V2 tokenizer migration report",
        "",
        "Phase 4 asks whether a future from-scratch V2 should keep SmolLM2's "
        "49,152-token vocabulary or adopt 24,576, 32,768 or 40,960.",
        "",
        "**Scope.** A tokenizer cannot be transplanted into a trained model: "
        "every embedding row and every output logit is indexed by the "
        "vocabulary the model was trained under. Nothing here changes released "
        "V1 weights or Daedalus-Code, and no result below should be read as a "
        "gain on either.",
        "",
        "## The preregistered rule",
        "",
        f"> {RULE_TEXT}",
        "",
        f"Rule digest `{rule_digest()}`, written before any measurement. "
        f"Thresholds: no domain worse than "
        f"{MAX_DOMAIN_FERTILITY_REGRESSION_PCT:g}%, code "
        f"<= {MAX_CODE_FERTILITY_REGRESSION_PCT:g}%, tiny-model BPB "
        f"<= {MAX_TINY_BPB_REGRESSION_PCT:g}%, embedding bytes down at least "
        f"{MIN_EMBEDDING_BYTE_REDUCTION_PCT:g}%.",
        "",
        "## Bytes per token vs the incumbent (SmolLM2), by domain",
        "",
        "Negative is better: the candidate needs fewer tokens for the same "
        "bytes. This is the comparison the rule reads.",
        "",
    ]
    lines += _fertility_table(measurements, INCUMBENT_KEY)

    if MATCHED_CONTROL_KEY in measurements:
        lines += [
            "",
            "## Bytes per token vs a size-matched control",
            "",
            "`49152-matched` is 49,152 tokens trained on **this** sample with "
            "the same trainer. Candidate-vs-matched isolates the cost of "
            "shrinking the vocabulary; candidate-vs-SmolLM2 above mixes that "
            "with the gain from retraining on this corpus, and would otherwise "
            "credit *size* for a *corpus-match* effect.",
            "",
        ]
        lines += _fertility_table(measurements, MATCHED_CONTROL_KEY)

    lines += ["", "## Artifact cost", "",
              "| tokenizer | embedding params | Q6_K MiB | vs incumbent | "
              "KV bytes/context-token |", "|---|---|---|---|---|"]
    base = measurements[INCUMBENT_KEY]["embedding"]["q6_k_bytes"]
    for label, reading in measurements.items():
        embedding = reading["embedding"]
        lines.append(
            f"| {label} | {embedding['parameters']:,} | "
            f"{embedding['q6_k_bytes'] / 2**20:.2f} | "
            f"{100 * (base - embedding['q6_k_bytes']) / base:+.1f}% | "
            f"{reading['kv']['kv_bytes_per_context_token']:,} |")
    lines += [
        "",
        "`token_embd.weight` ships Q6_K and is tied, so that one tensor is "
        "both the input table and the output projection "
        "(`runs/preflight/token-embd-quant-grid.md`). The KV column is "
        "identical by construction: the cache is attention-shaped, so a "
        "vocabulary change moves the embedding tensor and nothing in it.",
        "",
        "## Byte coverage and round trips",
        "",
        "| tokenizer | byte characters | round trip |", "|---|---|---|",
    ]
    for label, reading in measurements.items():
        coverage = reading["round_trip"]["byte_alphabet"]
        lines.append(
            f"| {label} | {coverage['covered']}/256 | "
            f"{'pass' if reading['round_trip']['passed'] else 'FAIL'} |")
    lines += [
        "",
        "The incumbent's row is a finding, not a formatting error. SmolLM2's "
        "vocabulary is missing 21 byte-level characters; most stand for bytes "
        "that never occur in valid UTF-8, but 0xf1-0xf3 are four-byte lead "
        "bytes, so code points U+40000-U+FFFFF are silently dropped rather "
        "than rejected. Every candidate covers all 256.",
        "",
        "## Held-out bits per byte",
        "",
        "Bits per **byte**, never per-token perplexity: a larger vocabulary "
        "improves per-token likelihood by packing more bytes into each token, "
        "with no improvement in the model.",
        "",
        "| arm | protocol | tokens trained | BPB | vs incumbent |",
        "|---|---|---|---|---|",
    ]
    scored_files = sorted((root / "scored").glob("*.json"))
    scored = [json.loads(p.read_text()) for p in scored_files]
    incumbent_bpb = {r["protocol"]: r["bpb"] for r in scored
                     if r["label"] == INCUMBENT_KEY}
    for record in scored:
        reference = incumbent_bpb.get(record["protocol"])
        delta = (f"{100 * (record['bpb'] - reference) / reference:+.2f}%"
                 if reference else "n/a")
        lines.append(
            f"| {record['label']} | {record['protocol']} | "
            f"{record.get('tokens_seen') or 0:,} | {record['bpb']:.4f} | "
            f"{delta} |")

    if gguf:
        lines += [
            "", "## Can stock llama.cpp convert these?", "",
            "The constraint that decides whether any of this is actionable: "
            "unmodified stock llama.cpp is a fixed program decision. "
            "`conversion/base.py::get_vocab_base_pre` identifies a BPE "
            "pre-tokenizer by hashing the token **ids** a fixed probe string "
            "encodes to, checks that against a hard-coded list, and raises "
            "`NotImplementedError` for anything absent. The hash is over ids, "
            "so it moves with the vocabulary and the merges -- a newly trained "
            "tokenizer cannot match a registered hash however faithfully it "
            "copies SmolLM2's pre-tokenizer configuration.",
            "",
            "| tokenizer | converts | pre-tokenizer recognised |",
            "|---|---|---|",
        ]
        for label, record in gguf.items():
            if not record.get("ran"):
                lines.append(f"| {label} | not run | {record.get('reason')} |")
                continue
            lines.append(
                f"| {label} | {'yes' if record['converted'] else 'no'} | "
                f"{'no' if record.get('pre_tokenizer_unrecognized') else 'yes'} |")

    if verdict:
        lines += [
            "", "## Verdict", "",
            f"**Selected: {verdict.get('selected') or 'none'}.** "
            f"{verdict.get('reason', '')}",
            "",
            "| candidate | selectable | failed clauses |", "|---|---|---|",
        ]
        for candidate in verdict.get("candidates", []):
            lines.append(
                f"| {candidate['vocab_size']} | "
                f"{'yes' if candidate['selectable'] else 'no'} | "
                f"{', '.join(candidate['failed']) or '-'} |")

    if sample:
        lines += [
            "", "## The sample", "",
            f"{sample['totals']['bytes'] / 1e9:.3f} GB across "
            f"{len(sample['sources'])} sources, split disjointly by SHA-256 of "
            f"each document's bytes: "
            + ", ".join(f"{split} {size / 1e6:.0f} MB"
                        for split, size in sample["totals"]["by_split"].items())
            + ".",
            "",
            "| domain | bytes |", "|---|---|",
        ]
        for domain, size in sample["totals"]["by_domain"].items():
            lines.append(f"| {domain} | {size / 1e6:,.1f} MB |")
        if sample.get("shortfalls"):
            lines += ["", "Sources that could not fill their share, recorded "
                          "rather than force-filled:", ""]
            for key, record in sample["shortfalls"].items():
                lines.append(
                    f"- `{key}`: {record['achieved_bytes'] / 1e6:.2f} MB of "
                    f"{record['planned_bytes'] / 1e6:.2f} MB "
                    f"({record['fill_frac']:.1%})")

    path = root / "v2-tokenizer-migration.md"
    path.write_text("\n".join(lines) + "\n")
    return path


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

    pack = sub.add_parser("pack", help="shard the LM splits under each vocabulary")
    pack.add_argument("--sample", default="data/tokenizer-lab/sample")
    pack.add_argument("--tokenizers", default="data/tokenizer-lab/tokenizers")
    pack.add_argument("--out", default="data/tokenizer-lab/shards")
    pack.add_argument("--only", action="append", default=[],
                      help="measurement label to pack; repeatable, default all")
    pack.add_argument("--byte-budget", type=int, default=None,
                      help="bytes of LM-train every arm reads (default: the "
                           "whole split)")

    probe = sub.add_parser("probe", help="train one tiny-model arm")
    probe.add_argument("--label", required=True)
    probe.add_argument("--protocol", required=True,
                       choices=("equal-bytes", "equal-tokens"))
    probe.add_argument("--shards", default="data/tokenizer-lab/shards")
    probe.add_argument("--tokenizers", default="data/tokenizer-lab/tokenizers")
    probe.add_argument("--device", default="cuda")
    probe.add_argument("--max-attempts", type=int, default=3)

    sweep = sub.add_parser("sweep", help="train and score every arm in order")
    sweep.add_argument("--shards", default="data/tokenizer-lab/shards")
    sweep.add_argument("--tokenizers", default="data/tokenizer-lab/tokenizers")
    sweep.add_argument("--device", default="cuda")
    sweep.add_argument("--include-matched", action="store_true",
                       help="also probe the size-matched control (diagnostic, "
                            "not read by the rule)")
    sweep.add_argument("--rerun", action="store_true",
                       help="re-train arms that already have a score")
    sweep.add_argument("--score-only", action="store_true",
                       help="re-score already-trained arms without training; "
                            "implies --rerun")

    score = sub.add_parser("score", help="held-out BPB for one arm")
    score.add_argument("--label", required=True)
    score.add_argument("--protocol", required=True,
                       choices=("equal-bytes", "equal-tokens"))
    score.add_argument("--shards", default="data/tokenizer-lab/shards")
    score.add_argument("--tokenizers", default="data/tokenizer-lab/tokenizers")
    score.add_argument("--device", default="cuda")

    addendum = sub.add_parser(
        "addendum", help="record decisions the preregistration left open, "
                         "refused once any arm has been scored")
    addendum.add_argument("--force", action="store_true")

    gguf = sub.add_parser(
        "gguf-check", help="can stock llama.cpp convert each vocabulary?")
    gguf.add_argument("--tokenizers", default="data/tokenizer-lab/tokenizers")
    gguf.add_argument("--out", default="runs/tokenizer-lab/gguf-check")
    gguf.add_argument("--llama-cpp-dir", default=None)

    sub.add_parser("decide", help="apply the preregistered rule")
    report = sub.add_parser("report", help="render the migration report")
    report.add_argument("--sample", default="data/tokenizer-lab/sample")

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
        for label, size, path in measurement_targets(args.tokenizers):
            print(f"  {label}: {path}", flush=True)
            readings[label] = measure_tokenizer(
                load_tokenizer(path), name=path, vocab_size=size,
                root=args.sample, identifiers=identifiers,
                require_round_trip=label != INCUMBENT_KEY)
        out = root / "measurements.json"
        out.write_text(json.dumps(readings, indent=2, sort_keys=True) + "\n")
        for reference in (INCUMBENT_KEY, MATCHED_CONTROL_KEY):
            if reference not in readings:
                continue
            print(f"\n  bytes/token vs {reference} "
                  f"(negative = candidate needs fewer tokens):")
            for label, _size, _path in measurement_targets(args.tokenizers):
                if label == reference:
                    continue
                deltas = fertility_deltas(readings[label], readings[reference])
                print(f"    {label:16s} " + "  ".join(
                    f"{domain} {delta:+.2f}%"
                    for domain, delta in sorted(deltas.items())))
        print(f"\nwrote {out}")
        return 0

    if args.command == "pack":
        measurements = json.loads((root / "measurements.json").read_text())
        targets = [t for t in measurement_targets(args.tokenizers)
                   if not args.only or t[0] in args.only]
        # The whole LM-train split, not a computed budget. Every arm still
        # reads exactly the same bytes, which is all the byte-matched protocol
        # requires, and spending the split removes the need to predict its
        # fertility from the holdout's -- a prediction that was 2.8-3.4% low
        # and left two arms under the equal-tokens budget.
        available = int(json.loads(
            (Path(args.sample) / "sample-manifest.json").read_text()
        )["totals"]["by_split"]["lm-train"])
        floor = equal_byte_budget(max_bytes_per_token=max(
            reading["fertility"]["__all__"]["bytes_per_token"]
            for reading in measurements.values()))
        if available < floor:
            raise SystemExit(
                f"LM-train holds {available:,} bytes, below the {floor:,} the "
                f"worst arm needs for {LM_PROBE_TOKENS:,} tokens; rebuild the "
                f"sample with a larger target")
        budget = args.byte_budget or available
        print(f"byte budget {budget:,} ({budget/1e6:.1f} MB); the worst arm "
              f"needs at least {floor:,} for {LM_PROBE_TOKENS:,} tokens",
              flush=True)
        packed = {}
        for label, size, path in targets:
            packed[label] = pack_for_tokenizer(
                sample_root=args.sample, tokenizer_path=path, vocab_size=size,
                label=label, out_root=args.out, byte_budget=budget)
        (root / "packed.json").write_text(
            json.dumps({"byte_budget": budget,
                        "arms": {label: {k: v for k, v in record.items()
                                         if k != "train"}
                                 for label, record in packed.items()}},
                       indent=2, sort_keys=True, default=str) + "\n")
        short = {label: record["train"]["total_tokens"]
                 for label, record in packed.items()
                 if record["train"]["total_tokens"] < LM_PROBE_TOKENS}
        for label, record in packed.items():
            print(f"  {label:16s} {record['train']['total_tokens']:>12,} tokens"
                  f"  {record['measured_bytes_per_token']:.4f} bytes/token"
                  f"{'   SHORT' if label in short else ''}")
        if short:
            # Loud, not silent: an arm below the budget would either train on
            # less than the others or quietly re-read data they saw once, and
            # both are the comparison failing rather than a run failing.
            print(f"\n{len(short)} arm(s) packed fewer than the "
                  f"{LM_PROBE_TOKENS:,}-token equal-tokens budget: {short}")
            return 1
        return 0

    if args.command == "probe":
        size, path = resolve_label(args.label, args.tokenizers)
        report = launch_probe(vocab_size=size, label=args.label,
                              protocol=args.protocol, shard_root=args.shards,
                              tokenizer_path=path, device=args.device,
                              max_attempts=args.max_attempts)
        out = root / "probes" / f"{args.label}-{args.protocol}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                  default=str) + "\n")
        print(json.dumps({k: report[k] for k in
                          ("run", "protocol", "total_tokens", "packed_tokens")},
                         indent=2))
        return 0

    if args.command == "sweep":
        results = run_sweep(
            root=root, shard_root=args.shards, tokenizer_root=args.tokenizers,
            device=args.device, include_matched=args.include_matched,
            skip_existing=not (args.rerun or args.score_only),
            score_only=args.score_only)
        print(json.dumps({name: record["bpb"]
                          for name, record in results.items()},
                         indent=2, sort_keys=True))
        return 0

    if args.command == "score":
        size, path = resolve_label(args.label, args.tokenizers)
        record = score_probe(vocab_size=size, label=args.label,
                             protocol=args.protocol, shard_root=args.shards,
                             tokenizer_path=path, device=args.device)
        out = root / "scored" / f"{args.label}-{args.protocol}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    if args.command == "gguf-check":
        results = {}
        for label, size, path in measurement_targets(args.tokenizers):
            print(f"  converting with {label} ...", flush=True)
            results[label] = check_stock_gguf_conversion(
                label=label, vocab_size=size, tokenizer_path=path,
                out_dir=args.out, llama_cpp_dir=args.llama_cpp_dir)
            print(f"    converted={results[label].get('converted')} "
                  f"unrecognized_pretokenizer="
                  f"{results[label].get('pre_tokenizer_unrecognized')}",
                  flush=True)
        out = root / "gguf-check.json"
        out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out}")
        return 0

    if args.command == "addendum":
        written = write_addendum(root, build_addendum(), force=args.force)
        print(f"recorded {written}")
        return 0

    if args.command == "decide":
        verdict = decide_from_artifacts(root)
        (root / "verdict.json").write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: verdict[k] for k in
                          ("selected", "negative_result", "reason")}, indent=2))
        return 0

    if args.command == "report":
        path = write_report(root, sample_root=args.sample)
        print(f"wrote {path}")
        return 0

    return 1


def _exit(code: int) -> None:
    """Leave the process without running interpreter finalization.

    `datasets`' parquet streaming leaves a pyarrow thread pool behind, and
    CPython's shutdown of it aborts with SIGABRT
    (`PyGILState_Release: thread state ... must be current when releasing`)
    *after* `main` has returned. The controller reads the exit status, so a
    completed sample that had written and fsynced every output was recorded as
    a failed phase, halting the program.

    Exiting here rather than suppressing the abort keeps the status meaningful:
    the code below is this script's own verdict, decided before any
    finalization runs, so a genuine failure still reports one. Everything this
    script writes goes through `write_text` or a closed handle, so there is no
    buffered output for finalization to flush.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        _code = main()
    except SystemExit as _exit_request:               # argparse and raise SystemExit
        _code = _exit_request.code if isinstance(_exit_request.code, int) else 1
    except BaseException:                             # noqa: BLE001
        import traceback
        traceback.print_exc()
        _code = 1
    _exit(_code or 0)
