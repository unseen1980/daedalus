"""Daedalus `dataprep` job (AGENT.md SS4): stream the stable-phase mixture,
cross-dedup, decontaminate against the eval suite, tokenize, and shard.

Built on top of `daedalus/data.py`'s primitives (`ShardWriter`,
`tokenize_document`, `NearDupFilter`, n-gram decontam, Hub upload). This
module adds: the mixture table itself, per-source streaming with row-level
filters, a memory-bounded dedup strategy across the whole ~45B-token corpus,
and run/resume orchestration that checkpoints to `data/manifest.json` after
every source so an interrupted run doesn't lose completed sources.

## Substitutions for gated/incomplete sources

AGENT.md SS4's mixture table names three sources that turned out not to be
directly usable (verified live against the Hub before writing this module --
see STATUS.md). Per AGENT.md's explicit policy ("if a dataset is gated or
renamed, substitute the closest equivalent, note it in the manifest, and
continue -- do not stall"), each is substituted rather than blocking the run.
Full detail in `GATED_SUBSTITUTION_NOTES` below, written into every run's
manifest.

## Memory-bounded dedup

Exact-duplicate detection (a 64-bit hash of normalized text) is cheap enough
to keep as one set per *near-dup group* (see below) -- tens of millions of
documents cost a few GB, well within this box's 30GB RAM, but it is no
longer literally one set for the whole run now that groups run in separate
processes (see "Parallelism" below); the one case that actually matters --
catching the same document recurring across sources -- is exactly the
overlap case already scoped to a shared group, so this loses no real
coverage in practice.

Near-duplicate detection (MinHash/LSH) is NOT global: an LSH index sized to
the full corpus (tens of millions of documents, each costing >1KB for its
hash permutation vector) would exceed available RAM well before disk becomes
the constraint -- this was flagged as an open risk in STATUS.md before this
module was written. Instead, `DedupState` gives each source its own
`NearDupFilter`, periodically reset every `near_dup_reset_every` documents,
which bounds peak memory to O(reset_every) regardless of total corpus size.
This is a deliberate, documented approximation: it only catches near-dups
within a `reset_every`-sized window of stream order, not across the whole
corpus. The one cross-source case AGENT.md/the blueprint explicitly call out
-- fineweb-edu and DCLM's ~32% mutual overlap (Zyda-2) -- is still covered:
those two sources share a single filter (see `MIXTURE`'s `near_dup_group`).

## Parallelism

A live-timed check of `run_source` against the real Hub (see STATUS.md)
measured ~70-140k tokens/sec per source running *alone*, and roughly the same
per-source rate when three sources ran *concurrently* -- confirming this is
I/O/parse bound, not CPU-saturation bound, on this 16-core box. Run fully
sequentially (the original design), the ~45B-token target would take on the
order of 100+ hours, far past AGENT.md's ~8h/~$6 estimate. `run_dataprep`
therefore dispatches one process per `near_dup_group` (multiple sources
sharing a group, e.g. the fineweb-edu+dclm-baseline overlap pair, still run
*sequentially within that one process* so they can share one `DedupState` --
correctness for the documented cross-source case comes first, and the
wall-clock win comes from the ~10 otherwise-independent groups running
concurrently instead of one 16-core-idle process running everything in turn).

This relies on `multiprocessing`'s **fork** start method specifically: worker
processes are dispatched by *key*, not by passing `SourceSpec` objects across
the process boundary, because several `SourceSpec.text_fn`/`filter_fn`
fields are closures (e.g. `_text_field(...)`) that plain `pickle` cannot
serialize. Fork's copy-on-write semantics let a worker see the parent's
already-built `_ACTIVE_MIXTURE_BY_KEY` (set right before the pool starts) for
free, with no pickling of function objects involved. This module must not be
adapted to the `spawn` start method without redesigning that lookup.
"""
import gc
import hashlib
import json
import multiprocessing
import os
import resource
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import psutil

# Belt-and-suspenders alongside the submit-before-wandb-init ordering in
# run_dataprep (see its docstring): disables the `tokenizers` crate's
# internal rayon thread pool outright, which is HF's own documented
# mitigation for its fork-safety check ever triggering in a forked worker.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from daedalus.data import (
    DEFAULT_EOS_ID,
    NearDupFilter,
    ShardWriter,
    TOKEN_DTYPE,
    build_eval_ngram_index,
    get_tokenizer,
    is_contaminated,
    tokenize_document,
    upload_shards,
)

# ----------------------------------------------------------------- mixture ---

_WEB_OVERLAP_GROUP = "fineweb-edu+dclm"  # shared near-dup filter, see module docstring


def _text_field(name: str) -> Callable[[dict], str]:
    return lambda row: row.get(name) or ""


def _flatten_messages(row: dict) -> str:
    """Turns a chat-format {"messages": [{"role", "content"}, ...]} row into
    plain text, for conversational sources mixed into pretraining data."""
    parts = []
    for m in row.get("messages") or []:
        content = m.get("content")
        if not content:
            continue
        role = m.get("role")
        parts.append(f"{role}: {content}" if role else content)
    return "\n".join(parts)


@dataclass
class SourceSpec:
    key: str
    dataset: str
    share: float
    config: Optional[str] = None
    split: str = "train"
    revision: Optional[str] = None
    text_fn: Callable[[dict], str] = _text_field("text")
    filter_fn: Optional[Callable[[dict], bool]] = None
    max_epochs: int = 1
    near_dup_group: Optional[str] = None  # defaults to `key` if unset
    load_kwargs: dict = field(default_factory=dict)
    note: str = ""


MIXTURE: List[SourceSpec] = [
    SourceSpec("fineweb-edu", "HuggingFaceFW/fineweb-edu", share=0.375, config="default",
               filter_fn=lambda r: (r.get("int_score") or 0) >= 3,
               near_dup_group=_WEB_OVERLAP_GROUP,
               note="AGENT.md base share 30% + Nemotron-CC-v2's redistributed 7.5pp"),
    SourceSpec("dclm-baseline", "mlfoundations/dclm-baseline-1.0", share=0.225, config="default",
               near_dup_group=_WEB_OVERLAP_GROUP,
               note="AGENT.md base share 18% + Nemotron-CC-v2's redistributed 4.5pp"),
    SourceSpec("stack-edu-python", "codeparrot/github-code", share=0.09,
               revision="refs/convert/parquet", text_fn=_text_field("code"),
               load_kwargs={"data_files": "Python-all/partial-train/*.parquet"},
               note="substitute for smollm-corpus's Stack-Edu/Python config, see "
                    "GATED_SUBSTITUTION_NOTES. Uses the parquet-converted revision's "
                    "per-language data_files directly (Python-all/), NOT config='default' "
                    "+ a language=='Python' row filter -- the 'default' config interleaves "
                    "all languages with Python vanishingly rare in stream order (scanned "
                    "646k consecutive rows with zero Python matches in a live check), so a "
                    "row filter over it would take unbounded time to fill the token budget."),
    SourceSpec("finepdfs-edu", "HuggingFaceFW/finepdfs-edu", share=0.08, config="eng_Latn"),
    SourceSpec("finephrase", "HuggingFaceFW/finephrase", share=0.07, config="all"),
    SourceSpec("finemath-3plus", "HuggingFaceTB/finemath", share=0.03, config="finemath-3plus"),
    SourceSpec("infiwebmath-3plus", "HuggingFaceTB/finemath", share=0.03, config="infiwebmath-3plus",
               note="finemath-3plus + infiwebmath-3plus together cover AGENT.md's 6% "
                    "'finemath ∪ Nemotron-CC-Math-v1' bucket, see GATED_SUBSTITUTION_NOTES"),
    SourceSpec("cosmopedia-v2", "HuggingFaceTB/cosmopedia-v2", share=0.05, config="cosmopedia-v2"),
    SourceSpec("finewiki-en", "HuggingFaceFW/finewiki", share=0.03, config="en"),
    SourceSpec("everyday-conversations", "HuggingFaceTB/everyday-conversations-llama3.1-2k",
               share=0.02, split="train_sft", text_fn=_flatten_messages, max_epochs=20,
               note="~2.2k rows total; even at max_epochs=20 this falls far short of a "
                    "literal 2% token share -- the shortfall is reported, not force-filled, "
                    "see GATED_SUBSTITUTION_NOTES"),
]

assert abs(sum(s.share for s in MIXTURE) - 1.0) < 1e-9, "mixture shares must sum to 1.0"


def build_budget_mixture(budgets: Dict[str, int],
                         mixture: Optional[List[SourceSpec]] = None,
                         ) -> Tuple[List[SourceSpec], int]:
    """Turn absolute per-source token budgets into a (mixture, target_tokens)
    pair that `run_dataprep` can consume unchanged.

    Everything downstream derives a source's budget as
    `round(spec.share * target_tokens)` (`_run_group_worker`), so there is no
    way to ask for "3.75B of fineweb-edu and 2.05B of dclm" through the
    blueprint's proportional shares alone. This rewrites the shares so that
    identity reproduces the requested absolute numbers exactly, with
    `target_tokens` set to their sum.

    Needed because the corpus this project actually built is *not*
    proportional: `finemath-3plus`/`infiwebmath-3plus`/`finephrase`
    overshot their blueprint shares while `fineweb-edu`/`dclm-baseline`
    starved (issue #4 §4.2). Finishing the corpus therefore means topping up
    specific sources to specific absolute numbers, not scaling every source
    by one global target.

    Pass a budget for *every* source you want represented, including ones
    already complete -- give those their current on-disk token count and they
    no-op on `--resume` while still being recovered into the run manifest.
    Sources omitted from `budgets` are dropped from the mixture entirely.
    """
    mixture = mixture if mixture is not None else MIXTURE
    by_key = {s.key: s for s in mixture}
    unknown = sorted(set(budgets) - set(by_key))
    if unknown:
        raise ValueError(f"unknown source key(s) in budgets: {unknown}; "
                         f"known keys are {sorted(by_key)}")
    if not budgets:
        raise ValueError("budgets must name at least one source")
    if any(v < 0 for v in budgets.values()):
        raise ValueError("source budgets must be non-negative")
    total = sum(budgets.values())
    if total <= 0:
        raise ValueError("source budgets must sum to a positive number of tokens")
    out = [replace(by_key[k], share=budgets[k] / total)
           for k in (s.key for s in mixture) if k in budgets]
    return out, total

GATED_SUBSTITUTION_NOTES = {
    "nvidia/Nemotron-CC-v2": (
        "manually-gated on the Hub; access was not granted at run time (verified live "
        "before this run). Its 12% share was redistributed proportionally to "
        "fineweb-edu (+7.5pp) and dclm-baseline-1.0 (+4.5pp), the closest-in-kind "
        "general high-quality web corpora already in the mixture."
    ),
    "nvidia/Nemotron-CC-Math-v1": (
        "auto-gated on the Hub; access was not granted at run time. Its portion of "
        "AGENT.md's 6% 'finemath ∪ Nemotron-CC-Math-v1' bucket is covered entirely by "
        "HuggingFaceTB/finemath (finemath-3plus + infiwebmath-3plus), already in the mixture."
    ),
    "HuggingFaceTB/smollm-corpus (Stack-Edu/Python config)": (
        "ships only Stack-v2 blob_id/repo/path metadata, not raw text (Stack-v2 license "
        "terms require joining against gated raw-content storage). Substituted with "
        "codeparrot/github-code (parquet-converted revision) filtered to language=='Python'."
    ),
}


# ------------------------------------------------------------------- dedup ---

def _normalize_for_hash(text: str) -> str:
    return " ".join(text.split()).lower()


def exact_hash(text: str) -> int:
    """64-bit hash of normalized text, for a cheap global exact-dup set."""
    digest = hashlib.blake2b(_normalize_for_hash(text).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


@dataclass
class DedupState:
    """See module docstring for the memory-bounded near-dup design.

    `exact_seen` is bounded the same way as the near-dup filter: one set per
    `near_dup_group`, reset every `near_dup_reset_every` kept docs. Without
    this, a long-running group (e.g. fineweb-edu+dclm's 27B-token share at
    full scale) would grow an unbounded Python set of 64-bit hashes for the
    life of the process -- exactly the "accumulate instead of stream" failure
    ADDENDUM 2 rule 2 calls out. This is the same documented tradeoff as the
    near-dup filter: exact dups are only caught within a reset_every-sized
    window of stream order, not across the whole group.
    """
    near_dup_reset_every: int = 2_000_000
    num_perm: int = 64
    threshold: float = 0.85
    counters: Dict[str, int] = field(default_factory=lambda: {
        "kept": 0, "exact_dup": 0, "near_dup": 0, "contaminated": 0})
    _filters: Dict[str, NearDupFilter] = field(default_factory=dict, repr=False)
    _exact_seen: Dict[str, set] = field(default_factory=dict, repr=False)
    _group_counts: Dict[str, int] = field(default_factory=dict, repr=False)

    def _filter_for(self, group: str) -> NearDupFilter:
        n = self._group_counts.get(group, 0)
        if group not in self._filters or n >= self.near_dup_reset_every:
            self._filters[group] = NearDupFilter(threshold=self.threshold, num_perm=self.num_perm)
            self._exact_seen[group] = set()
            self._group_counts[group] = 0
        return self._filters[group]

    def keep(self, text: str, group: str, eval_ngram_index: Optional[set]) -> bool:
        """Exact-dup -> decontam -> near-dup, cheapest check first."""
        self._filter_for(group)  # ensures _exact_seen[group] exists (and is reset on schedule)
        h = exact_hash(text)
        if h in self._exact_seen[group]:
            self.counters["exact_dup"] += 1
            return False
        if eval_ngram_index is not None and is_contaminated(text, eval_ngram_index):
            self.counters["contaminated"] += 1
            return False
        if self._filters[group].is_duplicate(text):
            self.counters["near_dup"] += 1
            return False
        self._exact_seen[group].add(h)
        self._group_counts[group] = self._group_counts.get(group, 0) + 1
        self.counters["kept"] += 1
        return True


def _build_eval_index(n: int = 13, limit: Optional[int] = 2000) -> set:
    """Union of n-grams over every eval task's context+continuation text, for
    8-13-gram decontamination (AGENT.md SS4)."""
    from eval import load_all_tasks  # repo-root module, see eval.py

    tasks = load_all_tasks(limit=limit)
    texts = [ctx + " " + cont for examples in tasks.values() for ex in examples
             for ctx, cont in ex.candidates]
    return build_eval_ngram_index(texts, n=n)


# ---------------------------------------------------------------- sourcing ---

class _RowStream:
    """Iterable over one source's raw HF rows that can also report its exact
    position in the stream (`state()`) and restore one (`stream_state`).

    This is issue #3's O(1) resume. `datasets` 5.x `IterableDataset` exposes
    `state_dict()`/`load_state_dict()`, which record the *shard* (parquet
    file) index plus the row offset inside it -- so resuming re-reads at most
    the one file the previous chunk stopped in, never the whole prefix of the
    stream. Measured live against `fineweb-edu`: after consuming 20,000 rows,
    `load_state_dict` on a brand-new stream returned the *exact* next row
    (byte-identical id and text) in 7.6 s, versus 16.2 s to replay the same
    prefix by iterating -- and the replay cost grows without bound while the
    restore cost is capped by one file (fineweb-edu ships 3,036 of them).

    That distinction is the whole fix. The previous contract resumed a
    soft-stopped source by re-streaming every already-processed row
    (`_documents`' `skip`), which is O(n) per respawn and therefore
    O(n^2/chunk) over a source: at fineweb-edu's ~15.9M documents and the
    measured ~150K-document chunk size that is ~8.5e8 rows of pure
    re-reading, ~68 h for one source. Worse, the replay itself left ~0.6 GB
    of extra resident memory in the resumed worker (measured: a fresh worker
    started at 1.17 GB, a `skip`-resumed one at 1.75 GB), which pushed it
    across the 3.0 GB soft threshold after only ~6,000 new documents -- so
    each respawn did less work than the one before while paying more to
    start, and the run converged to making no progress at all. See the
    `validate_tighter_rss` measurements in issue #3.

    `state()` returns None when the underlying object has no `state_dict`
    (a monkeypatched test stub, or a `datasets` too old to support it); every
    caller treats that as "no O(1) resume available" and falls back to the
    old `skip` replay rather than failing."""

    def __init__(self, spec: SourceSpec, stream_state: Optional[dict] = None, log=print):
        from datasets import load_dataset

        kwargs = dict(spec.load_kwargs)
        if spec.revision:
            kwargs["revision"] = spec.revision
        self.ds = load_dataset(spec.dataset, spec.config, split=spec.split,
                                streaming=True, **kwargs)
        self.resumed = False
        if stream_state:
            try:
                self.ds.load_state_dict(stream_state)
                self.resumed = True
            except Exception as e:  # unsupported/stale state -- caller falls back to `skip`
                log(f"[{spec.key}] could not restore stream position ({e!r}); "
                    f"falling back to replaying the stream")

    def __iter__(self) -> Iterator[dict]:
        return iter(self.ds)

    def state(self) -> Optional[dict]:
        try:
            return self.ds.state_dict()
        except Exception:
            return None


def _stream_rows(spec: SourceSpec, stream_state: Optional[dict] = None,
                  log=print) -> Iterable[dict]:
    """Returns an iterable of raw rows for `spec`. With `stream_state` (a
    prior `_RowStream.state()`) the stream resumes at that exact position.

    Kept as a plain function -- rather than making callers build `_RowStream`
    themselves -- because it is the seam the whole test suite monkeypatches
    with a one-argument stub returning a plain iterator. Callers therefore
    never assume the result has `.state()`/`.resumed`; they use `getattr`."""
    return _RowStream(spec, stream_state, log=log)


# finepdfs-edu/finemath-3plus-scale sources have a heavy-tailed document-length
# distribution (median ~6k chars, p99 ~130-140k, live outliers past 500k -- a
# live probe against the real Hub data found a single 43k-char document alone
# cost ~330 MB of permanent worker RSS through tokenizer/minhash processing,
# not reclaimed afterward). Truncating the pathological tail bounds worst-case
# per-document memory; it also stops one huge document from dominating dozens
# of consecutive packed training windows. 100k chars is well above the
# packing seq_len's needs (2048 tokens is ~8-10k chars) and left ~98-99% of a
# live finepdfs-edu sample untouched.
_MAX_DOC_CHARS = 100_000


class _DocumentStream:
    """Iterable of a source's usable document texts (filtered, non-empty,
    length-capped) that also tracks its exact position in the underlying
    stream, so a chunk boundary can be resumed later in O(1).

    Resume precedence, and why both mechanisms exist:

    * `stream_state` (a prior `state()`) is the fast path -- `_RowStream`
      restores the position directly and no already-processed row is read
      again beyond the current file. This is the contract `run_dataprep`'s
      respawn loop and `--resume` both use now.
    * `skip` is the O(n) fallback: replay the stream and silently discard the
      first `skip` would-be-yielded documents. It is applied *only* when the
      fast path was unavailable or failed (`_RowStream.resumed` is False), so
      passing both is safe and never double-skips. Kept because a source
      whose `datasets` builder doesn't support `state_dict` must still resume
      correctly rather than duplicating a chunk's worth of documents.

    `state()` is safe to call at any point the iterator is suspended -- i.e.
    right after a `break` out of a `for` over it -- and then points at the
    row *after* the last one yielded (verified live against `fineweb-edu`:
    the resumed stream's first row was byte-identical to the un-resumed
    stream's next row). It returns None before iteration starts (when there
    is nothing new to report it echoes the state it was constructed with)
    and None when the underlying stream can't report a position."""

    def __init__(self, spec: SourceSpec, max_docs: Optional[int] = None, skip: int = 0,
                  stream_state: Optional[dict] = None, log=print):
        self.spec = spec
        self.max_docs = max_docs
        self.skip = skip
        self._initial = stream_state or None
        self._log = log
        self._rows = None
        self._epoch = int((stream_state or {}).get("epoch", 0))
        self._replay_done = True

    def state(self) -> Optional[dict]:
        """`{"epoch": int, "hf_state": {...}}` -- JSON-serializable, so it
        round-trips through the manifest and through the respawn executor's
        pickled arguments unchanged.

        Mid-replay it deliberately reports the state it was constructed with,
        not the live position. During a `skip` replay the live position runs
        ahead of what the caller's counters describe -- the caller's `n_seen`
        already covers the whole skipped prefix, because those documents were
        processed by an earlier chunk. Saving the live position then (say a
        network error interrupts a 1,350,000-document replay after 100,000)
        would make the next resume start from row ~100,000 with `skip` no
        longer applied, re-tokenizing a quarter-million already-flushed
        documents into duplicate shards. Reporting the prior state instead
        just costs a repeated replay, which is the safe direction."""
        if not self._replay_done:
            return self._initial
        getter = getattr(self._rows, "state", None)
        if getter is None:
            return self._initial
        hf_state = getter()
        if hf_state is None:
            return self._initial
        return {"epoch": self._epoch, "hf_state": hf_state}

    def __iter__(self) -> Iterator[str]:
        n = 0
        skipped = 0
        for epoch in range(self._epoch, self.spec.max_epochs):
            self._epoch = epoch
            resume_hf = None
            if epoch == int((self._initial or {}).get("epoch", 0)):
                resume_hf = (self._initial or {}).get("hf_state")
            self._rows = (_stream_rows(self.spec, resume_hf, log=self._log)
                          if resume_hf else _stream_rows(self.spec))
            # Only replay-skip when the O(1) restore didn't happen; see the
            # class docstring's resume-precedence note.
            skip_target = 0 if getattr(self._rows, "resumed", False) else self.skip
            self._replay_done = skipped >= skip_target
            if not self._replay_done:
                # A replay of a source that got far (finephrase reached
                # 1,350,000 documents) takes minutes during which nothing
                # else logs, because `n_seen` only advances for yielded
                # documents. Say so, so a long replay isn't read as a hang.
                self._log(f"[{self.spec.key}] no saved stream position -- replaying "
                          f"{skip_target - skipped:,} already-processed documents to reach the "
                          f"resume point (one-off: a position is saved at the next chunk boundary)")
            for row in self._rows:
                if self.spec.filter_fn is not None and not self.spec.filter_fn(row):
                    continue
                text = self.spec.text_fn(row)
                if not text:
                    continue
                if len(text) > _MAX_DOC_CHARS:
                    text = text[:_MAX_DOC_CHARS]
                if skipped < skip_target:
                    skipped += 1
                    if skipped >= skip_target:
                        self._replay_done = True
                        self._log(f"[{self.spec.key}] replay complete, resuming real work")
                    continue
                yield text
                n += 1
                if self.max_docs is not None and n >= self.max_docs:
                    return


def _documents(spec: SourceSpec, max_docs: Optional[int], skip: int = 0,
                stream_state: Optional[dict] = None, log=print) -> _DocumentStream:
    """See `_DocumentStream`. Returns the stream object (which is iterable,
    so existing `for text in _documents(...)` / `list(_documents(...))` call
    sites are unchanged) rather than a bare generator, because the caller
    needs `state()` to checkpoint the position at a chunk boundary."""
    return _DocumentStream(spec, max_docs=max_docs, skip=skip,
                            stream_state=stream_state, log=log)


class WorkerMemoryExceeded(MemoryError):
    """Raised by `_check_worker_rss` when this process's own resident memory
    exceeds its cap. A `MemoryError` subclass so it reads correctly in a
    traceback; caught by `_run_group_worker`'s existing per-source
    try/except, which records it as a failed source and moves on rather than
    letting one over-sized document (a giant PDF, a huge code file) push the
    whole worker toward swapping the box."""


def _malloc_trim() -> None:
    """Returns freed glibc arena memory to the OS. A live probe against real
    finepdfs-edu streaming (see STATUS.md) found RSS climbing ~1 GB over
    15,000 documents purely from processing overhead (parsing/decoding/
    tokenizing, not the near-dup filter -- pyarrow's own memory pool stayed
    under 200 MB the whole time), consistent with glibc malloc arena
    fragmentation from many differently-sized allocations rather than a
    logical leak: freed blocks sit in the arena instead of being returned to
    the OS. `malloc_trim(0)` is glibc's documented call for exactly this;
    called alongside every RSS check (`_check_worker_rss`) it cut that same
    15,000-document run's growth by roughly 5x live. No-op, never raises, on
    non-glibc libc (e.g. musl) where the symbol doesn't exist."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _measure_rss_gb() -> float:
    """Trims the allocator first (`_malloc_trim`) so a transient fragmentation
    spike doesn't read as real growth, then returns this process's own
    resident memory in GB. Shared by `_check_worker_rss` (the hard cap) and
    `run_source`'s soft-limit check (issue #3's within-source respawn)."""
    _malloc_trim()
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _check_worker_rss(limit_gb: Optional[float]) -> None:
    """Precise, per-worker enforcement of ADDENDUM 2 rule 1 ("never let
    resident memory exceed ~20 GB"). Deliberately checks RSS, not
    `RLIMIT_AS` -- a live measurement showed `import torch; import
    transformers` alone reserves ~4.3 GB of *virtual* address space while
    using well under 1 GB RSS (CUDA/allocator reservations, not real usage),
    so an RLIMIT_AS cap tight enough to matter kills every worker before it
    does any work. See `_set_worker_memory_limit` for the separate, coarse
    RLIMIT_AS backstop this complements.

    Trims the allocator first (`_malloc_trim`) so a transient fragmentation
    spike doesn't trip the cap and abandon a healthy source -- see STATUS.md's
    writeup of the fineweb-edu/finepdfs-edu/finemath-3plus incident this
    fixed, where 3 of 9 sources failed at 2-11% of their token budget."""
    if not limit_gb:
        return
    rss_gb = _measure_rss_gb()
    if rss_gb > limit_gb:
        raise WorkerMemoryExceeded(f"worker RSS {rss_gb:.2f} GB exceeded the {limit_gb:.2f} GB cap")


def _recover_source_stats(key: str, out_root: str) -> Optional[dict]:
    """Best-effort recovery of a source's real progress from disk, for use
    when the parent can't get a normal stats dict back (e.g. a sibling
    worker's hard crash -- a raw C-level `malloc` failure, not a catchable
    Python exception -- poisons the whole `ProcessPoolExecutor`:
    `concurrent.futures` marks *every* still-pending future across the pool
    as `BrokenProcessPool`, not just the one that actually died, so a group
    that was making real progress in its own live, unaffected OS process
    gets the exact same "0 tokens" treatment as the one that crashed).

    `run_source` flushes shards via `ShardWriter._flush` continuously,
    independently of and *before* its own end-of-run `write_manifest()`
    call, so real token data (whole `shard_tokens`-sized `.bin` files) can
    be sitting on disk even when the per-source `manifest.json` was never
    written, or was truncated mid-write by the same crash. Caught live: the
    eighth incident (see STATUS.md/COSTS.md) reported `stack-edu-python`,
    `finephrase`, and `finepdfs-edu` at 0 tokens in `data/manifest.json`
    despite 700M/300M/400M real tokens (7/3/4 complete shards) already on
    disk for each. Falls back from the per-source manifest to scanning
    `.bin` files directly when that manifest is missing or corrupt. The
    very last, not-yet-flushed partial shard (still in the writer's
    in-memory buffer when the process died) is genuinely unrecoverable and
    is not counted -- but every whole flushed shard is real, already-on-disk
    data and must never be reported as 0 tokens."""
    source_dir = os.path.join(out_root, key)
    manifest_path = os.path.join(source_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                m = json.load(f)
            return {"key": key, "dataset": m.get("source_dataset"),
                    "config": m.get("source_config"),
                    "tokens": m.get("total_tokens", 0), "shards": m.get("shards", []),
                    # Written by `run_source` for exactly this case: with it,
                    # a hard-crashed source resumes at its last chunk
                    # boundary instead of re-streaming from row 0.
                    "stream_state": m.get("stream_state"),
                    "n_seen": m.get("n_seen", 0), "n_kept": m.get("n_kept", 0),
                    "n_too_short": m.get("n_too_short", 0)}
        except (OSError, json.JSONDecodeError):
            pass  # truncated by the same crash -- fall through to scanning .bin files
    if not os.path.isdir(source_dir):
        return None
    shard_files = sorted(f for f in os.listdir(source_dir)
                          if f.startswith(f"{key}_") and f.endswith(".bin"))
    if not shard_files:
        return None
    itemsize = np.dtype(TOKEN_DTYPE).itemsize
    shards, total_tokens = [], 0
    for name in shard_files:
        n_tokens = os.path.getsize(os.path.join(source_dir, name)) // itemsize
        shards.append({"file": name, "tokens": n_tokens})
        total_tokens += n_tokens
    return {"key": key, "dataset": None, "tokens": total_tokens, "shards": shards}


def _write_source_manifest(writer, spec: SourceSpec, stream_state, stats: dict) -> str:
    """Write the per-source manifest, including the resume position.

    Durability: this file is the only record that survives a worker dying hard
    (an OS kill, a C-level malloc failure, a clean low-memory abort), so the
    resume position and counters live here, not just in the stats dict
    returned through the executor. See `_recover_source_stats`.
    """
    return writer.write_manifest({
        "source_dataset": spec.dataset, "source_config": spec.config,
        "source_key": spec.key, "eos_id": DEFAULT_EOS_ID,
        "stream_state": stream_state,
        "n_seen": stats["n_seen"], "n_kept": stats["n_kept"],
        "n_too_short": stats["n_too_short"],
    })


def run_source(spec: SourceSpec, tokenizer, token_budget: int, out_dir: str,
                dedup: DedupState, eval_ngram_index: Optional[set],
                shard_tokens: int = 100_000_000, max_docs: Optional[int] = None,
                min_chars: int = 200, progress_every: int = 200_000,
                rss_limit_gb: Optional[float] = None, rss_check_every: int = 5_000,
                checkpoint_every: int = 50_000,
                rss_soft_limit_gb: Optional[float] = None,
                resume_skip: int = 0, resume_seed: Optional[dict] = None,
                resume_stream_state: Optional[dict] = None,
                log=print) -> dict:
    """`rss_soft_limit_gb` (issue #3's within-source respawn fix) is a second,
    *lower* threshold checked alongside the existing hard `rss_limit_gb` cap.
    Crossing it stops this source cleanly -- no exception, shards flushed and
    manifested as usual -- and marks the result `incomplete=True` with a
    `resume_skip` count instead of `error`, so the caller (`_run_group_worker`
    / `run_dataprep`) can hand the rest of this source to a *fresh* forked
    process via `resume_skip`/`resume_seed` rather than either aborting it
    (the old contract) or letting one long-lived worker's RSS climb
    unboundedly until it hits the hard cap. Measured live: `finemath-3plus`,
    `infiwebmath-3plus`, and `finephrase` all hit the hard 4.0 GB cap only
    very late (95%, 88%, 44% through their token budget respectively) despite
    `_run_group_worker`'s group-boundary trim (`3527c9f`) -- proof the growth
    is substantially *within* a single long-running source, not just
    reused-worker carryover between groups, so only a real process respawn
    (not another `gc.collect`/`malloc_trim`) fixes it. See issue #3.

    `resume_stream_state`/`resume_skip`/`resume_seed` resume a source's own
    prior chunk. `resume_stream_state` (that chunk's returned
    `stats["stream_state"]`) restores the exact stream position in O(1) and
    is the path every caller uses now; `resume_skip` is the O(n) replay
    fallback, applied only if the restore was unavailable or failed -- see
    `_DocumentStream`'s resume-precedence note for why both exist and why
    passing both never double-skips. `resume_seed` (that prior chunk's own
    returned `stats`, used verbatim) seeds this call's counters and
    `ShardWriter` so shards accumulate instead of restarting at `_00000`.
    All three unset (the defaults) reproduce the original, unchunked
    behavior exactly.

    The returned `stats` always carries a fresh `stream_state` when the
    stream could report one, and that state is also written into the
    per-source shard manifest, so progress survives not just a soft stop but
    a hard worker crash -- `_recover_source_stats` reads it back and the next
    `--resume` continues from it instead of re-streaming from row 0."""
    resume_seed = resume_seed or {}
    # `resume_seed` is shaped like a prior chunk's own returned `stats` dict
    # (its "tokens" key, not ShardWriter.resume_from's "total_tokens") -- so
    # a caller can pass that prior stats dict straight through unmodified.
    writer = ShardWriter(out_dir, shard_tokens=shard_tokens, prefix=spec.key,
                          resume_from={"shards": resume_seed["shards"],
                                       "total_tokens": resume_seed.get("tokens", 0)}
                          if resume_seed.get("shards") else None)
    group = spec.near_dup_group or spec.key
    stats = {"key": spec.key, "dataset": spec.dataset, "config": spec.config,
             "n_seen": resume_seed.get("n_seen", 0), "n_kept": resume_seed.get("n_kept", 0),
             "n_too_short": resume_seed.get("n_too_short", 0), "tokens": resume_seed.get("tokens", 0)}
    prior_elapsed_s = resume_seed.get("elapsed_s", 0.0)
    t0 = time.time()
    incomplete = False
    exhausted = False
    stream = _documents(spec, max_docs, skip=resume_skip,
                         stream_state=resume_stream_state, log=log)
    try:
        if stats["tokens"] < token_budget:
            for text in stream:
                stats["n_seen"] += 1
                if len(text) < min_chars:
                    stats["n_too_short"] += 1
                    continue
                if not dedup.keep(text, group, eval_ngram_index):
                    continue
                ids = tokenize_document(tokenizer, text, DEFAULT_EOS_ID)
                writer.write(ids)
                stats["n_kept"] += 1
                stats["tokens"] += len(ids)
                if stats["n_seen"] % progress_every == 0:
                    log(f"[{spec.key}] seen={stats['n_seen']:,} kept={stats['n_kept']:,} "
                        f"tokens={stats['tokens']:,}/{token_budget:,} "
                        f"elapsed={time.time() - t0:.0f}s")
                if checkpoint_every and stats["n_seen"] % checkpoint_every == 0:
                    # Durable progress, independent of RSS. Before this, the
                    # per-source manifest -- the only thing that survives a
                    # hard worker death -- was written *once*, when the source
                    # stopped. That was tolerable while a 3.0 GB soft limit
                    # made stops frequent; raising it to 7.0 GB to cure the
                    # respawn thrashing silently removed the checkpoint
                    # cadence with it, and a clean low-memory abort ~50 min
                    # into attempt 9 threw away every token written since the
                    # source started. Flush first so the manifest describes
                    # exactly what is on disk.
                    writer.flush_partial()
                    _write_source_manifest(writer, spec, stream.state(), stats)
                if stats["n_seen"] % rss_check_every == 0:
                    _check_worker_rss(rss_limit_gb)
                    if rss_soft_limit_gb:
                        rss_gb = _measure_rss_gb()
                        if rss_gb > rss_soft_limit_gb:
                            incomplete = True
                            log(f"[{spec.key}] soft RSS stop: {rss_gb:.2f} GB > "
                                f"{rss_soft_limit_gb:.2f} GB soft limit at seen={stats['n_seen']:,} "
                                f"tokens={stats['tokens']:,}/{token_budget:,} -- will resume in a "
                                f"fresh worker")
                            break
                if stats["tokens"] >= token_budget:
                    break
            else:
                # for-else: the stream ran out rather than the loop breaking on
                # the budget or a soft stop. Worth recording, because since
                # `_demote_short_sources` a source below its budget is retried
                # on the next --resume -- and an exhausted source is below its
                # budget permanently, so without this marker it would be
                # re-dispatched on every run to find nothing, forever.
                # Exhaustion is a property of the stream, not the budget, so it
                # stays true when the budget is later raised.
                exhausted = True
                log(f"[{spec.key}] stream exhausted at {stats['tokens']:,} tokens of a "
                    f"{token_budget:,} budget -- this source has no more documents")
    except Exception as e:
        # A mid-stream failure (RSS cap, a bad document, a network hiccup)
        # must not orphan shards already flushed to disk -- ADDENDUM 2 rule 2
        # is "stream, flush to disk, free", and a flushed shard is real,
        # usable data no matter why the source stopped early. Record the
        # error but still close/manifest whatever's already on disk below,
        # instead of leaving unmanifested .bin files no loader can see.
        # Caught live: a 6th sweep-scale dataprep attempt tripped the RSS cap
        # on fineweb-edu after 300M/750M tokens (3 full shards already
        # flushed) and the old code discarded them -- see STATUS.md.
        stats["error"] = repr(e)
        log(f"FAILED source {spec.key}: {e!r}; {stats['tokens']:,} tokens "
            f"already flushed will still be manifested")
    # Snapshot the position *before* anything else can advance the stream.
    # `stream` is suspended at whichever `break`/exception ended the loop, so
    # this points at the first row not yet accounted for in `stats`.
    stream_state = stream.state()
    writer.close()
    manifest_path = _write_source_manifest(writer, spec, stream_state, stats)
    stats.update(shards=writer.shards, manifest_path=manifest_path,
                 stream_state=stream_state,
                 elapsed_s=prior_elapsed_s + (time.time() - t0), token_budget=token_budget,
                 achieved_fraction=(stats["tokens"] / token_budget) if token_budget else 0.0)
    if incomplete and "error" not in stats:
        stats["incomplete"] = True
        stats["resume_skip"] = stats["n_seen"]
    if exhausted and "error" not in stats:
        stats["exhausted"] = True
    return stats


# ----------------------------------------------------------- orchestration ---

# Set by `run_dataprep` right before the pool starts; forked workers inherit it
# via copy-on-write. See the module docstring's "Parallelism" section for why
# this exists instead of passing `SourceSpec` objects through the pool's queue.
_ACTIVE_MIXTURE_BY_KEY: Dict[str, SourceSpec] = {}


# RLIMIT_AS backstop for worker processes -- deliberately coarse, see
# `_set_worker_memory_limit`'s docstring for why it can't be tight. Not a
# CLI-tunable knob: `run_source`'s `_check_worker_rss` (driven by
# `per_worker_mem_limit_gb`) is the precise, tunable enforcement of rule 1;
# this is only a hard stop against truly pathological virtual-memory growth.
_WORKER_VMEM_HARD_CAP_GB = 12.0

# Address space a worker needs *above* its resident budget. This used to be
# folded into the constant above, which made the backstop silently
# incompatible with any RSS cap larger than ~4 GB.
#
# Measured on attempt 7's live workers: VSZ 9.40-9.74 GB while RSS was only
# 2.19-2.87 GB -- roughly 6.5-7.5 GB of address space reserved but not
# resident (torch/CUDA allocator arenas, thread stacks, the tokenizers and
# pyarrow Rust heaps). That overhead is a property of the imports, not of how
# much data the worker is holding, so it does not shrink when the RSS budget
# is small and it does not grow when the RSS budget is large.
#
# Attempt 8 raised `per_worker_mem_limit_gb` 4.0 -> 8.0 while this stayed
# pinned at 12.0. A worker that legitimately grew toward its new 8 GB resident
# budget crossed 12 GB of *address space* long before it got there and died on
# a raw `memory allocation of 1048576 bytes failed` inside Rust -- an
# uncatchable hard crash, which poisons the whole `ProcessPoolExecutor` and
# marked all six in-flight groups `BrokenProcessPool` at once. 12 GB of
# headroom covers the measured 7.5 GB with room for allocator fragmentation
# in a long-lived worker.
#
# Raising an *address space* cap is not a RAM risk: physical memory stays
# bounded by `_check_worker_rss` (per worker) and by the parent's
# `min_available_gb` poll (system-wide). RLIMIT_AS only ever governed VIRT.
_WORKER_VMEM_HEADROOM_GB = 12.0


def worker_vmem_cap_gb(rss_limit_gb: Optional[float]) -> float:
    """Address-space cap that leaves room for `rss_limit_gb` of real data.

    Must scale with the resident cap, or the two guards contradict each other
    and the coarse one wins by killing the process uncatchably.
    """
    if not rss_limit_gb:
        return _WORKER_VMEM_HARD_CAP_GB
    return max(_WORKER_VMEM_HARD_CAP_GB, rss_limit_gb + _WORKER_VMEM_HEADROOM_GB)


def _set_worker_memory_limit(limit_gb: float = _WORKER_VMEM_HARD_CAP_GB) -> None:
    """Caps this process's virtual address space (ADDENDUM 2 rule 6: "a
    runaway allocation fails loudly with a traceback you can read, instead of
    silently wedging the machine"). Deliberately generous: a live measurement
    showed `import torch; import transformers` alone reserves ~4.3 GB of
    *virtual* address space (CUDA/allocator reservations) while using well
    under 1 GB of actual resident memory -- a cap anywhere near this box's
    ~2-3 GB real per-worker RSS budget kills every worker on import, before
    it does any work (caught live: the first version of this guard set 2.5
    GB and every worker died with `MemoryError` inside `import transformers`,
    not from a real leak). This backstop exists only to catch truly
    pathological growth; `_check_worker_rss`/`per_worker_mem_limit_gb` is the
    real, tunable rule-1 enforcement, checked against RSS not VIRT. No-op if
    `limit_gb` is falsy or the sandbox forbids adjusting RLIMIT_AS (soft==hard
    means this process can never raise it again, which is fine: workers are
    short-lived, one group's worth of work)."""
    if not limit_gb:
        return
    limit_bytes = int(limit_gb * (1024 ** 3))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError):
        pass


def _run_group_worker(group_key: str, spec_keys: List[str], target_tokens: int, out_root: str,
                       shard_tokens: int, max_docs_per_source: Optional[int],
                       eval_ngram_index: Optional[set], rss_limit_gb: Optional[float] = None,
                       rss_soft_limit_gb: Optional[float] = None,
                       resume_state: Optional[Dict[str, dict]] = None,
                       rss_check_every: int = 5_000, checkpoint_every: int = 50_000):
    """Runs every source in one near-dup group sequentially, inside one forked
    worker process, sharing a single `DedupState` -- so cross-source dedup
    within the group (e.g. fineweb-edu+dclm-baseline's ~32% overlap) still
    works. Looks specs up from `_ACTIVE_MIXTURE_BY_KEY` (inherited via fork)
    rather than receiving them as arguments, since their text_fn/filter_fn
    closures aren't picklable.

    Setup (the memory limit, the tokenizer load, `DedupState()`) is wrapped
    the same way as each individual source below: a failure here must not
    propagate through `fut.result()` in the parent, which would raise out of
    `run_dataprep` entirely and kill every *other* group's in-flight progress
    too, not just this one -- a strictly worse outcome than one failed
    source. Caught live: a worker died in `get_tokenizer()`'s `import
    transformers` before this fix existed, taking down all 6 concurrent
    groups at once instead of just the one (see STATUS.md).

    With more groups than workers, `ProcessPoolExecutor` reuses each OS
    process for multiple group tasks over its lifetime (see the "reused
    worker" note by the executor's construction in `run_dataprep`) --
    `DedupState` and everything `run_source` allocated for this group is
    dropped and trimmed (`_malloc_trim`, see issue #3) before returning, so
    the next group handed to this same process starts from close to a clean
    RSS baseline instead of inheriting this group's fragmentation. Caught
    live: without this, `cosmopedia-v2` and `finewiki-en` -- both small,
    light sources -- tripped the RSS cap within their first ~5-25K documents
    after landing on a worker that had just finished a heavy group
    (`stack-edu-python`), while a source given a fresh worker sailed past
    the same cap with room to spare (see issue #3).

    `resume_state` (issue #3's within-source respawn fix) maps a `spec_keys`
    entry to its prior chunk's `{"resume_skip", "resume_seed"}` -- passed
    straight through to `run_source` for that key so a source that soft-
    stopped (see `rss_soft_limit_gb`/`run_source`'s docstring) continues from
    where it left off instead of restarting from row 0. Keys not in
    `resume_state` run fresh, exactly as before; `resume_state=None` (the
    default) reproduces the original, unchunked behavior exactly."""
    resume_state = resume_state or {}
    try:
        # Scaled to this worker's resident budget -- a fixed cap here crashed
        # every group of attempt 8 uncatchably. See `worker_vmem_cap_gb`.
        _set_worker_memory_limit(worker_vmem_cap_gb(rss_limit_gb))
        tokenizer = get_tokenizer()
        dedup = DedupState()
    except Exception as e:
        print(f"FAILED group {group_key} setup: {e!r}; recording every source as failed")
        results = []
        for k in spec_keys:
            stats = _recover_source_stats(k, out_root)
            if stats is None:
                stats = {"key": k, "dataset": _ACTIVE_MIXTURE_BY_KEY[k].dataset, "tokens": 0}
            stats["error"] = repr(e)
            results.append(stats)
        gc.collect()
        _malloc_trim()
        return group_key, results, {}

    results = []
    rss_exceeded = False
    for i, key in enumerate(spec_keys):
        spec = _ACTIVE_MIXTURE_BY_KEY[key]
        token_budget = int(round(spec.share * target_tokens))
        source_out = os.path.join(out_root, spec.key)
        if rss_exceeded:
            # A prior source in this group already tripped its RSS cap, so
            # this process's memory is already elevated -- see the eighth
            # incident (STATUS.md/COSTS.md): starting the next source
            # anyway is exactly what produced a raw C-level `malloc`
            # failure moments after fineweb-edu's own graceful RSS-cap
            # failure, which then took the whole worker process down hard
            # enough to poison every OTHER group's in-flight progress too
            # (see _recover_source_stats's docstring). Bail out of this
            # group's remaining sources instead of pressing on; a fresh
            # worker on the next --resume attempt gets a clean budget.
            print(f"skipping {spec.key}: an earlier source in group {group_key} "
                  f"already exceeded its RSS cap in this worker; not risking a "
                  f"harder crash by starting another source in the same process")
            stats = _recover_source_stats(spec.key, out_root)
            if stats is None:
                stats = {"key": spec.key, "dataset": spec.dataset, "tokens": 0}
            stats["error"] = "skipped: an earlier source in this worker exceeded its RSS cap"
            results.append(stats)
            continue
        print(f"=== {spec.key} ({spec.dataset}) share={spec.share:.3f} "
              f"budget={token_budget:,} tokens (group={group_key}) ===")
        resume = resume_state.get(key) or {}
        if resume:
            how = ("stream-position restore (O(1))" if resume.get("stream_state")
                   else f"replaying {resume.get('resume_skip', 0):,} docs (no saved stream position)")
            print(f"resuming {spec.key} from a prior chunk via {how}: "
                  f"{resume.get('resume_seed', {}).get('tokens', 0):,} tokens already flushed")
        if i > 0:
            _malloc_trim()  # reclaim fragmentation from the prior source in this same call
        try:
            stats = run_source(spec, tokenizer, token_budget, source_out, dedup,
                                eval_ngram_index, shard_tokens=shard_tokens,
                                max_docs=max_docs_per_source, rss_limit_gb=rss_limit_gb,
                                rss_soft_limit_gb=rss_soft_limit_gb,
                                rss_check_every=rss_check_every,
                                checkpoint_every=checkpoint_every,
                                resume_skip=resume.get("resume_skip", 0),
                                resume_seed=resume.get("resume_seed"),
                                resume_stream_state=resume.get("stream_state"))
        except Exception as e:  # a broken source must not sink the whole run
            print(f"FAILED source {spec.key}: {e!r}; recording and continuing")
            stats = _recover_source_stats(spec.key, out_root)
            if stats is None:
                stats = {"key": spec.key, "dataset": spec.dataset, "tokens": 0}
            stats["error"] = repr(e)
        if isinstance(stats.get("error"), str) and "WorkerMemoryExceeded" in stats["error"]:
            rss_exceeded = True
        results.append(stats)
    dedup_counters = dict(dedup.counters)
    del dedup, tokenizer
    gc.collect()
    _malloc_trim()
    return group_key, results, dedup_counters


def _terminate_children(log) -> None:
    """Terminate every live child process of this process. Used by the
    low-memory abort path: `ProcessPoolExecutor` exposes no public API to
    kill in-flight work, so we reach for the real process tree instead
    (ADDENDUM 2 rule 4 -- "a clean stop you can diagnose beats a hang the
    operator must reboot")."""
    children = psutil.Process().children(recursive=True)
    for p in children:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    if children:
        log(f"terminated {len(children)} in-flight worker process(es)")


def _tokens_on_disk(out_root: str) -> int:
    """Live token count from shard file sizes, for progress reporting only.

    `manifest["sources"]` is only updated when a *source* finishes, so
    `total_tokens` reads exactly flat for hours during a long build -- which is
    the one number the operator checks from a phone, and a flat line is
    indistinguishable from a hang (this is the reporting gap called out in the
    agent instructions). Shards are `uint16`, so bytes/2 is the token count,
    and a `stat()` per shard file is cheap enough for a 15 s poll.

    Best-effort and never raises: a file being written concurrently, or a
    directory disappearing mid-scan, must not take down the run it is only
    reporting on.
    """
    total = 0
    try:
        for entry in os.scandir(out_root):
            if not entry.is_dir():
                continue
            try:
                for f in os.scandir(entry.path):
                    if f.name.endswith(".bin"):
                        total += f.stat().st_size // 2
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _tree_rss_gb() -> float:
    """This process's + all live children's combined RSS, in GB -- the same
    metric `measure_full2.py`'s live incident-verification script polled.
    Used only for W&B progress reporting; never raises."""
    try:
        me = psutil.Process()
        procs = [me] + me.children(recursive=True)
        return sum(p.memory_info().rss for p in procs if p.is_running()) / (1024 ** 3)
    except psutil.Error:
        return 0.0


def _demote_short_sources(mixture: List[SourceSpec], manifest: dict, target_tokens: int,
                          out_root: str, done_keys: set, resume_state_by_key: Dict[str, dict],
                          log=print) -> None:
    """Un-finish sources whose recorded tokens fall short of this run's budget.

    "Done" has to mean *reached the budget this run asks for*, not merely
    *has a manifest entry*. The old rule -- an entry without an error is
    finished -- is right for crash-resume, where the budgets are unchanged and
    the only question is where to continue, and silently wrong for a top-up:
    raising a source's `--source-budget` could never take effect, because the
    source was skipped on the strength of its older, smaller entry.

    That is how the 60B corpus top-up "succeeded" in twelve seconds having
    written nothing (2026-08-10 18:03Z). All ten sources were short of their
    new budgets -- 3,499,030,510 tokens short in total, precisely the amount
    the run existed to add -- and all ten were skipped as already recorded.

    Continuing is only safe where a real resume point exists. `run_source`
    seeds its counters and its `ShardWriter` from `resume_seed`, so shards
    accumulate from the right index; but a source with shards and *no* saved
    position would restart at row 0 with the writer overwriting from `_00000`,
    destroying tokens already paid for -- fineweb-edu alone is 3.75B. So a
    short source with no position stays done and says so. Falling short is
    recoverable by running again; overwriting the corpus is not.

    Mutates `done_keys` and `resume_state_by_key` in place, matching how
    `run_dataprep` already assembles both.
    """
    by_key = {s["key"]: s for s in manifest.get("sources", []) if "key" in s}
    for spec in mixture:
        entry = by_key.get(spec.key)
        if spec.key not in done_keys or not entry:
            continue
        # The identical formula to the worker's own (`_run_group_worker`), so
        # the two cannot disagree by a rounding step about what "short" means.
        budget = int(round(spec.share * target_tokens))
        have = entry.get("tokens", 0)
        if have >= budget:
            continue
        if entry.get("exhausted"):
            # Below budget permanently: the stream has no more documents, so
            # re-dispatching would replay it to find nothing on every run.
            log(f"{spec.key}: {have:,} tokens of a {budget:,} budget, but the stream is "
                f"exhausted -- nothing more to fetch")
            continue
        stream_state = entry.get("stream_state")
        if not stream_state:
            stream_state = (_recover_source_stats(spec.key, out_root) or {}).get("stream_state")
        n_seen = entry.get("n_seen", 0)
        if not stream_state and not n_seen:
            log(f"{spec.key}: {have:,} tokens on disk against a {budget:,} budget, but no "
                f"resume position -- leaving it complete rather than restarting from row 0 "
                f"and overwriting {have:,} tokens")
            continue
        done_keys.discard(spec.key)
        resume_state_by_key[spec.key] = {"resume_skip": n_seen,
                                          "resume_seed": entry,
                                          "stream_state": stream_state}
        how = "O(1) stream restore" if stream_state else f"replaying {n_seen:,} docs"
        log(f"{spec.key}: {have:,} tokens on disk against a {budget:,} budget -- "
            f"continuing for {budget - have:,} more ({how})")


def run_dataprep(target_tokens: int = 45_000_000_000, out_root: str = "data/shards",
                  manifest_path: str = "data/manifest.json", hf_repo: Optional[str] = None,
                  hf_token: Optional[str] = None, max_docs_per_source: Optional[int] = None,
                  shard_tokens: int = 100_000_000, resume: bool = True,
                  eval_task_limit: Optional[int] = 2000, skip_decontam: bool = False,
                  mixture: Optional[List[SourceSpec]] = None, log=print,
                  max_workers: int = 4, per_worker_mem_limit_gb: Optional[float] = 4.0,
                  rss_soft_limit_gb: Optional[float] = None, rss_check_every: int = 5_000,
                  checkpoint_every: int = 50_000,
                  min_available_gb: float = 6.0, mem_poll_interval_s: float = 15.0,
                  wandb_enabled: bool = True, wandb_project: Optional[str] = None,
                  wandb_entity: Optional[str] = None, run_name: Optional[str] = None) -> dict:
    """Runs every `near_dup_group` in `mixture` (default `MIXTURE`) as its own
    process, up to `max_workers` concurrently -- see the module docstring's
    "Parallelism" section for why (a purely sequential run was measured at
    100+ hours for the full ~45B-token target). Checkpoints `manifest_path`
    after each group completes, so a `--resume` rerun skips finished sources
    rather than restarting the whole job.

    A source that crosses the *soft* `rss_soft_limit_gb` threshold (issue
    #3's within-source respawn fix, see `run_source`'s docstring) is handled
    automatically, inside this same call: its partial progress (`resume_skip`/
    `resume_seed`) is handed to a brand-new, freshly forked single-worker
    `ProcessPoolExecutor` -- a real OS-level process respawn, not just
    `gc.collect`/`malloc_trim` -- so a source too large to finish in one
    worker's RSS budget still completes without operator intervention.
    `rss_soft_limit_gb=None` (the default) disables this entirely, reproducing
    the original all-or-nothing-per-source behavior exactly.

    A source that instead trips the *hard* `per_worker_mem_limit_gb` cap (an
    `error`-recorded entry) ends this call with real progress -- shards on
    disk plus a full `n_seen`/`tokens` stats dict -- but no in-process
    continuation happens for it (the worker that held it is gone). That
    progress is not lost, though: the next external `--resume` recovers any
    such entry with `tokens>0` and on-disk `shards` into `resume_state_by_key`
    and seeds it into the fresh run's *initial* group dispatch via the same
    resume_skip/resume_seed path a live soft-stop uses -- a real cross-
    process resume, not a restart from row 0. Four dataprep-full attempts in
    a row (see COSTS.md) discarded every errored entry this way before this
    fix; see issue #3. Only an errored entry with zero recorded progress
    (nothing ever got flushed) is dropped and redone from scratch, same as
    always.

    RAM discipline (ADDENDUM 2, added after a real incident where 9-10
    concurrent workers thrashed the box's 31.2 GB of system RAM until it
    became unreachable): every worker gets a generous `RLIMIT_AS` backstop
    (`_set_worker_memory_limit`, rule 6) plus a precise, tunable resident-
    memory cap enforced from inside `run_source` (`per_worker_mem_limit_gb`,
    checked against RSS every `rss_check_every` documents -- see
    `_check_worker_rss`'s docstring for why RLIMIT_AS alone can't be tight:
    torch/transformers reserve several GB of *virtual* address space on
    import alone, well above any real per-worker RSS budget). The main
    process also polls system memory every `mem_poll_interval_s` seconds; if
    available memory drops below `min_available_gb`, all in-flight workers
    are terminated, the reason is recorded in the manifest, and the function
    returns cleanly instead of continuing toward a hang (rule 4). Default
    `max_workers=4` with `per_worker_mem_limit_gb=4.0` bounds worst-case
    worker RSS at 4*4.0=16 GB, leaving headroom under the ~20 GB
    resident-memory ceiling (rule 1) for the parent process and the OS -- see
    STATUS.md for the measured (not guessed, per rule 3) per-worker RSS this
    is based on.

    These were `max_workers=6`/`per_worker_mem_limit_gb=2.5` until a live
    2B-token production run showed the cap was too tight: each forked
    worker's *baseline* RSS (torch/transformers/tokenizers/datasets state,
    duplicated per worker by Python refcount-triggered copy-on-write despite
    `fork`) reached 2.6-2.9 GB within the first couple of minutes, before
    the source's own data had a chance to dominate -- including on
    `fineweb-edu` itself, the critical-path source. `_check_worker_rss`
    correctly killed those sources per its contract, but a cap tighter than
    real baseline overhead meant most sources would have failed this way,
    reproducing the "run completes cleanly but yields ~0 useful tokens"
    failure mode for a new reason. See STATUS.md's "Fifth incident".

    Progress is logged to W&B (project/entity default to `$WANDB_PROJECT`/
    `$WANDB_ENTITY`, tag "dataprep") once per `mem_poll_interval_s` tick --
    tokens/sources completed so far, groups still running, tree RSS, and
    available memory -- so a multi-hour run shows a live, monotonically
    advancing signal on the operator's phone instead of an idle-looking
    dashboard (see the reporting-gaps addendum in the agent instructions).
    Uses `daedalus.wandb_logger.WandbLogger` (not `train.py`'s re-export --
    see that module's docstring for why importing torch into this parent
    process before it forks workers is actively unsafe here) and degrades to
    offline the same way; never blocks or crashes the run."""
    global _ACTIVE_MIXTURE_BY_KEY
    mixture = mixture if mixture is not None else MIXTURE
    _ACTIVE_MIXTURE_BY_KEY = {s.key: s for s in mixture}

    eval_ngram_index = None
    if not skip_decontam:
        log("building eval n-gram decontamination index ...")
        eval_ngram_index = _build_eval_index(limit=eval_task_limit)
        log(f"eval n-gram index: {len(eval_ngram_index):,} 13-grams")

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    manifest = {
        "target_tokens": target_tokens, "sources": [],
        "substitutions": GATED_SUBSTITUTION_NOTES,
        "dedup": {
            "exact": "64-bit blake2b hash set over normalized text, scoped per near_dup_group "
                     "(one process per group -- see daedalus/dataprep.py docstring's Parallelism section)",
            "near_dup": f"MinHashLSH threshold={DedupState.threshold} num_perm={DedupState.num_perm}, "
                        f"reset every {DedupState.near_dup_reset_every} docs per source-group "
                        f"(bounded memory approximation, see daedalus/dataprep.py docstring)",
        },
        "decontam": ("8/13-gram overlap vs HellaSwag/ARC-Easy/PIQA/OpenBookQA/WinoGrande"
                     if not skip_decontam else "skipped (skip_decontam=True)"),
    }
    if resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        log(f"resuming from {manifest_path}: {len(manifest.get('sources', []))} source(s) already done")
        # The resumed file carries the *previous* run's target. Overwrite it,
        # or a run that deliberately retargets the corpus (e.g. 45B -> 13B
        # balanced, issue #4 section 4.2) would keep reporting the old,
        # abandoned goal as its denominator everywhere downstream.
        manifest["target_tokens"] = target_tokens
        # Same argument, and the same bug one field over: these describe the
        # run that *wrote* the file, and only the abort path ever sets them.
        # Resuming loaded them back and the success path re-serialized them
        # untouched, so one low-memory abort marked the manifest aborted
        # permanently -- every later run inherited the claim no matter how
        # cleanly it finished. Attempt 10 completed all ten sources
        # ("=== done fineweb-edu: tokens=3,750,002,609 ===") and still wrote
        # `aborted_low_memory: true` with a reason naming "5 group(s) still
        # running", which it never had -- it ran with --max-workers 2. Nothing
        # reads these in code, so the cost is purely a false signal to whoever
        # reads the manifest next; it misled me for several minutes at 00:14Z
        # on 2026-08-10 into thinking the corpus build had died.
        manifest.pop("aborted_low_memory", None)
        manifest.pop("aborted_reason", None)

    # A source that hit an error (e.g. an RSS-cap trip) only got a partial
    # share of its token budget -- see STATUS.md's malloc_trim incident,
    # where finepdfs-edu and finemath-3plus stopped at 11% and 2.8%. Treating
    # its manifest entry as "done" would skip it forever on every future
    # --resume.
    #
    # Its `stats` dict -- whether it came from a live soft RSS stop or a
    # hard cap trip -- still carries real n_seen/tokens/shards for whatever
    # was actually flushed to disk before the failure (run_source always
    # calls writer.close()/stats.update() on the way out, success or not).
    # Discarding that on every --resume was exactly the problem issue #3's
    # round-two fix addresses: four dataprep-full attempts in a row dropped
    # every errored entry and re-streamed the same documents from byte 0,
    # producing zero carried-forward tokens across ~$2 of failed attempts.
    # So an errored entry with real recorded progress (tokens>0, shards on
    # disk) is now seeded into `resume_state_by_key` and handed to a fresh
    # worker via the same resume_skip/resume_seed continuation path a live
    # soft-stop uses (see the loop below) -- a real cross-process resume, not
    # just the in-run one. An errored entry with zero progress (e.g. it
    # failed before writing anything, or was skipped outright -- see
    # dclm-baseline's "skipped: an earlier source ..." entries) has nothing
    # to recover and is simply dropped, same as before: retried from scratch.
    # Note: dedup_counters is a diagnostic aggregate only (not used to
    # reconstruct shards), so a resumed/retried source's counts stacking on
    # top of its failed attempt's counts is a small, accepted inaccuracy.
    retry_keys = {s["key"] for s in manifest.get("sources", []) if s.get("error")}
    resume_state_by_key: Dict[str, dict] = {}
    if retry_keys:
        log(f"will retry {len(retry_keys)} previously-failed source(s) on resume: {sorted(retry_keys)}")
        recoverable = []
        for s in manifest.get("sources", []):
            if s.get("error") and s.get("tokens", 0) > 0 and s.get("shards"):
                # Prefer the manifest's own saved stream position; fall back
                # to the on-disk per-source manifest, which `run_source`
                # writes at every chunk boundary and therefore still has a
                # position even for an entry recorded by the crash-recovery
                # path (`_recover_source_stats`) before that path knew to
                # carry one.
                stream_state = s.get("stream_state")
                if not stream_state:
                    recovered = _recover_source_stats(s["key"], out_root) or {}
                    stream_state = recovered.get("stream_state")
                resume_state_by_key[s["key"]] = {"resume_skip": s.get("n_seen", 0),
                                                  "resume_seed": s,
                                                  "stream_state": stream_state}
                recoverable.append(s["key"])
        if recoverable:
            log(f"recovering partial progress for {len(recoverable)} source(s) instead of "
                f"restarting from scratch: {sorted(recoverable)}")

    # An errored entry stays in `manifest["sources"]` rather than being
    # deleted here, and is *replaced* when the source is redone (see
    # `_record_source`). Deleting it made the on-disk manifest lose that
    # source's recorded progress the moment this run started -- so a run that
    # died before the source finished (killed, out of credit, box rebooted)
    # left nothing behind, and the next --resume had only whatever the
    # per-source shard manifest happened to hold. That bit for real: killing
    # attempt 5 mid-replay cost `finephrase` and `infiwebmath-3plus` their
    # n_seen (2.57B tokens' worth of resume position), recoverable only from
    # this file's git history. `done_keys` excludes errored entries, so they
    # are still retried exactly as before.
    done_keys = {s["key"] for s in manifest.get("sources", []) if "key" in s and not s.get("error")}

    # A recorded source may still be short of the budget *this* run asks for.
    # See `_demote_short_sources`.
    if resume:
        _demote_short_sources(mixture, manifest, target_tokens, out_root,
                              done_keys, resume_state_by_key, log)

    manifest.setdefault("dedup_counters", {"kept": 0, "exact_dup": 0, "near_dup": 0, "contaminated": 0})

    def _record_source(stats: dict) -> None:
        """Replace this source's manifest entry, or append if it is new."""
        sources = manifest.setdefault("sources", [])
        for i, existing in enumerate(sources):
            if existing.get("key") == stats.get("key"):
                sources[i] = stats
                return
        sources.append(stats)

    if resume:
        # A source that was still *in flight* when the run ended -- killed,
        # out of credit, box rebooted, parent OOM'd -- has no run-level
        # manifest entry at all: `run_dataprep` only appends one when a
        # source finishes, and the loop above only recovers entries that
        # exist and carry an `error`. So the disk holds real shards and a
        # real resume position that nothing was looking at, and the next
        # --resume restarted the source from row 0 *and* had its ShardWriter
        # overwrite those shards from `_00000`. Found by killing a live
        # validation run mid-source: 37,365,481 tokens across 5 shards, with
        # a valid saved position, versus a run-level manifest listing no
        # sources at all.
        #
        # Only seed when disk actually yields a resume point (a saved stream
        # position, or an `n_seen` to replay). Shards with no position at all
        # -- e.g. recovered by scanning .bin files after the per-source
        # manifest was truncated mid-write -- are deliberately left to the
        # from-scratch path: re-doing the source costs time, whereas
        # appending to shards whose stream position is unknown would silently
        # duplicate documents into the corpus.
        for spec in mixture:
            if spec.key in done_keys or spec.key in resume_state_by_key:
                continue
            recovered = _recover_source_stats(spec.key, out_root)
            if not recovered or recovered.get("tokens", 0) <= 0 or not recovered.get("shards"):
                continue
            if not recovered.get("stream_state") and not recovered.get("n_seen"):
                log(f"{spec.key}: {recovered['tokens']:,} tokens on disk but no resume point "
                    f"recorded -- redoing this source from scratch")
                continue
            resume_state_by_key[spec.key] = {"resume_skip": recovered.get("n_seen", 0),
                                              "resume_seed": recovered,
                                              "stream_state": recovered.get("stream_state")}
            log(f"{spec.key}: recovered {recovered['tokens']:,} tokens from an interrupted "
                f"in-flight run ({len(recovered['shards'])} shard(s) on disk) -- continuing "
                f"rather than restarting")

    groups: Dict[str, List[str]] = {}
    for spec in mixture:
        if spec.key in done_keys:
            log(f"skip {spec.key}: already recorded in manifest (resume)")
            continue
        groups.setdefault(spec.near_dup_group or spec.key, []).append(spec.key)

    if groups:
        workers = max(1, min(max_workers, len(groups)))
        log(f"dispatching {len(groups)} group(s) across {workers} worker process(es) "
            f"(per-worker mem limit={per_worker_mem_limit_gb} GB, "
            f"abort if available memory < {min_available_gb} GB)")

        t0 = time.time()
        n_shards_uploaded = 0

        # NOTE: `max_tasks_per_child` (which would force a fresh OS process
        # per group, avoiding RSS carryover across a reused worker -- see
        # STATUS.md's "shard buffer" incident writeup) is NOT used here: it
        # raises ValueError with the 'fork' start method, and fork is
        # required (workers inherit `_ACTIVE_MIXTURE_BY_KEY`'s closures --
        # `text_fn`/`filter_fn` -- via copy-on-write; 'spawn'/'forkserver'
        # can't pickle those or reliably time the inheritance). Reused-worker
        # RSS carryover turned out to still be real with more groups than
        # workers (issue #3: `cosmopedia-v2`/`finewiki-en` tripped the cap
        # almost immediately after landing on a worker that had just
        # finished a heavy group) -- `_run_group_worker` now explicitly
        # `gc.collect()`s and `_malloc_trim()`s before returning, so a reused
        # process hands the next group a close-to-clean RSS baseline. That
        # alone still wasn't enough for sources that grow *within* their own
        # run (issue #3, round 2: finemath-3plus/infiwebmath-3plus/finephrase
        # all hit the hard cap late, at 95%/88%/44% through their budget) --
        # for those, `rss_soft_limit_gb` below gets a genuine process respawn
        # via a throwaway single-worker executor, see the loop below.
        ctx = multiprocessing.get_context("fork")  # required: see Parallelism note in module docstring
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        aborted_reason = None
        throwaway_executors = []  # every one-shot respawn executor ever created, for final cleanup
        try:
            # This process must never initialise wandb -- see
            # `daedalus/wandb_sidecar.py`. wandb starts an asyncio manager on
            # init, and every process forked afterwards inherits one that
            # belongs to the parent, so touching wandb there raises
            # ForkedError. Respawns fork throughout the run by construction,
            # so no amount of ordering saves them: `dataprep-full-attempt5`
            # lost finepdfs-edu, cosmopedia-v2 and finewiki-en that way, each
            # at its first chunk boundary. The W&B run lives in a child
            # process now and this one only ever appends to a file.
            fut_to_group = {
                ex.submit(_run_group_worker, group_key, keys, target_tokens, out_root,
                          shard_tokens, max_docs_per_source, eval_ngram_index,
                          per_worker_mem_limit_gb, rss_soft_limit_gb,
                          {k: resume_state_by_key[k] for k in keys if k in resume_state_by_key} or None,
                          rss_check_every, checkpoint_every,
                          ): (group_key, keys)
                for group_key, keys in groups.items()
            }
            # `ex` never receives another submit() after this -- every continuation
            # goes through its own throwaway `tw_ex` (see `_submit_respawn`). Without
            # this, a worker that finishes its last queued group (by completing OR
            # soft-stopping) just sits idle, holding its trimmed-but-nonzero RSS
            # (~2 GB observed) for the rest of the run, for every one of `workers`
            # slots -- e.g. 4 workers x ~2 GB = ~8 GB permanently stranded on top of
            # the actively-cycling throwaway workers' own budget, found live via a
            # 1-group/1-worker validation where the sole worker had nothing left to
            # do after its first soft-stop. `shutdown(wait=False)` here does NOT
            # cancel the already-submitted futures above (cancel_futures defaults to
            # False) -- it only tells the pool to stop each worker as soon as it
            # drains the queue instead of leaving it idle until the final `finally`.
            ex.shutdown(wait=False)
            fut_to_respawn_executor: Dict = {}  # continuation futures only; main-pool futures absent
            pending = set(fut_to_group)
            n_groups_total = len(pending)
            groups_fully_done = 0

            def _submit_respawn(group_key, continuation_keys, resume_state):
                # A fresh, throwaway single-worker executor -- NOT `ex`, the
                # persistent pool -- so this really forks a brand-new OS
                # process with a clean RSS baseline (see the note above on
                # why `max_tasks_per_child` can't do this with 'fork').
                tw_ctx = multiprocessing.get_context("fork")
                tw_ex = ProcessPoolExecutor(max_workers=1, mp_context=tw_ctx)
                throwaway_executors.append(tw_ex)
                new_fut = tw_ex.submit(
                    _run_group_worker, group_key, continuation_keys, target_tokens, out_root,
                    shard_tokens, max_docs_per_source, eval_ngram_index,
                    per_worker_mem_limit_gb, rss_soft_limit_gb, resume_state, rss_check_every,
                    checkpoint_every)
                fut_to_group[new_fut] = (group_key, continuation_keys)
                fut_to_respawn_executor[new_fut] = tw_ex
                pending.add(new_fut)

            from daedalus.wandb_sidecar import WandbSidecar
            wb = WandbSidecar(
                project=wandb_project or os.environ.get("WANDB_PROJECT", "daedalus"),
                entity=wandb_entity or os.environ.get("WANDB_ENTITY"),
                name=run_name or "dataprep",
                config={"target_tokens": target_tokens, "n_groups": len(groups),
                        "max_workers": workers, "per_worker_mem_limit_gb": per_worker_mem_limit_gb,
                        "rss_soft_limit_gb": rss_soft_limit_gb, "rss_check_every": rss_check_every,
                        "checkpoint_every": checkpoint_every,
                        "min_available_gb": min_available_gb, "shard_tokens": shard_tokens,
                        "skip_decontam": skip_decontam},
                tags=["dataprep"], enabled=wandb_enabled,
                progress_path=os.path.join(os.path.dirname(manifest_path) or ".",
                                            f"{run_name or 'dataprep'}_progress.jsonl"),
                log=log,
            )
            url = wb.run_url(timeout_s=45)
            if url:
                log(f"W&B run: {url}")
            while pending:
                done, pending = wait(pending, timeout=mem_poll_interval_s,
                                      return_when=FIRST_COMPLETED)
                available_gb = psutil.virtual_memory().available / (1024 ** 3)
                wb.log({
                    "total_tokens": sum(s.get("tokens", 0) for s in manifest["sources"]),
                    # Advances every poll instead of only when a source
                    # finishes -- see `_tokens_on_disk`.
                    "tokens_on_disk": _tokens_on_disk(out_root),
                    # Errored entries are kept in `sources` now (they hold
                    # real tokens and a resume position), so "done" counts
                    # only the ones that actually finished.
                    "sources_done": sum(1 for s in manifest["sources"] if not s.get("error")),
                    "groups_done": groups_fully_done,
                    "groups_remaining": len(pending),
                    "tree_rss_gb": _tree_rss_gb(),
                    "available_mem_gb": available_gb,
                    "elapsed_s": time.time() - t0,
                })
                if available_gb < min_available_gb:
                    aborted_reason = (
                        f"available memory {available_gb:.1f} GB dropped below the "
                        f"{min_available_gb:.1f} GB floor with {len(pending)} group(s) still "
                        f"running -- aborted cleanly per ADDENDUM 2 rule 4 instead of risking "
                        f"the box thrashing"
                    )
                    log(f"ABORT: {aborted_reason}")
                    _terminate_children(log)
                    break
                for fut in done:
                    fallback_group_key, fallback_keys = fut_to_group.pop(fut)
                    respawn_ex = fut_to_respawn_executor.pop(fut, None)
                    try:
                        group_key, results, group_counters = fut.result()
                    except Exception as e:
                        # A worker crashing entirely (segfault, OS-killed, an
                        # exception from setup that somehow still escaped
                        # _run_group_worker's own try/except) must not sink
                        # every *other* group's in-flight progress -- same
                        # rationale as _run_group_worker's own per-source and
                        # per-group try/except, one layer further out.
                        log(f"FAILED group {fallback_group_key} (worker crashed): {e!r}; "
                            f"recording and continuing")
                        group_key = fallback_group_key
                        group_counters = {}
                        # concurrent.futures marks EVERY still-pending future across the
                        # whole pool broken when any one worker hard-crashes, even groups
                        # whose own OS process was alive and making real progress -- see
                        # _recover_source_stats's docstring (the eighth incident). Recover
                        # whatever real tokens/shards are already flushed to disk instead
                        # of reporting them as 0.
                        results = []
                        for k in fallback_keys:
                            stats = _recover_source_stats(k, out_root)
                            if stats is None:
                                stats = {"key": k, "dataset": _ACTIVE_MIXTURE_BY_KEY[k].dataset, "tokens": 0}
                            stats["error"] = repr(e)
                            results.append(stats)
                    finally:
                        if respawn_ex is not None:
                            respawn_ex.shutdown(wait=False, cancel_futures=True)

                    continuation_keys = []
                    resume_state: Dict[str, dict] = {}
                    for stats in results:
                        if stats.get("incomplete"):
                            # Soft-stopped, not an error: hand its exact progress to a
                            # fresh respawned process instead of recording it "done" or
                            # making the operator run another --resume by hand.
                            # `stats` is already shaped exactly like what run_source's
                            # `resume_seed` param expects (its own prior return value) --
                            # pass it straight through, no re-keying needed.
                            resume_state[stats["key"]] = {
                                "resume_skip": stats["resume_skip"],
                                "resume_seed": stats,
                                "stream_state": stats.get("stream_state"),
                            }
                            continuation_keys.append(stats["key"])
                            log(f"{stats['key']}: soft RSS stop at {stats['tokens']:,}/"
                                f"{stats.get('token_budget', 0):,} tokens ({stats['n_seen']:,} docs "
                                f"seen) -- resuming in a freshly forked process")
                        else:
                            _record_source(stats)
                            log(f"=== done {stats['key']}: tokens={stats.get('tokens', 0):,} ===")
                            if hf_repo and stats.get("shards"):
                                log(f"uploading {stats['key']} shards to {hf_repo} ...")
                                upload_shards(os.path.join(out_root, stats["key"]), hf_repo, token=hf_token)
                                n_shards_uploaded += len(stats["shards"])
                                wb.log({"shards_uploaded": n_shards_uploaded})
                    for k, v in group_counters.items():
                        manifest["dedup_counters"][k] = manifest["dedup_counters"].get(k, 0) + v
                    manifest["total_tokens"] = sum(s.get("tokens", 0) for s in manifest["sources"])
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)

                    if continuation_keys:
                        _submit_respawn(group_key, continuation_keys, resume_state)
                    else:
                        groups_fully_done += 1
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            for tw_ex in throwaway_executors:
                tw_ex.shutdown(wait=False, cancel_futures=True)

        if aborted_reason:
            manifest["aborted_low_memory"] = True
            manifest["aborted_reason"] = aborted_reason
            manifest["total_tokens"] = sum(s.get("tokens", 0) for s in manifest["sources"])
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            wb.log({"aborted_low_memory": True})
            wb.finish()
            return manifest

        wb.log({
            "total_tokens": manifest["total_tokens"],
            "sources_done": sum(1 for s in manifest["sources"] if not s.get("error")),
            "groups_done": groups_fully_done, "groups_remaining": 0,
            "elapsed_s": time.time() - t0,
        })
        wb.finish()

    manifest["total_tokens"] = sum(s.get("tokens", 0) for s in manifest["sources"])
    manifest["achieved_fraction"] = (manifest["total_tokens"] / target_tokens) if target_tokens else 0.0
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# --------------------------------------------------------------------- cli ---

def _cli():
    import argparse

    p = argparse.ArgumentParser(description="AGENT.md dataprep: build the ~45B-token training shard set.")
    p.add_argument("--target-tokens", type=int, default=45_000_000_000)
    p.add_argument("--out", default="data/shards")
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--hf-repo", default=None, help="private HF dataset repo to upload shards to")
    p.add_argument("--max-docs-per-source", type=int, default=None,
                   help="cap documents per source (testing only)")
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--skip-decontam", action="store_true")
    p.add_argument("--eval-task-limit", type=int, default=2000)
    p.add_argument("--max-workers", type=int, default=4,
                   help="concurrent source-group processes (see module docstring's Parallelism note "
                        "and run_dataprep's RAM discipline note -- ADDENDUM 2)")
    p.add_argument("--per-worker-mem-limit-gb", type=float, default=4.0,
                   help="per-worker RSS cap, checked periodically inside run_source; a source that "
                        "pushes this worker over it fails loudly and is recorded, instead of the "
                        "worker silently growing (ADDENDUM 2 rule 1/6). 0 disables. Separate from "
                        "the generous fixed RLIMIT_AS backstop -- see _set_worker_memory_limit.")
    p.add_argument("--rss-soft-limit-gb", type=float, default=None,
                   help="lower, soft RSS threshold (issue #3): crossing it stops the current source "
                        "cleanly and resumes it in a freshly forked process instead of letting RSS "
                        "climb toward --per-worker-mem-limit-gb. Unset disables the respawn path "
                        "entirely (sources run start-to-finish in one process, the original behavior).")
    p.add_argument("--rss-check-every", type=int, default=5_000,
                   help="how many documents between RSS checks (both --per-worker-mem-limit-gb and "
                        "--rss-soft-limit-gb). Growth was measured live to be bursty, not smooth -- up "
                        "to ~0.7 GB within a single 5,000-doc window -- so a respawned worker can cross "
                        "both the soft and hard threshold between two checks if this is too coarse "
                        "relative to the soft/hard margin. See issue #3's diag_growth2 measurements.")
    p.add_argument("--min-available-gb", type=float, default=6.0,
                   help="abort cleanly if system available memory drops below this (ADDENDUM 2 rule 4)")
    p.add_argument("--mem-poll-interval-s", type=float, default=15.0)
    p.add_argument("--checkpoint-every", type=int, default=50_000,
                   help="flush and record the resume position every N documents. "
                        "Bounds how much work a hard crash or a clean abort can "
                        "throw away; before this existed the position was saved "
                        "only when a source stopped.")
    p.add_argument("--source-budget", action="append", default=[], metavar="KEY=TOKENS",
                   help="absolute token budget for one source, repeatable. If given at "
                        "all, the mixture becomes exactly the listed sources with exactly "
                        "these budgets and --target-tokens is ignored (their sum is used "
                        "instead). This is how the corpus gets finished to specific "
                        "absolute per-source numbers rather than to global proportions -- "
                        "see build_budget_mixture and issue #4 section 4.2.")
    p.add_argument("--run-name", default=None, help="W&B run name; defaults to 'dataprep'")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN_WRITE")

    mixture, target_tokens = None, args.target_tokens
    if args.source_budget:
        budgets = {}
        for item in args.source_budget:
            key, _, raw = item.partition("=")
            if not raw:
                p.error(f"--source-budget expects KEY=TOKENS, got {item!r}")
            budgets[key.strip()] = int(raw.replace("_", ""))
        try:
            mixture, target_tokens = build_budget_mixture(budgets)
        except ValueError as e:
            p.error(str(e))
        print(f"per-source budgets given; target is their sum: {target_tokens:,} tokens "
              f"across {len(mixture)} source(s)")
        for s in mixture:
            print(f"  {s.key:26s} {int(round(s.share * target_tokens)):>14,}")

    manifest = run_dataprep(
        target_tokens=target_tokens, out_root=args.out, manifest_path=args.manifest,
        mixture=mixture,
        hf_repo=args.hf_repo, hf_token=hf_token, max_docs_per_source=args.max_docs_per_source,
        shard_tokens=args.shard_tokens, resume=not args.no_resume,
        eval_task_limit=args.eval_task_limit, skip_decontam=args.skip_decontam,
        max_workers=args.max_workers, per_worker_mem_limit_gb=args.per_worker_mem_limit_gb or None,
        rss_soft_limit_gb=args.rss_soft_limit_gb, rss_check_every=args.rss_check_every,
        checkpoint_every=args.checkpoint_every,
        min_available_gb=args.min_available_gb, mem_poll_interval_s=args.mem_poll_interval_s,
        wandb_enabled=not args.no_wandb, wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity, run_name=args.run_name,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    _cli()
