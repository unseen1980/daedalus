"""Daedalus data pipeline: streaming tokenization to uint16 shards, a
document-aligned packed loader, near-dup / contamination filters, and HF Hub
shard sync.

Tokenizer is reused byte-identical from SmolLM2 (AGENT.md SS2) -- never modify
its files, llama.cpp's converter fingerprints the pre-tokenizer by hash.

Packing scheme: documents are tokenized and concatenated back-to-back with an
EOS separator into one flat token stream per shard (the standard dense-packing
scheme used by nanoGPT-speedrun-era trainers -- no padding waste). Any
contiguous `seq_len + 1` window is valid training data. `PackedTokenDataset`
can additionally return per-token document ids (derived from EOS positions) so
a future FlexAttention block-causal mask in train.py/model.py can prevent
cross-document attention; plain dense packing (no masking) is the default and
is what training uses until that lands.
"""
import array
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

TOKEN_DTYPE = np.uint16          # vocab_size 49152 < 65536
SMOLLM2_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
# SmolLM2's tokenizer maps bos/eos/unk all to `<|endoftext|>` (id 0) -- verified
# against the real tokenizer_config, see config.py's to_hf_dict for detail.
DEFAULT_EOS_ID = 0


def get_tokenizer(name: Optional[str] = None):
    """Load a tokenizer -- SmolLM2 byte-identical by default.

    `name` accepts a Hub id or a local directory, so Phase 4's candidate
    vocabularies can be packed and scored through this same pipeline. `None`
    (the default, and what every existing caller passes) resolves to SmolLM2,
    so no shipped run changes behaviour.

    The explicit-path form exists because `vocab_size` is not a tokenizer.
    Two shard directories built under different vocabularies are the same
    dtype, the same shape and the same manifest schema; nothing about reading
    the wrong one raises, it just embeds ids that mean something else. The
    matching guard is `assert_manifest_tokenizer`.

    Import is local so importing this module (e.g. from train.py for
    `PackedTokenDataset`) doesn't require `transformers` unless tokenization is
    actually used.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name or SMOLLM2_TOKENIZER)


def tokenizer_fingerprint(tokenizer, name: Optional[str] = None) -> dict:
    """What a manifest records about the tokenizer that produced its ids.

    The vocabulary size alone is not enough -- two 32,768-token vocabularies
    trained on different samples share it and agree on nothing else -- so the
    fingerprint also carries the ids of the pinned specials and a digest of the
    tokenization of a fixed probe string. That digest is the same trick
    llama.cpp's converter uses to identify a pre-tokenizer, and it is cheap
    enough to check before every pass over a shard directory.

    A tokenizer that does not expose the full `PreTrainedTokenizer` surface --
    the stand-ins several tests pack shards with, for instance -- yields a
    `partial` fingerprint carrying only what it could answer. That is recorded
    rather than faked, and `assert_manifest_tokenizer` compares only fields
    both sides actually have, so a partial fingerprint degrades to the same
    non-guard as a manifest written before this field existed.
    """
    import hashlib

    probe = ("Daedalus tokenizer fingerprint probe\n\tdef f(x):\n"
             "        return x ** 2  # 中文 🚀 ∀x∈ℝ\n")
    fingerprint = {
        "name": name or getattr(tokenizer, "name_or_path", None) or SMOLLM2_TOKENIZER,
    }
    try:
        ids = tokenizer.encode(probe, add_special_tokens=False)
        fingerprint["probe_digest"] = hashlib.sha256(
            str(list(ids)).encode()).hexdigest()[:32]
    except TypeError:
        pass
    for field, read in (("vocab_size", lambda: int(tokenizer.vocab_size)),
                        ("eos_id",
                         lambda: tokenizer.convert_tokens_to_ids("<|endoftext|>"))):
        try:
            fingerprint[field] = read()
        except (AttributeError, TypeError, ValueError):
            pass
    if "probe_digest" not in fingerprint or "vocab_size" not in fingerprint:
        fingerprint["partial"] = True
    return fingerprint


def assert_manifest_tokenizer(manifest: dict, tokenizer, name: Optional[str] = None
                              ) -> None:
    """Refuse a shard directory that was packed under a different tokenizer.

    Silent by default is the whole problem: `PackedTokenDataset` will happily
    serve 32,768-vocabulary ids to a 49,152-row embedding, and the run trains,
    logs a loss and exports. A manifest written before this field existed
    carries no fingerprint and is passed through rather than refused, so older
    corpora keep working.
    """
    recorded = manifest.get("tokenizer")
    if not recorded:
        return
    current = tokenizer_fingerprint(tokenizer, name)
    for field in ("vocab_size", "probe_digest"):
        if field not in recorded or field not in current:
            continue
        if recorded[field] != current[field]:
            raise ValueError(
                f"shard tokenizer mismatch on {field!r}: shards were packed by "
                f"{recorded.get('name')!r} ({recorded.get(field)!r}) but "
                f"{current['name']!r} gives {current[field]!r}. Reading these "
                f"ids under this tokenizer is not an error anything raises -- "
                f"it silently trains on a different vocabulary.")


# --------------------------------------------------------------- streaming ----

def stream_dataset(dataset_name: str, split: str = "train",
                    text_column: str = "text", **load_kwargs) -> Iterator[str]:
    """Yield text strings from a streaming HF dataset. No local download."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=True, **load_kwargs)
    for row in ds:
        text = row.get(text_column)
        if text:
            yield text


def tokenize_document(tokenizer, text: str, eos_id: int) -> List[int]:
    ids = tokenizer.encode(text)
    ids.append(eos_id)
    return ids


# ------------------------------------------------------------------ shards ----

@dataclass
class ShardWriter:
    """Buffers uint16 token ids and flushes fixed-size `.bin` shards to disk.

    The buffer is an `array.array("H")`, not a `list[int]`: a plain Python
    list of `shard_tokens` (100,000,000 by default) boxed ints costs ~3.6 GB
    of RSS (measured live -- each boxed `PyLong` outside the small-int cache
    is ~28 bytes, plus 8 bytes/slot for the list's own pointer array), while
    `array.array("H")` stores the same 100M uint16 values packed, ~200 MB --
    an ~18x reduction. This was the dominant cause of a live sweep-scale
    `dataprep` run OOMing every worker (see STATUS.md/COSTS.md, "shard
    buffer" incident): any source processing more than one shard's worth of
    tokens grew this buffer to multiple GB before its first flush."""
    out_dir: str
    shard_tokens: int = 100_000_000   # ~200 MB per shard at uint16
    prefix: str = "shard"
    resume_from: Optional[dict] = None
    """Seeds shard numbering/totals from a prior chunk's already-flushed
    shards (dataprep.py's within-source RSS-respawn continuation), e.g.
    `{"shards": [...], "total_tokens": N}` -- so a continuation writer picks
    up shard numbering where the interrupted chunk left off instead of
    restarting at `_00000` and clobbering already-flushed files. `None`
    (default) is the original from-scratch behavior, used by every other
    caller (a fresh source, or a full from-scratch retry)."""

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._buf: array.array = array.array("H")
        if self.resume_from:
            self.shards: List[dict] = list(self.resume_from.get("shards", []))
            self.total_tokens = int(self.resume_from.get("total_tokens", 0))
        else:
            self.shards = []
            self.total_tokens = 0
        self._shard_idx = len(self.shards)

    def write(self, ids: Sequence[int]) -> None:
        self._buf.extend(ids)
        while len(self._buf) >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, n: int) -> None:
        chunk = self._buf[:n]
        del self._buf[:n]
        arr = np.asarray(chunk, dtype=TOKEN_DTYPE)
        name = f"{self.prefix}_{self._shard_idx:05d}.bin"
        arr.tofile(os.path.join(self.out_dir, name))
        self.shards.append({"file": name, "tokens": int(arr.size)})
        self.total_tokens += int(arr.size)
        self._shard_idx += 1

    def flush_partial(self) -> None:
        """Flush whatever is buffered as a short shard.

        Used for mid-source checkpointing: afterwards everything counted in
        `total_tokens` is durably on disk, so a manifest written at this
        instant describes exactly what a reader would find. Shards may end up
        shorter than `shard_tokens`, which nothing downstream cares about --
        the loader concatenates per the manifest and the final shard has
        always been partial anyway.
        """
        if self._buf:
            self._flush(len(self._buf))

    def close(self) -> List[dict]:
        self.flush_partial()
        return self.shards

    def write_manifest(self, extra: Optional[dict] = None) -> str:
        manifest = {
            "shards": self.shards,
            "total_tokens": self.total_tokens,
            "dtype": "uint16",
        }
        if extra:
            manifest.update(extra)
        path = os.path.join(self.out_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        return path


def tokenize_and_pack(tokenizer, documents: Iterable[str], out_dir: str,
                       eos_id: int = DEFAULT_EOS_ID,
                       shard_tokens: int = 100_000_000,
                       manifest_extra: Optional[dict] = None,
                       tokenizer_name: Optional[str] = None) -> dict:
    """Stream-tokenize `documents` into fixed-size uint16 shards under
    `out_dir`, EOS-separated dense packing. Writes and returns the manifest.

    The manifest records which tokenizer produced the ids (see
    `tokenizer_fingerprint`). That was always worth recording and became
    necessary in Phase 4, where four shard directories differing only in
    vocabulary sit side by side.
    """
    writer = ShardWriter(out_dir, shard_tokens=shard_tokens)
    n_docs = 0
    for text in documents:
        writer.write(tokenize_document(tokenizer, text, eos_id))
        n_docs += 1
    writer.close()
    extra = {"n_documents": n_docs, "eos_id": eos_id,
             "tokenizer": tokenizer_fingerprint(tokenizer, tokenizer_name)}
    if manifest_extra:
        extra.update(manifest_extra)
    writer.write_manifest(extra)
    return {"total_tokens": writer.total_tokens, "n_documents": n_docs,
            "shards": writer.shards, "manifest_path": os.path.join(out_dir, "manifest.json")}


# ------------------------------------------------------------------ loader ----

class PackedTokenDataset(Dataset):
    """Random-access `seq_len`-token windows over a directory of uint16 shards
    written by `tokenize_and_pack`.

    `__getitem__` returns a single `[seq_len]` int64 tensor -- `Daedalus.forward`
    takes one same-length sequence as *both* `input_ids` and `targets` and does
    its own internal shift (see `model.py`'s docstring and `smoke.py`'s
    `net(x, targets=x)`), so callers pass this tensor unchanged as both
    arguments rather than pre-splitting it into an (input, target) pair.

    When `return_doc_ids=True`, also returns a `[seq_len]` tensor incrementing
    at each EOS crossing, for a future document-aware attention mask.

    `stride` controls the gap between consecutive window start offsets.
    `stride=1` (the default) is dense sliding-window sampling: every offset is
    a valid window start, which is what training wants -- with `shuffle=True`
    it gives near-unlimited distinct starting positions for a fixed
    `total_tokens` budget, at the cost of `__len__` being much larger than the
    shard's actual token count. A full or bounded *pass* over data for a
    metric (eval's held-out bpb) instead wants `stride=seq_len`: non-
    overlapping windows that visit each token exactly once, so `__len__`
    windows covers the whole shard rather than exploding by ~`seq_len`x.
    """

    def __init__(self, shard_dir: str, seq_len: int,
                 manifest: Optional[dict] = None, return_doc_ids: bool = False,
                 stride: int = 1):
        self.shard_dir = shard_dir
        self.seq_len = seq_len
        self.return_doc_ids = return_doc_ids
        self.stride = stride
        if manifest is None:
            with open(os.path.join(shard_dir, "manifest.json")) as f:
                manifest = json.load(f)
        self.manifest = manifest
        self.eos_id = manifest.get("eos_id", DEFAULT_EOS_ID)
        self._mmaps = [
            np.memmap(os.path.join(shard_dir, s["file"]), dtype=TOKEN_DTYPE, mode="r")
            for s in manifest["shards"]
        ]
        windows_per_shard = [max(0, (len(m) - seq_len) // stride + 1) for m in self._mmaps]
        self._cum = np.cumsum([0] + windows_per_shard)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_i = int(np.searchsorted(self._cum, idx, side="right") - 1)
        offset = (idx - int(self._cum[shard_i])) * self.stride
        window = self._mmaps[shard_i][offset: offset + self.seq_len]
        ids = torch.from_numpy(window.astype(np.int64))
        if not self.return_doc_ids:
            return ids
        doc_ids = (ids == self.eos_id).cumsum(0)
        doc_ids = torch.cat([doc_ids.new_zeros(1), doc_ids[:-1]])  # shift: id *after* crossing
        return ids, doc_ids


def make_loader(shard_dir: str, seq_len: int, batch_size: int, shuffle: bool = True,
                 num_workers: int = 2, stride: int = 1,
                 generator: Optional[torch.Generator] = None,
                 num_samples: Optional[int] = None, **kwargs) -> DataLoader:
    """DataLoader over a shard directory.

    `shuffle=True` samples **with replacement** rather than using DataLoader's
    `shuffle=`, because `shuffle=True` installs a `RandomSampler` that does
    `torch.randperm(len(ds)).tolist()` -- a whole permutation of the dataset,
    materialized as a Python list, before the first batch comes out.

    With `stride=1` (what training uses -- dense sliding windows) `len(ds)` is
    one window *per token*, so that permutation is sized by the corpus, not by
    the run. Measured on this box at ~32 bytes per window (an int64 tensor plus
    a boxed Python int each):

        200M-token sweep stand-in    ->    6.0 GB   (survivable; this is what
                                                     the first sweep paid)
        fineweb-edu at 3.75B         ->  109.1 GB   fatal
        the full 14B corpus          ->  418.1 GB   fatal

    So `hero` and `abl-arch` could not have run on the real mixture at all, and
    would have wedged the box on the way down rather than raising -- the same
    failure mode as the 2026-08-08 thrash. Sampling with replacement draws
    indices lazily in blocks of 32 via `torch.randint`, which is O(1) memory and
    independent of corpus size.

    The statistical cost is negligible here: `hero` draws ~19.5M windows from
    ~14B, so expected collisions are ~1.4e4 pairs, under 0.1%. Neighbouring
    stride-1 windows already overlap in 2047 of 2048 tokens, so exact-index
    repeats are not the meaningful form of repetition anyway.

    **Collisions are not the quantity that matters, though, and the paragraph
    above used to be the whole story.** "How often is the same window drawn
    twice" (0.069%) is a different question from "how much of the corpus is
    drawn at all", and only the second one affects the model. A token is
    covered by any of the `seq_len` windows starting within `seq_len` of it, so
    with `n` draws the coverage count is Poisson with mean
    `n * seq_len / corpus_tokens` -- which is just that source's epoch count.

    **The corpus-wide average is the wrong number to quote, and it used to be
    the only one here.** `MixtureBatchSource` draws each source at its own
    mixture share, so coverage is per-source, and the epoch counts at the 60B
    budget span 1.71 to 4.00 (`runs/preflight/epochs-at-60b.md`). The sources
    with the *most* data left over are the ones covered *worst*, because their
    share is fixed while their disk is large:

        4.00 epochs (7 of 10 sources, incl. fineweb-edu)  ->  1.83% unseen
        2.60 epochs (finephrase)                          ->  7.43% unseen
        1.71 epochs (finemath-3plus, infiwebmath-3plus)   -> 18.09% unseen

    So ~18.1% of the two maths sources' unique tokens are never drawn, and the
    budget they would have filled is spent re-drawing the other 82%. That is a
    real cost and it is *not* negligible in the way collisions are. It is
    accepted rather than fixed -- see the two reasons below, which the smaller
    corpus-wide figure understated rather than changed.

    (The corpus-wide figure, for orientation: 60B over the ~17.2B train split
    is a mean of ~3.5, so ~3% of the corpus overall. The earlier version of
    this docstring quoted 6.00% at 40B over 14.2B; both the budget and the
    corpus have since moved, which is why
    `test_hero_coverage_is_priced_at_the_budget_hero_actually_defaults_to`
    now reads the budget off `hero._build_parser()` instead of restating it.)

    That is a real cost and it is *not* negligible in the way collisions are.
    It is accepted rather than fixed, for two reasons worth stating so a later
    reader does not have to re-derive them:

    - **It is small where it lands.** At ~2.8 epochs the data-constrained
      scaling result the corpus target was chosen from says repeated tokens are
      nearly as valuable as fresh ones, so losing the tail 6% is worth much
      less than 6% of a fresh-data equivalent. Coverage improves fast with
      budget (13.5% unseen at 2 epochs, 4.98% at 3, 1.83% at 4).
    - **The alternative is not free of risk.** See below.

    **The 418 GB is a consequence of `stride=1`, not an independent
    constraint** -- which the numbers above do not say and a reader could
    easily miss. With `stride=seq_len` (non-overlapping windows, the
    conventional packing arrangement) the index space collapses from one window
    per *token* to one per *window*: ~6.9M entries for this corpus, ~222 MB as
    a `randperm` list, entirely affordable. A true permutation would then give
    100% coverage with each token seen exactly 2.813 times.

    So the honest statement is that this is a **tradeoff, not a forced move**:
    stride=1 buys varied context boundaries (a token sits at a different
    position in the window each time it is seen, which is mild augmentation
    against position-specific overfitting) and costs 6% of unique tokens;
    stride=seq_len buys full coverage and fixed boundaries. The choice was made
    for stride=1 and is kept, because the effect is small in both directions and
    the pipeline is validated and about to run a ~$44 job -- not because the
    permutation was impossible. `tests/test_data.py` pins the coverage
    arithmetic so this claim cannot rot.

    `num_samples` only sets how many indices the sampler yields before the
    iterator ends (`train.py` wraps it in `_cycle` and just keeps going); it is
    an iteration count, never an allocation.
    """
    ds = PackedTokenDataset(shard_dir, seq_len, stride=stride)
    if shuffle:
        sampler = RandomSampler(ds, replacement=True,
                                num_samples=num_samples or len(ds),
                                generator=generator)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=True, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=True, **kwargs)


def select_holdout_shards(shards: List[dict], total_tokens: int,
                          holdout_frac: float = 0.02) -> Tuple[List[dict], List[dict]]:
    """`(train_shards, holdout_shards)` — which whole shards go to the holdout.

    Split out of `make_holdout_split` so the *selection* can be evaluated
    without materializing directories. `scripts/mixture_margin.py` needs
    exactly this to say what mixture `hero` will really train on, and it
    previously modelled it as a flat `n * (1 - holdout_frac)` haircut per
    source. That model is wrong wherever a source's final shard is small
    relative to the target: `stack-edu-python` (13 shards, last one 10.96M
    against a 24.2M target) has to take the 100M shard before it as well, so
    its real carve is **9.16%**, not 2%. Uneven carves move the *mixture*, and
    on the built corpus that difference is `l1_skew` 9.01 (modelled) against
    **10.21** (real) at 60B — the wrong side of `hero`'s 10.0 refusal limit.

    Reserves shards from the end until at least `holdout_frac` of the tokens is
    covered, always at least one, never all of them.
    """
    holdout: List[dict] = []
    seen = 0
    target = int(total_tokens * holdout_frac)
    for s in reversed(shards):
        holdout.insert(0, s)
        seen += s["tokens"]
        if seen >= target or len(holdout) >= len(shards) - 1:
            break
    return shards[: len(shards) - len(holdout)], holdout


def make_holdout_split(shard_dir: str, train_dir: str, holdout_dir: str,
                        holdout_frac: float = 0.02) -> Dict[str, dict]:
    """Split one source's shards (written by `tokenize_and_pack`) into disjoint
    train/holdout manifests, by reserving whole shard *files* for holdout --
    never slicing tokens out of a file -- so no `PackedTokenDataset` window
    can straddle the train/holdout boundary. Neither `train.py`'s `--data-dir`
    nor `eval.py`'s `--shard-dir` support a mixture/subset of shards, only "all
    shards in this directory" (see their manifest-per-directory convention),
    so this materializes two real directories, each with its own
    `manifest.json`, rather than passing a shard subset through some other
    mechanism. Shard files themselves are hardlinked (not copied) into both
    directories to avoid duplicating GBs of token data on disk.

    Reserves shards from the end of the list until at least `holdout_frac` of
    total tokens is covered (always >=1 shard). Raises if the source has only
    one shard, since a 0-token train or holdout split is never useful.
    """
    with open(os.path.join(shard_dir, "manifest.json")) as f:
        manifest = json.load(f)
    shards = manifest["shards"]
    if len(shards) < 2:
        raise ValueError(
            f"{shard_dir} has only {len(shards)} shard(s); need >=2 to hold "
            "one out without leaving the train or holdout split empty")
    train_shards, holdout_shards = select_holdout_shards(
        shards, manifest["total_tokens"], holdout_frac)

    def _write_split(out_dir: str, split_shards: List[dict]) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        for s in split_shards:
            src = os.path.join(shard_dir, s["file"])
            dst = os.path.join(out_dir, s["file"])
            # Reuse `dst` only when it is *the same file* as `src`, by
            # (device, inode) -- not merely the same name. Shard names restart
            # at _00000 for every source, so re-pointing a split root at a
            # different corpus finds same-named, different-content files: on
            # this box data/shards/fineweb-edu/fineweb-edu_00000.bin and
            # data/shards-sweep/fineweb-edu/fineweb-edu_00000.bin are distinct
            # inodes. A name-only check would keep the stale bytes and write a
            # manifest describing the new ones -- training silently reads data
            # that does not match its own manifest, with nothing to see in any
            # log. Relink instead.
            if os.path.exists(dst):
                d, s_ = os.stat(dst), os.stat(src)
                if (d.st_dev, d.st_ino) == (s_.st_dev, s_.st_ino):
                    continue
                os.remove(dst)
            os.link(src, dst)
        split_manifest = dict(manifest)
        split_manifest["shards"] = split_shards
        split_manifest["total_tokens"] = sum(s["tokens"] for s in split_shards)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(split_manifest, f, indent=2)
        return split_manifest

    train_manifest = _write_split(train_dir, train_shards)
    holdout_manifest = _write_split(holdout_dir, holdout_shards)
    return {"train": train_manifest, "holdout": holdout_manifest}


def make_mixture_holdout_split(mixture_root: str, train_root: str, holdout_root: str,
                                holdout_frac: float = 0.02) -> Dict[str, dict]:
    """`make_holdout_split`, applied per-source across a `dataprep.py` mixture
    root (one subdirectory per source, each with its own manifest.json -- the
    same layout `train.py`'s `MixtureBatchSource` reads). Needed for `abl-arch`,
    which trains on the real mixture rather than sweep's single-source stand-in.

    Sources with fewer than 2 shards can't be split (see `make_holdout_split`)
    -- skipped with a warning rather than raising, so evaluating against a
    partially-built mixture doesn't crash, matching `MixtureBatchSource`'s own
    "train on whatever's present" tolerance for a mixture still being built.
    """
    sources = sorted(
        name for name in os.listdir(mixture_root)
        if os.path.exists(os.path.join(mixture_root, name, "manifest.json")))
    if not sources:
        raise ValueError(f"no source under {mixture_root!r} has a manifest.json")

    splits: Dict[str, dict] = {}
    for name in sources:
        try:
            splits[name] = make_holdout_split(
                os.path.join(mixture_root, name),
                os.path.join(train_root, name),
                os.path.join(holdout_root, name),
                holdout_frac=holdout_frac)
        except ValueError as e:
            print(f"make_mixture_holdout_split: skipping {name!r} ({e})")
    if not splits:
        raise ValueError(f"every source under {mixture_root!r} had too few "
                         "shards to split; nothing to train or evaluate on")
    return splits


# ------------------------------------------------------------- near-dedup ----

def shingles(text: str, n: int = 5) -> List[str]:
    words = text.split()
    if len(words) <= n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def minhash_signature(text: str, num_perm: int = 128, shingle_size: int = 5):
    """MinHash signature over word shingles, for ~0.85-Jaccard cross-dedup
    (AGENT.md `dataprep`: fineweb-edu/DCLM/Nemotron overlap ~32%)."""
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for s in shingles(text, shingle_size):
        m.update(s.encode("utf-8"))
    return m


class NearDupFilter:
    """Streaming near-duplicate filter backed by MinHashLSH. `is_duplicate`
    inserts the signature as a side effect when the doc is kept, so later
    documents are checked against everything kept so far."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        from datasketch import MinHashLSH
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._next_id = 0

    def is_duplicate(self, text: str) -> bool:
        m = minhash_signature(text, num_perm=self.num_perm)
        if self.lsh.query(m):
            return True
        self.lsh.insert(str(self._next_id), m)
        self._next_id += 1
        return False


# ------------------------------------------------------------- decontam ----

def ngram_set(text: str, n: int) -> set:
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def build_eval_ngram_index(eval_texts: Iterable[str], n: int = 13) -> set:
    """Union of n-grams across every eval-set text, for 8-13-gram decontam."""
    idx = set()
    for t in eval_texts:
        idx |= ngram_set(t, n)
    return idx


def is_contaminated(text: str, eval_ngram_index: set, n: int = 13) -> bool:
    return not eval_ngram_index.isdisjoint(ngram_set(text, n))


# --------------------------------------------------------------- hub sync ----

def upload_shards(local_dir: str, repo_id: str, token: Optional[str] = None,
                   private: bool = True) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="dataset")
    return repo_id


def download_shards(repo_id: str, local_dir: str, token: Optional[str] = None) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, repo_type="dataset",
                             local_dir=local_dir, token=token)


# --------------------------------------------------------------------- cli ----

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Tokenize one HF dataset into uint16 shards.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--out", required=True)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    args = p.parse_args()

    tok = get_tokenizer()
    docs = stream_dataset(args.dataset, split=args.split, text_column=args.text_column)
    if args.max_docs:
        from itertools import islice
        docs = islice(docs, args.max_docs)
    result = tokenize_and_pack(tok, docs, args.out, eos_id=tok.eos_token_id,
                               shard_tokens=args.shard_tokens,
                               manifest_extra={"source_dataset": args.dataset})
    print(json.dumps({k: v for k, v in result.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    _cli()
