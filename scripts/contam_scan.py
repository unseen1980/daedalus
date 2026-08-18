"""Measure how much of the built corpus the partial decontamination let through.

Why this exists
---------------
`STATUS.md` records an honest caveat on every Daedalus eval number, `hero`'s
included: the corpus was decontaminated with `_build_eval_index(limit=2000)`
(`daedalus/dataprep.py:299`), so the n-gram index covered only **2,000 items per
task**. HellaSwag has 10,042 -- **19.9% covered** -- and ARC-Easy 2,376 (84.2%).
PIQA, OpenBookQA and WinoGrande are fully covered. Documents matching an
*uncovered* eval item were never filtered.

That caveat is currently an argument, not a number. The mission is "beats
comparable models on benchmarks", and the first thing a reader does to a
surprising small-model score is ask whether it saw the test set. "We only
indexed a fifth of HellaSwag, but here is why we think it is fine" is a much
weaker answer than a measured exposure rate, and the measurement costs no GPU
and a few minutes of CPU.

Writing it also turned up a second gap the caveat did not know about. The build
logs record an index of **183,359** 13-grams; the same code at the same
`limit=2000` builds **214,682** today. The difference is `334c86c` (2026-08-09
18:09Z), which moved ARC-Easy and OpenBookQA onto their `test` splits to match
lm-evaluation-harness. Every source built before it indexed those tasks'
`validation` splits -- so for most of this corpus the splits we are *scored on*
were never filtered at all. Reproducing 183,359 exactly is what pinned that
down; `--build-splits` is how it is expressed here.

What it measures
----------------
For a token-weighted sample of the real shards, the fraction of **training
tokens that live in a document containing a 13-gram from an eval item** --
against three *disjoint* indices, one per distinct reason a document could have
slipped through:

  filtered    what `dataprep` really matched on (ARC/OBQA `validation`, 2,000
              items per task).  **Negative control.**
  split_gap   n-grams of the scored splits that `filtered` never contained
  limit_gap   everything past `--eval-task-limit 2000` (HellaSwag 19.9%)

`filtered` is the reason this is evidence rather than a number. Those n-grams
were removed at build time, so its rate must come back at ~0. If it does not,
the measurement disagrees with the pipeline (tokenizer round-trip, whitespace,
document splitting) and neither exposure row can be trusted. A scan that reports
only the number you were hoping to see has no way to fail.

All three are built with this repo's own `ngram_set`/`build_eval_ngram_index`
(`daedalus/data.py:458,463`) and `eval.TASK_LOADERS`, not a reimplementation, so
"contaminated" means here exactly what it meant during the build.

Sampling, and which way each choice biases the answer
-----------------------------------------------------
Windows of `--window-tokens` are read at evenly-spaced offsets over each
source's whole token space, so every token is equally likely to be sampled and
the token-weighted rate is unbiased. Each window is split on the source's
`eos_id` (`tokenize_document` appends it to every document), and only documents
**fully contained** in the window are scored -- a partial document at either
edge cannot be classified, and guessing would be the kind of silent error this
file exists to remove.

That exclusion is the one bias worth stating: a document longer than the window
can never be fully contained, long documents contain more n-grams, so the
estimate leans **low** for them. The report prints the excluded token mass so
the size of that lean is visible rather than assumed. Default window 32,768
tokens is ~13x the median document here, which keeps the excluded mass small.

Memory (ADDENDUM 2)
-------------------
Bounded and measured -- 1.05 GB peak RSS, flat in sample size because the
indices are the only unbounded structures and 20 windows cost the same as 60:
shards
are read through `np.memmap` a window at a time, and per-window n-gram sets are
freed each iteration. `--max-rss-gb` sets `RLIMIT_AS` so a runaway allocation
raises a traceback instead of wedging the box, and the scan stops cleanly if
available memory falls below `--min-available-gb`.

Usage
-----
    python scripts/contam_scan.py --out runs/preflight/contam-exposure.md
"""
import argparse
import json
import os
import resource
import sys
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_USED_LIMIT = 2000
DEFAULT_WINDOW_TOKENS = 32_768
DEFAULT_WINDOWS_PER_SOURCE = 120


# ------------------------------------------------------------------ indices ---

def _task_texts(limit: Optional[int], splits: Optional[Dict[str, str]] = None):
    """`(texts, per_task_item_counts)` for every eval task.

    Goes through `eval.TASK_LOADERS` rather than `load_all_tasks` so a caller
    can ask for a *historical* split per task -- which is the whole point of
    the `filtered` index below.
    """
    import eval as E
    texts, counts = [], {}
    for name, loader in E.TASK_LOADERS.items():
        kwargs = {"limit": limit}
        chosen = (splits or {}).get(name, E.TASK_SPLITS.get(name))
        if chosen is not None:
            kwargs["split"] = chosen
        try:
            examples = loader(**kwargs)
        except Exception as e:                       # matches load_all_tasks
            print(f"WARNING: could not load task {name!r} ({e}); skipping")
            counts[name] = 0
            continue
        counts[name] = len(examples)
        texts += [ctx + " " + cont for ex in examples for ctx, cont in ex.candidates]
    return texts, counts


def build_indices(used_limit: int = DEFAULT_USED_LIMIT, n: int = 13,
                  full_limit: Optional[int] = None,
                  build_splits: Optional[Dict[str, str]] = None
                  ) -> Tuple[Dict[str, set], dict]:
    """Three **disjoint** indices, one per distinct reason a document could
    have slipped through, plus the coverage metadata for the report.

      `filtered`   exactly what `dataprep` matched on while building this
                   corpus. **Negative control** -- these were removed, so its
                   rate must come back ~0.
      `split_gap`  n-grams of the splits we actually *score* that `filtered`
                   does not contain. On this corpus that is ARC-Easy and
                   OpenBookQA `test`: every source built before `334c86c`
                   (2026-08-09 18:09Z) indexed their `validation` splits
                   instead, so the scored items were never filtered.
      `limit_gap`  everything left in the full task sets -- the exposure from
                   `_build_eval_index(limit=2000)` covering only 2,000 items
                   per task (HellaSwag 19.9%).

    Disjoint by construction (successive set differences), so a document
    hitting two rows really hit two different eval items rather than being
    counted twice. Keeping them apart is what makes `filtered` a control: if
    the split gap were folded into it, a non-zero rate could mean either "the
    corpus is exposed" or "this scan disagrees with the pipeline", and there
    would be no way to tell which.
    """
    from daedalus.data import build_eval_ngram_index

    built_texts, built_counts = _task_texts(used_limit, build_splits)
    scored_texts, scored_counts = _task_texts(used_limit, None)
    full_texts, full_counts = _task_texts(full_limit, None)

    filtered = build_eval_ngram_index(built_texts, n=n)
    split_gap = build_eval_ngram_index(scored_texts, n=n) - filtered
    limit_gap = (build_eval_ngram_index(full_texts, n=n) - filtered) - split_gap

    coverage = {name: {"indexed": scored_counts.get(name, 0),
                       "total": total,
                       "build_split": (build_splits or {}).get(name)}
                for name, total in full_counts.items()}
    return ({"filtered": filtered, "split_gap": split_gap,
             "limit_gap": limit_gap}, coverage)


INDEX_NAMES = ("filtered", "split_gap", "limit_gap")

INDEX_BLURB = {
    "filtered": "**negative control** — n-grams `dataprep` did remove",
    "split_gap": "scored splits the build never indexed (ARC-Easy/OpenBookQA "
                 "`test` vs `validation`, pre-`334c86c`)",
    "limit_gap": "eval items beyond `--eval-task-limit 2000` (HellaSwag 19.9% "
                 "covered)",
}


# ------------------------------------------------------------------ sampling --

def window_offsets(total_tokens: int, window: int, k: int) -> List[int]:
    """`k` evenly-spaced start offsets covering `[0, total_tokens)`.

    Systematic rather than random: same token-uniformity, lower variance, and
    reproducible without threading a seed through every caller. Returns fewer
    than `k` offsets when the source is too small to hold that many disjoint
    windows -- oversampling a small source would weight its tokens more heavily
    than a large one's and quietly bias the corpus-level rate.
    """
    if total_tokens <= 0 or window <= 0 or k <= 0:
        return []
    if total_tokens <= window:
        return [0]
    k = min(k, total_tokens // window)
    span = total_tokens - window
    if k == 1:
        return [span // 2]
    return [round(i * span / (k - 1)) for i in range(k)]


def locate(shards: List[dict], offset: int) -> Optional[Tuple[int, int, int]]:
    """Global token `offset` -> `(shard_index, local_offset, tokens_left_in_shard)`.

    The remaining-tokens figure is returned rather than the caller assuming the
    shard is full: the last shard of a source is short, and reading past it
    would not raise -- memmap silently returns a shorter slice than the
    accounting claims.
    """
    seen = 0
    for i, s in enumerate(shards):
        n = int(s["tokens"])
        if offset < seen + n:
            local = offset - seen
            return i, local, n - local
        seen += n
    return None


def read_window(shard_dir: str, shards: List[dict], offset: int, window: int):
    """`window` tokens starting at global `offset`, crossing shard files.

    Reading up to the end of the containing shard instead would be simpler and
    wrong in a way that is invisible in the output: shard boundaries are not
    random with respect to content, and truncating every window that straddles
    one under-samples the tail of each shard -- the tokens written last, which
    on a resumed source are precisely the ones a later `--eval-task-limit` was
    meant to protect. It also silently shrinks the sample (the first version of
    this scan returned 1 token for a window landing 1 token before a boundary),
    and a scan that reports 0 hits because it read almost nothing looks exactly
    like a clean corpus.
    """
    import numpy as np

    loc = locate(shards, offset)
    if loc is None:
        return np.empty(0, dtype=np.uint16)
    idx, local, _ = loc
    parts, need = [], window
    for s in shards[idx:]:
        n = int(s["tokens"])
        take = min(need, n - local)
        if take > 0:
            arr = np.memmap(os.path.join(shard_dir, s["file"]),
                            dtype=np.uint16, mode="r")
            parts.append(np.array(arr[local:local + take]))   # copy; frees the map
            del arr
            need -= take
        local = 0
        if need <= 0:
            break
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)


# ------------------------------------------------------------------ scanning --

def split_documents(tokens, eos_id: int) -> Tuple[List[list], int, int]:
    """`(whole_documents, excluded_tokens, partial_fragments)`.

    Documents end with `eos_id` (`daedalus/data.py:55`). Only fragments bounded
    by an eos on *both* sides are whole; the head of the window (tail of some
    earlier document) and the tail (head of a later one) are excluded and
    reported, because a partial document cannot be classified either way.
    """
    ids = list(tokens)
    bounds = [i for i, t in enumerate(ids) if t == eos_id]
    if len(bounds) < 2:
        return [], len(ids), (1 if ids else 0)
    docs = []
    for a, b in zip(bounds, bounds[1:]):
        doc = ids[a + 1:b]
        if doc:
            docs.append(doc)
    # The head before the first eos and the tail after the last one: content of
    # documents the window only partly contains. The eos separators themselves
    # belong to no document's text and are counted in neither.
    excluded = bounds[0] + (len(ids) - bounds[-1] - 1)
    partial = int(bounds[0] > 0) + int(bounds[-1] < len(ids) - 1)
    return docs, excluded, partial


def classify_document(text: str, indices: Dict[str, set], n: int = 13) -> dict:
    """Which of `indices` this document's 13-grams hit.

    Builds `ngram_set` once and calls `isdisjoint` per index, rather than
    calling `is_contaminated` once per index -- that would rebuild the
    document's n-gram set each time, and building it is the dominant cost of
    the whole scan. Same predicate, proven by
    `test_classify_document_agrees_with_is_contaminated`.
    """
    from daedalus.data import ngram_set
    grams = ngram_set(text, n)
    return {name: not idx.isdisjoint(grams) for name, idx in indices.items()}


def scan_source(shard_dir: str, tokenizer, indices: Dict[str, set],
                window: int = DEFAULT_WINDOW_TOKENS,
                k: int = DEFAULT_WINDOWS_PER_SOURCE, n: int = 13,
                min_available_gb: float = 6.0, max_examples: int = 4) -> dict:
    with open(os.path.join(shard_dir, "manifest.json")) as f:
        manifest = json.load(f)
    shards = manifest["shards"]
    eos_id = int(manifest.get("eos_id", 0))
    total = int(manifest["total_tokens"])

    out = {"source": os.path.basename(shard_dir), "source_tokens": total,
           "eos_id": eos_id, "windows": 0, "tokens_scanned": 0,
           "tokens_excluded": 0, "docs": 0, "doc_tokens": 0,
           "stopped_early": None, "examples": []}
    for name in indices:
        out[f"docs_{name}"] = 0
        out[f"tokens_{name}"] = 0

    for off in window_offsets(total, window, k):
        if _available_gb() < min_available_gb:
            out["stopped_early"] = f"available memory below {min_available_gb} GB"
            break
        chunk = read_window(shard_dir, shards, off, window)
        if chunk.size == 0:
            continue
        out["windows"] += 1
        out["tokens_scanned"] += int(chunk.size)

        docs, excluded, _ = split_documents(chunk.tolist(), eos_id)
        out["tokens_excluded"] += excluded
        for doc in docs:
            text = tokenizer.decode(doc)
            hit = classify_document(text, indices, n=n)
            out["docs"] += 1
            out["doc_tokens"] += len(doc)
            for name, did in hit.items():
                if did:
                    out[f"docs_{name}"] += 1
                    out[f"tokens_{name}"] += len(doc)
                    if len(out["examples"]) < max_examples:
                        out["examples"].append(
                            _example(name, text, indices[name], n))
    return out


def _example(index_name: str, text: str, index: set, n: int) -> dict:
    """A hit, in enough detail to judge it by eye.

    A count alone cannot distinguish "a document reproducing a HellaSwag item"
    from "a physics tutorial that happens to share thirteen words of ordinary
    prose with a science question". Those have very different implications for
    a benchmark claim, and only the text tells them apart -- so the report
    shows the matched n-gram and its surroundings rather than asking the reader
    to take the rate on trust.
    """
    from daedalus.data import ngram_set
    matched = sorted(index & ngram_set(text, n))
    head = matched[0] if matched else ""
    at = text.find(head.split()[0]) if head else -1
    lo = max(0, at - 120)
    return {"index": index_name, "n_matched": len(matched), "ngram": head,
            "excerpt": text[lo:at + 320].replace("\n", " ") if at >= 0 else text[:320]}


def _available_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return float("inf")


# -------------------------------------------------------------------- stats ---

def wilson_upper(hits: int, n: int, z: float = 1.96) -> float:
    """Upper bound of the Wilson score interval.

    Wilson rather than normal-approximation because the interesting case is
    `hits` at or near 0, where `p +/- z*sqrt(p(1-p)/n)` collapses to the useless
    interval [0, 0] and would report certainty from an absence of evidence.
    """
    if n <= 0:
        return 1.0
    p = hits / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return min(1.0, (centre + half) / d)


def corpus_rates(per_source: List[dict], names=INDEX_NAMES) -> dict:
    """Corpus-level rates, weighted by each source's real token share.

    A plain pooled rate would let `everyday-conversations` (0.4M tokens) count
    for as much as `fineweb-edu` (4.9B), because both get the same number of
    sampled windows. That is not the quantity anyone cares about: what a reader
    wants is the share of the *training stream* that is exposed.
    """
    tok = sum(s["doc_tokens"] for s in per_source)
    docs = sum(s["docs"] for s in per_source)
    corpus = sum(s["source_tokens"] for s in per_source)
    out = {"corpus_tokens": corpus, "docs": docs, "doc_tokens": tok,
           "sampled_frac": (tok / corpus) if corpus else 0.0}
    for name in names:
        rate = 0.0
        for s in per_source:
            share = (s["source_tokens"] / corpus) if corpus else 0.0
            if s["doc_tokens"]:
                rate += share * s[f"tokens_{name}"] / s["doc_tokens"]
        hits = sum(s[f"docs_{name}"] for s in per_source)
        out[f"docs_{name}"] = hits
        out[f"token_rate_{name}"] = rate
        out[f"doc_rate_{name}_upper95"] = wilson_upper(hits, docs)
    return out


# ------------------------------------------------------------------- report ---

def format_report(per_source: List[dict], totals: dict, coverage: dict,
                  index_sizes: dict, params: dict) -> str:
    L = []
    A = L.append
    A("# Contamination exposure of the built corpus — measured, not argued")
    A("")
    A(f"Written by `scripts/contam_scan.py`. Window {params['window']:,} tokens, "
      f"up to {params['k']} windows per source, {params['n']}-grams.")
    A("")
    A("## Verdict")
    A("")
    A("| index | what a hit means | 13-grams | rate over sampled training tokens | docs hit | 95% upper bound |")
    A("|---|---|---|---|---|---|")
    for name in INDEX_NAMES:
        A(f"| `{name}` | {INDEX_BLURB[name]} | {index_sizes[name]:,} | "
          f"**{totals[f'token_rate_{name}']*100:.4f}%** | "
          f"{totals[f'docs_{name}']:,} / {totals['docs']:,} | "
          f"{totals[f'doc_rate_{name}_upper95']*100:.4f}% |")
    A("")
    A("`filtered` is the load-bearing row and the reason the other two can be "
      "believed: those n-grams *were* removed at build time, so a non-zero rate "
      "there would mean this scan and the pipeline disagree about what a "
      "document is — and the exposure rows would be measuring that "
      "disagreement rather than the corpus.")
    A("")
    A("## Coverage the build actually had")
    A("")
    A("| task | items indexed | items in the scored split | covered | split the build indexed |")
    A("|---|---|---|---|---|")
    for name, c in sorted(coverage.items()):
        pct = 100.0 * c["indexed"] / c["total"] if c["total"] else 0.0
        built = c.get("build_split") or "(as scored)"
        A(f"| {name} | {c['indexed']:,} | {c['total']:,} | {pct:.1f}% | {built} |")
    A("")
    A("## Per source")
    A("")
    A("| source | source tokens | sampled docs | doc tokens | "
      + " | ".join(f"`{n}`" for n in INDEX_NAMES) + " |")
    A("|---|---|---|---|" + "---|" * len(INDEX_NAMES))
    for s in sorted(per_source, key=lambda x: -x["source_tokens"]):
        cells = " | ".join(str(s[f"docs_{n}"]) for n in INDEX_NAMES)
        A(f"| {s['source']} | {s['source_tokens']:,} | {s['docs']:,} | "
          f"{s['doc_tokens']:,} | {cells} |")
    A("")
    excluded = sum(s["tokens_excluded"] for s in per_source)
    scanned = sum(s["tokens_scanned"] for s in per_source)
    hits = [(s["source"], e) for s in per_source for e in s.get("examples", [])]
    if hits:
        A("## Every hit, in full")
        A("")
        A("A rate cannot tell a document that reproduces an eval item from one "
          "that shares thirteen words of ordinary prose with it, and the two "
          "mean very different things for a benchmark claim.")
        A("")
        for src, e in hits:
            A(f"**`{e['index']}` in {src}** — {e['n_matched']} matching "
              f"{params['n']}-gram(s)")
            A("")
            A(f"> matched: `{e['ngram']}`")
            A("")
            A(f"> context: …{e['excerpt'].strip()}…")
            A("")
    A(f"Sampled {totals['doc_tokens']:,} tokens of {totals['corpus_tokens']:,} "
      f"({totals['sampled_frac']*100:.3f}% of the corpus). "
      f"{excluded:,} of {scanned:,} scanned tokens ({100.0*excluded/scanned:.2f}%) "
      "sat in a partial document at a window edge and were excluded — that is "
      "the size of the documented downward lean on long documents.")
    A("")
    for s in per_source:
        if s["stopped_early"]:
            A(f"- **{s['source']} stopped early**: {s['stopped_early']}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- cli ---

def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards-root", default="data/shards")
    p.add_argument("--out", default="runs/preflight/contam-exposure.md")
    p.add_argument("--json-out", default="runs/preflight/contam-exposure.json")
    p.add_argument("--used-limit", type=int, default=DEFAULT_USED_LIMIT,
                   help="the --eval-task-limit the corpus was built with")
    p.add_argument("--build-splits",
                   default="arc_easy=validation,openbookqa=validation",
                   help="task=split pairs naming the splits the corpus was "
                        "actually indexed against, which on this corpus are "
                        "not the ones it is scored on (334c86c). Verified by "
                        "reproducing the 183,359-gram index the build logged; "
                        "pass an empty string to assume the build used today's "
                        "splits.")
    p.add_argument("--window-tokens", type=int, default=DEFAULT_WINDOW_TOKENS)
    p.add_argument("--windows-per-source", type=int,
                   default=DEFAULT_WINDOWS_PER_SOURCE)
    p.add_argument("--n", type=int, default=13)
    p.add_argument("--max-rss-gb", type=float, default=12.0)
    p.add_argument("--min-available-gb", type=float, default=6.0)
    p.add_argument("--hit-examples", type=int, default=4,
                   help="per-source cap on hits quoted in the report")
    p.add_argument("--sources", default=None,
                   help="comma-separated subset; default is every source with "
                        "a manifest.json")
    a = p.parse_args(argv)

    if a.max_rss_gb > 0:
        cap = int(a.max_rss_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    if _available_gb() < a.min_available_gb:
        print(f"REFUSE: only {_available_gb():.1f} GB available, need "
              f"{a.min_available_gb} GB", file=sys.stderr)
        return 2

    build_splits = dict(kv.split("=", 1) for kv in a.build_splits.split(",")
                        if "=" in kv)
    print(f"building indices (used_limit={a.used_limit}, "
          f"build_splits={build_splits or 'as scored'}) ...", flush=True)
    indices, coverage = build_indices(a.used_limit, n=a.n,
                                      build_splits=build_splits or None)
    print("  " + "  ".join(f"{k}={len(v):,}" for k, v in indices.items()),
          flush=True)

    from daedalus.data import get_tokenizer
    tokenizer = get_tokenizer()

    names = ([s.strip() for s in a.sources.split(",")] if a.sources else
             sorted(x for x in os.listdir(a.shards_root)
                    if os.path.exists(os.path.join(a.shards_root, x, "manifest.json"))))
    per_source = []
    for name in names:
        d = os.path.join(a.shards_root, name)
        print(f"scanning {name} ...", flush=True)
        s = scan_source(d, tokenizer, indices,
                        window=a.window_tokens, k=a.windows_per_source, n=a.n,
                        min_available_gb=a.min_available_gb,
                        max_examples=a.hit_examples)
        print(f"  docs={s['docs']:,} " +
              " ".join(f"{n}={s[f'docs_{n}']}" for n in INDEX_NAMES), flush=True)
        per_source.append(s)

    totals = corpus_rates(per_source)
    params = {"window": a.window_tokens, "k": a.windows_per_source, "n": a.n,
              "used_limit": a.used_limit, "build_splits": build_splits}
    sizes = {k: len(v) for k, v in indices.items()}
    report = format_report(per_source, totals, coverage, sizes, params)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(report)
    with open(a.json_out, "w") as f:
        json.dump({"per_source": per_source, "totals": totals,
                   "coverage": coverage, "index_sizes": sizes,
                   "params": params}, f, indent=2)
    print(report)
    print(f"wrote {a.out} and {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
