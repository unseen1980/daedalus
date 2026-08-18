"""Alternating CPU decode benchmark: our GGUF against a peer's, honestly.

The headline half of this project's stated bar is "concede SmolLM2-135M on
quality, beat it decisively on CPU decode". Until now nothing measured the
second half at all -- `runs/eval/peer-table.md` has quality for five peers and
no decode number for any of them, and `abl-arch` only ever compares our two
arms to each other.

Two things this script exists to prevent, both learned the expensive way on
this project:

1. **Non-alternating runs drift.** A back-to-back "A three times, then B three
   times" comparison on this box reported hybrid-vs-dense at 1.29x when
   alternating rounds put the truth at 1.15x -- the box's own load moved
   underneath the measurement. So rounds alternate A, B, A, B, ... and every
   round is reported, not just the mean.
2. **Thread counts must match.** llama-bench's default is (cores - something)
   depending on build; comparing a run at 8 threads with one at 6 is comparing
   schedulers, not models.

Absolute numbers depend on what else the box is doing, and are only comparable
within one invocation. The *ratio* is the durable part. That caveat is written
into the JSON rather than left to the reader.

Usage:
  python scripts/decode_bench.py --models ours=/path/a.gguf smollm2=/path/b.gguf \\
      --threads 8 --rounds 3 --n-gen 128 --out runs/eval/decode-bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional

DEFAULT_BENCH = "/workspace/llama.cpp/build/bin/llama-bench"


def run_once(bench_bin: str, model: str, threads: int, n_gen: int,
             depth: int = 0, timeout_s: float = 900.0) -> Optional[float]:
    """One llama-bench decode measurement, in tokens/sec, or None if it failed.

    `-p 0` drops the prefill row: prompt processing is a different regime and
    averaging the two hides the number a chat user actually feels.

    `depth` (`-d`) is the interesting axis for *this* architecture and the one
    a default llama-bench invocation silently omits. At depth 0 every model
    decodes into an empty context; attention's per-token cost then barely
    shows. A gated short conv's decode cost is flat in context length while
    attention's KV reads grow with it, so a hybrid measured only at depth 0 is
    measured exactly where it has the least to gain -- and 2048 is the context
    this project trains for.
    """
    cmd = [bench_bin, "-m", model, "-p", "0", "-n", str(n_gen),
           "-d", str(depth), "-t", str(threads), "-r", "1", "-o", "json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {' '.join(cmd)}", flush=True)
        return None
    if out.returncode != 0:
        print(f"  FAILED rc={out.returncode}: {out.stderr.strip()[-400:]}",
              flush=True)
        return None
    try:
        rows = json.loads(out.stdout)
    except json.JSONDecodeError:
        print(f"  UNPARSEABLE: {out.stdout.strip()[:400]}", flush=True)
        return None
    # llama-bench emits one row per (prompt, gen) config; with -p 0 there is
    # exactly one, but select on n_gen rather than trusting the position.
    for row in rows:
        if int(row.get("n_gen", 0)) == n_gen:
            return float(row["avg_ts"])
    return float(rows[-1]["avg_ts"]) if rows else None


def bench(models: Dict[str, str], threads: int, rounds: int, n_gen: int,
          bench_bin: str = DEFAULT_BENCH, depth: int = 0) -> dict:
    """Alternate every model through `rounds` rounds at one thread count."""
    samples: Dict[str, List[float]] = {name: [] for name in models}
    for r in range(1, rounds + 1):
        for name, path in models.items():
            ts = run_once(bench_bin, path, threads, n_gen, depth=depth)
            print(f"  round {r} {name}: "
                  f"{'FAILED' if ts is None else f'{ts:.1f} tok/s'}", flush=True)
            if ts is not None:
                samples[name].append(ts)

    result = {"threads": threads, "n_gen": n_gen, "depth": depth,
              "rounds": rounds, "models": {}}
    for name, path in models.items():
        xs = samples[name]
        result["models"][name] = {
            "path": path,
            "file_mb": round(os.path.getsize(path) / 1e6, 2)
                       if os.path.exists(path) else None,
            "samples": [round(x, 2) for x in xs],
            "mean": round(statistics.mean(xs), 2) if xs else None,
            # Population stdev on n=1 is 0 and pstdev on n<1 raises; guard both
            # rather than reporting a spread that does not exist.
            "stdev": round(statistics.stdev(xs), 2) if len(xs) > 1 else None,
        }
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help="name=/path/to.gguf, in the order to alternate")
    p.add_argument("--threads", type=int, nargs="+", default=[8],
                   help="one pass per thread count; models are matched within "
                        "each pass, never across")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--depths", type=int, nargs="+", default=[0],
                   help="context depth to decode at (llama-bench -d). Depth 0 "
                        "is where a conv hybrid has least to gain; 2048 is "
                        "what this project trains for.")
    p.add_argument("--bench-bin", default=DEFAULT_BENCH)
    p.add_argument("--out", default=None)
    p.add_argument("--note", default=None,
                   help="what else the box was doing; absolutes are only "
                        "comparable within one invocation")
    a = p.parse_args(argv)

    models: Dict[str, str] = {}
    for spec in a.models:
        if "=" not in spec:
            print(f"--models wants name=path, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"no such model: {path}", file=sys.stderr)
            return 2
        models[name] = path

    passes = []
    for t in a.threads:
        for d in a.depths:
            print(f"=== {t} threads, depth {d}, {a.rounds} alternating rounds "
                  f"===", flush=True)
            passes.append(bench(models, t, a.rounds, a.n_gen, a.bench_bin,
                                depth=d))

    report = {"passes": passes, "note": a.note,
              "metric": "llama-bench decode tok/s (-p 0), alternating rounds; "
                        "absolute values depend on concurrent box load and are "
                        "comparable only within one invocation -- the ratio is "
                        "the durable number"}
    print(json.dumps(report, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
