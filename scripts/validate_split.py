"""Validate every shard `hero` will read, before $59.85 depends on it.

Two checks, both cheap and both metadata-only until the last one:

1. Every shard file exists and its size on disk equals `tokens * itemsize`.
   A truncated shard is the likely corruption from a dataprep that was
   interrupted, and this project's dataprep was interrupted many times. It
   would surface at `hero` startup as a memmap/shape error, `run_with_resume`
   would burn its 10 attempts over ~1 h of backoff, and the operator would wake
   to a dead run.

2. Manifest totals agree with the sum of their shard entries.

3. Then build the real `MixtureBatchSource` over the real split and pull
   batches, so the constructor path itself is exercised.

Read-only. Memmaps, so RSS stays flat.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/daedalus")
os.chdir("/workspace/daedalus")

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/shards-hero-split/train"

DTYPES = {"uint16": 2, "uint32": 4, "int32": 4}

bad = []
total_tokens = 0
total_bytes = 0
n_shards = 0

sources = sorted(d for d in os.listdir(ROOT)
                 if os.path.isfile(os.path.join(ROOT, d, "manifest.json")))
print(f"{len(sources)} sources under {ROOT}\n")

for src in sources:
    d = os.path.join(ROOT, src)
    man = json.load(open(os.path.join(d, "manifest.json")))
    itemsize = DTYPES.get(man.get("dtype", "uint16"))
    if itemsize is None:
        bad.append((src, "-", f"unknown dtype {man.get('dtype')!r}"))
        continue
    summed = 0
    for entry in man["shards"]:
        n_shards += 1
        path = os.path.join(d, entry["file"])
        want = int(entry["tokens"]) * itemsize
        summed += int(entry["tokens"])
        try:
            got = os.path.getsize(path)
        except OSError as e:
            bad.append((src, entry["file"], f"unreadable: {e}"))
            continue
        total_bytes += got
        if got != want:
            bad.append((src, entry["file"],
                        f"size {got:,} != tokens*{itemsize} = {want:,} "
                        f"({got - want:+,} bytes)"))
    total_tokens += summed
    declared = int(man.get("total_tokens") or 0)
    flag = "" if declared == summed else f"  <-- manifest says {declared:,}"
    print(f"  {src:24s} {len(man['shards']):4d} shards  {summed:>15,} tokens{flag}")
    if declared != summed:
        bad.append((src, "manifest.json",
                    f"total_tokens {declared:,} != sum of shards {summed:,}"))

print(f"\n{n_shards} shards, {total_tokens:,} tokens, {total_bytes / 1e9:.2f} GB")

if bad:
    print(f"\n*** {len(bad)} PROBLEM(S) ***")
    for src, f, why in bad[:40]:
        print(f"  {src}/{f}: {why}")
    sys.exit(1)

print("every shard's byte length matches its manifest token count\n")

# --- 3. the real loader, over the real split -------------------------------
import torch  # noqa: E402

from train import MixtureBatchSource  # noqa: E402

print("building MixtureBatchSource over the real split (cpu) ...")
src = MixtureBatchSource(ROOT, micro_batch=2, device="cpu",
                         total_run_tokens=58_000_000_000, seed=0)
print(f"  sources resolved: {len(src.names)}")
for name, prob in sorted(zip(src.names, src.probs), key=lambda kv: -kv[1]):
    print(f"    {name:24s} {prob * 100:6.3f}%")

seen, lo, hi = 0, 10 ** 9, -1
for _ in range(8):
    x = src.get_batch(2048)
    assert x.shape == (2, 2048), x.shape
    seen += x.numel()
    lo, hi = min(lo, int(x.min())), max(hi, int(x.max()))
print(f"\npulled 8 batches at seq 2048: {seen:,} tokens, id range [{lo}, {hi}]")
