"""Check a recovery run's inputs before spending GPU hours on them.

The first Phase 3 smoke run produced a non-finite loss on its very first step
and then span: `train_step` returns early on a non-finite loss without
advancing `self.step` or `tokens_seen`, so neither the `max_steps` nor the
`total_tokens` break condition is ever reached. It wrote 2,794 identical
skipped-update rows in ten minutes and would have run until the deadline.

Both halves of that need catching earlier than "the loss is NaN", because that
symptom is shared by causes with nothing in common:

- **the weights**, if the released checkpoint carries a tensor the Q4_0 grid
  cannot represent -- a block whose scale overflows fp16, or a value that is
  already non-finite on disk;
- **the data**, if a shard holds a token id outside the model's vocabulary. On
  CUDA an out-of-bounds embedding gather does not raise where it happens; it
  corrupts the context, and *every* later kernel returns garbage. That is why
  a data fault and a weight fault look identical from the metrics row: both
  end up with a NaN loss *and* a NaN `qat_rel_rmse`.

So the two are separated here by construction. Weights are checked on CPU,
where an out-of-range index raises instead of poisoning the device, and the
shards are checked by reading token ids directly rather than by running a
forward pass over them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def shard_token_range(shard_dir) -> dict:
    """Min and max token id across one source's shards, read off the files.

    Read from the memory-mapped `uint16` data rather than inferred from the
    manifest: the manifest records how many tokens were written, not what they
    were, and a tokenizer mismatch shows up only in the values.
    """
    shard_dir = Path(shard_dir)
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    lo, hi, total = None, None, 0
    for entry in manifest.get("shards", []):
        path = shard_dir / entry["file"]
        if not path.exists():
            return {"source": shard_dir.name, "ok": False,
                    "reason": f"manifest names {entry['file']}, which is absent"}
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if data.size == 0:
            continue
        shard_lo, shard_hi = int(data.min()), int(data.max())
        lo = shard_lo if lo is None else min(lo, shard_lo)
        hi = shard_hi if hi is None else max(hi, shard_hi)
        total += int(data.size)
    return {"source": shard_dir.name, "ok": True, "min_id": lo, "max_id": hi,
            "tokens": total, "shards": len(manifest.get("shards", []))}


def check_data(data_root, vocab_size: int) -> dict:
    """Every source's token ids against the model's vocabulary.

    An id at or above `vocab_size` is the failure this exists to catch: on CUDA
    it is not an exception, it is a corrupted context and a NaN loss that looks
    exactly like a numerical problem in the model.
    """
    data_root = Path(data_root)
    sources, failures = [], []
    for child in sorted(data_root.iterdir()):
        if not (child / "manifest.json").exists():
            continue
        report = shard_token_range(child)
        if report["ok"] and report.get("max_id") is not None:
            report["in_vocab"] = report["max_id"] < vocab_size
            if not report["in_vocab"]:
                failures.append(
                    f"{report['source']}: max token id {report['max_id']} "
                    f">= vocab_size {vocab_size}")
        elif not report["ok"]:
            failures.append(f"{report['source']}: {report['reason']}")
        sources.append(report)
    return {"passed": not failures, "failures": failures, "sources": sources,
            "vocab_size": vocab_size}


def check_weights(checkpoint: str, config: str) -> dict:
    """Load on CPU, put the model on the grid, and report what is not finite.

    Deliberately per tensor. "The loss is NaN" says nothing about where to
    look; "these three tensors leave the grid" says exactly where.
    """
    from daedalus import qat
    from daedalus.config import PRESETS
    from daedalus.model import Daedalus
    from train import load_checkpoint

    cfg = PRESETS[config]
    model = Daedalus(cfg)
    load_checkpoint(checkpoint, model, map_location="cpu")

    stored: List[dict] = []
    for name, parameter in model.named_parameters():
        if torch.isfinite(parameter).all():
            continue
        stored.append({"tensor": name,
                       "nan": int(torch.isnan(parameter).sum()),
                       "inf": int(torch.isinf(parameter).sum())})

    on_grid: List[dict] = []
    with torch.no_grad():
        for name, module, kind in qat.plan_qat(model):
            weight = qat.master_weight(module).detach().float()
            quantized = qat._QDQ[kind](weight)
            if torch.isfinite(quantized).all():
                continue
            # The scale is what overflows: `d = signed_absmax / -8` is stored
            # as fp16, and `(q - 8) * inf` is NaN wherever q lands on 8.
            absmax = float(weight.abs().max())
            on_grid.append({
                "tensor": name, "kind": kind, "absmax": absmax,
                "implied_scale": absmax / 8.0,
                "nan": int(torch.isnan(quantized).sum()),
                "inf": int(torch.isinf(quantized).sum()),
            })

    error = qat.quantization_error(model)
    return {
        "passed": not stored and not on_grid and math.isfinite(
            error["qat_rel_rmse"]),
        "non_finite_stored": stored,
        "non_finite_on_grid": on_grid,
        "qat_rel_rmse": error["qat_rel_rmse"],
        "qat_tensors": error["qat_tensors"],
        "checkpoint": checkpoint,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="daedalus-150m")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from daedalus.config import PRESETS
    report: Dict[str, object] = {}

    if args.data_dir:
        report["data"] = check_data(args.data_dir,
                                    PRESETS[args.config].vocab_size)
    if args.checkpoint:
        report["weights"] = check_weights(args.checkpoint, args.config)

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    failed = [k for k, v in report.items()
              if isinstance(v, dict) and not v.get("passed")]
    if failed:
        print(f"\nPREFLIGHT FAILED: {', '.join(failed)}")
        return 1
    print("\npreflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
