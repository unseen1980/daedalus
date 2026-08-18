"""Attribute training memory between the loss head and everything else.

The chunked loss head bounds its own peak at `loss_chunk_size * vocab_size * 4`
bytes, but that only helps if the loss head is what dominates. This probe
measures, for a given batch size, how peak memory responds to `loss_chunk_size`
so the remaining floor (parameters, optimizer state, block activations) is
visible separately.

By default the probe now also builds Muon + AdamW (`--optim`) and runs one
full `backward(); step()` so its peak reflects what `smoke.py`'s real loop
holds: fp32 params, fp32 grads, Muon momentum buffers, and AdamW's two fp32
moment buffers -- none of which the forward+backward-only probe captured.
Add `--compile` to also fold in `torch.compile`/Inductor autotune workspace,
and `--isolate` to run every (batch, chunk) point in its own subprocess, so
one data point's peak can't be inflated by memory the caching allocator
didn't hand back from the previous point (matches how `smoke.py` loops over
batch sizes in a single process).

Run: python tools/mem_probe.py --batches 8,16,32 --chunks 1024 --optim --grad-checkpoint
"""
import argparse
import copy
import subprocess
import sys

import torch

from daedalus.config import PRESETS
from daedalus.model import Daedalus
from daedalus.muon import build_optimizers, momentum_warmup


def probe(cfg_name, batch, seq, chunk, grad_ckpt=False, use_optim=False,
          use_compile=False, device="cuda"):
    """Run one forward+backward(+optimizer step) and report peak memory, or None on OOM."""
    cfg = copy.deepcopy(PRESETS[cfg_name])
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, seq)
    cfg.loss_chunk_size = chunk
    cfg.gradient_checkpointing = grad_ckpt

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = Daedalus(cfg).to(device)
    model.train()
    muon = adamw = None
    if use_optim:
        muon, adamw, _ = build_optimizers(model, muon_lr=0.02, adam_lr=3e-4)
    params_gb = torch.cuda.memory_allocated() / 1e9

    net = torch.compile(model) if use_compile else model
    x = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    try:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss, _ = net(x, targets=x)
        loss.backward()
        if use_optim:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in muon.param_groups:
                g["momentum"] = momentum_warmup(0)
            muon.step(); adamw.step()
            muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        result = {"chunk": chunk, "peak_GB": round(peak, 2),
                  "params_GB": round(params_gb, 2),
                  "loss": round(loss.item(), 3)}
    except torch.cuda.OutOfMemoryError:
        result = {"chunk": chunk, "peak_GB": None, "params_GB": round(params_gb, 2),
                  "loss": None}
    finally:
        del model, net, x, muon, adamw
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return result


def probe_isolated(cfg_name, batch, seq, chunk, grad_ckpt, use_optim, use_compile):
    """Run `probe()` in a fresh subprocess so its peak can't inherit memory a
    prior data point's caching allocator kept reserved in this process."""
    cmd = [sys.executable, __file__, "--config", cfg_name,
           "--batches", str(batch), "--seq", str(seq), "--chunks", str(chunk),
           "--_single"]
    if grad_ckpt:
        cmd.append("--grad-checkpoint")
    if use_optim:
        cmd.append("--optim")
    if use_compile:
        cmd.append("--compile")
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.strip().startswith("{"):
            import ast
            return ast.literal_eval(line.strip())
    return {"chunk": chunk, "peak_GB": None, "params_GB": None, "loss": None,
            "stderr_tail": out.stderr[-2000:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="daedalus-150m")
    p.add_argument("--batches", default="8")
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--chunks", default="1024")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute block activations in backward")
    p.add_argument("--optim", action="store_true",
                   help="build Muon+AdamW and run one step so their state "
                        "(momentum buffers, fp32 grads) is in the peak")
    p.add_argument("--compile", action="store_true",
                   help="wrap the model in torch.compile before the step")
    p.add_argument("--isolate", action="store_true",
                   help="run each (batch, chunk) point in its own subprocess "
                        "so peaks can't inherit memory a prior point's "
                        "caching allocator kept reserved in-process")
    p.add_argument("--_single", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._single:
        # subprocess worker mode: run exactly one point, print it, exit
        batch = int(args.batches.split(",")[0])
        chunk = int(args.chunks.split(",")[0])
        r = probe(args.config, batch, args.seq, chunk, args.grad_checkpoint,
                  args.optim, args.compile)
        print(r)
        return

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu       : {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
    print(f"seq       : {args.seq}")
    print(f"block ckpt: {args.grad_checkpoint}")
    print(f"optim     : {args.optim}")
    print(f"compile   : {args.compile}")
    print(f"isolate   : {args.isolate}")
    print("-" * 68)

    for batch in [int(b) for b in args.batches.split(",")]:
        for chunk in [int(c) for c in args.chunks.split(",")]:
            if args.isolate:
                r = probe_isolated(args.config, batch, args.seq, chunk,
                                    args.grad_checkpoint, args.optim, args.compile)
            else:
                r = probe(args.config, batch, args.seq, chunk,
                          args.grad_checkpoint, args.optim, args.compile)
            theoretical = chunk * PRESETS[args.config].vocab_size * 4 / 1e9
            label = f"  batch {batch:3d} chunk {chunk:5d}:"
            if r["peak_GB"] is None:
                print(f"{label} OOM           "
                      f"(loss-head fp32 chunk alone = {theoretical:.2f} GB)")
            else:
                print(f"{label} peak {r['peak_GB']:5.2f} GB  "
                      f"params {r['params_GB']:4.2f} GB  "
                      f"(loss-head chunk = {theoretical:.2f} GB)  "
                      f"loss {r['loss']}")


if __name__ == "__main__":
    main()
