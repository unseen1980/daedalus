#!/usr/bin/env python
"""Can llama.cpp's `lfm2` graph load a model whose short-conv width < hidden size?

This answers the one question `runs/preflight/conv-death-decision-rule.md` leaves
open for option C ("prune the dead channels at export"):

    "It is gated only on whether llama.cpp's `lfm2` graph tolerates conv width !=
     hidden size, which is an export question answerable on CPU at any time."

Nobody had answered it. The finding it gates is worth answering for: ~48% of the
short-conv channels in `hero` contribute exactly nothing, and structurally
removing them would cut weight bytes per decoded token -- which is the number
this project leads with.

Rather than reading `src/models/lfm2.cpp` and reasoning about it, this builds the
artifact and hands it to a real `llama.cpp` binary. That is deliberate: every
expensive bug on this project has been a green argument around an unexecuted
artifact.

Method: take a **real** Daedalus GGUF, rewrite it with every short-conv tensor
narrowed from `n_embd` to `--pruned-width` (a uniform prune -- the *easiest*
case for a loader, since a real per-layer prune gives each conv block its own
width), and run `llama-perplexity` against it.

Uniform is the right case to test first: if the loader rejects uniform narrowing
it rejects per-layer narrowing a fortiori, so a failure here closes option C
without needing to build the harder artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

# The three tensors llama.cpp creates at fixed `n_embd` in
# llama_model_lfm2::load_arch_tensors (src/models/lfm2.cpp).
#   conv     {n_shortconv_l_cache, n_embd}   -> numpy (n_embd, L)
#   in_proj  {n_embd, 3 * n_embd}            -> numpy (3*n_embd, n_embd)
#   out_proj {n_embd, n_embd}                -> numpy (n_embd, n_embd)
# numpy shape is the reverse of the GGUF `ne` ordering.
CONV_SUFFIX = ".shortconv.conv.weight"
INPROJ_SUFFIX = ".shortconv.in_proj.weight"
OUTPROJ_SUFFIX = ".shortconv.out_proj.weight"


def narrow(name: str, arr: np.ndarray, width: int) -> np.ndarray:
    """Slice a short-conv tensor down to `width` conv channels.

    Which axis is the conv-channel axis differs per tensor, so this is spelled
    out rather than inferred -- getting it wrong would produce a file that fails
    to load for the wrong reason and would answer the question falsely.
    """
    if name.endswith(CONV_SUFFIX):
        return arr[:width, :]           # (n_embd, L)      -> (width, L)
    if name.endswith(INPROJ_SUFFIX):
        return arr[: 3 * width, :]      # (3*n_embd, n_embd)-> (3*width, n_embd)
    if name.endswith(OUTPROJ_SUFFIX):
        return arr[:, :width]           # (n_embd, n_embd) -> (n_embd, width)
    return arr


def is_conv_tensor(name: str) -> bool:
    return name.endswith((CONV_SUFFIX, INPROJ_SUFFIX, OUTPROJ_SUFFIX))


def copy_kv(reader: GGUFReader, writer: GGUFWriter) -> int:
    """Copy every KV field except the ones GGUFWriter emits itself."""
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
            "general.architecture"}
    n = 0
    for key, field in reader.fields.items():
        if key in skip:
            continue
        if not field.types:
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            sub = field.types[1]
            if sub == GGUFValueType.STRING:
                vals = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
            else:
                vals = [field.parts[i].tolist()[0] for i in field.data]
            writer.add_key_value(key, vals, GGUFValueType.ARRAY, sub_type=sub)
        elif vtype == GGUFValueType.STRING:
            writer.add_key_value(
                key, bytes(field.parts[field.data[0]]).decode("utf-8"), vtype)
        else:
            writer.add_key_value(key, field.parts[field.data[0]].tolist()[0], vtype)
        n += 1
    return n


def build_pruned(src: str, dst: str, width: int) -> dict:
    reader = GGUFReader(src)
    arch = bytes(
        reader.fields["general.architecture"].parts[
            reader.fields["general.architecture"].data[0]]).decode("utf-8")
    writer = GGUFWriter(dst, arch)
    n_kv = copy_kv(reader, writer)

    touched, n_embd = 0, None
    for t in reader.tensors:
        arr = np.array(t.data)
        if is_conv_tensor(t.name):
            if n_embd is None and t.name.endswith(OUTPROJ_SUFFIX):
                n_embd = arr.shape[0]
            arr = narrow(t.name, arr, width)
            touched += 1
        writer.add_tensor(t.name, arr, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"kv_copied": n_kv, "conv_tensors_narrowed": touched,
            "n_embd": int(n_embd) if n_embd else None}


def try_load(gguf_path: str, binary: str, text_file: str, timeout: int = 600) -> dict:
    """Hand the file to a real llama.cpp binary and record what it says."""
    cmd = [binary, "-m", gguf_path, "-f", text_file, "-c", "512", "--chunks", "1",
           "-t", "2"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"rc": None, "loaded": False, "error": "timeout"}
    out = (p.stdout or "") + (p.stderr or "")
    return {
        "rc": p.returncode,
        "loaded": p.returncode == 0,
        "wrong_shape": "wrong shape" in out,
        "tail": "\n".join(
            [ln for ln in out.splitlines() if ln.strip()][-6:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-gguf",
                    default="runs/abl-arch-daedalus-150m/export/model-f16.gguf")
    ap.add_argument("--work-dir", default="/tmp/conv-prune-probe")
    ap.add_argument("--pruned-width", type=int, default=640,
                    help="conv channels to keep (< n_embd)")
    ap.add_argument("--llama-perplexity",
                    default="/workspace/llama.cpp/build/bin/llama-perplexity")
    ap.add_argument("--text-file", default=None,
                    help="corpus for the load attempt; a tiny one is generated "
                         "if omitted, since we are testing load not quality")
    ap.add_argument("--out", default="runs/preflight/conv-prune-feasibility.json")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    text_file = args.text_file
    if text_file is None:
        text_file = os.path.join(args.work_dir, "tiny.txt")
        with open(text_file, "w") as fh:
            fh.write(("the quick brown fox jumps over the lazy dog. " * 400) + "\n")

    if not os.path.exists(args.src_gguf):
        print(f"source GGUF not found: {args.src_gguf}", file=sys.stderr)
        return 2

    result: dict = {"src_gguf": args.src_gguf, "pruned_width": args.pruned_width}

    # Control 1: the unmodified file must load, or a failure below proves nothing.
    print(f"[control]   loading unmodified {args.src_gguf}")
    result["control"] = try_load(args.src_gguf, args.llama_perplexity, text_file)
    print(f"  rc={result['control']['rc']} loaded={result['control']['loaded']}")

    # Control 2: the same file rebuilt through *this script's* writer at full
    # width. Without this, a rejected narrow file is ambiguous -- it could mean
    # llama.cpp refuses the narrowing, or merely that the writer emits a broken
    # GGUF. Only the difference between control 2 and the pruned build isolates
    # the narrowing as the cause.
    n_embd = GGUFReader(args.src_gguf)
    n_embd = next(int(np.array(t.data).shape[0]) for t in n_embd.tensors
                  if t.name.endswith(OUTPROJ_SUFFIX))
    rt = os.path.join(args.work_dir, f"roundtrip-{n_embd}.gguf")
    print(f"[roundtrip] rebuilding at full width n_embd={n_embd} -> {rt}")
    result["roundtrip_build"] = build_pruned(args.src_gguf, rt, n_embd)
    result["roundtrip"] = try_load(rt, args.llama_perplexity, text_file)
    print(f"  rc={result['roundtrip']['rc']} loaded={result['roundtrip']['loaded']}")

    pruned = os.path.join(args.work_dir, f"pruned-{args.pruned_width}.gguf")
    print(f"[build]     narrowing short-conv tensors -> {args.pruned_width}")
    result["build"] = build_pruned(args.src_gguf, pruned, args.pruned_width)
    print(f"  {result['build']}")

    print(f"[pruned]    loading {pruned}")
    result["pruned"] = try_load(pruned, args.llama_perplexity, text_file)
    print(f"  rc={result['pruned']['rc']} loaded={result['pruned']['loaded']} "
          f"wrong_shape={result['pruned'].get('wrong_shape')}")

    if not result["control"]["loaded"]:
        result["verdict"] = "INCONCLUSIVE: the unmodified control did not load"
    elif not result["roundtrip"]["loaded"]:
        result["verdict"] = ("INCONCLUSIVE: this script's writer cannot round-trip "
                             "a loadable GGUF at full width, so a rejected narrow "
                             "build says nothing about the narrowing")
    elif args.pruned_width >= n_embd:
        result["verdict"] = (f"NOT A TEST: --pruned-width {args.pruned_width} is not "
                             f"below n_embd {n_embd}; nothing was narrowed")
    elif result["pruned"]["loaded"]:
        result["verdict"] = "FEASIBLE: llama.cpp loaded a narrowed short-conv model"
    else:
        result["verdict"] = ("INFEASIBLE: llama.cpp rejects conv width != n_embd; "
                             "option C requires patching llama.cpp")
    print(f"\n{result['verdict']}")
    if result["pruned"].get("tail"):
        print("pruned load said:")
        print(result["pruned"]["tail"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
