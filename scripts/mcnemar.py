#!/usr/bin/env python
"""Paired comparison of two models on the same eval items.

`scripts/eval_noise.py` reports the *unpaired* error of a difference, which is
an upper bound: every model answers the identical questions, so the items both
get right and both get wrong carry no information about which is better. This
throws those away and tests only the disagreements — McNemar's test — which is
the right instrument when the claim being made is decided on ~1 point.

Why that matters here specifically. The success definition is "beat
Pythia-160M, OPT-125M and GPT-neo-125M", and those three sit inside a
1.1-point band on our harness against an unpaired sigma of ±0.83. Under the
unpaired bound a 1-point win is unresolvable; paired, a 1-point win over 15,000
items usually is not.

Reads the `<out stem>.items.json` sidecars `eval.py` writes.

  python scripts/mcnemar.py runs/eval/a.items.json runs/eval/b.items.json \
      --a-label "us @5B" --b-label "Pythia-160M"
"""
import argparse
import json
import math
import os
from typing import Optional

TASKS = ["hellaswag", "arc_easy", "piqa", "openbookqa", "winogrande"]


def _pick(entry: dict) -> Optional[list]:
    """The outcome list matching the metric a published table would quote."""
    if entry is None:
        return None
    key = "items_" + entry.get("headline", "acc")
    return entry.get(key) or entry.get("items_acc")


def _only_model(side: dict) -> dict:
    """A sidecar holds {model_path: {task: ...}}. These files are written one
    model at a time, so exactly one key is expected; more than one is a caller
    error worth raising on rather than silently taking the first."""
    models = side.get("models") or {}
    if len(models) != 1:
        raise SystemExit(
            f"expected exactly one model per sidecar, found {len(models)}: "
            f"{sorted(models)}")
    return next(iter(models.values()))


def mcnemar(a_items: list, b_items: list) -> dict:
    """Exact-ish McNemar on the discordant pairs.

    b01 = a right / b wrong, b10 = a wrong / b right. The statistic uses the
    continuity-corrected chi-square, which is the standard reporting form and
    is well behaved down to a handful of discordant pairs; below ~10 it is
    reported but flagged, since the normal approximation is thin there.
    """
    if len(a_items) != len(b_items):
        raise ValueError(f"item counts differ: {len(a_items)} vs {len(b_items)}")
    b01 = sum(1 for x, y in zip(a_items, b_items) if x and not y)
    b10 = sum(1 for x, y in zip(a_items, b_items) if y and not x)
    n_disc = b01 + b10
    if n_disc == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "z": 0.0, "p": 1.0,
                "diff_pts": 0.0}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n_disc if n_disc else 0.0
    z = math.sqrt(max(chi2, 0.0))
    # Two-sided normal tail. math.erfc keeps this dependency-free.
    p = math.erfc(z / math.sqrt(2))
    return {"b01": b01, "b10": b10, "n_discordant": n_disc,
            "z": z, "p": p,
            "diff_pts": 100.0 * (b10 - b01) / len(a_items)}


def compare(a: dict, b: dict) -> dict:
    """Per task plus a pooled result over all five."""
    a_m, b_m = _only_model(a), _only_model(b)
    per_task, mismatches = {}, []
    pooled_a, pooled_b = [], []
    for task in TASKS:
        ea, eb = a_m.get(task), b_m.get(task)
        ia, ib = _pick(ea), _pick(eb)
        if ia is None or ib is None:
            continue
        if ea.get("digest") != eb.get("digest") or len(ia) != len(ib):
            # The failure this guards is silent and total: pairing item 7 of one
            # run against a different question in the other yields a confident,
            # meaningless p-value.
            mismatches.append(task)
            continue
        per_task[task] = mcnemar(ia, ib)
        pooled_a += ia
        pooled_b += ib
    out = {"per_task": per_task, "mismatched_tasks": mismatches}
    if pooled_a:
        out["pooled"] = mcnemar(pooled_a, pooled_b)
        out["pooled"]["n_items"] = len(pooled_a)
    return out


def render(result: dict, a_label: str, b_label: str) -> str:
    out = [f"# Paired comparison — {b_label} vs {a_label}", "",
           "McNemar over items both models answered. Only disagreements carry "
           "information, so this is a much tighter instrument than the "
           "unpaired error bar in `scripts/eval_noise.py`.", "",
           f"| task | {a_label} only | {b_label} only | Δ (pts) | z | p |",
           "|---|---|---|---|---|---|"]
    for task, r in result["per_task"].items():
        out.append(f"| {task} | {r['b01']} | {r['b10']} | {r['diff_pts']:+.2f} "
                   f"| {r['z']:.2f} | {r['p']:.4f} |")
    pooled = result.get("pooled")
    if pooled:
        out.append(f"| **pooled ({pooled['n_items']} items)** | {pooled['b01']} "
                   f"| {pooled['b10']} | **{pooled['diff_pts']:+.2f}** | "
                   f"{pooled['z']:.2f} | **{pooled['p']:.4f}** |")
        verdict = ("resolved" if pooled["p"] < 0.05 else "**not resolved**")
        out += ["", f"Pooled verdict at p<0.05: {verdict}. "
                    f"{pooled['n_discordant']} of {pooled['n_items']} items "
                    f"were discordant; the rest carry no information either way."]
        if pooled["n_discordant"] < 10:
            out.append("⚠ Fewer than 10 discordant pairs — the normal "
                       "approximation is thin here, read the counts not the p.")
    if result["mismatched_tasks"]:
        out += ["", "## Not compared", "",
                "These tasks' item fingerprints differ between the two runs, so "
                "pairing them would compare different questions: "
                + ", ".join(f"`{t}`" for t in result["mismatched_tasks"])
                + ". Re-score both sides with the same `--task-limit` and "
                  "dataset revision."]
    # Pooling across tasks weights each task by its item count, so HellaSwag
    # (10,042) dominates OpenBookQA (500). That is the right weighting for "did
    # this model answer more questions correctly" and the wrong one for the
    # 5-task *mean* the peer table reports, which weights tasks equally.
    out += ["", "Pooled counts items, not tasks — HellaSwag's 10,042 outweigh "
                "OpenBookQA's 500. That answers \"more questions right\", which "
                "is not the same question as the equally-weighted 5-task mean "
                "the peer table reports. Read both."]
    return "\n".join(out) + "\n"


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--a-label", default=None)
    p.add_argument("--b-label", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    result = compare(a, b)
    text = render(result,
                  args.a_label or os.path.basename(args.a),
                  args.b_label or os.path.basename(args.b))
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    _cli()
