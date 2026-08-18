#!/usr/bin/env python
"""How much of an eval difference is sampling noise?

The gate's central quality argument is a comparison of small numbers: 40.2 at
0.5B tokens against a 42.2 bar, with a 5B point arriving to make it a slope.
Nothing in this project has ever attached an error bar to those, and the tasks
are small -- OpenBookQA is 500 items, so a single question is 0.2 points of its
own score.

Unlike seed variance (which nothing here has measured -- `train.py` has no
`--seed` flag), this noise *is* computable: each task is n independent
Bernoulli trials, so the standard error of its accuracy is sqrt(p(1-p)/n), and
the 5-task mean's is sqrt(sum(se_i^2))/5.

Two honest caveats, stated in the output rather than buried here:

  - This is the *sampling* error of the benchmark only. It says nothing about
    seed variance, which at this scale the literature puts far higher (2-3
    points on accuracy) and which no measurement here bounds.
  - Comparing two checkpoints on the *same* items is a paired comparison, so
    the true error of the difference is smaller than the unpaired
    sqrt(2)*se reported here -- by how much depends on how correlated the two
    models' answers are, which needs per-item outputs this harness does not
    keep. The unpaired figure is therefore an upper bound: if a difference
    clears it, it is real; if it does not, this tool cannot say.

Usage:
  python scripts/eval_noise.py runs/eval/ours-sweep-lr0.02-500M.json
  python scripts/eval_noise.py a.json b.json --labels "0.5B" "5B"
"""
import argparse
import json
import math
import os
from typing import Optional

# The five tasks the success definition is measured on, in the order the peer
# table renders them. Each JSON stores the headline metric under the bare task
# name (acc_norm where the harness normalizes, acc for winogrande) and its item
# count under `<task>_n`.
TASKS = ["hellaswag", "arc_easy", "piqa", "openbookqa", "winogrande"]


def _metrics(data: dict) -> dict:
    """Both shapes this repo writes: a peer file is flat, ours has `mean`."""
    return data.get("mean") or data


def task_stderr(p: float, n: float) -> float:
    """Standard error of a proportion, in points."""
    if not n or n <= 0:
        return float("nan")
    return 100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def summarize(data: dict) -> dict:
    m = _metrics(data)
    per_task, var = {}, 0.0
    scores = []
    for task in TASKS:
        p, n = m.get(task), m.get(f"{task}_n")
        if p is None or n is None:
            continue
        se = task_stderr(float(p), float(n))
        per_task[task] = {"acc_pts": 100.0 * float(p), "n": int(n),
                          "stderr_pts": se}
        var += se ** 2
        scores.append(100.0 * float(p))
    if not scores:
        return {"per_task": {}, "mean_pts": None, "mean_stderr_pts": None}
    return {"per_task": per_task,
            "mean_pts": sum(scores) / len(scores),
            # The mean of k tasks, each an independent estimate: se = sqrt(sum
            # of variances) / k. Not sqrt(mean of variances) -- averaging
            # reduces the error, and conflating the two overstates it by k.
            "mean_stderr_pts": math.sqrt(var) / len(scores)}


def render(summaries: list, labels: list) -> str:
    out = ["# Eval sampling noise", "",
           "Standard error of each task's accuracy (`sqrt(p(1-p)/n)`) and of "
           "the 5-task mean. **Benchmark sampling only** — this does not bound "
           "seed variance, which nothing in this project has measured.", ""]
    out += ["| model | " + " | ".join(TASKS) + " | **5-task mean** |",
            "|---" * (len(TASKS) + 2) + "|"]
    for label, s in zip(labels, summaries):
        cells = []
        for task in TASKS:
            t = s["per_task"].get(task)
            cells.append("-" if t is None
                         else f"{t['acc_pts']:.1f} ± {t['stderr_pts']:.2f}")
        mean = ("-" if s["mean_pts"] is None
                else f"**{s['mean_pts']:.2f} ± {s['mean_stderr_pts']:.2f}**")
        out.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")
    out.append("")
    for label, s in zip(labels, summaries):
        t = s["per_task"].get("openbookqa")
        if t:
            out.append(f"`{label}`: OpenBookQA is {t['n']} items, so one "
                       f"question is {100.0/t['n']:.1f} points of that column "
                       f"alone.")
            break

    if len(summaries) == 2 and all(s["mean_pts"] is not None for s in summaries):
        a, b = summaries
        diff = b["mean_pts"] - a["mean_pts"]
        se = math.sqrt(a["mean_stderr_pts"] ** 2 + b["mean_stderr_pts"] ** 2)
        out += ["", f"**{labels[1]} − {labels[0]} = {diff:+.2f} points**, "
                    f"against an unpaired standard error of ±{se:.2f} "
                    f"({abs(diff)/se:.1f}σ).",
                "",
                "That σ is an **upper bound**: both models answer the same "
                "items, so the paired error is smaller by an amount only "
                "per-item outputs could quantify. A difference that clears it "
                "is real; one that does not is simply unresolved by this "
                "measurement.",
                "",
                f"For scale, a difference must reach **{2*se:.2f} points** to "
                f"be two of these sigmas."]
    return "\n".join(out) + "\n"


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+", help="eval JSON files")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    summaries, labels = [], []
    for i, path in enumerate(args.results):
        with open(path) as f:
            summaries.append(summarize(json.load(f)))
        labels.append(args.labels[i] if args.labels and i < len(args.labels)
                      else os.path.basename(path).replace(".json", ""))
    text = render(summaries, labels)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    _cli()
