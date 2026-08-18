"""Collate runs/eval/peer-*.json (and optionally Daedalus's own results) into
the comparison table README's bar is stated in.

Exists because the bar has to be *measured*, not quoted. The published numbers
in README come from the MobileLLM paper's own evaluation setup (arXiv:2402.14905
Table 3); this harness reproduces them only to within 1-3 points per task, which
is the same size as the gaps between the peers themselves. So the honest
comparison is peers-and-us through one harness, with the published column kept
alongside as a cross-check rather than as the yardstick.

Usage: python scripts/peer_table.py [--ours runs/eval/results.json]
"""
import argparse
import glob
import json
import os

TASKS = ["hellaswag", "arc_easy", "piqa", "openbookqa", "winogrande"]

# MobileLLM arXiv:2402.14905 Table 3. Their 8-task average includes ARC-c,
# BoolQ and SIQA, which this project does not run (AGENT.md SS3 drops the noisy
# ones at this scale), so `published_avg` is NOT comparable to our 5-task mean
# and is carried only as the number the operator's bar was originally set from.
#
# `tokens` is each model's *published* training budget, checked against the
# primary source on 2026-08-10 rather than quoted from memory -- it is the
# column the "token count is not destiny" argument in the hero gate rests on,
# so a wrong entry there would misprice a $41 decision:
#   pythia-160m   299,892,736,000 tokens, stated exactly on the EleutherAI card
#   gpt-neo-125m  300B over 572,300 steps, stated on the EleutherAI card
#   opt-125m      180B / 800GB -- the OPT corpus (Zhang et al. 2022). Corrected
#                 from 300B on 2026-08-10; that figure was never sourced. The
#                 card gives no 125M-specific budget, so this is the corpus.
#   mobilellm-125m 1T, MobileLLM paper
#   smollm2-135m  2T, SmolLM2 paper
# openai-community/gpt2 is deliberately absent: OpenAI never disclosed GPT-2's
# training duration ("The training duration was not disclosed, nor were the
# exact details of training" -- the HF card), so it renders as "-" rather than
# as an invented estimate. See runs/eval/peer-token-budgets.md.
PUBLISHED = {
    "pythia-160m":    {"hellaswag": 29.9, "arc_easy": 40.0, "piqa": 62.0,
                       "openbookqa": 31.2, "winogrande": 50.9,
                       "published_avg": 42.5, "tokens": "300B"},
    "opt-125m":       {"hellaswag": 31.1, "arc_easy": 41.3, "piqa": 62.0,
                       "openbookqa": 31.2, "winogrande": 50.8,
                       "published_avg": 42.6, "tokens": "180B"},
    "gpt-neo-125m":   {"hellaswag": 29.7, "arc_easy": 40.7, "piqa": 62.5,
                       "openbookqa": 31.6, "winogrande": 50.7,
                       "published_avg": 42.9, "tokens": "300B"},
    "mobilellm-125m": {"hellaswag": 38.9, "arc_easy": 43.9, "piqa": 65.3,
                       "openbookqa": 39.5, "winogrande": 53.1,
                       "published_avg": 46.3, "tokens": "1T"},
    "smollm2-135m":   {"published_avg": 50.7, "tokens": "2T"},
}


def _slug(path: str) -> str:
    return os.path.basename(path)[len("peer-"):-len(".json")].split("/")[-1]


def _published_key(slug: str) -> str:
    """Match a filename slug to a PUBLISHED entry by suffix, not by splitting
    on the first '-': slugs carry the HF org (`eleutherai-pythia-160m`) and a
    positional split yields `160m`, which silently matches nothing and prints a
    table of dashes that looks like "no published number exists"."""
    for key in PUBLISHED:
        if slug == key or slug.endswith("-" + key):
            return key
    return slug


def load_rows(pattern: str = "runs/eval/peer-*.json"):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)["per_checkpoint"][0]
        slug = _slug(path)
        key = _published_key(slug)
        rows.append({"name": r.get("checkpoint", slug), "key": key,
                     **{t: r[t] * 100 for t in TASKS if t in r},
                     # Item counts ride along so the mean can carry an error
                     # bar. Without them the table renders differences of ~1
                     # point -- which is the entire spread of the peer group --
                     # as if they were exact.
                     **{f"{t}_n": r[f"{t}_n"] for t in TASKS
                        if f"{t}_n" in r}})
    return rows


def mean5(row) -> float:
    vals = [row[t] for t in TASKS if isinstance(row.get(t), float)]
    return sum(vals) / len(vals) if vals else float("nan")


def mean5_stderr(row):
    """Binomial standard error of the 5-task mean, in points, or None if the
    item counts are absent.

    The peers sit inside a 1.1-point band and this table is where that band is
    read, so rendering the mean bare invites a 0.3-point difference being taken
    for a result. `sqrt(sum(se_i^2))/k` -- averaging k tasks divides by k, it
    does not average the variances.

    Sampling error only. It does not bound seed variance, which at this scale
    is reported far larger and which nothing in this project has measured.
    """
    var, k = 0.0, 0
    for t in TASKS:
        p, n = row.get(t), row.get(f"{t}_n")
        if not isinstance(p, float) or not n:
            continue
        var += 100.0 * 100.0 * (p / 100.0) * (1 - p / 100.0) / float(n)
        k += 1
    return (var ** 0.5) / k if k else None


def render(rows) -> str:
    head = ("| model | " + " | ".join(t.replace("_", "-") for t in TASKS)
            + " | our 5-task mean | their published 8-task avg | tokens |")
    sep = "|" + "---|" * (len(TASKS) + 4)
    lines = [head, sep]
    for row in rows:
        pub = PUBLISHED.get(row["key"], {})
        cells = [f"{row[t]:.1f}" if isinstance(row.get(t), float) else "-"
                 for t in TASKS]
        # A row's own token count wins over the published table's: our rows are
        # not in PUBLISHED and must still state a budget (see ours_rows).
        toks = row.get("tokens", pub.get("tokens", "-"))
        se = mean5_stderr(row)
        mean_cell = (f"**{mean5(row):.1f}**" if se is None
                     else f"**{mean5(row):.1f}** ± {se:.2f}")
        lines.append(
            f"| {row['name']} | " + " | ".join(cells)
            + f" | {mean_cell} | {pub.get('published_avg', '-')} "
            + f"| {toks} |")
    lines += ["",
              "± is the binomial standard error of the 5-task mean "
              "(`sqrt(p(1-p)/n)` per task, `sqrt(sum)/5` for the mean). A "
              "difference must reach ~1.65 points to be two of these sigmas, "
              "so most gaps in this table are ties. Sampling error only — it "
              "does not bound seed variance, which is unmeasured here and "
              "larger. See `scripts/eval_noise.py` and `scripts/mcnemar.py` "
              "for the paired test, which is tighter."]
    return "\n".join(lines)


def render_delta(rows) -> str:
    """Per-task gap between what this harness measures and what the paper
    reports -- the evidence for using our own measurements as the bar."""
    lines = ["| model | " + " | ".join(TASKS) + " |", "|" + "---|" * (len(TASKS) + 1)]
    for row in rows:
        pub = PUBLISHED.get(row["key"], {})
        cells = []
        for t in TASKS:
            if isinstance(row.get(t), float) and t in pub:
                cells.append(f"{row[t] - pub[t]:+.1f}")
            else:
                cells.append("-")
        lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ours_rows(paths, labels, tokens):
    """Build our own rows, one per --ours file.

    Repeatable on purpose: a single score is a dot, and the question this table
    is asked ("is 42.2 reachable at 40B?") is about a *slope*, so the 0.5B probe
    and the 5B ablation checkpoint have to sit in the table together.

    A missing file raises rather than being skipped. The previous behaviour --
    `if os.path.exists(...)` -- silently rendered a peer-only table on a typo'd
    path, which reads exactly like "we scored and landed among the peers" while
    containing no Daedalus row at all.
    """
    if not paths:
        return []
    for name, seq in (("--ours-label", labels), ("--ours-tokens", tokens)):
        if seq and len(seq) != len(paths):
            raise SystemExit(
                f"{name} given {len(seq)} time(s) but --ours given "
                f"{len(paths)}; they pair positionally")
    rows = []
    for i, path in enumerate(paths):
        with open(path) as f:  # missing file -> FileNotFoundError, loudly
            m = json.load(f)["mean"]
        label = labels[i] if i < len(labels) else "Daedalus-150M"
        # "?" not "-": every peer row states a token budget, and this table is
        # meaningless without one. An unlabelled row invites reading a 0.5B
        # probe as the finished model.
        tok = tokens[i] if i < len(tokens) else "?"
        rows.append({"name": f"**{label}**", "key": f"daedalus-{i}",
                     "tokens": tok,
                     **{t: m[t] * 100 for t in TASKS if t in m},
                     # Same item counts as `load_rows` carries, for the same
                     # reason -- and kept in both because these two row
                     # builders have drifted before: the peer path once
                     # recorded `<task>_n` and the checkpoint path did not,
                     # which let a 500-example score print beside 10,042-example
                     # peers with nothing in either artifact showing it.
                     **{f"{t}_n": m[f"{t}_n"] for t in TASKS
                        if f"{t}_n" in m}})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="runs/eval/peer-*.json")
    p.add_argument("--ours", action="append", default=[],
                   help="our own eval results.json, appended as a row; repeatable")
    p.add_argument("--ours-label", action="append", default=[],
                   help="display name for the matching --ours row (pairs positionally)")
    p.add_argument("--ours-tokens", action="append", default=[],
                   help="training tokens for the matching --ours row (pairs positionally)")
    args = p.parse_args()

    rows = load_rows(args.pattern)
    rows += ours_rows(args.ours, args.ours_label, args.ours_tokens)

    print("### Measured on this harness, full validation splits, fp32\n")
    print(render(rows))
    print("\n### Our measurement minus the published table (points)\n")
    print(render_delta(rows))


if __name__ == "__main__":
    main()
