#!/usr/bin/env python
"""Render `runs/abl-arch/results.json` into the head-to-head table the
`[ASK HUMAN] ready for hero` issue and the writeup both need.

Why a script rather than reading the JSON by hand at the gate: `abl-arch`
finishes at ~06:45Z and the box idles at $10.78/day from that moment until the
operator answers, so the gate has to go up in minutes. Hand-carrying numbers
into an operator-facing decision document is also exactly where this project
has already been bitten -- two wrong peer token budgets and a mislabelled
success bar, all transcription rather than measurement.

Deliberately total: an arm that failed, or that is missing decode speed or the
perplexity check, renders as `-` with the reason kept, rather than raising.
Half an ablation is still worth reporting, and a renderer that crashes on a
partial result is useless at precisely the moment it is needed.
"""
import argparse
import json
import math
import os
from typing import Optional

# The hybrid is the model the project ships; the dense twin exists only as its
# param-matched control. Naming them here keeps the "who won" sentence from
# depending on dict ordering.
HYBRID = "daedalus-150m"
DENSE = "dense-150m"

# --- The pre-registered decision rule ---------------------------------------
# Full rationale, prediction and escalation arithmetic:
# runs/preflight/abl-arch-decision-rule.md, written before either arm scored.
#
# Relative held-out BPB gap below which the two arms are called a tie.
#
# Deliberately NOT imported from `abl_arch.NOISE_FRAC`, and deliberately not
# pinned equal to it by a test: that constant was calibrated for 0.5B-token *lr*
# probes and this is a 5B-token *architecture* comparison, so equality would be
# a coincidence rather than a shared quantity. The value is inherited and
# unmeasured -- `train.py` has no --seed flag, so every run this project has
# launched shares seed 0 and nothing here has ever measured seed variance in
# bits-per-byte. The rule is built so that being wrong about it is cheap: it
# decides whether the *operator is asked*, never whether the config silently
# switches.
QUALITY_NOISE_FRAC = 0.005

# Decode speed, unlike quality, has a measured sigma -- export.py's bench
# reports tok_per_sec_stddev over repeated llama-bench runs -- so the gap is
# called against the data rather than against a constant.
DECODE_SIGMAS = 2.0

# The context depth the headline decode ratio is quoted at.
#
# A bare llama-bench decodes into an *empty* context, which is the one regime
# where a conv hybrid has almost nothing to gain: its advantage is that only 6
# of 18 blocks keep a KV cache, and at depth 0 there is no KV cache to re-read.
# Measured on this box at matched threads and alternating rounds, the same two
# GGUFs read 1.15x at depth 0 and 1.83x at depth 2048
# (`runs/eval/decode-hybrid-vs-dense.json`). Quoting depth 0 would understate
# the project's headline Pareto claim by ~60% of itself.
#
# 2048 is the context these models are trained for, so it is the depth the
# claim is about. Depth 0 is still reported beside it -- it is what a reader
# reproducing with a default `llama-bench` invocation will see, and hiding that
# would make the result look irreproducible.
HEADLINE_DEPTH = 2048

# What switching `hero` to the dense twin costs -- derived, not restated.
#
# The dense arm trains at this fraction of the hybrid's rate (preflight:
# 100,561.4 vs 113,183.6 tok/s at micro-batch 16), so `hero`'s wall clock and
# cost both scale by 1/ratio. These were hardcoded as "+$4.99, +11.1 h" and
# "a 92-hour unattended run" -- correct at the 40B budget, and still being
# rendered into the gate's own headline table after the operator raised `hero`
# to 60B. Cost is linear in tokens, so pricing the budget explicitly lets the
# figures follow it; `test_the_dense_switch_cost_is_priced_at_heros_real_budget`
# fails if `hero.py`'s default moves away from HERO_BUDGET_PRICED.
DENSE_TRAIN_RATE_RATIO = 0.888
HERO_BUDGET_PRICED = 60_000_000_000
HERO_HOURS = 137.9
HERO_COST_USD = 61.89

DENSE_HOURS = HERO_HOURS / DENSE_TRAIN_RATE_RATIO
DENSE_EXTRA_HOURS = DENSE_HOURS - HERO_HOURS
DENSE_EXTRA_USD = HERO_COST_USD / DENSE_TRAIN_RATE_RATIO - HERO_COST_USD


def _get(entry: dict, *path, default=None):
    cur = entry
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _fmt(value, spec: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def attempts_from_metrics(run_name: Optional[str],
                          runs_root: str = "runs") -> Optional[int]:
    """How many times `train.py` was started for this run, counted from its own
    metrics.

    `run_abl_arch` records `attempts` on the entry it writes; the arm-2 recovery
    path (`scripts/finish_dense_arm.py`) records `resumed`/`resume_note` and no
    count, so the table printed `-` for the *only* arm that was ever resumed --
    a blank exactly where the caveat belongs, next to a `1` for the arm that ran
    clean.

    An attempt boundary is `step` going backwards, the same signal
    `watchdog.records_since_resume` uses: a resumed trainer continues from the
    checkpoint's step, so the counter drops. Counting here rather than fixing
    the writer is deliberate -- the writer is a live unattended process holding
    a --wait loop, and re-arming it hours before a $59.85 launch to add a
    cosmetic field is the worse trade. This reads what it already produced.
    """
    if not run_name:
        return None
    path = os.path.join(runs_root, run_name, "metrics.jsonl")
    attempts, prev = 1, None
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    step = json.loads(line).get("step")
                except json.JSONDecodeError:
                    continue
                if step is None:
                    continue
                if prev is not None and step < prev:
                    attempts += 1
                prev = step
    except OSError:
        return None
    return attempts if prev is not None else None


def summarize(data: dict, runs_root: str = "runs") -> dict:
    """Pull the four reported quantities per arm out of results.json."""
    out = {}
    for config, entry in (data.get("runs") or {}).items():
        export = _get(entry, "export", default={}) or {}
        speed = _get(export, "decode_speed", default={}) or {}
        # `by_depth` is keyed by depth-as-string in results.json; normalise to
        # int here so callers never have to know that. Absent for any result
        # written before export.py grew depths, in which case the top-level
        # (depth-0) keys are all there is and everything below falls back to
        # them rather than reporting nothing.
        by_depth = {}
        for key, item in (speed.get("by_depth") or {}).items():
            try:
                by_depth[int(key)] = item
            except (TypeError, ValueError):
                continue
        out[config] = {
            "config": config,
            "run_name": entry.get("run_name"),
            "error": entry.get("error"),
            "val_bpb": entry.get("val_bpb"),
            "decode_tok_per_sec": speed.get("tok_per_sec"),
            "decode_stddev": speed.get("tok_per_sec_stddev"),
            "decode_by_depth": by_depth,
            "n_threads": speed.get("n_threads"),
            "q4_0_delta_pct": export.get("delta_pct"),
            "passes_threshold": export.get("passes_threshold"),
            "attempts": (entry.get("attempts")
                         if entry.get("attempts") is not None
                         else attempts_from_metrics(entry.get("run_name"),
                                                    runs_root)),
            "resumed": entry.get("resumed"),
        }
    return out


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or not b:
        return None
    return a / b


def quality_gap(rows: dict) -> Optional[float]:
    """Relative held-out BPB gap, positive when the *hybrid* is better.

    None when either arm is missing a score, so callers cannot mistake an
    absent comparison for a tie.
    """
    hyb, den = rows.get(HYBRID), rows.get(DENSE)
    if not hyb or not den:
        return None
    hb, db = hyb.get("val_bpb"), den.get("val_bpb")
    if hb is None or db is None or not hb:
        return None
    return (db - hb) / hb


def decode_at(row: Optional[dict], depth: Optional[int]) -> tuple:
    """(tok_per_sec, stddev) for one arm at one depth.

    `depth=None` means the top-level keys, which are depth 0 by construction in
    `export.measure_decode_speed`. A depth that was requested but failed is
    stored with `tok_per_sec: None`, so it reads as missing here rather than as
    a measurement -- deliberately, because the deep benchmark is best-effort and
    a failure must not become a silent zero in a ratio.
    """
    if not row:
        return (None, None)
    if depth is None:
        return (row.get("decode_tok_per_sec"), row.get("decode_stddev"))
    item = (row.get("decode_by_depth") or {}).get(depth)
    if not item:
        return (None, None)
    return (item.get("tok_per_sec"), item.get("tok_per_sec_stddev"))


def headline_depth(rows: dict) -> Optional[int]:
    """The deepest depth both arms actually measured, preferring the trained
    context. None means neither arm has per-depth data and callers should fall
    back to the top-level depth-0 numbers.
    """
    hyb, den = rows.get(HYBRID), rows.get(DENSE)
    shared = sorted(set(d for d in (hyb or {}).get("decode_by_depth", {})
                        if decode_at(hyb, d)[0] is not None)
                    & set(d for d in (den or {}).get("decode_by_depth", {})
                          if decode_at(den, d)[0] is not None))
    if not shared:
        return None
    if HEADLINE_DEPTH in shared:
        return HEADLINE_DEPTH
    return shared[-1]


def decode_separation(rows: dict, depth: Optional[int] = None) -> Optional[float]:
    """How many combined sigmas separate the two decode measurements.

    `sqrt(sigma_h^2 + sigma_d^2)`, i.e. the sigma of the *difference*. None
    when either arm lacks a measurement or a stddev -- an unreported sigma must
    read as "unknown", never as "zero", which would call every gap significant.
    """
    hyb, den = rows.get(HYBRID), rows.get(DENSE)
    if not hyb or not den:
        return None
    hs, sh = decode_at(hyb, depth)
    ds, sd = decode_at(den, depth)
    if None in (hs, ds, sh, sd):
        return None
    combined = math.sqrt(sh ** 2 + sd ** 2)
    if not combined:
        return float("inf") if hs != ds else 0.0
    return abs(hs - ds) / combined


def verdict(rows: dict) -> str:
    """The two-sentence read of the ablation, stated so neither half can be
    quietly dropped: the claim is a *Pareto* one (quality and CPU decode), and
    reporting only the half that won would misrepresent it.

    Both halves are reported against a floor. The quality half used to declare
    a winner on any gap at all -- at 0.02% it would have printed "dense wins on
    held-out BPB" into the gate issue and the writeup, inside its own unmeasured
    noise, which is the defect `check_sweep.py` already fixed for the lr grid.
    """
    hyb, den = rows.get(HYBRID), rows.get(DENSE)
    if not hyb or not den:
        return ("Only one arm is present, so there is no comparison yet.")
    lines = []
    hb, db = hyb["val_bpb"], den["val_bpb"]
    dq = quality_gap(rows)
    if dq is not None:
        gap = abs(dq) * 100
        if gap < QUALITY_NOISE_FRAC * 100:
            lines.append(
                f"**Quality:** too close to call — hybrid {hb:.6f} vs dense "
                f"{db:.6f} is a {gap:.2f}% gap, inside the "
                f"{QUALITY_NOISE_FRAC*100:.1f}% floor these arms cannot resolve "
                f"(one seed, no measured seed sigma). Report it as a tie, not "
                f"as a win for "
                f"{'the hybrid' if dq > 0 else 'the dense twin'}.")
        else:
            better = "hybrid" if dq > 0 else "dense"
            lines.append(
                f"**Quality:** {better} wins on held-out BPB — hybrid {hb:.6f} "
                f"vs dense {db:.6f}, a {gap:.2f}% gap (lower is better).")
    else:
        lines.append("**Quality:** not comparable — an arm is missing val_bpb.")

    depth = headline_depth(rows)
    hs, _ = decode_at(hyb, depth)
    ds, _ = decode_at(den, depth)
    r = _ratio(hs, ds)
    sigmas = decode_separation(rows, depth)
    if r is not None:
        sig = ""
        if sigmas is None:
            sig = (" Separation is unmeasured — an arm reported no stddev, so "
                   "treat the ratio as indicative.")
        elif sigmas < DECODE_SIGMAS:
            sig = (f" But the gap is only {sigmas:.1f}σ of the bench's own "
                   f"noise (floor {DECODE_SIGMAS:.0f}σ), so the two decode at "
                   f"the same speed as far as this measurement can tell.")
        # Say which depth, always. The same two models read 1.15x at depth 0
        # and 1.83x at 2048, so a ratio quoted without its depth is not a
        # number anyone can check.
        if depth is None:
            where = ("at depth 0 (empty context) — the only depth this result "
                     "carries, which is where a conv hybrid has least to gain, "
                     "so read it as a floor")
        else:
            where = f"at context depth {depth}"
        base_hs, _ = decode_at(hyb, 0)
        base_ds, _ = decode_at(den, 0)
        base_r = _ratio(base_hs, base_ds)
        also = ""
        if depth and base_r is not None and depth != 0:
            also = (f" A default `llama-bench` run measures depth 0 instead and "
                    f"would report **{base_r:.2f}×**; the gap between those two "
                    f"numbers is the KV-cache traffic this architecture avoids, "
                    f"and it grows with context.")
        lines.append(
            f"**CPU decode ({where}):** hybrid {hs:.1f} tok/s vs dense "
            f"{ds:.1f} tok/s — **{r:.2f}×**. This is the measured half of the "
            f"Pareto claim and the reason the experiment exists.{sig}{also}")
    else:
        lines.append("**CPU decode:** not comparable — an arm is missing a "
                     "decode measurement.")
    return "\n\n".join(lines)


def decision(rows: dict) -> dict:
    """Apply the pre-registered rule and name the config `hero` should train.

    Returns {"hero_config", "escalate", "reason"}. `escalate` True means the
    gate must put the choice to the operator rather than proceed: it is set
    only when the dense twin wins quality by more than the floor, because that
    is the one outcome where the two halves of the mission disagree and the
    trade costs real money (+$4.99, +11.1 h, and the thinnest memory margin in
    the plan). See runs/preflight/abl-arch-decision-rule.md.

    Deliberately total. This is read at the launch of a $41.26 job, so a
    malformed or half-written results.json must fall through to the blueprint
    default rather than raise.
    """
    dq = quality_gap(rows)
    if dq is None:
        return {"hero_config": HYBRID, "escalate": False,
                "reason": ("The ablation did not decide it — an arm is missing "
                           "a held-out BPB. The blueprint default stands.")}
    gap = abs(dq) * 100
    if gap < QUALITY_NOISE_FRAC * 100:
        return {"hero_config": HYBRID, "escalate": False,
                "reason": (f"Quality is a tie ({gap:.2f}% < "
                           f"{QUALITY_NOISE_FRAC*100:.1f}%), and a tie goes to "
                           f"the blueprint — as it did for the sweep's lr. The "
                           f"hybrid is also ~${DENSE_EXTRA_USD:.2f} cheaper and "
                           f"{DENSE_EXTRA_HOURS:.1f} h faster to train, and it "
                           f"is the arm the CPU-decode claim needs.")}
    if dq > 0:
        return {"hero_config": HYBRID, "escalate": False,
                "reason": (f"The hybrid wins quality by {gap:.2f}%, beyond the "
                           f"{QUALITY_NOISE_FRAC*100:.1f}% floor. Unambiguous, "
                           f"and the Pareto claim is clean.")}
    return {"hero_config": DENSE, "escalate": True,
            "reason": (f"The dense twin wins quality by {gap:.2f}%, beyond the "
                       f"{QUALITY_NOISE_FRAC*100:.1f}% floor. This is the one "
                       f"outcome the rule refuses to decide alone: switching "
                       f"costs +${DENSE_EXTRA_USD:.2f} and +{DENSE_EXTRA_HOURS:.1f} h "
                       f"of `hero`, puts a {DENSE_HOURS:.0f}-hour "
                       f"unattended run on ~28.4 GB of 32.6, and gives up the "
                       f"CPU-decode half of the Pareto claim — the only axis on "
                       f"which the strongest peer is beaten at all.")}


def apply_paired(rows: dict, paired: Optional[dict]) -> bool:
    """Replace both arms' decode numbers with one alternating pass over both.

    Each arm benchmarks itself inside its own export step, and those steps are
    ~12 h apart -- exactly the non-simultaneous comparison that once reported
    1.29x where alternating rounds put the same measurement at 1.15x. When
    `scripts/rebench_arms.py` has produced a paired file, its numbers are the
    comparable ones and these are what the ratio should be computed from.

    All-or-nothing: if the paired file does not cover *both* arms it is ignored
    entirely, because half a paired measurement mixed with half a stale one is
    worse than either alone. Returns whether it was applied.
    """
    if not paired:
        return False
    by_config: dict = {}
    for pass_ in (paired.get("passes") or []):
        try:
            depth = int(pass_.get("depth"))
        except (TypeError, ValueError):
            continue
        for config, item in (pass_.get("models") or {}).items():
            if not item or item.get("mean") is None:
                continue
            by_config.setdefault(config, {})[depth] = {
                "depth": depth,
                "tok_per_sec": item["mean"],
                "tok_per_sec_stddev": item.get("stdev"),
            }
    if not all(rows.get(c) and by_config.get(c) for c in (HYBRID, DENSE)):
        return False
    for config in (HYBRID, DENSE):
        rows[config]["decode_by_depth"] = by_config[config]
        base = by_config[config].get(0)
        if base:
            rows[config]["decode_tok_per_sec"] = base["tok_per_sec"]
            rows[config]["decode_stddev"] = base["tok_per_sec_stddev"]
    return True


def load_paired(results_path: str) -> Optional[dict]:
    """The paired decode file that sits beside results.json, if it exists.

    Found by convention rather than by flag: this is rendered at ~06:00Z with
    nobody watching, and a flag nobody passes is a measurement nobody uses.
    """
    path = os.path.join(os.path.dirname(results_path) or ".",
                        "decode-paired.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        # Read at the launch of a $41.26 job: a corrupt sidecar must degrade to
        # the per-arm numbers, never raise.
        return None


def _depth_section(rows: dict, paired: bool = False) -> list:
    """Decode speed against context depth, for both arms.

    Emitted only when both arms carry per-depth data, because a one-row table
    claiming to be a depth sweep is worse than no table. The point of showing
    every depth rather than just the headline is that the *trend* is the
    mechanism: a gated short conv's decode cost is flat in context length,
    attention's KV reads grow with it, so a ratio that climbs with depth is the
    architecture doing what it was chosen to do -- and a flat one would mean
    something is wrong with the claim.
    """
    hyb, den = rows.get(HYBRID), rows.get(DENSE)
    depths = sorted(set((hyb or {}).get("decode_by_depth", {}))
                    & set((den or {}).get("decode_by_depth", {})))
    if not depths:
        return []
    out = ["", "### CPU decode against context depth", "",
           "| depth | hybrid | dense twin | ratio |", "|---|---|---|---|"]
    for d in depths:
        hs, hsd = decode_at(hyb, d)
        ds, dsd = decode_at(den, d)
        r = _ratio(hs, ds)
        hcell = _fmt(hs, ".1f") + (f" ± {hsd:.1f}" if hs is not None
                                   and hsd is not None else "")
        dcell = _fmt(ds, ".1f") + (f" ± {dsd:.1f}" if ds is not None
                                   and dsd is not None else "")
        mark = "**" if d == HEADLINE_DEPTH else ""
        out.append(f"| {mark}{d}{mark} | {hcell} | {dcell} | "
                   f"{mark}{_fmt(r, '.2f')}×{mark} |" if r is not None
                   else f"| {d} | {hcell} | {dcell} | - |")
    out += ["",
            f"Depth {HEADLINE_DEPTH} is the trained context and the depth the "
            f"Pareto claim is about; depth 0 is what a default `llama-bench` "
            f"invocation reports. Only 6 of the hybrid's 18 blocks keep a KV "
            f"cache against all 24 of the dense twin's, so the dense twin "
            f"re-reads exactly 2× the KV bytes per decoded token and the gap "
            f"widens with context."]
    if paired:
        out += ["",
                "Both arms measured in **one alternating pass** at matched "
                "thread counts (`scripts/rebench_arms.py`), not in their "
                "separate export steps ~12 h apart — absolute llama-bench "
                "numbers move with whatever else the box is doing, so only a "
                "single invocation gives a comparable ratio."]
    else:
        out += ["",
                "⚠ Each arm measured itself during its own export, ~12 h "
                "apart. Absolute llama-bench numbers move with box load, so "
                "this ratio is indicative; run `scripts/rebench_arms.py` for "
                "an alternating measurement of both arms in one pass."]
    return out


def render(data: dict, paired: Optional[dict] = None) -> str:
    rows = summarize(data)
    is_paired = apply_paired(rows, paired)
    tokens = data.get("total_tokens_per_run")
    lr = data.get("lr") or {}
    out = ["# `abl-arch` — hybrid vs dense, param-matched, identical data", ""]
    out.append(
        f"Both arms: **{_fmt(tokens, ',')} tokens** each, Muon lr "
        f"**{_fmt(lr.get('muon_lr'))}** (source: {lr.get('source', 'unknown')}), "
        f"same seed and same data order.")
    depth = headline_depth(rows)
    col = ("CPU decode (tok/s) @ depth 0" if depth is None
           else f"CPU decode (tok/s) @ depth {depth}")
    out += ["",
            f"| arm | val_bpb | {col} | fp16→Q4_0 Δ | attempts |",
            "|---|---|---|---|---|"]
    for config in (HYBRID, DENSE):
        r = rows.get(config)
        if r is None:
            out.append(f"| `{config}` | _absent_ | - | - | - |")
            continue
        ts, sd = decode_at(r, depth)
        speed = _fmt(ts, ".1f")
        if ts is not None and sd is not None:
            speed += f" ± {sd:.1f}"
        delta = _fmt(r["q4_0_delta_pct"], ".3f")
        if r["q4_0_delta_pct"] is not None:
            delta += "%" + ("" if r["passes_threshold"] else " ⚠")
        out.append(f"| `{config}` | {_fmt(r['val_bpb'], '.6f')} | {speed} | "
                   f"{delta} | {_fmt(r['attempts'])} |")
    out += ["", verdict(rows)]
    out += _depth_section(rows, is_paired)

    # The rule was fixed before either arm scored, which is the only reason the
    # claim it produces is worth anything. Rendering it here rather than
    # applying it by hand at the gate keeps that true.
    d = decision(rows)
    out += ["", "## What `hero` trains", "",
            f"**`{d['hero_config']}`**" +
            ("  — **but this one escalates.**" if d["escalate"] else ""),
            "", d["reason"], "",
            "Rule pre-registered before either arm scored → "
            "`runs/preflight/abl-arch-decision-rule.md`."]

    failed = [(c, r["error"]) for c, r in rows.items() if r.get("error")]
    if failed:
        out += ["", "## Arms that did not complete", ""]
        for config, err in failed:
            out.append(f"- `{config}`: {err}")
    # A ⚠ above is not a failure of the ablation -- it is the Q4_0 quality bar
    # AGENT.md SS3 sets for the *shipped* artifact, and abl-arch's arms are 5B
    # probes without QAT. Saying so here stops it being read as a broken export.
    if any(r["passes_threshold"] is False for r in rows.values()):
        out += ["", f"⚠ = fp16→Q4_0 delta above the 1% bar. Expected here: "
                    f"these are 5B probes trained without the QAT tail that "
                    f"`hero` runs over its final 5%."]
    return "\n".join(out) + "\n"


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="runs/abl-arch/results.json")
    p.add_argument("--out", default=None, help="write markdown here too")
    args = p.parse_args(argv)

    if not os.path.exists(args.results):
        raise SystemExit(f"no results file at {args.results}")
    with open(args.results) as f:
        data = json.load(f)
    text = render(data, load_paired(args.results))
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    _cli()
