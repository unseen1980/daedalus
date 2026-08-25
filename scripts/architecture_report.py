"""Phase 6 stage A scoring: what the fifteen arms bought, and what advances.

`architecture_sweep.py` deliberately stopped at launching. This is the other
half: read the finished arms on a held-out full pass, put the analytic columns
beside the measured one, and apply the stage B advancement rule. The rule is in
this file, committed, *before* any arm has a score -- which is the only order in
which a preregistered threshold means anything.

Four decisions carry the weight.

**The measurement is a full pass on the matched holdout.** Every arm logged a
`val_bpb` during training, and none of those numbers can decide this. The last
one lands at step 1,500 of 1,536, where the learning rate is still an order of
magnitude above its floor -- it measures a model mid-decay, not the model the
arm produced. It is also a bounded sample. So the arms are re-scored from their
final checkpoints, over every held-out window, and the scorecard records
`bpb_mode: full`; a sampled card is refused rather than read.

Matched means `fineweb-edu` alone. The arms trained on that source and nothing
else; the two other sources in the box's holdout root would measure transfer,
which is a real question and not this one, at three times the cost.

**The control is a grid point, and every column is a delta against it.**
`a8-kv4` is the shipped model's own ratio at the probe's width and depth. An
absolute BPB at 105M parameters over 100M tokens means very little; the same
number as a distance from the shipped ratio, measured under an identical
schedule, means what the phase is asking.

**A quality win inside the parameter margin is not a win.** The grid is
parameter-matched only to +/-1.5%: a conv block is dearer than an attention
block, so trading attention away *adds* parameters, and the sparsest arm carries
3.05% more than the densest. That bias points at exactly the arms a KV-savings
phase hopes will win. Every row therefore carries `credited_bpb_delta_pct`
alongside the raw delta -- the raw delta with the improvement its parameter
surplus could explain removed -- and `quality_win_survives_param_margin`, which
is false for an arm whose whole advantage is that it is bigger.

The discount uses `PARAM_SCALING_EXPONENT = 0.34`, Chinchilla's exponent on N.
That is deliberately an upper bound rather than an estimate: total loss carries
an irreducible term, so the true sensitivity of loss to parameters is strictly
smaller than 0.34, and over-crediting the surplus makes it *harder* for a sparse
arm to be called a winner. Conservative in the direction the bias runs.

**The floor is the plan's own gate, not a new one.** An arm advances only if its
raw BPB is no worse than the control by more than 0.5% -- the number phase 6 was
preregistered with, applied to the raw delta rather than the discounted one so
that no threshold is invented here after the fact. The discount is reported, and
carried into stage B, where 150M candidates can be matched more tightly than a
single 256-wide FFN step allows at this scale.

Among arms that clear the floor, what advances is the Pareto frontier on (KV
bytes down, BPB down), cheapest cache first. The floor already says "does not
lose quality"; among shapes that keep quality, the valuable one is the one whose
cache costs least. The frontier matters because KV cost ties: 8 attention layers
with 1 KV head, 4 with 2, and 2 with 4 all cost the same bytes per context
token, and at equal cost only the better-scoring shape is worth stage B's hours.

Subcommands: `score`, `report`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daedalus.scorecard import (ArtifactRef, ScorecardError, load_scorecard,  # noqa: E402
                                sha256_file)
from scripts.architecture_sweep import (ARMS, CONTROL, REPORT_ROOT,  # noqa: E402
                                        RUN_ROOT, STAGE_A, ArchArm, StageShape,
                                        arm_checkpoint_path, arm_run_name,
                                        parameter_spread, selected_arms)
from scripts.bpb_eval import _git_short_sha, run_bpb_eval  # noqa: E402


# ============================================================ preregistered ===
# Named here, in the commit that precedes the first score. A threshold chosen
# after the numbers are on screen is not a threshold.

#: The one source the stage-A arms trained on, and therefore the only one whose
#: held-out BPB answers this screen's question.
MATCHED_HOLDOUT_SOURCE = "fineweb-edu"

#: Phase 6's own gate: an arm is out if its BPB is worse than the control's by
#: more than this, in percent. Applied to the raw delta -- see the module
#: docstring on why the parameter discount informs rather than gates.
STAGE_B_FLOOR_PCT = 0.5

#: How many non-control arms stage B re-runs at 150M over 250M tokens. Stage C
#: takes at most two finalists, so a wider stage B would buy resolution the next
#: stage cannot spend. The control runs in every stage regardless.
STAGE_B_MAX_ARMS = 3

#: Chinchilla's exponent on N in `L = E + A/N^0.34 + B/D^0.28`, used to discount
#: the parameter surplus. An upper bound, not an estimate: `E` is irreducible,
#: so `dlogL/dlogN` is strictly smaller than this in magnitude. Over-crediting
#: the surplus is the conservative error here.
PARAM_SCALING_EXPONENT = 0.34

#: Where the per-arm scorecards land, beside the sweep artifact rather than
#: inside the run directories -- a scorecard must never be mistaken for a
#: checkpoint.
SCORECARD_ROOT = f"{REPORT_ROOT}/scorecards"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# =================================================================== scoring ===

def scorecard_name(arm: ArchArm, tag: str = "stagea") -> str:
    return f"arch-{tag}-{arm.name}-bpb"


def scorecard_path(arm: ArchArm, *, tag: str = "stagea",
                   out_dir: str = SCORECARD_ROOT) -> Path:
    return Path(out_dir) / f"{scorecard_name(arm, tag)}.json"


def scored_from(path, checkpoint_sha: str) -> bool:
    """True when `path` already scores exactly these bytes.

    Keyed on the checkpoint digest rather than on the file existing, so a
    re-scored arm is skipped only when the thing that was scored is the thing
    that is there now. A rerun after `--refresh` retrained an arm must not
    silently keep the old arm's number.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        card = load_scorecard(path)
    except (ScorecardError, KeyError, ValueError, OSError):
        return False
    return card.provenance.artifact.sha256 == checkpoint_sha


def make_checkpoint_bpb_fn(arm: ArchArm, checkpoint, *, device: str,
                           seq_len: int, batch_size: int
                           ) -> Callable[[Path], float]:
    """Load one arm's final weights and return its per-source BPB callable.

    Imports live inside because this module's selection logic is pure and its
    tests must not need torch to exercise the rule that decides the phase.
    """
    from daedalus.config import PRESETS
    from daedalus.data import get_tokenizer
    from daedalus.model import Daedalus
    from eval import evaluate_bpb
    from train import load_checkpoint

    tokenizer = get_tokenizer(None)
    model = Daedalus(PRESETS[arm.config]).to(device)
    load_checkpoint(str(checkpoint), model, map_location=device)
    model.eval()

    def bpb_fn(source_dir: Path) -> float:
        # max_batches=None is the full pass the gate requires.
        return evaluate_bpb(model, str(source_dir), seq_len, tokenizer, device,
                            batch_size=batch_size, max_batches=None)

    return bpb_fn


def _release(device: str) -> None:
    """Drop the previous arm's weights before the next 105M model lands.

    Fifteen arms scored in one process on a 24GB card is fifteen models plus
    fifteen sets of activations unless each is released.
    """
    import gc

    gc.collect()
    if not device.startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:                                   # pragma: no cover
        pass


def score_arm(arm: ArchArm, *, holdout_root: str, tag: str = "stagea",
              run_root: str = RUN_ROOT, out_dir: str = SCORECARD_ROOT,
              source: str = MATCHED_HOLDOUT_SOURCE, device: str = "cuda",
              seq_len: int = 2048, batch_size: int = 8, seed: int = 20260824,
              refresh: bool = False,
              bpb_factory: Callable[..., Callable[[Path], float]]
              = make_checkpoint_bpb_fn) -> dict:
    """Full-pass BPB for one finished arm, written as a scorecard."""

    checkpoint = arm_checkpoint_path(arm, tag, run_root)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"arm {arm.name} has no checkpoint at {checkpoint}; it either "
            "never ran or its run directory was moved")
    digest = sha256_file(checkpoint)
    path = scorecard_path(arm, tag=tag, out_dir=out_dir)
    if not refresh and scored_from(path, digest):
        return {"arm": arm.name, "scorecard": str(path), "skipped": "already-scored",
                "checkpoint_sha256": digest}

    bpb_fn = bpb_factory(arm, checkpoint, device=device, seq_len=seq_len,
                         batch_size=batch_size)
    try:
        written = run_bpb_eval(
            name=scorecard_name(arm, tag), holdout_root=holdout_root,
            out_dir=out_dir,
            artifact=ArtifactRef(path=str(checkpoint), sha256=digest,
                                 kind="checkpoint", config=arm.config),
            tokenizer_ref=ArtifactRef(path="<smollm2-default>", sha256="0" * 64,
                                      kind="tokenizer"),
            seed=seed, git_sha=_git_short_sha(), bpb_fn=bpb_fn,
            max_batches=None, sources=[source],
            runtime={"device": device, "seq_len": seq_len,
                     "batch_size": batch_size},
            details_extra={"phase": "phase6-architecture", "arm": arm.name,
                           "preset": arm.config, "tag": tag,
                           "run": arm_run_name(arm, tag)})
    finally:
        del bpb_fn
        _release(device)
    return {"arm": arm.name, "scorecard": str(written["scorecard"]),
            "checkpoint_sha256": digest}


def score_arms(arms: Sequence[ArchArm] = ARMS, **kwargs) -> dict:
    """Score every arm, control first, leaving each scorecard as it lands.

    Re-entrant for the same reason the sweep is: this is a GPU pass over fifteen
    checkpoints, and a session that ends mid-pass must cost only the arm it was
    on.
    """
    results = []
    for arm in arms:
        results.append(score_arm(arm, **kwargs))
    return {"scored": results}


# ================================================================= reporting ===

def credited_bpb_delta_pct(bpb_delta_pct: float, param_surplus_pct: float,
                           exponent: float = PARAM_SCALING_EXPONENT) -> float:
    """The raw delta with the improvement its parameter surplus could explain
    removed.

    An arm 3.05% larger than the control could buy about 1.04% of BPB with size
    alone under an exponent that is already an upper bound. An arm that came in
    0.5% better is therefore 0.54% *worse* once its extra parameters are paid
    for -- which is the reading this phase has to make, because the arms it
    hopes to advance are precisely the larger ones.
    """
    return bpb_delta_pct + exponent * param_surplus_pct


def arm_bpb(card, source: str = MATCHED_HOLDOUT_SOURCE) -> float:
    """The measured BPB for one source, refusing anything that is not a full
    pass over it.

    A bounded sample and a full pass are different measurements, and this one
    feeds a gate. Reading `metrics["bpb"]` would have taken whatever weighting
    the card happened to carry; reading the item asserts which corpus it is.
    """
    if card.provenance.bpb_mode != "full":
        raise ScorecardError(
            f"scorecard {card.name!r} was measured in bpb_mode "
            f"{card.provenance.bpb_mode!r}; a stage-A gate reads a full pass "
            "only")
    for item in card.items or ():
        if item.get("id") == source:
            return float(item["bpb"])
    raise ScorecardError(
        f"scorecard {card.name!r} has no item for source {source!r}; "
        f"it scored {[item.get('id') for item in card.items or ()]}")


def read_rows(arms: Sequence[ArchArm] = ARMS, *, tag: str = "stagea",
              out_dir: str = SCORECARD_ROOT,
              source: str = MATCHED_HOLDOUT_SOURCE) -> List[dict]:
    """One row per scored arm, measured column joined to analytic ones."""

    control_row = None
    rows: List[dict] = []
    for arm in arms:
        path = scorecard_path(arm, tag=tag, out_dir=out_dir)
        if not path.exists():
            continue
        card = load_scorecard(path)
        analytic = arm.describe()
        row = {
            "arm": arm.name,
            "preset": arm.config,
            "is_control": arm.is_control,
            "attention_layers": analytic["attention_layers"],
            "num_key_value_heads": analytic["num_key_value_heads"],
            "kv_bytes_per_context_token": analytic["kv_bytes_per_context_token"],
            "parameters": analytic["parameters"],
            "q4_0_MB": analytic["q4_0_MB"],
            "bpb": arm_bpb(card, source),
            "checkpoint_sha256": card.provenance.artifact.sha256,
            "scorecard": str(path),
        }
        rows.append(row)
        if arm.is_control:
            control_row = row

    if control_row is None:
        raise ScorecardError(
            f"no scorecard for the control arm {CONTROL.name!r} under "
            f"{out_dir}; every stage-A column is a delta against it, so a "
            "table without it is unreadable rather than merely incomplete")

    for row in rows:
        row["bpb_delta_pct"] = 100.0 * (row["bpb"] - control_row["bpb"]) \
            / control_row["bpb"]
        row["param_surplus_pct"] = 100.0 * (
            row["parameters"] - control_row["parameters"]) \
            / control_row["parameters"]
        row["param_explained_bpb_pct"] = -PARAM_SCALING_EXPONENT \
            * row["param_surplus_pct"]
        row["credited_bpb_delta_pct"] = credited_bpb_delta_pct(
            row["bpb_delta_pct"], row["param_surplus_pct"])
        row["kv_saving_pct"] = 100.0 * (
            control_row["kv_bytes_per_context_token"]
            - row["kv_bytes_per_context_token"]) \
            / control_row["kv_bytes_per_context_token"]
        row["passes_floor"] = row["bpb_delta_pct"] <= STAGE_B_FLOOR_PCT
        row["quality_win_survives_param_margin"] = (
            row["bpb_delta_pct"] < 0.0 and row["credited_bpb_delta_pct"] < 0.0)
    return rows


def pareto_frontier(rows: Sequence[dict]) -> List[dict]:
    """Rows no other row beats on both KV bytes and BPB, cheapest cache first.

    Needed because KV cost ties across the grid -- 8 attention layers with 1 KV
    head costs exactly what 4 with 2 and 2 with 4 do -- and at equal cache cost
    only the better-scoring shape is worth stage B's hours.
    """
    frontier = [
        row for row in rows
        if not any(
            other is not row
            and other["kv_bytes_per_context_token"] <= row["kv_bytes_per_context_token"]
            and other["bpb"] <= row["bpb"]
            and (other["kv_bytes_per_context_token"] < row["kv_bytes_per_context_token"]
                 or other["bpb"] < row["bpb"])
            for other in rows)
    ]
    return sorted(frontier, key=lambda row: (row["kv_bytes_per_context_token"],
                                             row["arm"]))


def select_stage_b(rows: Sequence[dict], *, floor_pct: float = STAGE_B_FLOOR_PCT,
                   max_arms: int = STAGE_B_MAX_ARMS) -> dict:
    """Apply the preregistered advancement rule and say what it decided."""

    eligible = [row for row in rows
                if not row["is_control"] and row["bpb_delta_pct"] <= floor_pct]
    frontier = pareto_frontier(eligible)
    selected = frontier[:max_arms]
    verdict = "advance" if selected else "no-advance"
    decision = {
        "rule": {
            "floor_pct": floor_pct,
            "max_arms": max_arms,
            "param_scaling_exponent": PARAM_SCALING_EXPONENT,
            "floor": "raw bpb_delta_pct <= floor_pct, phase 6's own gate",
            "order": "Pareto frontier on (kv_bytes down, bpb down), "
                     "cheapest cache first",
        },
        "eligible": [row["arm"] for row in eligible],
        "frontier": [row["arm"] for row in frontier],
        "selected": [row["arm"] for row in selected],
        "dropped_from_frontier": [row["arm"] for row in frontier[max_arms:]],
        "verdict": verdict,
    }
    if verdict == "no-advance":
        decision["note"] = (
            "no arm held the control's quality within the 0.5% floor, so no "
            "attention or KV-head reduction advances. The shipped ratio stands "
            "on this evidence; stage B does not run. Recorded as a negative "
            "result rather than re-run at a looser threshold.")
    else:
        unsurvived = [row["arm"] for row in selected
                      if row["bpb_delta_pct"] < 0
                      and not row["quality_win_survives_param_margin"]]
        if unsurvived:
            decision["note"] = (
                f"{unsurvived} score better than the control on raw BPB, but "
                "not by more than their parameter surplus could explain. They "
                "advance on the KV saving, not on a quality win, and must not "
                "be reported as beating the shipped ratio.")
    return decision


def build_report(rows: Sequence[dict], *, tag: str = "stagea",
                 shape: StageShape = STAGE_A,
                 source: str = MATCHED_HOLDOUT_SOURCE) -> dict:
    control = next(row for row in rows if row["is_control"])
    ordered = sorted(rows, key=lambda row: (row["kv_bytes_per_context_token"],
                                            row["arm"]))
    return {
        "tag": tag,
        "created_at": _utcnow(),
        "holdout_source": source,
        "bpb_mode": "full",
        "shape": {"name": shape.name, "seq_len": shape.seq_len,
                  "total_tokens": shape.total_tokens, "steps": shape.steps},
        "control": control["arm"],
        "control_bpb": control["bpb"],
        "parameter_spread": parameter_spread(),
        "rows": ordered,
        "stage_b": select_stage_b(rows),
        "caveats": [
            "The grid is parameter-matched only to +/-1.5%, and the residual "
            "favours attention-sparse arms: a conv block is dearer than an "
            "attention block, so cutting attention adds parameters. "
            "credited_bpb_delta_pct discounts that surplus at Chinchilla's "
            "0.34 exponent on N, which is an upper bound and therefore "
            "conservative against exactly the arms this phase hopes to "
            "advance.",
            "A ranking measured on 105M-parameter proxies over 100M tokens is "
            "a ranking at that scale. Stage B re-runs the survivors at 150M "
            "over 250M tokens for that reason, and nothing here should be "
            "quoted as a property of a larger successor.",
            "BPB is the only measured column. Retrieval by depth, GGUF "
            "export/load and decode shape are preregistered stage-6 gates that "
            "this pass does not measure; no arm is recommended on BPB alone.",
            f"Scored on {source} alone, the one source the arms trained on. "
            "Transfer to held-out code and other web text is unmeasured at "
            "stage A.",
        ],
    }


def render_markdown(report: dict) -> str:
    """The table a human reads, with the discount column beside the raw one."""

    lines = [
        f"# Phase 6 stage A: attention x KV-head screen ({report['tag']})",
        "",
        f"Full-pass held-out BPB on `{report['holdout_source']}`, "
        f"{report['shape']['total_tokens']:,} tokens at "
        f"{report['shape']['seq_len']} context. Control `{report['control']}` "
        f"= {report['control_bpb']:.4f} BPB.",
        "",
        "| arm | attn | kv | KV B/tok | KV saved | params | BPB | vs ctrl % | "
        "param surplus % | credited % | floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in report["rows"]:
        mark = " (control)" if row["is_control"] else ""
        lines.append(
            f"| `{row['arm']}`{mark} | {row['attention_layers']} | "
            f"{row['num_key_value_heads']} | "
            f"{row['kv_bytes_per_context_token']} | "
            f"{row['kv_saving_pct']:+.0f}% | "
            f"{row['parameters'] / 1e6:.1f}M | {row['bpb']:.4f} | "
            f"{row['bpb_delta_pct']:+.2f} | {row['param_surplus_pct']:+.2f} | "
            f"{row['credited_bpb_delta_pct']:+.2f} | "
            f"{'pass' if row['passes_floor'] else 'FAIL'} |")

    stage_b = report["stage_b"]
    lines += [
        "",
        "## Stage B selection",
        "",
        f"- rule: raw delta <= {stage_b['rule']['floor_pct']}% of control, then "
        f"{stage_b['rule']['order']}, capped at {stage_b['rule']['max_arms']}",
        f"- eligible: {stage_b['eligible'] or 'none'}",
        f"- frontier: {stage_b['frontier'] or 'none'}",
        f"- **selected: {stage_b['selected'] or 'none'}**",
        f"- verdict: `{stage_b['verdict']}`",
    ]
    if stage_b.get("note"):
        lines += ["", f"> {stage_b['note']}"]
    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in report["caveats"]]
    return "\n".join(lines) + "\n"


# ====================================================================== cli ====

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--scorecard-root", default=SCORECARD_ROOT)
    parser.add_argument("--tag", default="stagea")
    parser.add_argument("--source", default=MATCHED_HOLDOUT_SOURCE)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score")
    score.add_argument("--holdout-root", required=True,
                       help="the root holding one manifest-backed directory "
                            "per source; only --source is measured")
    score.add_argument("--arms", default=None,
                       help="comma-separated subset, control first")
    score.add_argument("--device", default="cuda")
    score.add_argument("--seq-len", type=int, default=STAGE_A.seq_len)
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--seed", type=int, default=20260824)
    score.add_argument("--refresh", action="store_true",
                       help="re-score arms whose scorecard already matches "
                            "their checkpoint")

    sub.add_parser("report")

    args = parser.parse_args(argv)

    if args.command == "score":
        report = score_arms(selected_arms(args.arms), holdout_root=args.holdout_root,
                            tag=args.tag, run_root=args.run_root,
                            out_dir=args.scorecard_root, source=args.source,
                            device=args.device, seq_len=args.seq_len,
                            batch_size=args.batch_size, seed=args.seed,
                            refresh=args.refresh)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "report":
        rows = read_rows(ARMS, tag=args.tag, out_dir=args.scorecard_root,
                         source=args.source)
        report = build_report(rows, tag=args.tag, source=args.source)
        json_path = Path(args.report_root) / f"{args.tag}-report.json"
        markdown_path = Path(args.report_root) / f"{args.tag}-report.md"
        _write_json(json_path, report)
        markdown_path.write_text(render_markdown(report))
        print(render_markdown(report))
        print(f"wrote {json_path} and {markdown_path}")
        return 0

    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
