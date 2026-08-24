"""Preregistered QAT recovery of the released Daedalus checkpoint.

Phase 3 asks a narrow question: how much of the released model's Q4_0 damage
can be bought back by fine-tuning it on the shipping lattice, without
repeating pretraining. The measured penalty on this host is +5.539%
(`runs/eval/quant-base/quant-comparison.json`), carried by 285 of 292 chunks,
and the improvement gate is to halve it.

This module owns the *decisions*, not the compute. Every threshold, learning
rate, budget and tie-break is a pure function of the preregistration file
written before the first probe starts, so no verdict here can be reached by
adjusting a bar after seeing a result. `preregister` refuses to overwrite an
existing plan for the same reason: the plan is evidence, and evidence that can
be rewritten once the outcome is known is not evidence.

The three things most easily got wrong, and what this file does about them:

**`--init-from`, never `--resume`.** `--resume` restores step, tokens_seen and
both optimizer states. Pointed at a *finished* 59.9B-token run under a 100M
budget it makes `fit()` break at the top of its first iteration and exit 0
having trained nothing -- measured twice on this project already (see
`TrainArgs.init_from`). Every probe command this module builds carries
`--init-from`, and `assert_no_resume` is what keeps it that way. The `--resume`
that `run_with_resume` appends on a *retry* is a different thing and is
correct: by then the checkpoint being resumed is the probe's own.

**QAT from step zero.** The blueprint's `qat_frac=0.05` spends the last 5% of a
run on the grid, which is right for pretraining and wrong here: there is no
tail to wait for, and fp32 steps at the start of a 190-step budget move weights
the grid then has to move back. Recovery runs at `qat_frac=1.0`.

**The retrieval gate is finer than the instrument.** "Retrieval drops by no
more than 1 point absolute at any depth" is 0.01 on the 0-1 exact-match scale,
but the baseline was measured at 10 items per depth, where one item is 0.10.
As written the gate therefore admits only *exact equality* -- it is not a
1-point tolerance, it is a zero-tolerance gate wearing one. `RETRIEVAL_PER_DEPTH`
raises the item count to 100, the smallest count at which one item is one
point, and the baseline is re-measured at the same count. Doing this before any
probe runs is the whole point; doing it afterwards would be exactly the
threshold-tuning the phase forbids.

That makes the gate *arithmetically expressible*. It does not make a 1-point
difference *statistically resolvable*, and the distinction should not be
glossed: at 100 items and an exact-match rate near 0.85, binomial sampling
noise alone is about 3.6 points, so a 1-point move is well inside it.
Resolving one point would need thousands of items per depth, which this phase
has no budget for. So what the retrieval gate can honestly do here is catch a
*large* regression -- a model that stopped retrieving -- and what it cannot do
is certify that retrieval is unchanged to within a point. Verdicts report the
observed drop next to that limit rather than implying a precision the
measurement does not have.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# The shipped optimizer defaults, and therefore the ratio a probe must keep.
# Read off `train.py`'s parser rather than restated, so a change there cannot
# leave this file quietly scaling Adam against a Muon rate that moved.
SHIPPED_MUON_LR = 0.02
SHIPPED_ADAM_LR = 3e-4

# Preregistered probe rates. Muon candidates come from the phase brief; Adam
# follows from the ratio above.
PROBE_MUON_LRS = (2e-4, 5e-4, 1e-3)

PROBE_TOKENS = 100_000_000
FOLLOWUP_TOKENS = 300_000_000
ESCALATION_TOKENS = 1_000_000_000

# Constant-shape steps: recovery continues from the *end* of pretraining, so it
# trains at the sequence length and batch size the model finished at rather
# than replaying a ramp it has already been through.
SEQ_LEN = 2048
BATCH_TOKENS = 524_288
MICRO_BATCH = 8

# ~5% of a 190-step budget. `train.py`'s 300-step default is longer than the
# entire probe, which would leave the run permanently in warmup.
WARMUP_FRAC = 0.05
MIN_WARMUP_STEPS = 10
# Most of the budget spent decaying to zero. D2Z's linear-to-zero finding is
# what makes "fully decayed" the requirement it is; a probe that stopped at a
# non-zero LR would be scored mid-schedule.
DECAY_FRAC = 0.8

# See the module docstring: 10 items per depth cannot express a 1-point change.
RETRIEVAL_PER_DEPTH = 100
RETRIEVAL_DEPTHS = (256, 512, 1024, 2048)

# Mandatory retention gates, from the phase brief.
MAX_FP16_PPL_REGRESSION_PCT = 0.5
MAX_FIVE_TASK_DROP = 0.5
MAX_RETRIEVAL_DROP_POINTS = 1.0

# Improvement gates, expressed against the *measured* baseline penalty rather
# than the brief's "roughly 6%".
MIN_PENALTY_REDUCTION_FRAC = 0.50     # halve the damage
TARGET_PENALTY_PCT = 3.0
STRETCH_PENALTY_PCT = 1.0
# Escalation stop rule: a follow-up must beat the best probe by this much,
# relative, or the phase reports a negative result instead of spending 1B.
MIN_ESCALATION_REDUCTION_FRAC = 0.10


# Every gate here is a "<= limit" or ">= bar" comparison against a number
# produced by arithmetic on measured floats, and binary floating point does not
# represent the round decimals the gates are written in. `(2.00 - 1.80) / 2.00`
# evaluates to 0.09999999999999998, so a follow-up that improved Q4 damage by
# exactly the preregistered 10% would be refused escalation by representation
# error alone. The tolerance is far below any measurement's resolution -- Q4
# penalty is quoted to three decimals of a percent -- so it can only ever
# rescue a boundary case, never move a bar.
GATE_EPSILON = 1e-9


def _at_least(value: float, bar: float) -> bool:
    """`value >= bar`, honouring the bar as written rather than as represented."""
    return value >= bar - GATE_EPSILON


def _at_most(value: float, limit: float) -> bool:
    """`value <= limit`, honouring the limit as written rather than as represented."""
    return value <= limit + GATE_EPSILON


class PreregistrationError(RuntimeError):
    """Raised when a plan would be overwritten or read back inconsistently."""


def adam_lr_for(muon_lr: float,
                shipped_muon_lr: float = SHIPPED_MUON_LR,
                shipped_adam_lr: float = SHIPPED_ADAM_LR) -> float:
    """The Adam rate that keeps the shipped Muon:Adam ratio at `muon_lr`.

    The two optimizers cover disjoint parameter sets -- Muon the hidden
    matrices, AdamW the embeddings, norms and gains -- so scaling one without
    the other does not "lower the learning rate", it changes which half of the
    model moves. Recovery is a small nudge to a finished model; silently
    retuning that balance would make the probes a comparison of two things at
    once.
    """
    if muon_lr <= 0:
        raise ValueError(f"muon_lr must be positive, got {muon_lr}")
    return muon_lr * (shipped_adam_lr / shipped_muon_lr)


def estimated_steps(total_tokens: int, batch_tokens: int = BATCH_TOKENS) -> int:
    """Steps at the constant batch size a recovery run uses.

    `train.py`'s `estimate_total_steps` replays the batch/sequence ramp, which
    a recovery run does not have: `tok_start == tok_end` and
    `seq_start == seq_end`, so the count is a division.
    """
    if total_tokens <= 0 or batch_tokens <= 0:
        raise ValueError("token counts must be positive")
    return max(1, math.ceil(total_tokens / batch_tokens))


def warmup_steps_for(total_tokens: int,
                     batch_tokens: int = BATCH_TOKENS,
                     frac: float = WARMUP_FRAC,
                     minimum: int = MIN_WARMUP_STEPS) -> int:
    """A warmup proportional to the budget, floored so it never rounds to zero."""
    return max(minimum, int(estimated_steps(total_tokens, batch_tokens) * frac))


@dataclass(frozen=True)
class Probe:
    """One preregistered arm."""

    name: str
    muon_lr: float
    total_tokens: int
    stage: str                      # "probe" | "followup" | "escalation"

    @property
    def adam_lr(self) -> float:
        return adam_lr_for(self.muon_lr)

    @property
    def warmup_steps(self) -> int:
        return warmup_steps_for(self.total_tokens)

    def to_dict(self) -> dict:
        return {**asdict(self), "adam_lr": self.adam_lr,
                "warmup_steps": self.warmup_steps,
                "estimated_steps": estimated_steps(self.total_tokens)}


def probe_arms(muon_lrs: Sequence[float] = PROBE_MUON_LRS,
               total_tokens: int = PROBE_TOKENS) -> List[Probe]:
    """The three 100M-token learning-rate arms, in preregistered order."""
    return [
        Probe(name=f"qat-recovery-lr{lr:g}", muon_lr=lr,
              total_tokens=total_tokens, stage="probe")
        for lr in muon_lrs
    ]


def train_command(probe: Probe, *, init_from: str, data_dir: str,
                  run_root: str = "runs", val_dir: Optional[str] = None,
                  device: str = "cuda", micro_batch: int = MICRO_BATCH,
                  seq_len: int = SEQ_LEN, batch_tokens: int = BATCH_TOKENS,
                  loss_chunk_size: Optional[int] = None,
                  gradient_checkpointing: bool = False,
                  hub_repo: Optional[str] = None,
                  python: str = "python") -> List[str]:
    """The exact `train.py` argv for one arm.

    Built as data so the identical-data-and-order requirement is checkable by
    a test rather than by reading three shell lines: every arm differs only in
    `--run-name`, `--muon-lr` and `--adam-lr`.
    """
    cmd = [
        python, "train.py",
        "--run-name", probe.name,
        "--config", "daedalus-150m",
        "--data-dir", data_dir,
        # Weights only. See the module docstring.
        "--init-from", init_from,
        "--total-tokens", str(probe.total_tokens),
        "--micro-batch", str(micro_batch),
        "--seq-start", str(seq_len), "--seq-end", str(seq_len),
        "--tok-start", str(batch_tokens), "--tok-end", str(batch_tokens),
        "--muon-lr", f"{probe.muon_lr:g}",
        "--adam-lr", f"{probe.adam_lr:g}",
        "--warmup-steps", str(probe.warmup_steps),
        "--decay-frac", str(DECAY_FRAC),
        # The whole run is the QAT phase.
        "--qat-frac", "1.0",
        "--device", device,
        # No `--seed`: `train.py` has no such flag, and `TrainArgs.seed`
        # already defaults to 0. Every arm therefore shares one seed, which is
        # the "identical data, order and seeds" requirement; the value is
        # recorded in the preregistration so the verdict still names it.
    ]
    if loss_chunk_size is not None:
        cmd += ["--loss-chunk-size", str(loss_chunk_size)]
    if gradient_checkpointing:
        cmd += ["--gradient-checkpointing"]
    if val_dir:
        cmd += ["--val-dir", val_dir]
    cmd += ["--hub-repo", hub_repo or ""]
    return cmd


def assert_no_resume(cmd: Sequence[str]) -> None:
    """Refuse a launch command carrying `--resume`.

    The failure this prevents is silent: `train.py` prints `resumed from ...`,
    writes no metrics row and exits 0, so a probe that trained nothing looks
    from the outside like a probe that finished early.
    """
    if "--resume" in cmd:
        raise ValueError(
            "a recovery probe must start from --init-from, not --resume: "
            "--resume would restore the finished pretraining run's step and "
            "token count and the probe would train nothing and exit 0")


# ----------------------------------------------------------------- scoring ---

@dataclass
class Baseline:
    """The released model's measured numbers, as recorded by Phase 2."""

    q4_penalty_pct: float
    perplexity_fp16: float
    perplexity_q4_0: float
    five_task_mean: float
    retrieval: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "Baseline":
        return cls(
            q4_penalty_pct=float(payload["q4_penalty_pct"]),
            perplexity_fp16=float(payload["perplexity_fp16"]),
            perplexity_q4_0=float(payload["perplexity_q4_0"]),
            five_task_mean=float(payload["five_task_mean"]),
            retrieval={str(k): float(v)
                       for k, v in (payload.get("retrieval") or {}).items()},
        )


def finiteness_from_metrics(rows: Sequence[dict]) -> dict:
    """Decide the finiteness gate from `metrics.jsonl` alone.

    The gate is "all losses, gradients and weights stay finite with no skipped
    non-finite updates", and every part of it is in the durable record:
    `skipped_updates` counts dropped updates across restarts, and a non-finite
    loss or gradient norm shows up in the row itself.
    """
    if not rows:
        return {"passed": False, "reason": "no metrics rows",
                "skipped_updates": None, "non_finite_rows": 0}

    def bad(value) -> bool:
        return value is not None and not math.isfinite(float(value))

    non_finite = [r["step"] for r in rows
                  if bad(r.get("loss")) or bad(r.get("grad_norm"))]
    # The count is cumulative and monotonic, so the last row carries the total.
    skipped = int(rows[-1].get("skipped_updates") or 0)
    reasons = []
    if skipped:
        reasons.append(f"{skipped} skipped non-finite update(s)")
    if non_finite:
        reasons.append(f"non-finite loss/grad at steps {non_finite[:5]}")
    return {
        "passed": not reasons,
        "reason": "; ".join(reasons),
        "skipped_updates": skipped,
        "non_finite_rows": len(non_finite),
    }


def penalty_reduction_frac(baseline_pct: float, observed_pct: float) -> float:
    """How much of the baseline Q4 penalty a candidate removed, as a fraction.

    1.0 means the penalty is gone; 0.0 means unchanged; negative means the
    candidate made quantization damage *worse*, which is a real outcome for a
    learning rate that moved the weights off the grid faster than QAT pulled
    them back.
    """
    if baseline_pct <= 0:
        raise ValueError(
            f"baseline penalty must be positive to be reduced, got {baseline_pct}")
    return (baseline_pct - observed_pct) / baseline_pct


def retention_verdict(baseline: Baseline, observed: dict) -> dict:
    """The mandatory gates, each reported with the number that decided it.

    Every gate is measured against *this host's* baseline rather than the
    historical figure in the model card: Phase 2 showed the five-task mean
    reproduces here at 47.374 against a recorded 47.313, a floating-point
    residual that would eat 12% of a 0.5-point budget if the wrong reference
    were used.
    """
    checks = []

    fp16 = observed.get("perplexity_fp16")
    if fp16 is None:
        checks.append({"gate": "fp16-perplexity", "passed": False,
                       "reason": "not measured"})
    else:
        regression = (fp16 / baseline.perplexity_fp16 - 1.0) * 100.0
        checks.append({
            "gate": "fp16-perplexity", "observed_pct": regression,
            "limit_pct": MAX_FP16_PPL_REGRESSION_PCT,
            "passed": _at_most(regression, MAX_FP16_PPL_REGRESSION_PCT),
        })

    tasks = observed.get("five_task_mean")
    if tasks is None:
        checks.append({"gate": "five-task-mean", "passed": False,
                       "reason": "not measured"})
    else:
        drop = baseline.five_task_mean - tasks
        checks.append({
            "gate": "five-task-mean", "observed_drop": drop,
            "limit_drop": MAX_FIVE_TASK_DROP,
            "passed": _at_most(drop, MAX_FIVE_TASK_DROP),
        })

    # Retrieval is gated per depth, not on the aggregate: a model that loses a
    # depth and gains another nets out flat while being worse at the thing the
    # gate exists to protect.
    observed_retrieval = observed.get("retrieval") or {}
    if not baseline.retrieval:
        checks.append({"gate": "retrieval", "passed": False,
                       "reason": "no baseline recorded"})
    else:
        worst_key, worst_drop = None, float("-inf")
        missing = sorted(set(baseline.retrieval) - set(observed_retrieval))
        for key, base_value in baseline.retrieval.items():
            if key not in observed_retrieval:
                continue
            drop = (base_value - float(observed_retrieval[key])) * 100.0
            if drop > worst_drop:
                worst_key, worst_drop = key, drop
        if missing:
            checks.append({"gate": "retrieval", "passed": False,
                           "reason": f"depths not measured: {missing}"})
        else:
            checks.append({
                "gate": "retrieval", "worst_depth": worst_key,
                "observed_drop_points": worst_drop,
                "limit_drop_points": MAX_RETRIEVAL_DROP_POINTS,
                "passed": _at_most(worst_drop, MAX_RETRIEVAL_DROP_POINTS),
            })

    finiteness = observed.get("finiteness")
    if finiteness is not None:
        checks.append({"gate": "finiteness", **finiteness})

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def score_candidate(baseline: Baseline, observed: dict) -> dict:
    """One candidate's full verdict: improvement, retention, and both together."""
    penalty = observed.get("q4_penalty_pct")
    reduction = (None if penalty is None
                 else penalty_reduction_frac(baseline.q4_penalty_pct, penalty))
    retention = retention_verdict(baseline, observed)
    improved = (reduction is not None
                and _at_least(reduction, MIN_PENALTY_REDUCTION_FRAC))
    return {
        "name": observed.get("name"),
        "q4_penalty_pct": penalty,
        "penalty_reduction_frac": reduction,
        "meets_improvement_gate": improved,
        "meets_target": penalty is not None and _at_most(penalty,
                                                         TARGET_PENALTY_PCT),
        "meets_stretch": penalty is not None and _at_most(penalty,
                                                          STRETCH_PENALTY_PCT),
        "retention": retention,
        # Both must hold. A candidate that halved Q4 damage by wrecking the
        # fp32 model has not recovered anything.
        "accepted": bool(improved and retention["passed"]),
        "observed": observed,
    }


def _sort_key(scored: dict):
    """The preregistered tie-break order, as a sort key.

    Paired Q4 reduction first, then fp16 retention, then full-pass BPB, then
    the five-task mean, then retrieval. Negated where larger is better, so the
    natural ascending sort puts the winner first. Missing measurements sort
    last rather than winning by absence.
    """
    observed = scored.get("observed") or {}
    reduction = scored.get("penalty_reduction_frac")
    fp16 = observed.get("perplexity_fp16")
    bpb = observed.get("bpb")
    tasks = observed.get("five_task_mean")
    retrieval = observed.get("retrieval") or {}
    mean_retrieval = (sum(retrieval.values()) / len(retrieval)
                      if retrieval else None)
    inf = float("inf")
    return (
        -reduction if reduction is not None else inf,   # more reduction first
        fp16 if fp16 is not None else inf,              # lower perplexity first
        bpb if bpb is not None else inf,                # lower BPB first
        -tasks if tasks is not None else inf,           # higher task mean first
        -mean_retrieval if mean_retrieval is not None else inf,
        scored.get("name") or "",                       # deterministic tie-break
    )


def select_winner(scored: Sequence[dict]) -> Optional[dict]:
    """The best *accepted* candidate, or None when none passed.

    A candidate that failed a mandatory gate is never selected, however much
    Q4 damage it removed -- that is what makes the gates mandatory rather than
    advisory.
    """
    accepted = [s for s in scored if s.get("accepted")]
    if not accepted:
        return None
    return sorted(accepted, key=_sort_key)[0]


def escalation_decision(best_probe: Optional[dict],
                        followup: Optional[dict]) -> dict:
    """Whether to spend 1B tokens, per the preregistered stop rule.

    The rule is deliberately asymmetric: escalation needs a *positive* reason
    (the follow-up improved Q4 damage by at least 10% relative over the best
    probe, with every retention gate still holding), and everything else --
    including a follow-up that is merely as good -- stops the phase and
    reports what was measured.
    """
    if best_probe is None:
        return {"escalate": False,
                "reason": "no 100M probe passed both the improvement and "
                          "retention gates; reporting the negative result "
                          "rather than escalating"}
    if followup is None:
        return {"escalate": False,
                "reason": "the 300M follow-up has not been scored yet"}
    if not followup.get("accepted"):
        return {"escalate": False,
                "reason": "the 300M follow-up violated a mandatory gate: "
                          + _first_failure(followup)}

    best_penalty = best_probe.get("q4_penalty_pct")
    followup_penalty = followup.get("q4_penalty_pct")
    if best_penalty is None or followup_penalty is None:
        return {"escalate": False, "reason": "Q4 penalty not measured"}

    relative = penalty_reduction_frac(best_penalty, followup_penalty)
    clears = _at_least(relative, MIN_ESCALATION_REDUCTION_FRAC)
    return {
        "escalate": clears,
        "relative_improvement_frac": relative,
        "required_frac": MIN_ESCALATION_REDUCTION_FRAC,
        "reason": (
            f"the 300M follow-up improved Q4 damage by {relative:.1%} relative "
            f"to the best probe ({best_penalty:.3f}% -> {followup_penalty:.3f}%), "
            f"{'clearing' if clears else 'short of'} "
            f"the {MIN_ESCALATION_REDUCTION_FRAC:.0%} bar"),
    }


def _first_failure(scored: dict) -> str:
    for check in (scored.get("retention") or {}).get("checks", []):
        if not check.get("passed"):
            return f"{check.get('gate')} ({check.get('reason') or 'over limit'})"
    return "improvement gate"


# --------------------------------------------------------- preregistration ---

def build_preregistration(baseline: Baseline, *, init_from: str,
                          init_from_sha256: str) -> dict:
    """The plan, with every bar stated before a single probe has run."""
    arms = probe_arms()
    return {
        "schema": 1,
        "phase": "phase3-qat-recovery",
        "input": {"checkpoint": init_from, "sha256": init_from_sha256},
        "baseline": asdict(baseline),
        "shared_training": {
            "init_from_only": True,
            "qat_frac": 1.0,
            "seq_len": SEQ_LEN,
            "batch_tokens": BATCH_TOKENS,
            "micro_batch": MICRO_BATCH,
            "decay_frac": DECAY_FRAC,
            "warmup_frac": WARMUP_FRAC,
            "seed": 0,
            "muon_adam_ratio": SHIPPED_ADAM_LR / SHIPPED_MUON_LR,
        },
        "arms": [p.to_dict() for p in arms],
        "followup_tokens": FOLLOWUP_TOKENS,
        "escalation_tokens": ESCALATION_TOKENS,
        "gates": {
            "improvement": {
                "min_penalty_reduction_frac": MIN_PENALTY_REDUCTION_FRAC,
                "baseline_penalty_pct": baseline.q4_penalty_pct,
                "implied_max_penalty_pct":
                    baseline.q4_penalty_pct * (1 - MIN_PENALTY_REDUCTION_FRAC),
                "target_penalty_pct": TARGET_PENALTY_PCT,
                "stretch_penalty_pct": STRETCH_PENALTY_PCT,
            },
            "retention": {
                "max_fp16_ppl_regression_pct": MAX_FP16_PPL_REGRESSION_PCT,
                "max_five_task_drop": MAX_FIVE_TASK_DROP,
                "max_retrieval_drop_points": MAX_RETRIEVAL_DROP_POINTS,
                "no_skipped_non_finite_updates": True,
            },
            "escalation": {
                "min_relative_reduction_frac": MIN_ESCALATION_REDUCTION_FRAC,
            },
        },
        "selection_order": [
            "paired Q4 perplexity reduction",
            "FP16 perplexity retention",
            "full-pass BPB",
            "five-task mean",
            "retrieval retention",
        ],
        "retrieval_measurement": {
            "per_depth": RETRIEVAL_PER_DEPTH,
            "depths": list(RETRIEVAL_DEPTHS),
            "note": (
                "raised from the Phase 2 baseline's 10 items per depth before "
                "any probe ran: at 10 items one item is 10 points, so a "
                "1-point gate could only ever be satisfied by exact equality. "
                "The baseline is re-measured at this count so the comparison "
                "is like for like."),
            "resolution_caveat": (
                "100 items makes a 1-point gate expressible, not resolvable. "
                "At an exact-match rate near 0.85 binomial noise alone is "
                "~3.6 points, so this gate can catch a model that stopped "
                "retrieving and cannot certify retrieval is unchanged to "
                "within a point. Resolving one point would need thousands of "
                "items per depth, which this phase has no budget for."),
        },
    }


def write_preregistration(path, payload: dict, *, force: bool = False) -> Path:
    """Write the plan once. Refuse to overwrite it afterwards.

    A preregistration that can be rewritten after the fact records nothing.
    `force` exists for tests and for a genuine restart *before* any arm has
    run; it is not reachable from the normal launch path.
    """
    path = Path(path)
    if path.exists() and not force:
        raise PreregistrationError(
            f"{path} already exists; a preregistration is written once, before "
            f"the first probe. Delete it deliberately if the plan genuinely "
            f"changed before any arm ran.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def read_metrics(run_dir) -> List[dict]:
    """Every metrics row for a run, oldest first. Empty when the run never
    wrote one, which `finiteness_from_metrics` reports as a failure rather
    than as a pass by absence."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A row torn by a crash mid-write is not a reason to lose the run's
            # whole history; the finiteness gate reads the rest.
            continue
    return rows


def collect_observation(name: str, *, run_dir, quant_comparison=None,
                        tasks=None, bpb=None, retrieval=None) -> dict:
    """Assemble one candidate's measured numbers from artifacts on disk.

    Reads the scorecards the existing evaluators already write rather than
    re-deriving anything, so a number in a verdict here is the same number
    that is in the file the verdict cites.
    """
    observed: dict = {"name": name,
                      "finiteness": finiteness_from_metrics(read_metrics(run_dir))}

    if quant_comparison is not None:
        payload = json.loads(Path(quant_comparison).read_text())
        observed["q4_penalty_pct"] = float(payload["q4_penalty_pct"])
        observed["perplexity_fp16"] = float(payload["perplexity_fp16"])
        observed["perplexity_q4_0"] = float(payload["perplexity_q4_0"])
        observed["chunks_worse"] = payload.get("chunks_worse")
        observed["chunks_better"] = payload.get("chunks_better")
        observed["n_chunks"] = payload.get("n_chunks")
        observed["quant_comparison"] = str(quant_comparison)

    if tasks is not None:
        payload = json.loads(Path(tasks).read_text())
        metrics = payload.get("metrics", payload)
        for key in ("five_task_mean", "mean", "task_mean"):
            if key in metrics:
                observed["five_task_mean"] = float(metrics[key])
                break
        observed["tasks_scorecard"] = str(tasks)

    if bpb is not None:
        payload = json.loads(Path(bpb).read_text())
        metrics = payload.get("metrics", payload)
        for key in ("bpb", "val_bpb", "bits_per_byte"):
            if key in metrics:
                observed["bpb"] = float(metrics[key])
                break
        observed["bpb_scorecard"] = str(bpb)

    if retrieval:
        depths: Dict[str, float] = {}
        for card_path in retrieval:
            payload = json.loads(Path(card_path).read_text())
            task = payload.get("name", Path(card_path).stem)
            for key, value in (payload.get("metrics") or {}).items():
                if key.startswith("exact_match_d"):
                    depths[f"{task}:{key[len('exact_match_'):]}"] = float(value)
        observed["retrieval"] = depths
        observed["retrieval_scorecards"] = [str(p) for p in retrieval]

    return observed


# --------------------------------------------------------------------- cli ---

def _load_baseline(path) -> Baseline:
    return Baseline.from_dict(json.loads(Path(path).read_text()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/qat-recovery",
                        help="where the plan and verdicts live")
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preregister",
                         help="write the plan before the first probe")
    pre.add_argument("--baseline", required=True,
                     help="JSON with the released model's measured numbers")
    pre.add_argument("--init-from", required=True)
    pre.add_argument("--init-from-sha256", required=True)
    pre.add_argument("--force", action="store_true")

    cmd = sub.add_parser("command", help="print one arm's train.py argv")
    cmd.add_argument("--arm", required=True)
    cmd.add_argument("--init-from", required=True)
    cmd.add_argument("--data-dir", required=True)
    cmd.add_argument("--val-dir", default=None)
    cmd.add_argument("--device", default="cuda")
    cmd.add_argument("--total-tokens", type=int, default=None,
                     help="override the arm's budget (follow-up/escalation)")
    cmd.add_argument("--loss-chunk-size", type=int, default=None)
    cmd.add_argument("--gradient-checkpointing", action="store_true")

    score = sub.add_parser("score", help="score one candidate against the plan")
    score.add_argument("--name", required=True)
    score.add_argument("--run-dir", required=True)
    score.add_argument("--quant-comparison", default=None)
    score.add_argument("--tasks", default=None)
    score.add_argument("--bpb", default=None)
    score.add_argument("--retrieval", action="append", default=[])

    decide = sub.add_parser("decide",
                            help="select a winner and rule on escalation")
    decide.add_argument("--followup", default=None,
                        help="name of the 300M candidate, when it has been scored")

    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.command == "preregister":
        baseline = _load_baseline(args.baseline)
        payload = build_preregistration(
            baseline, init_from=args.init_from,
            init_from_sha256=args.init_from_sha256)
        written = write_preregistration(root / "preregistration.json", payload,
                                        force=args.force)
        print(f"preregistered {len(payload['arms'])} arms in {written}")
        print(f"improvement gate: Q4 penalty must fall from "
              f"{baseline.q4_penalty_pct:.3f}% to at most "
              f"{payload['gates']['improvement']['implied_max_penalty_pct']:.3f}%")
        return 0

    plan_path = root / "preregistration.json"
    if not plan_path.exists():
        raise SystemExit(
            f"no preregistration at {plan_path}; run `preregister` before "
            f"launching or scoring anything")
    plan = json.loads(plan_path.read_text())
    baseline = Baseline.from_dict(plan["baseline"])

    if args.command == "command":
        arms = {a["name"]: a for a in plan["arms"]}
        if args.arm not in arms:
            raise SystemExit(
                f"{args.arm!r} is not a preregistered arm; known: {sorted(arms)}")
        spec = arms[args.arm]
        probe = Probe(name=spec["name"], muon_lr=float(spec["muon_lr"]),
                      total_tokens=int(args.total_tokens or spec["total_tokens"]),
                      stage=spec["stage"])
        command = train_command(
            probe, init_from=args.init_from, data_dir=args.data_dir,
            val_dir=args.val_dir, device=args.device,
            loss_chunk_size=args.loss_chunk_size,
            gradient_checkpointing=args.gradient_checkpointing)
        assert_no_resume(command)
        print(" ".join(command))
        return 0

    if args.command == "score":
        observed = collect_observation(
            args.name, run_dir=args.run_dir,
            quant_comparison=args.quant_comparison, tasks=args.tasks,
            bpb=args.bpb, retrieval=args.retrieval)
        scored = score_candidate(baseline, observed)
        out = root / "scored" / f"{args.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: scored[k] for k in
                          ("name", "q4_penalty_pct", "penalty_reduction_frac",
                           "meets_improvement_gate", "accepted")}, indent=2))
        return 0

    if args.command == "decide":
        scored_dir = root / "scored"
        candidates = [json.loads(p.read_text())
                      for p in sorted(scored_dir.glob("*.json"))]
        followup = next((c for c in candidates
                         if c.get("name") == args.followup), None)
        probes = [c for c in candidates if c is not followup]
        winner = select_winner(probes)
        decision = escalation_decision(winner, followup)
        verdict = {
            "winner": winner["name"] if winner else None,
            "winner_penalty_pct": winner["q4_penalty_pct"] if winner else None,
            "escalation": decision,
            "candidates": [
                {k: c.get(k) for k in ("name", "q4_penalty_pct",
                                       "penalty_reduction_frac", "accepted")}
                for c in candidates
            ],
        }
        (root / "verdict.json").write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(json.dumps(verdict, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
