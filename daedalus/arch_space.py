"""Phase 6: the space of architecture candidates, and what can be known for free.

Depth, attention count, KV heads and FFN width are four knobs with a large
product, and training is the expensive way to eliminate a candidate. Most of
them can be eliminated without it: a shape that stock llama.cpp will not convert,
or that blows the KV budget, or that is not parameter-matched to the control, is
not a worse model -- it is not a comparison at all. This module is that filter,
and it is arithmetic rather than GPU time.

Three decisions are worth stating, because each is a way this kind of sweep
usually goes wrong.

**Candidates are parameter-matched by construction, not by hand.** `block_ff_dim`
is *solved* for the target rather than gridded over, so every candidate lands
within a rounding step of the control. The shipped `daedalus-150m-deep` preset is
the counter-example this exists for: at 24x640 it carries `block_ff_dim=1792` and
totals 148.2M against the control's 160.5M, so it is 7.7% smaller. A depth
comparison run against it measures depth *and* 12M missing parameters, and the
plan says in as many words not to use it as proof. `depth_matched_candidate`
solves the same shape to 2048 and 160.0M.

**KV bytes per context token is the constraint that binds, and it is analytic.**
`2 (K and V) * kv_heads * head_dim * attention_layers * 2 bytes` -- no training
required, and no candidate that fails it can be rescued by a good loss curve.
The shipped model sits at exactly 6,144, which is the plan's ceiling, so every
candidate that improves on long-context cost has to *cut* attention layers or KV
heads rather than trade them for something else.

**The successor's scale is recorded, and the proxy's is not it.** The operator
fixed a ~500M-parameter successor after this phase was written for 150M, and KV
cost does not scale with parameters -- it scales with attention layers times KV
heads, which a deeper 500M model has more of. So the space is generated at both:
the analytic screen runs at whichever target is asked for, and a quality ranking
measured on 150M proxies is a ranking at 150M. `SUCCESSOR_PARAMS` is here so that
distinction is a named constant rather than a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from daedalus.config import DaedalusConfig, PRESETS


#: The shipped model, and the control every stage of this phase runs against.
CONTROL_PRESET = "daedalus-150m"

#: The operator's successor target, recorded 2026-08-25. Analytic screening runs
#: here; proxy *training* does not, and a ranking measured at the proxy scale is
#: a ranking at that scale.
SUCCESSOR_PARAMS = 500_000_000

#: llama.cpp keeps the KV cache in f16 by default, and stores both K and V.
KV_BYTES_PER_ELEMENT = 2

#: The plan's ceiling and its preferred value, in bytes per context token. The
#: shipped model is at the ceiling exactly, which is why it is a ceiling.
MAX_KV_BYTES_PER_CONTEXT_TOKEN = 6144
PREFERRED_KV_BYTES_PER_CONTEXT_TOKEN = 4096

#: k-quants operate on 256-element rows, so a tensor dimension that is not a
#: multiple of 256 falls back to a different type or refuses outright.
QUANT_BLOCK = 256

#: SwiGLU inner width as a multiple of `hidden_size`. The shipped model is
#: 2048/768 = 2.67x, which is also Llama's ratio, and the band brackets it wide
#: enough to keep depth 12 through 36 reachable at every width in the grid.
#:
#: This is a validity rule, not a preference. `block_ff_dim` is *solved* for the
#: parameter target, so a shape with almost no attention absorbs its whole
#: budget into the FFN: at the 500M target, hidden 512 with one attention layer
#: solves to 25,088, a 49x aspect ratio. That is not an architecture that lost a
#: comparison, it is the solver reporting that the shape cannot hold 500M
#: parameters any other way, and carrying it forward would fill the Pareto set
#: with candidates nobody would train.
MIN_FF_RATIO = 1.5
MAX_FF_RATIO = 5.0

#: A candidate is parameter-matched when it lands within this of the target.
#: One `block_ff_dim` step is `3 * hidden * layers * 256` parameters, which at
#: the shipped shape is 1.1% of the model -- so the bound cannot be tighter than
#: a rounding step without emptying the space.
PARAM_MATCH_TOLERANCE = 0.02


@dataclass(frozen=True)
class ArchCandidate:
    """One stock-LFM2-compatible shape, parameter-matched to a target."""

    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_blocks: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    block_ff_dim: int
    vocab_size: int = 49152
    max_position_embeddings: int = 2048

    def config(self, **overrides) -> DaedalusConfig:
        """The `DaedalusConfig` this candidate names.

        Built through the shipped dataclass rather than a parallel one, so a
        candidate that this module accepts is a candidate `train.py` and
        `export.py` can already take, and `__post_init__`'s own assertions run.
        """
        fields = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_blocks": self.num_attention_blocks,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "block_ff_dim": self.block_ff_dim,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
        }
        fields.update(overrides)
        return DaedalusConfig(**fields)


def candidate_from_config(name: str, cfg: DaedalusConfig) -> ArchCandidate:
    """Read an existing preset back as a candidate, so the control competes.

    `num_attention_blocks` is the *requested* count and `layer_types` is what
    the interleaver produced; they can differ, and the realised count is the one
    every downstream number depends on. Taken from `n_attn_layers` for that
    reason.
    """
    return ArchCandidate(
        name=name,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_blocks=cfg.n_attn_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        block_ff_dim=cfg.block_ff_dim,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
    )


# ------------------------------------------------------------- analytic cost ---

def kv_bytes_per_context_token(cfg: DaedalusConfig,
                               bytes_per_element: int = KV_BYTES_PER_ELEMENT
                               ) -> int:
    """Bytes of KV cache one context token costs, across the whole stack.

    Both K and V, over every *attention* layer -- conv layers have no cache, and
    counting them is how a hybrid's long-context advantage gets erased on paper.
    Read from `layer_types` rather than from `num_attention_blocks`, because the
    interleaver is what decides how many attention layers there actually are.
    """
    return (2 * cfg.num_key_value_heads * cfg.head_dim
            * cfg.n_attn_layers * bytes_per_element)


def kv_bytes_at_context(cfg: DaedalusConfig, context: int) -> int:
    return kv_bytes_per_context_token(cfg) * context


def attention_fraction(cfg: DaedalusConfig) -> float:
    return cfg.n_attn_layers / cfg.num_hidden_layers


# ---------------------------------------------------------------- validation ---

def validation_failures(candidate: ArchCandidate, *,
                        target_params: Optional[int] = None,
                        tolerance: float = PARAM_MATCH_TOLERANCE) -> List[str]:
    """Every rule this candidate breaks, named. Empty means it is comparable.

    Returned as a list rather than raised one at a time so a rejected candidate
    can be reported with all of its reasons; a sweep that surfaces one failure,
    gets it fixed, and then surfaces the next costs a round trip per rule.
    """
    failures = []

    if candidate.num_attention_heads % candidate.num_key_value_heads:
        failures.append(
            f"num_attention_heads {candidate.num_attention_heads} is not "
            f"divisible by num_key_value_heads {candidate.num_key_value_heads}")

    # LFM2 projects q/o against `hidden_size`. Letting heads x head_dim differ
    # from it is representable here and is an export risk with no upside, so it
    # is refused rather than discovered by a failed conversion after training.
    projected = candidate.num_attention_heads * candidate.head_dim
    if projected != candidate.hidden_size:
        failures.append(
            f"num_attention_heads x head_dim = {projected} does not equal "
            f"hidden_size {candidate.hidden_size}")

    # `block_ff_dim` and `vocab_size` are the two `DaedalusConfig.__post_init__`
    # asserts on, so a candidate breaking either cannot be built at all.
    # `hidden_size` is deliberately *not* here: 640 is not a multiple of 256 and
    # both `daedalus-150m-deep` and `dense-150m` ship at that width, and the
    # plan asks for the 24x640 depth comparison by name. An unaligned hidden
    # size costs a k-quant fallback on the tensors whose rows it sizes, which is
    # a Pareto column (`kquant_aligned_hidden` in `describe`) rather than a
    # reason the candidate is not a comparison.
    for field in ("block_ff_dim", "vocab_size"):
        value = getattr(candidate, field)
        if value % QUANT_BLOCK:
            failures.append(
                f"{field} {value} is not a multiple of {QUANT_BLOCK}, so "
                f"k-quants cannot use their {QUANT_BLOCK}-element blocks")

    if not 0 < candidate.num_attention_blocks <= candidate.num_hidden_layers:
        failures.append(
            f"num_attention_blocks {candidate.num_attention_blocks} is not "
            f"within 1..{candidate.num_hidden_layers}")

    if failures:                    # a shape this broken cannot be instantiated
        return failures

    # Below here the shape builds, so every remaining rule reports rather than
    # short-circuits: a candidate outside the FFN band is usually also off the
    # parameter target, and surfacing one reason at a time costs a round trip
    # per rule.
    ratio = candidate.block_ff_dim / candidate.hidden_size
    if not MIN_FF_RATIO <= ratio <= MAX_FF_RATIO:
        failures.append(
            f"block_ff_dim {candidate.block_ff_dim} is {ratio:.2f}x "
            f"hidden_size, outside the {MIN_FF_RATIO:g}-{MAX_FF_RATIO:g}x band "
            f"(the shipped model is 2.67x)")

    cfg = candidate.config()
    kv = kv_bytes_per_context_token(cfg)
    if kv > MAX_KV_BYTES_PER_CONTEXT_TOKEN:
        failures.append(
            f"KV cache is {kv:,} bytes per context token, over the "
            f"{MAX_KV_BYTES_PER_CONTEXT_TOKEN:,} ceiling")

    if target_params is not None:
        total = cfg.param_count()["total"]
        drift = abs(total - target_params) / target_params
        if drift > tolerance:
            failures.append(
                f"{total:,} parameters is {100 * drift:.1f}% from the "
                f"{target_params:,} target, over the {100 * tolerance:.0f}% "
                f"match tolerance")
    return failures


def is_comparable(candidate: ArchCandidate, **kw) -> bool:
    return not validation_failures(candidate, **kw)


# ------------------------------------------------------- parameter matching ----

def solve_ff_dim(*, hidden_size: int, num_hidden_layers: int,
                 num_attention_blocks: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int, target_params: int,
                 vocab_size: int = 49152, multiple: int = QUANT_BLOCK) -> int:
    """The `block_ff_dim` that puts this shape closest to `target_params`.

    Solved rather than searched: parameter count is affine in `block_ff_dim`
    (`3 * hidden * layers` per unit), so two evaluations give the intercept and
    the slope and the answer is one division. Rounded to `multiple` because a
    dimension that is not a multiple of 256 fails the quantization rule above,
    which would make the solution unusable.
    """

    def total(ff: int) -> int:
        return ArchCandidate(
            name="probe", hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_blocks=num_attention_blocks,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads, head_dim=head_dim,
            block_ff_dim=ff, vocab_size=vocab_size).config().param_count()["total"]

    base = total(multiple)
    step = total(2 * multiple) - base
    if step <= 0:                                    # not reachable for LFM2
        raise ValueError("parameter count does not grow with block_ff_dim")
    units = round((target_params - base) / step) + 1
    return max(multiple, units * multiple)


def matched_candidate(name: str, *, hidden_size: int, num_hidden_layers: int,
                      num_attention_blocks: int, num_key_value_heads: int,
                      head_dim: int = 64, target_params: int,
                      vocab_size: int = 49152) -> ArchCandidate:
    """A candidate at this shape, its FFN solved for the parameter target.

    `num_attention_heads` follows from `hidden_size / head_dim` rather than
    being a free knob, because the validation rule above requires their product
    to be `hidden_size` -- so it is derived here instead of being supplied and
    then rejected.
    """
    if hidden_size % head_dim:
        raise ValueError(
            f"hidden_size {hidden_size} is not divisible by head_dim {head_dim}")
    heads = hidden_size // head_dim
    ff = solve_ff_dim(
        hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
        num_attention_blocks=num_attention_blocks, num_attention_heads=heads,
        num_key_value_heads=num_key_value_heads, head_dim=head_dim,
        target_params=target_params, vocab_size=vocab_size)
    return ArchCandidate(
        name=name, hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_blocks=num_attention_blocks, num_attention_heads=heads,
        num_key_value_heads=num_key_value_heads, head_dim=head_dim,
        block_ff_dim=ff, vocab_size=vocab_size)


def depth_matched_candidate(target_params: Optional[int] = None
                            ) -> ArchCandidate:
    """The corrected 24x640 depth comparison the plan asks for by name.

    The shipped `daedalus-150m-deep` preset is this shape at `block_ff_dim=1792`
    and 148.2M parameters against the control's 160.5M. Running depth against it
    measures depth and a 7.7% parameter deficit together, and the plan says not
    to use it as proof. Solved here instead.
    """
    control = PRESETS[CONTROL_PRESET].param_count()["total"]
    return matched_candidate(
        "deep-24x640", hidden_size=640, num_hidden_layers=24,
        num_attention_blocks=8, num_key_value_heads=2,
        target_params=target_params or control)


# ---------------------------------------------------------------- the space ----

#: The knobs, and why each range stops where it does.
#:
#: `hidden_size` multiples of 256 that divide by 64, spanning the control both
#: ways -- narrower needs more depth to hold the parameters, wider needs less,
#: and both directions move KV cost.
#: `attention_fraction` from a third (the shipped ratio) down to a twelfth: the
#: cache is the whole reason to run a hybrid, and every candidate that improves
#: on 6,144 bytes does it here.
#: `num_key_value_heads` powers of two only, because GQA replicates KV heads
#: across query heads and a non-divisor is rejected by the rule above anyway.
DEPTHS = (12, 18, 24, 30, 36)
HIDDEN_SIZES = (512, 640, 768, 896, 1024, 1280, 1536)
ATTENTION_FRACTIONS = (1 / 3, 1 / 4, 1 / 6, 1 / 9, 1 / 12)
KV_HEADS = (1, 2, 4, 8)


def _attention_blocks(depth: int, fraction: float) -> int:
    return max(1, round(depth * fraction))


def generate(*, target_params: int,
             depths: Sequence[int] = DEPTHS,
             hidden_sizes: Sequence[int] = HIDDEN_SIZES,
             attention_fractions: Sequence[float] = ATTENTION_FRACTIONS,
             kv_heads: Sequence[int] = KV_HEADS,
             tolerance: float = PARAM_MATCH_TOLERANCE) -> List[ArchCandidate]:
    """Every comparable candidate in the grid, parameter-matched to the target.

    Only the ones that survive `validation_failures` are returned: an
    incomparable shape is not a candidate that lost, and carrying it forward as
    one is how a sweep reports having explored a space it never could.
    """
    seen: Dict[tuple, ArchCandidate] = {}
    for hidden in hidden_sizes:
        if hidden % 64:
            continue
        for depth in depths:
            for fraction in attention_fractions:
                blocks = _attention_blocks(depth, fraction)
                for kv in kv_heads:
                    if (hidden // 64) % kv:
                        continue
                    try:
                        candidate = matched_candidate(
                            f"d{depth}x{hidden}-a{blocks}-kv{kv}",
                            hidden_size=hidden, num_hidden_layers=depth,
                            num_attention_blocks=blocks,
                            num_key_value_heads=kv,
                            target_params=target_params)
                    except (ValueError, AssertionError):
                        continue
                    if not is_comparable(candidate, target_params=target_params,
                                         tolerance=tolerance):
                        continue
                    key = (candidate.hidden_size, candidate.num_hidden_layers,
                           candidate.num_attention_blocks,
                           candidate.num_key_value_heads,
                           candidate.block_ff_dim)
                    seen.setdefault(key, candidate)
    return sorted(seen.values(), key=lambda c: c.name)


def describe(candidate: ArchCandidate,
             target_params: Optional[int] = None) -> dict:
    """Everything knowable about a candidate without training it."""
    cfg = candidate.config()
    counts = cfg.param_count()
    kv = kv_bytes_per_context_token(cfg)
    record = {
        "name": candidate.name,
        "hidden_size": candidate.hidden_size,
        "num_hidden_layers": candidate.num_hidden_layers,
        "attention_layers": cfg.n_attn_layers,
        "attention_fraction": attention_fraction(cfg),
        "num_key_value_heads": candidate.num_key_value_heads,
        "num_attention_heads": candidate.num_attention_heads,
        "head_dim": candidate.head_dim,
        "block_ff_dim": candidate.block_ff_dim,
        "parameters": counts["total"],
        "non_embedding": counts["non_embedding"],
        "embedding_frac": counts["embedding_frac"],
        "q4_0_MB": counts["q4_0_MB"],
        "kv_bytes_per_context_token": kv,
        "kv_MB_at_2048": kv_bytes_at_context(cfg, 2048) / 1e6,
        "kv_within_ceiling": kv <= MAX_KV_BYTES_PER_CONTEXT_TOKEN,
        "kv_at_or_under_preferred": kv <= PREFERRED_KV_BYTES_PER_CONTEXT_TOKEN,
        # Rows of `token_embd` and the attention projections are `hidden_size`
        # long, so an unaligned width makes llama.cpp fall back off k-quants for
        # them. Not a rejection -- the shipped 640-wide presets would fail that
        # -- but it belongs beside the artifact size it changes.
        "kquant_aligned_hidden": candidate.hidden_size % QUANT_BLOCK == 0,
        "ff_ratio": candidate.block_ff_dim / candidate.hidden_size,
        "layer_types": list(cfg.layer_types),
    }
    if target_params is not None:
        record["param_drift_pct"] = 100.0 * (
            counts["total"] - target_params) / target_params
    return record


def control_candidate() -> ArchCandidate:
    return candidate_from_config(CONTROL_PRESET, PRESETS[CONTROL_PRESET])


def analytic_screen(*, target_params: int,
                    tolerance: float = PARAM_MATCH_TOLERANCE) -> dict:
    """The whole free half of phase 6: the space, the control, and the cuts.

    Quality is not in here and cannot be. What this decides is which candidates
    are *worth* a GPU, which is a different question and the one that can be
    answered before spending one.
    """
    control = control_candidate()
    candidates = generate(target_params=target_params, tolerance=tolerance)
    described = [describe(candidate, target_params) for candidate in candidates]
    control_record = describe(control, target_params)
    improves = [record for record in described
                if record["kv_bytes_per_context_token"]
                < control_record["kv_bytes_per_context_token"]]
    # The control is the shipped 160M model at every target, because the KV
    # ceiling the plan gates on is *its* 6,144 bytes and that reference does not
    # move with parameter count. Its parameter column does, though, so a screen
    # at the successor scale says outright that the control is not
    # parameter-matched to it -- otherwise the control row reads as a candidate
    # that came in 68% under budget.
    control_drift = abs(control_record["parameters"] - target_params) / target_params
    control_note = None
    if control_drift > tolerance:
        control_note = (
            f"the control is the shipped {control_record['parameters']:,}-"
            f"parameter model and is not matched to the {target_params:,} "
            f"target; it is the KV and layout reference, and its parameter, "
            f"artifact-size and quality columns are not comparable here")

    return {
        "target_params": target_params,
        "tolerance": tolerance,
        "control": control_record,
        "control_is_parameter_matched": control_drift <= tolerance,
        "control_note": control_note,
        "candidates": described,
        "counts": {
            "generated": len(described),
            "kv_under_control": len(improves),
            "kv_at_or_under_preferred": sum(
                1 for record in described
                if record["kv_at_or_under_preferred"]),
        },
        "grid": {
            "depths": list(DEPTHS),
            "hidden_sizes": list(HIDDEN_SIZES),
            "attention_fractions": list(ATTENTION_FRACTIONS),
            "kv_heads": list(KV_HEADS),
        },
    }
