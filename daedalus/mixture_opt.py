"""Phase 7 steps 7 and 8: which mixture weights, and decided by what rule?

The plan asks for candidate mixture weights *derived* from tiny-model per-source
excess loss, compared against at least a baseline and a quality-heavy arm under
equal compute, and selected "by aggregate BPB plus domain floors, not by
aggregate loss alone". This module is the whole decision half of that: the arms,
the derivation rule, the floors, and the selection. It trains nothing and reads
no checkpoints, so every threshold in it can be -- and is -- committed before the
first arm finishes.

That ordering is the point. `scripts/mixture_opt.py` spends GPU hours; this file
spends none, and the plan's standing rule is that thresholds are not tuned after
seeing outcomes. A rule that lands in the same commit as the numbers it judges is
indistinguishable from one fitted to them.

**Excess loss is measured against a specialist, not against a prior.** For each
source `s`, `excess(s) = bpb_baseline(s) - bpb_specialist_s(s)`, where the
specialist is the same tiny model under the same budget and schedule trained on
`s` alone. The specialist is what this architecture, at this scale, on this
budget, can do with that source; the gap is the part the mixture is leaving
unclaimed. Every other reading of "excess" needs a quantity nothing on this box
measures -- a source's intrinsic entropy, or a reference model trained on data
this program does not have.

A negative excess is kept as measured rather than clipped to zero. It is a real
outcome: a specialist that has to re-read a short source can lose to the
mixture-trained model on that source's own holdout, and the honest consequence is
that the rule down-weights it. `EXCESS_RATIO_CAP` bounds how far in either
direction one such number can move the mixture.

**The floors are stated as fractions of the blueprint, not as absolutes.** "No
floored domain may keep less than `DOMAIN_FLOOR_FRACTION` of the share the
blueprint gave it" is a rule that means the same thing over the full ten-source
corpus and over the three sources this box holds, and it needs no invented
constants. The floored domains are exactly the plan's: the general-web backbone,
math, and code.

**A floor over a subset of the corpus is a weaker guarantee than the same floor
over all of it, and `unrepresented_floored_domains` says so.** On this box the
math sources are not on disk at all, so the math floor is vacuous here. That is
reported rather than silently satisfied -- a selection artifact claiming three
floors held when one had no source to hold is describing a corpus that was not
measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

#: Which domain each blueprint source belongs to. Written out rather than
#: inferred from the dataset name: `finepdfs-edu` and `fineweb-edu` share a
#: prefix with `finephrase` and `finemath-3plus` and belong to three different
#: domains, so any rule cheap enough to be a heuristic gets at least one wrong.
DOMAIN_OF_SOURCE: Dict[str, str] = {
    "fineweb-edu": "web-edu",
    "finepdfs-edu": "web-edu",
    "dclm-baseline": "web-raw",
    "finephrase": "web-raw",
    "finemath-3plus": "math",
    "infiwebmath-3plus": "math",
    "stack-edu-python": "code",
    "cosmopedia-v2": "synthetic",
    "finewiki-en": "reference",
    "everyday-conversations": "dialogue",
}

#: The domains the plan puts a floor under: "Preserve a minimum general-web
#: backbone and explicit math/code floors."
#:
#: `web-raw` rather than all web is the deliberate reading. `fineweb-edu` is web
#: too, so flooring "web" as a whole would be satisfied by a mixture that is
#: entirely educational filtering -- which is the shift the floor exists to bound.
#: What a backbone buys is unfiltered distributional breadth, and only the raw
#: sources carry it.
FLOORED_DOMAINS = ("web-raw", "math", "code")

#: A floored domain may not fall below this fraction of its blueprint share.
#: 0.4 is a real constraint -- it forbids collapsing raw web, math or code to a
#: token presence -- while leaving room for a mixture that genuinely wants to
#: move 60% of a domain's mass elsewhere.
DOMAIN_FLOOR_FRACTION = 0.4

#: The quality-heavy arm scales every raw-web source to this fraction of its
#: blueprint share and redistributes what it frees across the filtered sources in
#: proportion to their blueprint shares.
#:
#: 0.45 rather than 0.4: constructing an arm that sits exactly on its own floor
#: makes its admissibility a question about float comparison rather than about
#: the mixture. Above the floor by a margin, the arm is admissible by
#: construction and the floors stay a check on the *derived* arm, which is the
#: one no one has seen yet.
QUALITY_HEAVY_RAW_SCALE = 0.45

#: Bits per byte of excess loss that doubles a source's weight, in
#: `exp(excess / T)`. Held-out BPB at this scale sits around 1.0-1.5, and a
#: per-source gap of 0.10 bits/byte between a mixture-trained model and a
#: specialist is a large one -- so T = 0.10 makes a large gap worth a 2.7x ask
#: before the cap trims it, and a 0.01 gap worth 1.1x.
EXCESS_TEMPERATURE = 0.10

#: The most, and least, the rule may scale one source's blueprint share by
#: before floors and renormalization. A single specialist arm that diverged,
#: got a short holdout, or simply ran unlucky can produce one wild excess
#: figure; without a cap that one number writes the mixture.
#:
#: The cost of the cap is that a share it sets is no longer a measured optimum,
#: and nothing downstream can tell the two apart from the number alone --
#: `cap_saturation` is what says which is which.
EXCESS_RATIO_CAP = 2.0

#: A candidate may not regress any single source's held-out BPB by more than
#: this, relative to the baseline arm, however good its aggregate is. This is
#: the "not by aggregate loss alone" half of the plan's selection rule that the
#: floors cannot express: floors constrain the *weights*, and a mixture can
#: satisfy every floor while the model it produces has fallen off a cliff on one
#: source.
MAX_SOURCE_REGRESSION = 0.05

#: How much better than the baseline arm a candidate's aggregate BPB must be
#: before the mixture is changed at all. Below this the verdict is
#: `keep-baseline`: a proxy that cannot separate the arms is evidence about the
#: arms, and the plan's own instruction is to record a negative result rather
#: than to advance the best of a tie.
MIN_AGGREGATE_GAIN = 0.005

_TOL = 1e-9


# ------------------------------------------------------------------- weights ---

def blueprint_shares() -> Dict[str, float]:
    """`dataprep.MIXTURE`'s shares, keyed by source.

    Imported at call time rather than at module import: this module is the
    decision layer and is imported by tests that have no reason to pull in the
    dataprep dependency stack.
    """
    from daedalus.dataprep import MIXTURE

    return {spec.key: float(spec.share) for spec in MIXTURE}


def domain_of(key: str) -> str:
    """The domain a source belongs to, or a refusal naming it.

    Unknown sources raise rather than defaulting to an "other" bucket. A source
    silently placed outside every floored domain is a source the floors do not
    constrain, and finding that out from a selection artifact is finding it out
    too late.
    """
    try:
        return DOMAIN_OF_SOURCE[key]
    except KeyError:
        raise KeyError(
            f"source {key!r} has no domain; add it to DOMAIN_OF_SOURCE so the "
            f"floors know whether it is part of the general-web backbone, of "
            f"math, or of code") from None


def normalized(weights: Mapping[str, float]) -> Dict[str, float]:
    """Shares that sum to 1, refusing the inputs that cannot be made to."""
    negative = sorted(name for name, value in weights.items() if value < 0)
    if negative:
        raise ValueError(f"negative mixture share(s) for {negative}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("mixture shares sum to zero; nothing would be sampled")
    return {name: value / total for name, value in sorted(weights.items())}


def restrict(weights: Mapping[str, float],
             sources: Sequence[str]) -> Dict[str, float]:
    """`weights` over `sources` only, renormalized.

    The same reduction `train.resolve_mixture` performs against what is on disk,
    done here so an arm's *stated* weights and its *sampled* weights are the
    same numbers rather than two computations that agree until one changes.
    """
    wanted = list(dict.fromkeys(sources))
    unknown = [name for name in wanted if name not in weights]
    if unknown:
        raise ValueError(
            f"{unknown} are not in the weights being restricted "
            f"(have {sorted(weights)})")
    return normalized({name: weights[name] for name in wanted})


def domain_shares(weights: Mapping[str, float]) -> Dict[str, float]:
    """Total share per domain, over the domains `weights` actually names."""
    shares: Dict[str, float] = {}
    for name, value in weights.items():
        shares[domain_of(name)] = shares.get(domain_of(name), 0.0) + float(value)
    return shares


def sources_by_domain(sources: Sequence[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for name in sorted(sources):
        grouped.setdefault(domain_of(name), []).append(name)
    return grouped


def domain_floors(baseline: Mapping[str, float],
                  fraction: float = DOMAIN_FLOOR_FRACTION) -> Dict[str, float]:
    """The floor each floored domain has under `baseline`.

    Stated against the baseline the arms are actually compared to, so a floor
    over the three sources this box holds and a floor over the full corpus are
    the same rule -- "keep at least `fraction` of what the blueprint gave this
    domain" -- rather than the same constant meaning two different things.

    A floored domain with no source in `baseline` gets no floor, because there
    is nothing for a floor to bind. `unrepresented_floored_domains` is what says
    so out loud.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"floor fraction must be in [0, 1], got {fraction}")
    shares = domain_shares(baseline)
    return {domain: shares[domain] * fraction
            for domain in FLOORED_DOMAINS
            if shares.get(domain, 0.0) > 0.0}


def unrepresented_floored_domains(sources: Sequence[str]) -> List[str]:
    """Floored domains with no source under this root.

    On this box that is `math`: the corpus subset fetched here carries web,
    educational web and Python, so the math floor has nothing to constrain. A
    report that lists three floors as held when one had no source is describing a
    corpus that was not measured, so this is carried into the artifact rather
    than left to be noticed.
    """
    present = set(sources_by_domain(sources))
    return [domain for domain in FLOORED_DOMAINS if domain not in present]


def floor_violations(weights: Mapping[str, float],
                     floors: Mapping[str, float]) -> List[dict]:
    """Every floored domain `weights` puts below its floor, worst first."""
    shares = domain_shares(weights)
    rows = [{"domain": domain, "share": shares.get(domain, 0.0), "floor": floor,
             "shortfall": floor - shares.get(domain, 0.0)}
            for domain, floor in floors.items()
            if shares.get(domain, 0.0) < floor - _TOL]
    return sorted(rows, key=lambda row: (-row["shortfall"], row["domain"]))


def apply_floors(weights: Mapping[str, float],
                 floors: Mapping[str, float]) -> Dict[str, float]:
    """`weights` with every floored domain raised to its floor, renormalized.

    Water-filling, and it has to iterate for the same reason
    `train.cap_weights_by_epochs` does: taking mass off the unfloored sources to
    lift one domain can push a second domain below *its* floor, which only a
    re-checking loop catches. Bounded by the number of floors, since each pass
    locks at least one domain at its floor and a locked domain is never revisited.

    A domain at zero share that has sources and a floor is lifted by spreading
    the floor equally across them -- the alternative, scaling by
    `floor / 0`, has no answer, and dropping the floor silently is the failure
    this module exists to avoid.
    """
    current = dict(normalized(weights))
    if sum(floors.values()) > 1.0 + _TOL:
        raise ValueError(
            f"floors sum to {sum(floors.values()):.4f}, which no mixture can "
            f"satisfy; floors are {dict(sorted(floors.items()))}")

    grouped = sources_by_domain(list(current))
    locked: Dict[str, float] = {}
    for _ in range(len(floors) + 1):
        shares = domain_shares(current)
        deficient = [domain for domain, floor in floors.items()
                     if domain in grouped and domain not in locked
                     and shares.get(domain, 0.0) < floor - _TOL]
        if not deficient:
            return {name: current[name] for name in sorted(current)}
        for domain in deficient:
            floor, share = floors[domain], shares.get(domain, 0.0)
            members = grouped[domain]
            if share <= 0.0:
                for name in members:
                    current[name] = floor / len(members)
            else:
                for name in members:
                    current[name] *= floor / share
            locked[domain] = floor

        free = [name for name in current if domain_of(name) not in locked]
        free_mass = sum(current[name] for name in free)
        remaining = 1.0 - sum(locked.values())
        if remaining < -_TOL:
            raise ValueError(
                f"floors {dict(sorted(locked.items()))} already exceed the "
                f"whole mixture; nothing is left for {sorted(free)}")
        if not free:
            # Every source is inside a floored domain, so the floors *are* the
            # mixture; renormalizing them is the only consistent answer.
            return normalized(current)
        if free_mass <= 0.0:
            for name in free:
                current[name] = remaining / len(free)
        else:
            for name in free:
                current[name] *= remaining / free_mass
    raise RuntimeError(
        "floor water-filling did not settle; this should be impossible since "
        "each pass locks a domain")


def quality_heavy(baseline: Mapping[str, float],
                  raw_scale: float = QUALITY_HEAVY_RAW_SCALE
                  ) -> Dict[str, float]:
    """The preregistered quality-heavy arm: less raw web, more filtered text.

    Constructed from the baseline rather than typed out as three numbers, so the
    arm means the same thing over whatever sources a root happens to hold, and
    so "quality-heavy" has a definition -- scale the unfiltered general-web
    backbone down, redistribute what that frees over the filtered sources in
    proportion to their blueprint shares -- instead of being whatever weights
    someone chose on the day.
    """
    baseline = normalized(baseline)
    raw = [name for name in baseline if domain_of(name) == "web-raw"]
    rest = [name for name in baseline if name not in raw]
    if not raw:
        raise ValueError(
            "no raw-web source in this mixture, so there is nothing for a "
            "quality-heavy arm to move mass away from")
    if not rest:
        raise ValueError(
            "every source in this mixture is raw web, so there is nothing for "
            "a quality-heavy arm to move mass toward")
    freed = sum(baseline[name] for name in raw) * (1.0 - raw_scale)
    rest_mass = sum(baseline[name] for name in rest)
    out = {name: baseline[name] * raw_scale for name in raw}
    out.update({name: baseline[name] + freed * baseline[name] / rest_mass
                for name in rest})
    return normalized(out)


# -------------------------------------------------------------- excess loss ---

def excess_loss(baseline_bpb: Mapping[str, float],
                specialist_bpb: Mapping[str, float]) -> Dict[str, float]:
    """Per-source `bpb_baseline - bpb_specialist`, in bits per byte.

    Both arguments are per-source held-out BPB measured on the *same* holdout:
    `baseline_bpb` from the arm trained on the baseline mixture, and
    `specialist_bpb[s]` from the arm trained on `s` alone.

    A source present in `baseline_bpb` and missing from `specialist_bpb` raises.
    Treating it as zero excess would give it a score of exactly 1.0 and leave its
    weight at the blueprint's -- a mixture that looks derived and is, for that
    source, the mixture it started from, with nothing in the artifact to say a
    specialist arm had failed.
    """
    missing = sorted(set(baseline_bpb) - set(specialist_bpb))
    if missing:
        raise ValueError(
            f"no specialist BPB for {missing}; excess loss is defined against a "
            f"specialist and a source without one has no measured excess")
    for name, value in list(baseline_bpb.items()) + list(specialist_bpb.items()):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite BPB {value!r} for source {name!r}")
    return {name: float(baseline_bpb[name]) - float(specialist_bpb[name])
            for name in sorted(baseline_bpb)}


def _uncapped_ratio(value: float, temperature: float) -> float:
    """`exp(excess / T)`, with an overflow answered rather than raised.

    The cap exists for "a single specialist arm that diverged", and a diverged
    arm is exactly the input that reaches this: its holdout BPB stays finite --
    so `excess_loss` admits it -- while being large enough that `exp(excess/T)`
    is not representable. `math.exp` raises `OverflowError` there, which would
    take down the derivation at the one input the cap was written to absorb, and
    do it from inside a call that already knows the answer is "more than the cap
    allows". Infinity is that answer, and the clip below turns it into the cap.
    """
    try:
        return math.exp(float(value) / temperature)
    except OverflowError:
        return math.inf


def excess_scores(excess: Mapping[str, float],
                  temperature: float = EXCESS_TEMPERATURE,
                  ratio_cap: float = EXCESS_RATIO_CAP) -> Dict[str, float]:
    """`exp(excess / T)`, clipped to `[1/cap, cap]`."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if ratio_cap < 1.0:
        raise ValueError(f"ratio cap must be at least 1, got {ratio_cap}")
    scores = {}
    for name, value in sorted(excess.items()):
        raw = _uncapped_ratio(value, temperature)
        scores[name] = min(ratio_cap, max(1.0 / ratio_cap, raw))
    return scores


def cap_saturation(excess: Mapping[str, float],
                   temperature: float = EXCESS_TEMPERATURE,
                   ratio_cap: float = EXCESS_RATIO_CAP) -> Dict[str, dict]:
    """The sources whose score the *cap* set, rather than the measurement.

    A saturated source's derived share is not a measured optimum. It is the
    largest (or smallest) share the rule was willing to grant it, and the
    measurement only says the ask was somewhere past that bound -- 2.1x and 84x
    both arrive as 2.0x, and the artifact that records `2.0` cannot tell them
    apart afterwards. On this box `stack-edu-python` saturates: its 0.213
    bits/byte of headroom asks for 8.4x its blueprint share, and the derived
    arm's 0.178 code share is what the cap allowed, which is a different claim
    from "0.178 is the code share the evidence picked".

    Disclosure, deliberately, and not a rule. Every threshold in this module was
    committed before the first arm trained, and the plan's standing instruction
    is that thresholds are not tuned after seeing outcomes -- so a saturated
    source does not become inadmissible, get a wider cap, or change the
    selection. It gets said out loud in the artifact that reports the share.

    Only saturated sources are returned, so an empty mapping means every share
    the rule produced came from inside the cap.
    """
    scores = excess_scores(excess, temperature=temperature, ratio_cap=ratio_cap)
    saturated: Dict[str, dict] = {}
    for name, value in sorted(excess.items()):
        raw = _uncapped_ratio(value, temperature)
        if raw >= ratio_cap:
            bound = "upper"
        elif raw <= 1.0 / ratio_cap:
            bound = "lower"
        else:
            continue
        saturated[name] = {"bound": bound, "excess": float(value),
                           "score": scores[name], "uncapped_ratio": raw,
                           "ratio_cap": float(ratio_cap),
                           "temperature": float(temperature)}
    return saturated


def derive_weights(baseline: Mapping[str, float],
                   excess: Mapping[str, float],
                   *,
                   temperature: float = EXCESS_TEMPERATURE,
                   ratio_cap: float = EXCESS_RATIO_CAP,
                   floors: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    """The derived mixture: blueprint shares tilted by excess loss, then floored.

    Every source in `baseline` must have a measured excess. A derivation that
    quietly scores an unmeasured source at 1.0 produces a mixture that is partly
    derived and partly inherited while describing itself as derived.
    """
    baseline = normalized(baseline)
    missing = sorted(set(baseline) - set(excess))
    if missing:
        raise ValueError(
            f"no excess loss for {missing}; every source in the mixture needs "
            f"one before the derived arm can be built")
    scores = excess_scores(excess, temperature=temperature, ratio_cap=ratio_cap)
    tilted = normalized({name: baseline[name] * scores[name] for name in baseline})
    if floors is None:
        floors = domain_floors(baseline)
    return apply_floors(tilted, floors)


# --------------------------------------------------------------------- arms ---

@dataclass(frozen=True)
class MixtureArm:
    """One arm: a name, the weights it trains on, and where they came from."""

    name: str
    weights: Dict[str, float]
    basis: str
    #: The baseline arm is the point every other arm is read against, and is
    #: shared between the reference and candidate stages rather than trained
    #: twice.
    is_baseline: bool = False

    def describe(self) -> dict:
        return {
            "arm": self.name,
            "basis": self.basis,
            "is_baseline": self.is_baseline,
            "weights": {name: round(value, 6)
                        for name, value in sorted(self.weights.items())},
            "domain_shares": {domain: round(value, 6) for domain, value
                              in sorted(domain_shares(self.weights).items())},
        }


def specialist_name(source: str) -> str:
    return f"only-{source}"


def baseline_arm(sources: Sequence[str]) -> MixtureArm:
    """The blueprint mixture restricted to what is on disk.

    This is also the evaluation weighting every arm is scored under -- see
    `evaluation_weights`. The two being the same set of numbers is deliberate:
    the question the phase asks is which *training* mixture produces the best
    model over the corpus the program intends to serve, and the corpus it intends
    to serve is the blueprint.
    """
    return MixtureArm(name="baseline", weights=restrict(blueprint_shares(), sources),
                      basis="dataprep.MIXTURE shares, renormalized over the "
                            "sources present under the data root",
                      is_baseline=True)


def evaluation_weights(sources: Sequence[str]) -> Dict[str, float]:
    """The fixed weighting every arm's aggregate BPB is computed under.

    Fixed, and emphatically not each arm's own training weights. `train.py`'s
    in-run `val_bpb` weights the holdout by the mixture that run samples --
    correct for a single run, which wants held-out BPB under its own training
    distribution, and useless across arms, because it would score six models on
    six different corpora and then subtract the numbers. The arms are compared
    through `scripts/bpb_eval.py` under these weights instead.
    """
    return restrict(blueprint_shares(), sources)


def reference_arms(sources: Sequence[str]) -> List[MixtureArm]:
    """The baseline, then one specialist per source.

    Baseline first for the reason phase 5 and phase 6 put their controls first:
    a specialist's BPB means nothing until the point it is measured against
    exists, so a sweep the deadline cuts short should lose specialists, not the
    arm they are all read against.

    A specialist is expressed as a weight of 1.0 on its source and 0.0 on the
    rest rather than as a different `--data-dir`, so every arm in the phase
    shares one data root, one holdout, and one set of shard files. Pointing arms
    at different roots would make "identical except for the mixture" a property
    of two paths matching rather than of one number differing.
    """
    sources = list(dict.fromkeys(sources))
    arms = [baseline_arm(sources)]
    for source in sources:
        weights = {name: (1.0 if name == source else 0.0) for name in sources}
        arms.append(MixtureArm(
            name=specialist_name(source), weights=weights,
            basis=f"all mass on {source}: the achievable-BPB bound that "
                  f"per-source excess loss is measured against"))
    return arms


def candidate_arms(sources: Sequence[str],
                   derived: Optional[Mapping[str, float]] = None
                   ) -> List[MixtureArm]:
    """The baseline, the quality-heavy arm, and -- once measured -- the derived one.

    `derived` is None until the reference stage has produced per-source excess
    loss. The arm is omitted rather than stubbed with the blueprint, so a
    candidate sweep run too early trains two arms instead of silently training
    the baseline twice under two names.
    """
    baseline = baseline_arm(list(dict.fromkeys(sources)))
    arms = [baseline, MixtureArm(
        name="quality-heavy", weights=quality_heavy(baseline.weights),
        basis=f"raw-web sources scaled to {QUALITY_HEAVY_RAW_SCALE:g} of their "
              f"blueprint share, the freed mass redistributed over the filtered "
              f"sources in proportion to theirs")]
    if derived is not None:
        arms.append(MixtureArm(
            name="derived", weights=normalized(derived),
            basis=f"blueprint shares tilted by exp(excess/{EXCESS_TEMPERATURE:g}), "
                  f"clipped to {EXCESS_RATIO_CAP:g}x, then floored at "
                  f"{DOMAIN_FLOOR_FRACTION:g} of each floored domain's blueprint "
                  f"share"))
    return arms


# ---------------------------------------------------------------- selection ---

@dataclass
class ArmResult:
    """One scored arm, as the selection rule reads it."""

    name: str
    weights: Dict[str, float]
    aggregate_bpb: float
    per_source_bpb: Dict[str, float] = field(default_factory=dict)


def select_mixture(results: Sequence[ArmResult],
                   floors: Mapping[str, float],
                   *,
                   baseline_name: str = "baseline",
                   max_source_regression: float = MAX_SOURCE_REGRESSION,
                   min_gain: float = MIN_AGGREGATE_GAIN) -> dict:
    """Which mixture the evidence selects, and the refusals behind that.

    Three preregistered rules, in order:

    1. an arm whose *weights* put a floored domain under its floor is
       inadmissible, whatever its BPB;
    2. an arm whose *model* regresses any single source's held-out BPB by more
       than `max_source_regression` relative to the baseline is inadmissible --
       the floors constrain the mixture, and this constrains what the mixture
       produced, which is not the same thing;
    3. among what is left, the lowest aggregate BPB wins, and only if it beats
       the baseline by at least `min_gain` relative. Otherwise the verdict is
       `keep-baseline`.

    Refusals are returned rather than filtered out. An artifact that lists the
    winner and drops the arms that were excluded cannot be read back to check
    whether the rule or the data did the excluding.
    """
    by_name = {result.name: result for result in results}
    if baseline_name not in by_name:
        raise ValueError(
            f"no {baseline_name!r} arm among {sorted(by_name)}; every rule here "
            f"is stated relative to the baseline")
    baseline = by_name[baseline_name]

    refusals: List[dict] = []
    admissible: List[ArmResult] = []
    for result in results:
        if result.name == baseline_name:
            continue
        violations = floor_violations(result.weights, floors)
        if violations:
            refusals.append({"arm": result.name, "reason": "floor",
                             "detail": violations})
            continue
        regressions = []
        for source, value in sorted(result.per_source_bpb.items()):
            reference = baseline.per_source_bpb.get(source)
            if reference is None or reference <= 0:
                continue
            relative = (value - reference) / reference
            if relative > max_source_regression + _TOL:
                regressions.append({"source": source, "bpb": value,
                                    "baseline_bpb": reference,
                                    "relative_regression": relative})
        if regressions:
            refusals.append({"arm": result.name, "reason": "source-regression",
                             "detail": regressions})
            continue
        admissible.append(result)

    ranked = sorted(admissible, key=lambda r: (r.aggregate_bpb, r.name))
    verdict = {
        "baseline": baseline_name,
        "baseline_aggregate_bpb": baseline.aggregate_bpb,
        "floors": {domain: floors[domain] for domain in sorted(floors)},
        "max_source_regression": max_source_regression,
        "min_aggregate_gain": min_gain,
        "admissible": [r.name for r in ranked],
        "refusals": refusals,
        "ranking": [{"arm": r.name, "aggregate_bpb": r.aggregate_bpb,
                     "relative_gain": ((baseline.aggregate_bpb - r.aggregate_bpb)
                                       / baseline.aggregate_bpb)
                     if baseline.aggregate_bpb else 0.0}
                    for r in ranked],
    }
    if not ranked:
        return {**verdict, "verdict": "keep-baseline", "selected": baseline_name,
                "reason": "no candidate arm was admissible"}
    best = ranked[0]
    gain = ((baseline.aggregate_bpb - best.aggregate_bpb) / baseline.aggregate_bpb
            if baseline.aggregate_bpb else 0.0)
    if gain < min_gain:
        return {**verdict, "verdict": "keep-baseline", "selected": baseline_name,
                "reason": f"best admissible arm {best.name!r} gains "
                          f"{gain:.4%} of aggregate BPB, under the "
                          f"preregistered {min_gain:.4%}"}
    return {**verdict, "verdict": "adopt", "selected": best.name,
            "selected_weights": {name: round(value, 6)
                                 for name, value in sorted(best.weights.items())},
            "relative_gain": gain}
