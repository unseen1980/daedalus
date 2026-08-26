"""Tests for daedalus/mixture_opt.py and scripts/mixture_opt.py.

The module under test is the *preregistration* for phase 7 steps 7 and 8: the
arms, the derivation rule, the floors and the selection all land before a single
arm has been scored. So what these assert is not that the numbers come out well
-- nobody knows yet -- but that the rule says what it claims to say, that it
refuses the inputs it cannot honestly act on, and that an arm cannot quietly
train on a mixture other than its own.

Run: python -m pytest tests/test_mixture_opt.py -v
"""
import json
from pathlib import Path

import pytest

from daedalus.data import ShardWriter
from daedalus.mixture_opt import (DOMAIN_FLOOR_FRACTION, EXCESS_RATIO_CAP,
                                  EXCESS_TEMPERATURE, FLOORED_DOMAINS,
                                  MAX_SOURCE_REGRESSION, MIN_AGGREGATE_GAIN,
                                  QUALITY_HEAVY_RAW_SCALE, ArmResult,
                                  apply_floors, baseline_arm, blueprint_shares,
                                  candidate_arms, derive_weights, domain_floors,
                                  domain_of, domain_shares, evaluation_weights,
                                  excess_loss, excess_scores, floor_violations,
                                  normalized, quality_heavy, reference_arms,
                                  restrict, select_mixture, specialist_name,
                                  unrepresented_floored_domains)
from scripts.mixture_opt import (PROBE, arm_checkpoint_path, arm_preflight,
                                 arm_run_name, build_report, derive,
                                 discover_sources, finished_run, foreign_run,
                                 render_markdown, run_arm, score_arm,
                                 scorecard_name, scoring_sources, stage_arms,
                                 sweep, train_command, verdict_path,
                                 weight_args, weighted_bpb)


#: The three sources this box actually holds under `data/shards-train`.
BOX_SOURCES = ["dclm-baseline", "fineweb-edu", "stack-edu-python"]


def _write_source(root, name, n_tokens, shard_tokens=200):
    """One source's shard dir, as `dataprep.run_dataprep` would leave it."""
    out_dir = root / name
    writer = ShardWriter(str(out_dir), shard_tokens=shard_tokens)
    # Ids cycle rather than counting up: shards are uint16, and these roots are
    # sized in hundreds of thousands of tokens so that no arm comes near the
    # epoch cap.
    writer.write([index % 4096 for index in range(n_tokens)])
    writer.close()
    writer.write_manifest({"eos_id": 0, "source_key": name})
    return out_dir


#: A smoke budget of two probe steps, and a root large enough that no arm --
#: including a specialist putting all of its mass on one source -- comes near
#: the epoch cap. Sized from the arm that reads the most: 2 x batch_tokens at a
#: share of 1.0.
SMOKE_TOKENS = 2 * 131_072


def _mixture_root(tmp_path, sizes=None):
    root = tmp_path / "shards-train"
    default = {name: SMOKE_TOKENS for name in BOX_SOURCES}
    for name, tokens in (sizes or default).items():
        _write_source(root, name, n_tokens=tokens, shard_tokens=tokens // 4 or 1)
    return root


# ------------------------------------------------------------------ weights ---

def test_the_baseline_is_the_blueprint_restricted_to_what_is_on_disk():
    """Not a fourth set of hand-typed shares. The baseline is the mixture the
    program intends to serve, and the ratios between the sources that survive
    the restriction have to be the blueprint's."""
    blueprint = blueprint_shares()
    baseline = baseline_arm(BOX_SOURCES).weights
    assert sum(baseline.values()) == pytest.approx(1.0)
    assert set(baseline) == set(BOX_SOURCES)
    assert (baseline["fineweb-edu"] / baseline["dclm-baseline"]
            == pytest.approx(blueprint["fineweb-edu"] / blueprint["dclm-baseline"]))


def test_the_evaluation_weighting_is_the_baseline_and_is_the_same_for_every_arm():
    """`train.py`'s in-run val_bpb weights the holdout by the mixture each run
    samples, so six arms' val_bpb columns are six models scored on six corpora.
    The comparison has to be one weighting, and the one that answers the phase's
    question is the corpus the program intends to serve."""
    assert evaluation_weights(BOX_SOURCES) == pytest.approx(
        baseline_arm(BOX_SOURCES).weights)


def test_restricting_to_a_source_that_is_not_in_the_weights_is_refused():
    with pytest.raises(ValueError, match="finemath-3plus"):
        restrict(blueprint_shares(), BOX_SOURCES + ["not-a-source"])
    with pytest.raises(ValueError):
        restrict({"fineweb-edu": 1.0}, ["finemath-3plus"])


def test_an_unknown_source_has_no_domain_and_says_so():
    """Defaulting to an 'other' bucket would put the source outside every floor,
    and the artifact would report three floors held over a mixture one of them
    never touched."""
    with pytest.raises(KeyError, match="DOMAIN_OF_SOURCE"):
        domain_of("some-new-corpus")


def test_normalized_refuses_a_mixture_that_cannot_be_one():
    with pytest.raises(ValueError, match="negative"):
        normalized({"fineweb-edu": -0.1, "dclm-baseline": 1.1})
    with pytest.raises(ValueError, match="sum to zero"):
        normalized({"fineweb-edu": 0.0, "dclm-baseline": 0.0})


# ------------------------------------------------------------------- floors ---

def test_the_floored_domains_are_the_plans_three():
    assert set(FLOORED_DOMAINS) == {"web-raw", "math", "code"}


def test_floors_are_a_fraction_of_the_blueprint_rather_than_an_invented_constant():
    """The same rule over three sources and over ten: keep at least
    DOMAIN_FLOOR_FRACTION of what the blueprint gave this domain."""
    baseline = baseline_arm(BOX_SOURCES).weights
    floors = domain_floors(baseline)
    shares = domain_shares(baseline)
    assert floors["web-raw"] == pytest.approx(shares["web-raw"] * DOMAIN_FLOOR_FRACTION)
    assert floors["code"] == pytest.approx(shares["code"] * DOMAIN_FLOOR_FRACTION)
    # No math source on this box, so nothing for a math floor to bind.
    assert "math" not in floors


def test_a_floored_domain_with_no_source_here_is_named_rather_than_assumed_held():
    assert unrepresented_floored_domains(BOX_SOURCES) == ["math"]
    assert unrepresented_floored_domains(
        BOX_SOURCES + ["finemath-3plus"]) == []


def test_apply_floors_lifts_a_violated_domain_and_renormalizes():
    baseline = baseline_arm(BOX_SOURCES).weights
    floors = domain_floors(baseline)
    starved = normalized({"fineweb-edu": 0.95, "dclm-baseline": 0.03,
                          "stack-edu-python": 0.02})
    assert floor_violations(starved, floors)

    fixed = apply_floors(starved, floors)
    assert sum(fixed.values()) == pytest.approx(1.0)
    assert floor_violations(fixed, floors) == []
    # The floors bind exactly, and the rest of the mixture keeps its ordering.
    assert domain_shares(fixed)["web-raw"] == pytest.approx(floors["web-raw"])
    assert domain_shares(fixed)["code"] == pytest.approx(floors["code"])
    assert fixed["fineweb-edu"] > fixed["dclm-baseline"]


def test_apply_floors_re_checks_after_redistributing():
    """Taking mass off the unfloored sources to lift one domain can push a
    second under its own floor. Only a re-checking loop catches it -- the same
    reason `cap_weights_by_epochs` water-fills over multiple rounds."""
    floors = {"web-raw": 0.30, "code": 0.30}
    start = normalized({"fineweb-edu": 0.90, "dclm-baseline": 0.09,
                        "stack-edu-python": 0.01})
    fixed = apply_floors(start, floors)
    assert domain_shares(fixed)["web-raw"] >= 0.30 - 1e-9
    assert domain_shares(fixed)["code"] >= 0.30 - 1e-9
    assert sum(fixed.values()) == pytest.approx(1.0)


def test_apply_floors_leaves_a_compliant_mixture_alone():
    baseline = baseline_arm(BOX_SOURCES).weights
    assert apply_floors(baseline, domain_floors(baseline)) == pytest.approx(baseline)


def test_floors_that_no_mixture_can_satisfy_are_refused():
    with pytest.raises(ValueError, match="no mixture can"):
        apply_floors(baseline_arm(BOX_SOURCES).weights,
                     {"web-raw": 0.7, "code": 0.7})


# ------------------------------------------------------------ quality-heavy ---

def test_quality_heavy_moves_raw_web_mass_onto_the_filtered_sources():
    baseline = baseline_arm(BOX_SOURCES).weights
    arm = quality_heavy(baseline)
    assert sum(arm.values()) == pytest.approx(1.0)
    assert arm["dclm-baseline"] == pytest.approx(
        baseline["dclm-baseline"] * QUALITY_HEAVY_RAW_SCALE)
    assert arm["fineweb-edu"] > baseline["fineweb-edu"]
    assert arm["stack-edu-python"] > baseline["stack-edu-python"]


def test_quality_heavy_is_admissible_by_construction():
    """Constructed above its own floor by a margin, so the floors stay a check
    on the derived arm -- the one nobody has seen -- rather than a question
    about how two nearly equal floats compare."""
    baseline = baseline_arm(BOX_SOURCES).weights
    assert QUALITY_HEAVY_RAW_SCALE > DOMAIN_FLOOR_FRACTION
    assert floor_violations(quality_heavy(baseline), domain_floors(baseline)) == []


def test_quality_heavy_needs_something_to_move_mass_between():
    with pytest.raises(ValueError, match="nothing for a quality-heavy arm"):
        quality_heavy({"fineweb-edu": 1.0})
    with pytest.raises(ValueError, match="nothing for a quality-heavy arm"):
        quality_heavy({"dclm-baseline": 1.0})


# -------------------------------------------------------------- excess loss ---

def test_excess_loss_is_the_gap_to_the_specialist():
    excess = excess_loss({"fineweb-edu": 1.20, "stack-edu-python": 0.90},
                         {"fineweb-edu": 1.10, "stack-edu-python": 0.95})
    assert excess["fineweb-edu"] == pytest.approx(0.10)
    # Kept as measured, not clipped: a specialist that had to re-read a short
    # source can lose to the mixture on that source's own holdout, and the
    # honest consequence is that the rule down-weights it.
    assert excess["stack-edu-python"] == pytest.approx(-0.05)


def test_a_source_without_a_specialist_has_no_measured_excess():
    """Scoring it 1.0 would leave its weight at the blueprint's, and the
    artifact would describe as derived a mixture that is, for that source, the
    one it started from."""
    with pytest.raises(ValueError, match="no specialist BPB"):
        excess_loss({"fineweb-edu": 1.2, "stack-edu-python": 0.9},
                    {"fineweb-edu": 1.1})


def test_a_non_finite_bpb_is_refused():
    with pytest.raises(ValueError, match="non-finite"):
        excess_loss({"fineweb-edu": float("nan")}, {"fineweb-edu": 1.1})


def test_excess_scores_are_bounded_in_both_directions():
    """One diverged specialist arm must not be allowed to write the mixture."""
    scores = excess_scores({"a": 10.0, "b": -10.0, "c": 0.0},
                           temperature=EXCESS_TEMPERATURE,
                           ratio_cap=EXCESS_RATIO_CAP)
    assert scores["a"] == pytest.approx(EXCESS_RATIO_CAP)
    assert scores["b"] == pytest.approx(1.0 / EXCESS_RATIO_CAP)
    assert scores["c"] == pytest.approx(1.0)


# ----------------------------------------------------------------- deriving ---

def test_derive_gives_more_weight_to_the_source_with_more_headroom():
    baseline = baseline_arm(BOX_SOURCES).weights
    derived = derive_weights(baseline, {"fineweb-edu": 0.0,
                                        "dclm-baseline": 0.0,
                                        "stack-edu-python": 0.08})
    assert sum(derived.values()) == pytest.approx(1.0)
    assert derived["stack-edu-python"] > baseline["stack-edu-python"]
    assert derived["fineweb-edu"] < baseline["fineweb-edu"]


def test_a_uniform_excess_leaves_the_blueprint_where_it_was():
    """The rule tilts on *differences* in headroom. If every source is equally
    far from its specialist, there is nothing to tilt toward."""
    baseline = baseline_arm(BOX_SOURCES).weights
    derived = derive_weights(baseline, {name: 0.05 for name in BOX_SOURCES})
    assert derived == pytest.approx(baseline)


def test_the_derived_mixture_is_floored():
    """The floors exist for exactly this arm: the one produced by a rule from
    numbers nobody has seen yet."""
    baseline = baseline_arm(BOX_SOURCES).weights
    derived = derive_weights(baseline, {"fineweb-edu": 5.0,
                                        "dclm-baseline": -5.0,
                                        "stack-edu-python": -5.0})
    floors = domain_floors(baseline)
    assert floor_violations(derived, floors) == []
    assert domain_shares(derived)["web-raw"] == pytest.approx(floors["web-raw"])


def test_deriving_without_every_sources_excess_is_refused():
    with pytest.raises(ValueError, match="no excess loss for"):
        derive_weights(baseline_arm(BOX_SOURCES).weights,
                       {"fineweb-edu": 0.1, "dclm-baseline": 0.1})


# --------------------------------------------------------------------- arms ---

def test_the_reference_stage_is_the_baseline_and_one_specialist_per_source():
    arms = reference_arms(BOX_SOURCES)
    assert [arm.name for arm in arms] == ["baseline"] + [
        specialist_name(name) for name in BOX_SOURCES]
    # Baseline first, for the reason phases 5 and 6 put their controls first: a
    # specialist's number means nothing until the point it is read against
    # exists, so a truncated sweep should lose specialists, not the baseline.
    assert arms[0].is_baseline
    assert sum(1 for arm in arms if arm.is_baseline) == 1


def test_a_specialist_is_one_hot_over_the_same_root_as_every_other_arm():
    """Not a different --data-dir. Arms pointed at different roots differ in
    shard files, packing and holdout as well as in mixture."""
    arm = {a.name: a for a in reference_arms(BOX_SOURCES)}["only-stack-edu-python"]
    assert set(arm.weights) == set(BOX_SOURCES)
    assert arm.weights["stack-edu-python"] == pytest.approx(1.0)
    assert all(value == pytest.approx(0.0) for name, value in arm.weights.items()
               if name != "stack-edu-python")


def test_every_arms_weights_sum_to_one():
    arms = reference_arms(BOX_SOURCES) + candidate_arms(
        BOX_SOURCES, {"fineweb-edu": 0.5, "dclm-baseline": 0.3,
                      "stack-edu-python": 0.2})
    for arm in arms:
        assert sum(arm.weights.values()) == pytest.approx(1.0), arm.name


def test_the_derived_arm_is_omitted_until_it_has_been_derived():
    """Stubbing it with the blueprint would train the baseline twice under two
    names and report one of them as derived."""
    names = [arm.name for arm in candidate_arms(BOX_SOURCES)]
    assert names == ["baseline", "quality-heavy"]
    assert "derived" in [arm.name for arm in candidate_arms(
        BOX_SOURCES, {"fineweb-edu": 0.4, "dclm-baseline": 0.35,
                      "stack-edu-python": 0.25})]


def test_the_reference_stage_refuses_derived_weights():
    with pytest.raises(SystemExit, match="reference stage"):
        stage_arms("reference", BOX_SOURCES, {"fineweb-edu": 1.0})


# ---------------------------------------------------------------- selection ---

def _result(name, weights, aggregate, per_source=None):
    return ArmResult(name=name, weights=weights, aggregate_bpb=aggregate,
                     per_source_bpb=per_source or {})


def _baseline_result(aggregate=1.00, per_source=None):
    return _result("baseline", baseline_arm(BOX_SOURCES).weights, aggregate,
                   per_source or {name: 1.00 for name in BOX_SOURCES})


def test_the_best_admissible_aggregate_wins_when_it_clears_the_bar():
    floors = domain_floors(baseline_arm(BOX_SOURCES).weights)
    verdict = select_mixture(
        [_baseline_result(),
         _result("quality-heavy", quality_heavy(baseline_arm(BOX_SOURCES).weights),
                 0.97, {name: 0.97 for name in BOX_SOURCES})],
        floors)
    assert verdict["verdict"] == "adopt"
    assert verdict["selected"] == "quality-heavy"
    assert verdict["relative_gain"] == pytest.approx(0.03)


def test_a_candidate_that_barely_wins_keeps_the_baseline():
    """A proxy that cannot separate the arms is evidence about the arms. The
    plan's instruction is to record the negative result, not to advance the best
    of a tie."""
    floors = domain_floors(baseline_arm(BOX_SOURCES).weights)
    verdict = select_mixture(
        [_baseline_result(1.00),
         _result("quality-heavy", quality_heavy(baseline_arm(BOX_SOURCES).weights),
                 1.00 - MIN_AGGREGATE_GAIN / 2,
                 {name: 0.999 for name in BOX_SOURCES})],
        floors)
    assert verdict["verdict"] == "keep-baseline"
    assert verdict["selected"] == "baseline"
    # Still ranked and reported: the arm was admissible, it just did not earn a
    # change of mixture.
    assert verdict["admissible"] == ["quality-heavy"]


def test_a_floor_violation_is_inadmissible_however_good_the_aggregate():
    floors = domain_floors(baseline_arm(BOX_SOURCES).weights)
    starved = normalized({"fineweb-edu": 0.97, "dclm-baseline": 0.02,
                          "stack-edu-python": 0.01})
    verdict = select_mixture(
        [_baseline_result(),
         _result("derived", starved, 0.50, {name: 0.50 for name in BOX_SOURCES})],
        floors)
    assert verdict["verdict"] == "keep-baseline"
    assert [row["arm"] for row in verdict["refusals"]] == ["derived"]
    assert verdict["refusals"][0]["reason"] == "floor"


def test_a_source_that_falls_off_a_cliff_is_inadmissible_even_inside_the_floors():
    """The half of the plan's rule the floors cannot express. Floors constrain
    the weights; this constrains what the weights produced."""
    baseline_weights = baseline_arm(BOX_SOURCES).weights
    floors = domain_floors(baseline_weights)
    verdict = select_mixture(
        [_baseline_result(1.00, {"dclm-baseline": 1.0, "fineweb-edu": 1.0,
                                 "stack-edu-python": 1.0}),
         _result("quality-heavy", quality_heavy(baseline_weights), 0.90,
                 {"dclm-baseline": 0.85, "fineweb-edu": 0.85,
                  "stack-edu-python": 1.0 + 2 * MAX_SOURCE_REGRESSION})],
        floors)
    assert verdict["verdict"] == "keep-baseline"
    assert verdict["refusals"][0]["reason"] == "source-regression"
    assert verdict["refusals"][0]["detail"][0]["source"] == "stack-edu-python"


def test_selection_without_a_baseline_arm_is_refused():
    with pytest.raises(ValueError, match="no 'baseline' arm"):
        select_mixture([_result("derived", baseline_arm(BOX_SOURCES).weights, 0.9)],
                       domain_floors(baseline_arm(BOX_SOURCES).weights))


# ======================================================= the sweep launcher ===

def test_every_arm_differs_only_in_its_mixture_weights():
    """The whole experiment. Two arms that differ in schedule, batch shape or
    budget are measuring something other than the mixture."""
    arms = reference_arms(BOX_SOURCES)
    commands = [train_command(arm, data_dir="d", run_name=arm_run_name(arm, "probe"))
                for arm in arms]

    def without_run_specific(command):
        out, skip = [], False
        for index, part in enumerate(command):
            if skip:
                skip = False
                continue
            if part in ("--mixture-weight", "--run-name"):
                skip = True
                continue
            out.append(part)
        return out

    assert all(without_run_specific(c) == without_run_specific(commands[0])
               for c in commands[1:])


def test_the_probe_is_phase_fours_recipe_at_the_shipped_vocabulary():
    """Re-used rather than re-derived, so the throughput, the memory headroom
    and the schedule shape are measured facts on this box."""
    from scripts import tokenizer_lab

    assert PROBE.config == "tok-probe-49152"
    assert PROBE.seq_len == tokenizer_lab.PROBE_SEQ_LEN
    assert PROBE.batch_tokens == tokenizer_lab.PROBE_BATCH_TOKENS
    assert PROBE.micro_batch == tokenizer_lab.PROBE_MICRO_BATCH
    assert PROBE.muon_lr == tokenizer_lab.PROBE_MUON_LR
    assert PROBE.adam_lr == tokenizer_lab.PROBE_ADAM_LR
    assert PROBE.warmup_steps == tokenizer_lab.PROBE_WARMUP_STEPS
    assert PROBE.decay_frac == tokenizer_lab.PROBE_DECAY_FRAC
    # A whole number of steps: a truncated final batch makes the step count a
    # rounding artefact rather than a property of the plan.
    assert PROBE.total_tokens % PROBE.batch_tokens == 0
    assert PROBE.steps == PROBE.total_tokens // PROBE.batch_tokens


def test_weight_arguments_are_stable_across_launches():
    """`finished_run` compares commands exactly, so an argv that varied with
    dict ordering would make a completed arm look like a new experiment and
    retrain it over its own checkpoint."""
    arm = baseline_arm(BOX_SOURCES)
    assert weight_args(arm) == weight_args(baseline_arm(list(reversed(BOX_SOURCES))))
    names = [part.split("=")[0] for part in weight_args(arm)
             if part != "--mixture-weight"]
    assert names == sorted(names)


def test_the_sources_are_discovered_from_the_root_not_read_off_the_blueprint(tmp_path):
    """Building arms from the blueprint would name seven sources that are not on
    this box, and a specialist for a source with no shards cannot run."""
    root = _mixture_root(tmp_path)
    assert discover_sources(root) == sorted(BOX_SOURCES)
    with pytest.raises(SystemExit, match="mixture root"):
        discover_sources(tmp_path / "empty" if (tmp_path / "empty").mkdir()
                         or True else tmp_path)


def test_an_arm_whose_mixture_would_be_reweighted_is_refused_before_the_gpu(tmp_path):
    """`cap_weights_by_epochs` clamps a source that cannot supply its share and
    water-fills the difference onto the others. Correct for a production run,
    fatal here: the arm would train on a mixture that is not the one it names."""
    root = _mixture_root(tmp_path, {"fineweb-edu": 400_000, "dclm-baseline": 400_000,
                                    "stack-edu-python": 1_000})
    with pytest.raises(SystemExit, match="points of L1 away"):
        arm_preflight(baseline_arm(BOX_SOURCES), data_dir=str(root),
                      total_tokens=100_000)


def test_an_arm_that_would_re_read_its_own_source_past_the_cap_is_refused(tmp_path):
    """The failure `capped_sources` cannot tell from the one above. In the
    all-capped regime the target mixture is *kept* and repetition is accepted,
    so there is no skew to see -- and an arm whose advantage could be a fourth
    pass over its own data is not evidence about mixtures."""
    root = _mixture_root(tmp_path, {name: 10_000 for name in BOX_SOURCES})
    arms = {arm.name: arm for arm in reference_arms(BOX_SOURCES)}
    with pytest.raises(SystemExit, match="epoch cap"):
        arm_preflight(arms["only-stack-edu-python"], data_dir=str(root),
                      total_tokens=100_000)


def test_a_well_stocked_arm_passes_preflight_and_reports_what_it_will_draw(tmp_path):
    root = _mixture_root(tmp_path, {name: 400_000 for name in BOX_SOURCES})
    summary = arm_preflight(baseline_arm(BOX_SOURCES), data_dir=str(root),
                            total_tokens=100_000)
    assert summary["capped_sources"] == []
    assert summary["l1_skew_pts"] == pytest.approx(0.0)


def test_a_source_missing_from_the_root_is_the_rewrite_the_skew_cannot_see(tmp_path):
    """`resolve_mixture` drops a missing source and takes `target_probs` *after*
    renormalizing, so this rewrite reports zero skew. A baseline arm on a root
    that had lost the code source would train on web alone and record a
    perfectly clean mixture summary."""
    root = _mixture_root(tmp_path, {"fineweb-edu": 400_000,
                                    "dclm-baseline": 400_000})
    from train import mixture_preflight

    quiet = mixture_preflight(str(root), 100_000, weights=baseline_arm(
        BOX_SOURCES).weights, verbose=False)
    assert quiet["l1_skew_pts"] == pytest.approx(0.0)

    with pytest.raises(SystemExit, match="stack-edu-python"):
        arm_preflight(baseline_arm(BOX_SOURCES), data_dir=str(root),
                      total_tokens=100_000)


def test_a_specialists_zero_weight_sources_must_be_present_too(tmp_path):
    """They are what makes a specialist arm the same experiment as the others:
    one root, one holdout, one set of shard files. A root missing one is not
    that root."""
    root = _mixture_root(tmp_path, {"fineweb-edu": 400_000,
                                    "dclm-baseline": 400_000})
    arms = {arm.name: arm for arm in reference_arms(BOX_SOURCES)}
    with pytest.raises(SystemExit, match="stack-edu-python"):
        arm_preflight(arms["only-fineweb-edu"], data_dir=str(root),
                      total_tokens=100_000)


def _fake_supervisor(monkeypatch, seen=None):
    import daedalus.supervise as supervise

    def fake_run_with_resume(cmd, ckpt_path, **kw):
        if seen is not None:
            seen.update({"cmd": list(cmd), "ckpt": ckpt_path, "kw": kw})
        return {"attempts": 1, "resumed": False, "returncodes": [0]}

    monkeypatch.setattr(supervise, "run_with_resume", fake_run_with_resume)
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)


def test_run_arm_supervises_the_checkpoint_and_records_the_mixture(tmp_path,
                                                                   monkeypatch):
    """A relaunch after a lost session has to be able to tell one arm's
    interrupted run from another's, and the thing that distinguishes them here
    is the mixture."""
    root = _mixture_root(tmp_path)
    seen = {}
    _fake_supervisor(monkeypatch, seen)

    arm = {a.name: a for a in reference_arms(BOX_SOURCES)}["only-fineweb-edu"]
    report = run_arm(arm, data_dir=str(root), tag="probe",
                     run_root=str(tmp_path / "runs"), device="cpu",
                     total_tokens=2 * PROBE.batch_tokens)

    assert seen["ckpt"] == str(arm_checkpoint_path(arm, "probe",
                                                   str(tmp_path / "runs")))
    assert seen["kw"]["halt_marker"].endswith("HALTED")
    extra = seen["kw"]["inflight_extra"]
    assert extra["phase"] == "phase7-mixture"
    assert extra["arm"] == "only-fineweb-edu"
    assert extra["weights"]["fineweb-edu"] == pytest.approx(1.0)
    assert report["steps"] == 2
    assert report["preflight"]["capped_sources"] == []


def _close_a_finished_run(arm, tmp_path, **overrides):
    """Exactly what a completed arm leaves: a checkpoint and a closed marker.

    Written through the real `write_inflight`/`mark_inflight_done` rather than
    hand-built, so this pins the on-disk schema the guard reads against the one
    the supervisor writes.
    """
    from daedalus.supervise import mark_inflight_done, write_inflight

    kwargs = {"data_dir": "d", "run_name": arm_run_name(arm, "probe"),
              "total_tokens": 2 * PROBE.batch_tokens, "device": "cpu"}
    kwargs.update(overrides)
    command = train_command(arm, **kwargs)
    ckpt = arm_checkpoint_path(arm, "probe", str(tmp_path / "runs"))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"the phase 7 reference arm")
    write_inflight(str(ckpt.parent), list(command), str(ckpt))
    mark_inflight_done(str(ckpt.parent), "completed")
    return ckpt


def test_a_finished_arm_is_not_retrained_over_its_own_checkpoint(tmp_path,
                                                                 monkeypatch):
    """The supervisor gives no guard here: a closed marker is correctly not
    resumed, and `train.py` then starts at step 0 and overwrites the checkpoint
    on its first save."""
    import daedalus.supervise as supervise

    root = _mixture_root(tmp_path)
    arm = {a.name: a for a in reference_arms(BOX_SOURCES)}["baseline"]
    _close_a_finished_run(arm, tmp_path, data_dir=str(root))

    def refuse(cmd, ckpt_path, **kw):
        raise AssertionError(f"retrained a finished arm over {ckpt_path}")

    monkeypatch.setattr(supervise, "run_with_resume", refuse)
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)

    report = run_arm(arm, data_dir=str(root), tag="probe",
                     run_root=str(tmp_path / "runs"), device="cpu",
                     total_tokens=2 * PROBE.batch_tokens)
    # Recorded, not omitted: an artifact that drops a skipped arm and one that
    # never ran it look identical to a reader.
    assert report["skipped"] == "already-completed"
    assert report["arm"] == "baseline"


def test_a_finished_run_of_a_different_budget_is_still_rerun(tmp_path, monkeypatch):
    """A changed budget in the same directory is a rerun and must retrain --
    which is exactly why `finished_run` cannot be the guard against a redefined
    arm."""
    root = _mixture_root(tmp_path)
    arm = {a.name: a for a in reference_arms(BOX_SOURCES)}["baseline"]
    _close_a_finished_run(arm, tmp_path, data_dir=str(root),
                          total_tokens=PROBE.batch_tokens)
    _fake_supervisor(monkeypatch)

    report = run_arm(arm, data_dir=str(root), tag="probe",
                     run_root=str(tmp_path / "runs"), device="cpu",
                     total_tokens=2 * PROBE.batch_tokens)
    assert "skipped" not in report


def test_a_run_directory_holding_another_mixture_is_refused(tmp_path, monkeypatch):
    """Phase 6's hazard in this phase's currency: an arm *definition* that
    changed -- an edited blueprint share, a moved floor, a source added to the
    root -- while the run name stayed the same."""
    root = _mixture_root(tmp_path)
    arm = {a.name: a for a in reference_arms(BOX_SOURCES)}["baseline"]
    _close_a_finished_run(arm, tmp_path, data_dir=str(root))
    _fake_supervisor(monkeypatch)

    from daedalus.mixture_opt import MixtureArm

    redefined = MixtureArm(name="baseline", basis="an edited blueprint",
                           weights={"dclm-baseline": 0.2, "fineweb-edu": 0.5,
                                    "stack-edu-python": 0.3},
                           is_baseline=True)
    command = train_command(redefined, data_dir=str(root),
                            run_name=arm_run_name(redefined, "probe"),
                            total_tokens=2 * PROBE.batch_tokens)
    ckpt = arm_checkpoint_path(redefined, "probe", str(tmp_path / "runs"))
    assert foreign_run(command, ckpt) is not None
    assert finished_run(command, ckpt) is None

    with pytest.raises(SystemExit, match="already holds a run of mixture"):
        run_arm(redefined, data_dir=str(root), tag="probe",
                run_root=str(tmp_path / "runs"), device="cpu",
                total_tokens=2 * PROBE.batch_tokens)


def test_the_sweep_artifact_carries_the_yardstick_and_what_it_could_not_measure(
        tmp_path, monkeypatch):
    """An artifact that does not record the evaluation weighting cannot be
    checked for having used the same one twice, and one that lists three floors
    as held when a domain had no source is describing a corpus nobody measured.
    """
    root = _mixture_root(tmp_path)
    _fake_supervisor(monkeypatch)

    report = sweep(data_dir=str(root), stage="reference", tag="probe",
                   run_root=str(tmp_path / "runs"),
                   report_root=str(tmp_path / "report"), device="cpu",
                   total_tokens=2 * PROBE.batch_tokens)

    assert [row["arm"] for row in report["arms"]][0] == "baseline"
    assert report["unrepresented_floored_domains"] == ["math"]
    assert report["evaluation_weights"] == pytest.approx(
        evaluation_weights(sorted(BOX_SOURCES)), abs=1e-6)

    written = json.loads(
        (tmp_path / "report" / "mixture-sweep-reference.json").read_text())
    assert written["evaluation_weights"] == report["evaluation_weights"]
    assert len(written["arms"]) == len(report["arms"])


# ============================================= scoring the arms that finished ==
# The half that turns four finished checkpoints into the numbers the
# preregistered rule reads. What these assert is that the pass measures what it
# says it measures -- one weighting for every arm, a full pass, the checkpoint
# that is actually there -- and that the derivation refuses every input it
# cannot honestly subtract.

def _holdout_root(tmp_path, sources=BOX_SOURCES, tokens=400):
    """A holdout root shaped like `data/holdout`: one manifest per source."""
    root = tmp_path / "holdout"
    for name in sources:
        _write_source(root, name, n_tokens=tokens, shard_tokens=tokens // 2)
    return root


def _trained(arm, tmp_path, payload=b"the phase 7 arm"):
    """The checkpoint a finished arm leaves behind."""
    ckpt = arm_checkpoint_path(arm, "probe", str(tmp_path / "runs"))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(payload)
    return ckpt


def _bpb_factory(values):
    """A `bpb_factory` that answers from a `{source: bpb}` table.

    Stands in for loading 4 checkpoints and forward-passing 220M held-out
    tokens, so the rules that decide the phase are testable without a GPU.
    """
    def factory(checkpoint, *, config, device, seq_len, batch_size):
        def bpb_fn(source_dir):
            return values[Path(str(source_dir)).name]
        return bpb_fn
    return factory


def _score(arm, tmp_path, values, **kwargs):
    kwargs.setdefault("holdout_root", str(_holdout_root(tmp_path)))
    return score_arm(arm, sources=BOX_SOURCES, tag="probe",
                     run_root=str(tmp_path / "runs"),
                     out_dir=str(tmp_path / "scorecards"), device="cpu",
                     bpb_factory=_bpb_factory(values), **kwargs)


#: A baseline that trails each specialist on that specialist's own source, by
#: very different margins: code has 0.30 bits/byte of headroom, raw web 0.02,
#: and educational web none at all.
BASELINE_BPB = {"dclm-baseline": 1.20, "fineweb-edu": 1.00,
                "stack-edu-python": 0.90}
SPECIALIST_BPB = {"dclm-baseline": 1.18, "fineweb-edu": 1.00,
                  "stack-edu-python": 0.60}


def _score_reference(tmp_path):
    """Every reference arm scored, as `score --stage reference` leaves them."""
    holdout = str(_holdout_root(tmp_path))
    arms = {arm.name: arm for arm in reference_arms(BOX_SOURCES)}
    _trained(arms["baseline"], tmp_path, b"baseline weights")
    _score(arms["baseline"], tmp_path, BASELINE_BPB, holdout_root=holdout)
    for source in BOX_SOURCES:
        arm = arms[specialist_name(source)]
        _trained(arm, tmp_path, f"{source} weights".encode())
        _score(arm, tmp_path, {source: SPECIALIST_BPB[source]},
               holdout_root=holdout)
    return arms


def test_a_specialist_is_read_only_on_the_source_it_specialised_in(tmp_path):
    """Excess loss reads a specialist through exactly one number. Scoring it on
    the other two sources answers a transfer question this phase never asks, at
    three times the GPU hours per specialist."""
    arms = {arm.name: arm for arm in reference_arms(BOX_SOURCES)}
    specialist = arms["only-stack-edu-python"]
    assert scoring_sources(specialist, BOX_SOURCES) == ["stack-edu-python"]
    assert scoring_sources(arms["baseline"], BOX_SOURCES) == BOX_SOURCES

    _trained(specialist, tmp_path)
    report = _score(specialist, tmp_path, {"stack-edu-python": 0.60})

    card = json.loads(Path(report["scorecard"]).read_text())
    assert card["details"]["sources_requested"] == ["stack-edu-python"]
    assert card["details"]["scored_role"] == "specialist"
    # No aggregate weighting on a one-source card: there is nothing to weight,
    # and `summarize_bpb` refuses weights naming sources it did not measure.
    assert card["details"]["weights"] is None
    assert card["metrics"]["n_sources"] == 1.0


def test_every_mixture_arm_is_scored_under_the_same_fixed_weighting(tmp_path):
    """The whole reason this pass exists rather than reading `val_bpb`: each
    arm's own training mixture would score six models on six different corpora
    and then subtract the numbers."""
    baseline = baseline_arm(BOX_SOURCES)
    _trained(baseline, tmp_path)
    report = _score(baseline, tmp_path, BASELINE_BPB)

    card = json.loads(Path(report["scorecard"]).read_text())
    fixed = evaluation_weights(BOX_SOURCES)
    assert card["details"]["weights"] == pytest.approx(fixed, abs=1e-9)
    assert card["provenance"]["bpb_mode"] == "full"
    expected = sum(BASELINE_BPB[name] * share for name, share in fixed.items())
    assert card["metrics"]["bpb"] == pytest.approx(expected)
    for name, value in BASELINE_BPB.items():
        assert card["metrics"][f"bpb_{name}"] == pytest.approx(value)


def test_scoring_an_arm_that_never_trained_is_refused(tmp_path):
    """A missing checkpoint is an arm that did not run, not an arm worth a
    partial score."""
    with pytest.raises(SystemExit, match="has no checkpoint"):
        _score(baseline_arm(BOX_SOURCES), tmp_path, BASELINE_BPB)


def test_a_rescore_is_skipped_for_the_same_bytes_and_not_for_new_ones(tmp_path):
    """Keyed on the checkpoint digest rather than on the scorecard existing: an
    arm retrained under `--refresh` must not keep the old arm's number."""
    baseline = baseline_arm(BOX_SOURCES)
    _trained(baseline, tmp_path, b"first weights")
    first = _score(baseline, tmp_path, BASELINE_BPB)
    again = _score(baseline, tmp_path, BASELINE_BPB)
    assert again["skipped"] == "already-scored"
    assert again["checkpoint_sha256"] == first["checkpoint_sha256"]

    _trained(baseline, tmp_path, b"retrained weights")
    after = _score(baseline, tmp_path, {**BASELINE_BPB, "fineweb-edu": 0.95})
    assert "skipped" not in after
    assert after["checkpoint_sha256"] != first["checkpoint_sha256"]
    card = json.loads(Path(after["scorecard"]).read_text())
    assert card["metrics"]["bpb_fineweb-edu"] == pytest.approx(0.95)


def test_the_derivation_tilts_toward_the_source_with_measured_headroom(tmp_path):
    """`exp(excess/T)` on the measured gap to each specialist, then the floors.
    Code has 0.30 bits/byte of headroom here and educational web none, so the
    derived mixture has to move mass the first way and not the second."""
    _score_reference(tmp_path)
    record = derive(sources=BOX_SOURCES, tag="probe",
                    out_dir=str(tmp_path / "scorecards"))

    blueprint = baseline_arm(BOX_SOURCES).weights
    weights = record["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert weights["stack-edu-python"] > blueprint["stack-edu-python"]
    assert weights["fineweb-edu"] < blueprint["fineweb-edu"]
    assert record["excess_loss"]["stack-edu-python"] == pytest.approx(0.30)
    assert record["excess_loss"]["fineweb-edu"] == pytest.approx(0.0)
    # The cap is what stops one wild specialist writing the mixture.
    assert record["excess_scores"]["stack-edu-python"] == pytest.approx(
        EXCESS_RATIO_CAP)

    # The floors are the plan's, and `math` has no source on this box -- a
    # record claiming three floors held when one had nothing to hold would be
    # describing a corpus nobody measured.
    assert record["rule"]["unrepresented_floored_domains"] == ["math"]
    assert set(record["rule"]["floors"]) == {"web-raw", "code"}
    assert record["rule"]["temperature"] == EXCESS_TEMPERATURE

    # Recomputable from the artifact rather than trusted: every BPB it used
    # comes with the scorecard and the checkpoint bytes it was read from.
    assert record["baseline"]["bpb"] == pytest.approx(BASELINE_BPB)
    for source in BOX_SOURCES:
        entry = record["specialists"][source]
        assert entry["bpb"] == pytest.approx(SPECIALIST_BPB[source])
        assert Path(entry["scorecard"]).exists()
        assert len(entry["checkpoint_sha256"]) == 64
    assert derive_weights(blueprint, record["excess_loss"]) == pytest.approx(
        weights, abs=1e-6)


def test_deriving_before_every_specialist_is_scored_is_refused(tmp_path):
    """A source without a specialist has no measured excess, and scoring it at
    1.0 would produce a mixture that is partly derived and partly inherited
    while describing itself as derived."""
    holdout = str(_holdout_root(tmp_path))
    arms = {arm.name: arm for arm in reference_arms(BOX_SOURCES)}
    _trained(arms["baseline"], tmp_path, b"baseline weights")
    _score(arms["baseline"], tmp_path, BASELINE_BPB, holdout_root=holdout)

    with pytest.raises(SystemExit, match="only-dclm-baseline"):
        derive(sources=BOX_SOURCES, tag="probe",
               out_dir=str(tmp_path / "scorecards"))


def test_a_sampled_scorecard_cannot_feed_the_derivation(tmp_path):
    """A bounded sample and a full pass are different measurements, and excess
    loss subtracts one from the other."""
    from daedalus.scorecard import ArtifactRef
    from scripts.bpb_eval import run_bpb_eval

    arms = _score_reference(tmp_path)
    run_bpb_eval(
        name=scorecard_name(arms["baseline"], "probe"),
        holdout_root=str(_holdout_root(tmp_path)),
        out_dir=str(tmp_path / "scorecards"),
        artifact=ArtifactRef(path="c", sha256="a" * 64, kind="checkpoint",
                             config=PROBE.config),
        tokenizer_ref=ArtifactRef(path="<smollm2-default>", sha256="0" * 64,
                                  kind="tokenizer"),
        seed=1, git_sha="deadbee", bpb_fn=lambda d: BASELINE_BPB[Path(d).name],
        max_batches=4, weights=evaluation_weights(BOX_SOURCES))

    with pytest.raises(SystemExit, match="bpb_mode 'sample'"):
        derive(sources=BOX_SOURCES, tag="probe",
               out_dir=str(tmp_path / "scorecards"))


def test_scorecards_measured_against_different_holdouts_are_not_subtracted(
        tmp_path):
    """Two BPBs over different corpora are not each other's units; subtracting
    them derives a mixture from an arithmetic error, and every field in the
    artifact would still look right."""
    arms = _score_reference(tmp_path)
    elsewhere = _holdout_root(tmp_path / "other", tokens=600)
    specialist = arms["only-fineweb-edu"]
    _trained(specialist, tmp_path, b"fineweb-edu weights, rescored")
    _score(specialist, tmp_path, {"fineweb-edu": 0.98},
           holdout_root=str(elsewhere))

    with pytest.raises(SystemExit, match="not measured against one holdout"):
        derive(sources=BOX_SOURCES, tag="probe",
               out_dir=str(tmp_path / "scorecards"))


# ------------------------------------------------------------------ verdict ---
# `select_mixture`'s rules are exercised above against hand-built results. What
# these assert is that the reporting pass hands it the right numbers: one
# weighting for every arm, over the whole corpus, from the checkpoint each arm
# actually trained -- and that it refuses every input where that is not true.

def _score_candidates(tmp_path, *, quality, derived_bpb):
    """Both stages as `score` leaves them.

    The reference arms first, because the derived mixture is a function of their
    scorecards, then the two candidate arms trained against it. The baseline is
    shared across the stages under one tag, so the report reads the same card
    the derivation did -- which is how it runs on the box.
    """
    _score_reference(tmp_path)
    weights = derive(sources=BOX_SOURCES, tag="probe",
                     out_dir=str(tmp_path / "scorecards"))["weights"]
    arms = {arm.name: arm for arm in candidate_arms(BOX_SOURCES, weights)}
    for name, values in (("quality-heavy", quality), ("derived", derived_bpb)):
        _trained(arms[name], tmp_path, f"{name} weights".encode())
        _score(arms[name], tmp_path, values)
    return weights, arms


def _report(tmp_path, weights):
    return build_report(sources=BOX_SOURCES, tag="probe",
                        out_dir=str(tmp_path / "scorecards"), derived=weights)


def _better_by(delta, base=None):
    """A per-source BPB table `delta` bits/byte below the baseline's."""
    return {name: value - delta for name, value in (base or BASELINE_BPB).items()}


def test_the_verdict_adopts_the_best_admissible_arm_and_shows_its_working(
        tmp_path):
    """The aggregate is recomputed here under the fixed evaluation weighting
    rather than read off each card, so "one weighting for every arm" is true by
    construction; the card's own number is then a free check on it."""
    weights, _ = _score_candidates(tmp_path, quality=_better_by(0.02),
                                   derived_bpb=_better_by(0.05))
    report = _report(tmp_path, weights)

    selection = report["selection"]
    assert selection["verdict"] == "adopt"
    assert selection["selected"] == "derived"
    assert [row["arm"] for row in selection["ranking"]] == ["derived",
                                                            "quality-heavy"]
    assert selection["refusals"] == []

    fixed = evaluation_weights(BOX_SOURCES)
    rows = {row["arm"]: row for row in report["arms"]}
    assert sorted(rows) == ["baseline", "derived", "quality-heavy"]
    for row in rows.values():
        assert row["aggregate_bpb"] == pytest.approx(
            weighted_bpb(row["per_source_bpb"], fixed))
        # The card's own aggregate agrees, because it was written under this
        # same weighting -- that is the check, not a second measurement.
        assert row["scorecard_bpb"] == pytest.approx(row["aggregate_bpb"])
        assert sorted(row["per_source_bpb"]) == BOX_SOURCES
        assert Path(row["scorecard"]).exists()
        assert len(row["checkpoint_sha256"]) == 64
    assert rows["derived"]["run"] == arm_run_name(
        candidate_arms(BOX_SOURCES, weights)[-1], "probe")
    assert selection["relative_gain"] == pytest.approx(
        (rows["baseline"]["aggregate_bpb"] - rows["derived"]["aggregate_bpb"])
        / rows["baseline"]["aggregate_bpb"])

    # Recomputable from the artifact: the yardstick, the holdout, the context.
    assert report["evaluation_weights"] == pytest.approx(fixed, abs=1e-6)
    assert report["measured_on"]["seq_len"] == PROBE.seq_len
    assert Path(report["measured_on"]["holdout_root"]).name == "holdout"
    assert report["unrepresented_floored_domains"] == ["math"]

    page = render_markdown(report)
    assert "adopt `derived`" in page
    for name in rows:
        assert f"`{name}`" in page


def test_a_candidate_that_cannot_clear_the_bar_keeps_the_baseline(tmp_path):
    """A proxy that cannot separate the arms is evidence about the arms. The
    plan's instruction is to record that, not to advance the best of a tie."""
    weights, _ = _score_candidates(tmp_path, quality=_better_by(0.001),
                                   derived_bpb=_better_by(0.002))
    report = _report(tmp_path, weights)

    selection = report["selection"]
    assert selection["verdict"] == "keep-baseline"
    assert selection["selected"] == "baseline"
    # Admissible and still not selected: the arms cleared the floors and the
    # per-source bound, and simply did not win by enough.
    assert selection["admissible"] == ["derived", "quality-heavy"]
    assert f"{MIN_AGGREGATE_GAIN:.4%}" in selection["reason"]
    assert "keep-baseline `baseline`" in render_markdown(report)


def test_an_arm_that_falls_off_a_cliff_is_refused_and_the_next_one_wins(
        tmp_path):
    """The floors constrain the mixture; this constrains what the mixture
    produced. An arm can satisfy every floor and still have lost a source."""
    cliff = {**_better_by(0.10), "stack-edu-python":
             BASELINE_BPB["stack-edu-python"] * (1 + MAX_SOURCE_REGRESSION * 2)}
    weights, _ = _score_candidates(tmp_path, quality=_better_by(0.02),
                                   derived_bpb=cliff)
    report = _report(tmp_path, weights)

    selection = report["selection"]
    assert [refusal["arm"] for refusal in selection["refusals"]] == ["derived"]
    assert selection["refusals"][0]["reason"] == "source-regression"
    assert selection["refusals"][0]["detail"][0]["source"] == "stack-edu-python"
    # The refused arm had the better aggregate, which is the whole point.
    rows = {row["arm"]: row for row in report["arms"]}
    assert rows["derived"]["aggregate_bpb"] < rows["quality-heavy"][
        "aggregate_bpb"]
    assert selection["verdict"] == "adopt"
    assert selection["selected"] == "quality-heavy"

    # Rendered, not dropped: a table of the survivors reads as though the field
    # was smaller than it was.
    page = render_markdown(report)
    assert "no -- source-regression" in page
    assert "## Refusals" in page


def test_a_verdict_that_never_saw_the_derived_arm_is_refused(tmp_path):
    """`candidate_arms` omits the derived arm when it has no weights -- right
    for a sweep, and here it would silently make the artifact a comparison of
    two arms that describes itself as the comparison of three."""
    _score_candidates(tmp_path, quality=_better_by(0.02),
                      derived_bpb=_better_by(0.05))
    with pytest.raises(SystemExit, match="never saw the derived arm"):
        _report(tmp_path, None)


def test_reporting_before_the_candidates_are_scored_names_the_missing_pass(
        tmp_path):
    """An arm without a scorecard is not a smaller comparison, it is a
    different one -- and the refusal has to send the reader to the candidate
    stage rather than to the reference stage the derivation needs."""
    _score_reference(tmp_path)
    weights = derive(sources=BOX_SOURCES, tag="probe",
                     out_dir=str(tmp_path / "scorecards"))["weights"]

    with pytest.raises(SystemExit, match="quality-heavy.*has no scorecard"):
        _report(tmp_path, weights)
    with pytest.raises(SystemExit, match="score --stage candidates"):
        _report(tmp_path, weights)


def _write_card(arm, tmp_path, values, *, weights, sources=None,
                train_weights=None):
    """A scorecard written straight through `run_bpb_eval`, to build the ones
    `score_arm` would never produce."""
    from daedalus.scorecard import ArtifactRef
    from scripts.bpb_eval import run_bpb_eval

    run_bpb_eval(
        name=scorecard_name(arm, "probe"),
        holdout_root=str(_holdout_root(tmp_path)),
        out_dir=str(tmp_path / "scorecards"),
        artifact=ArtifactRef(path="c", sha256="a" * 64, kind="checkpoint",
                             config=PROBE.config),
        tokenizer_ref=ArtifactRef(path="<smollm2-default>", sha256="0" * 64,
                                  kind="tokenizer"),
        seed=1, git_sha="deadbee", bpb_fn=lambda d: values[Path(d).name],
        max_batches=None, weights=weights, sources=sources,
        runtime={"seq_len": PROBE.seq_len},
        details_extra={"train_weights": train_weights} if train_weights else {})


def test_an_arm_scored_under_a_different_weighting_is_not_comparable(tmp_path):
    """An equal-weighted aggregate and a blueprint-weighted one are two
    different averages of the same numbers. Subtracting them ranks arms by which
    weighting their card happened to carry."""
    weights, arms = _score_candidates(tmp_path, quality=_better_by(0.02),
                                      derived_bpb=_better_by(0.05))
    quality = arms["quality-heavy"]
    _write_card(quality, tmp_path, _better_by(0.02), weights=None,
                train_weights={name: round(value, 6) for name, value
                               in quality.weights.items()})

    with pytest.raises(SystemExit, match="different weighting"):
        _report(tmp_path, weights)


def test_a_card_that_did_not_measure_the_whole_corpus_is_refused(tmp_path):
    """Both halves of the rule need every source: the aggregate is an average
    over the corpus and the regression check compares source by source."""
    weights, arms = _score_candidates(tmp_path, quality=_better_by(0.02),
                                      derived_bpb=_better_by(0.05))
    quality = arms["quality-heavy"]
    _write_card(quality, tmp_path, _better_by(0.02), weights=None,
                sources=["fineweb-edu"],
                train_weights={name: round(value, 6) for name, value
                               in quality.weights.items()})

    with pytest.raises(SystemExit, match="not \\['dclm-baseline'"):
        _report(tmp_path, weights)


def test_a_card_scored_before_the_mixture_was_redefined_is_refused(tmp_path):
    """The arm definitions are computed, not typed: `derived` comes from an
    artifact a re-run of `derive` can legitimately change. Nothing else catches
    it -- the run directory, the scorecard name and the tag all still match."""
    weights, _ = _score_candidates(tmp_path, quality=_better_by(0.02),
                                   derived_bpb=_better_by(0.05))
    rederived = normalized({**weights,
                            "stack-edu-python": weights["stack-edu-python"] * 1.5})

    with pytest.raises(SystemExit) as refused:
        _report(tmp_path, rederived)
    # Every share, not just the one that moved: a re-derivation renormalizes,
    # so one source's excess loss changing shifts the whole mixture.
    assert "measures a different mixture under the same name" in str(refused.value)
    assert "differ on ['dclm-baseline', 'fineweb-edu', 'stack-edu-python']" in str(
        refused.value)
    # And the same arms under the weights they trained on still report.
    assert _report(tmp_path, weights)["selection"]["verdict"] == "adopt"


def test_the_verdict_is_tag_scoped_like_every_other_stage_artifact():
    """Two stages sharing one filename is how a rerun overwrites the evidence
    another stage was launched from."""
    assert verdict_path("probe").name == "mixture-verdict-probe.json"
    assert verdict_path("probe") != verdict_path("rerun")
