"""Tests for scripts/conv_health.py -- the phase 5 arm sweep and its verdict.

The arms are the experiment, so most of what can go wrong here is an arm that
differs from the others in something nobody meant to vary, or a verdict that
reads a clean-looking result the fix note already said is unreadable. Both are
cheap to assert and expensive to discover from a finished sweep.

Run: python -m pytest tests/test_conv_health_sweep.py -v
"""
from dataclasses import replace

import pytest

from daedalus.config import PRESETS
from daedalus.muon import conv_proj_wd_schedule
from scripts.conv_health import (
    ARMS,
    ARMS_BY_NAME,
    CONTROL,
    CONTROL_DEATH_FLOOR,
    MAX_DEAD_FRACTION,
    MAX_NORM_RATIO,
    PAIRED_SHAPE,
    PROBE_SHAPE,
    ConvArm,
    arm_checkpoint_path,
    arm_run_name,
    load_stage,
    matched_ablation_set,
    probe_train_command,
    probe_total_tokens,
    recommendation,
    render_report,
    selected_arms,
    verdict,
    write_report,
)


# ------------------------------------------------------------------- arms -----

def test_the_four_preregistered_arms_are_the_ones_in_the_plan():
    assert [arm.name for arm in ARMS] == [
        "shipped-0.1", "weak-0.0133", "warmup-0-to-0.1", "weak-then-0.1"]
    assert CONTROL.name == "shipped-0.1"
    assert CONTROL.start == 0.1 and CONTROL.end is None


def test_there_is_no_pure_zero_decay_arm():
    """Decay 0 wins on death outright and still cannot be the recommendation:
    `conv-death-fix-validated.md` measured 6.8x/10.5x projection growth over
    600 steps, which is 0.5% of a real run, against a decay whose stated
    purpose is late-training stability. The two varying arms are the shapes
    that can stop the death *and* keep an equilibrium."""
    for arm in ARMS:
        final = conv_proj_wd_schedule(10_000, 10_000, arm.start, end=arm.end,
                                      ramp_frac=arm.ramp_frac,
                                      hold_frac=arm.hold_frac)
        assert final > 0.0, f"{arm.name} ends at decay {final}, with no equilibrium"


def test_the_control_runs_in_the_same_two_group_layout_as_every_other_arm():
    """Running the control as the shipped *single*-group split would make the
    arms differ in optimizer layout as well as in decay, and the comparison
    rests on one variable. 0.1 in its own group is the shipped decay with the
    layout matched."""
    assert "--conv-proj-wd" in CONTROL.train_flags()
    for arm in ARMS:
        assert arm.train_flags()[0] == "--conv-proj-wd"


def test_only_the_decay_flags_differ_between_arms():
    """The identity of everything else *is* the experiment."""
    commands = {
        arm.name: probe_train_command(arm, data_dir="d", run_name="r",
                                      total_tokens=1_000, device="cpu")
        for arm in ARMS}

    def strip(command):
        """Drop each conv-decay flag *and its value*, leaving the shared run."""
        out, skip = [], False
        for part in command:
            if skip:
                skip = False
                continue
            if part.startswith("--conv-proj-wd"):
                skip = True
                continue
            out.append(part)
        return out

    stripped = {name: strip(command) for name, command in commands.items()}
    reference = stripped[CONTROL.name]
    for name, command in stripped.items():
        assert command == reference, f"{name} differs outside the decay flags"


def test_the_ramp_flags_reach_the_command_only_for_the_varying_arms():
    constant = probe_train_command(ARMS_BY_NAME["weak-0.0133"], data_dir="d",
                                   run_name="r", total_tokens=1, device="cpu")
    ramped = probe_train_command(ARMS_BY_NAME["warmup-0-to-0.1"], data_dir="d",
                                 run_name="r", total_tokens=1, device="cpu")

    assert "--conv-proj-wd-end" not in constant
    assert "--conv-proj-wd-end" in ramped
    assert ramped[ramped.index("--conv-proj-wd-ramp-frac") + 1] == repr(0.1)


def test_every_arm_command_parses_and_configures_the_schedule_it_names():
    """The flags have to survive `train.py`'s own parsing, or an arm is a
    string that looks right and trains the shipped schedule."""
    import train as train_mod

    for arm in ARMS:
        command = probe_train_command(arm, data_dir="d", run_name=arm.name,
                                      total_tokens=1_000, device="cpu")
        args = train_mod.parse_args(command[2:])   # drop python, train.py
        assert args.conv_proj_wd == arm.start
        assert args.conv_proj_wd_end == arm.end
        assert args.conv_proj_wd_ramp_frac == arm.ramp_frac


def test_the_probe_holds_sequence_length_and_batch_flat():
    """The ramp buys throughput on a long run, and it would make `lr x steps`
    -- the clock the arms are compared on -- mean something different early and
    late."""
    command = probe_train_command(CONTROL, data_dir="d", run_name="r",
                                  total_tokens=1, device="cpu")
    assert command[command.index("--seq-start") + 1] == command[
        command.index("--seq-end") + 1]
    assert command[command.index("--tok-start") + 1] == command[
        command.index("--tok-end") + 1]


def test_the_probe_shape_is_the_established_control():
    """hidden 256 at the shipped 2:1 conv:attention ratio."""
    cfg = PRESETS["conv-probe"]
    assert cfg.hidden_size == 256
    assert cfg.num_hidden_layers == 9
    assert cfg.layer_types.count("conv") == 6
    assert cfg.n_attn_layers == 3


def test_probe_token_budget_matches_the_step_count_it_claims():
    assert probe_total_tokens(600) == 600 * 8 * 256


def test_the_supervisor_watches_the_path_the_trainer_actually_writes():
    """The bug this test exists for, caught by a 20-step smoke.

    `run_with_resume` is handed a checkpoint path so it can resume from it, and
    `train.py` resolves its own run directory as `runs/<run_name>` with no CLI
    flag to move it. A launcher that watched `runs/conv-health/<run_name>`
    instead left the in-flight marker beside a file that was never written:
    every relaunch would have started from step 0, and nothing in any log would
    have said so -- which is the exact failure phase 5 was told to fix before
    starting a long run.
    """
    import train as train_mod

    for arm in ARMS:
        command = probe_train_command(arm, data_dir="d",
                                      run_name=arm_run_name(arm, "probe"),
                                      total_tokens=1_000, device="cpu")
        parsed = train_mod.parse_args(command[2:])
        assert str(arm_checkpoint_path(arm, "probe")) == \
            train_mod.checkpoint_path_for(parsed), (
                f"{arm.name}: the supervisor and the trainer disagree about "
                f"where the checkpoint lives")


def test_reports_are_kept_out_of_the_run_directories():
    """Runs live where `train.py` puts them; a verdict next to a checkpoint is
    one `rm -rf runs/<name>` away from being mistaken for one."""
    from scripts.conv_health import REPORT_ROOT, RUN_ROOT

    assert RUN_ROOT == "runs"
    assert REPORT_ROOT.startswith("runs/")
    assert REPORT_ROOT != RUN_ROOT


def test_arm_run_names_are_distinct_and_carry_the_tag():
    names = {arm_run_name(arm, "probe") for arm in ARMS}
    assert len(names) == len(ARMS)
    assert arm_run_name(CONTROL, "paired") != arm_run_name(CONTROL, "probe")


# ------------------------------------------------------------------ shape -----

def test_the_escalation_is_the_shipped_shape_at_the_plans_budget():
    """Phase 5 step 6: a paired 150M-parameter, 500M-token run at lr 0.04. The
    150M part is not incidental -- a claim about *this model's* channels cannot
    be established at the probe's hidden 256."""
    assert PAIRED_SHAPE.config == "daedalus-150m"
    assert PRESETS[PAIRED_SHAPE.config].hidden_size == 768
    assert PAIRED_SHAPE.total_tokens == 500_000_000
    assert PAIRED_SHAPE.muon_lr == 0.04


def test_the_escalation_batch_gives_the_decay_enough_steps_to_act():
    """The thing that has to carry over from the probe is the decay clock, not
    the learning rate: Muon decays once per *optimizer step*, so a batch large
    enough would spend 500M tokens without the control losing a channel.

    At hero's 512k tokens/step the clock is ~29 and `exp(-2.9)` of shrink would
    not cross a threshold that needs ~`exp(-4.6)`. This asserts the shape that
    was actually chosen clears it with room."""
    assert PAIRED_SHAPE.batch_tokens == 131_072
    assert PAIRED_SHAPE.steps == 3815
    shrink_exponent = PAIRED_SHAPE.decay_clock * CONTROL.start
    assert shrink_exponent > 4.6 * 2, (
        f"the shipped arm's decay clock only reaches exp(-{shrink_exponent:.1f}); "
        f"that may not produce material death in 500M tokens")


def test_the_escalation_holds_batch_and_sequence_flat_like_the_probe():
    command = probe_train_command(CONTROL, data_dir="d", run_name="r",
                                  total_tokens=PAIRED_SHAPE.total_tokens,
                                  device="cpu", shape=PAIRED_SHAPE)

    assert command[command.index("--seq-start") + 1] == command[
        command.index("--seq-end") + 1]
    assert command[command.index("--tok-start") + 1] == command[
        command.index("--tok-end") + 1]


def test_only_the_decay_flags_differ_between_escalation_arms():
    """The same identity property the probe has, asserted at the shape that
    costs GPU-hours to get wrong."""
    commands = {
        arm.name: probe_train_command(arm, data_dir="d", run_name="r",
                                      total_tokens=PAIRED_SHAPE.total_tokens,
                                      device="cpu", shape=PAIRED_SHAPE)
        for arm in ARMS}

    def strip(command):
        out, skip = [], False
        for part in command:
            if skip:
                skip = False
                continue
            if part.startswith("--conv-proj-wd"):
                skip = True
                continue
            out.append(part)
        return out

    reference = strip(commands[CONTROL.name])
    for name, command in commands.items():
        assert strip(command) == reference, f"{name} differs outside the decay"


def test_the_escalation_command_parses_and_carries_its_own_shape():
    import train as train_mod

    command = probe_train_command(ARMS_BY_NAME["weak-then-0.1"], data_dir="d",
                                  run_name="r", shape=PAIRED_SHAPE,
                                  total_tokens=PAIRED_SHAPE.total_tokens,
                                  device="cpu")
    args = train_mod.parse_args(command[2:])

    assert args.config == "daedalus-150m"
    assert args.muon_lr == 0.04
    assert args.total_tokens == 500_000_000
    assert args.seq_start == args.seq_end == 2048
    assert args.tok_start == args.tok_end == 131_072
    assert args.micro_batch == 8
    assert args.conv_proj_wd_ramp_frac == 0.3


def test_the_supervisor_watches_the_escalation_checkpoint_too():
    """The phase 4 failure, re-asserted at the other shape: the escalation runs
    a different preset, and a checkpoint path that assumed the probe's would
    hand `run_with_resume` a file that never appears."""
    import train as train_mod

    for arm in ARMS:
        command = probe_train_command(
            arm, data_dir="d", run_name=arm_run_name(arm, "paired"),
            total_tokens=PAIRED_SHAPE.total_tokens, device="cpu",
            shape=PAIRED_SHAPE)
        parsed = train_mod.parse_args(command[2:])
        assert str(arm_checkpoint_path(arm, "paired",
                                       config=PAIRED_SHAPE.config)) == \
            train_mod.checkpoint_path_for(parsed)


def test_a_smoke_step_count_overrides_the_shapes_budget():
    """`--steps` exists for smokes. It must not be able to silently shorten a
    preregistered run, so the budget comes from the shape unless asked."""
    assert probe_total_tokens(4, PAIRED_SHAPE) == 4 * 131_072
    assert probe_total_tokens(600, PROBE_SHAPE) == PROBE_SHAPE.total_tokens


# ------------------------------------------------------------ arm selection ---

def test_a_subset_always_carries_the_control_and_carries_it_first():
    """Every criterion an arm is read against is relative to the control -- 2x
    its alive-channel norms, 0.5% of its held-out loss, its flagged set's size.
    A subset without it is unreadable, not cheaper."""
    chosen = selected_arms("weak-0.0133,weak-then-0.1")

    assert [arm.name for arm in chosen] == [
        CONTROL.name, "weak-0.0133", "weak-then-0.1"]


def test_naming_the_control_does_not_duplicate_it():
    chosen = selected_arms(f"{CONTROL.name},weak-0.0133")

    assert [arm.name for arm in chosen] == [CONTROL.name, "weak-0.0133"]


def test_no_subset_means_every_preregistered_arm():
    assert list(selected_arms(None)) == list(ARMS)
    assert list(selected_arms("")) == list(ARMS)


def test_a_misspelled_arm_stops_the_run_rather_than_shrinking_it():
    """Silently dropping an unknown name would run a two-arm escalation that
    reports as the three-arm one it was asked for."""
    with pytest.raises(SystemExit, match="weak-0.013"):
        selected_arms("weak-0.013")


# -------------------------------------------------------- matched ablation ----

def _health_with(alive_per_layer, hidden=256):
    """Layer healths where layer `i` has `alive_per_layer[i]` live channels.

    Built from the real `layer_health`, not a stub, so `weakest_alive`'s actual
    ordering and alive test are what the bookkeeping is measured against.
    """
    import torch

    from daedalus.conv_health import layer_health
    from daedalus.model import ShortConv

    cfg = replace(PRESETS["tiny"], hidden_size=hidden, block_ff_dim=256)
    layers = []
    for index, alive in enumerate(alive_per_layer):
        layer = ShortConv(cfg)
        torch.manual_seed(index)
        with torch.no_grad():
            layer.in_proj.weight.normal_(0.0, 0.02)
            layer.conv.weight.normal_(0.0, 0.05)
            layer.out_proj.weight.normal_(0.0, 0.02)
            # Collapse every channel past `alive`, which the coupled reading
            # then flags as dead.
            layer.out_proj.weight[:, alive:] *= 1e-11
        layers.append(layer_health(layer, index))
    return layers


def test_the_matched_set_reports_a_layer_that_could_not_supply_its_k():
    """The 2026-08-25 probe defect. `weakest_alive` slices a list, so a layer
    with fewer alive channels than the control's k silently returns all of them
    and the ablation stops being baseline-sized. That has to reach the report,
    because the delta on its own looks like the check passing."""
    layers = _health_with([200, 40], hidden=256)

    matched, requested, short = matched_ablation_set(layers, {0: 150, 1: 150})

    assert requested == 300
    assert len(matched[0]) == 150            # 200 alive, k met
    assert len(matched[1]) == 40             # 40 alive, k of 150 impossible
    assert list(short) == [1]
    assert short[1] == {"requested": 150, "delivered": 40}


def test_a_matched_set_that_met_every_k_reports_no_shortfall():
    layers = _health_with([200, 200], hidden=256)

    matched, requested, short = matched_ablation_set(layers, {0: 150, 1: 150})

    assert requested == 300
    assert sum(len(v) for v in matched.values()) == 300
    assert short == {}


def test_the_matched_set_never_includes_a_channel_the_instrument_called_dead():
    """Matched means weakest *alive*: including flagged channels would make the
    control and the flagged ablation the same measurement."""
    layers = _health_with([200], hidden=256)
    dead = {index for index in range(200, 256)}

    matched, _, _ = matched_ablation_set(layers, {0: 150})

    assert not (set(matched[0]) & dead)


# ---------------------------------------------------------------- verdict -----

def _score(name, *, dead, in_proj=0.08, out_proj=0.05, kernel=0.05,
           loss=5.0, matched_delta=0.6, flagged_delta=0.0, hidden=256,
           layers=6, matched=None):
    ablate_matched = {"delta": matched_delta}
    if matched is not None:
        ablate_matched.update(matched)
    return {
        "arm": name,
        "health": {"defined": True, "dead_fraction": dead,
                   "per_layer": [{"layer_index": i, "hidden_size": hidden,
                                  "dead_channels": int(dead * hidden)}
                                 for i in range(layers)]},
        "held_out_loss": loss,
        "ablate_flagged": {"delta": flagged_delta},
        "ablate_matched": ablate_matched,
        "projection_norms": {"alive_weighted": {
            "in_proj": in_proj, "out_proj": out_proj, "kernel": kernel,
            "alive_channels": hidden * layers}},
    }


def _scored(*arms):
    return {"tag": "probe", "control": CONTROL.name, "arms": list(arms)}


def test_a_control_that_did_not_die_invalidates_the_whole_sweep():
    """The fix note's clause 2: an arm that looks clean because *nothing* died
    in the probe is not validation. Without this, a probe where no arm died
    reads as four successes."""
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.0),
        _score("weak-0.0133", dead=0.0)))

    assert decision["valid"] is False
    assert decision["positive_control"]["exhibits_death"] is False
    assert decision["passing"] == []


def test_a_dying_control_makes_the_sweep_readable():
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67),
        _score("weak-0.0133", dead=0.0)))

    assert decision["valid"] is True
    assert decision["passing"] == ["weak-0.0133"]


def test_the_control_death_floor_is_the_bar_it_claims():
    below = verdict(_scored(_score(CONTROL.name,
                                   dead=CONTROL_DEATH_FLOOR - 1e-6)))
    at = verdict(_scored(_score(CONTROL.name, dead=CONTROL_DEATH_FLOOR)))

    assert below["valid"] is False
    assert at["valid"] is True


def test_an_arm_that_stopped_the_death_by_growing_is_rejected():
    """The risk decay 0 does not retire, made a check: an arm can reach 0% dead
    by letting the projections grow without bound, and that trades channel
    death for a late-training stability question and worse Q4 damage."""
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67, in_proj=0.08, out_proj=0.05),
        _score("warmup-0-to-0.1", dead=0.0, in_proj=0.54, out_proj=0.53)))

    arm = decision["arms"][0]
    assert arm["dead_fraction"] < MAX_DEAD_FRACTION
    assert arm["checks"]["norms_within_2x"] is False
    assert arm["passes"] is False
    assert max(arm["norm_ratio"].values()) > MAX_NORM_RATIO


def test_an_arm_whose_matched_ablation_costs_nothing_is_rejected():
    """The false positive this whole phase is built to avoid: 0% dead because
    the metric never fired. If removing the arm's weakest baseline-sized set is
    free, its channels were not carrying anything either."""
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67),
        _score("weak-0.0133", dead=0.0, matched_delta=0.0)))

    arm = decision["arms"][0]
    assert arm["checks"]["matched_ablation_bites"] is False
    assert arm["passes"] is False


def test_a_matched_ablation_that_could_not_be_baseline_sized_is_not_credited():
    """`weakest_alive` slices a list, so an arm with fewer alive channels than
    the control's k silently gets all of them instead of k of them. On the
    2026-08-25 probe that turned three arms' matched ablation into "remove
    every channel still alive", whose large delta then read as the check
    passing. Removing every live channel hurting is a tautology, so the size
    has to be part of the claim."""
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67),
        _score("warmup-0-to-0.1", dead=0.72, matched_delta=1.77,
               matched={"channels": 422, "requested_channels": 1112,
                        "baseline_sized": False})))

    arm = decision["arms"][0]
    assert arm["matched_ablation_delta"] > 0.0
    assert arm["checks"]["matched_ablation_bites"] is False
    assert arm["matched_ablation_channels"] == 422
    assert arm["matched_ablation_requested"] == 1112


def test_a_matched_ablation_that_got_its_full_k_is_credited():
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67),
        _score("weak-0.0133", dead=0.0, matched_delta=0.6,
               matched={"channels": 1112, "requested_channels": 1112,
                        "baseline_sized": True})))

    arm = decision["arms"][0]
    assert arm["checks"]["matched_ablation_bites"] is True
    assert arm["passes"] is True


def test_a_score_written_before_the_size_was_recorded_still_reads():
    """Backwards compatibility, deliberately: a score with neither field never
    recorded the fact, and failing it on that would rewrite an old result
    rather than read it."""
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67),
        _score("weak-0.0133", dead=0.0, matched_delta=0.6)))

    assert decision["arms"][0]["checks"]["matched_ablation_bites"] is True


def test_an_arm_that_costs_held_out_loss_is_rejected():
    """`muon.py` calls the decay what keeps Muon stable, so a fix that trades
    dead channels for a worse model is not a fix."""
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67, loss=5.0),
        _score("weak-0.0133", dead=0.0, loss=5.1)))

    arm = decision["arms"][0]
    assert arm["checks"]["loss_not_worse"] is False
    assert arm["passes"] is False
    assert arm["loss_regression"] == pytest.approx(0.02)


def test_an_arm_that_still_dies_is_rejected_on_the_dead_fraction_alone():
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67),
        _score("weak-0.0133", dead=0.15)))

    arm = decision["arms"][0]
    assert arm["checks"]["dead_fraction_under_1pc"] is False
    assert arm["passes"] is False


def test_a_clean_arm_passes_every_check_at_once():
    decision = verdict(_scored(
        _score(CONTROL.name, dead=0.67, in_proj=0.08, out_proj=0.05),
        _score("weak-then-0.1", dead=0.0, in_proj=0.12, out_proj=0.09,
               loss=4.95, matched_delta=0.6)))

    arm = decision["arms"][0]
    assert all(arm["checks"].values()), arm["checks"]
    assert decision["passing"] == ["weak-then-0.1"]


def test_the_verdict_records_the_thresholds_it_applied():
    """A verdict that does not carry its own bar cannot be re-read later
    against the bar that was actually in force."""
    decision = verdict(_scored(_score(CONTROL.name, dead=0.67)))

    assert decision["thresholds"]["max_dead_fraction"] == MAX_DEAD_FRACTION
    assert decision["thresholds"]["max_norm_ratio"] == MAX_NORM_RATIO


def test_an_unknown_arm_name_is_not_silently_scored_as_the_control():
    with pytest.raises(KeyError):
        verdict({"tag": "probe", "control": "not-an-arm",
                 "arms": [_score(CONTROL.name, dead=0.67)]})


def test_a_custom_arm_still_builds_a_valid_command():
    """`ConvArm` is the unit a follow-up sweep would add an arm with."""
    arm = ConvArm("weak-then-0.1-held", start=0.0133, end=0.1, ramp_frac=0.3,
                  hold_frac=0.1)
    flags = arm.train_flags()

    assert flags[flags.index("--conv-proj-wd-hold-frac") + 1] == repr(0.1)


# ----------------------------------------------------------------- report -----

def _stage(tag, arms, *, control_dead=0.67, flagged_delta=1e-8, shape=None):
    """A stage as `load_stage` returns one, built from the real `verdict`."""
    scored = {"tag": tag, "control": CONTROL.name,
              "arms": [_score(CONTROL.name, dead=control_dead,
                              flagged_delta=flagged_delta)] + list(arms)}
    return {"tag": tag, "verdict": verdict(scored),
            "shape": shape or {"name": tag, "config": "conv-probe",
                               "total_tokens": 1_228_800, "muon_lr": 0.15}}


def _paired_like():
    """The two stages as the escalation actually produced them.

    Numbers from `runs/conv-health/verdict-{probe,paired}.json` so the renderer
    is exercised on the shape of result it has to describe, rather than on a
    clean case that never occurred.
    """
    probe = _stage("probe", [
        _score("weak-0.0133", dead=0.0553, in_proj=0.1018, out_proj=0.0805,
               kernel=0.0377, loss=5.0029, matched_delta=0.388,
               matched={"channels": 1106, "requested_channels": 1112,
                        "baseline_sized": False}),
        _score("weak-then-0.1", dead=0.6888, in_proj=0.0764, out_proj=0.0479,
               kernel=0.0496, loss=5.0012, matched_delta=1.631,
               matched={"channels": 478, "requested_channels": 1112,
                        "baseline_sized": False}),
    ], control_dead=0.7240, flagged_delta=-8.94e-08)
    paired = _stage("paired", [
        _score("weak-0.0133", dead=0.1452, in_proj=0.1459, out_proj=0.1165,
               kernel=0.1065, loss=5.0071, matched_delta=3.954,
               matched={"channels": 4855, "requested_channels": 4964,
                        "baseline_sized": False}),
        _score("weak-then-0.1", dead=0.4243, in_proj=0.0751, out_proj=0.0496,
               kernel=0.0628, loss=4.9845, matched_delta=4.017,
               matched={"channels": 4021, "requested_channels": 4964,
                        "baseline_sized": False}),
    ], control_dead=0.5386, flagged_delta=2.98e-08,
        shape={"name": "paired", "config": "daedalus-150m",
               "total_tokens": 500_000_000, "muon_lr": 0.04})
    return [probe, paired]


def test_the_report_records_a_negative_result_when_no_arm_cleared_the_rule():
    """The outcome the escalation actually produced. A report that renders a
    table and stops leaves the reader to decide what it meant, which is the
    point at which a bar gets relaxed to suit the numbers."""
    verdicts = _paired_like()

    advice = recommendation(verdicts)
    text = render_report(verdicts)

    assert advice["negative_result"] is True
    assert advice["selected"] is None
    assert advice["decisive_stage"] == "paired"
    assert "negative result" in text.lower()


def test_the_report_names_the_arm_when_one_actually_clears_the_rule():
    verdicts = [_stage("paired", [
        _score("weak-then-0.1", dead=0.0, in_proj=0.09, out_proj=0.06,
               kernel=0.05, loss=4.95, matched_delta=0.6,
               matched={"channels": 1112, "requested_channels": 1112,
                        "baseline_sized": True})])]

    advice = recommendation(verdicts)

    assert advice["negative_result"] is False
    assert advice["selected"] == "weak-then-0.1"
    assert "weak-then-0.1" in render_report(verdicts)


def test_a_stage_whose_control_did_not_die_is_reported_as_unreadable():
    """`valid: false` means the sweep measured nothing, and an arm's 0% dead in
    that stage is uninterpretable rather than a success. Rendering its table
    without saying so is how a broken stage reads as a clean one."""
    verdicts = [_stage("probe", [_score("weak-0.0133", dead=0.0)],
                       control_dead=0.0)]

    advice = recommendation(verdicts)
    text = render_report(verdicts)

    assert advice["decisive_stage"] is None
    assert advice["negative_result"] is True
    assert "unreadable" in text.lower()


def test_the_report_shows_an_uncredited_ablation_as_short_not_as_its_delta():
    """A 4.0-nat delta from 4,021 of a requested 4,964 channels is "every live
    channel was removed", not "the arm's weakest baseline-sized set bites". The
    delta alone reads as the check passing, so the size travels with it."""
    text = render_report(_paired_like())

    assert "4021/4964" in text
    assert "uncredited" in text.lower()


def test_the_report_shows_how_the_norm_cost_moved_between_the_two_shapes():
    """The escalation's job. Weakening the decay cost 1.61x the control's
    out_proj at the screen and 2.33x at the escalation, so the trade worsens
    with the decay clock -- which is only visible with both stages side by
    side, and is the reason a probe result was not enough."""
    text = render_report(_paired_like())

    assert "1.61" in text and "2.33" in text


def test_the_report_states_that_it_cannot_revive_the_released_models_channels():
    """Phase 5 step 8. The runs here are from-initialization arms at a proxy
    shape; nothing in them touches V1's trained weights, and a reader carrying
    "channel death fixed" into the release notes would be wrong."""
    text = render_report(_paired_like())

    assert "V2" in text
    assert "revive" in text.lower() or "revived" in text.lower()


def test_the_report_reads_the_control_ablation_as_the_instrument_working():
    """The control flagged 53.9% of channels and removing all of them moved
    held-out loss by 3e-08. That is the finding, not a footnote: the shipped
    decay's dead channels were carrying nothing."""
    text = render_report(_paired_like())

    assert "53.86%" in text
    assert "+2.98e-08 nats" in text
    assert "positive control" in text.lower()


def test_a_stage_that_was_never_scored_is_skipped_rather_than_invented(tmp_path):
    (tmp_path / "verdict-paired.json").write_text(
        __import__("json").dumps(_paired_like()[1]["verdict"]))

    assert load_stage(tmp_path, "probe") is None
    stage = load_stage(tmp_path, "paired")
    assert stage["tag"] == "paired"
    assert stage["shape"] == {}


def test_the_cross_shape_table_carries_the_control_whose_own_death_moved():
    """Every norm ratio is against the control *of its own stage*, and the
    control's dead fraction fell 72.40% -> 53.86% between them. `weak-then-0.1`
    fell 68.88% -> 42.43% over the same pair, which is following the baseline
    down rather than improving on it -- unreadable without the control row."""
    text = render_report(_paired_like())

    assert f"`{CONTROL.name}` (control) | 72.40% -> 53.86%" in text


def test_a_clause_no_arm_could_meet_is_reported_as_having_decided_nothing():
    """All four matched ablations came back short, so the clause rejected every
    arm without discriminating between any of them. A reader counting it as
    evidence would overstate how much of the rule actually applied."""
    verdicts = _paired_like()

    advice = recommendation(verdicts)
    text = render_report(verdicts)

    kinds = [finding["kind"] for finding in advice["findings"]]
    assert "ablation-clause-never-applied" in kinds
    finding = next(f for f in advice["findings"]
                   if f["kind"] == "ablation-clause-never-applied")
    assert finding["decided_any_verdict"] == []
    assert "decided no verdict" in text


def test_an_arm_rejected_only_by_the_unmeetable_clause_is_flagged_for_re_reading():
    """The case that would matter: an arm clearing dead fraction, norms and
    loss, and failing only on a set that could not be baseline-sized. That is
    a rejection resting on a measurement that was never taken."""
    verdicts = [_stage("paired", [
        _score("weak-then-0.1", dead=0.0, in_proj=0.09, out_proj=0.06,
               kernel=0.05, loss=4.95, matched_delta=1.9,
               matched={"channels": 400, "requested_channels": 1112,
                        "baseline_sized": False})])]

    advice = recommendation(verdicts)

    finding = next(f for f in advice["findings"]
                   if f["kind"] == "ablation-clause-never-applied")
    assert finding["decided_any_verdict"] == ["weak-then-0.1"]
    assert "re-read" in render_report(verdicts)


def test_the_report_ends_without_a_blank_line_so_it_can_be_committed():
    """`git diff --check` rejects a blank line at EOF, and the approved wrapper
    runs it before every commit -- so a report that renders correctly and ends
    in two newlines is a report the phase cannot push."""
    text = render_report(_paired_like())

    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_write_report_refuses_when_nothing_was_ever_scored(tmp_path):
    with pytest.raises(SystemExit, match="no verdict"):
        write_report(tmp_path, tags=("probe", "paired"))
