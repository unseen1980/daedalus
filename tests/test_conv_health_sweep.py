"""Tests for scripts/conv_health.py -- the phase 5 arm sweep and its verdict.

The arms are the experiment, so most of what can go wrong here is an arm that
differs from the others in something nobody meant to vary, or a verdict that
reads a clean-looking result the fix note already said is unreadable. Both are
cheap to assert and expensive to discover from a finished sweep.

Run: python -m pytest tests/test_conv_health_sweep.py -v
"""
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
    ConvArm,
    arm_checkpoint_path,
    arm_run_name,
    probe_train_command,
    probe_total_tokens,
    verdict,
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


# ---------------------------------------------------------------- verdict -----

def _score(name, *, dead, in_proj=0.08, out_proj=0.05, kernel=0.05,
           loss=5.0, matched_delta=0.6, flagged_delta=0.0, hidden=256,
           layers=6):
    return {
        "arm": name,
        "health": {"defined": True, "dead_fraction": dead,
                   "per_layer": [{"layer_index": i, "hidden_size": hidden,
                                  "dead_channels": int(dead * hidden)}
                                 for i in range(layers)]},
        "held_out_loss": loss,
        "ablate_flagged": {"delta": flagged_delta},
        "ablate_matched": {"delta": matched_delta},
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
