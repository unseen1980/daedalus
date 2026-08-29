"""Tests for scripts/architecture_sweep.py and the phase 6 stage-A presets.

The arms are the experiment, so what can go wrong here is an arm that differs
from the others in something nobody meant to vary, a shape stock llama.cpp would
refuse after the GPU time is spent, or a supervisor watching a checkpoint path
the trainer never writes. All three are cheap to assert and expensive to find in
a finished sweep.

Run: python -m pytest tests/test_architecture_sweep.py -v
"""
import pytest

from daedalus.arch_space import (MAX_KV_BYTES_PER_CONTEXT_TOKEN,
                                 PARAM_MATCH_TOLERANCE, QUANT_BLOCK,
                                 candidate_from_config, kv_bytes_per_context_token,
                                 validation_failures)
from daedalus.config import (ARCH_PROBE_ATTENTION_BLOCKS, ARCH_PROBE_CONTROL,
                             ARCH_PROBE_DEPTH, ARCH_PROBE_HIDDEN,
                             ARCH_PROBE_KV_HEADS, ARCH_STAGEB_HIDDEN, PRESETS,
                             arch_probe_preset_name, arch_stageb_config,
                             arch_stageb_preset_name)
from scripts.architecture_sweep import (ARMS, ARMS_BY_NAME, CONTROL, SHAPES,
                                        STAGE_A, STAGE_B, arm_checkpoint_path,
                                        arm_run_name, arms_for,
                                        parameter_spread, run_arm,
                                        selected_arms, train_command)


CONTROL_PRESET = PRESETS["daedalus-150m"]


# ------------------------------------------------------------------- grid -----

def test_the_grid_is_the_plans_two_knobs_fully_crossed():
    """Five attention fractions by three KV-head counts, and nothing else."""
    assert len(ARMS) == len(ARCH_PROBE_ATTENTION_BLOCKS) * len(ARCH_PROBE_KV_HEADS)
    assert {(arm.num_attention_blocks, arm.num_key_value_heads) for arm in ARMS} == {
        (blocks, kv)
        for blocks in ARCH_PROBE_ATTENTION_BLOCKS
        for kv in ARCH_PROBE_KV_HEADS
    }


def test_the_control_is_the_shipped_models_own_ratio():
    """Attention every third layer with four KV heads -- the released model's
    point in this grid. Reading the arms against the released *checkpoint*
    instead would compare architectures and 55M parameters and two different
    training budgets at once."""
    assert (CONTROL.num_attention_blocks,
            CONTROL.num_key_value_heads) == ARCH_PROBE_CONTROL
    assert CONTROL.is_control
    assert sum(1 for arm in ARMS if arm.is_control) == 1

    shipped = CONTROL_PRESET
    assert (CONTROL.num_attention_blocks / ARCH_PROBE_DEPTH
            == pytest.approx(shipped.n_attn_layers / shipped.num_hidden_layers))
    assert CONTROL.num_key_value_heads == shipped.num_key_value_heads


def test_depth_24_makes_every_attention_fraction_a_different_model():
    """The grid spans 1/3 down to 1/12, and 18 layers cannot express those as
    five distinct counts -- 1/9 and 1/12 both land on 2. That collision would
    silently turn a five-point sweep into a four-point one with a duplicated
    arm."""
    assert len(set(ARCH_PROBE_ATTENTION_BLOCKS)) == len(ARCH_PROBE_ATTENTION_BLOCKS)
    fractions = {blocks / ARCH_PROBE_DEPTH for blocks in ARCH_PROBE_ATTENTION_BLOCKS}
    assert len(fractions) == len(ARCH_PROBE_ATTENTION_BLOCKS)


def test_every_arm_realises_the_attention_count_it_names():
    """`num_attention_blocks` is a *request*; `layer_types` is what the
    interleaver produced, and every KV byte and every FLOP follows the realised
    count. An arm whose interleaver dropped a layer is mislabelled in every
    artifact it writes."""
    for arm in ARMS:
        cfg = PRESETS[arm.config]
        assert cfg.n_attn_layers == arm.num_attention_blocks, arm.name
        assert cfg.num_key_value_heads == arm.num_key_value_heads, arm.name


def test_only_attention_count_and_kv_heads_differ_between_arms():
    """The identity of every other field is the experiment."""
    varying = {"num_attention_blocks", "num_key_value_heads", "layer_types"}
    reference = PRESETS[CONTROL.config]
    for arm in ARMS:
        cfg = PRESETS[arm.config]
        for field in vars(reference):
            if field in varying:
                continue
            assert getattr(cfg, field) == getattr(reference, field), \
                f"{arm.name} differs in {field}"


# ------------------------------------------------------ parameter matching ----

def test_holding_the_ffn_fixed_keeps_the_grid_parameter_matched():
    """Within the tolerance `arch_space` screens the candidate space on."""
    spread = parameter_spread()

    assert spread["max_drift_from_midpoint"] < PARAM_MATCH_TOLERANCE
    assert spread["per_arm"].keys() == {arm.name for arm in ARMS}


def test_solving_the_ffn_per_arm_would_match_parameters_worse_than_fixing_it():
    """The reason the stage-A presets do not use `matched_candidate`.

    `block_ff_dim` has to stay a multiple of 256 for k-quants, and one step is
    `3 * hidden * layers * 256` parameters. At this width and depth that is 9%
    of the model -- four times the whole match tolerance -- so snapping each arm
    to its nearest solved FFN moves arms further apart than leaving the FFN
    alone does. If a future stage-A shape grows enough for that to stop being
    true, this test is the thing that says so.
    """
    spread = parameter_spread()
    ff_step_params = 3 * ARCH_PROBE_HIDDEN * ARCH_PROBE_DEPTH * QUANT_BLOCK

    assert ff_step_params / spread["midpoint"] > PARAM_MATCH_TOLERANCE
    # Half a step is the worst-case snap error, and it is worse than the spread
    # the fixed FFN leaves.
    assert 0.5 * ff_step_params / spread["midpoint"] > spread["max_drift_from_midpoint"]


def test_the_residual_parameter_spread_favours_the_attention_sparse_arms():
    """A conv block is dearer than an attention block, so cutting attention adds
    parameters -- which flatters exactly the arms a KV-savings phase hopes will
    win. The direction is asserted so it stays in the record and in the scoring,
    rather than being discovered as a surprise in a ranking."""
    spread = parameter_spread()
    largest = ARMS_BY_NAME[spread["max_arm"]]
    smallest = ARMS_BY_NAME[spread["min_arm"]]

    assert largest.num_attention_blocks == min(ARCH_PROBE_ATTENTION_BLOCKS)
    assert smallest.num_attention_blocks == max(ARCH_PROBE_ATTENTION_BLOCKS)
    assert spread["spread_over_min"] > 0.0


# ---------------------------------------------------------------- KV cost -----

def test_kv_cost_follows_the_analytic_formula_the_screen_uses():
    """Two tensors x KV heads x head_dim x attention layers x 2 bytes. Computed
    by `arch_space` in both places so a sweep artifact and the phase 6 screen
    cannot disagree about the same model."""
    for arm in ARMS:
        cfg = PRESETS[arm.config]
        expected = (2 * arm.num_key_value_heads * cfg.head_dim
                    * arm.num_attention_blocks * 2)
        assert arm.kv_bytes_per_context_token == expected, arm.name
        assert arm.kv_bytes_per_context_token == kv_bytes_per_context_token(cfg)


def test_the_shipped_ratio_costs_more_KV_at_depth_24_than_the_shipped_model_does():
    """A finding rather than a defect: at a fixed attention *fraction*, depth
    buys KV cost. The control is over the plan's 6,144 ceiling for exactly that
    reason, which is why the ceiling is a rule for recommended successors and
    not for the control they are read against."""
    assert CONTROL.kv_bytes_per_context_token > MAX_KV_BYTES_PER_CONTEXT_TOKEN
    assert kv_bytes_per_context_token(CONTROL_PRESET) == MAX_KV_BYTES_PER_CONTEXT_TOKEN


def test_every_arm_but_the_control_is_a_shape_the_screen_would_accept():
    """`validation_failures` is what phase 6 rejects candidates with, so an arm
    it refuses is not a candidate that lost -- it is GPU time spent on a
    comparison that was never valid."""
    for arm in ARMS:
        candidate = candidate_from_config(arm.name, PRESETS[arm.config])
        failures = validation_failures(candidate)
        if arm.is_control:
            assert len(failures) == 1 and "KV cache" in failures[0]
        else:
            assert failures == [], f"{arm.name}: {failures}"


def test_arms_run_control_first_then_down_the_KV_curve():
    """A sweep the deadline cuts short then leaves a contiguous walk down from
    the shipped ratio, which reads as a curve; an arbitrary subset does not."""
    assert ARMS[0] is CONTROL
    costs = [arm.kv_bytes_per_context_token for arm in ARMS[1:]]
    assert costs == sorted(costs, reverse=True)


# --------------------------------------------------------------- the shape ----

def test_the_screen_runs_at_the_shipped_context_length():
    """Halving the context would nearly double throughput and would flatter
    every attention-sparse arm, because attention is worth least at short
    context. That is the direction this phase is hoping to find, so the cheap
    screen is the one measurement it must not take."""
    assert STAGE_A.seq_len == CONTROL_PRESET.max_position_embeddings == 2048
    for arm in ARMS:
        assert PRESETS[arm.config].max_position_embeddings == STAGE_A.seq_len


def test_the_stage_shape_divides_into_whole_gradient_accumulation_steps():
    """A batch that is not a whole number of micro-batches is a different token
    budget per step than the schedule was sized for."""
    assert STAGE_A.batch_tokens % (STAGE_A.micro_batch * STAGE_A.seq_len) == 0
    assert STAGE_A.grad_accum == STAGE_A.batch_tokens // (
        STAGE_A.micro_batch * STAGE_A.seq_len)
    assert STAGE_A.steps == STAGE_A.total_tokens // STAGE_A.batch_tokens


def test_warmup_and_decay_leave_a_fully_decayed_schedule():
    """WSD only pays off if it reaches zero; a warmup that eats the run does not
    leave room for it."""
    assert STAGE_A.warmup_steps < 0.1 * STAGE_A.steps
    assert 0.0 < STAGE_A.decay_frac <= 1.0
    assert STAGE_A.warmup_steps + STAGE_A.decay_frac * STAGE_A.steps <= STAGE_A.steps


# ------------------------------------------------------------- the command ----

def test_only_the_config_flag_differs_between_arm_commands():
    commands = {
        arm.name: train_command(arm, data_dir="d", run_name="r",
                                total_tokens=1_000, device="cpu")
        for arm in ARMS}

    def strip(command):
        out, skip = [], False
        for part in command:
            if skip:
                skip = False
                continue
            if part == "--config":
                skip = True
                continue
            out.append(part)
        return out

    reference = strip(commands[CONTROL.name])
    for name, command in commands.items():
        assert strip(command) == reference, f"{name} differs outside --config"


def test_every_arm_command_parses_and_names_the_preset_it_claims():
    """The flags have to survive `train.py`'s own parsing, or an arm is a string
    that looks right and trains something else."""
    import train as train_mod

    for arm in ARMS:
        command = train_command(arm, data_dir="d", run_name=arm_run_name(arm),
                                total_tokens=STAGE_A.total_tokens, device="cpu",
                                val_dir="v")
        args = train_mod.parse_args(command[2:])   # drop python, train.py

        assert args.config == arm.config
        assert args.config == arch_probe_preset_name(arm.num_attention_blocks,
                                                     arm.num_key_value_heads)
        assert args.seq_start == args.seq_end == STAGE_A.seq_len
        assert args.tok_start == args.tok_end == STAGE_A.batch_tokens
        assert args.micro_batch == STAGE_A.micro_batch
        assert args.total_tokens == STAGE_A.total_tokens
        assert args.warmup_steps == STAGE_A.warmup_steps
        assert args.decay_frac == STAGE_A.decay_frac
        assert args.muon_lr == STAGE_A.muon_lr
        assert args.adam_lr == STAGE_A.adam_lr
        assert args.val_dir == "v"


def test_the_supervisor_watches_the_path_the_trainer_actually_writes():
    """The phase 5 regression, kept. A launcher that composes its own run
    directory hands `run_with_resume` a path that never appears: the in-flight
    marker sits beside a file nothing writes, every relaunch starts from step 0,
    and no log says so."""
    import train as train_mod

    for arm in ARMS:
        command = train_command(arm, data_dir="d", run_name=arm_run_name(arm),
                                total_tokens=1_000, device="cpu")
        args = train_mod.parse_args(command[2:])

        assert str(arm_checkpoint_path(arm)) == train_mod.checkpoint_path_for(args)


# ------------------------------------------------------------------ subset ----

def test_selected_arms_always_includes_the_control_and_puts_it_first():
    chosen = selected_arms("a2-kv1,a4-kv2")

    assert chosen[0] is CONTROL
    assert [arm.name for arm in chosen] == [CONTROL.name, "a2-kv1", "a4-kv2"]


def test_selected_arms_does_not_duplicate_an_explicitly_named_control():
    chosen = selected_arms(f"{CONTROL.name},a2-kv1")

    assert [arm.name for arm in chosen] == [CONTROL.name, "a2-kv1"]


def test_an_unknown_arm_is_refused_rather_than_silently_dropped():
    with pytest.raises(SystemExit):
        selected_arms("a8-kv3")


# ------------------------------------------------------------- supervision ----

def test_run_arm_supervises_the_checkpoint_and_records_the_arm(tmp_path, monkeypatch):
    """`run_arm` must hand the supervisor the checkpoint it can resume from and
    a marker that says which arm it belongs to, or a relaunch after a lost
    session cannot tell one arm's interrupted run from another's."""
    import daedalus.supervise as supervise

    seen = {}

    def fake_run_with_resume(cmd, ckpt_path, **kw):
        seen["cmd"] = list(cmd)
        seen["ckpt"] = ckpt_path
        seen["kw"] = kw
        return {"attempts": 1, "resumed": False, "returncodes": [0]}

    monkeypatch.setattr(supervise, "run_with_resume", fake_run_with_resume)
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)

    arm = ARMS_BY_NAME["a2-kv1"]
    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     total_tokens=2 * STAGE_A.batch_tokens)

    assert seen["ckpt"] == str(arm_checkpoint_path(arm, run_root=str(tmp_path)))
    assert seen["kw"]["halt_marker"].endswith("HALTED")
    extra = seen["kw"]["inflight_extra"]
    assert extra["arm"] == "a2-kv1"
    assert extra["preset"] == arm.config
    assert extra["kv_bytes_per_context_token"] == arm.kv_bytes_per_context_token
    assert report["steps"] == 2
    assert report["total_tokens"] == 2 * STAGE_A.batch_tokens


# --------------------------------------------------------------- re-entry ----

def _close_a_finished_run(arm, tmp_path, **overrides):
    """Leave behind exactly what a completed arm leaves: a checkpoint and the
    marker `run_with_resume` closes over it.

    Written through the real `write_inflight`/`mark_inflight_done` rather than
    hand-built, so this pins the on-disk schema the guard reads against the one
    the supervisor writes.
    """
    from daedalus.supervise import mark_inflight_done, write_inflight

    kwargs = {"data_dir": "d", "run_name": arm_run_name(arm),
              "total_tokens": 2 * STAGE_A.batch_tokens, "device": "cpu"}
    kwargs.update(overrides)
    command = train_command(arm, **kwargs)
    ckpt = arm_checkpoint_path(arm, run_root=str(tmp_path))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"the stage-A result")
    write_inflight(str(ckpt.parent), list(command), str(ckpt))
    mark_inflight_done(str(ckpt.parent), "completed")
    return ckpt


def _no_trainer(monkeypatch):
    """Fail loudly if the supervisor is entered at all."""
    import daedalus.supervise as supervise

    def refuse(cmd, ckpt_path, **kw):
        raise AssertionError(f"retrained a finished arm over {ckpt_path}")

    monkeypatch.setattr(supervise, "run_with_resume", refuse)
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)


def test_a_finished_arm_is_not_retrained_over_its_own_checkpoint(tmp_path,
                                                                 monkeypatch):
    """The checkpoint beside a closed marker *is* the stage-A result.

    The supervisor cannot protect it: a completed run's marker is closed, so
    `interrupted_marker` rightly declines to resume it, and `train.py` then
    starts at step 0 and overwrites the checkpoint on its first save. A sweep
    relaunched after a lost session would destroy all fifteen finished arms in
    the course of reproducing them, and nothing in any log would say so.
    """
    _no_trainer(monkeypatch)
    arm = ARMS_BY_NAME["a2-kv1"]
    ckpt = _close_a_finished_run(arm, tmp_path)

    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     total_tokens=2 * STAGE_A.batch_tokens)

    assert report["skipped"] == "already-completed"
    assert report["returncodes"] == []
    assert ckpt.read_bytes() == b"the stage-A result"


def test_a_skipped_arm_is_recorded_rather_than_omitted(tmp_path, monkeypatch):
    """A sweep artifact that drops a skipped arm and one that never ran it read
    identically, and the second is a gap in the curve."""
    _no_trainer(monkeypatch)
    arm = ARMS_BY_NAME["a2-kv1"]
    _close_a_finished_run(arm, tmp_path)

    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     total_tokens=2 * STAGE_A.batch_tokens)

    assert report["arm"] == "a2-kv1"
    assert report["preset"] == arm.config
    assert report["steps"] == 2
    assert report["command"] == train_command(
        arm, data_dir="d", run_name=arm_run_name(arm),
        total_tokens=2 * STAGE_A.batch_tokens, device="cpu")


def test_a_finished_run_of_a_different_budget_is_not_reused(tmp_path,
                                                            monkeypatch):
    """A shorter smoke that happens to share a run directory is a different
    experiment, and reusing it would report a 100M-token arm that trained on
    2M."""
    import daedalus.supervise as supervise

    seen = {}
    monkeypatch.setattr(supervise, "run_with_resume",
                        lambda cmd, ckpt_path, **kw: seen.update(cmd=list(cmd))
                        or {"attempts": 1, "resumed": False, "returncodes": [0]})
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)

    arm = ARMS_BY_NAME["a2-kv1"]
    _close_a_finished_run(arm, tmp_path, total_tokens=STAGE_A.batch_tokens)

    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     total_tokens=2 * STAGE_A.batch_tokens)

    assert "skipped" not in report
    assert "--total-tokens" in seen["cmd"]


def test_a_halted_arm_is_not_mistaken_for_a_finished_one(tmp_path, monkeypatch):
    """`mark_inflight_done` closes the marker for a watchdog halt too. Reading
    `completed` rather than `outcome` would bank a diverged run as a result."""
    from daedalus.supervise import mark_inflight_done

    import daedalus.supervise as supervise
    seen = {}
    monkeypatch.setattr(supervise, "run_with_resume",
                        lambda cmd, ckpt_path, **kw: seen.update(ran=True)
                        or {"attempts": 1, "resumed": False, "returncodes": [0]})
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)

    arm = ARMS_BY_NAME["a2-kv1"]
    ckpt = _close_a_finished_run(arm, tmp_path)
    mark_inflight_done(str(ckpt.parent), "halted:diverged")

    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     total_tokens=2 * STAGE_A.batch_tokens)

    assert "skipped" not in report
    assert seen.get("ran") is True


def test_refresh_retrains_a_finished_arm_deliberately(tmp_path, monkeypatch):
    """The guard protects against an accidental relaunch, not against an
    operator who means it."""
    import daedalus.supervise as supervise

    seen = {}
    monkeypatch.setattr(supervise, "run_with_resume",
                        lambda cmd, ckpt_path, **kw: seen.update(ran=True)
                        or {"attempts": 1, "resumed": False, "returncodes": [0]})
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)

    arm = ARMS_BY_NAME["a2-kv1"]
    _close_a_finished_run(arm, tmp_path)

    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     total_tokens=2 * STAGE_A.batch_tokens, refresh=True)

    assert seen.get("ran") is True
    assert "skipped" not in report


# ================================================================== stage B ====
# Stage B re-runs the stage-A survivors at ~150M over 250M tokens. What has to
# hold is that it is the *same experiment at another scale*: the same grid
# points, the same cache costs, the same schedule shape -- and that the two
# stages cannot land in each other's run directories.

STAGE_B_ARMS = arms_for(STAGE_B)
STAGE_B_BY_NAME = {arm.name: arm for arm in STAGE_B_ARMS}


def test_stage_b_re_runs_the_same_grid_points_under_the_same_names():
    """Stage A's decision is a list of arm names. If stage B named its points
    differently, handing that decision on would need a translation table, and a
    translation table is somewhere for `a4-kv2` to become `a4-kv1`."""
    assert [arm.name for arm in STAGE_B_ARMS] == [arm.name for arm in ARMS]
    for arm in STAGE_B_ARMS:
        assert arm.config == arch_stageb_preset_name(arm.num_attention_blocks,
                                                     arm.num_key_value_heads)
        assert arm.config != ARMS_BY_NAME[arm.name].config


def test_stage_b_arms_cost_exactly_the_cache_their_stage_a_twins_did():
    """The KV column is what the stage-A screen selected on, and it is
    `2 * kv * head_dim * attention_layers * 2` -- depth and head_dim and nothing
    else stage B widens. Holding them means stage B re-runs the shapes that were
    chosen rather than shapes whose saving moved underneath the choice."""
    for arm in STAGE_B_ARMS:
        twin = ARMS_BY_NAME[arm.name]
        assert arm.kv_bytes_per_context_token == twin.kv_bytes_per_context_token
        assert (PRESETS[arm.config].layer_types
                == PRESETS[twin.config].layer_types)


def test_stage_b_lands_on_the_shipped_models_parameter_count():
    """"150M candidates" is the plan's wording, and the shipped 160.5M model is
    what it means. A stage B that quietly ran at 130M would be a third stage of
    proxy rather than the scale a recommendation rests on."""
    control = PRESETS[STAGE_B_BY_NAME[CONTROL.name].config].param_count()["total"]
    shipped = CONTROL_PRESET.param_count()["total"]

    drift = abs(control - shipped) / shipped
    assert drift < PARAM_MATCH_TOLERANCE, f"{control:,} vs {shipped:,}"
    # ...and materially bigger than the stage-A proxy it is scaling up from.
    stage_a = PRESETS[CONTROL.config].param_count()["total"]
    assert control > 1.4 * stage_a


def test_the_documented_stage_b_arithmetic_is_the_arithmetic():
    """The preset comment justifies 768/1280 with specific numbers, and those
    numbers are the whole argument for the shape. Pinned so prose and model
    cannot drift apart -- a comment claiming a 0.97% miss beside a preset that
    misses by 8% is worse than no comment."""
    control = PRESETS[STAGE_B_BY_NAME[CONTROL.name].config]
    total = control.param_count()["total"]
    shipped = CONTROL_PRESET.param_count()["total"]

    assert total == pytest.approx(158.9e6, abs=0.05e6)
    assert 100.0 * (total - shipped) / shipped == pytest.approx(-0.97, abs=0.01)
    assert control.block_ff_dim / control.hidden_size == pytest.approx(1.67,
                                                                      abs=0.01)
    # The next FFN step up overshoots, which is why 1280 is the choice.
    over = arch_stageb_config(CONTROL.num_attention_blocks,
                              CONTROL.num_key_value_heads)
    over.block_ff_dim += QUANT_BLOCK
    assert (100.0 * (over.param_count()["total"] - shipped) / shipped
            == pytest.approx(7.9, abs=0.1))

    spreads = (parameter_spread(ARMS)["max_drift_from_midpoint"],
               parameter_spread(STAGE_B_ARMS)["max_drift_from_midpoint"])
    assert spreads[0] == pytest.approx(0.015, abs=0.001)
    assert spreads[1] == pytest.approx(0.022, abs=0.001)


def test_stage_b_width_is_the_only_one_the_grid_can_use():
    """768 is forced, not preferred: `num_attention_heads` is `hidden/head_dim`
    and every KV-head count has to divide it. 640 gives ten heads and 4 does not
    divide 10, which would drop a third of the grid."""
    heads = ARCH_STAGEB_HIDDEN // PRESETS[CONTROL.config].head_dim
    for kv in ARCH_PROBE_KV_HEADS:
        assert heads % kv == 0
    assert 640 % PRESETS[CONTROL.config].head_dim == 0
    assert any(640 // PRESETS[CONTROL.config].head_dim % kv
               for kv in ARCH_PROBE_KV_HEADS)


def test_every_stage_b_arm_is_a_shape_the_screen_would_accept():
    """Same bar stage A is held to: a shape `validation_failures` refuses is not
    a candidate that lost, it is GPU time spent on an invalid comparison. The
    control is over the KV ceiling for the same structural reason it is at stage
    A -- depth 24 at the shipped attention *fraction* costs 8,192 bytes."""
    for arm in STAGE_B_ARMS:
        candidate = candidate_from_config(arm.name, PRESETS[arm.config])
        failures = validation_failures(candidate)
        if arm.is_control:
            assert len(failures) == 1 and "KV cache" in failures[0]
        else:
            assert failures == [], f"{arm.name}: {failures}"


def test_only_attention_count_and_kv_heads_differ_between_stage_b_arms():
    varying = {"num_attention_blocks", "num_key_value_heads", "layer_types"}
    reference = PRESETS[STAGE_B_BY_NAME[CONTROL.name].config]
    for arm in STAGE_B_ARMS:
        cfg = PRESETS[arm.config]
        assert cfg.n_attn_layers == arm.num_attention_blocks, arm.name
        for field in vars(reference):
            if field in varying:
                continue
            assert getattr(cfg, field) == getattr(reference, field), \
                f"{arm.name} differs in {field}"


def test_solving_the_ffn_per_arm_is_no_better_at_stage_b_than_at_stage_a():
    """The reason stage B holds the FFN fixed too.

    One `block_ff_dim` step is `3 * hidden * layers * 256`, and widening to 768
    makes it *larger* relative to the model, not smaller. Snapping each arm to
    its nearest solved FFN would therefore still move arms further apart than
    leaving the FFN alone does.
    """
    spread = parameter_spread(STAGE_B_ARMS)
    ff_step = 3 * ARCH_STAGEB_HIDDEN * ARCH_PROBE_DEPTH * QUANT_BLOCK

    assert ff_step / spread["midpoint"] > PARAM_MATCH_TOLERANCE
    assert 0.5 * ff_step / spread["midpoint"] > spread["max_drift_from_midpoint"]


def test_stage_bs_residual_parameter_spread_is_worse_than_stage_as():
    """Recorded because the scoring depends on it and the intuition is
    backwards: a bigger model is not a better-matched grid here. Widening raises
    the conv-block premium over an attention block faster than it raises the
    model, so the fixed-FFN grid spreads *further* at 768 than at 512 -- which
    makes `credited_bpb_delta_pct` more load-bearing at stage B, not less."""
    stage_a = parameter_spread(ARMS)
    stage_b = parameter_spread(STAGE_B_ARMS)

    assert stage_b["max_drift_from_midpoint"] > stage_a["max_drift_from_midpoint"]
    # Still the same direction, so the discount still points the same way.
    assert (STAGE_B_BY_NAME[stage_b["max_arm"]].num_attention_blocks
            == min(ARCH_PROBE_ATTENTION_BLOCKS))


# ------------------------------------------------------------ stage B shape ----

def test_stage_b_is_stage_as_schedule_run_two_and_a_half_times_as_long():
    """Tokens per step, context, and both learning rates are identical, so the
    stages differ in how long the schedule runs and not in what a step is. A
    re-tuned LR between stages would move the comparison the scale-up exists to
    make."""
    assert STAGE_B.batch_tokens == STAGE_A.batch_tokens
    assert STAGE_B.seq_len == STAGE_A.seq_len
    assert STAGE_B.muon_lr == STAGE_A.muon_lr
    assert STAGE_B.adam_lr == STAGE_A.adam_lr
    assert STAGE_B.decay_frac == STAGE_A.decay_frac
    assert STAGE_B.steps == 2.5 * STAGE_A.steps
    assert STAGE_B.total_tokens >= 250_000_000


def test_the_stage_b_shape_divides_into_whole_steps_and_decays_fully():
    assert STAGE_B.total_tokens % STAGE_B.batch_tokens == 0
    assert STAGE_B.batch_tokens % (STAGE_B.micro_batch * STAGE_B.seq_len) == 0
    assert STAGE_B.steps == STAGE_B.total_tokens // STAGE_B.batch_tokens
    assert STAGE_B.warmup_steps < 0.1 * STAGE_B.steps
    assert (STAGE_B.warmup_steps + STAGE_B.decay_frac * STAGE_B.steps
            <= STAGE_B.steps)
    # The warmup is the same fraction of the run stage A's was, so the WSD curve
    # has one shape across both stages.
    assert (STAGE_B.warmup_steps / STAGE_B.steps
            == pytest.approx(STAGE_A.warmup_steps / STAGE_A.steps, abs=0.005))


def test_the_stage_b_micro_batch_stays_under_the_known_good_activation_size():
    """Peak activation memory scales with `micro_batch * hidden_size`. Stage A
    is measured to fit at 8 x 512 on this card; stage B must not be the run that
    discovers 4 x 768 does not, two hours into a seven-hour sweep."""
    assert (STAGE_B.micro_batch * ARCH_STAGEB_HIDDEN
            <= STAGE_A.micro_batch * ARCH_PROBE_HIDDEN)
    assert STAGE_B.grad_accum == STAGE_B.batch_tokens // (
        STAGE_B.micro_batch * STAGE_B.seq_len)


def test_a_stage_b_command_parses_and_carries_the_stage_b_shape():
    import train as train_mod

    for arm in STAGE_B_ARMS:
        command = train_command(arm, data_dir="d",
                                run_name=arm_run_name(arm, STAGE_B.tag),
                                total_tokens=STAGE_B.total_tokens, device="cpu",
                                shape=STAGE_B)
        args = train_mod.parse_args(command[2:])

        assert args.config == arm.config
        assert args.micro_batch == STAGE_B.micro_batch
        assert args.total_tokens == STAGE_B.total_tokens
        assert args.warmup_steps == STAGE_B.warmup_steps
        assert args.seq_start == args.seq_end == STAGE_B.seq_len
        assert str(arm_checkpoint_path(arm, STAGE_B.tag)) == \
            train_mod.checkpoint_path_for(args)


# ------------------------------------------------- keeping the stages apart ----

def test_each_shape_owns_a_distinct_tag_and_preset_family():
    tags = [shape.tag for shape in SHAPES.values()]
    families = [shape.preset_family for shape in SHAPES.values()]

    assert len(set(tags)) == len(tags)
    assert len(set(families)) == len(families)


def test_a_stage_b_run_defaults_into_its_own_run_directory(tmp_path, monkeypatch):
    """The tag comes from the shape rather than from a CLI default, so a stage-B
    sweep launched without `--tag` cannot land in `runs/arch-stagea-*`."""
    import daedalus.supervise as supervise

    seen = {}
    monkeypatch.setattr(supervise, "run_with_resume",
                        lambda cmd, ckpt_path, **kw: seen.update(ckpt=ckpt_path)
                        or {"attempts": 1, "resumed": False, "returncodes": [0]})
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)

    arm = STAGE_B_BY_NAME["a4-kv2"]
    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     shape=STAGE_B, total_tokens=2 * STAGE_B.batch_tokens)

    assert report["run"] == "arch-stageb-a4-kv2"
    assert "arch-stagea-" not in seen["ckpt"]


def test_a_stage_b_arm_refuses_to_train_over_a_stage_a_run_directory(
        tmp_path, monkeypatch):
    """The failure a shared arm vocabulary creates, and the one guard that
    catches it.

    `finished_run` cannot: it compares whole commands, and a differing command
    is exactly what it must let through so that a changed budget re-runs. So a
    mistyped `--tag stagea` on a stage-B sweep would find the stage-A marker,
    decline to skip, and train a 159M arm from step 0 over a finished 105M one
    -- destroying a scored stage-A result with nothing in the log naming it.
    """
    _no_trainer(monkeypatch)
    stage_a_arm = ARMS_BY_NAME["a4-kv2"]
    ckpt = _close_a_finished_run(stage_a_arm, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        run_arm(STAGE_B_BY_NAME["a4-kv2"], data_dir="d",
                run_root=str(tmp_path), device="cpu", tag="stagea",
                shape=STAGE_B, total_tokens=2 * STAGE_B.batch_tokens)

    assert stage_a_arm.config in str(excinfo.value)
    assert ckpt.read_bytes() == b"the stage-A result"


def test_refresh_does_not_override_the_foreign_run_guard(tmp_path, monkeypatch):
    """`--refresh` means "retrain this arm", not "clobber whatever is there".
    Which of two experiments owns a directory is not a question a launcher gets
    to answer by guessing."""
    _no_trainer(monkeypatch)
    ckpt = _close_a_finished_run(ARMS_BY_NAME["a4-kv2"], tmp_path)

    with pytest.raises(SystemExit):
        run_arm(STAGE_B_BY_NAME["a4-kv2"], data_dir="d",
                run_root=str(tmp_path), device="cpu", tag="stagea",
                shape=STAGE_B, total_tokens=2 * STAGE_B.batch_tokens,
                refresh=True)

    assert ckpt.read_bytes() == b"the stage-A result"


# --------------------------------------------- the decision reaches the arms ----
# Stage A's conclusion is a list of arm names; stage B is the longest run this
# phase makes. A hand-typed `--arms` between them is a way to give 250M tokens
# each to shapes the screen did not select, with correct run directories, a
# correct schedule, and a final table naming arms nobody chose.


def _commit_stage_a_report(root, *, selected, verdict="advance"):
    """The half of the stage-A report the launcher reads, at the path it reads
    it from."""
    import json

    from scripts.architecture_report import report_path

    path = report_path(STAGE_A.tag, str(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "tag": STAGE_A.tag, "created_at": "2026-08-25T00:00:00Z",
        "control": CONTROL.name, "shape": {"name": STAGE_A.name},
        "rows": [{"arm": arm.name} for arm in ARMS],
        "stage_b": {"verdict": verdict, "selected": list(selected),
                    "frontier": list(selected), "eligible": list(selected),
                    "dropped_from_frontier": [],
                    "rule": {"floor_pct": 0.5, "max_arms": 3}},
    }, indent=2) + "\n")
    return path


def _capture_trained_arms(monkeypatch):
    trained = []

    def fake(cmd, ckpt_path, **kw):
        trained.append(kw["inflight_extra"]["arm"])
        return {"attempts": 1, "resumed": False, "returncodes": [0]}

    import daedalus.supervise as supervise
    monkeypatch.setattr(supervise, "run_with_resume", fake)
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)
    return trained


def test_stage_b_trains_exactly_the_arms_the_stage_a_report_advanced(
        tmp_path, monkeypatch):
    import json

    from scripts.architecture_sweep import main as sweep_main

    trained = _capture_trained_arms(monkeypatch)
    _commit_stage_a_report(tmp_path, selected=["a4-kv2", "a2-kv1"])

    assert sweep_main(["--run-root", str(tmp_path), "--report-root",
                       str(tmp_path), "--shape", "stage-b", "sweep",
                       "--arms-from-report", STAGE_A.tag, "--data-dir", "d",
                       "--device", "cpu",
                       "--total-tokens", str(STAGE_B.batch_tokens)]) == 0

    assert trained == [CONTROL.name, "a4-kv2", "a2-kv1"], "control first"
    artifact = json.loads((tmp_path / f"sweep-{STAGE_B.tag}.json").read_text())
    assert artifact["advanced_from"]["selected"] == ["a4-kv2", "a2-kv1"]
    assert artifact["advanced_from"]["report"].endswith("stagea-report.json")


def test_a_no_advance_report_stops_the_stage_b_launch(tmp_path, monkeypatch):
    """The refusal has to reach the launcher, not just the reader: a negative
    result that the next command walks past is not a gate."""
    from scripts.architecture_sweep import main as sweep_main

    trained = _capture_trained_arms(monkeypatch)
    _commit_stage_a_report(tmp_path, selected=[], verdict="no-advance")

    with pytest.raises(SystemExit):
        sweep_main(["--run-root", str(tmp_path), "--report-root", str(tmp_path),
                    "--shape", "stage-b", "sweep", "--arms-from-report",
                    STAGE_A.tag, "--data-dir", "d", "--device", "cpu"])

    assert trained == []


def test_naming_the_arms_twice_is_refused(tmp_path, monkeypatch):
    """The whole point of reading the list is that it is not also retyped; two
    sources of truth for it is the failure this closes, not a convenience."""
    from scripts.architecture_sweep import main as sweep_main

    trained = _capture_trained_arms(monkeypatch)
    _commit_stage_a_report(tmp_path, selected=["a4-kv2"])

    with pytest.raises(SystemExit):
        sweep_main(["--run-root", str(tmp_path), "--report-root", str(tmp_path),
                    "--shape", "stage-b", "sweep", "--arms-from-report",
                    STAGE_A.tag, "--arms", "a2-kv1", "--data-dir", "d",
                    "--device", "cpu"])

    assert trained == []


def test_a_changed_budget_on_the_same_preset_is_still_a_rerun(tmp_path,
                                                              monkeypatch):
    """The guard has to separate "another experiment's directory" from "the same
    experiment at a new budget". Only the first is refused; catching the second
    would make every budget change a manual cleanup."""
    import daedalus.supervise as supervise

    seen = {}
    monkeypatch.setattr(supervise, "run_with_resume",
                        lambda cmd, ckpt_path, **kw: seen.update(ran=True)
                        or {"attempts": 1, "resumed": False, "returncodes": [0]})
    monkeypatch.setattr(supervise, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(supervise, "stop_watchdog", lambda *a, **k: None)

    arm = ARMS_BY_NAME["a4-kv2"]
    _close_a_finished_run(arm, tmp_path, total_tokens=STAGE_A.batch_tokens)

    report = run_arm(arm, data_dir="d", run_root=str(tmp_path), device="cpu",
                     total_tokens=2 * STAGE_A.batch_tokens)

    assert seen.get("ran") is True
    assert "skipped" not in report
