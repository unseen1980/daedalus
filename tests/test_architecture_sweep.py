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
                             ARCH_PROBE_KV_HEADS, PRESETS,
                             arch_probe_preset_name)
from scripts.architecture_sweep import (ARMS, ARMS_BY_NAME, CONTROL, STAGE_A,
                                        arm_checkpoint_path, arm_run_name,
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
