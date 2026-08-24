"""Phase 3's decisions, tested where they are made.

The expensive failure in a recovery phase is not a crashed run -- it is a
verdict that looks measured and is not: a probe that trained nothing and
exited 0, a gate quietly satisfied by a missing measurement, or a bar that
moved once the result was in. Everything below targets one of those.
"""

import json

import pytest

from scripts import qat_recovery as qr


# ------------------------------------------------------------ probe shape ---

def test_adam_rate_preserves_the_shipped_ratio():
    """Muon and AdamW cover disjoint parameters, so scaling one alone changes
    which half of the model moves rather than the learning rate."""
    ratio = qr.SHIPPED_ADAM_LR / qr.SHIPPED_MUON_LR
    for muon in qr.PROBE_MUON_LRS:
        assert qr.adam_lr_for(muon) == pytest.approx(muon * ratio)
    # The shipped pair is its own fixed point.
    assert qr.adam_lr_for(qr.SHIPPED_MUON_LR) == pytest.approx(qr.SHIPPED_ADAM_LR)


def test_a_non_positive_muon_rate_is_refused():
    with pytest.raises(ValueError):
        qr.adam_lr_for(0.0)


def test_the_preregistered_arms_are_the_briefs_three_rates():
    arms = qr.probe_arms()
    assert [a.muon_lr for a in arms] == [2e-4, 5e-4, 1e-3]
    assert {a.total_tokens for a in arms} == {qr.PROBE_TOKENS}
    assert {a.stage for a in arms} == {"probe"}
    assert len({a.name for a in arms}) == 3


def test_warmup_is_shortened_to_fit_a_recovery_budget():
    """`train.py`'s 300-step default is longer than the whole 100M probe: a
    run that never leaves warmup is scored on a schedule it never ran."""
    steps = qr.estimated_steps(qr.PROBE_TOKENS)
    warmup = qr.warmup_steps_for(qr.PROBE_TOKENS)
    assert steps < 300, "the probe really is shorter than the default warmup"
    assert warmup < steps / 2
    assert warmup >= qr.MIN_WARMUP_STEPS


def test_a_tiny_budget_still_gets_a_non_zero_warmup():
    assert qr.warmup_steps_for(1) == qr.MIN_WARMUP_STEPS


# ------------------------------------------------------ the launch command ---

def _command(probe, **kw):
    kw.setdefault("init_from", "/root/daedalus/final/hero/checkpoint.pt")
    kw.setdefault("data_dir", "data/shards")
    return qr.train_command(probe, **kw)


def test_every_probe_starts_from_init_from_and_never_resume():
    """The failure this prevents is silent: `--resume` against a finished
    59.9B-token run under a 100M budget makes fit() break at the top of its
    first iteration and exit 0 having trained nothing."""
    for probe in qr.probe_arms():
        cmd = _command(probe)
        assert "--resume" not in cmd
        assert cmd[cmd.index("--init-from") + 1].endswith("checkpoint.pt")
        qr.assert_no_resume(cmd)


def test_assert_no_resume_rejects_a_command_carrying_resume():
    with pytest.raises(ValueError, match="init-from"):
        qr.assert_no_resume(["python", "train.py", "--resume", "x.pt"])


def test_every_probe_is_quantization_aware_from_the_first_step():
    """Phase 3 recovers a finished model: there is no tail to wait for, and
    fp32 opening steps would move weights the grid then has to move back."""
    for probe in qr.probe_arms():
        cmd = _command(probe)
        assert cmd[cmd.index("--qat-frac") + 1] == "1.0"


def test_the_arms_differ_only_in_the_learning_rates():
    """'Identical data, order and seeds' is the whole basis of the comparison,
    so it is asserted rather than left to three shell lines agreeing."""
    def without_rates(cmd):
        out, skip = [], {"--muon-lr", "--adam-lr", "--run-name"}
        i = 0
        while i < len(cmd):
            if cmd[i] in skip:
                i += 2
                continue
            out.append(cmd[i])
            i += 1
        return out

    shapes = {tuple(without_rates(_command(p))) for p in qr.probe_arms()}
    assert len(shapes) == 1, "arms differ by more than their learning rates"


def test_the_command_pins_a_constant_batch_and_sequence_shape():
    """Recovery continues from the end of pretraining, so it must not replay
    the batch/sequence ramp the model has already been through."""
    cmd = _command(qr.probe_arms()[0])
    assert cmd[cmd.index("--seq-start") + 1] == cmd[cmd.index("--seq-end") + 1]
    assert cmd[cmd.index("--tok-start") + 1] == cmd[cmd.index("--tok-end") + 1]


def test_the_command_carries_the_shortened_warmup_and_full_decay():
    probe = qr.probe_arms()[0]
    cmd = _command(probe)
    assert cmd[cmd.index("--warmup-steps") + 1] == str(probe.warmup_steps)
    assert cmd[cmd.index("--decay-frac") + 1] == str(qr.DECAY_FRAC)


def test_the_memory_flags_are_absent_unless_asked_for():
    """They are a repair for an OOM, not part of the preregistered arm: a run
    that silently enabled checkpointing would not be the run that was planned."""
    plain = _command(qr.probe_arms()[0])
    assert "--gradient-checkpointing" not in plain
    assert "--loss-chunk-size" not in plain

    constrained = _command(qr.probe_arms()[0], gradient_checkpointing=True,
                           loss_chunk_size=256)
    assert "--gradient-checkpointing" in constrained
    assert constrained[constrained.index("--loss-chunk-size") + 1] == "256"


def test_the_command_parses_against_train_pys_real_parser():
    """A flag this module invents but train.py does not accept would fail only
    at launch, after the box has been idle waiting for it."""
    import train as train_mod
    cmd = _command(qr.probe_arms()[0], val_dir="data/holdout")
    parsed = train_mod.parse_args(cmd[2:])   # drop "python train.py"
    assert parsed.qat_frac == 1.0
    assert parsed.init_from.endswith("checkpoint.pt")
    assert parsed.resume is None
    assert parsed.warmup_steps == qr.probe_arms()[0].warmup_steps
    assert parsed.decay_frac == qr.DECAY_FRAC
    assert parsed.seq_start == parsed.seq_end == qr.SEQ_LEN


# ---------------------------------------------------------------- finiteness ---

def test_a_clean_run_passes_the_finiteness_gate():
    rows = [{"step": 1, "loss": 3.0, "grad_norm": 0.4, "skipped_updates": 0},
            {"step": 2, "loss": 2.9, "grad_norm": 0.3, "skipped_updates": 0}]
    verdict = qr.finiteness_from_metrics(rows)
    assert verdict["passed"] and verdict["skipped_updates"] == 0


def test_a_skipped_update_fails_the_gate_even_when_every_loss_is_finite():
    """The skipped step never writes a non-finite row of its own -- it returns
    early -- so the count is the only evidence it happened."""
    rows = [{"step": 1, "loss": 3.0, "grad_norm": 0.4, "skipped_updates": 0},
            {"step": 2, "loss": 2.9, "grad_norm": 0.3, "skipped_updates": 1}]
    verdict = qr.finiteness_from_metrics(rows)
    assert not verdict["passed"]
    assert verdict["skipped_updates"] == 1
    assert "skipped" in verdict["reason"]


def test_a_non_finite_loss_fails_the_gate():
    rows = [{"step": 1, "loss": float("nan"), "skipped_updates": 0}]
    assert not qr.finiteness_from_metrics(rows)["passed"]


def test_a_non_finite_gradient_norm_fails_the_gate():
    rows = [{"step": 1, "loss": 3.0, "grad_norm": float("inf"),
             "skipped_updates": 0}]
    assert not qr.finiteness_from_metrics(rows)["passed"]


def test_a_run_with_no_metrics_fails_rather_than_passing_by_absence():
    """A run that died before its first metrics row has not demonstrated
    finiteness; treating silence as success is how a dead probe gets scored."""
    assert not qr.finiteness_from_metrics([])["passed"]


def test_read_metrics_survives_a_row_torn_by_a_crash(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 3.0}\n{"step": 2, "los\n{"step": 3, "loss": 2.5}\n')
    rows = qr.read_metrics(tmp_path)
    assert [r["step"] for r in rows] == [1, 3]


# --------------------------------------------------------------- the gates ---

BASELINE = qr.Baseline(
    q4_penalty_pct=5.539,
    perplexity_fp16=6.6135,
    perplexity_q4_0=6.9798,
    five_task_mean=47.374,
    retrieval={"retrieval-passkey:d256": 0.90, "retrieval-passkey:d2048": 0.80},
)


def _observed(**overrides):
    observed = {
        "name": "candidate",
        "q4_penalty_pct": 2.0,
        "perplexity_fp16": 6.6135,
        "five_task_mean": 47.374,
        "bpb": 1.0,
        "retrieval": dict(BASELINE.retrieval),
        "finiteness": {"passed": True, "reason": "", "skipped_updates": 0,
                       "non_finite_rows": 0},
    }
    observed.update(overrides)
    return observed


def test_penalty_reduction_is_measured_against_the_hosts_own_baseline():
    assert qr.penalty_reduction_frac(5.539, 2.7695) == pytest.approx(0.5)
    assert qr.penalty_reduction_frac(5.539, 0.0) == pytest.approx(1.0)
    # A candidate can make quantization damage worse, and that must read as
    # negative rather than as a small win.
    assert qr.penalty_reduction_frac(5.539, 8.0) < 0


def test_halving_the_penalty_exactly_clears_the_improvement_gate():
    """The bar is '>= 50%', so the boundary case must pass -- an off-by-one
    here silently raises the gate above what was preregistered."""
    scored = qr.score_candidate(BASELINE, _observed(q4_penalty_pct=5.539 / 2))
    assert scored["meets_improvement_gate"] and scored["accepted"]


def test_falling_just_short_of_halving_does_not_clear_it():
    scored = qr.score_candidate(BASELINE, _observed(q4_penalty_pct=2.7696))
    assert not scored["meets_improvement_gate"] and not scored["accepted"]


def test_target_and_stretch_are_reported_separately_from_the_gate():
    assert qr.score_candidate(BASELINE, _observed(q4_penalty_pct=2.5))["meets_target"]
    assert not qr.score_candidate(BASELINE, _observed(q4_penalty_pct=2.5))["meets_stretch"]
    assert qr.score_candidate(BASELINE, _observed(q4_penalty_pct=0.9))["meets_stretch"]


def test_an_fp16_regression_over_half_a_percent_fails_retention():
    over = BASELINE.perplexity_fp16 * 1.0051
    scored = qr.score_candidate(BASELINE, _observed(perplexity_fp16=over))
    assert not scored["accepted"]
    check = next(c for c in scored["retention"]["checks"]
                 if c["gate"] == "fp16-perplexity")
    assert not check["passed"] and check["observed_pct"] > 0.5


def test_a_five_task_drop_over_half_a_point_fails_retention():
    scored = qr.score_candidate(
        BASELINE, _observed(five_task_mean=BASELINE.five_task_mean - 0.6))
    assert not scored["accepted"]


def test_retrieval_is_gated_per_depth_not_on_the_average():
    """A model that loses one depth and gains another nets out flat while
    being worse at exactly the thing the gate protects."""
    retrieval = {"retrieval-passkey:d256": 0.90 - 0.02,   # 2 points worse
                 "retrieval-passkey:d2048": 0.80 + 0.02}  # 2 points better
    scored = qr.score_candidate(BASELINE, _observed(retrieval=retrieval))
    assert not scored["accepted"]
    check = next(c for c in scored["retention"]["checks"]
                 if c["gate"] == "retrieval")
    assert check["worst_depth"] == "retrieval-passkey:d256"
    assert check["observed_drop_points"] == pytest.approx(2.0)


def test_an_unmeasured_depth_fails_rather_than_passing_by_absence():
    scored = qr.score_candidate(
        BASELINE, _observed(retrieval={"retrieval-passkey:d256": 0.90}))
    assert not scored["accepted"]
    check = next(c for c in scored["retention"]["checks"]
                 if c["gate"] == "retrieval")
    assert not check["passed"] and "d2048" in check["reason"]


def test_an_unmeasured_metric_fails_rather_than_passing_by_absence():
    for missing in ("perplexity_fp16", "five_task_mean"):
        observed = _observed()
        observed.pop(missing)
        assert not qr.score_candidate(BASELINE, observed)["accepted"], missing


def test_a_skipped_update_blocks_acceptance_however_good_the_penalty_is():
    """Halving Q4 damage in a run that dropped updates is not a recovery, it
    is an unexplained model."""
    observed = _observed(q4_penalty_pct=0.5,
                         finiteness={"passed": False, "reason": "1 skipped",
                                     "skipped_updates": 1, "non_finite_rows": 0})
    assert not qr.score_candidate(BASELINE, observed)["accepted"]


# ------------------------------------------------------------- selection ---

def test_the_winner_is_the_largest_paired_q4_reduction():
    scored = [qr.score_candidate(BASELINE, _observed(name="a", q4_penalty_pct=2.5)),
              qr.score_candidate(BASELINE, _observed(name="b", q4_penalty_pct=1.5)),
              qr.score_candidate(BASELINE, _observed(name="c", q4_penalty_pct=2.0))]
    assert qr.select_winner(scored)["name"] == "b"


def test_fp16_retention_breaks_a_tie_on_the_penalty():
    scored = [
        qr.score_candidate(BASELINE, _observed(
            name="worse-fp16", q4_penalty_pct=2.0, perplexity_fp16=6.62)),
        qr.score_candidate(BASELINE, _observed(
            name="better-fp16", q4_penalty_pct=2.0, perplexity_fp16=6.60)),
    ]
    assert qr.select_winner(scored)["name"] == "better-fp16"


def test_bpb_breaks_a_tie_below_fp16_retention():
    scored = [
        qr.score_candidate(BASELINE, _observed(name="hi-bpb", bpb=1.10)),
        qr.score_candidate(BASELINE, _observed(name="lo-bpb", bpb=1.05)),
    ]
    assert qr.select_winner(scored)["name"] == "lo-bpb"


def test_a_rejected_candidate_never_wins_however_much_damage_it_removed():
    """This is what makes the retention gates mandatory rather than advisory."""
    scored = [
        qr.score_candidate(BASELINE, _observed(
            name="reckless", q4_penalty_pct=0.1,
            perplexity_fp16=BASELINE.perplexity_fp16 * 1.02)),
        qr.score_candidate(BASELINE, _observed(name="careful", q4_penalty_pct=2.7)),
    ]
    assert qr.select_winner(scored)["name"] == "careful"


def test_no_accepted_candidate_selects_nothing_rather_than_the_least_bad():
    scored = [qr.score_candidate(BASELINE, _observed(name="a", q4_penalty_pct=5.0)),
              qr.score_candidate(BASELINE, _observed(name="b", q4_penalty_pct=4.0))]
    assert qr.select_winner(scored) is None


# ------------------------------------------------------------ escalation ---

def test_no_passing_probe_stops_the_phase_with_a_negative_result():
    decision = qr.escalation_decision(None, None)
    assert not decision["escalate"] and "negative result" in decision["reason"]


def test_a_followup_short_of_ten_percent_relative_does_not_escalate():
    best = qr.score_candidate(BASELINE, _observed(name="best", q4_penalty_pct=2.00))
    followup = qr.score_candidate(BASELINE, _observed(name="f", q4_penalty_pct=1.85))
    decision = qr.escalation_decision(best, followup)
    assert not decision["escalate"]
    assert decision["relative_improvement_frac"] == pytest.approx(0.075)


def test_a_followup_clearing_ten_percent_relative_escalates():
    best = qr.score_candidate(BASELINE, _observed(name="best", q4_penalty_pct=2.00))
    followup = qr.score_candidate(BASELINE, _observed(name="f", q4_penalty_pct=1.80))
    decision = qr.escalation_decision(best, followup)
    assert decision["escalate"]
    assert decision["relative_improvement_frac"] == pytest.approx(0.10)


def test_a_bar_met_exactly_is_not_refused_by_representation_error():
    """`(2.00 - 1.80) / 2.00` is 0.09999999999999998 in binary floating point.
    Without a tolerance, a follow-up that improved Q4 damage by exactly the
    preregistered 10% would be refused escalation by arithmetic rather than by
    the rule -- which is a moved bar, just an accidental one."""
    assert qr.penalty_reduction_frac(2.00, 1.80) < qr.MIN_ESCALATION_REDUCTION_FRAC
    assert qr._at_least(qr.penalty_reduction_frac(2.00, 1.80),
                        qr.MIN_ESCALATION_REDUCTION_FRAC)
    # The tolerance is far below the resolution of anything being measured, so
    # it cannot rescue a candidate that genuinely missed.
    assert not qr._at_least(0.0999, qr.MIN_ESCALATION_REDUCTION_FRAC)
    assert not qr._at_most(0.5001, qr.MAX_FP16_PPL_REGRESSION_PCT)


def test_a_followup_that_broke_a_gate_does_not_escalate():
    best = qr.score_candidate(BASELINE, _observed(name="best", q4_penalty_pct=2.0))
    followup = qr.score_candidate(BASELINE, _observed(
        name="f", q4_penalty_pct=0.5,
        perplexity_fp16=BASELINE.perplexity_fp16 * 1.02))
    decision = qr.escalation_decision(best, followup)
    assert not decision["escalate"] and "fp16" in decision["reason"]


def test_an_unscored_followup_does_not_escalate_on_the_probes_alone():
    best = qr.score_candidate(BASELINE, _observed(name="best", q4_penalty_pct=2.0))
    assert not qr.escalation_decision(best, None)["escalate"]


# ------------------------------------------------------- preregistration ---

def test_the_plan_states_every_bar_before_any_probe_runs(tmp_path):
    payload = qr.build_preregistration(BASELINE, init_from="/x/checkpoint.pt",
                                       init_from_sha256="a" * 64)
    gates = payload["gates"]
    assert gates["improvement"]["implied_max_penalty_pct"] == pytest.approx(
        5.539 * 0.5)
    assert gates["retention"]["max_fp16_ppl_regression_pct"] == 0.5
    assert gates["retention"]["no_skipped_non_finite_updates"] is True
    assert gates["escalation"]["min_relative_reduction_frac"] == 0.10
    assert payload["selection_order"][0].startswith("paired Q4")
    assert payload["shared_training"]["qat_frac"] == 1.0
    assert payload["shared_training"]["init_from_only"] is True
    assert len(payload["arms"]) == 3


def test_the_plan_records_the_input_checkpoint_it_was_written_against(tmp_path):
    """A verdict about "the released model" that cannot name which file it
    started from is not a verdict about the released model."""
    payload = qr.build_preregistration(BASELINE, init_from="/x/checkpoint.pt",
                                       init_from_sha256="b" * 64)
    assert payload["input"] == {"checkpoint": "/x/checkpoint.pt",
                                "sha256": "b" * 64}


def test_the_plan_raises_the_retrieval_item_count_before_measuring():
    """At 10 items per depth one item is 10 points, so a 1-point gate could
    only be met by exact equality. Raising the count afterwards would be the
    threshold-tuning the phase forbids; raising it in the plan is not."""
    payload = qr.build_preregistration(BASELINE, init_from="/x/c.pt",
                                       init_from_sha256="c" * 64)
    measurement = payload["retrieval_measurement"]
    assert measurement["per_depth"] >= 100
    assert measurement["per_depth"] > 10
    assert list(measurement["depths"]) == [256, 512, 1024, 2048]


def test_a_written_plan_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "preregistration.json"
    payload = qr.build_preregistration(BASELINE, init_from="/x/c.pt",
                                       init_from_sha256="d" * 64)
    qr.write_preregistration(path, payload)
    with pytest.raises(qr.PreregistrationError, match="written once"):
        qr.write_preregistration(path, {"schema": 1})
    # The original survives the refused write.
    assert json.loads(path.read_text())["gates"]["improvement"][
        "baseline_penalty_pct"] == pytest.approx(5.539)


def test_scoring_before_preregistration_refuses_to_run(tmp_path):
    """Scoring against a plan that does not exist yet is scoring against a bar
    that can still be chosen."""
    with pytest.raises(SystemExit, match="preregistration"):
        qr.main(["--root", str(tmp_path), "score", "--name", "x",
                 "--run-dir", str(tmp_path)])


# -------------------------------------------------------- artifact reading ---

def test_collect_observation_reads_the_numbers_from_the_written_scorecards(
        tmp_path):
    """A verdict must cite the same number the evaluator wrote, not a
    re-derivation of it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 3.0, "grad_norm": 0.2,
                    "skipped_updates": 0}) + "\n")

    quant = tmp_path / "quant-comparison.json"
    quant.write_text(json.dumps({
        "q4_penalty_pct": 2.5, "perplexity_fp16": 6.61,
        "perplexity_q4_0": 6.775, "chunks_worse": 100, "chunks_better": 192,
        "n_chunks": 292}))

    retrieval = tmp_path / "retrieval-passkey.json"
    retrieval.write_text(json.dumps({
        "name": "retrieval-passkey",
        "metrics": {"exact_match": 0.83, "exact_match_d256": 0.9,
                    "exact_match_d2048": 0.8, "n_d256": 100.0}}))

    observed = qr.collect_observation("cand", run_dir=run_dir,
                                      quant_comparison=quant,
                                      retrieval=[retrieval])
    assert observed["q4_penalty_pct"] == 2.5
    assert observed["perplexity_fp16"] == 6.61
    assert observed["finiteness"]["passed"]
    # Only the per-depth figures become gate inputs; the aggregate and the
    # item counts must not be mistaken for depths.
    assert observed["retrieval"] == {"retrieval-passkey:d256": 0.9,
                                     "retrieval-passkey:d2048": 0.8}


def test_collect_observation_reports_a_dead_run_as_failing(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    observed = qr.collect_observation("dead", run_dir=run_dir)
    assert not observed["finiteness"]["passed"]
    assert qr.score_candidate(BASELINE, observed)["accepted"] is False


# ----------------------------------------------------------- the cli path ---

def test_preregister_then_command_then_score_round_trips(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "q4_penalty_pct": 5.539, "perplexity_fp16": 6.6135,
        "perplexity_q4_0": 6.9798, "five_task_mean": 47.374,
        "retrieval": {"retrieval-passkey:d256": 0.9}}))

    assert qr.main(["--root", str(tmp_path), "preregister",
                    "--baseline", str(baseline_path),
                    "--init-from", "/x/checkpoint.pt",
                    "--init-from-sha256", "e" * 64]) == 0
    assert (tmp_path / "preregistration.json").exists()

    arm = qr.probe_arms()[0].name
    assert qr.main(["--root", str(tmp_path), "command", "--arm", arm,
                    "--init-from", "/x/checkpoint.pt",
                    "--data-dir", "data/shards"]) == 0

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 3.0, "skipped_updates": 0}) + "\n")
    assert qr.main(["--root", str(tmp_path), "score", "--name", arm,
                    "--run-dir", str(run_dir)]) == 0
    scored = json.loads((tmp_path / "scored" / f"{arm}.json").read_text())
    # No quantization measurement yet, so it is explicitly not accepted rather
    # than accepted on the strength of the metrics it does have.
    assert scored["accepted"] is False


def test_launch_supervises_with_a_halt_marker_and_no_resume_on_attempt_one(
        tmp_path, monkeypatch):
    """The three supervision pieces, checked where they are wired rather than
    where they are documented.

    The halt marker is the one that costs most when missing: without it a
    supervisor reads the watchdog's SIGTERM as an ordinary crash, resumes the
    diverged checkpoint with no watchdog left running, and trains a broken
    model for the rest of the budget before exiting 0.
    """
    seen = {}

    def fake_run_with_resume(cmd, ckpt_path, **kw):
        seen["cmd"] = list(cmd)
        seen["ckpt_path"] = ckpt_path
        seen["kw"] = kw
        return {"attempts": 1, "resumed": False, "returncodes": [0]}

    def fake_start_watchdog(run_name, run_dir, target_tokens, **kw):
        seen["watchdog"] = {"run_name": run_name, "run_dir": run_dir,
                            "target_tokens": target_tokens, **kw}
        return "watchdog-proc"

    stopped = []
    monkeypatch.setattr("daedalus.supervise.run_with_resume",
                        fake_run_with_resume)
    monkeypatch.setattr("daedalus.supervise.start_watchdog",
                        fake_start_watchdog)
    monkeypatch.setattr("daedalus.supervise.stop_watchdog", stopped.append)

    probe = qr.probe_arms()[0]
    command = _command(probe)
    report = qr.launch_supervised(probe, command, run_root=str(tmp_path))

    assert report["returncodes"] == [0]
    assert "--resume" not in seen["cmd"], "attempt one must not resume"
    assert seen["kw"]["halt_marker"].endswith("HALTED")
    # The retry resumes the probe's own checkpoint, never the released one.
    assert seen["ckpt_path"] == str(tmp_path / probe.name / "checkpoint.pt")
    assert seen["watchdog"]["supervised"] is True
    assert seen["watchdog"]["target_tokens"] == probe.total_tokens
    # The arm is identifiable from the inflight marker a boot resume reads.
    assert seen["kw"]["inflight_extra"]["arm"] == probe.name


def test_launch_stops_the_watchdog_even_when_the_arm_fails(tmp_path,
                                                           monkeypatch):
    """A failed arm must not leave a watchdog polling a dead directory for the
    rest of the night."""
    def boom(cmd, ckpt_path, **kw):
        raise RuntimeError("arm died")

    stopped = []
    monkeypatch.setattr("daedalus.supervise.run_with_resume", boom)
    monkeypatch.setattr("daedalus.supervise.start_watchdog",
                        lambda *a, **k: "watchdog-proc")
    monkeypatch.setattr("daedalus.supervise.stop_watchdog", stopped.append)

    with pytest.raises(RuntimeError, match="arm died"):
        qr.launch_supervised(qr.probe_arms()[0], _command(qr.probe_arms()[0]),
                             run_root=str(tmp_path))
    assert stopped == ["watchdog-proc"]


def test_launch_refuses_a_command_carrying_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("daedalus.supervise.run_with_resume",
                        lambda *a, **k: pytest.fail("should not have launched"))
    monkeypatch.setattr("daedalus.supervise.start_watchdog",
                        lambda *a, **k: None)
    monkeypatch.setattr("daedalus.supervise.stop_watchdog", lambda p: None)
    with pytest.raises(ValueError, match="init-from"):
        qr.launch_supervised(qr.probe_arms()[0],
                             _command(qr.probe_arms()[0]) + ["--resume", "x.pt"],
                             run_root=str(tmp_path))


def test_the_cli_refuses_an_arm_that_was_never_preregistered(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "q4_penalty_pct": 5.539, "perplexity_fp16": 6.6135,
        "perplexity_q4_0": 6.9798, "five_task_mean": 47.374}))
    qr.main(["--root", str(tmp_path), "preregister", "--baseline",
             str(baseline_path), "--init-from", "/x/c.pt",
             "--init-from-sha256", "f" * 64])
    with pytest.raises(SystemExit, match="not a preregistered arm"):
        qr.main(["--root", str(tmp_path), "command", "--arm", "qat-recovery-lr9",
                 "--init-from", "/x/c.pt", "--data-dir", "data/shards"])
