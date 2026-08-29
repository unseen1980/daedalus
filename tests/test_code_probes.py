"""Gate 1's three arms: what they are, and what a relaunch of them does.

The subject here is not the training -- it is everything around it that decides
whether three numbers can be compared at the end: that the arms differ in the
learning rate and nothing else, that they start from the released base and never
from `--resume`, that the mixture they read is the composed one rather than
three sets of flags, and that a sweep interrupted at arm two resumes rather than
restarts.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import code_probes as CP


def _shard_dir(root, name, tokens=100_000_000):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"total_tokens": tokens, "shards": [], "eos_id": 0}))
    return path


def _record(tmp_path, weights=None, tokens=None) -> dict:
    """A composed mixture and the record `corpus mixture` writes beside it."""
    weights = weights or {"code-python": 0.65, "fineweb-edu": 0.20,
                          "finemath-3plus": 0.15}
    train_root = tmp_path / "mix" / "train"
    holdout_root = tmp_path / "mix" / "holdout"
    for name in weights:
        _shard_dir(train_root, name, tokens=(tokens or {}).get(name, 400_000_000))
    _shard_dir(holdout_root, "code-python", tokens=2_000_000)
    record = {"schema": 1, "weights": weights,
              "train_root": str(train_root), "holdout_root": str(holdout_root),
              "corpus_shares": dict(CP.__dict__.get("CORPUS_SHARES", {}))}
    path = tmp_path / "train-mixture.json"
    path.write_text(json.dumps(record))
    return {"path": str(path), "record": record}


def _base_checkpoint(tmp_path, body=b"released base weights") -> str:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(body)
    return str(path)


def test_the_arms_are_the_rates_this_gate_preregistered():
    arms = CP.probe_arms()

    assert [arm.muon_lr for arm in arms] == [5e-4, 1e-3, 2e-3]
    assert [arm.name for arm in arms] == ["code-probe-lr0.0005",
                                          "code-probe-lr0.001",
                                          "code-probe-lr0.002"]
    assert {arm.total_tokens for arm in arms} == {250_000_000}


def test_adam_is_scaled_with_muon_rather_than_left_where_it_was():
    """Muon and AdamW cover disjoint parameter sets -- the hidden matrices
    against the embeddings, norms and gains -- so moving one alone changes which
    half of the model trains rather than lowering "the" learning rate."""
    from scripts.qat_recovery import SHIPPED_ADAM_LR, SHIPPED_MUON_LR

    ratio = SHIPPED_ADAM_LR / SHIPPED_MUON_LR
    for arm in CP.probe_arms():
        assert arm.adam_lr == pytest.approx(arm.muon_lr * ratio)


def test_the_arms_differ_in_exactly_the_learning_rate(tmp_path):
    """"Identical data, order and seed" is a claim three shell lines cannot
    support and a diff of three argvs can."""
    built = _record(tmp_path)
    base = _base_checkpoint(tmp_path)
    commands = [CP.train_command(arm, init_from=base, record=built["record"])
                for arm in CP.probe_arms()]

    for other in commands[1:]:
        differing = {a for a, b in zip(commands[0], other) if a != b}
        assert len(commands[0]) == len(other)
        # The run name and the two rates, and nothing else.
        assert len(differing) == 3
        assert any(part.startswith("code-probe-lr") for part in differing)


def test_an_arm_starts_from_the_base_weights_and_never_resumes(tmp_path):
    """`--resume` on attempt one would restore the finished pretraining run's
    step and token count: the probe writes no metrics row and exits 0, which
    from outside looks like an arm that finished early."""
    built = _record(tmp_path)
    base = _base_checkpoint(tmp_path)

    command = CP.train_command(CP.probe_arms()[0], init_from=base,
                               record=built["record"])

    assert "--resume" not in command
    assert command[command.index("--init-from") + 1] == base
    CP.assert_no_resume(command)


def test_an_arm_reads_the_composed_mixture_rather_than_flags_typed_again(
        tmp_path):
    from train import parse_mixture_weights

    built = _record(tmp_path)
    command = CP.train_command(CP.probe_arms()[0],
                               init_from=_base_checkpoint(tmp_path),
                               record=built["record"])

    assert command[command.index("--data-dir") + 1] == built["record"]["train_root"]
    assert command[command.index("--val-dir") + 1] == built["record"]["holdout_root"]
    pairs = [command[i + 1] for i, part in enumerate(command)
             if part == "--mixture-weight"]
    assert parse_mixture_weights(pairs) == pytest.approx(
        built["record"]["weights"])


def test_the_whole_budget_is_spent_decaying_to_zero(tmp_path):
    """A probe scored mid-schedule is not a fully decayed probe, and "fully
    decayed" is what the gate compares."""
    from scripts.qat_recovery import DECAY_FRAC

    command = CP.train_command(CP.probe_arms()[0],
                               init_from=_base_checkpoint(tmp_path),
                               record=_record(tmp_path)["record"])

    assert command[command.index("--decay-frac") + 1] == str(DECAY_FRAC)
    warmup = int(command[command.index("--warmup-steps") + 1])
    # train.py's 300-step default is longer than a third of this budget, which
    # would leave the arm in warmup for most of it.
    assert 10 <= warmup < 100


def test_a_mixture_root_that_lost_a_source_is_refused(tmp_path):
    """`resolve_mixture` renormalizes over what it finds, so a root missing half
    its sources trains a healthy-looking arm on a mixture nothing describes."""
    built = _record(tmp_path)
    os.remove(os.path.join(built["record"]["train_root"], "fineweb-edu",
                           "manifest.json"))

    with pytest.raises(ValueError, match="missing 1 of the mixture's 3"):
        CP.load_mixture(built["path"])


def test_a_record_that_was_never_composed_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"schema": 1}))

    with pytest.raises(ValueError, match="corpus mixture"):
        CP.load_mixture(str(path))


def test_a_base_checkpoint_that_is_not_the_pinned_one_is_refused(tmp_path):
    """Every arm's result is a difference against this file. A substituted one
    still trains three arms that still compare against each other."""
    base = _base_checkpoint(tmp_path)
    digest = CP.sha256_of(base)

    assert CP.assert_base_checkpoint(base, digest) == digest
    with pytest.raises(ValueError, match="not the pinned"):
        CP.assert_base_checkpoint(base, "0" * 64)
    with pytest.raises(ValueError, match="no base checkpoint"):
        CP.assert_base_checkpoint(str(tmp_path / "gone.pt"), None)


def _metrics(run_dir, tokens):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text("".join(
        json.dumps({"step": i, "tokens": t, "loss": 3.5 - i * 0.01}) + "\n"
        for i, t in enumerate(tokens)))


def test_completion_is_read_off_the_metrics_not_off_the_checkpoint(tmp_path):
    """The checkpoint is written throughout a run, so its existence says "this
    can be resumed", not "this is done" -- and a sweep that skipped on it would
    skip the arm it was meant to resume."""
    arm = CP.probe_arms()[0]
    run_dir = tmp_path / arm.name
    _metrics(run_dir, [1_000_000, 60_300_000])
    (run_dir / "checkpoint.pt").write_bytes(b"600MB, notionally")

    assert CP.arm_is_complete(run_dir, arm.total_tokens) is False

    _metrics(run_dir, [1_000_000, 60_300_000, arm.total_tokens])
    assert CP.arm_is_complete(run_dir, arm.total_tokens) is True


def test_a_torn_last_metrics_line_is_not_a_verdict(tmp_path):
    arm = CP.probe_arms()[0]
    run_dir = tmp_path / arm.name
    _metrics(run_dir, [arm.total_tokens])
    with (run_dir / "metrics.jsonl").open("a") as handle:
        handle.write('{"step": 99, "tok')

    assert CP.arm_is_complete(run_dir, arm.total_tokens) is True


def test_the_in_flight_marker_says_which_phase_is_resuming(tmp_path,
                                                           monkeypatch):
    """`boot_resume.py` continues a run from that marker after a reboot and the
    keeper reads it to know the box is busy, so the phase name in it is
    provenance. `qat_recovery.launch_supervised` is otherwise exactly this
    function and hardcodes phase 3's name."""
    import daedalus.supervise as supervise

    seen = {}

    def fake_run_with_resume(command, checkpoint, **kwargs):
        seen.update(kwargs)
        seen["checkpoint"] = checkpoint
        return {"attempts": 1, "returncodes": [0]}

    monkeypatch.setattr(supervise, "run_with_resume", fake_run_with_resume)
    monkeypatch.setattr(supervise, "start_watchdog",
                        lambda *a, **k: "watchdog")
    monkeypatch.setattr(supervise, "stop_watchdog", lambda handle: None)
    arm = CP.probe_arms()[0]

    CP.launch_supervised(arm, ["python", "train.py"], run_root=str(tmp_path))

    assert seen["inflight_extra"]["phase"] == "phase8-code-probes"
    assert seen["inflight_extra"]["arm"] == arm.name
    assert seen["inflight_extra"]["total_tokens"] == arm.total_tokens
    # Beside the arm's own checkpoint, and with the halt marker that makes a
    # watchdog stop stick.
    assert seen["checkpoint"] == str(tmp_path / arm.name / "checkpoint.pt")
    assert seen["halt_marker"] == str(tmp_path / arm.name / "HALTED")


def test_a_supervised_arm_refuses_a_command_carrying_resume(tmp_path):
    with pytest.raises(ValueError, match="--init-from"):
        CP.launch_supervised(CP.probe_arms()[0],
                             ["python", "train.py", "--resume", "x.pt"],
                             run_root=str(tmp_path))


def test_the_summary_reports_the_validation_that_happened_not_the_last_row(
        tmp_path):
    """Validation is periodic and the final step is rarely a multiple of the
    interval, so reading `val_bpb` off the last row reports null for every arm
    that ran -- which reads as validation being broken rather than as it having
    last run 200 steps ago."""
    arm = CP.probe_arms()[0]
    run_dir = tmp_path / arm.name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"step": 250, "tokens": 131_072_000, "loss": 2.0, "val_bpb": 0.65,
         "val_bpb_per_source": {"code-python": 0.53}},
        {"step": 477, "tokens": arm.total_tokens, "loss": 1.9, "val_bpb": None},
    ]))

    summary = CP.arm_summary(arm, run_dir)

    assert summary["step"] == 477 and summary["tokens"] == arm.total_tokens
    assert summary["val_bpb"] == 0.65
    # Stamped with where it came from, so it cannot be read as end-of-run.
    assert summary["val_bpb_step"] == 250
    assert summary["val_bpb_per_source"] == {"code-python": 0.53}


def _fake_launcher(monkeypatch, launched, *, fail=(), run_root=None):
    def launch(arm, command, **kwargs):
        launched.append(arm.name)
        if arm.name in fail:
            raise RuntimeError(f"{arm.name} died")
        if run_root is not None:
            _metrics(run_root / arm.name, [arm.total_tokens])
        return {"attempts": 1, "returncodes": [0]}

    monkeypatch.setattr(CP, "launch_supervised", launch)
    return launched


def test_a_relaunched_sweep_resumes_instead_of_restarting(tmp_path,
                                                          monkeypatch):
    """The failure this closes cost phase 4 an arm: a session exited mid-sweep,
    the next one restarted the arm from scratch beside a checkpoint it never
    read."""
    built = _record(tmp_path)
    run_root = tmp_path / "runs"
    arms = CP.probe_arms()
    _metrics(run_root / arms[0].name, [arms[0].total_tokens])
    launched = _fake_launcher(monkeypatch, [], run_root=run_root)

    report = CP.sweep(init_from=_base_checkpoint(tmp_path),
                      mixture_record=built["path"], run_root=str(run_root),
                      json_out=str(tmp_path / "probes.json"))

    assert launched == [arms[1].name, arms[2].name]
    assert report["arms"][0]["skipped"] is True
    assert report["arms"][0]["summary"]["complete"] is True


def test_one_arm_failing_does_not_lose_the_arms_beside_it(tmp_path,
                                                          monkeypatch):
    built = _record(tmp_path)
    run_root = tmp_path / "runs"
    arms = CP.probe_arms()
    _fake_launcher(monkeypatch, [], fail={arms[1].name}, run_root=run_root)

    report = CP.sweep(init_from=_base_checkpoint(tmp_path),
                      mixture_record=built["path"], run_root=str(run_root),
                      json_out=str(tmp_path / "probes.json"))

    assert "error" in report["arms"][1] and "died" in report["arms"][1]["error"]
    assert report["arms"][0]["summary"]["complete"] is True
    assert report["arms"][2]["summary"]["complete"] is True


def test_the_report_is_on_disk_before_the_last_arm_finishes(tmp_path,
                                                            monkeypatch):
    """A sweep that dies in arm three must not take arms one and two with it."""
    built = _record(tmp_path)
    run_root = tmp_path / "runs"
    out = tmp_path / "probes.json"
    arms = CP.probe_arms()
    seen = []

    def launch(arm, command, **kwargs):
        seen.append(json.loads(out.read_text()))
        _metrics(run_root / arm.name, [arm.total_tokens])
        return {"attempts": 1, "returncodes": [0]}

    monkeypatch.setattr(CP, "launch_supervised", launch)
    CP.sweep(init_from=_base_checkpoint(tmp_path), mixture_record=built["path"],
             run_root=str(run_root), json_out=str(out))

    # Written before the first arm ran, and one arm longer each time after.
    assert [len(record["arms"]) for record in seen] == [0, 1, 2]
    assert seen[0]["init_from"]["sha256"]
    assert len(seen[-1]["arms"]) == len(arms) - 1


def test_a_mixture_that_is_a_different_experiment_at_the_budget_is_refused(
        tmp_path, monkeypatch):
    """The epoch cap moves shares. A mixture the corpus cannot hold at 250M is
    not the mixture that was composed, and an hour and a half of GPU should not
    go into finding that out."""
    built = _record(tmp_path, tokens={"code-python": 400_000,
                                      "fineweb-edu": 400_000,
                                      "finemath-3plus": 400_000_000})
    launched = _fake_launcher(monkeypatch, [])

    with pytest.raises(ValueError, match="past the 5-point limit"):
        CP.sweep(init_from=_base_checkpoint(tmp_path),
                 mixture_record=built["path"], run_root=str(tmp_path / "runs"),
                 json_out=str(tmp_path / "probes.json"))
    assert launched == []


def test_a_smokes_report_does_not_answer_for_the_gate(tmp_path, monkeypatch):
    """The report is what a scorer reads. Naming the gate on a four-step run is
    how a smoke ends up answering "did the 250M probes run"."""
    built = _record(tmp_path)
    run_root = tmp_path / "runs"
    _fake_launcher(monkeypatch, [], run_root=run_root)

    smoke = CP.sweep(init_from=_base_checkpoint(tmp_path),
                     mixture_record=built["path"], run_root=str(run_root),
                     arms=CP.probe_arms(tag="smoke", steps=4), json_out=None)
    gate = CP.sweep(init_from=_base_checkpoint(tmp_path),
                    mixture_record=built["path"], run_root=str(run_root),
                    json_out=None)

    assert smoke["gate"] is None and smoke["smoke"] is True
    assert gate["gate"] == "probes_250m" and gate["smoke"] is False


def test_a_smoke_runs_under_its_own_name_and_budget(tmp_path):
    arms = CP.probe_arms(tag="smoke", steps=4)

    assert [arm.name for arm in arms] == ["code-smoke-lr0.0005",
                                          "code-smoke-lr0.001",
                                          "code-smoke-lr0.002"]
    assert {arm.total_tokens for arm in arms} == {4 * CP.BATCH_TOKENS}
    # And the gate's own arms are untouched by the existence of a smoke.
    assert {arm.total_tokens for arm in CP.probe_arms()} == {CP.PROBE_TOKENS}


def test_a_shortened_run_may_not_wear_the_gates_name(tmp_path, capsys):
    """Under the gate's name a shortened arm lands in the gate's run directory,
    and the next sweep either resumes it as the real arm or reads it as one that
    finished at a budget nobody chose."""
    built = _record(tmp_path)

    rc = CP._cli(["sweep", "--init-from", _base_checkpoint(tmp_path),
                  "--mixture-record", built["path"], "--steps", "4",
                  "--run-root", str(tmp_path / "runs"),
                  "--json-out", str(tmp_path / "probes.json")])

    assert rc == 2
    assert "needs its own --tag" in capsys.readouterr().err
    assert not (tmp_path / "probes.json").exists()


def test_the_cli_prints_one_arms_command(tmp_path, capsys):
    built = _record(tmp_path)

    rc = CP._cli(["command", "--arm", "code-probe-lr0.001",
                  "--init-from", _base_checkpoint(tmp_path),
                  "--mixture-record", built["path"]])

    printed = capsys.readouterr().out
    assert rc == 0
    assert "--muon-lr 0.001" in printed and "--init-from" in printed


def test_the_cli_refuses_an_arm_this_gate_does_not_have(tmp_path, capsys):
    built = _record(tmp_path)

    rc = CP._cli(["command", "--arm", "code-probe-lr0.005",
                  "--init-from", _base_checkpoint(tmp_path),
                  "--mixture-record", built["path"]])

    assert rc == 2
    assert "no arm" in capsys.readouterr().err


def test_the_cli_fails_when_an_arm_did_not_reach_its_budget(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """A sweep whose arms exited 0 without training their budget must not read
    as a passing gate: the verdict downstream is a comparison of three numbers
    that would then be three different amounts of training."""
    built = _record(tmp_path)
    run_root = tmp_path / "runs"

    def launch(arm, command, **kwargs):
        _metrics(run_root / arm.name, [arm.total_tokens // 2])
        return {"attempts": 1, "returncodes": [0]}

    monkeypatch.setattr(CP, "launch_supervised", launch)

    rc = CP._cli(["sweep", "--init-from", _base_checkpoint(tmp_path),
                  "--mixture-record", built["path"],
                  "--run-root", str(run_root),
                  "--json-out", str(tmp_path / "probes.json")])

    assert rc == 3
    assert "did not reach its budget" in capsys.readouterr().err
