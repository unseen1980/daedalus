"""The staged 2B extension: what it refuses, and what it hands the controller.

Every mistake available here produces a run that trains, exits 0 and reports a
number nobody can interpret: launched on the 250M gate's authority instead of the
1B one, started from the released base instead of the branch's weights, extending
a first stage that never finished, at a rate nobody chose, over a schedule sized
for someone else's budget, drawing a replay bucket the epoch cap thinned, or
started so late it cannot finish before finalization. So the subject here is the
refusals, and the two agreements a supervised launch depends on -- argv against
the checkpoint it is told to resume, and the plan against what actually happened
in the stage before it.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import code_branch as CB
from scripts import code_extension as CE
from scripts import code_probes as CP


# ----------------------------------------------------------------- fixtures ---

def _shard_dir(root, name, tokens=8_000_000_000):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"total_tokens": tokens, "shards": [], "eos_id": 0}))
    return path


def _record(tmp_path, *, tokens=None, weights=None, buckets=None,
            corpus_shares=None) -> dict:
    """A composed mixture shaped like the real one: buckets over weights.

    The default supply is large enough that 2B tokens fit inside the epoch cap
    for every source, so a test that wants a cap has to ask for one.
    """
    weights = weights or {"code-python": 0.65,
                          "fineweb-edu": 0.15, "dclm-baseline": 0.05,
                          "finemath-3plus": 0.15}
    buckets = buckets if buckets is not None else {
        "code": {"code-python": 1.0},
        "general-replay": {"fineweb-edu": 0.75, "dclm-baseline": 0.25},
        "technical": {"finemath-3plus": 1.0},
    }
    train_root = tmp_path / "mix" / "train"
    holdout_root = tmp_path / "mix" / "holdout"
    for name in weights:
        _shard_dir(train_root, name,
                   tokens=(tokens or {}).get(name, 8_000_000_000))
    _shard_dir(holdout_root, "code-python", tokens=2_000_000)
    record = {"schema": 1, "weights": weights, "buckets": buckets,
              "train_root": str(train_root), "holdout_root": str(holdout_root)}
    if corpus_shares is not None:
        record["corpus_shares"] = corpus_shares
    path = tmp_path / "train-mixture.json"
    path.write_text(json.dumps(record))
    return {"path": str(path), "record": record}


def _branch_run(tmp_path, *, name=CB.BRANCH_NAME, tokens=CB.BRANCH_TOKENS,
                lr=1e-3, rate=45_000.0, rows=4, body=b"1B branch weights"):
    """A finished branch run: its checkpoint, and metrics that carry its peak."""
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(rows):
        # Row 0 sits inside warmup, so the peak is not simply the first row.
        scheduled = lr * (0.4 if index == 0 else 1.0)
        lines.append(json.dumps({
            "step": 100 * (index + 1),
            "tokens": tokens // rows * (index + 1),
            "loss": 1.4, "lr": scheduled, "tok_per_sec": rate,
            "elapsed_h": 1.5 * (index + 1)}))
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    (run_dir / "checkpoint.pt").write_bytes(body)
    return str(run_dir / "checkpoint.pt")


def _branch_verdict(tmp_path, *, proceed=True, branch=CB.BRANCH_NAME,
                    gate="branch_1b", reason="5 of 5 clauses passed",
                    name="branch-1b-verdict.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps({
        "schema": 1, "continue": proceed, "reason": reason,
        "gate": {"gate": gate, "branch": branch, "continue": proceed,
                 "reason": reason}}))
    return str(path)


def _probe_verdict(tmp_path, *, selected="code-probe-lr0.001",
                   name="verdict.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps({
        "schema": 1, "continue": True, "selected": selected,
        "reason": f"{selected} is the lowest-code-BPB qualifying arm",
        "gate": {"gate": "probes_250m", "continue": True}}))
    return str(path)


def _state(tmp_path, *, hours_elapsed=64.0, hard_hours=144.0,
           reserve_hours=8.0, now=None) -> str:
    """A controller state whose deadline leaves `144 - 8 - hours_elapsed` hours."""
    now = now or datetime(2026, 8, 27, 3, 0, tzinfo=timezone.utc)
    started = now - timedelta(hours=hours_elapsed)
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "schema": 1, "phase": "phase8-extension-2b", "status": "running",
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "hard_hours": hard_hours, "reserve_hours": reserve_hours}))
    return str(path)


_NOW = datetime(2026, 8, 27, 3, 0, tzinfo=timezone.utc)


def _plan(tmp_path, monkeypatch, **kwargs):
    """A plan built with the trainer's own `runs/` under `tmp_path`."""
    built = _record(tmp_path)
    monkeypatch.chdir(tmp_path)
    defaults = dict(branch_verdict_path=_branch_verdict(tmp_path),
                    probe_verdict_path=_probe_verdict(tmp_path),
                    init_from=_branch_run(tmp_path),
                    mixture_record=built["path"], run_root="runs",
                    state_path=_state(tmp_path), now=_NOW)
    defaults.update(kwargs)
    return CE.extension_plan(**defaults)


# ------------------------------------------------------- the gate's answer ---

def test_the_250m_probe_verdict_does_not_authorise_the_extension(tmp_path):
    """Both verdicts live in one directory and both carry a passing `continue`.
    The probe gate authorised 1B tokens; reading it here would spend 2B more on
    an authority that was never given."""
    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.load_branch_verdict(_probe_verdict(tmp_path))

    assert "is a 'probes_250m' verdict" in str(excinfo.value)
    assert "branch_1b" in str(excinfo.value)


def test_a_branch_gate_that_said_stop_cannot_be_launched_anyway(tmp_path):
    """The plan's degradation policy names stopping Daedalus-Code at 1B as an
    acceptable outcome and an unfinishable extension as the expensive one."""
    path = _branch_verdict(tmp_path, proceed=False,
                           reason="general-bpb, retrieval did not pass")

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.load_branch_verdict(path)

    assert "says stop" in str(excinfo.value)
    assert "general-bpb, retrieval" in str(excinfo.value)


def test_a_verdict_about_another_model_does_not_authorise_this_one(tmp_path):
    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.load_branch_verdict(_branch_verdict(tmp_path, branch="code-probe-lr0.001"))

    assert "scored 'code-probe-lr0.001'" in str(excinfo.value)


def test_a_passing_branch_verdict_is_accepted(tmp_path):
    verdict = CE.load_branch_verdict(_branch_verdict(tmp_path))

    assert verdict["continue"] is True


# ----------------------------------------------------------------- the rate ---

def test_the_rate_is_half_what_the_branch_actually_ran_at(tmp_path):
    """Read off the branch's own metrics rather than re-derived from a document:
    those are two different claims, and only this one survives a hand relaunch."""
    checkpoint = _branch_run(tmp_path, lr=1e-3)

    rate = CE.extension_rate(branch_run_dir=os.path.dirname(checkpoint),
                             probe_verdict_path=_probe_verdict(tmp_path))

    assert rate["branch_muon_lr"] == pytest.approx(1e-3)
    assert rate["muon_lr"] == pytest.approx(5e-4)
    assert rate["frac"] == CE.EXTENSION_LR_FRAC == 0.5


def test_the_peak_is_the_top_of_the_schedule_not_the_first_window(tmp_path):
    """A run's first windows are inside warmup. Taking one would extend at half
    of a rate the branch only passed through."""
    checkpoint = _branch_run(tmp_path, lr=2e-3)

    assert CE.branch_peak_lr(os.path.dirname(checkpoint)) == pytest.approx(2e-3)


def test_a_branch_and_a_verdict_that_disagree_on_the_rate_are_refused(tmp_path):
    """Either source alone can be wrong invisibly: a verdict is a file that can
    be rewritten, and metrics can carry a rate somebody launched by hand."""
    checkpoint = _branch_run(tmp_path, lr=2e-3)

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.extension_rate(branch_run_dir=os.path.dirname(checkpoint),
                          probe_verdict_path=_probe_verdict(
                              tmp_path, selected="code-probe-lr0.001"))

    assert "does not describe the run this extends" in str(excinfo.value)


def test_a_branch_with_no_rate_rows_refuses_rather_than_guessing(tmp_path):
    run_dir = tmp_path / "runs" / CB.BRANCH_NAME
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "tokens": 1, "lr": None}) + "\n")

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.extension_rate(branch_run_dir=run_dir,
                          probe_verdict_path=_probe_verdict(tmp_path))

    assert "does not guess a rate" in str(excinfo.value)


def test_the_rate_is_lower_than_the_branchs_by_construction(tmp_path):
    """The branch arrives from a fully decayed schedule. Reopening at its peak
    re-injects exactly the update size that decay removed."""
    checkpoint = _branch_run(tmp_path, lr=1e-3)
    rate = CE.extension_rate(branch_run_dir=os.path.dirname(checkpoint),
                             probe_verdict_path=_probe_verdict(tmp_path))
    arm = CE.extension_arm(rate)

    assert 0 < arm.muon_lr < rate["branch_muon_lr"]
    # Adam follows through the shipped ratio: the two optimizers cover disjoint
    # parameters, so scaling one alone changes which half of the model moves.
    from scripts.qat_recovery import adam_lr_for
    assert arm.adam_lr == pytest.approx(adam_lr_for(arm.muon_lr))


# -------------------------------------------------- the weights it continues ---

def test_starting_from_the_released_base_is_refused_by_hash(tmp_path):
    """The mirror of gate 2's refusal. From the base this trains fine, exits 0,
    and is a fresh 2B run reported as staged 3B adaptation."""
    base = tmp_path / "hero" / "checkpoint.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"released base weights")

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.refuse_base_checkpoint(base, base_sha256=CP.sha256_of(base))

    assert "released base checkpoint" in str(excinfo.value)
    assert "fresh 2B run" in str(excinfo.value)


def test_the_branchs_own_checkpoint_is_accepted_and_hashed(tmp_path):
    checkpoint = _branch_run(tmp_path)

    digest = CE.refuse_base_checkpoint(checkpoint)

    assert digest == CP.sha256_of(checkpoint)


def test_a_missing_checkpoint_is_named_rather_than_hashed(tmp_path):
    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.refuse_base_checkpoint(tmp_path / "nowhere" / "checkpoint.pt")

    assert "no checkpoint at" in str(excinfo.value)


def test_a_probe_arms_checkpoint_is_still_refused(tmp_path, monkeypatch):
    """Gate 2's refusal has to keep holding here: this stage continues the 1B
    branch, and a 250M arm is neither the base nor the branch."""
    monkeypatch.chdir(tmp_path)
    arm = CP.probe_arms()[1]
    checkpoint = _branch_run(tmp_path, name=arm.name,
                             tokens=CP.PROBE_TOKENS, body=b"250M in")

    with pytest.raises(CB.BranchRefused):
        CE.extension_plan(branch_verdict_path=_branch_verdict(tmp_path),
                          probe_verdict_path=_probe_verdict(tmp_path),
                          init_from=checkpoint,
                          mixture_record=_record(tmp_path)["path"],
                          run_root="runs", state_path=_state(tmp_path),
                          now=_NOW)


def test_an_unfinished_branch_cannot_be_extended(tmp_path):
    """"Staged 1B -> 2B" is a claim about both stages, and gate 2's numbers were
    taken on the finished one."""
    checkpoint = _branch_run(tmp_path, tokens=400_000_000)

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.assert_branch_complete(os.path.dirname(checkpoint))

    assert "400,000,000 of 1,000,000,000" in str(excinfo.value)
    assert "2,400,000,000" in str(excinfo.value)


def test_a_finished_branch_reports_the_tokens_it_trained(tmp_path):
    checkpoint = _branch_run(tmp_path)

    assert CE.assert_branch_complete(
        os.path.dirname(checkpoint)) == CB.BRANCH_TOKENS


# ------------------------------------------------------------- the schedule ---

def test_the_warmup_is_recomputed_for_this_stages_budget(tmp_path):
    """Inheriting the branch's schedule would warm a 2B stage over a ramp sized
    for 1B, and `--resume` would restore a finished decay."""
    from scripts.qat_recovery import WARMUP_FRAC, estimated_steps

    checkpoint = _branch_run(tmp_path)
    rate = CE.extension_rate(branch_run_dir=os.path.dirname(checkpoint),
                             probe_verdict_path=_probe_verdict(tmp_path))
    arm = CE.extension_arm(rate)

    assert arm.total_tokens == CE.EXTENSION_TOKENS == 2_000_000_000
    assert arm.warmup_steps == int(
        estimated_steps(CE.EXTENSION_TOKENS) * WARMUP_FRAC)
    branch_warmup = CP.CodeProbe(name=CB.BRANCH_NAME, muon_lr=1e-3,
                                 total_tokens=CB.BRANCH_TOKENS).warmup_steps
    assert arm.warmup_steps == pytest.approx(branch_warmup * 2, rel=0.05)


def test_the_command_carries_this_stages_budget_name_and_no_resume(
        tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    command = plan["command"]

    assert "--resume" not in command
    assert command[command.index("--total-tokens") + 1] == "2000000000"
    assert command[command.index("--run-name") + 1] == CE.EXTENSION_NAME
    assert command[command.index("--init-from") + 1] == plan["init_from"]["path"]
    assert command[command.index("--muon-lr") + 1] == "0.0005"


def test_the_supervised_checkpoint_is_the_one_this_argv_writes(tmp_path,
                                                               monkeypatch):
    """A marker beside a file that never appears means every relaunch restarts
    from step zero and nothing reports a problem."""
    from scripts.vast_program import trainer_checkpoint_for

    plan = _plan(tmp_path, monkeypatch)

    assert (os.path.abspath(trainer_checkpoint_for(plan["command"]))
            == os.path.abspath(plan["supervise_checkpoint"]))
    assert CE.EXTENSION_NAME in plan["supervise_checkpoint"]


def test_this_stage_is_named_apart_from_the_branch_it_continues(tmp_path,
                                                                monkeypatch):
    """Under the branch's name a relaunch would reopen the finished 1B run,
    read its budget as met, and overwrite the checkpoint this starts from."""
    plan = _plan(tmp_path, monkeypatch)

    assert CE.EXTENSION_NAME != CB.BRANCH_NAME
    assert os.path.realpath(plan["run_dir"]) != os.path.realpath(
        os.path.dirname(plan["init_from"]["path"]))


def test_a_stage_that_would_write_over_its_own_input_is_refused(tmp_path,
                                                                monkeypatch):
    """The one arrangement that survives the name check: continuing from a
    checkpoint that already lives in this stage's own run directory."""
    monkeypatch.chdir(tmp_path)
    checkpoint = _branch_run(tmp_path, name=CE.EXTENSION_NAME,
                             tokens=CB.BRANCH_TOKENS)

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.extension_plan(branch_verdict_path=_branch_verdict(tmp_path),
                          probe_verdict_path=_probe_verdict(tmp_path),
                          init_from=checkpoint,
                          mixture_record=_record(tmp_path)["path"],
                          run_root="runs", state_path=_state(tmp_path),
                          now=_NOW)

    assert "own run directory" in str(excinfo.value)


def test_a_stage_that_already_trained_its_budget_is_not_relaunched(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / CE.EXTENSION_NAME
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 3815, "tokens": 2_000_000_000}) + "\n")

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        _plan(tmp_path, monkeypatch)

    assert "already trained" in str(excinfo.value)


# -------------------------------------------------------- the replay floor ---

def test_the_replay_floor_is_measured_at_this_stages_budget(tmp_path,
                                                             monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    floor = plan["mixture"]["replay_floor"]

    assert plan["mixture"]["preflight"]["total_run_tokens"] == 2_000_000_000
    assert floor["target_pts"] == pytest.approx(20.0)
    assert floor["effective_pts"] == pytest.approx(20.0)
    # A twenty-point bucket's pro-rata share of the mixture's own L1 budget.
    assert floor["tolerance_pts"] == pytest.approx(1.0)


def test_a_bucket_tolerance_is_tighter_than_the_aggregate_it_comes_from():
    """A shortfall the cap takes out of one bucket shows up twice in the
    aggregate L1 skew, so a per-bucket bar set at the aggregate 5 points could
    never fire before the mixture check -- it would not be a gate at all."""
    assert CE.replay_tolerance_pts(20.0) == pytest.approx(1.0)
    assert CE.replay_tolerance_pts(20.0) < CP.MAX_L1_SKEW_PTS / 2
    # The allowances are the aggregate budget distributed pro rata: three
    # buckets covering the mixture sum back to it.
    assert (CE.replay_tolerance_pts(65.0) + CE.replay_tolerance_pts(20.0)
            + CE.replay_tolerance_pts(15.0)) == pytest.approx(CP.MAX_L1_SKEW_PTS)


def test_a_replay_bucket_the_epoch_cap_thins_past_its_bar_is_refused(
        tmp_path, monkeypatch):
    """The replay sources are the small ones, so a larger draw caps them first
    and the aggregate skew can stay comfortable while replay alone thins."""
    # 15% of 2B is 300M tokens; 67M on disk is 4.5 epochs, so the cap holds this
    # source to 268M and takes ~1.6 points out of the replay bucket -- while the
    # aggregate L1 skew, which counts the loss and its redistribution, stays at
    # ~3 of its 5 permitted points.
    built = _record(tmp_path, tokens={"fineweb-edu": 67_000_000})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.extension_plan(branch_verdict_path=_branch_verdict(tmp_path),
                          probe_verdict_path=_probe_verdict(tmp_path),
                          init_from=_branch_run(tmp_path),
                          mixture_record=built["path"], run_root="runs",
                          state_path=_state(tmp_path), now=_NOW)

    message = str(excinfo.value)
    assert "replay bucket" in message and "fineweb-edu" in message
    assert "different experiment" in message


def test_a_record_whose_buckets_and_weights_disagree_is_refused(tmp_path):
    """A record that declares a 20-point replay bucket whose weights sum to 12
    would train at 12 while this stage reported holding the branch's floor."""
    built = _record(tmp_path, corpus_shares={"code": 0.65, "general-replay": 0.20,
                                             "technical": 0.15},
                    weights={"code-python": 0.73, "fineweb-edu": 0.09,
                             "dclm-baseline": 0.03, "finemath-3plus": 0.15})
    preflight = {"total_run_tokens": 2_000_000_000,
                 "per_source": {name: {"effective_share": share, "capped": False}
                                for name, share in built["record"]["weights"].items()}}

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.replay_floor(built["record"], preflight)

    assert "not the floor the record describes" in str(excinfo.value)


def test_a_record_without_buckets_cannot_state_a_floor(tmp_path):
    built = _record(tmp_path, buckets={})

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.replay_floor(built["record"], {"total_run_tokens": 1,
                                          "per_source": {}})

    assert "replay floor" in str(excinfo.value)


def test_a_replay_source_the_sampler_never_resolved_is_not_scored_around(
        tmp_path):
    """Summing over the sources that happen to be in the preflight is how "the
    same replay floor" quietly becomes "the floor over what resolved"."""
    built = _record(tmp_path)
    preflight = {"total_run_tokens": 2_000_000_000,
                 "per_source": {"fineweb-edu": {"effective_share": 0.15,
                                                "capped": False}}}

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.replay_floor(built["record"], preflight)

    assert "dclm-baseline" in str(excinfo.value)


# ---------------------------------------------------------- the projection ---

def test_the_projection_comes_from_the_branchs_own_throughput(tmp_path):
    """The branch is the same model, mixture, batch shape and box, and it has
    just run for hours -- a better estimate than the 250M probes'."""
    checkpoint = _branch_run(tmp_path, rate=40_000.0)

    projection = CE.projected_hours(branch_run_dir=os.path.dirname(checkpoint))

    assert projection["tok_per_sec"] == 40_000.0
    assert projection["hours"] == pytest.approx(
        2_000_000_000 / 40_000 / 3600 * CE.HOURS_MARGIN)


def test_a_branch_with_no_throughput_rows_refuses_rather_than_guessing(tmp_path):
    run_dir = tmp_path / "runs" / CB.BRANCH_NAME
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "tokens": 1, "tok_per_sec": None}) + "\n")

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.projected_hours(branch_run_dir=run_dir)

    assert "--estimated-hours" in str(excinfo.value)


def test_an_explicit_estimate_skips_the_projection(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch, estimated_hours=13.0)

    assert plan["estimated_hours"] == 13.0
    assert plan["projection"] is None


# ------------------------------------------------------------ the deadline ---

def test_a_stage_that_cannot_finish_before_finalization_is_refused(tmp_path):
    """The manifest makes this an only_if, so it is answered before anything is
    spawned rather than inside a detached child's log."""
    state = _state(tmp_path, hours_elapsed=130.0)

    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.assert_fits_deadline(14.0, state_path=state, now=_NOW)

    assert "does not fit" in str(excinfo.value)
    assert "stop Daedalus-Code at 1B" in str(excinfo.value)


def test_a_stage_that_fits_records_what_it_had_left(tmp_path):
    verdict = CE.assert_fits_deadline(
        14.0, state_path=_state(tmp_path, hours_elapsed=64.0), now=_NOW)

    assert verdict["fits"] is True
    assert verdict["stage"] == "active"
    assert verdict["hours_to_finalization"] == pytest.approx(72.0)


def test_the_deadline_uses_the_controllers_own_reserve(tmp_path):
    """Not a second implementation of the program's clock: the same reserve the
    controller refuses phases with, read from the same state file."""
    state = _state(tmp_path, hours_elapsed=100.0, reserve_hours=8.0)

    verdict = CE.assert_fits_deadline(30.0, state_path=state, now=_NOW)

    # 144 - 8 - 100 = 36 hours to finalization, so a 30h stage fits and a 40h
    # one does not.
    assert verdict["hours_to_finalization"] == pytest.approx(36.0)
    with pytest.raises(CE.ExtensionRefused):
        CE.assert_fits_deadline(40.0, state_path=state, now=_NOW)


def test_a_missing_state_file_refuses_rather_than_assuming_time(tmp_path):
    with pytest.raises(CE.ExtensionRefused) as excinfo:
        CE.assert_fits_deadline(1.0, state_path=str(tmp_path / "nowhere.json"),
                                now=_NOW)

    assert "not answerable without the deadline" in str(excinfo.value)


# ---------------------------------------------------------------- reporting ---

def test_the_plan_records_it_as_staged_adaptation(tmp_path, monkeypatch):
    """The manifest's report_as, carried in the launch document so the report is
    written from a record that already says what this was."""
    plan = _plan(tmp_path, monkeypatch)
    staged = plan["staged"]

    assert staged["stage"] == 2 and staged["of"] == 2
    assert staged["previous"]["run"] == CB.BRANCH_NAME
    assert staged["previous"]["tokens"] == 1_000_000_000
    assert staged["this_stage_tokens"] == 2_000_000_000
    assert staged["cumulative_tokens"] == 3_000_000_000
    assert "not one uninterrupted 3B run" in staged["report_as"]


# ----------------------------------------------------------------- launch ---

def test_the_launch_hands_the_controller_supervision_and_a_watchdog(
        tmp_path, monkeypatch):
    """The longest single run in the program. Dropped supervision means a
    relaunch starts at step zero beside a checkpoint nobody opened."""
    spawned = {}

    class _Fake:
        pid = 5150

    def spawn(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return _Fake()

    plan = _plan(tmp_path, monkeypatch, estimated_hours=14.0)
    started = CE.launch(plan, state_path=str(tmp_path / "state.json"),
                        log_path=str(tmp_path / "extension.log"), spawn=spawn)
    argv = spawned["argv"]

    assert started["pid"] == 5150
    assert spawned["kwargs"]["start_new_session"] is True
    assert argv[argv.index("--phase") + 1] == CE.PHASE == "phase8-extension-2b"
    assert argv[argv.index("--estimated-hours") + 1] == "14.0"
    assert (argv[argv.index("--supervise-checkpoint") + 1]
            == plan["supervise_checkpoint"])
    assert argv[argv.index("--watchdog-tokens") + 1] == "2000000000"
    assert argv[argv.index("--") + 1:] == plan["command"]


# -------------------------------------------------------------------- cli ---

def test_the_cli_refuses_a_stopped_gate_without_launching(tmp_path, capsys,
                                                          monkeypatch):
    built = _record(tmp_path)
    monkeypatch.chdir(tmp_path)

    rc = CE._cli(["launch",
                  "--verdict", _branch_verdict(tmp_path, proceed=False,
                                               reason="code-bpb did not pass"),
                  "--probe-verdict", _probe_verdict(tmp_path),
                  "--init-from", _branch_run(tmp_path),
                  "--mixture-record", built["path"],
                  "--run-root", "runs",
                  "--state", _state(tmp_path)])

    assert rc == 2
    assert "says stop" in capsys.readouterr().err


def test_plan_prints_the_rate_the_floor_and_the_argv(tmp_path, capsys,
                                                     monkeypatch):
    built = _record(tmp_path)
    monkeypatch.chdir(tmp_path)

    rc = CE._cli(["plan",
                  "--verdict", _branch_verdict(tmp_path),
                  "--probe-verdict", _probe_verdict(tmp_path),
                  "--init-from", _branch_run(tmp_path),
                  "--mixture-record", built["path"],
                  "--run-root", "runs",
                  "--state", _state(tmp_path),
                  "--estimated-hours", "14.0",
                  "--json-out", str(tmp_path / "extension-plan.json")])
    out = capsys.readouterr().out

    assert rc == 0
    assert "2,000,000,000 tokens" in out
    assert "3,000,000,000 cumulative tokens" in out
    assert "replay" in out and "train.py" in out
    written = json.loads((tmp_path / "extension-plan.json").read_text())
    assert written["gate"] == "extension_2b"
    assert written["arm"]["muon_lr"] == pytest.approx(5e-4)


# ---------------------------------------------------- against the real box ---

_MIXTURE = "runs/codeprep/train-mixture.json"

requires_corpus = pytest.mark.skipif(
    not os.path.exists(_MIXTURE),
    reason="needs the built code corpus")


@requires_corpus
def test_the_real_corpus_can_fill_the_extensions_budget():
    """Whether 2B tokens can be drawn from the corpus built for 250M arms --
    asked now rather than when the branch's verdict lands.

    The epoch cap moves shares with the budget, and this budget is eight times
    the one the corpus was composed at. Finding a skew or a thinned replay floor
    here leaves time to top the corpus up; finding it at launch means finding it
    with the GPU idle and a stage expected.
    """
    record = CP.load_mixture(_MIXTURE)
    preflight = CP.mixture_preflight_at(record, CE.EXTENSION_TOKENS)

    assert preflight["total_run_tokens"] == CE.EXTENSION_TOKENS
    assert preflight["l1_skew_pts"] <= CP.MAX_L1_SKEW_PTS, (
        f"the 2B extension cannot draw the preregistered mixture: "
        f"{preflight['l1_skew_pts']:.2f} points of skew, "
        f"{', '.join(preflight['capped_sources'])} capped")

    floor = CE.replay_floor(record, preflight)
    # Printed rather than only asserted: `-s` on this node is how the headroom
    # at 2B gets read off the corpus that exists, without a GPU or a launch.
    print(f"\n2B draw: L1 skew {preflight['l1_skew_pts']:.3f} pts, "
          f"max epochs {preflight['max_epochs_seen']:.2f} "
          f"({preflight['most_repeated_source']}), "
          f"capped {preflight['capped_sources'] or 'nothing'}; "
          f"replay {floor['effective_pts']:.3f} of {floor['target_pts']:.3f} "
          f"pts, bar {floor['tolerance_pts']:.2f}")

    assert floor["shortfall_pts"] <= floor["tolerance_pts"], (
        f"the replay floor thins to {floor['effective_pts']:.2f} of "
        f"{floor['target_pts']:.2f} points at 2B "
        f"({', '.join(floor['capped_sources'])} capped)")
