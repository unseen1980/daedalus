"""Gate 2's launch: what it refuses, and what it hands the controller.

The 1B branch is one command that spends about six hours of GPU, and every way
it can go wrong produces a run that trains, exits 0 and reports numbers nobody
can interpret: started from a probe arm instead of the base, launched at a rate
no arm ran, warmed up for a quarter of its budget, drawing a mixture the epoch
cap moved at the larger budget, or launched at all against a gate that said
stop. So the subject here is the refusals, and the agreement between the argv
and the checkpoint the supervisor is told to resume.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import code_branch as CB
from scripts import code_probes as CP


# ----------------------------------------------------------------- fixtures ---

def _shard_dir(root, name, tokens=4_000_000_000):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"total_tokens": tokens, "shards": [], "eos_id": 0}))
    return path


def _record(tmp_path, weights=None, tokens=None) -> dict:
    """A composed mixture with enough tokens that 1B fits inside the epoch cap."""
    weights = weights or {"code-python": 0.65, "fineweb-edu": 0.20,
                          "finemath-3plus": 0.15}
    train_root = tmp_path / "mix" / "train"
    holdout_root = tmp_path / "mix" / "holdout"
    for name in weights:
        _shard_dir(train_root, name,
                   tokens=(tokens or {}).get(name, 4_000_000_000))
    _shard_dir(holdout_root, "code-python", tokens=2_000_000)
    record = {"schema": 1, "weights": weights,
              "train_root": str(train_root), "holdout_root": str(holdout_root)}
    path = tmp_path / "train-mixture.json"
    path.write_text(json.dumps(record))
    return {"path": str(path), "record": record}


def _base_checkpoint(tmp_path, body=b"released base weights") -> str:
    path = tmp_path / "hero" / "checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return str(path)


def _verdict(tmp_path, *, selected="code-probe-lr0.001", proceed=True,
             reason="a reason", name="verdict.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps({"schema": 1, "continue": proceed,
                                "selected": selected, "reason": reason}))
    return str(path)


def _probe_metrics(run_root, *, rates, tokens=250_000_000):
    """One finished arm per entry of `rates`, at that windowed throughput."""
    for name, rate in rates.items():
        run_dir = run_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.jsonl").write_text("".join(
            json.dumps({"step": i, "tokens": tokens // 4 * (i + 1),
                        "loss": 2.0, "tok_per_sec": rate,
                        "elapsed_h": 0.35 * (i + 1)}) + "\n"
            for i in range(4)))


def _plan(tmp_path, monkeypatch, **kwargs):
    """A plan built with the trainer's own `runs/` under `tmp_path`.

    `train.py` has no `--run-dir` flag, so it writes `runs/<run-name>` relative
    to its working directory. Moving the *directory* rather than a flag is what
    production does, so the plan's checkpoint path is exercised as it is used.
    """
    built = _record(tmp_path)
    monkeypatch.chdir(tmp_path)
    _probe_metrics(tmp_path / "runs", rates={arm.name: 50_000.0
                                             for arm in CP.probe_arms()})
    defaults = dict(verdict_path=_verdict(tmp_path),
                    init_from=_base_checkpoint(tmp_path),
                    mixture_record=built["path"], run_root="runs")
    defaults.update(kwargs)
    return CB.branch_plan(**defaults)


# ------------------------------------------------------- the gate's answer ---

def test_a_gate_that_said_stop_cannot_be_launched_anyway():
    """The preregistered response to a failed probe gate is a recorded negative
    result. A launch here would spend 1B tokens against a gate that returned
    no, which is the one outcome preregistration exists to prevent."""
    verdict = {"continue": False, "selected": None,
               "reason": "no arm improved code BPB by >=2% or moved the "
                         "execution/syntax signal"}

    with pytest.raises(CB.BranchRefused) as excinfo:
        CB.selected_arm(verdict)

    assert "says stop" in str(excinfo.value)
    assert "no arm improved code BPB" in str(excinfo.value)


def test_a_gate_that_continued_without_naming_an_arm_is_refused():
    """`continue` and `selected` are set by different rules -- the gate's
    criteria and general retention -- so "continue, but every qualifying arm
    failed retention" is reachable, and there is no rate to launch at."""
    verdict = {"continue": True, "selected": None,
               "reason": "no qualifying arm held general retention"}

    with pytest.raises(CB.BranchRefused) as excinfo:
        CB.selected_arm(verdict)

    assert "selected no arm" in str(excinfo.value)


def test_the_rate_comes_from_the_preregistered_arms_not_from_the_verdict():
    """A verdict is a file on disk. Resolving the rate by name through
    `probe_arms()` means the only rates this can launch are the three that were
    preregistered and actually ran."""
    arm = CB.selected_arm({"continue": True, "selected": "code-probe-lr0.002"})

    assert arm.muon_lr == 2e-3
    assert arm.adam_lr == pytest.approx(CP.probe_arms()[2].adam_lr)


def test_an_arm_name_no_probe_ran_under_is_refused():
    with pytest.raises(CB.BranchRefused) as excinfo:
        CB.selected_arm({"continue": True, "selected": "code-probe-lr0.005"})

    assert "not one of the preregistered arms" in str(excinfo.value)


def test_a_probe_report_is_not_a_verdict(tmp_path):
    """`probes.json` and `verdict.json` sit in the same directory and one of
    them has already applied the gate."""
    path = tmp_path / "probes.json"
    path.write_text(json.dumps({"schema": 1, "gate": "probes_250m",
                                "arms": []}))

    with pytest.raises(CB.BranchRefused) as excinfo:
        CB.load_verdict(path)

    assert "no `continue` field" in str(excinfo.value)


# --------------------------------------------------------- fresh, not staged ---

def test_the_branch_is_named_apart_from_every_probe_arm():
    """Under a probe's name, `arm_is_complete` would read the finished 250M arm
    as this run's budget and a relaunch would resume into it."""
    arm = CB.selected_arm({"continue": True, "selected": "code-probe-lr0.001"})

    assert arm.name == CB.BRANCH_NAME
    assert arm.name not in {probe.name for probe in CP.probe_arms()}


def test_starting_from_a_probe_arms_checkpoint_is_refused_by_name(tmp_path):
    """"A fresh 1B branch from the base" and "continue the selected arm" are one
    --init-from apart, and the second trains fine: it produces 1.25B staged
    tokens reported as a fresh 1B run."""
    run_root = tmp_path / "runs"
    arm = CP.probe_arms()[1]
    (run_root / arm.name).mkdir(parents=True)
    (run_root / arm.name / "checkpoint.pt").write_bytes(b"250M in")

    with pytest.raises(CB.BranchRefused) as excinfo:
        CB.refuse_probe_checkpoint(run_root / arm.name / "checkpoint.pt",
                                   run_root=str(run_root))

    assert arm.name in str(excinfo.value)
    assert "staged 1.25B run" in str(excinfo.value)


def test_the_released_base_is_not_mistaken_for_a_probe_checkpoint(tmp_path):
    CB.refuse_probe_checkpoint(_base_checkpoint(tmp_path),
                               run_root=str(tmp_path / "runs"))


def test_a_base_checkpoint_that_is_not_the_pinned_artifact_is_refused(
        tmp_path, monkeypatch):
    """Every phase 8 number is a difference against this file; a substituted one
    still trains three arms that still compare against each other."""
    with pytest.raises(ValueError, match="not the pinned"):
        _plan(tmp_path, monkeypatch, init_from_sha256="0" * 64)


# ----------------------------------------------------------- the schedule ---

def test_the_warmup_is_recomputed_for_the_larger_budget():
    """Copying an arm's argv and editing --total-tokens leaves a 1B run warming
    up over a schedule sized for 250M."""
    probe = CP.probe_arms()[1]
    branch = CB.selected_arm({"continue": True, "selected": probe.name})

    from scripts.qat_recovery import WARMUP_FRAC, estimated_steps

    assert branch.total_tokens == CB.BRANCH_TOKENS == 1_000_000_000
    # 5% of *this* budget's steps, not of the probe's.
    assert branch.warmup_steps == int(
        estimated_steps(CB.BRANCH_TOKENS) * WARMUP_FRAC)
    assert branch.warmup_steps == pytest.approx(probe.warmup_steps * 4, rel=0.05)


def test_the_command_carries_the_branch_budget_and_no_resume(tmp_path,
                                                             monkeypatch):
    """--resume on attempt one restores the finished run's step and token count,
    so the phase writes no metrics row and exits 0."""
    plan = _plan(tmp_path, monkeypatch)
    command = plan["command"]

    assert "--resume" not in command
    assert command[command.index("--total-tokens") + 1] == "1000000000"
    assert command[command.index("--run-name") + 1] == CB.BRANCH_NAME
    assert command[command.index("--init-from") + 1] == plan["init_from"]["path"]


def test_the_supervised_checkpoint_is_the_one_this_argv_writes(tmp_path,
                                                               monkeypatch):
    """The launcher makes the same check, and it is the check that matters: a
    marker beside a file that never appears means every relaunch restarts from
    step zero and nothing reports a problem."""
    from scripts.vast_program import trainer_checkpoint_for

    plan = _plan(tmp_path, monkeypatch)

    assert (os.path.abspath(trainer_checkpoint_for(plan["command"]))
            == os.path.abspath(plan["supervise_checkpoint"]))


def test_a_run_root_cannot_move_where_the_trainer_writes(tmp_path, monkeypatch):
    """`train.py` has no --run-dir flag, so `--run-root` moves only where the
    *probe arms* are read from. Composing the branch's checkpoint from it
    instead would supervise a path that never appears."""
    plan = _plan(tmp_path, monkeypatch, run_root=str(tmp_path / "runs"))

    assert plan["supervise_checkpoint"] == os.path.join(
        "runs", CB.BRANCH_NAME, "checkpoint.pt")


# ------------------------------------------------------------- the mixture ---

def test_the_mixture_is_re_preflighted_at_the_branchs_own_budget(tmp_path,
                                                                 monkeypatch):
    """The epoch cap moves shares with the budget, so the mixture the 250M arms
    trained on is not automatically the one 1B tokens draw."""
    plan = _plan(tmp_path, monkeypatch)

    assert plan["mixture"]["preflight"]["total_run_tokens"] == 1_000_000_000


def test_a_mixture_the_epoch_cap_skews_past_the_limit_is_refused(tmp_path,
                                                                 monkeypatch):
    """At 1B tokens a source with 60M tokens cannot fill a 65% share inside four
    epochs, and the run that results is a different experiment."""
    built = _record(tmp_path, tokens={"code-python": 60_000_000})
    monkeypatch.chdir(tmp_path)
    _probe_metrics(tmp_path / "runs", rates={arm.name: 50_000.0
                                             for arm in CP.probe_arms()})

    with pytest.raises(CB.BranchRefused) as excinfo:
        CB.branch_plan(verdict_path=_verdict(tmp_path),
                       init_from=_base_checkpoint(tmp_path),
                       mixture_record=built["path"], run_root="runs")

    assert "points from the one asked for" in str(excinfo.value)


# ---------------------------------------------------------- the projection ---

def test_throughput_is_read_from_windows_not_from_elapsed_hours(tmp_path):
    """`elapsed_h` is wall-clock since *this process* started, so after a resume
    it restarts at zero while `tokens` keeps climbing. The ratio then overstates
    throughput, and an overstated rate understates the hours -- the direction
    that gets a run admitted inside the deadline reserve when it does not fit."""
    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        # A resumed run: tokens continue, elapsed_h restarts.
        {"step": 100, "tokens": 200_000_000, "tok_per_sec": 40_000,
         "elapsed_h": 0.01},
        {"step": 110, "tokens": 205_000_000, "tok_per_sec": 40_000,
         "elapsed_h": 0.05},
    ]))

    rate = CB.arm_throughput(run_dir)

    assert rate == 40_000
    # What the tokens/elapsed_h reading would have claimed, for contrast.
    assert 205_000_000 / 0.05 / 3600 > 1_000_000


def test_a_slow_first_window_does_not_set_the_estimate(tmp_path):
    """The first windows of a run carry torch.compile and are not the rate the
    other 99% of it runs at."""
    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"step": 1, "tokens": 1, "tok_per_sec": 900},
        {"step": 2, "tokens": 2, "tok_per_sec": 50_000},
        {"step": 3, "tokens": 3, "tok_per_sec": 50_100},
    ]))

    assert CB.arm_throughput(run_dir) == 50_000


def test_a_run_with_no_throughput_rows_reports_none(tmp_path):
    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "tokens": 1, "tok_per_sec": None}) + "\n")

    assert CB.arm_throughput(run_dir) is None


def test_the_projection_takes_the_slowest_arm(tmp_path):
    """The estimate feeds a deadline reserve. An optimistic one is refused at
    T+136h with the run half done."""
    run_root = tmp_path / "runs"
    names = [arm.name for arm in CP.probe_arms()]
    _probe_metrics(run_root, rates={names[0]: 50_000.0, names[1]: 40_000.0,
                                    names[2]: 60_000.0})

    projection = CB.projected_hours(run_root=str(run_root))

    assert projection["slowest_tok_per_sec"] == 40_000.0
    assert projection["hours"] == pytest.approx(
        1_000_000_000 / 40_000 / 3600 * CB.HOURS_MARGIN)


def test_no_measurable_probe_refuses_rather_than_guessing_hours(tmp_path):
    with pytest.raises(CB.BranchRefused) as excinfo:
        CB.projected_hours(run_root=str(tmp_path / "runs"))

    assert "--estimated-hours" in str(excinfo.value)


def test_an_explicit_estimate_skips_the_projection(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch, estimated_hours=9.0)

    assert plan["estimated_hours"] == 9.0
    assert plan["projection"] is None


# ------------------------------------------------------------ idempotence ---

def test_a_branch_that_already_trained_its_budget_is_not_relaunched(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / CB.BRANCH_NAME
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1907, "tokens": 1_000_000_000}) + "\n")

    with pytest.raises(CB.BranchRefused) as excinfo:
        _plan(tmp_path, monkeypatch)

    assert "already trained" in str(excinfo.value)


# ----------------------------------------------------------------- launch ---

def test_the_launch_hands_the_controller_supervision_and_a_watchdog(
        tmp_path, monkeypatch):
    """This is the phase `--supervise-checkpoint` was added for: one long run,
    no orchestrator. Dropped, a relaunch starts at step zero beside a 600MB
    checkpoint nobody opened."""
    spawned = {}

    class _Fake:
        pid = 4242

    def spawn(argv, **kwargs):
        spawned["argv"] = argv
        spawned["kwargs"] = kwargs
        return _Fake()

    plan = _plan(tmp_path, monkeypatch, estimated_hours=7.0)
    started = CB.launch(plan, state_path=str(tmp_path / "state.json"),
                        log_path=str(tmp_path / "branch.log"), spawn=spawn)
    argv = spawned["argv"]

    assert started["pid"] == 4242
    assert spawned["kwargs"]["start_new_session"] is True
    assert argv[argv.index("--phase") + 1] == CB.PHASE
    assert argv[argv.index("--estimated-hours") + 1] == "7.0"
    assert (argv[argv.index("--supervise-checkpoint") + 1]
            == plan["supervise_checkpoint"])
    assert argv[argv.index("--watchdog-tokens") + 1] == "1000000000"
    assert argv[argv.index("--max-attempts") + 1] == str(CB.MAX_ATTEMPTS)
    assert argv[argv.index("--") + 1:] == plan["command"]


def test_a_live_main_lane_lease_is_reported_rather_than_collided_with(tmp_path):
    """The detached child would take the same refusal, but in its own log, where
    nothing reads it until someone wonders why the branch never started."""
    state = tmp_path / "state.json"
    from scripts.vast_program import default_lease_name, process_start_ticks

    lock = state.with_name(default_lease_name())
    lock.write_text(json.dumps({"schema": 1, "pid": os.getpid(),
                                "start_ticks": process_start_ticks(os.getpid()),
                                "acquired_at": "2026-08-27T01:42:45Z"}))

    holder = CB.lease_holder(state)

    assert holder is not None and holder["pid"] == os.getpid()


def test_a_dead_lease_does_not_block_the_launch(tmp_path):
    state = tmp_path / "state.json"
    from scripts.vast_program import default_lease_name

    state.with_name(default_lease_name()).write_text(
        json.dumps({"schema": 1, "pid": 2 ** 22, "start_ticks": 1,
                    "acquired_at": "2026-08-27T01:42:45Z"}))

    assert CB.lease_holder(state) is None


def test_no_lease_file_at_all_is_not_a_holder(tmp_path):
    assert CB.lease_holder(tmp_path / "state.json") is None


# -------------------------------------------------------------------- cli ---

def test_the_cli_refuses_a_stopped_gate_without_launching(tmp_path, capsys,
                                                          monkeypatch):
    built = _record(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = CB._cli(["launch",
                  "--verdict", _verdict(tmp_path, proceed=False, selected=None),
                  "--init-from", _base_checkpoint(tmp_path),
                  "--init-from-sha256", "",
                  "--mixture-record", built["path"],
                  "--run-root", "runs",
                  "--state", str(tmp_path / "state.json")])

    assert rc == 2
    assert "says stop" in capsys.readouterr().err


def test_plan_prints_the_rate_the_budget_and_the_argv(tmp_path, capsys,
                                                      monkeypatch):
    built = _record(tmp_path)
    monkeypatch.chdir(tmp_path)
    _probe_metrics(tmp_path / "runs", rates={arm.name: 50_000.0
                                             for arm in CP.probe_arms()})

    rc = CB._cli(["plan", "--verdict", _verdict(tmp_path),
                  "--init-from", _base_checkpoint(tmp_path),
                  "--init-from-sha256", "",
                  "--mixture-record", built["path"],
                  "--run-root", "runs",
                  "--json-out", str(tmp_path / "branch-plan.json")])
    out = capsys.readouterr().out

    assert rc == 0
    assert "code-probe-lr0.001 -> code-branch-1b" in out
    assert "1,000,000,000 tokens" in out
    assert "train.py" in out
    written = json.loads((tmp_path / "branch-plan.json").read_text())
    assert written["gate"] == "branch_1b"
    assert written["selected_probe"] == "code-probe-lr0.001"


# ------------------------------------------------------- against the real box ---

_BASE = "/root/daedalus/final/hero/checkpoint.pt"
_MIXTURE = "runs/codeprep/train-mixture.json"

requires_corpus = pytest.mark.skipif(
    not (os.path.exists(_BASE) and os.path.exists(_MIXTURE)),
    reason="needs the released base checkpoint and the built code corpus")


@requires_corpus
def test_the_real_corpus_can_fill_the_branchs_budget(tmp_path):
    """Whether 1B tokens can be drawn from the corpus that was built for 250M
    arms -- asked now rather than at launch.

    The epoch cap moves shares with the budget, so a corpus that preflighted
    cleanly for three 250M arms can still skew past the limit at four times the
    draw. Finding that when the verdict lands would mean discovering it with the
    GPU idle and a launch expected; finding it here leaves time to repair the
    mixture. The hash and the throughput projection come along for free, which
    covers the other two inputs a launch depends on.

    Asked of the three inputs directly rather than through `branch_plan`,
    because the plan carries one more refusal that is *not* about the corpus:
    once the branch has trained its budget, `arm_is_complete` refuses to spend
    it a second time. That refusal is the launcher working. Routed through the
    plan, this test inverted at 09:51Z on 2026-08-27 -- it began reporting a
    corpus that can fill the budget as a corpus that cannot, at the moment the
    run it was gating succeeded.
    """
    record = CB.load_mixture(_MIXTURE)
    preflight = CB.mixture_preflight_at(record, CB.BRANCH_TOKENS)
    assert preflight["total_run_tokens"] == CB.BRANCH_TOKENS
    assert preflight["l1_skew_pts"] <= CB.MAX_L1_SKEW_PTS
    assert CB.assert_base_checkpoint(_BASE, None) == (
        "cfbf27dccf93a07caa2b93cbd630e483c174d52aed8785d104edb7addeb0e153")
    assert 0.0 < CB.projected_hours()["hours"] < 24.0

    # And the launcher on those same inputs: either it still has a launch to
    # authorise, or the run is done and it declines to repeat it. Anything else
    # is a refusal about the corpus, which is what this test is here to catch.
    try:
        plan = CB.branch_plan(verdict_path=_verdict(tmp_path), init_from=_BASE,
                              mixture_record=_MIXTURE)
    except CB.BranchRefused as exc:
        assert "already trained" in str(exc), (
            f"the 1B branch cannot launch as preregistered: {exc}")
    else:
        assert plan["mixture"]["preflight"] == preflight
        assert 0.0 < plan["estimated_hours"] < 24.0
