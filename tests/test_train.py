"""Tests for train.py. CPU-only, no compile, no network -- W&B and git are
exercised through fakes/local repos so the suite stays fast and offline.

Run: python -m pytest tests/test_train.py -v
"""
import json
import math
import os
import random
import subprocess
import sys
import time

import pytest
import torch

from daedalus.config import PRESETS
from daedalus.data import ShardWriter
from daedalus.model import Daedalus
from daedalus import ckpt_uploader as cu
from daedalus.muon import build_optimizers, decay_start_step, wsd_lr
import train as train_module
from train import (
    HUB_STALE_FACTOR,
    IntervalGate,
    MixtureBatchSource,
    ShardBatchSource,
    SyntheticBatchSource,
    TrainArgs,
    Trainer,
    WandbLogger,
    append_metrics,
    batch_tokens_schedule,
    bits_per_byte,
    cap_weights_by_epochs,
    estimate_total_steps,
    git_commit_and_push,
    grad_accum_steps,
    load_checkpoint,
    parse_args,
    ramp_progress,
    resolve_wandb_run_id,
    save_checkpoint,
    seq_len_schedule,
)


# ---------------------------------------------------------------- schedules ---

def test_ramp_progress_clamped():
    assert ramp_progress(0, 100) == 0.0
    assert ramp_progress(50, 100) == 0.5
    assert ramp_progress(200, 100) == 1.0
    assert ramp_progress(50, 0) == 1.0  # zero-length ramp is immediately "done"


def test_seq_len_schedule_ramps_and_snaps():
    assert seq_len_schedule(0, 1000, start=1024, end=2048, round_to=128) == 1024
    assert seq_len_schedule(1000, 1000, start=1024, end=2048, round_to=128) == 2048
    mid = seq_len_schedule(500, 1000, start=1024, end=2048, round_to=128)
    assert mid % 128 == 0
    assert 1024 <= mid <= 2048


def test_batch_tokens_schedule_linear():
    assert batch_tokens_schedule(0, 1000, start=128_000, end=512_000) == 128_000
    assert batch_tokens_schedule(1000, 1000, start=128_000, end=512_000) == 512_000
    mid = batch_tokens_schedule(500, 1000, start=128_000, end=512_000)
    assert 128_000 < mid < 512_000


def test_grad_accum_steps_rounds_to_nearest():
    assert grad_accum_steps(target_tokens=32_000, micro_batch=16, seq_len=2000) == 1
    assert grad_accum_steps(target_tokens=64_000, micro_batch=16, seq_len=2000) == 2
    assert grad_accum_steps(target_tokens=100, micro_batch=16, seq_len=2000) == 1  # floor at 1


def _replay_tokens(total_steps, ramp_tokens, micro_batch, seq_start, seq_end,
                   tok_start, tok_end):
    """Tokens the loop actually consumes over `total_steps` steps."""
    seen = 0
    for _ in range(total_steps):
        seq = seq_len_schedule(seen, ramp_tokens, seq_start, seq_end)
        bt = batch_tokens_schedule(seen, ramp_tokens, tok_start, tok_end)
        seen += micro_batch * seq * grad_accum_steps(bt, micro_batch, seq)
    return seen


def test_estimate_total_steps_matches_the_loops_own_token_accounting():
    total_tokens, ramp = 2_000_000_000, 200_000_000
    kw = dict(micro_batch=20, seq_start=1024, seq_end=2048,
              tok_start=128_000, tok_end=512_000)
    n = estimate_total_steps(total_tokens, ramp, **kw)
    # exactly the first step count that reaches the budget, not one more or less
    assert _replay_tokens(n, ramp, **kw) >= total_tokens
    assert _replay_tokens(n - 1, ramp, **kw) < total_tokens


def test_wsd_schedule_actually_decays_to_zero():
    """Regression: the lr multiplier on the final step must be ~0.

    The old estimator averaged tok_start/tok_end, overshooting the real step
    count by ~1.42x because the batch ramp is over after 10% of the budget.
    Every run therefore ended at ~0.66 of peak lr with no anneal -- the exact
    thing WSD exists to do.
    """
    from daedalus.muon import wsd_lr

    total_tokens, ramp = 2_000_000_000, 200_000_000
    kw = dict(micro_batch=20, seq_start=1024, seq_end=2048,
              tok_start=128_000, tok_end=512_000)
    n = estimate_total_steps(total_tokens, ramp, **kw)
    assert wsd_lr(n - 1, n, warmup=300, decay_frac=0.45) < 0.01

    naive = int(total_tokens / ((kw["tok_start"] + kw["tok_end"]) / 2))
    assert wsd_lr(n - 1, naive, warmup=300, decay_frac=0.45) > 0.5  # the old bug


# (step, tokens_seen) recorded from the live `abl-arch` hybrid arm on
# 2026-08-10, run `i7ues1xa`, at micro_batch 16 over a 5B budget. Frozen here
# rather than read from runs/ so the test does not depend on run state.
ARM1_LIVE_TOKENS = [
    (20, 2_621_440), (200, 27_656_192), (500, 77_846_528),
    (700, 118_331_392), (900, 165_572_608), (1000, 192_133_120),
    (1500, 359_940_096), (2000, 600_782_848), (3000, 1_125_070_848),
    (3680, 1_481_586_688),
]


def test_the_step_estimator_matches_a_real_run_token_for_token():
    """The estimator checked against hardware, not against itself.

    `test_estimate_total_steps_matches_the_loops_own_token_accounting` compares
    the estimator to a replay helper in this file -- a re-implementation of the
    same formula, so the two agreeing says nothing about whether either matches
    the training loop. These rows came off a real 12-hour GPU run, and they
    span the ramp (where the earlier WSD bug lived: the batch-token ramp
    finishes after 10% of the budget, and averaging across it overshot the step
    count by ~1.42x, ending every run at ~0.66 of peak lr with no anneal).

    All 184 rows of that run's metrics.jsonl matched exactly when this was
    written; ten spanning steps 20-3,680 are frozen above.
    """
    ramp = int(0.1 * 5_000_000_000)
    kw = dict(micro_batch=16, seq_start=1024, seq_end=2048,
              tok_start=128_000, tok_end=512_000)
    for step, tokens in ARM1_LIVE_TOKENS:
        assert _replay_tokens(step, ramp, **kw) == tokens, (
            f"step {step}: schedule predicts "
            f"{_replay_tokens(step, ramp, **kw)}, the live run recorded "
            f"{tokens}")


@pytest.mark.parametrize("budget,steps,decay", [
    (60_000_000_000, 124_684, 68_576),  # hero, as the operator raised it
    (40_000_000_000, 83_123, 45_717),   # hero, as recommended at the gate
    (30_000_000_000, 62_343, 34_288),   # hero, the cheaper gate option
    (5_000_000_000, 10_391, 5_715),     # abl-arch, per arm
])
def test_the_documented_schedules_are_what_the_code_computes(budget, steps, decay):
    """Pins the numbers the gate issue, STATUS.md and the model card quote.

    They are not decoration: `decay` is the step at which the milestone
    checkpoint -- the reusable branch point the whole checkpoint-durability
    precondition exists for -- is written, and `steps` is the denominator
    `wsd_lr` anneals over. Getting that denominator wrong is precisely the bug
    that cost four days and ~$44, and it was invisible in steady state.

    Recomputed here rather than retyped: checking hero's schedule by hand, I
    first got 82,792/45,535 by passing powers of two (131,072 -> 524,288) where
    TrainArgs defaults to 128,000 -> 512,000. The realized step size is
    identical either way -- grad_accum rounds 512,000 up to exactly 524,288
    tokens -- so the slip shows up only in the step *count*.
    """
    from daedalus.muon import decay_start_step
    n = estimate_total_steps(budget, int(0.1 * budget), micro_batch=16,
                             seq_start=1024, seq_end=2048,
                             tok_start=128_000, tok_end=512_000)
    assert n == steps
    assert decay_start_step(n, 0.45) == decay


@pytest.mark.parametrize(
    "budget", [58_000_000_000, 59_900_000_000, 60_000_000_000])
def test_the_schedule_still_anneals_at_the_60b_budget(budget):
    """Required before launching hero, measured at the launch budget specifically.

    The WSD bug in 95d7bc4 was a mis-priced batch ramp in the step estimator:
    the schedule never reached zero and every run ended at ~0.66 of peak lr.
    The fix replays the ramp for an exact count, but "it was fixed at 40B" is
    not evidence about 60B -- the ramp still finishes at 10% of the budget, so
    the shape of the error scales with it. So this asserts the two properties
    the operator asked for, at the real number rather than a scaled-down one.

    58B was added on the 11th. It is *the* number now: the materialized holdout
    carve pushed the real mixture skew to 10.21 against hero's own 10.0 limit,
    so 60B refuses and the gate asks for 58B. This test went on certifying 60B
    -- a budget that cannot launch -- while the budget that would launch had no
    schedule check anywhere. Which budget is the default is not the question
    this test should turn on; which budget can be launched is.

    59.9B was added when the operator replied `go 59.9B`, choosing the largest
    budget that launches rather than the recommended 58B. Same reasoning as
    above and the operator's own standing instruction -- "do not assume the fix
    generalises, measure it at the launch budget specifically" -- so the number
    that will actually be passed to `--total-tokens` is certified here before
    the run starts, not inferred from the two budgets either side of it.
    """
    from daedalus.muon import decay_start_step, wsd_lr

    n = estimate_total_steps(budget, int(0.1 * budget), micro_batch=16,
                             seq_start=1024, seq_end=2048,
                             tok_start=128_000, tok_end=512_000)
    d = decay_start_step(n, 0.45)

    # It genuinely anneals: the last step is ~1.8e-5 of peak, not 0.66 of it.
    assert wsd_lr(n - 1, n, warmup=300, decay_frac=0.45) < 0.01
    # And decay begins at 55% of the *true* step count.
    assert abs(d / n - 0.55) < 1e-4, f"decay starts at {d / n:.4%}, not 55%"
    # The branch point is at full lr -- that is what makes it a branch point.
    # A milestone written one decayed step late looks correct and is not.
    assert wsd_lr(d, n, warmup=300, decay_frac=0.45) == pytest.approx(1.0)


def test_trainer_caches_exact_total_steps_for_token_budget(tmp_path):
    args = TrainArgs(run_name="steps", config="tiny", micro_batch=2,
                     seq_start=16, seq_end=16, tok_start=32, tok_end=64,
                     total_tokens=100_000, ramp_frac=0.1, compile=False,
                     device="cpu", run_dir=str(tmp_path / "steps"),
                     wandb_enabled=False)
    t = Trainer(args)
    assert t.total_steps == t._estimated_total_steps()
    assert t.total_steps == estimate_total_steps(
        args.total_tokens, t.ramp_tokens, args.micro_batch, args.seq_start,
        args.seq_end, args.tok_start, args.tok_end)


def test_trainer_total_steps_is_max_steps_when_given(tmp_path):
    args = TrainArgs(run_name="steps2", config="tiny", max_steps=7,
                     micro_batch=2, seq_start=16, seq_end=16, compile=False,
                     device="cpu", run_dir=str(tmp_path / "steps2"),
                     wandb_enabled=False)
    assert Trainer(args).total_steps == 7


# --------------------------------------------------------------- checkpoint ---

def test_checkpoint_roundtrip_preserves_state(tmp_path):
    cfg = PRESETS["tiny"]
    torch.manual_seed(0)
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss, _ = model(x, targets=x)
    loss.backward()
    muon.step()
    adamw.step()

    path = save_checkpoint(str(tmp_path / "ckpt.pt"), model, muon, adamw,
                           step=7, tokens_seen=123, cfg=cfg)

    torch.manual_seed(99)  # different init
    model2 = Daedalus(cfg)
    muon2, adamw2, _ = build_optimizers(model2)
    info = load_checkpoint(path, model2, muon2, adamw2)

    assert info["step"] == 7
    assert info["tokens_seen"] == 123
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_weights_only_omits_optimizer(tmp_path):
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model)
    path = save_checkpoint(str(tmp_path / "ckpt.pt"), model, muon, adamw,
                           step=1, tokens_seen=1, cfg=cfg, save_optimizer=False)
    payload = torch.load(path, weights_only=False)
    assert "muon" not in payload and "adamw" not in payload

    model2 = Daedalus(cfg)
    info = load_checkpoint(path, model2)  # no optimizers passed -- must not crash
    assert info["step"] == 1


def test_checkpoint_never_leaves_partial_file(tmp_path):
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, muon, adamw, step=1, tokens_seen=1, cfg=cfg)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


# ------------------------------------------------------------------ metrics ---

def test_append_metrics_writes_valid_jsonl(tmp_path):
    append_metrics(str(tmp_path), {"step": 1, "loss": 2.0})
    append_metrics(str(tmp_path), {"step": 2, "loss": 1.5})
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 1, "loss": 2.0}
    assert json.loads(lines[1]) == {"step": 2, "loss": 1.5}


def test_bits_per_byte_matches_manual_formula():
    loss_nats, n_tokens, n_bytes = 2.0, 100, 400
    expected = (loss_nats * n_tokens) / (n_bytes * math.log(2))
    assert bits_per_byte(loss_nats, n_tokens, n_bytes) == pytest.approx(expected)


def test_bits_per_byte_handles_zero_bytes():
    assert math.isnan(bits_per_byte(1.0, 10, 0))


# --------------------------------------------------------------------- gate ---

def test_interval_gate_fires_first_call_then_waits():
    t = [0.0]
    gate = IntervalGate(10.0, clock=lambda: t[0])
    assert gate.ready() is True
    assert gate.ready() is False
    t[0] = 9.9
    assert gate.ready() is False
    t[0] = 10.0
    assert gate.ready() is True


# ------------------------------------------------------------------- wandb ---

class _FakeWandbRun:
    def __init__(self, fail_log=False):
        self.logs = []
        self.finished = False
        self._fail_log = fail_log

    def log(self, record, step=None):
        if self._fail_log:
            raise RuntimeError("log boom")
        self.logs.append((step, record))

    def finish(self):
        self.finished = True


class _FakeWandbModule:
    def __init__(self, fail_init=False, fail_log=False):
        self.fail_init = fail_init
        self.fail_log = fail_log
        self.last_run = None
        self.init_kwargs = []

    def init(self, **kwargs):
        self.init_kwargs.append(kwargs)
        if self.fail_init:
            raise RuntimeError("init boom")
        self.last_run = _FakeWandbRun(fail_log=self.fail_log)
        return self.last_run


def test_wandb_logger_disabled_is_noop():
    logger = WandbLogger("p", None, "n", {}, enabled=False)
    assert logger.run is None
    logger.log({"a": 1})
    logger.finish()


def test_wandb_logger_forces_offline_without_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule())
    logger = WandbLogger("p", None, "n", {}, enabled=True)
    assert os.environ.get("WANDB_MODE") == "offline"
    assert logger.run is not None


def test_wandb_logger_survives_init_failure(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule(fail_init=True))
    logger = WandbLogger("p", None, "n", {}, enabled=True)
    assert logger.run is None
    logger.log({"a": 1})  # must not raise


def test_wandb_logger_survives_log_failure_without_disabling_itself(monkeypatch):
    """A log failure must not reach the caller -- and must not be terminal.

    This test used to assert `logger.run is None` after one failure, pinning a
    permanent disable. That is the wrong contract for a ~92 h `hero` run: W&B is
    the operator's primary live view (AGENT.md SS5.1), and one transient blip
    would have frozen the dashboard for the rest of the run behind a single
    WARNING line -- indistinguishable, from a phone, from a dead run. The logger
    now mutes and retries; see tests/test_wandb_logger.py for the retry window,
    the failure cap, and the counter reset.
    """
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule(fail_log=True))
    logger = WandbLogger("p", None, "n", {}, enabled=True)
    assert logger.run is not None
    logger.log({"a": 1})            # triggers the fake failure internally
    assert logger.run is not None   # muted for the retry window, not disabled


# ------------------------------------------------- wandb run identity ---
# `supervise.run_with_resume` restarts train.py after a crash, and W&B mints a
# fresh run id per init() -- the run *name* is only a display label. Left
# alone, every restart during hero's ~92 h would strand the operator on the
# frozen URL published in STATUS.md and the gate issue while training carried
# on at a new one. Verified against the real client: two inits with identical
# name/project/tags returned different ids (ma562epj vs vlr1n10i).

def test_wandb_run_id_is_minted_and_persisted_on_a_fresh_run(tmp_path):
    run_dir = tmp_path / "run"
    run_id, mode = resolve_wandb_run_id(str(run_dir), resumed=False)
    assert run_id
    assert mode == "allow"
    assert (run_dir / "wandb-run-id.txt").read_text().strip() == run_id


def test_wandb_run_id_is_reattached_on_resume(tmp_path):
    """A supervisor restart must land in the *same* run, not a sibling."""
    run_dir = tmp_path / "run"
    first, _ = resolve_wandb_run_id(str(run_dir), resumed=False)
    again, mode = resolve_wandb_run_id(str(run_dir), resumed=True)
    assert again == first
    assert mode == "allow"


def test_fresh_run_under_a_recycled_name_does_not_reattach(tmp_path):
    """The counter-case, and the reason this is not keyed off the run name.

    `sweep` was thrown out and re-run under the same run names after the WSD
    bug. Re-attaching there would have appended the good curve to the
    discarded one at the same step numbers, silently interleaving two
    experiments in one chart.
    """
    run_dir = tmp_path / "run"
    first, _ = resolve_wandb_run_id(str(run_dir), resumed=False)
    second, _ = resolve_wandb_run_id(str(run_dir), resumed=False)
    assert second != first
    assert (run_dir / "wandb-run-id.txt").read_text().strip() == second


def test_resume_without_an_id_file_mints_one_rather_than_failing(tmp_path, capsys):
    """Disaster recovery: checkpoint restored from the Hub onto a clean box.

    The id of the original run is unknowable there, so continuity is lost --
    but the run must start, and must say why.
    """
    run_dir = tmp_path / "rebuilt"
    run_id, mode = resolve_wandb_run_id(str(run_dir), resumed=True)
    assert run_id and mode == "allow"
    assert "no wandb-run-id.txt" in capsys.readouterr().out


def test_wandb_logger_forwards_run_identity_to_init(monkeypatch):
    """The id has to actually reach wandb.init() -- the bug was an omission."""
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    fake = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    WandbLogger("p", None, "n", {}, enabled=True, run_id="abc123", resume="allow")
    assert fake.init_kwargs[-1]["id"] == "abc123"
    assert fake.init_kwargs[-1]["resume"] == "allow"


def test_wandb_logger_omits_identity_kwargs_when_not_given(monkeypatch):
    """Callers that pass neither must not start sending id=None to wandb."""
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    fake = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    WandbLogger("p", None, "n", {}, enabled=True)
    assert "id" not in fake.init_kwargs[-1]
    assert "resume" not in fake.init_kwargs[-1]


# --------------------------------------------------------------------- git ---

def _init_repo(path, with_remote=None):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    if with_remote:
        subprocess.run(["git", "remote", "add", "origin", str(with_remote)],
                       cwd=path, check=True)


def test_git_commit_and_push_with_local_remote(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, with_remote=remote)
    (repo / "f.txt").write_text("hello")

    ok = git_commit_and_push(str(repo), "test commit", ["f.txt"])
    assert ok is True
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True, check=True)
    assert "test commit" in log.stdout


def test_git_commit_and_push_noop_when_nothing_staged(tmp_path):
    repo = tmp_path / "repo2"
    repo.mkdir()
    _init_repo(repo)
    (repo / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    ok = git_commit_and_push(str(repo), "no changes", ["f.txt"])
    assert ok is False


def test_git_commit_and_push_skips_nonexistent_paths_but_stages_the_rest(tmp_path):
    """A path that doesn't exist yet (e.g. metrics.jsonl before the first
    write) must not sink a commit that has other real changes to push."""
    remote = tmp_path / "remote2.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo = tmp_path / "repo4"
    repo.mkdir()
    _init_repo(repo, with_remote=remote)
    (repo / "STATUS.md").write_text("status")

    ok = git_commit_and_push(str(repo), "partial commit",
                             ["STATUS.md", "runs/does-not-exist/metrics.jsonl"])
    assert ok is True
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True, check=True)
    assert "partial commit" in log.stdout


def test_git_commit_and_push_survives_missing_remote(tmp_path):
    """No 'origin' configured -> push fails -> function returns False, doesn't raise."""
    repo = tmp_path / "repo3"
    repo.mkdir()
    _init_repo(repo)
    (repo / "f.txt").write_text("hello")
    ok = git_commit_and_push(str(repo), "test commit", ["f.txt"])
    assert ok is False  # push failed, but no exception propagated


# ------------------------------------------------------------------- batch ---

def test_synthetic_batch_source_is_seed_deterministic():
    a = SyntheticBatchSource(vocab_size=100, micro_batch=2, device="cpu", seed=0)
    b = SyntheticBatchSource(vocab_size=100, micro_batch=2, device="cpu", seed=0)
    for _ in range(3):
        assert torch.equal(a.get_batch(8), b.get_batch(8))


# ----------------------------------------------------------- mixture source ---

def _write_source(root, name, n_tokens, shard_tokens=200):
    """Builds one source's shard dir under root/name, as dataprep.run_dataprep
    would, with distinguishable token ids (all multiples of a per-source base)
    so a batch's source can be identified from its content."""
    out_dir = os.path.join(str(root), name)
    w = ShardWriter(out_dir, shard_tokens=shard_tokens)
    w.write(list(range(n_tokens)))
    w.close()
    w.write_manifest({"eos_id": 0, "source_key": name})
    return out_dir


def test_cap_weights_by_epochs_leaves_well_stocked_sources_alone():
    w = {"a": 0.5, "b": 0.5}
    disk = {"a": 10_000, "b": 10_000}
    out = cap_weights_by_epochs(w, disk, total_run_tokens=10_000, max_epochs=4.0)
    assert out == pytest.approx({"a": 0.5, "b": 0.5})


def test_cap_weights_by_epochs_caps_a_tiny_source_and_redistributes():
    # "b" holds 1% of the run's tokens -> at 4 epochs it can supply 4% at most
    w = {"a": 0.5, "b": 0.5}
    disk = {"a": 1_000_000, "b": 10_000}
    out = cap_weights_by_epochs(w, disk, total_run_tokens=1_000_000, max_epochs=4.0)
    assert out["b"] == pytest.approx(0.04)
    assert out["a"] == pytest.approx(0.96)
    assert sum(out.values()) == pytest.approx(1.0)


def test_cap_weights_by_epochs_matches_the_real_everyday_conversations_case():
    """The case that motivated this: `everyday-conversations` is exhausted at
    403,573 tokens, so a 2% share of a 40B run is ~1,983 epochs."""
    total = 40_000_000_000
    w = {"everyday-conversations": 0.02, "fineweb-edu": 0.98}
    # fineweb-edu sized so its own cap does NOT bind (issue #4 4.2's 13B plan);
    # this isolates the single-starved-source case
    disk = {"everyday-conversations": 403_573, "fineweb-edu": 20_000_000_000}
    uncapped_epochs = w["everyday-conversations"] * total / disk["everyday-conversations"]
    assert uncapped_epochs > 1900

    out = cap_weights_by_epochs(w, disk, total_run_tokens=total, max_epochs=4.0)
    capped_epochs = out["everyday-conversations"] * total / disk["everyday-conversations"]
    assert capped_epochs == pytest.approx(4.0)
    assert sum(out.values()) == pytest.approx(1.0)


def test_cap_weights_by_epochs_water_fills_over_multiple_rounds():
    """Redistributing away from one capped source can push a second over its
    own cap -- the loop has to re-check, not cap once and stop."""
    # caps: big 40x (never binds), mid 0.4, small 0.04.
    # round 1 caps small; the redistribution then pushes mid from 0.33 to
    # 0.47, over *its* cap, which only a re-checking loop catches.
    w = {"big": 0.34, "mid": 0.33, "small": 0.33}
    disk = {"big": 10_000_000, "mid": 100_000, "small": 10_000}
    out = cap_weights_by_epochs(w, disk, total_run_tokens=1_000_000, max_epochs=4.0)
    assert out["small"] == pytest.approx(0.04)
    assert out["mid"] == pytest.approx(0.4)
    assert out["big"] == pytest.approx(0.56)
    assert sum(out.values()) == pytest.approx(1.0)


def test_cap_weights_by_epochs_keeps_the_mixture_when_every_source_is_short(capsys):
    """Whole corpus too small -> keep the target mixture and warn, rather than
    reweighting to the on-disk distribution. Reweighting would reproduce
    whatever skew exists (today: 21% FinePhrase, 5% FineWeb-Edu) which is worse
    for benchmarks than uniform over-repetition (issue #4 4.1)."""
    w = {"a": 0.25, "b": 0.75}
    disk = {"a": 20_000, "b": 10_000}
    out = cap_weights_by_epochs(w, disk, total_run_tokens=1_000_000, max_epochs=4.0)
    assert out == pytest.approx({"a": 0.25, "b": 0.75})   # mixture preserved
    err = capsys.readouterr().out
    assert "corpus is too small" in err
    assert "Build more data" in err          # loud, not silent (no silent caps)


def test_mixture_batch_source_applies_the_epoch_cap(tmp_path, capsys):
    for name, n_tokens in [("big", 4000), ("tiny", 10)]:
        d = tmp_path / name
        w = ShardWriter(str(d), shard_tokens=n_tokens)
        w.write([1] * n_tokens)
        w.close()
        w.write_manifest({"eos_id": 0})

    src = MixtureBatchSource(str(tmp_path), micro_batch=1, device="cpu",
                             weights={"big": 0.5, "tiny": 0.5},
                             total_run_tokens=1000, max_epochs=4.0)
    probs = dict(zip(src.names, src.probs))
    assert probs["tiny"] == pytest.approx(0.04)   # 4 * 10 / 1000
    assert probs["big"] == pytest.approx(0.96)
    assert "tiny share 0.5000 -> 0.0400" in capsys.readouterr().out


def test_mixture_batch_source_skips_the_cap_without_a_token_budget(tmp_path):
    """Callers that don't say how long the run is (tests, smoke) keep the old
    pure-share behaviour rather than getting a silent reweight."""
    for name in ("a", "b"):
        d = tmp_path / name
        w = ShardWriter(str(d), shard_tokens=100)
        w.write([1] * 100)
        w.close()
        w.write_manifest({"eos_id": 0})
    src = MixtureBatchSource(str(tmp_path), micro_batch=1, device="cpu",
                             weights={"a": 0.25, "b": 0.75})
    assert dict(zip(src.names, src.probs)) == pytest.approx({"a": 0.25, "b": 0.75})


def test_mixture_batch_source_raises_when_no_source_present(tmp_path):
    with pytest.raises(ValueError):
        MixtureBatchSource(str(tmp_path), micro_batch=2, device="cpu",
                           weights={"a": 0.5, "b": 0.5})


def test_mixture_batch_source_renormalizes_over_present_sources(tmp_path):
    _write_source(tmp_path, "a", n_tokens=500)
    # "b" and "c" are never written -- simulates dataprep still running or a
    # substituted/dropped source.
    src = MixtureBatchSource(str(tmp_path), micro_batch=2, device="cpu",
                             weights={"a": 0.5, "b": 0.3, "c": 0.2})
    assert src.names == ["a"]
    assert src.probs == [1.0]
    x = src.get_batch(8)
    assert x.shape == (2, 8)


def test_mixture_batch_source_samples_only_present_sources(tmp_path):
    _write_source(tmp_path, "a", n_tokens=500)
    _write_source(tmp_path, "b", n_tokens=500)
    src = MixtureBatchSource(str(tmp_path), micro_batch=2, device="cpu",
                             weights={"a": 0.9, "b": 0.1}, seed=0)
    seen = set()
    for _ in range(20):
        src.get_batch(8)
    # every draw came from one of the two configured sources' generators --
    # exercised indirectly via the deterministic seed below instead of
    # inspecting content, since ShardBatchSource.get_batch() shuffles.
    assert set(src.sources.keys()) == {"a", "b"}


def test_mixture_batch_source_sampling_matches_weights_with_fixed_seed():
    """rng.choices() is a plain random.Random(seed) stream, independent of
    the on-disk sources -- pin down the sampled-name distribution directly
    so a future refactor can't silently change the mixture proportions."""
    import random as random_module
    rng = random_module.Random(0)
    names = ["a", "b"]
    probs = [0.9, 0.1]
    picks = [rng.choices(names, weights=probs, k=1)[0] for _ in range(2000)]
    frac_a = picks.count("a") / len(picks)
    assert abs(frac_a - 0.9) < 0.03


def test_mixture_batch_source_rebuilds_on_seq_len_change(tmp_path):
    _write_source(tmp_path, "a", n_tokens=500)
    src = MixtureBatchSource(str(tmp_path), micro_batch=2, device="cpu",
                             weights={"a": 1.0})
    x8 = src.get_batch(8)
    x16 = src.get_batch(16)
    assert x8.shape == (2, 8)
    assert x16.shape == (2, 16)


def test_trainer_auto_detects_mixture_root(tmp_path):
    """Trainer.data_dir with no manifest.json directly inside it, but
    subdirectories that each have one, is treated as a mixture root rather
    than a single-source shard dir. Uses real daedalus.dataprep.MIXTURE keys
    since Trainer doesn't pass custom weights -- MixtureBatchSource falls
    back to MIXTURE's shares and matches subdirectory names against them."""
    data_root = tmp_path / "data"
    _write_source(data_root, "fineweb-edu", n_tokens=500, shard_tokens=200)
    _write_source(data_root, "stack-edu-python", n_tokens=500, shard_tokens=200)
    args = _tiny_args(tmp_path / "run", max_steps=2, data_dir=str(data_root))
    trainer = Trainer(args)
    assert isinstance(trainer.batch_source, MixtureBatchSource)
    assert set(trainer.batch_source.names) == {"fineweb-edu", "stack-edu-python"}


# ------------------------------------------------------- mixture reporting ---
# Added 2026-08-10. A DNS blip left dclm-baseline at 1.57B of 2.25B, and
# because cap_weights_by_epochs clamps a short source to 4 x on_disk /
# run_tokens, a 40B hero would have sampled 53.2% web against a 60% target.
# The only trace was one print() at startup. These pin the summary that now
# carries it into the W&B run config.
# See runs/preflight/mixture-cap-vs-hero-budget.md.

def test_mixture_summary_reports_the_cap_it_applied(tmp_path):
    _write_source(tmp_path, "big", n_tokens=4000, shard_tokens=1000)
    _write_source(tmp_path, "tiny", n_tokens=10, shard_tokens=10)
    src = MixtureBatchSource(str(tmp_path), micro_batch=1, device="cpu",
                             weights={"big": 0.5, "tiny": 0.5},
                             total_run_tokens=1000, max_epochs=4.0)
    s = src.mixture_summary()

    assert s["capped_sources"] == ["tiny"]
    assert s["per_source"]["tiny"]["target_share"] == pytest.approx(0.5)
    assert s["per_source"]["tiny"]["effective_share"] == pytest.approx(0.04)
    assert s["per_source"]["tiny"]["capped"] is True
    assert s["per_source"]["big"]["capped"] is False
    # tiny lost 0.46 and big gained it back: L1 counts both sides.
    assert s["l1_skew_pts"] == pytest.approx(92.0, abs=0.01)
    # a capped source sits exactly at the epoch limit by construction
    assert s["per_source"]["tiny"]["epochs"] == pytest.approx(4.0)
    assert s["total_run_tokens"] == 1000
    assert s["total_tokens_on_disk"] == 4010


def test_mixture_summary_is_clean_when_no_cap_binds(tmp_path):
    """The floor case: nothing capped means zero skew and an empty list, so a
    non-zero l1_skew_pts on the dashboard always means something real."""
    _write_source(tmp_path, "a", n_tokens=5000, shard_tokens=1000)
    _write_source(tmp_path, "b", n_tokens=5000, shard_tokens=1000)
    src = MixtureBatchSource(str(tmp_path), micro_batch=1, device="cpu",
                             weights={"a": 0.5, "b": 0.5},
                             total_run_tokens=1000, max_epochs=4.0)
    s = src.mixture_summary()
    assert s["capped_sources"] == []
    assert s["l1_skew_pts"] == pytest.approx(0.0)
    assert all(not r["capped"] for r in s["per_source"].values())
    assert s["per_source"]["a"]["epochs"] == pytest.approx(0.1)   # 0.5*1000/5000


def test_a_large_mixture_skew_warns_and_a_normal_one_stays_quiet(tmp_path, capsys):
    """The `capped:` line is not graded -- at hero's 40B the skew is 3.99 pts
    and at 50B it is 29.91, and both printed the same shape of line.

    The 50B case is the one that matters: the epoch cap binds hard, the web
    backbone collapses (fineweb-edu 37.5% -> 30.0%), and it is reached by
    raising a *token budget*, which does not look like a data change. Both
    sides are pinned here so the threshold cannot be silently widened past the
    budget it was chosen to protect.
    """
    # Real blueprint source names: the Trainer looks its weights up from
    # dataprep.MIXTURE, so a mixture root has to use them.
    skewed = tmp_path / "skewed-corpus"
    _write_source(skewed, "fineweb-edu", n_tokens=4000, shard_tokens=1000)
    _write_source(skewed, "finewiki-en", n_tokens=10, shard_tokens=10)

    Trainer(_tiny_args(tmp_path / "skewed-run", data_dir=str(skewed),
                       total_tokens=2000, max_steps=1))
    loud = capsys.readouterr().out
    assert "WARNING" in loud and "from the blueprint target" in loud
    assert "build more data" in loud

    # Nothing capped -> no warning at all. Without this the test would pass
    # with the threshold set to zero.
    even = tmp_path / "even-corpus"
    _write_source(even, "fineweb-edu", n_tokens=5000, shard_tokens=1000)
    _write_source(even, "finewiki-en", n_tokens=5000, shard_tokens=1000)
    Trainer(_tiny_args(tmp_path / "even-run", data_dir=str(even),
                       total_tokens=1000, max_steps=1))
    assert "from the blueprint target" not in capsys.readouterr().out


def test_the_skew_threshold_is_quiet_at_every_budget_the_corpus_was_built_for():
    """Guards the constant itself against the real numbers it was chosen from:
    10B-40B measure 3.97-3.99 pts and must not warn; 50B measures 29.91 and
    must. See runs/preflight/mixture-vs-token-budget.md."""
    from train import MAX_MIXTURE_SKEW_PTS
    for skew in (3.97, 3.98, 3.99):
        assert skew < MAX_MIXTURE_SKEW_PTS
    assert 29.91 > MAX_MIXTURE_SKEW_PTS


def test_mixture_summary_without_a_token_budget_reports_no_cap(tmp_path):
    """total_run_tokens=None skips capping entirely, so the summary must not
    invent an epochs figure it cannot compute."""
    _write_source(tmp_path, "a", n_tokens=500)
    _write_source(tmp_path, "b", n_tokens=500)
    src = MixtureBatchSource(str(tmp_path), micro_batch=1, device="cpu",
                             weights={"a": 0.25, "b": 0.75})
    s = src.mixture_summary()
    assert s["capped_sources"] == []
    assert s["total_run_tokens"] is None
    assert "epochs" not in s["per_source"]["a"]
    assert s["per_source"]["a"]["target_share"] == pytest.approx(0.25)


def test_trainer_puts_the_mixture_in_the_wandb_config(tmp_path):
    """The whole point: recoverable from the dashboard, not just a log line."""
    data_root = tmp_path / "data"
    _write_source(data_root, "fineweb-edu", n_tokens=4000, shard_tokens=1000)
    _write_source(data_root, "stack-edu-python", n_tokens=10, shard_tokens=10)
    args = _tiny_args(tmp_path / "run", max_steps=2, data_dir=str(data_root))
    args.total_tokens = 1000
    trainer = Trainer(args)

    cfg = trainer.wandb_config
    assert "data_mixture" in cfg
    mix = cfg["data_mixture"]
    assert mix["capped_sources"] == ["stack-edu-python"]
    assert mix["l1_skew_pts"] > 0
    assert mix["per_source"]["fineweb-edu"]["tokens_on_disk"] == 4000


def test_trainer_omits_the_mixture_block_for_a_single_source_dir(tmp_path):
    """A single-source run has no mixture to report; the key must be absent
    rather than present-and-empty, which would read as 'no skew'."""
    data_dir = _write_source(tmp_path, "a", n_tokens=500, shard_tokens=200)
    args = _tiny_args(tmp_path / "run", max_steps=2, data_dir=data_dir)
    trainer = Trainer(args)
    assert "data_mixture" not in trainer.wandb_config


def test_trainer_uses_shard_batch_source_for_single_source_dir(tmp_path):
    data_dir = _write_source(tmp_path, "a", n_tokens=500, shard_tokens=200)
    args = _tiny_args(tmp_path / "run", max_steps=2, data_dir=data_dir)
    trainer = Trainer(args)
    assert isinstance(trainer.batch_source, ShardBatchSource)


# ------------------------------------------------- sampler stream position ---
# The bug these pin down: `_rebuild` used to seed from the constant `self.seed`,
# so a resumed run replayed the window indices it had already trained on. A
# `hero` interrupted halfway would have spent its second half on the first
# half's data, silently.

def _first_batches(src, n=6, seq_len=8):
    return [src.get_batch(seq_len).clone() for _ in range(n)]


def test_stream_seed_is_stable_across_processes():
    """The seed derivation must not depend on anything per-process -- a resume
    is a new process, which is the one case where a `hash()`-based seed would
    silently stop being reproducible."""
    from train import stream_seed
    expect = [stream_seed(0, 0), stream_seed(0, 1), stream_seed(7, 10 ** 9)]
    code = ("import train;"
            "print([train.stream_seed(0,0),train.stream_seed(0,1),"
            "train.stream_seed(7,10**9)])")
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    assert eval(out.stdout.strip()) == expect
    assert all(s >= 0 for s in expect)          # manual_seed rejects negatives
    assert len(set(expect)) == 3


def test_stream_seed_separates_adjacent_positions():
    """tokens_seen values one step apart must not give neighbouring streams."""
    from train import stream_seed
    a, b = stream_seed(0, 1_000_000), stream_seed(0, 1_000_001)
    assert abs(a - b) > 10 ** 12


def test_shard_batch_source_position_changes_the_window_stream(tmp_path):
    d = _write_source(tmp_path, "a", n_tokens=4000, shard_tokens=4000)
    fresh = ShardBatchSource(d, micro_batch=2, device="cpu", seed=0)
    resumed = ShardBatchSource(d, micro_batch=2, device="cpu", seed=0)
    resumed.set_position(5_000_000_000)
    assert not all(torch.equal(x, y)
                   for x, y in zip(_first_batches(fresh), _first_batches(resumed)))


def test_shard_batch_source_position_is_deterministic(tmp_path):
    """Different from the pre-restart stream, but still reproducible: two
    resumes from the same checkpoint must train on the same data."""
    d = _write_source(tmp_path, "a", n_tokens=4000, shard_tokens=4000)

    def at(pos):
        s = ShardBatchSource(d, micro_batch=2, device="cpu", seed=0)
        s.set_position(pos)
        return _first_batches(s)

    assert all(torch.equal(x, y) for x, y in zip(at(1_234_567), at(1_234_567)))
    assert not all(torch.equal(x, y) for x, y in zip(at(1_234_567), at(7_654_321)))


def test_seq_len_rebuild_does_not_replay_the_same_windows(tmp_path):
    """The seq_len ramp rebuilds the loader mid-run. Re-seeding from a constant
    made each rebuild restart the *same* index stream, so the run re-read its
    opening windows every time the ramp moved."""
    d = _write_source(tmp_path, "a", n_tokens=4000, shard_tokens=4000)
    src = ShardBatchSource(d, micro_batch=2, device="cpu", seed=0)
    first = _first_batches(src, n=4, seq_len=8)
    _first_batches(src, n=1, seq_len=16)        # ramp moves -> rebuild
    again = _first_batches(src, n=4, seq_len=8)  # ramp returns -> rebuild
    assert not all(torch.equal(x, y) for x, y in zip(first, again))


def test_mixture_batch_source_position_moves_every_sub_source(tmp_path):
    _write_source(tmp_path, "a", n_tokens=2000, shard_tokens=2000)
    _write_source(tmp_path, "b", n_tokens=2000, shard_tokens=2000)
    src = MixtureBatchSource(str(tmp_path), micro_batch=2, device="cpu",
                             weights={"a": 0.5, "b": 0.5}, seed=0)
    src.set_position(3_000_000_000)
    assert all(s._stream == 3_000_000_000 for s in src.sources.values())
    # the source-pick stream moved too, not just the per-source samplers
    picks = [src.rng.choices(["a", "b"], k=1)[0] for _ in range(30)]
    assert picks != [random.Random(0).choices(["a", "b"], k=1)[0] for _ in range(30)]


def test_trainer_positions_the_sampler_after_resume(tmp_path):
    """End to end: the Trainer must apply the checkpoint's tokens_seen to the
    batch source, or none of the above helps the run that actually matters."""
    data_dir = _write_source(tmp_path, "a", n_tokens=4000, shard_tokens=4000)
    args = _tiny_args(tmp_path / "run", max_steps=2, data_dir=data_dir)
    fresh = Trainer(args)
    assert fresh.batch_source._stream == 0

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(str(ckpt), fresh.model, fresh.muon, fresh.adamw,
                    step=10, tokens_seen=8_000_000, cfg=fresh.cfg)
    resumed = Trainer(_tiny_args(tmp_path / "run2", max_steps=2,
                                 data_dir=data_dir, resume=str(ckpt)))
    assert resumed.tokens_seen == 8_000_000
    assert resumed.batch_source._stream == 8_000_000
    assert not all(torch.equal(x, y) for x, y in
                   zip(_first_batches(fresh.batch_source),
                       _first_batches(resumed.batch_source)))


class FixedBatchSource:
    """Replays a pre-built list of batches, one per get_batch() call --
    used to test resume against a controlled, known data sequence."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.i = 0

    def get_batch(self, seq_len):
        b = self.batches[self.i]
        self.i += 1
        return b


# ------------------------------------------------------------------ trainer ---

def _tiny_args(run_dir, **overrides):
    seq_len = overrides.pop("seq_len", 16)
    micro_batch = overrides.pop("micro_batch", 2)
    kwargs = dict(
        run_name="test", config="tiny", device="cpu", compile=False,
        micro_batch=micro_batch, seq_start=seq_len, seq_end=seq_len,
        tok_start=micro_batch * seq_len, tok_end=micro_batch * seq_len,
        wandb_enabled=False, seed=0, run_dir=str(run_dir),
        ckpt_every_sec=1e9, push_every_sec=1e9, metrics_every_steps=1,
        log_every_steps=1000000,
    )
    kwargs.update(overrides)
    return TrainArgs(**kwargs)


def test_trainer_runs_and_writes_metrics_and_checkpoint(tmp_path):
    args = _tiny_args(tmp_path / "run", max_steps=4)
    trainer = Trainer(args)
    trainer.fit()

    metrics_path = tmp_path / "run" / "metrics.jsonl"
    assert metrics_path.exists()
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) == 4
    assert trainer.step == 4
    assert trainer.tokens_seen == 4 * args.micro_batch * args.seq_start
    assert os.path.exists(trainer.ckpt_path)  # forced checkpoint at fit() end

    records = [json.loads(l) for l in lines]
    assert all(r["tok_per_sec"] > 0 for r in records)
    per_step_tokens = args.micro_batch * args.seq_start
    # single-step window each time (metrics_every_steps=1) -> tok_per_sec is
    # roughly per_step_tokens / (wall time for one step), so it should be a
    # sane finite number, not a bogus cumulative-since-epoch spike.
    assert all(r["tok_per_sec"] < per_step_tokens * 10_000 for r in records)


def test_trainer_pidfile_cleaned_up_after_fit(tmp_path):
    """watchdog.py reads runs/<name>/train.pid to check the training process
    is alive; it must not be left behind once fit() returns normally."""
    args = _tiny_args(tmp_path / "run", max_steps=2)
    trainer = Trainer(args)
    trainer.fit()
    assert not os.path.exists(os.path.join(str(tmp_path / "run"), "train.pid"))


def test_tok_per_sec_is_windowed_not_cumulative_after_resume(tmp_path):
    """A resumed run's first tok_per_sec sample must reflect only the new
    steps taken since resume, not (tokens_seen so far) / (wall time since
    process start) -- which would report a huge bogus spike."""
    seq_len, micro_batch = 16, 2
    args1 = _tiny_args(tmp_path / "run1", max_steps=4, seq_len=seq_len,
                       micro_batch=micro_batch)
    t1 = Trainer(args1)
    t1.fit()
    ckpt = t1.ckpt_path

    args2 = _tiny_args(tmp_path / "run2", max_steps=6, seq_len=seq_len,
                       micro_batch=micro_batch, resume=ckpt)
    t2 = Trainer(args2)
    assert t2.tokens_seen == t1.tokens_seen  # resumed with a "head start"
    t2.fit()

    lines = (tmp_path / "run2" / "metrics.jsonl").read_text().strip().splitlines()
    records = [json.loads(l) for l in lines]
    per_step_tokens = micro_batch * seq_len
    assert all(r["tok_per_sec"] < per_step_tokens * 10_000 for r in records)


def test_resume_reproduces_uninterrupted_training(tmp_path):
    seq_len, micro_batch, n_steps, k = 16, 2, 10, 4
    cfg = PRESETS["tiny"]
    torch.manual_seed(123)
    batches = [torch.randint(0, cfg.vocab_size, (micro_batch, seq_len))
              for _ in range(n_steps)]

    # uninterrupted reference run over all n_steps
    ref = Trainer(_tiny_args(tmp_path / "ref", max_steps=n_steps, seq_len=seq_len,
                             micro_batch=micro_batch))
    ref.batch_source = FixedBatchSource(batches)
    for _ in range(n_steps):
        ref.train_step()

    # first k steps only, then checkpoint (simulating an interruption)
    part1 = Trainer(_tiny_args(tmp_path / "part1", max_steps=n_steps, seq_len=seq_len,
                               micro_batch=micro_batch))
    part1.batch_source = FixedBatchSource(batches[:k])
    for _ in range(k):
        part1.train_step()
    ckpt_path = save_checkpoint(str(tmp_path / "interrupt.pt"), part1.model,
                                part1.muon, part1.adamw, part1.step,
                                part1.tokens_seen, part1.cfg)

    # fresh process resumes from the checkpoint and finishes the remaining steps
    resumed = Trainer(_tiny_args(tmp_path / "resumed", max_steps=n_steps, seq_len=seq_len,
                                 micro_batch=micro_batch, resume=str(ckpt_path)))
    assert resumed.step == k
    assert resumed.tokens_seen == part1.tokens_seen
    resumed.batch_source = FixedBatchSource(batches[k:])
    for _ in range(n_steps - k):
        resumed.train_step()

    assert resumed.step == ref.step == n_steps
    for p1, p2 in zip(ref.model.parameters(), resumed.model.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "resumed weights diverged from reference"


# --------------------------------------------------------------- CLI args ---

def _val_shards(tmp_path, n_tokens=4096):
    d = tmp_path / "holdout"
    w = ShardWriter(str(d), shard_tokens=n_tokens)
    w.write(list(torch.randint(0, PRESETS["tiny"].vocab_size, (n_tokens,)).tolist()))
    w.close()
    w.write_manifest({"eos_id": 0})
    return str(d)


class _ByteTokenizer:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_val_bpb_is_logged_on_its_own_cadence(tmp_path, monkeypatch):
    """AGENT.md SS5.2 requires val_bpb in metrics.jsonl; it used to be
    hard-coded None on every record, so a multi-day run had no val curve."""
    val_dir = _val_shards(tmp_path)
    args = _tiny_args(tmp_path / "run", max_steps=4, seq_len=16, micro_batch=2)
    args.val_dir = val_dir
    args.val_every_steps = 2
    args.val_batches = 1
    args.val_batch_size = 2
    args.seq_end = 16
    args.metrics_every_steps = 1

    t = Trainer(args)
    t._tokenizer = _ByteTokenizer()
    monkeypatch.setattr("daedalus.data.get_tokenizer", lambda *a, **k: _ByteTokenizer())
    t.batch_source = FixedBatchSource(
        [torch.randint(0, t.cfg.vocab_size, (2, 16)) for _ in range(4)])
    t.fit()

    records = [json.loads(l) for l in
              (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()]
    by_step = {r["step"]: r["val_bpb"] for r in records}
    assert by_step[1] is None and by_step[3] is None      # not due
    for step in (2, 4):
        assert by_step[step] is not None and math.isfinite(by_step[step])
        assert by_step[step] > 0


def _mixture_val_shards(tmp_path, names=("source-a", "source-b"), n_tokens=4096):
    """A `make_mixture_holdout_split`-shaped root: per-source subdirectories,
    each with its own manifest.json, and no manifest at the top level."""
    root = tmp_path / "mixholdout"
    for name in names:
        d = root / name
        w = ShardWriter(str(d), shard_tokens=n_tokens)
        w.write(list(torch.randint(0, PRESETS["tiny"].vocab_size,
                                   (n_tokens,)).tolist()))
        w.close()
        w.write_manifest({"eos_id": 0})
    return str(root)


def test_val_bpb_reads_a_mixture_holdout_root(tmp_path, monkeypatch, capsys):
    """`hero.py` carves a per-source holdout and passes the *root* as
    --val-dir. `evaluate_bpb` opens `<dir>/manifest.json`, which a mixture root
    does not have, so it raised FileNotFoundError -- and `_val_bpb` swallows
    every exception by design. The four-day run would have logged
    `val_bpb: null` at every interval behind a WARNING, with no val curve at
    all and nothing for the watchdog to act on."""
    val_dir = _mixture_val_shards(tmp_path)
    args = _tiny_args(tmp_path / "run", max_steps=2, seq_len=16, micro_batch=2)
    args.val_dir = val_dir
    args.val_every_steps = 1
    args.val_batches = 1
    args.val_batch_size = 2
    args.seq_end = 16
    args.metrics_every_steps = 1

    t = Trainer(args)
    t._tokenizer = _ByteTokenizer()
    monkeypatch.setattr("daedalus.data.get_tokenizer", lambda *a, **k: _ByteTokenizer())
    t.batch_source = FixedBatchSource(
        [torch.randint(0, t.cfg.vocab_size, (2, 16)) for _ in range(2)])
    t.fit()

    assert "val_bpb failed" not in capsys.readouterr().out
    records = [json.loads(l) for l in
              (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()]
    assert records, "no metrics written"
    for r in records:
        assert r["val_bpb"] is not None and math.isfinite(r["val_bpb"])
        assert r["val_bpb"] > 0


def test_val_weights_come_from_the_sampler_not_the_holdout(tmp_path):
    """val_bpb must weight sources the way the sampler draws them. Weighting by
    holdout size instead measures a blend nobody trained on -- `make_holdout_split`
    reserves whole shard files, so a source's holdout share is set by the size of
    its trailing partial shard."""
    args = _tiny_args(tmp_path / "run", max_steps=1, seq_len=16, micro_batch=2)
    t = Trainer(args)

    class _FakeMixture:
        names = ["source-a", "source-b"]
        probs = [0.25, 0.75]

    t.batch_source = _FakeMixture()
    assert t._val_weights() == {"source-a": 0.25, "source-b": 0.75}

    # A single-source run has no mixture weights; fall back rather than invent.
    t.batch_source = FixedBatchSource([torch.zeros(2, 16, dtype=torch.long)])
    assert t._val_weights() is None


def test_val_bpb_failure_does_not_kill_the_run(tmp_path, capsys):
    """A broken holdout dir must degrade like a W&B outage, not end a $44 run."""
    args = _tiny_args(tmp_path / "run", max_steps=2, seq_len=16, micro_batch=2)
    args.val_dir = str(tmp_path / "does-not-exist")
    args.val_every_steps = 1
    args.metrics_every_steps = 1

    t = Trainer(args)
    t.batch_source = FixedBatchSource(
        [torch.randint(0, t.cfg.vocab_size, (2, 16)) for _ in range(2)])
    t.fit()   # must not raise

    assert "val_bpb failed" in capsys.readouterr().out
    records = [json.loads(l) for l in
              (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(records) == 2
    assert all(r["val_bpb"] is None for r in records)


def test_val_bpb_off_by_default(tmp_path):
    args = _tiny_args(tmp_path / "run", max_steps=1, seq_len=16, micro_batch=2)
    args.metrics_every_steps = 1
    assert args.val_dir is None
    t = Trainer(args)
    t.batch_source = FixedBatchSource([torch.randint(0, t.cfg.vocab_size, (2, 16))])
    t.fit()
    record = json.loads(
        (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()[0])
    assert record["val_bpb"] is None


def test_parse_args_val_dir():
    args = parse_args(["--run-name", "x", "--val-dir", "/data/holdout",
                       "--val-every-steps", "50"])
    assert args.val_dir == "/data/holdout"
    assert args.val_every_steps == 50


def test_parse_args_muon_lr_and_tags_default():
    args = parse_args(["--run-name", "x"])
    assert args.muon_lr == 0.02
    assert args.adam_lr == 3e-4
    assert args.tags is None


def test_parse_args_muon_lr_override_for_sweep():
    args = parse_args(["--run-name", "sweep-lr0.01", "--muon-lr", "0.01",
                       "--tags", "sweep"])
    assert args.muon_lr == 0.01
    assert args.tags == ["sweep"]


def test_trainer_wandb_tags_include_run_name_and_extra_tags(tmp_path, monkeypatch):
    calls = {}

    class _RecordingWandbLogger:
        def __init__(self, *a, tags=None, **kw):
            calls["tags"] = tags

        def log(self, *a, **kw):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("train.WandbLogger", _RecordingWandbLogger)
    args = _tiny_args(tmp_path / "tagged", max_steps=1, tags=["sweep"])
    Trainer(args)
    assert calls["tags"] == [args.run_name, "sweep"]


# ------------------------------------------------- gradient accumulation ---
#
# Issue #4 section 2.3 listed this as "believed, not proven": only the
# step-count helper was tested, and nothing asserted that N accumulated
# micro-batches produce the same gradient as one backward over the
# concatenated batch. Every run this project does is accumulated (a 512k-token
# batch never fits), so if the arithmetic were wrong every result would be
# quietly wrong with no error anywhere.


class _ScriptedBatchSource:
    """Hands out a fixed list of batches, so both halves of the comparison
    train on exactly the same data in the same order."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.i = 0

    def get_batch(self, seq_len):
        b = self.batches[self.i % len(self.batches)]
        self.i += 1
        return b


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def test_grad_accumulation_equals_one_backward_over_the_concatenated_batch(tmp_path):
    torch.manual_seed(0)
    micro_batch, seq_len, accum = 2, 16, 2
    args = TrainArgs(
        run_name="accum", config="tiny", max_steps=1, micro_batch=micro_batch,
        seq_start=seq_len, seq_end=seq_len,
        # tok_start/tok_end pick the accumulation count: 64 / (2 * 16) = 2.
        tok_start=micro_batch * seq_len * accum, tok_end=micro_batch * seq_len * accum,
        compile=False, device="cpu", wandb_enabled=False,
        grad_clip=1e9,                      # make clipping a no-op
        run_dir=str(tmp_path / "accum"), ckpt_every_sec=1e9, push_every_sec=1e9,
    )
    t = Trainer(args)

    vocab = t.cfg.vocab_size
    micros = [torch.randint(0, vocab, (micro_batch, seq_len)) for _ in range(accum)]
    t.batch_source = _ScriptedBatchSource(micros)
    # Freeze the weights: we are comparing gradients, not trajectories.
    t.muon.step = lambda *a, **k: None
    t.adamw.step = lambda *a, **k: None

    stats = t.train_step()
    assert stats["accum"] == accum, stats
    accumulated = _grads(t.model)
    assert accumulated, "no gradients were produced"

    # One backward over the same tokens, concatenated.
    t.muon.zero_grad(set_to_none=True)
    t.adamw.zero_grad(set_to_none=True)
    big = torch.cat(micros, dim=0)
    _, loss, _ = t.model(big, targets=big)
    loss.backward()
    single = _grads(t.model)

    assert set(accumulated) == set(single)
    for name in single:
        a, b = accumulated[name], single[name]
        scale = max(float(b.abs().max()), 1e-8)
        assert torch.allclose(a, b, atol=2e-5 * scale, rtol=2e-4), (
            name, float((a - b).abs().max()), scale)


def test_grad_accumulation_loss_is_the_mean_not_the_sum(tmp_path):
    """`loss_sum` divided by `accum` must be the batch mean. Reporting the sum
    would make every logged loss `accum`x too large -- and `accum` changes
    during the batch ramp, so the curve would bend for no real reason."""
    micro_batch, seq_len, accum = 2, 16, 4
    args = TrainArgs(
        run_name="accum2", config="tiny", max_steps=1, micro_batch=micro_batch,
        seq_start=seq_len, seq_end=seq_len,
        tok_start=micro_batch * seq_len * accum, tok_end=micro_batch * seq_len * accum,
        compile=False, device="cpu", wandb_enabled=False,
        run_dir=str(tmp_path / "accum2"), ckpt_every_sec=1e9, push_every_sec=1e9,
    )
    t = Trainer(args)
    torch.manual_seed(1)
    micros = [torch.randint(0, t.cfg.vocab_size, (micro_batch, seq_len))
              for _ in range(accum)]
    t.batch_source = _ScriptedBatchSource(micros)
    t.muon.step = lambda *a, **k: None
    t.adamw.step = lambda *a, **k: None

    stats = t.train_step()
    assert stats["accum"] == accum

    with torch.no_grad():
        per_micro = [float(t.model(m, targets=m)[1]) for m in micros]
    assert stats["loss"] == pytest.approx(sum(per_micro) / accum, rel=1e-5)
    # Sanity: a fresh model's loss is ~ln(vocab), not accum x that.
    assert stats["loss"] < 2 * math.log(t.cfg.vocab_size)


def test_tokens_seen_counts_every_micro_batch(tmp_path):
    """`tokens_seen` drives the batch/seq ramps, the WSD schedule and the
    stopping condition. Counting one micro-batch per step instead of `accum`
    of them would stretch a 40B run by the accumulation factor."""
    micro_batch, seq_len, accum = 2, 16, 3
    args = TrainArgs(
        run_name="accum3", config="tiny", max_steps=1, micro_batch=micro_batch,
        seq_start=seq_len, seq_end=seq_len,
        tok_start=micro_batch * seq_len * accum, tok_end=micro_batch * seq_len * accum,
        compile=False, device="cpu", wandb_enabled=False,
        run_dir=str(tmp_path / "accum3"), ckpt_every_sec=1e9, push_every_sec=1e9,
    )
    t = Trainer(args)
    t.muon.step = lambda *a, **k: None
    t.adamw.step = lambda *a, **k: None
    before = t.tokens_seen
    stats = t.train_step()
    assert t.tokens_seen - before == micro_batch * seq_len * stats["accum"]


# --- end-to-end resume through the real CLI --------------------------------

def test_cli_device_flag_overrides_and_defaults():
    """--device exists so the subprocess test below can pin cpu. Unspecified
    must fall through to TrainArgs' own cuda-if-available default rather than
    being pinned to None."""
    assert parse_args(["--run-name", "r", "--device", "cpu"]).device == "cpu"
    default = parse_args(["--run-name", "r"]).device
    assert default == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.slow
def test_resume_across_a_real_subprocess(tmp_path):
    """Resume through `python train.py --resume`, in a fresh interpreter.

    Every other resume test builds a second Trainer in the same process. A
    real resume is a new process, which is precisely why the sampler seeding
    uses splitmix64 rather than hash() -- and it is the path abl_arch.py now
    takes when it retries a crashed arm, so it is load-bearing rather than
    hypothetical. The class of bug this catches is the one that has bitten
    this project twice: code that works in-process and fails only once a real
    subprocess is involved.

    cwd is tmp_path so nothing here can read or write the repo's own runs/.
    """
    shard_dir = tmp_path / "shards"
    w = ShardWriter(str(shard_dir), shard_tokens=8192)
    w.write(list(torch.randint(0, PRESETS["tiny"].vocab_size, (8192,)).tolist()))
    w.close()
    w.write_manifest({"eos_id": 0})

    base = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "train.py"),
        "--run-name", "e2e", "--config", "tiny", "--data-dir", str(shard_dir),
        "--device", "cpu", "--no-compile", "--no-wandb",
        "--micro-batch", "2", "--seq-start", "16", "--seq-end", "16",
        "--tok-start", "32", "--tok-end", "32", "--metrics-every-steps", "1"]
    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))}

    first = subprocess.run(base + ["--max-steps", "4"], cwd=str(tmp_path),
                           env=env, capture_output=True, text=True, timeout=600)
    assert first.returncode == 0, first.stderr[-3000:]

    ckpt = tmp_path / "runs" / "e2e" / "checkpoint.pt"
    assert ckpt.exists(), "fit() must force a checkpoint on normal completion"
    metrics_path = tmp_path / "runs" / "e2e" / "metrics.jsonl"
    steps_before = [json.loads(l)["step"] for l in
                    metrics_path.read_text().splitlines() if l.strip()]
    assert max(steps_before) == 4

    second = subprocess.run(base + ["--max-steps", "8", "--resume", str(ckpt)],
                            cwd=str(tmp_path), env=env, capture_output=True,
                            text=True, timeout=600)
    assert second.returncode == 0, second.stderr[-3000:]
    assert "resumed from" in second.stdout, second.stdout[-3000:]

    steps_after = [json.loads(l)["step"] for l in
                   metrics_path.read_text().splitlines() if l.strip()]
    # Continued rather than restarted. This is the assertion that actually
    # discriminates: a resume that was silently ignored would also exit 0,
    # also finish at step 8 and also leave a longer metrics.jsonl -- it would
    # just replay 1..8 instead of appending 5..8.
    assert steps_after[:len(steps_before)] == steps_before
    assert steps_after[len(steps_before):] == [5, 6, 7, 8]
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert ck["step"] == 8
    assert ck["tokens_seen"] > 0


def test_train_step_accepts_a_masked_label_batch(tmp_path):
    """A batch source may yield (x, y) so post.py's SFT can reuse this
    Trainer. Pretraining sources still yield a bare tensor and must behave
    exactly as before -- that path runs sweep and abl-arch tonight."""
    class PairSource:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.calls = 0

        def get_batch(self, seq_len):
            self.calls += 1
            return self.x, self.y

    torch.manual_seed(0)
    x = torch.randint(1, PRESETS["tiny"].vocab_size, (2, 16))
    y = x.clone()
    y[:, :8] = -100  # supervise only the second half

    args = _tiny_args(tmp_path / "sft", max_steps=1, seq_len=16, micro_batch=2)
    t = Trainer(args)
    t.batch_source = PairSource(x, y)
    stats = t.train_step()

    assert t.batch_source.calls >= 1
    assert not stats["skipped"]
    assert math.isfinite(stats["loss"])

    # Fully-masked labels make the loss independent of the model, which is the
    # cheapest proof that `y` -- not `x` -- is what reaches the loss.
    all_masked = torch.full_like(y, -100)
    t2 = Trainer(_tiny_args(tmp_path / "sft2", max_steps=1, seq_len=16, micro_batch=2))
    t2.batch_source = PairSource(x, all_masked)
    assert t2.train_step()["loss"] == 0.0


# ------------------------------------------------- the divergence guard ---
# train.py's in-loop NaN guard is the first line of defence for a ~92 h run
# (watchdog.py is the second), and it had no test at all -- only the negative
# assertion `not stats["skipped"]` elsewhere in this file.

def _nan_loss_trainer(tmp_path, name):
    t = Trainer(_tiny_args(tmp_path / name, max_steps=10, seq_len=16,
                           micro_batch=2))
    t.batch_source = SyntheticBatchSource(PRESETS["tiny"].vocab_size, 2, "cpu",
                                          seed=0)
    real = t.net

    def nan_forward(x, targets=None):
        logits, loss, aux = real(x, targets=targets)
        return logits, loss * float("nan"), aux

    t.net = nan_forward
    return t, real


def test_a_non_finite_loss_skips_the_update_instead_of_crashing(tmp_path):
    t, real = _nan_loss_trainer(tmp_path, "nan1")
    step_before, tokens_before = t.step, t.tokens_seen

    # Dirty the gradients first. Without this the grad assertion below is
    # vacuous -- a fresh model's grads are already None, so the test would
    # pass with the zero_grad calls deleted. It is also the real shape of the
    # bug being guarded against: at accum > 1 the NaN arrives on a later
    # micro-batch, after earlier ones have already accumulated real gradients
    # that must be thrown away rather than stepped.
    x = t.batch_source.get_batch(16)
    _, warmup_loss, _ = real(x, targets=x)
    warmup_loss.backward()
    assert any(p.grad is not None for p in t.model.parameters())
    before = [p.detach().clone() for p in t.model.parameters()]

    stats = t.train_step()

    assert stats["skipped"] is True
    assert math.isnan(stats["loss"])
    # The optimizers must not have stepped on NaN gradients: a single stepped
    # NaN poisons every weight it touches and no later batch can recover it.
    assert t.step == step_before
    assert t.tokens_seen == tokens_before
    assert all(p.grad is None for p in t.model.parameters())
    assert all(torch.equal(a, b)
               for a, b in zip(before, t.model.parameters()))
    assert all(torch.isfinite(p).all() for p in t.model.parameters())


def test_a_skipped_step_keeps_the_run_haltable(tmp_path):
    """The property that makes a NaN *reachable* by watchdog.py.

    A skipped step does not advance `self.step`, so `log_step`'s
    `step % metrics_every_steps` gate keeps whatever answer it had. That is
    what decides which of the two halt paths fires on a persistent NaN:

      - on a metrics step, the NaN reaches `metrics.jsonl`, where
        `watchdog.detect_divergence` sees a non-finite loss and halts within a
        poll (~2 min);
      - otherwise nothing is written, `metrics.jsonl`'s mtime freezes, and
        `watchdog.detect_stall` halts at 30 min.

    Both terminate, so the guard is safe either way -- the reason to pin it is
    that the *first* path only exists because the NaN is written rather than
    swallowed. Somebody "fixing" the duplicate-step rows by suppressing them
    would silently convert every divergence into the 30-minute path.
    """
    from watchdog import detect_divergence
    t, _ = _nan_loss_trainer(tmp_path, "nan2")
    t.log_step(t.train_step())

    records = [json.loads(l) for l in
               (tmp_path / "nan2" / "metrics.jsonl").read_text().splitlines()
               if l.strip()]
    assert records, "a skipped step on a metrics step must still be recorded"
    assert detect_divergence(records) is not None


def test_train_step_still_accepts_a_bare_tensor_batch(tmp_path):
    """Backward compatibility for every existing pretraining source."""
    class TensorSource:
        def __init__(self, x):
            self.x = x

        def get_batch(self, seq_len):
            return self.x

    torch.manual_seed(0)
    x = torch.randint(1, PRESETS["tiny"].vocab_size, (2, 16))
    args = _tiny_args(tmp_path / "pre", max_steps=1, seq_len=16, micro_batch=2)
    t = Trainer(args)
    t.batch_source = TensorSource(x)
    stats = t.train_step()
    assert not stats["skipped"] and math.isfinite(stats["loss"])


# ------------------------------------------------------- Hub durability ---
# AGENT.md SS0.2: "Never store state only on this box... If it isn't pushed, it
# doesn't exist." `hero` is ~95 GPU-hours; before this existed, losing the
# instance at hour 90 lost four days with nothing recoverable.


@pytest.fixture(autouse=True)
def _no_hub_token(monkeypatch):
    """This box exports HF_TOKEN_WRITE in production shells. Without this the
    suite's `hub_repo` tests would make real network calls on the operator's
    account -- offline by construction, not by luck."""
    monkeypatch.delenv("HF_TOKEN_WRITE", raising=False)


@pytest.fixture
def fake_hub_state():
    """Write the uploader's state file as if uploads had landed."""
    def write(outbox, uploads):
        with open(os.path.join(outbox, cu.STATE_FILENAME), "w") as f:
            json.dump({"uploads": uploads}, f)
    return write


def _hub_args(tmp_path, **overrides):
    overrides.setdefault("hub_repo", "me/daedalus-ckpt")
    # Never spawn the real uploader subprocess in the suite -- staging is what
    # the trainer is responsible for; transport is tested in test_ckpt_uploader.
    overrides.setdefault("hub_uploader", False)
    return _tiny_args(tmp_path, **overrides)


def _staged(outbox):
    return {r["path_in_repo"]: r for _, r in cu.pending_uploads(outbox)}


def test_milestone_step_is_exactly_the_wsd_decay_start():
    """The branch point's whole value is being the end of the stable phase. If
    this drifts from the schedule, the artifact is not what the model card
    says it is."""
    for total in (100, 1000, 87_413):
        for frac in (0.45, 0.2):
            start = decay_start_step(total, frac)
            assert wsd_lr(start - 1, total, warmup=0, decay_frac=frac) == 1.0
            assert wsd_lr(start + 1, total, warmup=0, decay_frac=frac) < 1.0


def test_trainer_milestone_step_tracks_decay_frac(tmp_path):
    args = _hub_args(tmp_path / "run", max_steps=1000, decay_frac=0.45)
    assert Trainer(args).milestone_step == 550


def test_milestone_written_at_decay_start_with_optimizer_state(tmp_path):
    """Full optimizer state, not just weights: a branch that has to rebuild
    Muon's momentum buffers and AdamW's moments loses ground on restart."""
    run = tmp_path / "run"
    # warmup_steps=1 so 10 steps actually reach the stable phase; hero's
    # milestone lands at ~48k steps, three orders of magnitude past its 300.
    args = _hub_args(run, max_steps=10, decay_frac=0.45, warmup_steps=1)
    trainer = Trainer(args)
    trainer.fit()

    record = json.loads((run / "milestone.json").read_text())
    assert record["step"] == 5 and record["revision"].endswith("-stable-end-step5")
    # lr is still at the peak at the branch point -- that is what makes it
    # cheaper to continue from than an annealed checkpoint.
    assert record["lr_mult_at_branch"] == 1.0

    staged = _staged(cu.outbox_dir(str(run)))
    milestone = staged["milestone/test/checkpoint.pt"]
    assert milestone["revision"] == record["revision"]
    assert milestone["kind"] == "milestone"

    payload = torch.load(os.path.join(cu.outbox_dir(str(run)), milestone["payload"]),
                         map_location="cpu", weights_only=False)
    assert "muon" in payload and "adamw" in payload
    assert payload["step"] == 5
    assert all(v.dtype == torch.float32 for v in payload["model"].values()
               if v.is_floating_point())


def test_milestone_revision_is_not_the_rolling_slot(tmp_path):
    """The rolling copy must not be able to overwrite the branch point."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=10, decay_frac=0.45))
    trainer.fit()
    staged = _staged(cu.outbox_dir(str(run)))
    assert staged["milestone/test/checkpoint.pt"]["revision"] != \
        staged["rolling/test/weights.pt"]["revision"]


def test_milestone_is_written_once_across_a_restart(tmp_path):
    """hero.py restarts train.py after a crash. A second milestone at a step
    that is no longer the branch point would quietly replace the real one."""
    run = tmp_path / "run"
    Trainer(_hub_args(run, max_steps=10, decay_frac=0.45)).fit()
    first = json.loads((run / "milestone.json").read_text())

    resumed = Trainer(_hub_args(run, max_steps=14, decay_frac=0.45,
                                resume=str(run / "checkpoint.pt")))
    assert resumed._milestone_done
    resumed.fit()
    assert json.loads((run / "milestone.json").read_text()) == first


def test_milestone_still_written_locally_without_a_hub_repo(tmp_path):
    """The artifact is the point; the upload is transport."""
    run = tmp_path / "run"
    Trainer(_tiny_args(run, max_steps=10, decay_frac=0.45)).fit()
    assert (run / "milestone.json").exists()
    assert (run / "milestone-checkpoint.pt").exists()
    assert not os.path.isdir(cu.outbox_dir(str(run)))  # nothing staged


def test_rolling_hub_copy_is_weights_only_and_bf16(tmp_path):
    """AGENT.md SS0.4: weights-only (~300MB) at intervals, optimizer state only
    at milestones. fp32 would be 642 MB per copy, ~48 copies over hero."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2))
    trainer.fit()

    rolling = _staged(cu.outbox_dir(str(run)))["rolling/test/weights.pt"]
    assert rolling["revision"] == "rolling" and rolling["kind"] == "rolling"
    payload = torch.load(os.path.join(cu.outbox_dir(str(run)), rolling["payload"]),
                         map_location="cpu", weights_only=False)
    assert "muon" not in payload and "adamw" not in payload
    assert all(v.dtype == torch.bfloat16 for v in payload["model"].values()
               if v.is_floating_point())


def test_rolling_copy_is_staged_before_the_first_interval_elapses(tmp_path):
    """A run that dies in its first two hours must still leave something on the
    Hub -- waiting for a full interval is a gap for no reason."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=1, hub_every_sec=1e9))
    trainer.fit()
    assert "rolling/test/weights.pt" in _staged(cu.outbox_dir(str(run)))


def test_a_bf16_weights_only_checkpoint_still_resumes(tmp_path):
    """The disaster-recovery contract: the bf16 rolling copy is what a run
    restarts from after the box is lost, so it must actually load."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=4))
    trainer.fit()
    outbox = cu.outbox_dir(str(run))
    rolling = _staged(outbox)["rolling/test/weights.pt"]
    ckpt = os.path.join(outbox, rolling["payload"])

    resumed = Trainer(_hub_args(tmp_path / "run2", max_steps=6, resume=ckpt))
    assert resumed.step == rolling["step"]
    assert resumed.tokens_seen == rolling["tokens_seen"]
    # Weights survive the fp32 -> bf16 -> fp32 round trip to bf16's precision.
    for name, p in resumed.model.named_parameters():
        ref = dict(trainer.model.named_parameters())[name]
        assert torch.allclose(p, ref, atol=0, rtol=2 ** -7), name
    resumed.fit()  # and training continues rather than raising


def test_no_hub_repo_stages_nothing_and_spawns_nothing(tmp_path):
    """Default-off keeps every existing test and smoke run offline."""
    run = tmp_path / "run"
    trainer = Trainer(_tiny_args(run, max_steps=2))
    trainer.fit()
    assert trainer.maybe_hub_upload(force=True) is None
    assert trainer.uploader_proc is None
    assert not os.path.isdir(cu.outbox_dir(str(run)))


def test_restore_from_hub_end_to_end(tmp_path, monkeypatch):
    """Stage -> upload -> download into a clean directory -> resume, with the
    trainer resolving a `hub://` URI itself. An untested restore path is not a
    backup; this is the offline half, `runs/preflight/hub-restore.md` is the
    live one against the real Hub."""
    hub = tmp_path / "fake-hub"

    class CopyingApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def create_branch(self, *a, **k):
            pass

        def upload_file(self, path_or_fileobj=None, path_in_repo=None,
                        revision=None, **k):
            dest = hub / revision / path_in_repo
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(path_or_fileobj, (bytes, bytearray)):
                dest.write_bytes(path_or_fileobj)
            else:
                dest.write_bytes(open(path_or_fileobj, "rb").read())

    def fake_download(repo_id=None, filename=None, revision=None,
                      local_dir=None, **k):
        os.makedirs(local_dir, exist_ok=True)
        out = os.path.join(local_dir, os.path.basename(filename))
        with open(out, "wb") as f:
            f.write((hub / revision / filename).read_bytes())
        return out

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", CopyingApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN_WRITE", "fake-token")  # enables transport
    # The exit drain is bounded by running the pass in a child process, which
    # would not see CopyingApi and would reach the real network. Run it in
    # process here: this test is about the restore contract, and the child
    # boundary itself is covered in tests/test_ckpt_uploader.py.
    monkeypatch.setattr(cu, "upload_once_bounded",
                        lambda outbox, repo, token=None, **k:
                        cu.upload_once(outbox, repo, token=token))

    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=4))
    trainer.fit()
    # fit()'s exit drain is what actually uploads: training is over, so a
    # synchronous transfer costs nothing and the last checkpoint is the one
    # most worth having off the box.
    assert cu.pending_uploads(cu.outbox_dir(str(run))) == []
    assert (hub / "rolling" / "rolling" / "test" / "weights.pt").exists()

    # Nothing of the original run is reachable from here -- a clean directory,
    # a fresh Trainer, and a URI.
    uri = "hub://me/daedalus-ckpt/rolling/test/weights.pt?rev=rolling"
    restored = Trainer(_hub_args(tmp_path / "restored", max_steps=6, resume=uri))
    assert restored.step == trainer.step
    assert restored.tokens_seen == trainer.tokens_seen
    restored.fit()
    assert restored.step == 6


def test_hub_repo_defaults_to_the_project_repo(monkeypatch):
    """abl-arch and hero inherit this without passing a flag. The overnight
    chain sources .env once at launch, so a variable added afterwards would
    never reach the jobs it starts -- the default has to live in code."""
    monkeypatch.delenv("DAEDALUS_HF_MODEL_REPO", raising=False)
    args = parse_args(["--run-name", "x"])
    assert args.hub_repo == cu.DEFAULT_MODEL_REPO


def test_hub_repo_env_override(monkeypatch):
    monkeypatch.setenv("DAEDALUS_HF_MODEL_REPO", "someone/else")
    assert parse_args(["--run-name", "x"]).hub_repo == "someone/else"


def test_empty_hub_repo_disables_upload():
    """How sweep.py opts its throwaway lr probes out."""
    assert parse_args(["--run-name", "x", "--hub-repo", ""]).hub_repo is None


def test_hub_health_is_absent_without_a_hub_repo(tmp_path):
    trainer = Trainer(_tiny_args(tmp_path / "run", max_steps=1))
    assert trainer._hub_health() == {}


def test_hub_health_reports_pending_before_anything_uploads(tmp_path):
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2))
    trainer.fit()
    health = trainer._hub_health()
    # Nothing was uploaded (no token in the suite), so the staged copies are
    # still pending and there is no uploaded step yet.
    assert health["hub_pending"] >= 1
    assert health["hub_uploaded_step"] is None


def test_hub_health_tracks_the_last_landed_rolling_step(tmp_path, fake_hub_state):
    """If this stops advancing while `step` climbs, uploads are failing --
    otherwise invisible for four days, since every upload failure is
    deliberately non-fatal."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=4))
    trainer.fit()
    fake_hub_state(trainer.outbox, {
        "rolling:rolling/test/weights.pt": {"step": 3, "kind": "rolling"},
        "hero-x:milestone/test/checkpoint.pt": {"step": 2, "kind": "milestone"},
    })
    health = trainer._hub_health()
    assert health["hub_uploaded_step"] == 3          # milestone is not the lag signal
    assert health["hub_lag_steps"] == trainer.step - 3


def test_hub_health_survives_a_corrupt_state_file(tmp_path):
    """Read mid-write must not take down a training run."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2))
    trainer.fit()
    with open(os.path.join(trainer.outbox, cu.STATE_FILENAME), "w") as f:
        f.write("{not json")
    assert trainer._hub_health()["hub_uploaded_step"] is None


def test_metrics_record_carries_hub_health(tmp_path):
    """It has to reach metrics.jsonl and W&B, not just exist as a method."""
    run = tmp_path / "run"
    Trainer(_hub_args(run, max_steps=2)).fit()
    last = json.loads((run / "metrics.jsonl").read_text().strip().split("\n")[-1])
    assert "hub_pending" in last and "hub_uploaded_step" in last


# A stalled uploader was recorded and never graded, which is why the 2026-08-10
# wedge ran for 20 minutes looking like a normal metrics line.

def test_a_stalled_hub_upload_is_warned_about_not_just_recorded(
        tmp_path, fake_hub_state, capsys):
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2))
    trainer.fit()
    stale_h = HUB_STALE_FACTOR * trainer.args.hub_every_sec / 3600.0 + 2.0
    fake_hub_state(trainer.outbox, {
        "rolling:rolling/test/weights.pt": {
            "step": 1, "kind": "rolling",
            "uploaded_at": time.time() - stale_h * 3600},
    })
    capsys.readouterr()
    health = trainer._hub_health()
    out = capsys.readouterr().out
    assert health["hub_stale_h"] > 0
    assert "no checkpoint has reached the Hub" in out, out
    assert "uninsured" in out


def test_a_healthy_cadence_is_silent(tmp_path, fake_hub_state, capsys):
    """Staleness sawtooths up to one full cadence in normal operation, so the
    threshold must not fire at the top of every ordinary interval."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2))
    trainer.fit()
    just_under = trainer.args.hub_every_sec * 0.99
    fake_hub_state(trainer.outbox, {
        "rolling:rolling/test/weights.pt": {
            "step": 1, "kind": "rolling", "uploaded_at": time.time() - just_under},
    })
    capsys.readouterr()
    trainer._hub_health()
    assert "no checkpoint has reached the Hub" not in capsys.readouterr().out


def test_the_stall_warning_does_not_repeat_on_every_metrics_line(
        tmp_path, fake_hub_state, capsys):
    """Four days of a real outage must stay visible without burying the log."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2))
    trainer.fit()
    stale_h = HUB_STALE_FACTOR * trainer.args.hub_every_sec / 3600.0 + 2.0
    fake_hub_state(trainer.outbox, {
        "rolling:rolling/test/weights.pt": {
            "step": 1, "kind": "rolling",
            "uploaded_at": time.time() - stale_h * 3600},
    })
    capsys.readouterr()
    for _ in range(5):
        trainer._hub_health()
    out = capsys.readouterr().out
    assert out.count("no checkpoint has reached the Hub") == 1, out


def test_a_failed_rolling_stage_does_not_end_the_run(tmp_path, monkeypatch):
    """A backup is insurance against losing the run; it must not become a way
    to end it. A full disk during the extra 321 MB write would otherwise take
    down a four-day job that was training perfectly well."""
    def boom(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(Trainer, "_stage", boom)
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=4))
    trainer.fit()                      # must not raise
    assert trainer.step == 4
    assert trainer.maybe_hub_upload(force=True) is None


def test_a_failed_milestone_does_not_end_the_run_and_is_retried(tmp_path,
                                                                monkeypatch):
    """Losing the branch point costs future flexibility; ending the run costs
    the run."""
    kinds = []

    def boom(*a, **k):
        kinds.append(k.get("kind"))
        raise OSError("No space left on device")

    monkeypatch.setattr(Trainer, "_stage", boom)
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=10, decay_frac=0.45,
                                warmup_steps=1))
    trainer.fit()                      # must not raise
    assert trainer.step == 10
    assert not (run / "milestone.json").exists()   # not falsely marked done
    assert not trainer._milestone_done
    # Retries are gated, not per-step: five more steps past the trigger, but
    # only one attempt. 1.4 GB of thrashing on top of a full disk would make a
    # bad situation worse.
    assert kinds.count("milestone") == 1


def test_milestone_retry_succeeds_on_a_later_attempt(tmp_path, monkeypatch):
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=10, decay_frac=0.45,
                                warmup_steps=1))
    real_stage = trainer._stage
    state = {"fail": True}

    def flaky(*a, **k):
        if state["fail"]:
            state["fail"] = False
            raise OSError("transient")
        return real_stage(*a, **k)

    monkeypatch.setattr(trainer, "_stage", flaky)
    monkeypatch.setattr(trainer, "_milestone_gate", IntervalGate(0.0))
    trainer.fit()
    assert (run / "milestone.json").exists() and trainer._milestone_done


# ------------------------------------------------------- init_from (fine-tune) ---
# `resume` continues an interrupted run; `init_from` starts a new one from a
# base model's weights. post.py conflated them, which made the whole SFT stage
# a silent no-op -- fit() exited before its first step because hero's
# step/tokens_seen came back with the weights. See TrainArgs.init_from.

def _hero_like_checkpoint(tmp_path, step=610_000, tokens_seen=40_000_000_000):
    """A checkpoint deep into a finished pretraining run, as hero's will be."""
    donor = Trainer(_tiny_args(tmp_path / "donor", max_steps=1))
    with torch.no_grad():
        for p in donor.model.parameters():
            p.add_(0.01)        # so "did the weights load" is answerable
    path = save_checkpoint(str(tmp_path / "hero.pt"), donor.model, donor.muon,
                           donor.adamw, step, tokens_seen, donor.cfg)
    return path, donor


def test_init_from_loads_weights_but_not_step_or_tokens(tmp_path):
    path, donor = _hero_like_checkpoint(tmp_path)
    t = Trainer(_tiny_args(tmp_path / "ft", max_steps=5, init_from=path))

    assert t.step == 0 and t.tokens_seen == 0
    for a, b in zip(donor.model.parameters(), t.model.parameters()):
        assert torch.allclose(a, b), "init_from did not load the base weights"


def test_init_from_actually_trains_where_resume_would_do_nothing(tmp_path):
    """The bug this guards: with `resume`, fit() breaks at the top of its first
    iteration because step (610000) >= max_steps, and post.py reports success
    having trained on nothing."""
    path, _ = _hero_like_checkpoint(tmp_path)

    broken = Trainer(_tiny_args(tmp_path / "broken", max_steps=5, resume=path))
    broken.fit()
    assert broken.step == 610_000, "sanity: resume restores hero's step"

    fixed = Trainer(_tiny_args(tmp_path / "fixed", max_steps=5, init_from=path))
    fixed.fit()
    assert fixed.step == 5


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_a_run_outside_the_repo_cannot_commit_to_it(tmp_path, monkeypatch):
    """`maybe_push` stages STATUS.md from the repo root, so a run living
    elsewhere used to publish the repo's working copy under its own name.

    Four `test: step 610000, 40000000000 tokens` commits reached origin/main
    from exactly the scenario below -- the test directly above this one. fit()
    breaks at the top of its first iteration (step 610000 >= max_steps 5), so
    metrics.jsonl is never written and STATUS.md is left as the only staged
    path. Each run of the suite published whatever STATUS.md contained at that
    instant, which during an editing session is a half-written file.

    Uses a real git repo and the real fit() path: the whole defect was that the
    production code reached a repository nobody intended it to touch, which a
    stubbed git call cannot demonstrate.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@example.com"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "STATUS.md").write_text("committed state\n")
    _git(["add", "STATUS.md"], repo)
    _git(["commit", "-qm", "init"], repo)
    head = _git(["rev-parse", "HEAD"], repo)

    # Dirty it, exactly as an agent mid-edit would leave it.
    (repo / "STATUS.md").write_text("half-written edit\n")

    path, _ = _hero_like_checkpoint(tmp_path / "donor")
    outside = tmp_path / "elsewhere"        # deliberately NOT inside `repo`
    monkeypatch.chdir(repo)
    trainer = Trainer(_tiny_args(outside, max_steps=5, resume=path))
    trainer.fit()

    assert _git(["rev-parse", "HEAD"], repo) == head, (
        "a run whose run_dir is outside the repo committed to it")
    assert (repo / "STATUS.md").read_text() == "half-written edit\n", (
        "the working copy of STATUS.md was published mid-edit")


def test_a_run_inside_the_repo_still_publishes(tmp_path, monkeypatch):
    """The guard must not silence the real thing: `hero` writes to runs/hero,
    which is inside the repo, and its 10-minute push is how the operator sees
    progress from a phone."""
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@example.com"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "STATUS.md").write_text("before\n")
    _git(["add", "STATUS.md"], repo)
    _git(["commit", "-qm", "init"], repo)
    head = _git(["rev-parse", "HEAD"], repo)
    (repo / "STATUS.md").write_text("progress\n")

    monkeypatch.chdir(repo)
    trainer = Trainer(_tiny_args(repo / "runs" / "hero", max_steps=2))
    trainer.fit()

    assert _git(["rev-parse", "HEAD"], repo) != head, (
        "a run inside the repo stopped publishing; the guard is too strict")
    assert "step 2" in _git(["log", "-1", "--format=%s"], repo)


def test_the_milestone_record_reaches_git_without_an_agent_hand_committing_it(
        tmp_path, monkeypatch):
    """The branch point's *record* has to survive losing this box.

    `milestone.json` names the revision, step, tokens seen and lr multiplier of
    the end-of-stable-phase checkpoint. `export.py:_load_milestone` reads it to
    render the model card's branch command -- hard precondition 4's deliverable
    -- and `train.py:976` reads it to decide the milestone is already done.

    Until this, `maybe_push` staged only `metrics.jsonl` and `STATUS.md`, so
    every milestone.json in this repo got there because an agent hand-committed
    it (f8cda13, 18cc9f2). `hero` writes its one milestone ~3 days in, and the
    after-run chain exists precisely because that agent loop may be dead by
    then. If it is, and the box is then lost, a recovery clones a `runs/hero/`
    with no milestone.json: `_milestone_done` reads False and the milestone
    **re-fires at the post-decay step the recovery resumed at**, publishing a
    checkpoint with lr_mult < 1.0 under the name of the branch point.

    Runs the real fit() against a real git repo, because the defect was about
    which paths reach a commit -- a stubbed git call cannot show that.
    """
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@example.com"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "STATUS.md").write_text("before\n")
    _git(["add", "STATUS.md"], repo)
    _git(["commit", "-qm", "init"], repo)

    monkeypatch.chdir(repo)
    # 10 steps with decay_frac 0.45 puts the milestone at step 5, so it fires
    # inside the run rather than being simulated.
    trainer = Trainer(_tiny_args(repo / "runs" / "hero", max_steps=10))
    assert trainer.milestone_step < 10, trainer.milestone_step
    trainer.fit()

    assert (repo / "runs" / "hero" / "milestone.json").exists(), \
        "sanity: the milestone did not fire, so this proves nothing"
    tracked = _git(["ls-files", "runs/hero"], repo).splitlines()
    assert "runs/hero/milestone.json" in tracked, (
        "the branch-point record never reached git; it survives only as long "
        f"as this box does. tracked: {tracked}")


def test_init_from_gets_a_fresh_lr_schedule(tmp_path):
    """Resuming hero's step also pins the WSD multiplier at 0 -- a fine-tune
    that runs but learns nothing."""
    path, _ = _hero_like_checkpoint(tmp_path)

    broken = Trainer(_tiny_args(tmp_path / "b", max_steps=5, resume=path))
    assert broken._lr_multiplier(broken._estimated_total_steps()) == 0.0

    fixed = Trainer(_tiny_args(tmp_path / "f", max_steps=100, init_from=path,
                               warmup_steps=10))
    fixed.step = 20     # past warmup, inside the stable phase
    assert fixed._lr_multiplier(100) == pytest.approx(1.0)


def test_init_from_leaves_optimizer_state_fresh(tmp_path):
    path, donor = _hero_like_checkpoint(tmp_path)
    donor.batch_source = FixedBatchSource(
        [torch.randint(0, donor.cfg.vocab_size, (2, 16)) for _ in range(3)])
    for _ in range(3):
        donor.train_step()      # give the donor's optimizers real momentum
    loaded = save_checkpoint(str(tmp_path / "hero2.pt"), donor.model, donor.muon,
                             donor.adamw, 610_000, 40_000_000_000, donor.cfg)

    t = Trainer(_tiny_args(tmp_path / "ft2", max_steps=1, init_from=loaded))
    assert all(not st for st in t.muon.state.values()), "Muon state carried over"
    assert all(not st for st in t.adamw.state.values()), "AdamW state carried over"


def test_resume_takes_precedence_over_init_from(tmp_path):
    """A crash-restart of a fine-tune must continue the fine-tune, not start it
    over from the base weights -- so a supervisor can relaunch one command line."""
    base, _ = _hero_like_checkpoint(tmp_path)
    ft = Trainer(_tiny_args(tmp_path / "ft", max_steps=4, init_from=base))
    ft.fit()
    mid = save_checkpoint(str(tmp_path / "ft.pt"), ft.model, ft.muon, ft.adamw,
                          ft.step, ft.tokens_seen, ft.cfg)

    restarted = Trainer(_tiny_args(tmp_path / "again", max_steps=8,
                                   init_from=base, resume=mid))
    assert restarted.step == 4


def test_missing_init_from_is_fatal(tmp_path):
    """Unlike `resume`, which is routinely absent on a first launch. Falling
    back to random init would fine-tune a fresh model for hours and succeed."""
    with pytest.raises(FileNotFoundError):
        Trainer(_tiny_args(tmp_path / "x", max_steps=1,
                           init_from=str(tmp_path / "does-not-exist.pt")))


def test_missing_resume_still_starts_from_scratch(tmp_path):
    t = Trainer(_tiny_args(tmp_path / "y", max_steps=1,
                           resume=str(tmp_path / "absent.pt")))
    assert t.step == 0


def test_init_from_is_exposed_on_the_cli(tmp_path):
    args = parse_args(["--init-from", "/tmp/base.pt", "--resume", "/tmp/r.pt"])
    assert args.init_from == "/tmp/base.pt" and args.resume == "/tmp/r.pt"


# ------------------------------------------------- finite sources end cleanly ---

class ExhaustibleBatchSource(FixedBatchSource):
    """post.py's SFT stream is one-shot: it raises StopIteration when the
    dataset runs out."""

    def get_batch(self, seq_len):
        if self.i >= len(self.batches):
            raise StopIteration
        return super().get_batch(seq_len)


def test_exhausted_batch_source_finishes_the_run_instead_of_crashing(tmp_path):
    """Letting StopIteration propagate skips the forced final checkpoint and
    wandb.finish(), so a complete fine-tune loses everything since the last
    rolling checkpoint."""
    run = tmp_path / "run"
    args = _tiny_args(run, max_steps=100)      # more steps than there is data
    t = Trainer(args)
    t.batch_source = ExhaustibleBatchSource(
        [torch.randint(0, t.cfg.vocab_size, (2, 16)) for _ in range(6)])

    t.fit()      # must not raise

    assert t.step == 6
    assert os.path.exists(t.ckpt_path), "final checkpoint was not forced"
    assert not os.path.exists(run / "train.pid")


# ----------------------------------------------------- the final metrics row ---
# `fit` breaks at the *top* of an iteration, so the last interval-written row
# covers a step up to `metrics_every_steps - 1` earlier and reports fewer tokens
# than the target. Three consumers compare that row's `tokens` against the
# target and read a finished run as unfinished:
# `watchdog.detect_completion` (which then falls through to `detect_stall`,
# whose halt marker makes `abl_arch.py` refuse to retry the arm),
# `scripts/eval_arm1_when_done.sh`, and the writeup. Measured on the live
# `abl-arch` arm 1: 10,391 steps is not a multiple of 20, so its last interval
# row is step 10,380 / 4,994,316,288 tokens against a 5,000,000,000 target.
# The three sweep probes ended at step 1040 and hid this completely.

def _tokens_run(run_dir, total_tokens, metrics_every_steps):
    """A token-budgeted run (max_steps=None) at 32 tokens/step."""
    args = _tiny_args(run_dir, max_steps=None, total_tokens=total_tokens,
                      metrics_every_steps=metrics_every_steps)
    t = Trainer(args)
    t.fit()
    records = [json.loads(l) for l in
               (run_dir / "metrics.jsonl").read_text().strip().splitlines()]
    return t, records


def test_final_metrics_row_reaches_the_token_target_off_interval(tmp_path):
    """32 tokens/step, target 200 -> the loop breaks after step 7, which is not
    a multiple of 5. Without the forced row the record stops at step 5 / 160
    tokens and every completion check reads the run as unfinished."""
    t, records = _tokens_run(tmp_path / "run", total_tokens=200,
                             metrics_every_steps=5)

    assert t.step == 7
    assert t.tokens_seen == 224
    last = records[-1]
    assert last["step"] == 7, [r["step"] for r in records]
    assert last["tokens"] >= 200, "durable record never reaches the target"
    # The interval alone would have stopped here -- this is what the bug was.
    interval_rows = [r for r in records if r["step"] % 5 == 0]
    assert interval_rows[-1]["tokens"] < 200


def test_final_metrics_row_is_not_duplicated_when_the_run_ends_on_interval(tmp_path):
    """Target 300 -> breaks after step 10, which *is* a multiple of 5. The
    interval already wrote that row; a second one double-counts in every
    consumer that replays metrics.jsonl."""
    t, records = _tokens_run(tmp_path / "run", total_tokens=300,
                             metrics_every_steps=5)

    assert t.step == 10
    steps = [r["step"] for r in records]
    assert steps == [5, 10], steps
    assert records[-1]["tokens"] >= 300


def test_watchdog_sees_a_finished_off_interval_run_as_complete(tmp_path):
    """The integration that actually matters: `detect_completion` must fire on
    the finished run, because when it does not the watchdog falls through to
    `detect_stall`, and a stall halt marker makes `abl_arch.py` abandon the
    chain rather than retry the arm."""
    import watchdog

    run = tmp_path / "run"
    _tokens_run(run, total_tokens=200, metrics_every_steps=5)
    records = watchdog.read_metrics(str(run))

    assert watchdog.detect_completion(records, 200) is not None
    # ...and the fallthrough it prevents: with only the interval rows, the same
    # metrics look unfinished, which is precisely the stall path.
    interval_only = [r for r in records if r["step"] % 5 == 0]
    assert watchdog.detect_completion(interval_only, 200) is None


def test_resume_from_an_already_complete_checkpoint_exits_cleanly(tmp_path):
    """The recovery path `scripts/guard_exit_drain.py` depends on, and that
    `abl_arch.py` takes on *any* retry of a finished arm.

    When the guard halts a trainer wedged in its exit drain, `abl_arch.py` sees
    a non-zero exit and relaunches with `--resume <ckpt>`. That attempt loads a
    checkpoint whose `tokens_seen` already meets the target, so `fit` must break
    at the top of its first iteration and exit 0 in seconds -- without training
    a step, and without tripping over `_last_stats` being None because no step
    ever ran. If this path raised, the guard would convert a recoverable stall
    into the chain abort it exists to prevent, which is strictly worse than
    doing nothing.

    Resumes into the run's **own** directory, which is the shape both recovery
    paths actually use -- `supervise.py:311` and `abl_arch.py:110` both build
    `--resume <run_dir>/checkpoint.pt`. An earlier version of this test pointed
    a *fresh* run directory at the finished checkpoint, which is not a relaunch
    at all but the branch-with-the-wrong-budget mistake `NoOpResume` now
    refuses (see below)."""
    run = tmp_path / "run"
    t, records = _tokens_run(run, total_tokens=200, metrics_every_steps=5)
    assert t.tokens_seen >= 200
    rows_before = len(records)

    resumed = Trainer(_tiny_args(run, max_steps=None,
                                 total_tokens=200, metrics_every_steps=5,
                                 resume=str(run / "checkpoint.pt")))
    assert resumed.tokens_seen >= 200, "resume did not restore the finished state"

    resumed.fit()      # must not raise, and must not train

    assert resumed.step == t.step, "a completed run trained further on resume"
    # No step ran, so no *new* row is forced -- the file still holds exactly
    # what the original run wrote. Forcing one from a None `_last_stats` is
    # the crash this guards against.
    rows_after = len((run / "metrics.jsonl").read_text().strip().splitlines())
    assert rows_after == rows_before, "a no-op resume appended a metrics row"
    assert os.path.exists(resumed.ckpt_path)
    assert rows_before > 0


def test_a_relaunch_that_left_no_metrics_is_still_treated_as_a_relaunch(tmp_path):
    """The metrics-history discriminator is not the only one, because a run can
    die before its first metrics write and still be relaunched from its own
    checkpoint. `--resume <this run's own dir>/checkpoint.pt` is a relaunch
    however little history it left, so it must exit cleanly rather than raise."""
    run = tmp_path / "run"
    t, _ = _tokens_run(run, total_tokens=200, metrics_every_steps=5)
    (run / "metrics.jsonl").unlink()

    resumed = Trainer(_tiny_args(run, max_steps=None, total_tokens=200,
                                 resume=str(run / "checkpoint.pt")))
    resumed.fit()
    assert resumed.step == t.step


def test_a_resume_that_would_train_nothing_refuses_instead_of_exiting_zero(tmp_path):
    """The defect this project has now shipped twice: a `--resume` whose
    checkpoint is already past `--total-tokens` breaks at the top of fit()'s
    first iteration, so the process prints `resumed from ...`, writes no
    metrics row, and **exits 0**.

    First occurrence: `post.py` passed hero's checkpoint as `resume` and its
    whole SFT stage silently did nothing. Second: the model card's
    branch-from-milestone command omitted `--total-tokens` and inherited the
    5e9 default against a 30.5e9-token milestone, so the one command hard
    precondition 4 exists to publish trained nothing at all. Measured rather
    than inferred -- `runs/preflight/branch-command.md`.

    A fresh run directory pointed at a foreign checkpoint is nobody's recovery
    path, so it is a mistake, and is refused naming the budget it needs."""
    run = tmp_path / "run"
    t, _ = _tokens_run(run, total_tokens=200, metrics_every_steps=5)

    with pytest.raises(train_module.NoOpResume) as excinfo:
        Trainer(_tiny_args(tmp_path / "branch", max_steps=None,
                           total_tokens=200,
                           resume=str(run / "checkpoint.pt")))
    message = str(excinfo.value)
    assert str(t.tokens_seen) in message.replace(",", ""), message
    assert "--total-tokens" in message and "--init-from" in message


def test_the_noop_guard_stays_out_of_the_way_of_a_real_branch(tmp_path):
    """The whole point of the milestone is branching onto a *larger* budget, so
    a budget above the checkpoint's tokens_seen must train rather than refuse.
    The guard is only allowed to fire where the loop would have done nothing."""
    run = tmp_path / "run"
    _tokens_run(run, total_tokens=200, metrics_every_steps=5)

    branched = Trainer(_tiny_args(tmp_path / "branch", max_steps=None,
                                  total_tokens=800, metrics_every_steps=1,
                                  resume=str(run / "checkpoint.pt")))
    branched.fit()
    assert branched.tokens_seen >= 800, "the branch did not train on"


def test_the_noop_guard_defers_to_an_explicit_max_steps(tmp_path):
    """`--max-steps` overrides the token budget, so `total_tokens` says nothing
    about whether the loop will run. Firing here would break every smoke and
    test path that resumes under a step cap."""
    run = tmp_path / "run"
    _tokens_run(run, total_tokens=200, metrics_every_steps=5)

    t = Trainer(_tiny_args(tmp_path / "capped", max_steps=100_000,
                           total_tokens=200,
                           resume=str(run / "checkpoint.pt")))
    assert t.tokens_seen >= 200      # constructed without raising


def test_unbounded_repetition_warns_where_the_skew_number_reads_perfect(tmp_path, capsys):
    """The mixture's *second* failure mode, which `l1_skew_pts` cannot see.

    When no allocation can satisfy the epoch cap -- every source over the limit
    -- `cap_weights_by_epochs` returns the target shares unchanged, so the skew
    is **0.00 by construction**: its best possible value at the one budget where
    repetition is bounded by nothing. That is hero at 60B against the 14.218B
    corpus, where `everyday-conversations` takes its full 2% from a dataset
    exhausted at 403,573 tokens. Before this, the graded signal was silent and
    the ungraded `capped:` line printed the same shape it prints when the cap is
    working normally.
    """
    tiny = tmp_path / "too-small"
    _write_source(tiny, "fineweb-edu", n_tokens=100, shard_tokens=100)
    _write_source(tiny, "finewiki-en", n_tokens=100, shard_tokens=100)

    src = Trainer(_tiny_args(tmp_path / "run", data_dir=str(tiny),
                             total_tokens=100_000, max_steps=1)).batch_source
    out = capsys.readouterr().out
    summary = src.mixture_summary()

    # The trap, asserted rather than described: the graded skew signal is at
    # its best possible value exactly here.
    assert summary["l1_skew_pts"] == pytest.approx(0.0)
    assert summary["max_epochs_seen"] > summary["max_epochs"]
    assert "repetition is UNBOUNDED" in out
    assert "is not measuring this" in out


def test_bounded_repetition_stays_quiet(tmp_path, capsys):
    """Without this the warning could fire whenever the cap merely binds, and a
    warning that fires on the normal case is one nobody reads. A properly capped
    source sits at exactly `max_epochs`, never above it."""
    corpus = tmp_path / "ok"
    _write_source(corpus, "fineweb-edu", n_tokens=8000, shard_tokens=1000)
    _write_source(corpus, "finewiki-en", n_tokens=8000, shard_tokens=1000)

    src = Trainer(_tiny_args(tmp_path / "run", data_dir=str(corpus),
                             total_tokens=2000, max_steps=1)).batch_source
    out = capsys.readouterr().out
    summary = src.mixture_summary()

    assert summary["max_epochs_seen"] <= summary["max_epochs"] + 1e-6
    assert "repetition is UNBOUNDED" not in out


def test_the_repetition_alarm_brackets_heros_real_budgets():
    """The constants it was chosen from, from the real manifests: at 40B the
    worst source sits at exactly the 4.00-epoch cap and must stay quiet; at 60B
    `everyday-conversations` reaches ~2,973 epochs and must fire. 60B is the
    budget the operator selected, so this is the gap the $1.89 corpus top-up
    (issue #5) closes -- and until it does, the alarm is what stands between a
    $61.89 run and a wrecked mixture."""
    max_epochs = 4.0
    for quiet in (1.33, 4.00):                 # the 40B range, per-source
        assert not quiet > max_epochs + 1e-6
    assert 2973.0 > max_epochs + 1e-6          # 60B, everyday-conversations


def test_the_forced_final_row_does_not_reprint_the_step_line(tmp_path, capsys):
    """A run ending exactly on the interval has already printed and recorded its
    final step; the forced call must suppress *both*, not just the row. A second
    identical `step NNNN loss ...` line reads as though the last step ran twice,
    which is a confusing thing to leave at the end of a 138-hour log."""
    run = tmp_path / "run"
    args = _tiny_args(run, max_steps=None, total_tokens=300,
                      metrics_every_steps=5, log_every_steps=5)
    Trainer(args).fit()

    out = capsys.readouterr().out
    step_lines = [l for l in out.splitlines() if l.startswith("step ")]
    assert len(step_lines) == len(set(step_lines)), step_lines
    finals = [l for l in step_lines if l.split()[1] == "10"]
    assert len(finals) == 1, step_lines


# ------------------------------ non-finite weights must not become durable ---

def _nan_one_parameter(trainer):
    with torch.no_grad():
        next(iter(trainer.model.parameters())).fill_(float("nan"))


def test_weights_are_finite_detects_a_single_nan_parameter(tmp_path):
    trainer = Trainer(_tiny_args(tmp_path / "run", max_steps=1))
    assert trainer.weights_are_finite()
    _nan_one_parameter(trainer)
    assert not trainer.weights_are_finite()


def test_a_diverged_model_does_not_overwrite_the_last_good_checkpoint(tmp_path):
    """The failure this guards: `train_step` skips the optimizer on a
    non-finite loss *without* incrementing `self.step`, so once the weights are
    NaN every step is skipped and metrics.jsonl goes quiet -- and watchdog.py
    needs up to --stall-min (30 min) to notice. The checkpoint gate is also 30
    min and runs every iteration, so the two clocks race. If the checkpoint one
    wins, the only artifact `hero` can resume from is NaN.
    """
    trainer = Trainer(_tiny_args(tmp_path / "run", max_steps=1))
    trainer.maybe_checkpoint(force=True)
    good = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    assert os.path.exists(trainer.ckpt_path)

    _nan_one_parameter(trainer)
    trainer.step += 1
    trainer.maybe_checkpoint(force=True)

    after = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    assert after["step"] == good["step"], "the NaN model overwrote the checkpoint"
    assert all(torch.isfinite(v).all() for v in after["model"].values()
               if v.is_floating_point())


def test_a_diverged_model_is_not_staged_for_the_hub(tmp_path):
    """A poisoned `rolling` revision is worse than a stale one: it is what a
    lost instance would be restored from, and the local copy is gone with the
    box."""
    args = _tiny_args(tmp_path / "run", max_steps=1)
    args.hub_repo = "someone/daedalus-checkpoints"
    trainer = Trainer(args)
    _nan_one_parameter(trainer)
    assert trainer.maybe_hub_upload(force=True) is None
    outbox = os.path.join(str(tmp_path / "run"), "hub_outbox")
    assert not [f for f in os.listdir(outbox)
                if f.endswith(".pt")] if os.path.isdir(outbox) else True


def test_a_diverged_milestone_is_retried_rather_than_marked_done(tmp_path):
    """The branch point is one-shot and is the artifact that makes a $63.78 run
    extendable later. Refusing to write a NaN one must not consume it -- a
    resume from the last good checkpoint can reach the same step with finite
    weights and still produce it."""
    args = _tiny_args(tmp_path / "run", max_steps=1)
    trainer = Trainer(args)
    trainer.milestone_step = 0
    trainer._milestone_done = False
    _nan_one_parameter(trainer)

    assert trainer.maybe_milestone() is None
    assert not trainer._milestone_done, "a refused milestone was consumed"
    assert not os.path.exists(trainer.milestone_path)


def test_the_refusal_is_logged_once_not_once_per_gate(tmp_path, capsys):
    """A diverged run keeps looping until the watchdog halts it. One ERROR line
    per 30-minute gate is fine; one per iteration would bury the non-finite
    loss row that says what actually happened."""
    trainer = Trainer(_tiny_args(tmp_path / "run", max_steps=1))
    _nan_one_parameter(trainer)
    for _ in range(3):
        trainer.maybe_checkpoint(force=True)
    out = capsys.readouterr().out
    assert out.count("refusing to overwrite") == 1


def test_a_healthy_run_still_checkpoints_normally(tmp_path):
    """The guard must be invisible when nothing is wrong -- the regression that
    would matter most is a finiteness check that somehow refuses a good model
    and silently stops checkpointing a four-day run."""
    trainer = Trainer(_tiny_args(tmp_path / "run", max_steps=3))
    trainer.fit()
    assert os.path.exists(trainer.ckpt_path)
    ckpt = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 3


# ------------------------------------------------ the uploader self-heal ---
# `_start_uploader` ran once at startup and the handle was never consulted
# again, so an uploader that died mid-run took the run's insurance with it
# silently for the rest of a 5.9-day job.

def test_a_dead_uploader_is_respawned(tmp_path, monkeypatch):
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2, hub_uploader=True))
    monkeypatch.setenv("HF_TOKEN_WRITE", "x")
    monkeypatch.setattr(cu, "uploader_is_live", lambda outbox: False)
    spawned = []
    monkeypatch.setattr(trainer, "_start_uploader", lambda: spawned.append(1))

    class Dead:
        def poll(self):
            return 1
    trainer.uploader_proc = Dead()
    trainer._ensure_uploader()
    assert spawned == [1]


def test_a_live_child_is_left_alone(tmp_path, monkeypatch):
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2, hub_uploader=True))
    monkeypatch.setenv("HF_TOKEN_WRITE", "x")
    spawned = []
    monkeypatch.setattr(trainer, "_start_uploader", lambda: spawned.append(1))

    class Alive:
        def poll(self):
            return None
    trainer.uploader_proc = Alive()
    trainer._ensure_uploader()
    assert spawned == []


def test_an_orphan_from_a_previous_attempt_is_not_duplicated(tmp_path, monkeypatch):
    """The case that makes `self.uploader_proc.poll()` alone the wrong test.

    A SIGKILLed trainer cannot reap its uploader, so the orphan keeps the lock
    and keeps delivering; *this* attempt's uploader exited on that lock
    immediately and by design. The child handle therefore reads "dead" for the
    rest of the run while uploads are entirely healthy -- respawning on it
    would fork a redundant uploader every two hours for six days.
    """
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2, hub_uploader=True))
    monkeypatch.setenv("HF_TOKEN_WRITE", "x")
    monkeypatch.setattr(cu, "uploader_is_live", lambda outbox: True)
    spawned = []
    monkeypatch.setattr(trainer, "_start_uploader", lambda: spawned.append(1))

    class Dead:
        def poll(self):
            return 1
    trainer.uploader_proc = Dead()
    trainer._ensure_uploader()
    assert spawned == []


def test_the_self_heal_is_off_without_a_hub_repo_or_token(tmp_path, monkeypatch):
    run = tmp_path / "run"
    trainer = Trainer(_tiny_args(run, max_steps=2))
    spawned = []
    monkeypatch.setattr(trainer, "_start_uploader", lambda: spawned.append(1))
    trainer._ensure_uploader()
    assert spawned == []

    trainer2 = Trainer(_hub_args(tmp_path / "run2", max_steps=2, hub_uploader=True))
    monkeypatch.delenv("HF_TOKEN_WRITE", raising=False)
    monkeypatch.setattr(trainer2, "_start_uploader", lambda: spawned.append(1))
    trainer2._ensure_uploader()
    assert spawned == []


def test_the_self_heal_never_raises(tmp_path, monkeypatch, capsys):
    """Insurance must not become a way to end the run."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2, hub_uploader=True))
    monkeypatch.setenv("HF_TOKEN_WRITE", "x")

    def boom(outbox):
        raise OSError("proc is gone")
    monkeypatch.setattr(cu, "uploader_is_live", boom)
    trainer.uploader_proc = None
    trainer._ensure_uploader()
    assert "could not check uploader liveness" in capsys.readouterr().out


def test_the_rolling_upload_checks_the_uploader_before_staging(tmp_path, monkeypatch):
    """Wired to the same 2 h gate as the upload it protects, so detection
    latency matches the cadence rather than needing a clock of its own."""
    run = tmp_path / "run"
    trainer = Trainer(_hub_args(run, max_steps=2))
    trainer.fit()
    calls = []
    monkeypatch.setattr(trainer, "_ensure_uploader", lambda: calls.append(1))
    trainer.maybe_hub_upload(force=True)
    assert calls == [1]


# ------------------------------------------------- git: bounded, not just ---
# -------------------------------------------------- exception-safe        ---
#
# `git_commit_and_push` runs synchronously inside the training loop every ~10
# minutes of a multi-day run. Its docstring promised "never raises", which is
# the wrong guarantee on its own: a hang is not an exception, so the `except`
# cannot see one, and a black-holed push would freeze the loop with the GPU
# idle for the rest of the run.

def _blackhole_server():
    """A TCP listener that accepts connections and then says nothing, ever.

    The point of the tests below is that a *real* `git push` against a real
    stalled endpoint returns. A monkeypatched `subprocess.run` would only prove
    that the mock raises what the mock was told to raise.
    """
    import socket
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held = []

    def accept_forever():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            held.append(conn)  # keep it open; never reply

    threading.Thread(target=accept_forever, daemon=True).start()
    return srv, srv.getsockname()[1], held


def test_a_black_holed_push_returns_instead_of_freezing_the_training_loop(
        tmp_path, monkeypatch):
    """The whole point: measured against a real stalled TCP endpoint.

    Without a timeout this call never returns, the training loop never takes
    another step, and nothing reports it -- the commit that would have said so
    is the one that is stuck. With one, it costs a single publish interval and
    a warning line.
    """
    import time

    srv, port, _held = _blackhole_server()
    try:
        repo = tmp_path / "wedged"
        repo.mkdir()
        _init_repo(repo, with_remote=f"http://127.0.0.1:{port}/daedalus.git")
        (repo / "f.txt").write_text("hello")
        monkeypatch.setattr(train_module, "GIT_PUSH_TIMEOUT_S", 3.0)

        t0 = time.monotonic()
        ok = git_commit_and_push(str(repo), "wedged push", ["f.txt"])
        elapsed = time.monotonic() - t0
    finally:
        srv.close()

    assert ok is False, "a push that never completes is not a successful push"
    assert elapsed < 30.0, (
        f"took {elapsed:.1f}s against a stalled endpoint -- the bound did not "
        f"bite, so the training loop would have hung here")

    # The commit itself is local and must survive, so the next interval pushes
    # it rather than losing the metrics row this call was publishing.
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True, check=True)
    assert "wedged push" in log.stdout


def test_every_git_subprocess_call_is_time_bounded():
    """Structural audit, because the defect is a class and its symptom is
    silence. A call site added later is unbounded by default."""
    import ast

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "train.py")
    fn = next(n for n in ast.walk(ast.parse(open(src).read()))
              if isinstance(n, ast.FunctionDef) and n.name == "git_commit_and_push")
    unbounded = [n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
                 and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "subprocess"
                 and not any(kw.arg == "timeout" for kw in n.keywords)]

    assert not unbounded, (
        f"train.py:{unbounded} run git with no timeout, inside the training "
        f"loop. A stall there is invisible and costs the rest of the run.")


def test_the_local_bound_is_far_wider_than_the_push_bound():
    """Asymmetric on purpose, and worth pinning so it is not "tidied" later.

    Only the push can stall on the network, so only the push needs a bound
    that bites. Killing a local `git commit` is the *risky* direction: SIGKILL
    can leave `.git/index.lock`, which breaks every later git operation in
    this repo -- the heartbeat and the auto-commit supervisor included.
    """
    assert train_module.GIT_LOCAL_TIMEOUT_S >= 60.0
    assert train_module.GIT_PUSH_TIMEOUT_S >= 120.0, \
        "must not be tight enough to fail a healthy push on a slow link"
    # Measured on this box: push round-trips in 0.70-0.82 s, local ops ~3 ms.
    assert train_module.GIT_LOCAL_TIMEOUT_S / 0.003 > \
           train_module.GIT_PUSH_TIMEOUT_S / 0.82


def test_the_push_carries_its_stall_guard_and_cannot_wait_on_a_prompt(
        tmp_path, monkeypatch):
    """Two hang surfaces the hard timeout alone would only paper over.

    The low-speed abort ends a stalled *transfer* as a clean git error the
    `except` already handles, rather than as a SIGKILL. `GIT_TERMINAL_PROMPT=0`
    turns a credential prompt -- this box authenticates pushes through the
    `gh auth git-credential` helper -- into an immediate failure instead of a
    wait on a stdin nobody is typing into.
    """
    calls = []
    real_run = subprocess.run

    def spy(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return real_run(cmd, **kwargs)

    remote = tmp_path / "remote-spy.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo = tmp_path / "spy"
    repo.mkdir()
    _init_repo(repo, with_remote=remote)
    (repo / "f.txt").write_text("hello")

    monkeypatch.setattr(train_module.subprocess, "run", spy)
    assert git_commit_and_push(str(repo), "spy commit", ["f.txt"]) is True

    push = next(c for c in calls if "push" in c["cmd"])
    assert "http.lowSpeedLimit=1000" in push["cmd"]
    assert "http.lowSpeedTime=60" in push["cmd"]
    # -c options are only honoured before the subcommand.
    assert push["cmd"].index("http.lowSpeedTime=60") < push["cmd"].index("push")
    assert push["kwargs"]["env"]["GIT_TERMINAL_PROMPT"] == "0"
    # The push inherits the rest of the environment -- dropping it would strip
    # PATH and the credential helper's own configuration.
    assert push["kwargs"]["env"].get("PATH") == os.environ.get("PATH")


# --------------------------------------------------------------------------
# Phase 3 recovery knobs: schedule and memory flags on the CLI, and the
# durable record a finiteness gate is decided from.
# --------------------------------------------------------------------------

def test_schedule_flags_default_to_the_shipped_values():
    """Absent flags must not change any run that is already scheduled."""
    a = train_module.parse_args(["--run-name", "x"])
    assert a.warmup_steps == 300
    assert a.decay_frac == 0.45
    assert a.loss_chunk_size is None
    assert a.gradient_checkpointing is None


def test_schedule_flags_reach_train_args_from_the_cli():
    a = train_module.parse_args([
        "--run-name", "x", "--warmup-steps", "40", "--decay-frac", "0.9",
        "--loss-chunk-size", "256", "--gradient-checkpointing",
    ])
    assert a.warmup_steps == 40
    assert a.decay_frac == 0.9
    assert a.loss_chunk_size == 256
    assert a.gradient_checkpointing is True


def test_gradient_checkpointing_can_be_forced_off_from_the_cli():
    """`--no-...` must be distinguishable from "unset", or a config that turns
    checkpointing on could never be overridden from the command line."""
    a = train_module.parse_args(["--run-name", "x", "--no-gradient-checkpointing"])
    assert a.gradient_checkpointing is False


def test_warmup_and_decay_flags_actually_move_the_schedule(tmp_path):
    """The flags exist to reshape a short run's LR curve, so assert on the
    curve rather than on the field they were stored in."""
    args = _tiny_args(tmp_path / "run", max_steps=100, warmup_steps=10,
                      decay_frac=0.5)
    t = Trainer(args)
    # Warmup is 10 steps, not 300: by step 10 the multiplier is at its ceiling
    # instead of the 3% a 300-step warmup would still be at.
    t.step = 10
    assert t._lr_multiplier(100) == pytest.approx(1.0)
    # Decay starts halfway, and the milestone tracks it.
    t.step = 50
    assert t._lr_multiplier(100) == pytest.approx(1.0)
    t.step = 75
    assert t._lr_multiplier(100) == pytest.approx(0.5)
    assert t.milestone_step == 50


def test_config_overrides_do_not_mutate_the_shared_preset(tmp_path):
    """`PRESETS` holds one instance per name. Mutating it would silently
    reconfigure every later Trainer, eval and export in the process."""
    before = (PRESETS["tiny"].loss_chunk_size,
              PRESETS["tiny"].gradient_checkpointing)
    args = _tiny_args(tmp_path / "run", max_steps=1, loss_chunk_size=64,
                      gradient_checkpointing=True)
    t = Trainer(args)
    assert t.cfg.loss_chunk_size == 64
    assert t.cfg.gradient_checkpointing is True
    assert (PRESETS["tiny"].loss_chunk_size,
            PRESETS["tiny"].gradient_checkpointing) == before
    assert t.cfg is not PRESETS["tiny"]


def test_an_unoverridden_run_still_gets_the_preset_object(tmp_path):
    args = _tiny_args(tmp_path / "run", max_steps=1)
    assert Trainer(args).cfg is PRESETS["tiny"]


def test_a_negative_loss_chunk_size_is_refused_rather_than_silently_odd(tmp_path):
    args = _tiny_args(tmp_path / "run", max_steps=1, loss_chunk_size=-1)
    with pytest.raises(ValueError, match="loss-chunk-size"):
        Trainer(args)


def test_gradient_checkpointing_still_trains_the_same_model(tmp_path):
    """The flag is a memory trade, not a model change: same seed, same data,
    same loss. If recomputation drifted from the stored activations the
    recovery run would be optimizing something other than what it exports."""
    plain = Trainer(_tiny_args(tmp_path / "a", max_steps=2))
    plain.fit()
    ckpt = Trainer(_tiny_args(tmp_path / "b", max_steps=2,
                              gradient_checkpointing=True))
    ckpt.fit()
    a = [json.loads(l)["loss"] for l in
         (tmp_path / "a" / "metrics.jsonl").read_text().strip().splitlines()]
    b = [json.loads(l)["loss"] for l in
         (tmp_path / "b" / "metrics.jsonl").read_text().strip().splitlines()]
    assert a == pytest.approx(b, rel=1e-5)


def test_metrics_carry_the_skipped_update_count_every_row(tmp_path):
    """The finiteness gate is decided from metrics.jsonl, so the count has to
    be in every row -- not only the rows where a skip happened."""
    args = _tiny_args(tmp_path / "run", max_steps=3)
    Trainer(args).fit()
    rows = [json.loads(l) for l in
            (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()]
    assert [r["skipped_updates"] for r in rows] == [0, 0, 0]


def test_a_non_finite_loss_increments_the_recorded_skip_count(tmp_path):
    """Previously a skip left only a WARNING on stdout and a `loss: nan` row
    indistinguishable from an ordinary interval row."""
    args = _tiny_args(tmp_path / "run", max_steps=2)
    t = Trainer(args)
    real_net = t.net

    def poisoned(x, targets=None, **kw):
        return None, torch.tensor(float("nan")), None

    t.net = poisoned
    stats = t.train_step()
    assert stats["skipped"] is True
    assert t._skipped_updates == 1
    # ... and the step did not advance, so the budget is unchanged.
    assert t.step == 0

    t.net = real_net
    t.log_step(stats, force=True)
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text()
                     .strip().splitlines()[-1])
    assert row["skipped_updates"] == 1


def test_the_skip_count_survives_a_crash_resume(tmp_path):
    """A resumed run that reset the count to zero would report a clean gate
    for a run that had already skipped updates."""
    args = _tiny_args(tmp_path / "run", max_steps=1)
    first = Trainer(args)
    first._skipped_updates = 3
    first.fit()

    second = Trainer(_tiny_args(tmp_path / "run", max_steps=2,
                                resume=str(tmp_path / "run" / "checkpoint.pt")))
    assert second._skipped_updates == 3


def test_a_fresh_init_from_run_starts_its_skip_count_at_zero(tmp_path):
    """`--init-from` is a new run. Inheriting the donor's skip count would
    charge a recovery probe for the pretraining run's history."""
    args = _tiny_args(tmp_path / "donor", max_steps=1)
    donor = Trainer(args)
    donor._skipped_updates = 5
    donor.fit()

    probe = Trainer(_tiny_args(tmp_path / "probe", max_steps=1,
                               init_from=str(tmp_path / "donor" / "checkpoint.pt")))
    assert probe._skipped_updates == 0
