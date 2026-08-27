"""Tests for phase 8 gate 2's collector.

The rule this feeds -- `code_gates.branch_1b_verdict` -- has its own tests, and
they are about thresholds. These are about the four measurements arriving as one
comparable pair, because every way that can go wrong produces a finite,
plausible, wrong retention number rather than an error:

  - the base's five-task and retrieval cards were written in phase 3, weeks of
    GPU time before the code-side ones. If they scored a different checkpoint,
    all four clauses move at once and none of them is about the branch;
  - retrieval scored at `--per-depth 10` against a base scored at 100 is a
    different denominator, not a drop;
  - a re-seeded retrieval run reuses the same item ids behind different needles;
  - a five-task mean over four tasks is smaller than one over five for a reason
    that has nothing to do with the model, against a one-point limit.

None of this needs torch: the collector reads scorecards and an `eval.py`
payload, so the fixtures write them.
"""

import json

import pytest

from daedalus.scorecard import (ArtifactRef, Provenance, Scorecard,
                                write_scorecard)
from scripts.code_branch_report import (BranchScoringError, CollectedScore,
                                        DEFAULT_BASE_RETRIEVAL_DIR,
                                        RETRIEVAL_TASKS, _cli,
                                        assert_one_artifact,
                                        assert_retrieval_paired,
                                        build_branch_verdict, collect,
                                        default_retrieval_paths,
                                        five_task_scores, harness_constraints,
                                        input_paths, missing_inputs,
                                        read_five_task_mean,
                                        read_tasks_payload,
                                        retrieval_identity_digest,
                                        retrieval_scores,
                                        tasks_artifact_sha256)
from scripts.code_probe_report import BASE_MODEL, ScoredModel

BASE_SHA = "b" * 64
BRANCH_SHA = "c" * 64
ZERO_SHA = "0" * 64
SEED = 20260824

#: Two depths is enough to show "at every depth" behaving; the real cards carry
#: four plus the control's one.
DEPTHS = (256, 512)


def _provenance(sha, *, seed=SEED, runtime=None):
    return Provenance(
        artifact=ArtifactRef(path="checkpoint.pt", sha256=sha,
                             kind="checkpoint", config="daedalus-150m"),
        tokenizer=ArtifactRef(path="<smollm2-default>", sha256=ZERO_SHA,
                              kind="tokenizer"),
        seed=seed, git_sha="abc1234",
        runtime=dict(runtime or {"backend": "torch", "max_new_tokens": 24}))


def _bpb_card(path, name, sha, bpb, **details):
    write_scorecard(path, Scorecard(
        kind="bpb", name=name, provenance=_provenance(sha),
        metrics={"bpb": bpb}, created_at="2026-08-27T00:00:00Z",
        item_count=1, details=dict(details)))


def _execution_card(path, name, sha, *, pass_at_1=0.0, pass_plus=0.0,
                    syntax=0.24, n=378, runtime=None):
    write_scorecard(path, Scorecard(
        kind="code-execution", name=name,
        provenance=_provenance(sha, runtime=runtime or {
            "backend": "torch", "max_new_tokens": 384}),
        metrics={"pass@1": pass_at_1, "pass@1_plus": pass_plus,
                 "syntax_valid": syntax, "n": float(n)},
        created_at="2026-08-27T00:00:00Z", item_count=n))


def _retrieval_items(task, *, depths=DEPTHS, per_depth=2, correct=1,
                     needle_shift=0.0):
    items = []
    for depth in depths:
        for index in range(per_depth):
            items.append({
                "id": f"{task}-d{depth}-{index}",
                "task": task,
                "depth": depth,
                "needle_depth_frac": needle_shift,
                "prompt_tokens": depth,
                "prompt": f"{task} at {depth} item {index}",
                "expected": f"{depth}{index}",
                # Model-dependent: excluded from the identity digest.
                "correct": correct,
                "query_accuracy": float(correct),
                "extracted": f"{depth}{index}" if correct else "",
                "response": f"{depth}{index}" if correct else "no",
            })
    return items


def _retrieval_card(path, task, sha, *, exact=None, depths=DEPTHS, per_depth=2,
                    seed=SEED, runtime=None, items=None):
    exact = exact or {depth: 1.0 for depth in depths}
    metrics = {"exact_match": sum(exact.values()) / len(exact),
               "query_accuracy": sum(exact.values()) / len(exact),
               "n": float(len(depths) * per_depth)}
    for depth in depths:
        metrics[f"exact_match_d{depth}"] = exact[depth]
        metrics[f"n_d{depth}"] = float(per_depth)
    write_scorecard(path, Scorecard(
        kind="retrieval", name=f"retrieval-{task}",
        provenance=_provenance(sha, seed=seed, runtime=runtime),
        metrics=metrics, created_at="2026-08-27T00:00:00Z",
        items=items if items is not None
        else _retrieval_items(task, depths=depths, per_depth=per_depth)))


def _tasks_payload(sha, *, mean=0.45, drop=None):
    scores = {task: mean for task in
              ("hellaswag", "arc_easy", "piqa", "openbookqa", "winogrande")}
    if drop:
        scores.pop(drop)
    return {"provenance": {"checkpoints": [{"path": "checkpoint.pt",
                                            "sha256": sha}]},
            "mean": {**scores, "hellaswag_n": 10042.0}}


def _write_model(root, name, sha, *, code_bpb=1.20, general_bpb=3.80,
                 tasks_mean=0.45, retrieval_exact=None, tasks_sha=None,
                 tasks_drop=None, syntax=0.24, retrieval_seed=SEED,
                 per_depth=2, retrieval_items=None):
    """A full card set for one model, in the layout the collector expects."""

    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _bpb_card(out_dir / "code-bpb.json", "code-bpb", sha, code_bpb)
    _bpb_card(out_dir / "general-bpb.json", "general-bpb", sha, general_bpb,
              bucket_share_covered=0.78,
              sources_requested=["dclm-baseline", "fineweb-edu"])
    for dataset in ("humaneval-plus", "mbpp-plus"):
        _execution_card(out_dir / f"{dataset}.json", dataset, sha,
                        syntax=syntax, n=164 if dataset == "humaneval-plus" else 378)
    for task in ("passkey", "mqar"):
        _retrieval_card(out_dir / f"retrieval-{task}.json", task, sha,
                        exact=retrieval_exact, seed=retrieval_seed,
                        per_depth=per_depth, items=retrieval_items)
    (out_dir / "tasks.json").write_text(json.dumps(
        _tasks_payload(tasks_sha or sha, mean=tasks_mean, drop=tasks_drop)))
    return ScoredModel(name=name, checkpoint=f"runs/{name}/checkpoint.pt",
                       out_dir=str(out_dir))


def _retrieval_paths(model):
    return [f"{model.out_dir}/retrieval-{task}.json"
            for task in ("passkey", "mqar")]


def _collect(model, **over):
    return collect(model, tasks=f"{model.out_dir}/tasks.json",
                   retrieval=_retrieval_paths(model), **over)


# ------------------------------------------------------------------- paths ---

def test_input_paths_label_every_card_by_what_it_supplies(tmp_path):
    model = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    paths = input_paths(model, tasks=f"{model.out_dir}/tasks.json",
                        retrieval=_retrieval_paths(model))

    assert set(paths) == {"code-bpb", "general-bpb", "humaneval-plus",
                          "mbpp-plus", "tasks", "retrieval-passkey",
                          "retrieval-mqar"}
    assert missing_inputs(paths) == []


def test_missing_inputs_names_the_absent_labels(tmp_path):
    model = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    (tmp_path / "code-branch-1b" / "retrieval-mqar.json").unlink()
    paths = input_paths(model, tasks=f"{model.out_dir}/tasks.json",
                        retrieval=_retrieval_paths(model))

    assert missing_inputs(paths) == ["retrieval-mqar"]


def test_a_retrieval_card_named_otherwise_is_refused(tmp_path):
    """Identified by name, so a differently named card would be read as neither."""
    model = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)

    with pytest.raises(BranchScoringError, match="retrieval-<task>.json"):
        input_paths(model, retrieval=[f"{model.out_dir}/passkey.json"])


def test_base_retrieval_defaults_point_at_the_phase_3_baseline():
    assert default_retrieval_paths(DEFAULT_BASE_RETRIEVAL_DIR) == [
        f"{DEFAULT_BASE_RETRIEVAL_DIR}/retrieval-{task}.json"
        for task in RETRIEVAL_TASKS]


# --------------------------------------------------------------- retrieval ---

def test_retrieval_scores_key_as_the_phase_3_baseline_does(tmp_path):
    """`<task>:d<depth>` -- the key `runs/qat-recovery/baseline.json` records."""
    model = _write_model(tmp_path, "base", BASE_SHA)
    collected = _collect(model)

    assert collected.score.retrieval == {
        "retrieval-passkey:d256": 1.0, "retrieval-passkey:d512": 1.0,
        "retrieval-mqar:d256": 1.0, "retrieval-mqar:d512": 1.0}


def test_the_undepthed_aggregate_is_not_a_key(tmp_path):
    """It cannot fail independently, so counting it would be a fifth depth."""
    model = _write_model(tmp_path, "base", BASE_SHA,
                         retrieval_exact={256: 1.0, 512: 0.5})
    collected = _collect(model)

    assert "retrieval-passkey:dexact_match" not in collected.score.retrieval
    assert all(":d" in key for key in collected.score.retrieval)
    assert len(collected.score.retrieval) == 4


def test_a_non_retrieval_card_is_refused(tmp_path):
    card = Scorecard(kind="bpb", name="retrieval-passkey",
                     provenance=_provenance(BASE_SHA), metrics={"bpb": 1.0},
                     created_at="2026-08-27T00:00:00Z", item_count=1)

    with pytest.raises(BranchScoringError, match="not 'retrieval'"):
        retrieval_scores({"retrieval-passkey": card})


def test_a_retrieval_card_without_per_depth_metrics_is_refused(tmp_path):
    card = Scorecard(kind="retrieval", name="retrieval-passkey",
                     provenance=_provenance(BASE_SHA),
                     metrics={"exact_match": 0.9},
                     created_at="2026-08-27T00:00:00Z", item_count=4)

    with pytest.raises(BranchScoringError, match="no per-depth exact match"):
        retrieval_scores({"retrieval-passkey": card})


def test_identity_digest_ignores_the_models_own_answers(tmp_path):
    """Two models on the same items pair; `paired_outcomes` would refuse them."""
    right = tmp_path / "right.json"
    wrong = tmp_path / "wrong.json"
    _retrieval_card(right, "passkey", BASE_SHA,
                    items=_retrieval_items("passkey", correct=1))
    _retrieval_card(wrong, "passkey", BRANCH_SHA,
                    items=_retrieval_items("passkey", correct=0))

    from daedalus.scorecard import load_scorecard
    assert retrieval_identity_digest(load_scorecard(right)) == \
        retrieval_identity_digest(load_scorecard(wrong))


def test_identity_digest_catches_a_reseeded_run_behind_the_same_ids(tmp_path):
    right = tmp_path / "right.json"
    reseeded = tmp_path / "reseeded.json"
    _retrieval_card(right, "passkey", BASE_SHA)
    _retrieval_card(reseeded, "passkey", BRANCH_SHA,
                    items=_retrieval_items("passkey", needle_shift=0.5))

    from daedalus.scorecard import load_scorecard
    with pytest.raises(BranchScoringError, match="identity digests differ"):
        assert_retrieval_paired(load_scorecard(right), load_scorecard(reseeded))


def test_a_different_seed_is_refused_before_the_digest_is_read(tmp_path):
    from daedalus.scorecard import load_scorecard
    left, right = tmp_path / "l.json", tmp_path / "r.json"
    _retrieval_card(left, "passkey", BASE_SHA, seed=SEED)
    _retrieval_card(right, "passkey", BRANCH_SHA, seed=SEED + 1)

    with pytest.raises(BranchScoringError, match="scored different needles"):
        assert_retrieval_paired(load_scorecard(left), load_scorecard(right))


def test_a_smaller_per_depth_denominator_is_refused(tmp_path):
    """`--per-depth 10` against a base at 100 is a different measurement."""
    from daedalus.scorecard import load_scorecard
    left, right = tmp_path / "l.json", tmp_path / "r.json"
    _retrieval_card(left, "passkey", BASE_SHA, per_depth=4)
    _retrieval_card(right, "passkey", BRANCH_SHA, per_depth=2)

    with pytest.raises(BranchScoringError, match="per-depth item counts differ"):
        assert_retrieval_paired(load_scorecard(left), load_scorecard(right))


def test_a_different_generation_budget_is_refused(tmp_path):
    from daedalus.scorecard import load_scorecard
    left, right = tmp_path / "l.json", tmp_path / "r.json"
    _retrieval_card(left, "passkey", BASE_SHA,
                    runtime={"backend": "torch", "max_new_tokens": 24})
    _retrieval_card(right, "passkey", BRANCH_SHA,
                    runtime={"backend": "torch", "max_new_tokens": 64})

    with pytest.raises(BranchScoringError, match="max_new_tokens"):
        assert_retrieval_paired(load_scorecard(left), load_scorecard(right))


def test_two_cards_naming_different_tasks_are_refused(tmp_path):
    from daedalus.scorecard import load_scorecard
    left, right = tmp_path / "l.json", tmp_path / "r.json"
    _retrieval_card(left, "passkey", BASE_SHA)
    _retrieval_card(right, "mqar", BRANCH_SHA)

    with pytest.raises(BranchScoringError, match="different tasks"):
        assert_retrieval_paired(load_scorecard(left), load_scorecard(right))


# -------------------------------------------------------------- five tasks ---

def test_five_task_mean_is_in_points(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(_tasks_payload(BASE_SHA, mean=0.45)))

    assert read_five_task_mean(read_tasks_payload(path), path) == \
        pytest.approx(45.0)


def test_a_mean_over_four_tasks_is_refused_by_name(tmp_path):
    """Smaller than a mean over five, for a reason that is not the model."""
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(_tasks_payload(BASE_SHA, drop="winogrande")))

    with pytest.raises(BranchScoringError, match="winogrande"):
        read_five_task_mean(read_tasks_payload(path), path)


def test_five_task_scores_are_reported_per_task(tmp_path):
    payload = _tasks_payload(BASE_SHA, mean=0.45)

    assert five_task_scores(payload) == {
        task: pytest.approx(45.0) for task in
        ("hellaswag", "arc_easy", "piqa", "openbookqa", "winogrande")}


def test_two_checkpoints_in_one_tasks_payload_are_refused(tmp_path):
    path = tmp_path / "tasks.json"
    payload = _tasks_payload(BASE_SHA)
    payload["provenance"]["checkpoints"].append({"sha256": BRANCH_SHA})
    path.write_text(json.dumps(payload))

    with pytest.raises(BranchScoringError, match="2 checkpoint digest"):
        tasks_artifact_sha256(read_tasks_payload(path), path)


def test_a_tasks_payload_with_no_checkpoint_is_refused(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"mean": {}}))

    with pytest.raises(BranchScoringError, match="0 checkpoint digest"):
        tasks_artifact_sha256(read_tasks_payload(path), path)


def test_an_unreadable_tasks_payload_is_refused(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text("{not json")

    with pytest.raises(BranchScoringError, match="cannot read"):
        read_tasks_payload(path)


# -------------------------------------------------------------- collection ---

def test_collect_builds_all_five_measurements(tmp_path):
    model = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA,
                         code_bpb=1.10, general_bpb=3.85, tasks_mean=0.44)
    collected = _collect(model)

    assert collected.sha256 == BRANCH_SHA
    assert collected.score.code_bpb == pytest.approx(1.10)
    assert collected.score.general_bpb == pytest.approx(3.85)
    assert collected.score.five_task_mean == pytest.approx(44.0)
    assert set(collected.score.execution) == {"humaneval-plus", "mbpp-plus"}
    assert len(collected.score.retrieval) == 4
    assert collected.details["general_bpb_share_covered"] == pytest.approx(0.78)


def test_collect_names_the_missing_card_and_its_path(tmp_path):
    model = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    (tmp_path / "code-branch-1b" / "general-bpb.json").unlink()

    with pytest.raises(BranchScoringError, match="general-bpb"):
        _collect(model)


def test_a_base_tasks_card_from_another_checkpoint_is_refused(tmp_path):
    """The headline failure: phase 3's artifacts must be the same base bytes."""
    model = _write_model(tmp_path, "base", BASE_SHA, tasks_sha=BRANCH_SHA)

    with pytest.raises(BranchScoringError, match="different checkpoints"):
        _collect(model)


def test_an_expected_digest_that_does_not_match_is_refused(tmp_path):
    model = _write_model(tmp_path, "base", BASE_SHA)

    with pytest.raises(BranchScoringError, match="expected checkpoint"):
        _collect(model, expected_sha256=BRANCH_SHA)


def test_assert_one_artifact_reports_every_label_it_saw():
    with pytest.raises(BranchScoringError, match="tasks="):
        assert_one_artifact({"code-bpb": BASE_SHA, "tasks": BRANCH_SHA},
                            model="base")


def test_harness_constraints_carry_what_a_matching_pass_must_use(tmp_path):
    model = _write_model(tmp_path, "base", BASE_SHA, per_depth=100)
    constraints = harness_constraints(_collect(model))

    assert constraints["retrieval-passkey"]["seed"] == SEED
    assert constraints["retrieval-passkey"]["per_depth_n"] == {
        "n_d256": 100, "n_d512": 100}
    assert constraints["mbpp-plus"]["max_new_tokens"] == 384
    assert "per_depth_n" not in constraints["mbpp-plus"]


# -------------------------------------------------------------------- gate ---

def _pair(tmp_path, **branch_over):
    base = _write_model(tmp_path, "base", BASE_SHA, code_bpb=1.20,
                        general_bpb=3.80, tasks_mean=0.45)
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA,
                          **{"code_bpb": 1.10, "general_bpb": 3.81,
                             "tasks_mean": 0.45, **branch_over})
    return _collect(base), _collect(branch)


def test_a_branch_that_clears_every_clause_continues(tmp_path):
    verdict = build_branch_verdict(*_pair(tmp_path))

    assert verdict["continue"] is True
    assert verdict["gate"]["failed"] == []
    assert verdict["models"]["base"]["sha256"] == BASE_SHA
    assert verdict["five_task_scores"]["code-branch-1b"]["piqa"] == \
        pytest.approx(45.0)


def test_a_general_bpb_regression_past_the_limit_stops_it(tmp_path):
    # 3.80 -> 3.88 is +2.1%, past the preregistered 1.5%.
    verdict = build_branch_verdict(*_pair(tmp_path, general_bpb=3.88))

    assert verdict["continue"] is False
    assert "general-bpb" in verdict["gate"]["failed"]


def test_a_retrieval_drop_at_one_depth_stops_it(tmp_path):
    verdict = build_branch_verdict(
        *_pair(tmp_path, retrieval_exact={256: 1.0, 512: 0.90}))

    assert verdict["continue"] is False
    assert "retrieval" in verdict["gate"]["failed"]
    retrieval = next(check for check in verdict["gate"]["checks"]
                     if check["gate"] == "retrieval")
    assert retrieval["worst_drop_points"] == pytest.approx(10.0)


def test_code_bpb_that_does_not_clear_two_percent_stops_it(tmp_path):
    # 1.20 -> 1.19 is 0.83%, under the borrowed 2.0% bar.
    verdict = build_branch_verdict(*_pair(tmp_path, code_bpb=1.19))

    assert verdict["continue"] is False
    assert "code-bpb" in verdict["gate"]["failed"]


def test_scoring_the_same_checkpoint_twice_is_refused(tmp_path):
    base = _write_model(tmp_path, "base", BASE_SHA)
    branch = _write_model(tmp_path, "code-branch-1b", BASE_SHA)

    with pytest.raises(BranchScoringError, match="same checkpoint"):
        build_branch_verdict(_collect(base), _collect(branch))


def test_a_branch_missing_a_retrieval_card_the_base_has_is_refused(tmp_path):
    base = _write_model(tmp_path, "base", BASE_SHA)
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    collected_base = _collect(base)
    collected_branch = _collect(branch)
    trimmed = CollectedScore(
        score=collected_branch.score, sha256=collected_branch.sha256,
        cards=collected_branch.cards,
        retrieval_cards={"retrieval-passkey":
                         collected_branch.retrieval_cards["retrieval-passkey"]},
        execution_cards=collected_branch.execution_cards,
        details=collected_branch.details)

    with pytest.raises(BranchScoringError, match="at every depth"):
        build_branch_verdict(collected_base, trimmed)


def test_execution_cards_from_different_harnesses_are_refused(tmp_path):
    base = _write_model(tmp_path, "base", BASE_SHA)
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    _execution_card(tmp_path / "code-branch-1b" / "mbpp-plus.json",
                    "mbpp-plus", BRANCH_SHA, n=378,
                    runtime={"backend": "torch", "max_new_tokens": 512})

    from scripts.code_probe_report import ProbeScoringError
    with pytest.raises(ProbeScoringError, match="not comparable"):
        build_branch_verdict(_collect(base), _collect(branch))


# --------------------------------------------------------------------- cli ---

def _cli_args(tmp_path, base, branch, *extra):
    return [*extra,
            "--base-checkpoint", base.checkpoint,
            "--base-out-dir", base.out_dir,
            "--base-tasks", f"{base.out_dir}/tasks.json",
            *sum((["--base-retrieval", path]
                  for path in _retrieval_paths(base)), []),
            "--branch-name", branch.name,
            "--branch-checkpoint", branch.checkpoint,
            "--branch-out-dir", branch.out_dir,
            "--branch-tasks", f"{branch.out_dir}/tasks.json",
            *sum((["--branch-retrieval", path]
                  for path in _retrieval_paths(branch)), [])]


def test_plan_exits_non_zero_while_the_branch_is_unscored(tmp_path, capsys):
    base = _write_model(tmp_path, "base", BASE_SHA)
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    (tmp_path / "code-branch-1b" / "tasks.json").unlink()

    code = _cli(_cli_args(tmp_path, base, branch, "plan"))
    out = capsys.readouterr().out

    assert code == 1
    assert "missing: tasks" in out
    # The base's own settings, so the branch pass is configured from provenance.
    assert "retrieval-passkey" in out and "\"seed\": 20260824" in out


def test_plan_exits_zero_when_both_sides_are_ready(tmp_path, capsys):
    base = _write_model(tmp_path, "base", BASE_SHA)
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)

    code = _cli(_cli_args(tmp_path, base, branch, "plan"))

    assert code == 0
    assert "both sides are ready" in capsys.readouterr().out


def test_verdict_writes_the_gate_payload(tmp_path, capsys):
    base = _write_model(tmp_path, "base", BASE_SHA, code_bpb=1.20)
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA, code_bpb=1.10)
    out_path = tmp_path / "branch-1b-verdict.json"

    code = _cli(_cli_args(tmp_path, base, branch, "verdict",
                          "--json-out", str(out_path)))
    payload = json.loads(out_path.read_text())

    assert code == 0
    assert payload["gate"]["gate"] == "branch_1b"
    assert payload["continue"] is True
    assert "continue" in capsys.readouterr().out


def test_verdict_refuses_rather_than_writing_a_partial_payload(tmp_path, capsys):
    base = _write_model(tmp_path, "base", BASE_SHA)
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    (tmp_path / "code-branch-1b" / "retrieval-mqar.json").unlink()
    out_path = tmp_path / "branch-1b-verdict.json"

    code = _cli(_cli_args(tmp_path, base, branch, "verdict",
                          "--json-out", str(out_path)))

    assert code == 2
    assert not out_path.exists()
    assert "REFUSE" in capsys.readouterr().err
