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
from pathlib import Path

import pytest

from daedalus.scorecard import (ArtifactRef, Provenance, Scorecard,
                                sha256_file, write_scorecard)
from scripts.code_branch import BRANCH_TOKENS
from scripts.code_branch_report import (BranchScoringError, CollectedScore,
                                        DEFAULT_BASE_RETRIEVAL_DIR,
                                        RETRIEVAL_TASKS, RetrievalSettings,
                                        _cli, assert_items_reproduce,
                                        assert_one_artifact,
                                        assert_retrieval_paired,
                                        branch_pass_plan,
                                        build_branch_verdict,
                                        build_stop_record, collect,
                                        default_retrieval_paths,
                                        five_task_scores, harness_constraints,
                                        input_paths, missing_inputs,
                                        paired_retrieval_evidence,
                                        read_five_task_mean,
                                        read_tasks_payload,
                                        retrieval_identity_digest,
                                        retrieval_scores,
                                        retrieval_settings_from, score_branch,
                                        tasks_artifact_sha256,
                                        tasks_scored_from, tasks_settings_from)
from scripts.code_probe_report import BASE_MODEL, ScoredModel

_sha = sha256_file

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


def _bpb_card(path, name, sha, bpb, *, per_source=None, **details):
    """A `bpb` card with the breakdown every real one carries.

    `summarize_bpb` emits one `bpb_<id>` per source on every card this program
    writes, and the collector reports that breakdown beside the aggregate, so a
    fixture without it would exercise a card shape that cannot be produced.
    """

    sources = list(per_source) if per_source else list(
        details.get("sources_requested") or ["code-python"])
    values = dict(per_source or {source: bpb for source in sources})
    metrics = {"bpb": bpb, "n_sources": float(len(sources))}
    for source in sources:
        metrics[f"bpb_{source}"] = float(values[source])
        metrics[f"tokens_{source}"] = 1000.0
    write_scorecard(path, Scorecard(
        kind="bpb", name=name, provenance=_provenance(sha),
        metrics=metrics, created_at="2026-08-27T00:00:00Z",
        item_count=len(sources), details=dict(details)))


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


def _tasks_payload(sha, *, mean=0.45, drop=None, config="daedalus-150m",
                   seed=SEED, limit=None, device="cuda"):
    scores = {task: mean for task in
              ("hellaswag", "arc_easy", "piqa", "openbookqa", "winogrande")}
    if drop:
        scores.pop(drop)
    return {"provenance": {"checkpoints": [{"path": "checkpoint.pt",
                                            "sha256": sha}],
                           "config": config, "seed": seed, "device": device,
                           "tasks": {task: {"limit": limit}
                                     for task in scores}},
            "mean": {**scores, "hellaswag_n": 10042.0}}


def _write_model(root, name, sha, *, code_bpb=1.20, general_bpb=3.80,
                 tasks_mean=0.45, retrieval_exact=None, tasks_sha=None,
                 tasks_drop=None, syntax=0.24, retrieval_seed=SEED,
                 per_depth=2, retrieval_items=None, code_bpb_by_source=None):
    """A full card set for one model, in the layout the collector expects."""

    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _bpb_card(out_dir / "code-bpb.json", "code-bpb", sha, code_bpb,
              per_source=code_bpb_by_source)
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


def test_the_verdict_reports_code_bpb_per_source_beside_the_clause(tmp_path):
    """The code clause is a 5% improvement in a mixture-weighted mean over the
    holdout's languages, and the two execution benchmarks beside it are
    Python-only. At 250M tokens that mean moved -23.6% overall and -2.3% on
    Python; the breakdown is what says which of those the gate cleared on."""
    base = _write_model(tmp_path, "base", BASE_SHA, code_bpb=1.20,
                        code_bpb_by_source={"code-python": 1.00,
                                            "code-typescript": 1.40})
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA,
                          code_bpb=1.10, general_bpb=3.81,
                          code_bpb_by_source={"code-python": 0.99,
                                              "code-typescript": 1.21})

    verdict = build_branch_verdict(_collect(base), _collect(branch))

    by_source = verdict["code_bpb_by_source"]
    assert by_source["code-python"]["improvement_pct"] == pytest.approx(1.0)
    assert by_source["code-typescript"]["improvement_pct"] == pytest.approx(13.57,
                                                                            abs=0.01)
    assert by_source["code-typescript"]["base"] == pytest.approx(1.40)
    # And it is on the model record too, so a single side can be read alone.
    assert verdict["models"]["base"]["details"]["code_bpb_by_source"][
        "code-python"]["bpb"] == pytest.approx(1.00)


def test_a_branch_scored_over_different_code_sources_is_refused(tmp_path):
    """Two aggregates over different holdouts are not a difference between
    models, and the breakdown is the only place that would have shown."""
    base = _write_model(tmp_path, "base", BASE_SHA,
                        code_bpb_by_source={"code-python": 1.0,
                                            "code-typescript": 1.4})
    branch = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA,
                          code_bpb_by_source={"code-python": 0.99})

    from scripts.code_probe_report import ProbeScoringError
    with pytest.raises(ProbeScoringError, match="different holdouts"):
        build_branch_verdict(_collect(base), _collect(branch))


def test_a_code_card_without_a_breakdown_is_refused_at_collection(tmp_path):
    """A card whose aggregate cannot be broken down is a refusal of the model's
    inputs, named as such, rather than a crash in the middle of the gate."""
    model = _write_model(tmp_path, "code-branch-1b", BRANCH_SHA)
    write_scorecard(Path(model.out_dir) / "code-bpb.json", Scorecard(
        kind="bpb", name="code-bpb", provenance=_provenance(BRANCH_SHA),
        metrics={"bpb": 1.10}, created_at="2026-08-27T00:00:00Z",
        item_count=1))

    with pytest.raises(BranchScoringError, match="no per-source"):
        _collect(model)


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


# ------------------------------------------------------------ scoring pass ---
# The pass writes the branch's half of the pair. Everything below is about it
# being configured from the base's own cards rather than from a command line
# retyped days later, because every knob that differs between the two sides is a
# difference in the harness that the gate would read as a difference in the
# model -- and the expensive version of finding out is at the verdict, after the
# GPU time.


class WhitespaceTokenizer:
    """Words as tokens: deterministic, and enough for the item generators."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


GEN_DEPTHS = (128, 256)


def _generated_retrieval(out_dir, sha, *, depths=GEN_DEPTHS, per_depth=2,
                         control_items=2, n_queries=4, seed=SEED,
                         max_new_tokens=24, correct=True):
    """Real generator output, written as the cards `retrieval_eval.py` writes.

    The identity check regenerates items and compares digests, so a fixture of
    hand-written items could only ever fail it. These are the real ones.
    """

    from daedalus.retrieval import make_all_items, score_items, summarize

    items_by_task = make_all_items(WhitespaceTokenizer(), depths=list(depths),
                                   per_depth=per_depth, seed=seed,
                                   n_queries=n_queries,
                                   control_items=control_items)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, items in items_by_task.items():
        records = score_items(items, [item.answer if correct else ""
                                      for item in items])
        for item, record in zip(items, records):
            record["prompt"] = item.prompt
        write_scorecard(out_dir / f"retrieval-{task}.json", Scorecard(
            kind="retrieval", name=f"retrieval-{task}",
            provenance=_provenance(sha, seed=seed,
                                   runtime={"backend": "torch",
                                            "device": "cuda",
                                            "max_new_tokens": max_new_tokens}),
            metrics=summarize(items, records),
            created_at="2026-08-27T00:00:00Z", items=records))
    return sorted(f"retrieval-{task}" for task in items_by_task)


def _base_with_real_items(tmp_path, sha=BASE_SHA, **over):
    """A full base card set whose retrieval cards came from the generator."""

    model = _write_model(tmp_path, "base", sha)
    for stale in ("passkey", "mqar"):
        (tmp_path / "base" / f"retrieval-{stale}.json").unlink()
        (tmp_path / "base" / f"retrieval-{stale}.items.json").unlink()
    names = _generated_retrieval(tmp_path / "base", sha, **over)
    return model, collect(model, tasks=f"{model.out_dir}/tasks.json",
                          retrieval=[f"{model.out_dir}/{name}.json"
                                     for name in names])


def _branch_model(tmp_path, *, name="code-branch-1b", body=b"branch weights",
                  tokens=BRANCH_TOKENS):
    """A branch checkpoint on disk, with the metrics row that says it finished."""

    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(body)
    if tokens is not None:
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 1, "tokens": tokens}) + "\n")
    out_dir = tmp_path / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return ScoredModel(name=name, checkpoint=str(checkpoint),
                       out_dir=str(out_dir))


def _recording_runner(codes=None):
    commands = []

    def run(command):
        commands.append([str(part) for part in command])
        return (codes or {}).get(Path(str(command[1])).name, 0)

    return commands, run


def test_retrieval_settings_are_read_off_the_base_cards(tmp_path):
    _, base = _base_with_real_items(tmp_path, per_depth=3, control_items=5,
                                    max_new_tokens=32)

    settings = retrieval_settings_from(base.retrieval_cards)

    assert settings.depths == GEN_DEPTHS
    assert settings.per_depth == 3
    assert settings.control_items == 5
    assert settings.max_new_tokens == 32
    assert settings.seed == SEED
    assert settings.backend == "torch"


def test_retrieval_settings_refuse_cards_scored_at_different_seeds(tmp_path):
    _, base = _base_with_real_items(tmp_path)
    reseeded = dict(base.retrieval_cards)
    card = reseeded["retrieval-passkey"]
    reseeded["retrieval-passkey"] = Scorecard(
        kind=card.kind, name=card.name,
        provenance=_provenance(BASE_SHA, seed=7,
                               runtime=dict(card.provenance.runtime)),
        metrics=card.metrics, created_at=card.created_at, items=card.items)

    with pytest.raises(BranchScoringError, match="disagree on the seed"):
        retrieval_settings_from(reseeded)


def test_retrieval_settings_refuse_a_ragged_per_depth(tmp_path):
    """`retrieval_eval.py` scores one --per-depth for every depth."""
    _, base = _base_with_real_items(tmp_path)
    card = base.retrieval_cards["retrieval-mqar"]
    metrics = {**card.metrics, "n_d256": 1.0}
    ragged = {**base.retrieval_cards, "retrieval-mqar": Scorecard(
        kind=card.kind, name=card.name, provenance=card.provenance,
        metrics=metrics, created_at=card.created_at, items=card.items)}

    with pytest.raises(BranchScoringError, match="per-depth item count"):
        retrieval_settings_from(ragged)


def test_retrieval_settings_refuse_a_missing_copy_control(tmp_path):
    """--control-items would be a guess, and the control is in this gate."""
    _, base = _base_with_real_items(tmp_path)
    without = {name: card for name, card in base.retrieval_cards.items()
               if name != "retrieval-copy-control"}

    with pytest.raises(BranchScoringError, match="control-items"):
        retrieval_settings_from(without)


def test_retrieval_settings_refuse_a_llama_cpp_base(tmp_path):
    """A GGUF card is a different harness, not a different model."""
    _, base = _base_with_real_items(tmp_path)
    swapped = {name: Scorecard(
        kind=card.kind, name=card.name,
        provenance=_provenance(BASE_SHA, seed=SEED,
                               runtime={"backend": "llama-cpp",
                                        "max_new_tokens": 24}),
        metrics=card.metrics, created_at=card.created_at, items=card.items)
        for name, card in base.retrieval_cards.items()}

    with pytest.raises(BranchScoringError, match="llama-cpp"):
        retrieval_settings_from(swapped)


def test_the_settings_regenerate_the_bases_own_items(tmp_path):
    _, base = _base_with_real_items(tmp_path)
    settings = retrieval_settings_from(base.retrieval_cards)

    digests = assert_items_reproduce(settings, base.retrieval_cards,
                                     tokenizer=WhitespaceTokenizer())

    assert sorted(digests) == ["retrieval-copy-control", "retrieval-mqar",
                               "retrieval-passkey"]


def test_the_one_unrecorded_knob_is_caught_before_the_model_loads(tmp_path):
    """--n-queries appears only inside a prompt, so the items are the check."""
    _, base = _base_with_real_items(tmp_path, n_queries=4)
    settings = retrieval_settings_from(base.retrieval_cards, n_queries=3)

    with pytest.raises(BranchScoringError,
                       match="retrieval-mqar: these settings would score"):
        assert_items_reproduce(settings, base.retrieval_cards,
                               tokenizer=WhitespaceTokenizer())


def test_a_reseeded_pass_is_caught_by_the_item_digest(tmp_path):
    _, base = _base_with_real_items(tmp_path, seed=SEED)
    settings = retrieval_settings_from(base.retrieval_cards)
    reseeded = RetrievalSettings(
        depths=settings.depths, per_depth=settings.per_depth,
        control_items=settings.control_items, seed=SEED + 1,
        max_new_tokens=settings.max_new_tokens)

    with pytest.raises(BranchScoringError, match="would score different items"):
        assert_items_reproduce(reseeded, base.retrieval_cards,
                               tokenizer=WhitespaceTokenizer())


def test_the_branch_pass_plan_carries_every_derived_setting(tmp_path):
    _, base = _base_with_real_items(tmp_path, per_depth=3, control_items=5)
    branch = _branch_model(tmp_path)

    plan = branch_pass_plan(branch, base, device="cuda")
    retrieval = " ".join(plan["commands"]["retrieval"])
    tasks = " ".join(plan["commands"]["tasks"])

    assert plan["config"] == "daedalus-150m"
    assert "--per-depth 3" in retrieval and "--control-items 5" in retrieval
    assert "--depths 128,256" in retrieval and "--n-queries 4" in retrieval
    assert f"--seed {SEED}" in retrieval and "--max-new-tokens 24" in retrieval
    assert f"--out-dir {branch.out_dir}" in retrieval
    assert f"--checkpoints {branch.checkpoint}" in tasks
    assert "--no-wandb" in tasks
    # The base scored the full validation splits, so the branch must too.
    assert "--task-limit" not in tasks
    assert "humaneval-plus" in plan["commands"]


def test_a_task_limit_the_base_used_is_carried_into_the_branch(tmp_path):
    model, base = _base_with_real_items(tmp_path)
    (tmp_path / "base" / "tasks.json").write_text(
        json.dumps(_tasks_payload(BASE_SHA, limit=500)))
    base = collect(model, tasks=f"{model.out_dir}/tasks.json",
                   retrieval=[f"{model.out_dir}/retrieval-{task}.json"
                              for task in RETRIEVAL_TASKS])

    plan = branch_pass_plan(_branch_model(tmp_path), base)

    assert "--task-limit 500" in " ".join(plan["commands"]["tasks"])


def test_score_branch_runs_the_passes_the_branch_has_no_card_for(tmp_path):
    _, base = _base_with_real_items(tmp_path)
    branch = _branch_model(tmp_path)
    _bpb_card(Path(branch.out_dir) / "code-bpb.json", "code-bpb",
              _sha(branch.checkpoint), 1.10)
    _bpb_card(Path(branch.out_dir) / "general-bpb.json", "general-bpb",
              _sha(branch.checkpoint), 3.85)
    commands, runner = _recording_runner()

    outcome = score_branch(branch, base=base, bpb_plan={},
                           tokenizer=WhitespaceTokenizer(), runner=runner)

    assert outcome["bpb"]["skipped"] == "already-scored"
    assert [Path(command[1]).name for command in commands] == [
        "code_eval.py", "code_eval.py", "eval.py", "retrieval_eval.py"]
    assert outcome["ran"]["retrieval"]["rescored"] == [
        "retrieval-copy-control", "retrieval-mqar", "retrieval-passkey"]


def test_score_branch_reuses_cards_that_score_exactly_these_bytes(tmp_path):
    _, base = _base_with_real_items(tmp_path)
    branch = _branch_model(tmp_path)
    digest = _sha(branch.checkpoint)
    for name, value in (("code-bpb", 1.10), ("general-bpb", 3.85)):
        _bpb_card(Path(branch.out_dir) / f"{name}.json", name, digest, value)
    for dataset in ("humaneval-plus", "mbpp-plus"):
        _execution_card(Path(branch.out_dir) / f"{dataset}.json", dataset,
                        digest, n=164 if dataset == "humaneval-plus" else 378)
    (Path(branch.out_dir) / "tasks.json").write_text(
        json.dumps(_tasks_payload(digest)))
    _generated_retrieval(Path(branch.out_dir), digest)
    commands, runner = _recording_runner()

    outcome = score_branch(branch, base=base, bpb_plan={},
                           tokenizer=WhitespaceTokenizer(), runner=runner)

    assert commands == []
    assert outcome["ran"]["tasks"]["skipped"] == "already-scored"
    assert outcome["ran"]["retrieval"]["skipped"] == "already-scored"


def test_score_branch_refuses_a_checkpoint_that_is_the_base(tmp_path):
    _, base = _base_with_real_items(tmp_path)
    branch = _branch_model(tmp_path)
    Path(branch.checkpoint).write_bytes(b"base weights")
    trimmed = CollectedScore(score=base.score, sha256=_sha(branch.checkpoint),
                             cards=base.cards,
                             retrieval_cards=base.retrieval_cards,
                             execution_cards=base.execution_cards,
                             details=base.details)
    commands, runner = _recording_runner()

    with pytest.raises(BranchScoringError, match="is the base"):
        score_branch(branch, base=trimmed, bpb_plan={},
                     tokenizer=WhitespaceTokenizer(), runner=runner)
    assert commands == []


def test_score_branch_refuses_a_run_that_has_not_reached_its_budget(tmp_path):
    """A checkpoint is written throughout a run; existence is not completion."""
    _, base = _base_with_real_items(tmp_path)
    branch = _branch_model(tmp_path, tokens=250_000_000)
    commands, runner = _recording_runner()

    with pytest.raises(BranchScoringError, match="1,000,000,000-token budget"):
        score_branch(branch, base=base, bpb_plan={},
                     tokenizer=WhitespaceTokenizer(), runner=runner)
    assert commands == []


def test_score_branch_refuses_a_missing_checkpoint(tmp_path):
    _, base = _base_with_real_items(tmp_path)
    branch = _branch_model(tmp_path)
    Path(branch.checkpoint).unlink()

    with pytest.raises(BranchScoringError, match="no checkpoint"):
        score_branch(branch, base=base, bpb_plan={},
                     tokenizer=WhitespaceTokenizer())


def test_score_branch_refuses_misconfigured_settings_before_any_pass(tmp_path):
    """The item check runs before the checkpoint is loaded, not after."""
    _, base = _base_with_real_items(tmp_path, n_queries=4)
    branch = _branch_model(tmp_path)
    commands, runner = _recording_runner()

    with pytest.raises(BranchScoringError, match="would score different items"):
        score_branch(branch, base=base, bpb_plan={}, n_queries=2,
                     tokenizer=WhitespaceTokenizer(), runner=runner)
    assert commands == []


def test_a_failed_pass_is_raised_with_its_command(tmp_path):
    _, base = _base_with_real_items(tmp_path)
    branch = _branch_model(tmp_path)
    for name, value in (("code-bpb", 1.10), ("general-bpb", 3.85)):
        _bpb_card(Path(branch.out_dir) / f"{name}.json", name,
                  _sha(branch.checkpoint), value)
    for dataset in ("humaneval-plus", "mbpp-plus"):
        _execution_card(Path(branch.out_dir) / f"{dataset}.json", dataset,
                        _sha(branch.checkpoint),
                        n=164 if dataset == "humaneval-plus" else 378)
    commands, runner = _recording_runner({"eval.py": 3})

    with pytest.raises(BranchScoringError, match="five-task pass exited 3"):
        score_branch(branch, base=base, bpb_plan={},
                     tokenizer=WhitespaceTokenizer(), runner=runner)


def test_tasks_scored_from_keys_on_the_checkpoint_digest(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(_tasks_payload(BRANCH_SHA)))

    assert tasks_scored_from(path, BRANCH_SHA) is True
    assert tasks_scored_from(path, BASE_SHA) is False
    assert tasks_scored_from(tmp_path / "absent.json", BRANCH_SHA) is False


def test_tasks_settings_refuse_a_payload_with_mixed_limits(tmp_path):
    payload = _tasks_payload(BASE_SHA)
    payload["provenance"]["tasks"]["piqa"]["limit"] = 500

    with pytest.raises(BranchScoringError, match="different limits"):
        tasks_settings_from(payload, "tasks.json")


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


# ------------------------------------------------------ the stop's evidence ---
#
# The gate has already answered. These are about the record of that answer being
# about the same items the gate differenced, and about it refusing to exist for a
# branch the gate continued.

def _scored_items(task, outcomes, *, depths=DEPTHS):
    """Items whose `correct` follows `outcomes[depth]`, everything else fixed.

    The identity digest excludes the outcome fields, so two models' items differ
    here in exactly the way two real scoring passes differ and in no other.
    """

    items = []
    for depth in depths:
        for index, correct in enumerate(outcomes[depth]):
            items.append({
                "id": f"{task}-d{depth}-{index}", "task": task, "depth": depth,
                "needle_depth_frac": 0.0, "prompt_tokens": depth,
                "prompt": f"{task} at {depth} item {index}",
                "expected": f"{depth}{index}",
                "correct": int(correct), "query_accuracy": float(correct),
                "extracted": f"{depth}{index}" if correct else "",
                "response": f"{depth}{index}" if correct else "no",
            })
    return items


def _exact_from(outcomes):
    return {depth: sum(values) / len(values)
            for depth, values in outcomes.items()}


def _stopped_pair(tmp_path, *, base_outcomes=None, branch_outcomes=None,
                  **branch_over):
    """A base and a branch the gate stops, with per-item retrieval on both."""

    base_outcomes = base_outcomes or {256: [1, 1, 1, 1], 512: [1, 1, 1, 1]}
    branch_outcomes = branch_outcomes or {256: [1, 1, 1, 1], 512: [0, 0, 1, 1]}
    base = _write_model(
        tmp_path, "base", BASE_SHA, code_bpb=1.20, general_bpb=3.80,
        tasks_mean=0.45, per_depth=4, retrieval_exact=_exact_from(base_outcomes),
        retrieval_items=_scored_items("passkey", base_outcomes))
    branch = _write_model(
        tmp_path, "code-branch-1b", BRANCH_SHA, per_depth=4,
        **{"code_bpb": 1.10, "general_bpb": 3.81, "tasks_mean": 0.45,
           "retrieval_exact": _exact_from(branch_outcomes),
           "retrieval_items": _scored_items("passkey", branch_outcomes),
           **branch_over})
    return _collect(base), _collect(branch)


def test_paired_evidence_counts_the_disagreements_at_each_depth(tmp_path):
    collected_base, collected_branch = _stopped_pair(tmp_path)

    evidence = paired_retrieval_evidence(
        collected_base.retrieval_cards["retrieval-passkey"],
        collected_branch.retrieval_cards["retrieval-passkey"])

    # Keyed as `retrieval_scores` keys, so a row sits beside its own drop.
    assert sorted(evidence) == ["retrieval-passkey:d256", "retrieval-passkey:d512"]
    unchanged = evidence["retrieval-passkey:d256"]
    assert unchanged["n_discordant"] == 0 and unchanged["drop_points"] == 0.0
    fell = evidence["retrieval-passkey:d512"]
    assert (fell["base_only"], fell["branch_only"]) == (2, 0)
    assert fell["base_correct"] == 4 and fell["branch_correct"] == 2
    # 2 of 4 items, and the gate's sign convention: positive is the branch worse.
    assert fell["drop_points"] == pytest.approx(50.0)
    assert fell["thin"] is True


def test_the_paired_drop_is_the_drop_the_gate_measured(tmp_path):
    """Two numbers over the same items that disagreed would mean one of them is
    about something else. The card's `exact_match_d<depth>` is the mean of the
    same `correct` field this pairs on, so they cannot."""
    collected_base, collected_branch = _stopped_pair(tmp_path)
    verdict = build_branch_verdict(collected_base, collected_branch)

    evidence = paired_retrieval_evidence(
        collected_base.retrieval_cards["retrieval-passkey"],
        collected_branch.retrieval_cards["retrieval-passkey"])
    gate = next(check for check in verdict["gate"]["checks"]
                if check["gate"] == "retrieval")

    for key, row in evidence.items():
        assert row["drop_points"] == pytest.approx(
            gate["per_key"][key]["drop_points"])


def test_paired_evidence_refuses_cards_that_scored_different_items(tmp_path):
    """`assert_retrieval_paired`'s refusals are this function's too. A p-value
    over two different needle sets is confident and meaningless."""
    collected_base, _ = _stopped_pair(tmp_path)
    other = _write_model(tmp_path, "reseeded", BRANCH_SHA, per_depth=4,
                         retrieval_seed=SEED + 1,
                         retrieval_items=_scored_items(
                             "passkey", {256: [1, 1, 1, 1], 512: [1, 1, 0, 0]}))

    with pytest.raises(BranchScoringError, match="seed"):
        paired_retrieval_evidence(
            collected_base.retrieval_cards["retrieval-passkey"],
            _collect(other).retrieval_cards["retrieval-passkey"])


def test_a_stop_record_carries_every_failed_clause_and_its_evidence(tmp_path):
    # 3.80 -> 3.88 is +2.1% general, past 1.5%, alongside the retrieval drop.
    collected_base, collected_branch = _stopped_pair(tmp_path, general_bpb=3.88)

    record = build_stop_record(collected_base, collected_branch)

    assert record["decision"] == "stop"
    assert record["verdict"]["continue"] is False
    assert sorted(check["gate"] for check in record["failed"]) == [
        "general-bpb", "retrieval"]
    assert record["evidence"]["retrieval_paired"][
        "retrieval-passkey:d512"]["base_only"] == 2
    # And which source the general regression is in, not just that there is one.
    by_source = record["evidence"]["general_bpb_by_source"]
    assert sorted(by_source) == ["dclm-baseline", "fineweb-edu"]
    assert by_source["fineweb-edu"]["improvement_pct"] < 0


def test_a_branch_the_gate_continued_has_no_stop_to_record(tmp_path):
    """The file outlives the session, so a record that says `stop` about a
    branch the gate continued misreports the program's own decision."""
    collected_base, collected_branch = _stopped_pair(
        tmp_path, branch_outcomes={256: [1, 1, 1, 1], 512: [1, 1, 1, 1]})

    with pytest.raises(BranchScoringError, match="no stop to record"):
        build_stop_record(collected_base, collected_branch)


def test_stop_record_writes_beside_the_verdict_without_rewriting_it(tmp_path,
                                                                   capsys):
    base = _write_model(tmp_path, "base", BASE_SHA, code_bpb=1.20,
                        general_bpb=3.80, per_depth=4,
                        retrieval_items=_scored_items(
                            "passkey", {256: [1, 1, 1, 1], 512: [1, 1, 1, 1]}))
    branch = _write_model(
        tmp_path, "code-branch-1b", BRANCH_SHA, code_bpb=1.10,
        general_bpb=3.88, per_depth=4,
        retrieval_exact={256: 1.0, 512: 0.5},
        retrieval_items=_scored_items("passkey",
                                      {256: [1, 1, 1, 1], 512: [0, 0, 1, 1]}))
    out_path = tmp_path / "branch-1b-stop.json"

    code = _cli(_cli_args(tmp_path, base, branch, "stop-record",
                          "--json-out", str(out_path)))
    payload = json.loads(out_path.read_text())

    assert code == 0
    assert payload["decision"] == "stop"
    assert payload["branch"] == "code-branch-1b"
    assert "decides nothing" in payload["reading"]
    assert not (tmp_path / "branch-1b-verdict.json").exists()
    assert "STOP" in capsys.readouterr().out
