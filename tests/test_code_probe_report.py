"""Tests for phase 8's out-of-band probe scoring pass.

The gate itself is `daedalus/code_gates.py` and has its own tests. What this
module adds is everything *around* the gate -- which holdout source feeds which
aggregate, which checkpoints get measured, and when a measurement may be reused
-- and each of those has a failure mode that produces a plausible number rather
than an error:

  - a code source counted as general replay reports code training as retention;
  - a general aggregate measured over two of six sources without saying so
    reads as the whole replay distribution;
  - a smoke report scored as the gate answers a question nobody asked;
  - a card reused because the file exists rather than because it scored these
    bytes answers a retrained arm with the old arm's number.

So the tests are mostly refusals. None of them needs torch or a GPU: the scoring
functions take the model loader as an argument for exactly that reason.
"""

import json

import pytest

from daedalus.code_gates import ProbeGateError
from daedalus.scorecard import (ArtifactRef, Provenance, Scorecard,
                                write_scorecard)
from scripts.code_probe_report import (BASE_MODEL, CODE_CARD, GENERAL_CARD,
                                       PROBE_RETENTION_REGRESSION_PCT_MAX,
                                       ProbeScoringError, ScoredModel,
                                       bucket_weights, build_verdict,
                                       completed_arms, execution_command,
                                       models_for, retention, score_bpb,
                                       score_execution, scored_from,
                                       scoring_plan)

ZERO_SHA = "0" * 64
ARM_SHA = "a" * 64

#: The composed mixture's own bucket block, cut down to what the holdout has.
RECORD = {
    "holdout_root": "",                     # filled in per test
    "buckets": {
        "code": {"code-python": 0.55,
                 "code-javascript-typescript-javascript-all": 0.1934,
                 "code-javascript-typescript-typescript-all": 0.2566},
        "general-replay": {"dclm-baseline": 0.2922, "fineweb-edu": 0.4870,
                           "cosmopedia-v2": 0.0649, "finephrase": 0.0909,
                           "finewiki-en": 0.0390,
                           "everyday-conversations": 0.0260},
        "technical": {"finemath-3plus": 0.2143, "finepdfs-edu": 0.5714,
                      "infiwebmath-3plus": 0.2143},
    },
    "caveats": ["stack-edu-python is not general retention"],
}

HOLDOUT_SOURCES = ("code-python",
                   "code-javascript-typescript-javascript-all",
                   "code-javascript-typescript-typescript-all",
                   "dclm-baseline", "fineweb-edu")


def _holdout(tmp_path, sources=HOLDOUT_SOURCES, tokens=1000):
    root = tmp_path / "holdout"
    for name in sources:
        source = root / name
        source.mkdir(parents=True)
        (source / "manifest.json").write_text(json.dumps({"total_tokens": tokens}))
    return root


def _record(tmp_path, **overrides):
    record = json.loads(json.dumps(RECORD))
    record["holdout_root"] = str(_holdout(tmp_path))
    record.update(overrides)
    return record


def _probe_report(arms=("code-probe-lr0.0005", "code-probe-lr0.001"), **over):
    report = {
        "gate": "probes_250m",
        "smoke": False,
        "arms": [{"arm": {"name": name}, "run_dir": f"runs/{name}",
                  "summary": {"complete": True}} for name in arms],
    }
    report.update(over)
    return report


# ------------------------------------------------------------------ buckets ---

def test_bucket_weights_renormalize_within_the_bucket(tmp_path):
    """Two of six replay sources are the replay aggregate, at 0.375/0.625."""
    spec = bucket_weights(_record(tmp_path), "general-replay", HOLDOUT_SOURCES)

    assert spec["sources"] == ["dclm-baseline", "fineweb-edu"]
    assert spec["weights"]["dclm-baseline"] == pytest.approx(0.2922 / 0.7792, rel=1e-4)
    assert spec["weights"]["fineweb-edu"] == pytest.approx(0.4870 / 0.7792, rel=1e-4)
    assert sum(spec["weights"].values()) == pytest.approx(1.0)


def test_bucket_weights_report_the_share_they_cover(tmp_path):
    """The number that separates "the replay distribution" from "78% of it"."""
    spec = bucket_weights(_record(tmp_path), "general-replay", HOLDOUT_SOURCES)

    assert spec["share_covered"] == pytest.approx(0.7792, rel=1e-3)
    assert set(spec["absent"]) == {"cosmopedia-v2", "finephrase", "finewiki-en",
                                   "everyday-conversations"}


def test_code_bucket_is_fully_covered(tmp_path):
    spec = bucket_weights(_record(tmp_path), "code", HOLDOUT_SOURCES)

    assert spec["share_covered"] == pytest.approx(1.0)
    assert spec["absent"] == []


def test_bucket_with_no_holdout_source_is_refused(tmp_path):
    with pytest.raises(ProbeScoringError, match="would be measured over nothing"):
        bucket_weights(_record(tmp_path), "technical", HOLDOUT_SOURCES)


def test_code_replay_source_may_not_be_general_retention(tmp_path):
    """A model trained on GitHub Python must not be credited for holding out on
    GitHub Python. The mixture record excludes it; this refuses a record that
    stops excluding it."""
    record = _record(tmp_path)
    record["buckets"]["general-replay"]["stack-edu-python"] = 0.09

    with pytest.raises(ProbeScoringError, match="stack-edu-python"):
        bucket_weights(record, "general-replay",
                       list(HOLDOUT_SOURCES) + ["stack-edu-python"])


def test_scoring_plan_names_sources_no_aggregate_claims(tmp_path):
    record = _record(tmp_path)
    (tmp_path / "holdout" / "finemath-3plus").mkdir()
    (tmp_path / "holdout" / "finemath-3plus" / "manifest.json").write_text(
        json.dumps({"total_tokens": 10}))

    plan = scoring_plan(record, record["holdout_root"])

    assert plan["unscored"] == ["finemath-3plus"]
    assert plan[CODE_CARD]["bucket"] == "code"
    assert plan[GENERAL_CARD]["bucket"] == "general-replay"
    assert plan["caveats"] == record["caveats"]


def test_a_source_in_both_buckets_is_refused(tmp_path):
    record = _record(tmp_path)
    record["buckets"]["general-replay"]["code-python"] = 0.1

    with pytest.raises(ProbeScoringError, match="both"):
        scoring_plan(record, record["holdout_root"])


# ------------------------------------------------------------------- inputs ---

def test_smoke_report_is_not_gate_evidence():
    with pytest.raises(ProbeScoringError, match="250M probes"):
        completed_arms(_probe_report(gate=None, smoke=True))


def test_incomplete_arm_is_refused():
    report = _probe_report()
    report["arms"][1]["summary"]["complete"] = False

    with pytest.raises(ProbeScoringError, match="did not reach its budget"):
        completed_arms(report)


def test_failed_arm_is_refused():
    report = _probe_report()
    report["arms"][0]["error"] = "RuntimeError('cuda oom')"

    with pytest.raises(ProbeScoringError, match="cuda oom"):
        completed_arms(report)


def test_models_put_the_base_first(tmp_path):
    models = models_for(_probe_report(), base_checkpoint="/base/checkpoint.pt",
                        base_out_dir="runs/eval/code-base",
                        eval_root="runs/eval")

    assert [model.name for model in models] == [
        BASE_MODEL, "code-probe-lr0.0005", "code-probe-lr0.001"]
    assert models[0].is_base and not models[1].is_base
    assert models[1].checkpoint == "runs/code-probe-lr0.0005/checkpoint.pt"
    assert models[1].out_dir == "runs/eval/code-probe-lr0.0005"


# ------------------------------------------------------------------ scoring ---

def _bpb_card(path, name, sha, *, bpb, sources, share_covered=1.0,
              per_source=None):
    """A `bpb` card as `run_bpb_eval` writes one, breakdown included.

    `per_source` defaults to the aggregate repeated for every source: every
    card this program writes goes through `summarize_bpb`, which always emits
    one `bpb_<id>` per source, so a fixture without a breakdown would be
    exercising a card shape that cannot be produced.
    """

    values = dict(per_source or {source: bpb for source in sources})
    metrics = {"bpb": bpb, "n_sources": float(len(sources))}
    for source in sources:
        metrics[f"bpb_{source}"] = float(values[source])
        metrics[f"tokens_{source}"] = 1000.0
    write_scorecard(path, Scorecard(
        kind="bpb", name=name,
        provenance=Provenance(
            artifact=ArtifactRef(path="checkpoint.pt", sha256=sha,
                                 kind="checkpoint"),
            tokenizer=ArtifactRef(path="<smollm2-default>", sha256=ZERO_SHA,
                                  kind="tokenizer"),
            seed=20260824, git_sha="abc1234", bpb_mode="full"),
        metrics=metrics,
        item_count=len(sources), created_at="2026-08-27T00:00:00Z",
        details={"sources_requested": list(sources),
                 "bucket_share_covered": share_covered}))


def _execution_card(path, name, sha, *, pass_at_1, syntax_valid, n,
                    max_new_tokens=384, seed=20260824):
    write_scorecard(path, Scorecard(
        kind="code-execution", name=name,
        provenance=Provenance(
            artifact=ArtifactRef(path="checkpoint.pt", sha256=sha,
                                 kind="checkpoint"),
            tokenizer=ArtifactRef(path="<embedded>", sha256=ZERO_SHA,
                                  kind="tokenizer"),
            seed=seed, git_sha="abc1234",
            runtime={"backend": "torch", "max_new_tokens": max_new_tokens}),
        metrics={"pass@1": pass_at_1, "pass@1_plus": pass_at_1,
                 "syntax_valid": syntax_valid, "n": float(n)},
        item_count=n, created_at="2026-08-27T00:00:00Z"))


def _checkpoint(tmp_path, name, payload=b"weights"):
    path = tmp_path / name / "checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_scored_from_is_keyed_on_the_bytes_not_the_file(tmp_path):
    path = tmp_path / "code-bpb.json"
    _bpb_card(path, CODE_CARD, ARM_SHA, bpb=1.0, sources=["code-python"])

    assert scored_from(path, ARM_SHA)
    assert not scored_from(path, ZERO_SHA)
    assert not scored_from(tmp_path / "absent.json", ARM_SHA)


def test_score_bpb_skips_a_model_whose_cards_match(tmp_path):
    checkpoint = _checkpoint(tmp_path, "arm")
    from daedalus.scorecard import sha256_file
    digest = sha256_file(checkpoint)
    out = tmp_path / "cards"
    out.mkdir()
    for name in (CODE_CARD, GENERAL_CARD):
        _bpb_card(out / f"{name}.json", name, digest, bpb=1.0,
                  sources=["code-python"])
    model = ScoredModel(name="arm", checkpoint=str(checkpoint), out_dir=str(out))

    def never(*args, **kwargs):                     # pragma: no cover - asserts
        raise AssertionError("a matching card must not be re-measured")

    result = score_bpb(model, plan={"holdout_root": "unused"},
                       bpb_factory=never, device="cpu")

    assert result["skipped"] == "already-scored"


def test_score_bpb_writes_both_buckets(tmp_path):
    record = _record(tmp_path)
    plan = scoring_plan(record, record["holdout_root"])
    checkpoint = _checkpoint(tmp_path, "arm")
    model = ScoredModel(name="arm", checkpoint=str(checkpoint),
                        out_dir=str(tmp_path / "cards"))
    seen = []

    def factory(path, **kwargs):
        def bpb_fn(source_dir):
            seen.append(source_dir.name)
            return 1.5 if source_dir.name.startswith("code-") else 0.9
        return bpb_fn

    result = score_bpb(model, plan=plan, bpb_factory=factory, device="cpu")

    assert sorted(result["cards"]) == [CODE_CARD, GENERAL_CARD]
    assert sorted(seen) == sorted(HOLDOUT_SOURCES)
    card = json.loads((tmp_path / "cards" / f"{GENERAL_CARD}.json").read_text())
    assert card["metrics"]["bpb"] == pytest.approx(0.9)
    assert card["provenance"]["bpb_mode"] == "full"
    assert card["details"]["bucket_share_covered"] == pytest.approx(0.7792, rel=1e-3)


def test_score_execution_skips_matching_cards_and_runs_missing_ones(tmp_path):
    checkpoint = _checkpoint(tmp_path, "arm")
    from daedalus.scorecard import sha256_file
    digest = sha256_file(checkpoint)
    out = tmp_path / "cards"
    out.mkdir()
    _execution_card(out / "humaneval-plus.json", "humaneval-plus", digest,
                    pass_at_1=0.0, syntax_valid=1.0, n=164)
    model = ScoredModel(name="arm", checkpoint=str(checkpoint), out_dir=str(out))
    launched = []

    def runner(command):
        launched.append(list(command))
        return 0

    result = score_execution(model, runner=runner, device="cpu")

    assert result["execution"]["humaneval-plus"]["skipped"] == "already-scored"
    assert len(launched) == 1
    assert "--dataset" in launched[0] and "mbpp-plus" in launched[0]


def test_score_execution_refuses_a_failed_benchmark(tmp_path):
    checkpoint = _checkpoint(tmp_path, "arm")
    model = ScoredModel(name="arm", checkpoint=str(checkpoint),
                        out_dir=str(tmp_path / "cards"))

    with pytest.raises(ProbeScoringError, match="exited 1"):
        score_execution(model, runner=lambda command: 1, device="cpu")


def test_execution_command_is_the_same_harness_for_every_model(tmp_path):
    base = ScoredModel(BASE_MODEL, "/base/checkpoint.pt", "runs/eval/code-base")
    arm = ScoredModel("arm", "runs/arm/checkpoint.pt", "runs/eval/arm")

    left = execution_command(base, "mbpp-plus", device="cuda")
    right = execution_command(arm, "mbpp-plus", device="cuda")

    differing = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
    # Only the checkpoint and its output directory may differ between models.
    assert [left[index] for index in differing] == ["/base/checkpoint.pt",
                                                    "runs/eval/code-base"]


# --------------------------------------------------------------------- gate ---

def _scored_pair(tmp_path, *, base_code=1.2, base_general=0.9,
                 arm_code=1.1, arm_general=0.9, arm_pass=0.0,
                 arm_syntax=0.2381, arm_n=378, arm_max_new_tokens=384,
                 base_code_by_source=None, arm_code_by_source=None):
    models = []
    for name, code_bpb, general_bpb, sha in (
            (BASE_MODEL, base_code, base_general, ZERO_SHA),
            ("arm", arm_code, arm_general, ARM_SHA)):
        out = tmp_path / name
        out.mkdir(parents=True, exist_ok=True)
        by_source = (base_code_by_source if name == BASE_MODEL
                     else arm_code_by_source)
        _bpb_card(out / f"{CODE_CARD}.json", CODE_CARD, sha, bpb=code_bpb,
                  sources=sorted(by_source) if by_source else ["code-python"],
                  per_source=by_source)
        _bpb_card(out / f"{GENERAL_CARD}.json", GENERAL_CARD, sha,
                  bpb=general_bpb, sources=["dclm-baseline", "fineweb-edu"],
                  share_covered=0.7792)
        is_base = name == BASE_MODEL
        _execution_card(out / "humaneval-plus.json", "humaneval-plus", sha,
                        pass_at_1=0.0 if is_base else arm_pass,
                        syntax_valid=1.0, n=164,
                        max_new_tokens=384 if is_base else arm_max_new_tokens)
        _execution_card(out / "mbpp-plus.json", "mbpp-plus", sha,
                        pass_at_1=0.0079 if is_base else arm_pass,
                        syntax_valid=0.2381 if is_base else arm_syntax,
                        n=378 if is_base else arm_n,
                        max_new_tokens=384 if is_base else arm_max_new_tokens)
        models.append(ScoredModel(name=name, checkpoint=f"{name}.pt",
                                  out_dir=str(out)))
    return models


def test_verdict_continues_and_selects_a_retaining_arm(tmp_path):
    # 8.3% code BPB improvement, general BPB unmoved.
    verdict = build_verdict(_scored_pair(tmp_path))

    assert verdict["gate"]["continue"]
    assert verdict["selected"] == "arm"
    assert verdict["continue"]
    assert verdict["retention"]["arm"]["regression_pct"] == pytest.approx(0.0)
    assert verdict["gate"]["arms"][0]["retention"]["retained"]


def test_a_qualifying_arm_that_loses_general_bpb_is_not_selected(tmp_path):
    verdict = build_verdict(_scored_pair(tmp_path, base_general=0.9,
                                         arm_general=0.9 * 1.02))

    assert verdict["gate"]["continue"]                  # the gate still says yes
    assert verdict["selected"] is None                  # retention says no
    assert verdict["continue"] is False
    assert verdict["retention_rejected"] == ["arm"]
    assert verdict["retention"]["arm"]["regression_pct"] == pytest.approx(2.0)


def test_retention_bound_is_the_preregistered_one():
    base = {"bpb": 1.0, "share_covered": 0.78}
    on_the_bound = retention(base, {"bpb": 1.0 + PROBE_RETENTION_REGRESSION_PCT_MAX / 100,
                                    "share_covered": 0.78})

    assert on_the_bound["threshold_pct"] == PROBE_RETENTION_REGRESSION_PCT_MAX
    assert on_the_bound["retained"]


def test_verdict_stops_when_no_arm_clears_either_criterion(tmp_path):
    verdict = build_verdict(_scored_pair(tmp_path, arm_code=1.2))

    assert not verdict["gate"]["continue"]
    assert verdict["selected"] is None
    assert verdict["continue"] is False


def test_a_differently_generated_arm_is_refused(tmp_path):
    models = _scored_pair(tmp_path, arm_max_new_tokens=512)

    with pytest.raises(ProbeScoringError, match="max_new_tokens"):
        build_verdict(models)


def test_a_shorter_benchmark_run_is_refused(tmp_path):
    """A --limit on one side turns a smaller denominator into an apparent gain.
    `code_gates` refuses it; this asserts the refusal survives the read-back."""
    models = _scored_pair(tmp_path, arm_n=100)

    with pytest.raises(ProbeGateError, match="not comparable"):
        build_verdict(models)


def test_general_bpb_carries_its_coverage_into_the_verdict(tmp_path):
    verdict = build_verdict(_scored_pair(tmp_path))

    assert verdict["general_bpb_base"]["share_covered"] == pytest.approx(0.7792)
    assert verdict["retention"]["arm"]["share_covered"] == pytest.approx(0.7792)


#: The shape phase 8's first two arms actually produced: a large aggregate code
#: BPB gain that is almost entirely TypeScript, on a corpus that is 55% Python.
#: Both execution benchmarks in the same verdict are Python-only.
UNEVEN_BASE = {"code-python": 0.52122,
               "code-javascript-typescript-typescript-all": 0.59634}
UNEVEN_ARM = {"code-python": 0.50899,
              "code-javascript-typescript-typescript-all": 0.21692}


def test_verdict_reports_code_bpb_per_source_beside_the_aggregate(tmp_path):
    """The aggregate alone invites a conclusion the breakdown does not support.

    -23.6% overall reads as "much better at code"; what happened was -63.6% on
    TypeScript and -2.3% on Python, which is the language the corpus is mostly
    made of. The gate is unchanged -- this is the number reported beside it.
    """
    verdict = build_verdict(_scored_pair(
        tmp_path, base_code=0.58714, arm_code=0.44845,
        base_code_by_source=UNEVEN_BASE, arm_code_by_source=UNEVEN_ARM))

    entry = verdict["gate"]["arms"][0]
    assert entry["code_bpb_improvement_pct"] == pytest.approx(23.62, abs=0.01)
    by_source = entry["code_bpb_by_source"]
    assert by_source["code-python"]["improvement_pct"] == pytest.approx(2.35,
                                                                        abs=0.01)
    assert by_source["code-javascript-typescript-typescript-all"][
        "improvement_pct"] == pytest.approx(63.63, abs=0.01)
    assert by_source["code-python"]["base"] == pytest.approx(0.52122)
    assert by_source["code-python"]["measured"] == pytest.approx(0.50899)


def test_verdict_reports_the_bases_own_breakdown(tmp_path):
    """Every arm's number is a difference against the base's, so the base's
    aggregate needs its breakdown for the same reason the arm's does."""
    verdict = build_verdict(_scored_pair(
        tmp_path, base_code=0.58714, arm_code=0.44845,
        base_code_by_source=UNEVEN_BASE, arm_code_by_source=UNEVEN_ARM))

    base_table = verdict["code_bpb_base_by_source"]

    assert sorted(base_table) == sorted(UNEVEN_BASE)
    assert base_table["code-python"]["bpb"] == pytest.approx(0.52122)


def test_an_arm_scored_over_different_code_sources_is_refused(tmp_path):
    """Two aggregates over different holdouts are already not comparable, and
    the breakdown is the only place that would have been visible."""
    models = _scored_pair(tmp_path, base_code_by_source=UNEVEN_BASE,
                          arm_code_by_source={"code-python": 0.5})

    with pytest.raises(ProbeScoringError, match="different holdouts"):
        build_verdict(models)


def test_verdict_without_a_base_is_refused(tmp_path):
    models = [model for model in _scored_pair(tmp_path) if not model.is_base]

    with pytest.raises(ProbeScoringError, match="no base model"):
        build_verdict(models)
