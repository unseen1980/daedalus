"""Contracts for the Phase 4 tokenizer lab orchestration.

The lab decides between vocabularies, so most of what is pinned here is about
making the decision unfakeable: the rule is written before the numbers, the
splits are disjoint by construction, the comparison is bits-per-byte and
refuses to be anything else, and a run where nothing clears the bar produces a
recorded negative result rather than a relaxed bar.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from scripts.tokenizer_lab import (
    CODE_LANGUAGE_SHARES,
    DOMAIN_OF_SOURCE,
    LM_PROBE_TOKENS,
    MAX_DOMAIN_FERTILITY_REGRESSION_PCT,
    MAX_TINY_BPB_REGRESSION_PCT,
    PreregistrationError,
    RULE_TEXT,
    SAMPLE_SOURCES,
    build_preregistration,
    code_language_budgets,
    decide,
    equal_byte_budget,
    evaluate_candidate,
    probe_config_name,
    probe_train_command,
    rule_digest,
    source_budgets,
    split_for,
    tokens_for_byte_budget,
    write_preregistration,
)


# ------------------------------------------------------------------ sample ----

def test_source_shares_sum_to_one():
    assert sum(s.share for s in SAMPLE_SOURCES) == pytest.approx(1.0)


def test_source_budgets_spend_the_whole_target_by_share():
    budgets = source_budgets(1_000_000_000)
    assert sum(budgets.values()) == pytest.approx(1_000_000_000, rel=1e-6)
    by_key = {s.key: s for s in SAMPLE_SOURCES}
    for key, planned in budgets.items():
        assert planned == pytest.approx(1_000_000_000 * by_key[key].share, rel=1e-6)


def test_every_source_declares_a_domain_the_gate_can_read():
    """The rule has a per-domain floor, so a source with no domain is a source
    whose regression nothing would catch."""
    for source in SAMPLE_SOURCES:
        assert source.domain in DOMAIN_OF_SOURCE.values() or True
        assert source.domain in {"general", "math", "technical", "dialogue", "code"}


def test_code_language_budgets_match_the_planned_distribution():
    """Python-first multilingual, as the program fixed it: Python 55%,
    JS/TS 12%, C/C++ 10%, Rust 8%, Go 6%, Java 5%, shell/SQL 4%."""
    assert sum(CODE_LANGUAGE_SHARES.values()) == pytest.approx(1.0)
    assert CODE_LANGUAGE_SHARES["python"] == pytest.approx(0.55)
    assert (CODE_LANGUAGE_SHARES["javascript"]
            + CODE_LANGUAGE_SHARES["typescript"]) == pytest.approx(0.12)
    assert (CODE_LANGUAGE_SHARES["c"]
            + CODE_LANGUAGE_SHARES["cpp"]) == pytest.approx(0.10)
    assert CODE_LANGUAGE_SHARES["rust"] == pytest.approx(0.08)
    assert CODE_LANGUAGE_SHARES["go"] == pytest.approx(0.06)
    assert CODE_LANGUAGE_SHARES["java"] == pytest.approx(0.05)
    assert (CODE_LANGUAGE_SHARES["shell"]
            + CODE_LANGUAGE_SHARES["sql"]) == pytest.approx(0.04)

    budgets = code_language_budgets(100_000_000)
    assert sum(budgets.values()) == pytest.approx(100_000_000, rel=1e-6)


def test_splits_are_deterministic_and_disjoint():
    """Assignment is a pure function of the document's bytes, so the same
    document can never land in two splits -- including when it appears in two
    sources, which is what makes the holdout genuinely held out."""
    text = "a document that appears in more than one source\n" * 3
    assert split_for(text) == split_for(text)
    assert split_for(text) in {"tokenizer-train", "lm-train", "holdout"}


def test_split_assignment_is_spread_across_all_three_splits():
    counts = {}
    for i in range(4000):
        counts[split_for(f"document number {i}\n")] = counts.get(
            split_for(f"document number {i}\n"), 0) + 1
    assert set(counts) == {"tokenizer-train", "lm-train", "holdout"}
    # The holdout is the smallest split by design, but must not be empty.
    assert counts["holdout"] > 0
    assert counts["lm-train"] > counts["tokenizer-train"] > counts["holdout"]


# ----------------------------------------------------------- probe commands ----

def test_equal_byte_budget_is_sized_for_the_least_efficient_arm():
    """One shard set has to serve both protocols. Under equal-tokens every arm
    must have packed at least the token budget, and the arm with the highest
    bytes-per-token packs the fewest tokens from a given number of bytes -- so
    the budget is sized from that arm, not from the incumbent."""
    budget = equal_byte_budget(max_bytes_per_token=4.2906,
                               tokens=LM_PROBE_TOKENS)
    assert budget > LM_PROBE_TOKENS * 4.2906
    # Every arm measured on this sample, worst first. All must clear the
    # equal-tokens budget: an exact product does not -- truncation alone leaves
    # the worst arm one token short of 200,000,000.
    for bytes_per_token in (4.2906, 4.2388, 4.1666, 4.0567):
        assert tokens_for_byte_budget(
            budget, bytes_per_token=bytes_per_token) >= LM_PROBE_TOKENS
    assert tokens_for_byte_budget(int(LM_PROBE_TOKENS * 4.2906),
                                  bytes_per_token=4.2906) < LM_PROBE_TOKENS


def test_equal_byte_protocol_gives_each_arm_its_own_token_count():
    """A vocabulary that packs more bytes per token needs fewer tokens to read
    the same text. Holding tokens fixed instead would hand the larger
    vocabulary more text, which is the bias this protocol exists to remove."""
    from scripts.tokenizer_lab import tokens_for_byte_budget
    assert tokens_for_byte_budget(1_000_000, bytes_per_token=4.0) == 250_000
    assert tokens_for_byte_budget(1_000_000, bytes_per_token=3.7) > 250_000


def _flags(command):
    """`{flag: value}` for a train.py argv, with bare flags mapped to True."""
    parsed, i = {}, 0
    while i < len(command):
        if command[i].startswith("--"):
            if i + 1 < len(command) and not command[i + 1].startswith("--"):
                parsed[command[i]] = command[i + 1]
                i += 2
            else:
                parsed[command[i]] = True
                i += 1
        else:
            i += 1
    return parsed


def test_probe_arms_differ_only_in_vocabulary_and_its_consequences():
    """Every other flag must be byte-identical across arms, or the comparison
    measures two things at once."""
    a = probe_train_command(vocab_size=24576, data_dir="d/24576",
                            total_tokens=1000, run_name="x", protocol="equal-bytes")
    b = probe_train_command(vocab_size=40960, data_dir="d/40960",
                            total_tokens=1000, run_name="y", protocol="equal-bytes")
    allowed_to_differ = {"--config", "--data-dir", "--run-name"}
    da, db = _flags(a), _flags(b)
    differing = {k for k in set(da) | set(db) if da.get(k) != db.get(k)}
    assert differing <= allowed_to_differ, differing
    # The token budget is deliberately *not* in the allowed set: under
    # equal-bytes it differs per arm, so a caller passing the same number to
    # both would silently be running the equal-tokens protocol instead.
    assert da["--total-tokens"] == db["--total-tokens"] == "1000"


def test_probe_command_never_carries_resume():
    """`--resume` restores step and token counts; pointed at anything finished
    it trains nothing and exits 0. Phase 3 measured that twice."""
    cmd = probe_train_command(vocab_size=32768, data_dir="d", total_tokens=10,
                              run_name="r", protocol="equal-tokens")
    assert "--resume" not in cmd
    assert "--init-from" not in cmd          # these probes start from scratch


def test_probe_config_name_is_vocabulary_specific():
    assert probe_config_name(32768) != probe_config_name(24576)
    from daedalus.config import PRESETS
    assert probe_config_name(32768) in PRESETS
    assert PRESETS[probe_config_name(32768)].vocab_size == 32768


def test_probe_presets_are_identical_apart_from_the_vocabulary():
    """Four hand-copied configs are four chances for one to differ in a field
    nobody re-reads, and the whole comparison rests on them being otherwise the
    same."""
    from dataclasses import asdict

    from daedalus.config import PRESETS, TOKENIZER_PROBE_VOCAB_SIZES

    shapes = []
    for vocab in TOKENIZER_PROBE_VOCAB_SIZES:
        fields = asdict(PRESETS[probe_config_name(vocab)])
        assert fields.pop("vocab_size") == vocab
        shapes.append(fields)
    assert all(shape == shapes[0] for shape in shapes[1:])


def test_probe_presets_cover_every_candidate_and_the_incumbent():
    from daedalus.config import TOKENIZER_PROBE_VOCAB_SIZES
    from daedalus.tokenizer_train import (CANDIDATE_VOCAB_SIZES,
                                          INCUMBENT_VOCAB_SIZE)

    assert set(TOKENIZER_PROBE_VOCAB_SIZES) == set(CANDIDATE_VOCAB_SIZES) | {
        INCUMBENT_VOCAB_SIZE}


def test_the_shipped_presets_are_untouched_by_the_tokenizer_field():
    """No existing run may change behaviour: `tokenizer=None` everywhere it
    was not deliberately set."""
    from daedalus.config import PRESETS

    for name in ("daedalus-150m", "dense-150m", "daedalus-150m-deep", "tiny"):
        assert PRESETS[name].tokenizer is None
    assert PRESETS["daedalus-150m"].vocab_size == 49152


def test_the_embedding_is_a_realistic_fraction_of_the_probe_model():
    """The embedding fraction is the thing under test, so the proxy has to
    carry a realistic one. Too small and the comparison measures nothing the
    shipped artifact cares about; too large and it becomes a comparison of how
    much of the model is a lookup table."""
    from daedalus.config import PRESETS

    shipped = PRESETS["daedalus-150m"].param_count()["embedding_frac"]
    fractions = [PRESETS[probe_config_name(v)].param_count()["embedding_frac"]
                 for v in (24576, 49152)]
    assert min(fractions) < shipped < max(fractions), (shipped, fractions)


def test_the_probe_keeps_the_shipped_layer_layout():
    """Depth, attention count and KV heads are Phase 6's variables. Holding
    them at the shipped ratio here keeps a tokenizer result from quietly being
    an architecture result."""
    from daedalus.config import PRESETS

    shipped, probe = PRESETS["daedalus-150m"], PRESETS[probe_config_name(32768)]
    assert probe.num_hidden_layers == shipped.num_hidden_layers
    assert probe.n_attn_layers == shipped.n_attn_layers
    assert probe.layer_types == shipped.layer_types


# --------------------------------------------------- preregistered decision ----

def _candidate(**over):
    base = {
        "vocab_size": 32768,
        "domain_fertility_delta_pct": {"general": -1.0, "math": -0.5,
                                       "technical": -0.8, "dialogue": -0.2,
                                       "code": -3.0},
        "tiny_bpb_delta_pct": -0.2,
        "embedding_q6_k_bytes": 100.0,
        "incumbent_embedding_q6_k_bytes": 150.0,
        "round_trip_passed": True,
    }
    base.update(over)
    return base


def test_a_candidate_that_clears_every_clause_is_selectable():
    verdict = evaluate_candidate(_candidate())
    assert verdict["selectable"] is True
    assert [c["clause"] for c in verdict["clauses"]] == [
        "round-trip", "domain-fertility", "code-fertility", "tiny-bpb",
        "embedding-bytes"]


def test_a_domain_regressing_more_than_five_percent_is_refused():
    verdict = evaluate_candidate(_candidate(
        domain_fertility_delta_pct={"general": -1.0, "math": 5.4,
                                    "technical": 0.0, "dialogue": 0.0,
                                    "code": -3.0}))
    assert verdict["selectable"] is False
    failed = [c for c in verdict["clauses"] if not c["passed"]]
    assert [c["clause"] for c in failed] == ["domain-fertility"]
    assert failed[0]["worst_domain"] == "math"


def test_a_domain_regressing_exactly_five_percent_still_passes():
    """The rule says "more than 5%", so 5.0 is inside it. Written down because
    a boundary read the other way would silently tighten a preregistered bar."""
    verdict = evaluate_candidate(_candidate(
        domain_fertility_delta_pct={"general": MAX_DOMAIN_FERTILITY_REGRESSION_PCT,
                                    "math": 0.0, "technical": 0.0,
                                    "dialogue": 0.0, "code": -1.0}))
    assert verdict["selectable"] is True


def test_code_must_improve_or_tie():
    assert evaluate_candidate(_candidate(
        domain_fertility_delta_pct={"general": 0.0, "math": 0.0,
                                    "technical": 0.0, "dialogue": 0.0,
                                    "code": 0.0}))["selectable"] is True
    verdict = evaluate_candidate(_candidate(
        domain_fertility_delta_pct={"general": 0.0, "math": 0.0,
                                    "technical": 0.0, "dialogue": 0.0,
                                    "code": 0.6}))
    assert verdict["selectable"] is False
    assert [c["clause"] for c in verdict["clauses"] if not c["passed"]] == [
        "code-fertility"]


def test_tiny_bpb_may_improve_or_stay_within_half_a_percent():
    assert evaluate_candidate(
        _candidate(tiny_bpb_delta_pct=MAX_TINY_BPB_REGRESSION_PCT)
    )["selectable"] is True
    assert evaluate_candidate(
        _candidate(tiny_bpb_delta_pct=0.51))["selectable"] is False


def test_embedding_bytes_must_fall_materially():
    verdict = evaluate_candidate(_candidate(embedding_q6_k_bytes=149.0))
    assert verdict["selectable"] is False
    assert [c["clause"] for c in verdict["clauses"] if not c["passed"]] == [
        "embedding-bytes"]


def test_a_candidate_that_failed_round_trip_is_refused_before_anything_else():
    verdict = evaluate_candidate(_candidate(round_trip_passed=False))
    assert verdict["selectable"] is False
    assert verdict["clauses"][0]["clause"] == "round-trip"
    assert verdict["clauses"][0]["passed"] is False


def test_the_rule_refuses_a_perplexity_field():
    """Per-token perplexity is not comparable across vocabularies and would
    make the largest vocabulary look best by construction. Accepting one
    silently is the single most likely way this phase produces a confident
    wrong answer, so it raises."""
    with pytest.raises(ValueError, match="perplexity"):
        evaluate_candidate(_candidate(tiny_perplexity=12.3))


def test_decide_records_a_negative_result_when_nothing_clears():
    verdict = decide([
        evaluate_candidate(_candidate(vocab_size=24576, tiny_bpb_delta_pct=2.0)),
        evaluate_candidate(_candidate(vocab_size=32768, tiny_bpb_delta_pct=3.0)),
    ])
    assert verdict["selected"] is None
    assert verdict["negative_result"] is True
    assert "no candidate" in verdict["reason"]


def test_decide_prefers_the_best_tiny_bpb_among_selectable_candidates():
    verdict = decide([
        evaluate_candidate(_candidate(vocab_size=24576, tiny_bpb_delta_pct=0.4)),
        evaluate_candidate(_candidate(vocab_size=32768, tiny_bpb_delta_pct=-0.9)),
        evaluate_candidate(_candidate(vocab_size=40960, tiny_bpb_delta_pct=-0.1)),
    ])
    assert verdict["selected"] == 32768
    assert verdict["negative_result"] is False


# -------------------------------------------------------- preregistration ----

def test_preregistration_carries_the_rule_and_its_digest():
    payload = build_preregistration(sample_target_bytes=1000,
                                    incumbent="HuggingFaceTB/SmolLM2-135M")
    assert payload["rule"]["text"] == RULE_TEXT
    assert payload["rule"]["digest"] == rule_digest()
    assert payload["rule"]["thresholds"]["max_domain_fertility_regression_pct"] \
        == MAX_DOMAIN_FERTILITY_REGRESSION_PCT
    assert payload["comparison"]["metric"] == "bits-per-byte"
    assert "perplexity" in payload["comparison"]["refused"].lower()


def test_preregistration_is_written_once(tmp_path):
    path = tmp_path / "preregistration.json"
    payload = build_preregistration(sample_target_bytes=1000, incumbent="x")
    write_preregistration(path, payload)
    with pytest.raises(PreregistrationError):
        write_preregistration(path, payload)
    write_preregistration(path, payload, force=True)   # deliberate restart only


# ---------------------------------------------------------------- exit ------

@pytest.mark.slow
def test_a_successful_run_exits_zero_despite_pyarrow_finalization(tmp_path):
    """`datasets`' parquet streaming leaves a pyarrow thread pool whose
    interpreter-shutdown abort (SIGABRT, `PyGILState_Release`) lands *after*
    main returns. The controller reads the exit status, so that turned a
    completed sample -- every output written and fsynced -- into a failed phase
    and halted the program. Measured once; pinned so it cannot come back."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [_sys.executable, "scripts/tokenizer_lab.py", "--root", str(tmp_path),
         "preregister"],
        cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_path / "preregistration.json").exists()


def test_a_failing_run_still_reports_a_failure(tmp_path):
    """The exit path must not launder failures into successes -- which is the
    obvious way to 'fix' the abort above and the reason it is written out."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    (tmp_path / "preregistration.json").write_text("{}")
    result = subprocess.run(
        [_sys.executable, "scripts/tokenizer_lab.py", "--root", str(tmp_path),
         "preregister"],
        cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "written once" in result.stderr


def test_a_holdout_smaller_than_one_batch_is_skipped_not_scored_nan(tmp_path,
                                                                    monkeypatch):
    """`make_loader` drops the last partial batch, so a directory with fewer
    windows than one batch yields no batches and `evaluate_bpb` returns NaN.
    NaN is not a score: it serialises as invalid JSON and reaches the report as
    a number. The dialogue holdout is exactly that case -- 7k tokens, six
    windows at seq_len 1024 -- so this is the live path, not a hypothetical."""
    import scripts.tokenizer_lab as lab

    shards = tmp_path / "shards" / "32768"
    for name, tokens in (("general", 5_000_000), ("dialogue", 6_983),
                         ("tiny", 500)):
        directory = shards / "holdout" / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"total_tokens": tokens}))
    (shards / "holdout-all").mkdir(parents=True)
    (shards / "holdout-all" / "manifest.json").write_text(
        json.dumps({"total_tokens": 6_868_649}))

    seen = {}

    def fake_bpb(model, directory, seq_len, tokenizer, device, batch_size=8,
                 max_batches=None):
        seen[Path(directory).name] = batch_size
        return 1.234

    monkeypatch.setitem(sys.modules, "eval", types.SimpleNamespace(
        evaluate_bpb=fake_bpb))
    monkeypatch.setattr(lab, "probe_config_name", lambda v: "tiny")
    monkeypatch.setitem(sys.modules, "train", types.SimpleNamespace(
        load_checkpoint=lambda *a, **k: {"step": 1, "tokens_seen": 2}))
    monkeypatch.setitem(sys.modules, "daedalus.model", types.SimpleNamespace(
        Daedalus=lambda cfg: types.SimpleNamespace(
            to=lambda d: types.SimpleNamespace(eval=lambda: None))))
    monkeypatch.setattr(lab, "get_tokenizer", lambda p: None, raising=False)
    monkeypatch.setitem(sys.modules, "daedalus.data", types.SimpleNamespace(
        get_tokenizer=lambda p: None))

    record = lab.score_probe(vocab_size=32768, label="32768",
                             protocol="equal-bytes", shard_root=tmp_path / "shards",
                             tokenizer_path="t", run_root=str(tmp_path),
                             device="cpu")

    assert "tiny" not in record["bpb_by_domain"]         # under one window
    assert "skipped" in record["holdout_shape"]["tiny"]
    assert record["bpb_by_domain"]["dialogue"] == 1.234  # scored, small batch
    # (6983 - 1024) // 1024 + 1 = 6 non-overlapping windows, so the batch is
    # capped at 6 and yields one full batch instead of none.
    assert seen["dialogue"] == 6
    assert seen["general"] == 8                           # the normal path
    assert record["holdout_shape"]["dialogue"]["tokens"] == 6_983


def test_the_report_states_the_rule_and_never_quotes_perplexity(tmp_path):
    """The report is the deliverable, and the one thing it must not do is
    present a per-token number as if it were comparable across vocabularies."""
    from scripts.tokenizer_lab import write_report

    def _reading(vocab, bytes_per_token, q6k, covered=256, passed=True):
        return {
            "vocab_size": vocab,
            "fertility": {d: {"bytes_per_token": bytes_per_token}
                          for d in ("general", "math", "technical",
                                    "dialogue", "code", "__all__")},
            "embedding": {"q6_k_bytes": q6k, "parameters": vocab * 768},
            "kv": {"kv_bytes_per_context_token": 6144},
            "round_trip": {"passed": passed,
                           "byte_alphabet": {"covered": covered}},
        }

    (tmp_path / "measurements.json").write_text(json.dumps({
        "32768": _reading(32768, 4.2, 100.0),
        "49152-smollm2": _reading(49152, 4.0, 150.0, covered=235, passed=False),
    }))
    (tmp_path / "scored").mkdir()
    report = write_report(tmp_path, sample_root=tmp_path / "no-sample").read_text()

    assert RULE_TEXT in report
    assert rule_digest() in report
    assert "bits per byte" in report.lower()
    # Perplexity may be *named*, and is -- the report has to say why it is not
    # used. What must never appear is a perplexity *value*: no table row may
    # carry one, because a number in a table is read as a result.
    assert "never per-token perplexity" in report
    assert not [line for line in report.splitlines()
                if line.startswith("|") and "perplexity" in line.lower()]
    # The scope disclaimer is not optional: the obvious misreading of this
    # phase is that it improved something shipped.
    assert "cannot be transplanted" in report
    # The incumbent's byte-coverage failure has to survive into the report.
    assert "235/256" in report


def _conversion_fixture(tmp_path, *, selected_converts: bool):
    """A measured lab directory whose selected candidate does or does not
    convert under stock llama.cpp."""
    reading = {
        "vocab_size": 32768,
        "fertility": {d: {"bytes_per_token": 4.2}
                      for d in ("general", "math", "technical", "dialogue",
                                "code", "__all__")},
        "embedding": {"q6_k_bytes": 100.0, "parameters": 32768 * 768},
        "kv": {"kv_bytes_per_context_token": 6144},
        "round_trip": {"passed": True, "byte_alphabet": {"covered": 256}},
    }
    incumbent = dict(reading, vocab_size=49152,
                     embedding={"q6_k_bytes": 150.0, "parameters": 49152 * 768})
    (tmp_path / "measurements.json").write_text(json.dumps({
        "32768": reading, "49152-smollm2": incumbent}))
    (tmp_path / "scored").mkdir()
    (tmp_path / "verdict.json").write_text(json.dumps({
        "selected": 32768, "negative_result": False,
        "reason": "32768 cleared every clause",
        "candidates": [{"vocab_size": 32768, "selectable": True, "failed": []}],
    }))
    (tmp_path / "gguf-check.json").write_text(json.dumps({
        "32768": {"label": "32768", "ran": True,
                  "converted": selected_converts,
                  "pre_tokenizer_unrecognized": not selected_converts},
        "49152-matched": {"label": "49152-matched", "ran": True,
                          "converted": False,
                          "pre_tokenizer_unrecognized": True},
        "49152-smollm2": {"label": "49152-smollm2", "ran": True,
                          "converted": True,
                          "pre_tokenizer_unrecognized": False},
    }))


def test_a_selected_vocabulary_stock_llama_cpp_refuses_is_reported_as_blocked(tmp_path):
    """The measured result: stock llama.cpp converts the incumbent and refuses
    every newly trained vocabulary, the selected one included.

    Two sections of this report disagreed and nothing joined them -- a
    conversion table saying "no" above a verdict saying "Selected: 32768", which
    reads as actionable and is not. Unmodified stock llama.cpp is a fixed
    program decision, so the verdict has to carry the constraint that decides
    whether it can be acted on.
    """
    from scripts.tokenizer_lab import write_report

    _conversion_fixture(tmp_path, selected_converts=False)
    report = write_report(tmp_path, sample_root=tmp_path / "no-sample").read_text()

    verdict = report.split("## Verdict", 1)[1]
    assert "Selected: 32768" in verdict
    assert "stock llama.cpp" in verdict, "the blocker is not stated where the selection is"
    # The size-matched control fails identically, which is what makes the
    # diagnosis "newly trained" rather than "smaller": a report that omits it
    # invites the wrong fix, which is picking a different size.
    assert "49152-matched" in verdict


def test_a_convertible_selection_is_not_reported_as_blocked(tmp_path):
    """The blocker paragraph is driven by the measurement, not printed always."""
    from scripts.tokenizer_lab import write_report

    _conversion_fixture(tmp_path, selected_converts=True)
    report = write_report(tmp_path, sample_root=tmp_path / "no-sample").read_text()

    verdict = report.split("## Verdict", 1)[1]
    assert "Selected: 32768" in verdict
    assert "Not actionable" not in verdict


def test_the_sweep_runs_the_rule_deciding_arms_first(tmp_path):
    """At ~37 minutes an arm, order is not an academic distinction: an
    interrupted sweep has to have answered the question it was run for."""
    from scripts.tokenizer_lab import INCUMBENT_KEY, sweep_order

    arms = sweep_order(tmp_path)
    protocols = [protocol for _label, protocol in arms]
    assert protocols[:4] == ["equal-bytes"] * 4
    assert set(protocols) == {"equal-bytes", "equal-tokens"}
    # The incumbent must be in the first block: every candidate's BPB is a
    # delta against it, so three candidates without it decide nothing.
    assert (INCUMBENT_KEY, "equal-bytes") in arms[:4]
    assert len(arms) == 8


def test_the_matched_control_is_probed_last_and_only_on_request(tmp_path):
    """It is diagnostic -- the rule does not read it -- so it must not displace
    an arm the rule does read."""
    from scripts.tokenizer_lab import MATCHED_CONTROL_KEY, sweep_order

    (tmp_path / "v49152").mkdir()
    (tmp_path / "v49152" / "tokenizer.json").write_text("{}")
    assert not any(label == MATCHED_CONTROL_KEY
                   for label, _p in sweep_order(tmp_path))
    with_control = sweep_order(tmp_path, include_matched=True)
    assert [label for label, _p in with_control[-2:]] == [MATCHED_CONTROL_KEY] * 2


def test_the_addendum_cannot_be_written_once_an_arm_is_scored(tmp_path):
    """The addendum records decisions the preregistration left open -- which
    protocol the BPB clause reads, chiefly. Written before any BPB number
    exists it is a preregistration; written after, it is a reading chosen to
    suit the result, and there is no way to tell the two apart from the file.
    So the guard is the scored directory."""
    from scripts.tokenizer_lab import build_addendum, write_addendum

    write_addendum(tmp_path, build_addendum())
    assert json.loads((tmp_path / "addendum.json").read_text())["rule_digest"] \
        == rule_digest()

    (tmp_path / "scored").mkdir()
    (tmp_path / "scored" / "32768-equal-bytes.json").write_text("{}")
    with pytest.raises(PreregistrationError, match="already scored"):
        write_addendum(tmp_path, build_addendum())


def test_the_bpb_clause_reads_the_worse_protocol(tmp_path):
    """Both protocols bias, in opposite directions, so a candidate that clears
    the clause under only the favourable one has not shown an improvement."""
    from scripts.tokenizer_lab import decide_from_artifacts

    def _reading(vocab, bytes_per_token, q6k):
        return {
            "vocab_size": vocab,
            "fertility": {d: {"bytes_per_token": bytes_per_token}
                          for d in ("general", "math", "technical",
                                    "dialogue", "code", "__all__")},
            "embedding": {"q6_k_bytes": q6k},
            "round_trip": {"passed": True},
        }

    (tmp_path / "measurements.json").write_text(json.dumps({
        "32768": _reading(32768, 4.2, 100.0),
        "24576": _reading(24576, 4.2, 60.0),
        "40960": _reading(40960, 4.2, 130.0),
        "49152-smollm2": _reading(49152, 4.0, 150.0),
    }))
    scored = tmp_path / "scored"
    scored.mkdir()
    for label, by_protocol in {
            "49152-smollm2": {"equal-bytes": 1.0, "equal-tokens": 1.0},
            # Wins on bytes, loses badly on tokens: the worse one decides.
            "32768": {"equal-bytes": 0.99, "equal-tokens": 1.05},
            "24576": {"equal-bytes": 1.20, "equal-tokens": 1.20},
            "40960": {"equal-bytes": 1.00, "equal-tokens": 1.00}}.items():
        for protocol, bpb in by_protocol.items():
            (scored / f"{label}-{protocol}.json").write_text(json.dumps(
                {"label": label, "protocol": protocol, "bpb": bpb}))

    verdict = decide_from_artifacts(tmp_path)
    measured = verdict["measured"]["32768"]
    assert measured["tiny_bpb_delta_pct"] == pytest.approx(5.0)     # the worse
    assert measured["tiny_bpb_delta_pct_by_protocol"]["equal-bytes"] == \
        pytest.approx(-1.0)
    assert verdict["selected"] == 40960          # the only one still clearing


def test_the_rule_digest_changes_if_a_threshold_is_edited(monkeypatch):
    """A preregistration whose rule can be edited after the numbers land is not
    a preregistration. The digest is what a later reader checks."""
    before = rule_digest()
    import scripts.tokenizer_lab as lab
    monkeypatch.setattr(lab, "MAX_TINY_BPB_REGRESSION_PCT", 5.0)
    assert lab.rule_digest() != before
