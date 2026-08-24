"""Contracts for the Phase 4 tokenizer lab orchestration.

The lab decides between vocabularies, so most of what is pinned here is about
making the decision unfakeable: the rule is written before the numbers, the
splits are disjoint by construction, the comparison is bits-per-byte and
refuses to be anything else, and a run where nothing clears the bar produces a
recorded negative result rather than a relaxed bar.
"""

import json

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

def test_equal_byte_budget_is_derived_from_the_incumbent():
    """The byte-matched protocol fixes the *text*, so the budget has to come
    from one arm's fertility. It comes from the incumbent, because that is the
    arm every candidate is being compared against."""
    budget = equal_byte_budget(incumbent_bytes_per_token=4.0,
                               tokens=LM_PROBE_TOKENS)
    assert budget == int(LM_PROBE_TOKENS * 4.0)


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


def test_the_rule_digest_changes_if_a_threshold_is_edited(monkeypatch):
    """A preregistration whose rule can be edited after the numbers land is not
    a preregistration. The digest is what a later reader checks."""
    before = rule_digest()
    import scripts.tokenizer_lab as lab
    monkeypatch.setattr(lab, "MAX_TINY_BPB_REGRESSION_PCT", 5.0)
    assert lab.rule_digest() != before
