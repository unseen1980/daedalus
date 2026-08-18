"""Tests for `scripts/source_headroom.py`.

The two cases that matter are real ones, so they are pinned with real numbers:
`stack-edu-python`, which genuinely ran out of documents on 2026-08-10 and must
come back EXHAUSTED, and `finepdfs-edu`, which the `hero` 60B mixture gate
depends on and must come back SAFE. A tool that got either backwards would be
worse than not having it -- an AT_RISK it cannot substantiate stalls a launch,
and a SAFE over an exhausted source is how the 139M shortfall surprised us in
the first place.

No test touches the network: `assess` and `furthest_shard_idx` are pure, and the
file lists are injected.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "source_headroom", os.path.join(_ROOT, "scripts", "source_headroom.py"))
sh = importlib.util.module_from_spec(_spec)
# Registered before exec because the module defines a @dataclass, and
# dataclasses resolves annotations via sys.modules[cls.__module__].
sys.modules[_spec.name] = sh
_spec.loader.exec_module(sh)


# ---------------------------------------------------------------- real shapes

# Copied from data/manifest.json at 2026-08-10 21:36Z, trimmed to the keys the
# reduction looks at. Both nest a `previous_state` beside an inner
# `examples_iterable`, and which one is ahead differs -- that is the whole
# reason the reduction is a max rather than a lookup.
STACK_STATE = {
    "epoch": 0,
    "hf_state": {"examples_iterable": {
        "examples_iterable": {"shard_idx": 10, "shard_example_idx": 0,
                              "type": "ArrowExamplesIterable"},
        "previous_state": {"shard_idx": 9, "shard_example_idx": 58000,
                           "type": "ArrowExamplesIterable"},
        "batch_idx": 641000, "num_chunks_since_previous_state": 10}},
}

FINEPDFS_STATE = {
    "epoch": 0,
    "hf_state": {"examples_iterable": {
        "examples_iterable": {"shard_idx": 0, "shard_example_idx": 89000,
                              "type": "ArrowExamplesIterable"},
        "previous_state": {"shard_idx": 0, "shard_example_idx": 210000,
                           "type": "ArrowExamplesIterable"},
        "batch_idx": 210890}},
}

# 10 files, ~1.79 GB total (measured from the Hub).
STACK_SIZES = [180_000_000, 181_000_000, 180_000_000, 181_000_000, 181_000_000,
               180_000_000, 180_000_000, 181_000_000, 181_000_000, 166_000_000]
# 100 files, ~298.67 GB total (measured from the Hub).
FINEPDFS_SIZES = [2_964_000_000] + [2_986_000_000] * 99


class TestFurthestShardIdx:
    def test_takes_the_max_across_nested_positions(self):
        # previous_state is behind the inner iterable here.
        assert sh.furthest_shard_idx(STACK_STATE) == 10

    def test_reads_zero_rather_than_none_when_still_on_the_first_file(self):
        # 0 and None mean very different things: "on file 0" vs "never ran".
        assert sh.furthest_shard_idx(FINEPDFS_STATE) == 0

    def test_none_when_there_is_no_position_at_all(self):
        assert sh.furthest_shard_idx(None) is None
        assert sh.furthest_shard_idx({}) is None
        assert sh.furthest_shard_idx({"epoch": 0}) is None

    def test_finds_a_position_nested_under_lists(self):
        assert sh.furthest_shard_idx({"a": [{"b": {"shard_idx": 7}}]}) == 7

    def test_ignores_booleans_which_are_ints_in_python(self):
        # `shard_idx: True` would otherwise reduce to 1 and fake progress.
        assert sh.furthest_shard_idx({"shard_idx": True}) is None


class TestTheRealExhaustedSource:
    """stack-edu-python: 1.211B tokens of a 1.35B budget, no documents left."""

    def _assess(self, budget=1_350_000_000):
        return sh.assess("stack-edu-python", 1_210_964_651, budget,
                         STACK_SIZES, sh.furthest_shard_idx(STACK_STATE))

    def test_is_exhausted_not_at_risk(self):
        r = self._assess()
        assert r.verdict == sh.EXHAUSTED
        assert "permanent" in r.note

    def test_reports_the_real_139m_shortfall(self):
        assert self._assess().tokens_needed == 1_350_000_000 - 1_210_964_651

    def test_has_no_bytes_left_to_offer(self):
        r = self._assess()
        assert r.files_consumed == r.files_total == 10
        assert r.bytes_remaining == 0

    def test_the_shortfall_was_predictable_from_a_prefix_of_the_source(self):
        # The point of the tool. Halfway through -- 5 of 10 files, tokens pro
        # rata -- the density already projects the whole 1.79 GB source to less
        # than its 1.35B budget, so the shortfall was knowable hours before the
        # stream actually ran dry, and AT_RISK fires while there is still time.
        half = sh.assess("stack-edu-python", 605_482_325, 1_350_000_000,
                         STACK_SIZES, shard_idx=5)
        assert half.verdict == sh.AT_RISK
        projected_whole_source = half.density_tok_per_byte * half.bytes_total
        assert projected_whole_source < 1_350_000_000

    def test_met_wins_over_exhausted_when_the_budget_was_reached(self):
        # Running out of documents exactly on budget is success, not failure.
        assert self._assess(budget=1_200_000_000).verdict == sh.MET


class TestTheSourceTheHeroGateDependsOn:
    """finepdfs-edu must reach 1,124,340,092 or `hero` refuses at 60B."""

    GATE_THRESHOLD = 1_124_340_092
    LIVE_TOKENS = 906_808_995

    def _assess(self, budget):
        return sh.assess("finepdfs-edu", self.LIVE_TOKENS, budget,
                         FINEPDFS_SIZES, sh.furthest_shard_idx(FINEPDFS_STATE))

    def test_is_safe_against_the_mixture_gate_threshold(self):
        assert self._assess(self.GATE_THRESHOLD).verdict == sh.SAFE

    def test_is_safe_against_its_full_budget(self):
        assert self._assess(1_200_000_000).verdict == sh.SAFE

    def test_the_headroom_is_orders_of_magnitude_not_marginal(self):
        # This is why the `go 58B` fallback insures against ~nothing.
        r = self._assess(1_200_000_000)
        assert r.cover_ratio > 100

    def test_only_the_first_file_is_credited_as_read(self):
        r = self._assess(1_200_000_000)
        assert r.files_consumed == 0
        assert r.bytes_remaining == sum(FINEPDFS_SIZES[1:])


class TestTheBoundsLeanTheSafeWay:
    def test_density_is_a_lower_bound_because_the_current_file_counts_whole(self):
        # 500 tokens came from part of file 0, but all 1000 bytes of file 0 are
        # charged against them, so the density understates the truth.
        r = sh.assess("s", 500, 10_000, [1000, 1000], shard_idx=0)
        assert r.density_tok_per_byte == 0.5
        assert r.bytes_remaining == 1000        # file 0's tail is not counted
        assert r.tokens_remaining_lower == 500

    def test_a_source_scraping_past_budget_does_not_read_as_comfortable(self):
        # 200 tokens from the first 200 bytes -> density 1.0; one untouched
        # 105-byte file left. Needs 100, can reach 105: over the line but
        # inside the 10% margin, so AT_RISK rather than SAFE.
        r = sh.assess("s", 200, 300, [100, 100, 105], shard_idx=1,
                      safety_margin=1.10)
        assert r.density_tok_per_byte == 1.0
        assert r.tokens_remaining_lower == 105
        assert r.tokens_needed == 100
        assert r.verdict == sh.AT_RISK

    def test_clear_headroom_is_safe(self):
        r = sh.assess("s", 100, 200, [100, 1000], shard_idx=0)
        assert r.verdict == sh.SAFE

    def test_at_risk_when_what_is_left_cannot_cover_the_need(self):
        r = sh.assess("s", 100, 10_000, [100, 100], shard_idx=0)
        assert r.verdict == sh.AT_RISK
        assert r.tokens_remaining_lower < r.tokens_needed


class TestDegenerateInputs:
    def test_unknown_rather_than_safe_when_no_tokens_have_been_produced(self):
        # Zero tokens gives no density; guessing one would be inventing data.
        r = sh.assess("s", 0, 1000, [100, 100], shard_idx=None)
        assert r.verdict == sh.UNKNOWN
        assert r.tokens_remaining_lower is None

    def test_a_shard_idx_past_the_file_list_is_clamped(self):
        r = sh.assess("s", 10, 1000, [100], shard_idx=99)
        assert r.files_consumed == 1
        assert r.bytes_remaining == 0

    def test_re_read_sources_are_not_alarmed_on(self):
        # everyday-conversations: ~2.2k rows, max_epochs=20, known shortfall.
        r = sh.assess("everyday-conversations", 403_573, 1_200_000_000,
                      [5_000_000], shard_idx=1, max_epochs=20)
        assert r.verdict == sh.SAFE
        assert "re-reads" in r.note

    def test_no_budget_means_met(self):
        assert sh.assess("s", 0, 0, [100], shard_idx=0).verdict == sh.MET


class TestExitStatus:
    def test_short_verdicts_are_the_ones_that_fail_a_run(self):
        assert sh.EXHAUSTED in sh._SHORT and sh.AT_RISK in sh._SHORT
        # UNKNOWN must not fail a run: a Hub hiccup is not evidence of trouble.
        for ok in (sh.SAFE, sh.MET, sh.UNKNOWN):
            assert ok not in sh._SHORT


class TestBudgetParsing:
    def test_parses_repeated_pairs_including_scientific_notation(self):
        got = sh.parse_budgets(["fineweb-edu=5625000000", "finewiki-en=4.5e8"])
        assert got == {"fineweb-edu": 5_625_000_000, "finewiki-en": 450_000_000}

    def test_reads_the_live_list_shaped_manifest(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text('{"sources": [{"key": "a", "tokens": 5}]}')
        assert sh.load_manifest(str(p))["a"]["tokens"] == 5

    def test_reads_a_dict_shaped_manifest_too(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text('{"sources": {"a": {"tokens": 5}}}')
        assert sh.load_manifest(str(p))["a"]["tokens"] == 5
