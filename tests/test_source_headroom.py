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
import math
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


# ------------------------------------------------- unique supply and epochs

# Copied from data/shards/fineweb-edu/manifest.json on this box. The shape is
# the trap: `total_tokens` describes the *fetched subset* and `subset_of` the
# whole released build, so reading the obvious key understates the source by
# 10.8x -- and understating supply is how a corpus reports a shortfall it does
# not have.
FINEWEB_SHARD_MANIFEST = {
    "source_key": "fineweb-edu",
    "source_dataset": "HuggingFaceFW/fineweb-edu",
    "stream_state": {"epoch": 0, "hf_state": {"examples_iterable": {
        "examples_iterable": {"shard_idx": 0, "shard_example_idx": 340000},
        "previous_state": {"shard_idx": 6, "shard_example_idx": 184000},
        "batch_idx": 4900000}}},
    "n_seen": 4900000, "n_kept": 4899632,
    "total_tokens": 479_185_034,
    "subset_of": {"shards": 103, "total_tokens": 5_193_493_853},
}

# data/shards/stack-edu-python/manifest.json: the source that genuinely ran out
# of documents, and the one whose manifest carries **no** stream position.
STACK_SHARD_MANIFEST = {
    "source_key": "stack-edu-python",
    "source_dataset": "codeparrot/github-code",
    "total_tokens": 200_000_000,
    "subset_of": {"shards": 13, "total_tokens": 1_210_964_651},
}


class TestRealizedTokensComeFromTheWholeBuild:
    def test_prefers_the_released_total_over_the_fetched_subset(self):
        tokens, basis = sh.realized_tokens(FINEWEB_SHARD_MANIFEST)
        assert tokens == 5_193_493_853
        assert "subset_of" in basis

    def test_falls_back_to_total_tokens_when_nothing_was_subsetted(self):
        tokens, basis = sh.realized_tokens({"total_tokens": 403_573})
        assert tokens == 403_573
        assert "total_tokens" in basis

    def test_a_manifest_with_neither_key_reports_nothing_rather_than_guessing(self):
        assert sh.realized_tokens({})[0] == 0


class TestSupplyRefusesToCreditFilesItCannotPlace:
    """No stream position means no density, and no density means no reachable
    remainder. Crediting file 0 as "the file we stopped in" would divide the
    whole build's tokens by one file's bytes and inflate a 1.2B source to
    something like 15B."""

    def test_no_stream_position_credits_only_what_was_realized(self):
        s = sh.supply_from_manifest("stack-edu-python", STACK_SHARD_MANIFEST,
                                    file_sizes=STACK_SIZES)
        assert s.unique_tokens == 1_210_964_651
        assert s.reachable_tokens is None
        assert s.density_tok_per_byte is None
        assert "no stream position" in s.basis

    def test_no_file_sizes_credits_only_what_was_realized(self):
        s = sh.supply_from_manifest("fineweb-edu", FINEWEB_SHARD_MANIFEST)
        assert s.unique_tokens == 5_193_493_853
        assert s.reachable_tokens is None

    def test_a_stream_position_adds_the_untouched_files(self):
        # 7 of 10 files touched at shard_idx 6; density is the build's own
        # tokens over those 7 files, and the remaining 3 are credited at it.
        s = sh.supply_from_manifest("fineweb-edu", FINEWEB_SHARD_MANIFEST,
                                    file_sizes=[1_000_000] * 10)
        assert s.files_consumed == 6
        assert s.density_tok_per_byte == 5_193_493_853 / 7_000_000
        assert s.reachable_tokens == int(s.density_tok_per_byte * 3_000_000)
        assert s.unique_tokens == 5_193_493_853 + s.reachable_tokens
        assert "untouched" in s.basis

    def test_an_exhausted_stream_has_nothing_left_to_reach(self):
        s = sh.supply_from_manifest("stack-edu-python",
                                    dict(STACK_SHARD_MANIFEST, stream_state=STACK_STATE),
                                    file_sizes=STACK_SIZES)
        assert s.files_consumed == s.files_total == 10
        assert s.reachable_tokens == 0
        assert s.unique_tokens == 1_210_964_651


class TestEpochCurve:
    """A budget times a share is a demand; a supply divided into it is epochs."""

    SPECS = [("fineweb-edu", 0.375, 1), ("stack-edu-python", 0.09, 1)]

    def _curve(self, budgets=(1_000_000_000_000,), supplies=None):
        supplies = supplies or {
            "fineweb-edu": sh.Supply(key="fineweb-edu", unique_tokens=5_193_493_853,
                                     realized_tokens=5_193_493_853, basis="realized"),
            "stack-edu-python": sh.Supply(key="stack-edu-python",
                                          unique_tokens=1_210_964_651,
                                          realized_tokens=1_210_964_651, basis="realized"),
        }
        return sh.epoch_curve(supplies, self.SPECS, budgets)

    def test_epochs_are_the_demand_over_the_supply(self):
        row = self._curve()[0]["sources"][0]
        assert row.key == "fineweb-edu"
        assert row.needed_tokens == 375_000_000_000
        assert abs(row.epochs - 375_000_000_000 / 5_193_493_853) < 1e-9
        assert row.epochs > 72          # 72.2 epochs of a corpus built for 3.5

    def test_the_shortfall_is_what_must_be_added_to_reach_the_four_epoch_bar(self):
        row = self._curve()[0]["sources"][0]
        # 375B at 4 epochs needs 93.75B unique; the source has 5.19B.
        assert row.shortfall_tokens == 93_750_000_000 - 5_193_493_853
        assert row.over_cap is True
        assert abs(row.growth_x - 93_750_000_000 / 5_193_493_853) < 1e-9

    def test_a_source_inside_the_bar_reports_no_shortfall(self):
        # 30B x 0.375 = 11.25B; at 4 epochs that needs 2.81B and it has 5.19B.
        row = self._curve(budgets=(30_000_000_000,))[0]["sources"][0]
        assert row.over_cap is False
        assert row.shortfall_tokens == 0
        assert row.epochs < 4

    def test_a_source_with_no_supply_is_infinite_rather_than_a_crash(self):
        supplies = {
            "fineweb-edu": sh.Supply(key="fineweb-edu", unique_tokens=0,
                                     realized_tokens=0, basis="none"),
            "stack-edu-python": sh.Supply(key="stack-edu-python", unique_tokens=1,
                                          realized_tokens=1, basis="realized"),
        }
        row = self._curve(supplies=supplies)[0]["sources"][0]
        assert row.epochs == float("inf")
        assert row.growth_x is None
        assert row.shortfall_tokens == 375_000_000_000 // 4

    def test_every_requested_budget_appears_once_in_ascending_order(self):
        budgets = (100_000_000_000, 30_000_000_000, 1_000_000_000_000)
        got = [point["budget"] for point in self._curve(budgets=budgets)]
        assert got == [30_000_000_000, 100_000_000_000, 1_000_000_000_000]

    def test_totals_name_the_binding_source_not_just_the_aggregate(self):
        totals = self._curve()[0]["totals"]
        # Aggregate epochs (1T over 6.40B unique) hides which source binds.
        assert totals["unique_tokens"] == 5_193_493_853 + 1_210_964_651
        assert totals["sources_over_cap"] == 2
        assert totals["binding_source"] == "stack-edu-python"   # 74.3 epochs
        assert totals["verdict"] == sh.SHORT
        assert totals["shortfall_tokens"] == sum(
            r.shortfall_tokens for r in self._curve()[0]["sources"])

    def test_a_supported_budget_says_so(self):
        totals = self._curve(budgets=(10_000_000_000,))[0]["totals"]
        assert totals["sources_over_cap"] == 0
        assert totals["verdict"] == sh.SUPPORTED

    def test_a_source_with_no_measured_supply_is_carried_as_unknown(self):
        # Missing from `supplies` entirely -- report it, do not drop it, and do
        # not let a silent omission read as a corpus that covers the budget.
        curve = sh.epoch_curve({}, self.SPECS, (30_000_000_000,))
        rows = curve[0]["sources"]
        assert [r.key for r in rows] == ["fineweb-edu", "stack-edu-python"]
        assert all(r.basis == sh.UNKNOWN.lower() or "unknown" in r.basis.lower()
                   for r in rows)
        assert curve[0]["totals"]["verdict"] == sh.SHORT


class TestBudgetLimits:
    """`cap x unique / share`: the budget question asked the other way round."""

    SPECS = [("fineweb-edu", 0.375, 1), ("stack-edu-python", 0.09, 1),
             ("everyday-conversations", 0.02, 20)]

    SUPPLIES = {
        "fineweb-edu": sh.Supply(key="fineweb-edu", unique_tokens=1_424_317_277_390,
                                 realized_tokens=5_193_493_853, basis="measured"),
        "stack-edu-python": sh.Supply(key="stack-edu-python",
                                      unique_tokens=1_210_964_651,
                                      realized_tokens=1_210_964_651, basis="measured"),
        "everyday-conversations": sh.Supply(key="everyday-conversations",
                                            unique_tokens=403_573,
                                            realized_tokens=403_573, basis="measured"),
    }

    def test_the_exhausted_code_source_caps_the_whole_corpus_near_54b(self):
        rows = {r["key"]: r for r in sh.budget_limits(self.SUPPLIES, self.SPECS)}
        assert abs(rows["stack-edu-python"]["max_total_budget"]
                   - 4 * 1_210_964_651 / 0.09) < 1
        assert 53.8e9 < rows["stack-edu-python"]["max_total_budget"] < 53.9e9

    def test_rows_are_ordered_worst_first_so_the_ceiling_reads_first(self):
        got = [r["key"] for r in sh.budget_limits(self.SUPPLIES, self.SPECS)]
        assert got == ["everyday-conversations", "stack-edu-python", "fineweb-edu"]

    def test_a_source_with_no_share_is_unbounded_rather_than_a_zero_divide(self):
        rows = sh.budget_limits(self.SUPPLIES, [("fineweb-edu", 0.0, 1)])
        assert math.isinf(rows[0]["max_total_budget"])

    def test_an_unmeasured_source_supports_nothing_and_says_why(self):
        rows = sh.budget_limits({}, [("fineweb-edu", 0.375, 1)])
        assert rows[0]["max_total_budget"] == 0.0
        assert "unknown" in rows[0]["basis"]


class TestTheCurveIsAReportNotAGate:
    def test_a_corpus_short_of_the_bar_still_exits_zero(self):
        """The expected answer at 1T is "no", and the phase must be able to
        record it. A non-zero exit marks the controller phase failed, which
        would turn the deliverable into a failure and invite someone to widen
        the bar until it passed."""
        assert sh.curve_exit_status([{"totals": {"verdict": sh.SHORT}}]) == 0
