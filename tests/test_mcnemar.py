"""Tests for scripts/mcnemar.py and eval.py's per-item sidecar.

The project's headline claim is decided on ~1 point across a peer group that
spans 1.1 points. Unpaired, that is unresolvable (±0.83). Paired, it usually is
not — but only if the pairing is *valid*, which is what most of these tests are
about: comparing item 7 of one run against a different question in the other
produces a confident, meaningless p-value, and nothing about the output would
look wrong.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from mcnemar import compare, mcnemar, render, _cli  # noqa: E402


def _side(model, **tasks):
    """A sidecar in the shape eval.py writes."""
    return {"models": {model: {
        name: {"n": len(items), "digest": digest, "headline": "acc",
               "items_acc": items, "items_acc_norm": None}
        for name, (items, digest) in tasks.items()}}}


def test_only_disagreements_carry_information():
    """The point of pairing: items both models get right or both get wrong are
    discarded, which is why this resolves differences the unpaired bound
    cannot."""
    a = [1, 1, 1, 1, 0, 0, 1, 0]
    b = [1, 1, 1, 1, 0, 0, 0, 1]      # differs on the last two only
    r = mcnemar(a, b)
    assert r["n_discordant"] == 2 and r["b01"] == 1 and r["b10"] == 1
    assert r["diff_pts"] == pytest.approx(0.0)


def test_a_consistent_one_sided_win_is_significant_at_scale():
    """60 items one way, 20 the other, out of 1000 -- the shape of a real
    1-point win. Unpaired this would be inside the noise."""
    a = [1] * 20 + [0] * 60 + [1] * 920
    b = [0] * 20 + [1] * 60 + [1] * 920
    r = mcnemar(a, b)
    assert r["b01"] == 20 and r["b10"] == 60
    assert r["diff_pts"] == pytest.approx(4.0)
    assert r["p"] < 0.01


def test_identical_models_are_not_a_result():
    r = mcnemar([1, 0, 1, 0], [1, 0, 1, 0])
    assert r["n_discordant"] == 0 and r["p"] == 1.0
    assert r["diff_pts"] == 0.0


def test_a_thin_discordant_count_is_flagged_not_quietly_reported():
    """Below ~10 disagreements the normal approximation is thin, and a p-value
    printed without that caveat invites a claim the data cannot support."""
    a = _side("a", piqa=([1, 0, 1, 1, 0, 1], "d1"))
    b = _side("b", piqa=([1, 1, 1, 1, 0, 1], "d1"))
    out = render(compare(a, b), "a", "b")
    assert "Fewer than 10 discordant pairs" in out


def test_mismatched_item_fingerprints_are_refused_not_compared():
    """The silent-and-total failure. A --task-limit on one side, or a dataset
    revision that reorders rows, pairs different questions together."""
    a = _side("a", piqa=([1, 0, 1], "digest-A"))
    b = _side("b", piqa=([0, 1, 1], "digest-B"))
    result = compare(a, b)
    assert result["mismatched_tasks"] == ["piqa"]
    assert "piqa" not in result["per_task"]
    assert "Not compared" in render(result, "a", "b")


def test_differing_item_counts_are_refused_even_if_a_digest_matched():
    a = _side("a", piqa=([1, 0, 1], "same"))
    b = _side("b", piqa=([1, 0], "same"))
    assert compare(a, b)["mismatched_tasks"] == ["piqa"]


def test_length_mismatch_raises_rather_than_zipping_short():
    """`zip` would silently truncate to the shorter list and report a result
    over a subset, which is the same class of bug as the digest mismatch."""
    with pytest.raises(ValueError):
        mcnemar([1, 0, 1], [1, 0])


def test_pooled_weights_items_and_the_output_says_so():
    """Pooling answers 'more questions right', not 'higher 5-task mean' --
    HellaSwag's 10,042 items outweigh OpenBookQA's 500. Conflating the two
    would let a pooled win be reported as beating the peer table's bar."""
    a = _side("a", hellaswag=([1] * 100, "h"), openbookqa=([0] * 10, "o"))
    b = _side("b", hellaswag=([1] * 100, "h"), openbookqa=([1] * 10, "o"))
    result = compare(a, b)
    assert result["pooled"]["n_items"] == 110
    out = render(result, "a", "b")
    assert "Pooled counts items, not tasks" in out
    assert "5-task mean" in out


def test_a_sidecar_with_two_models_is_an_error_not_a_guess():
    a = {"models": {"m1": {}, "m2": {}}}
    with pytest.raises(SystemExit):
        compare(a, _side("b", piqa=([1], "d")))


def test_the_headline_metric_is_the_one_compared():
    """acc_norm for the MC tasks, acc for WinoGrande -- the same selection the
    peer table quotes. Comparing acc while the table reports acc_norm would
    produce a p-value about a number nobody published."""
    a = {"models": {"a": {"piqa": {"n": 4, "digest": "d", "headline": "acc_norm",
                                   "items_acc": [1, 1, 1, 1],
                                   "items_acc_norm": [0, 0, 0, 0]}}}}
    b = {"models": {"b": {"piqa": {"n": 4, "digest": "d", "headline": "acc_norm",
                                   "items_acc": [1, 1, 1, 1],
                                   "items_acc_norm": [1, 1, 1, 1]}}}}
    r = compare(a, b)["per_task"]["piqa"]
    assert r["b10"] == 4 and r["b01"] == 0     # acc_norm differs, acc does not


def test_cli_writes_markdown(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_side("a", piqa=([1, 0, 1, 0], "d"))))
    b.write_text(json.dumps(_side("b", piqa=([1, 1, 1, 0], "d"))))
    out = tmp_path / "paired.md"
    _cli([str(a), str(b), "--out", str(out)])
    assert "McNemar" in out.read_text()


# --- the producer side: eval.py's sidecar -----------------------------------

def test_eval_records_per_item_outcomes_for_both_metrics():
    """Without these the paired test above has no input, and the only error bar
    available is the unpaired one this project's margins do not survive."""
    import eval as ev

    class _Ex:
        def __init__(self, label, lengths):
            self.candidates = [("ctx", " a"), ("ctx", " bb")]
            self.label = label
            self.choice_lengths = lengths

    examples = [_Ex(0, [1, 2]), _Ex(1, [1, 2])]
    scores = {0: [1.0, 0.0], 1: [1.0, 0.0]}     # always picks candidate 0
    res = ev.evaluate_cloze_task.__wrapped__ if hasattr(
        ev.evaluate_cloze_task, "__wrapped__") else ev.evaluate_cloze_task

    import unittest.mock as mock
    with mock.patch.object(ev, "score_example",
                           side_effect=lambda *a, **k: scores[0]):
        out = res(None, None, examples, "cpu")
    assert out["items_acc"] == [1, 0]           # right on ex0, wrong on ex1
    assert len(out["items_acc_norm"]) == 2
    assert out["correct"] == 1


def test_the_item_digest_changes_when_the_items_do():
    """It is what makes the mismatch refusal above possible."""
    import eval as ev

    class _Ex:
        def __init__(self, label, ctx):
            self.candidates = [(ctx, " a")]
            self.label = label
            self.choice_lengths = [1]

    a = [_Ex(0, "q1"), _Ex(1, "q2")]
    assert ev.item_digest(a) == ev.item_digest([_Ex(0, "q1"), _Ex(1, "q2")])
    assert ev.item_digest(a) != ev.item_digest([_Ex(0, "q1")])          # limit
    assert ev.item_digest(a) != ev.item_digest([_Ex(1, "q2"), _Ex(0, "q1")])
