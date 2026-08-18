"""Tests for scripts/eval_noise.py.

The project's success definition is "beat Pythia-160M, OPT-125M and
GPT-neo-125M on quality", and those three sit inside a 1.1-point band on our
harness. Nothing here had ever attached an error bar to a 5-task mean, so
whether a 1-point win is a result or a coin flip was an open question stated
nowhere. This computes the part that *is* computable.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from eval_noise import (TASKS, render, summarize, task_stderr,  # noqa: E402
                        _cli)


def _flat(**over):
    """A peer-shaped (flat) eval JSON at roughly our 0.5B numbers."""
    base = {"hellaswag": 0.273, "hellaswag_n": 10042,
            "arc_easy": 0.387, "arc_easy_n": 2376,
            "piqa": 0.560, "piqa_n": 1838,
            "openbookqa": 0.284, "openbookqa_n": 500,
            "winogrande": 0.507, "winogrande_n": 1267}
    base.update(over)
    return base


def test_stderr_is_the_binomial_one():
    assert task_stderr(0.5, 10_000) == pytest.approx(0.5, abs=1e-9)
    assert task_stderr(0.25, 500) == pytest.approx(
        100 * math.sqrt(0.25 * 0.75 / 500), rel=1e-12)


def test_a_bigger_split_is_a_tighter_estimate():
    assert task_stderr(0.3, 10_000) < task_stderr(0.3, 500)


def test_the_mean_is_tighter_than_its_worst_task():
    """Averaging k tasks divides by k, it does not average the variances.

    Getting this backwards -- sqrt(mean of variances) instead of
    sqrt(sum)/k -- overstates the mean's error by a factor of k, which would
    make every comparison look unresolvable and quietly kill an honest claim.
    """
    s = summarize(_flat())
    worst = max(t["stderr_pts"] for t in s["per_task"].values())
    assert s["mean_stderr_pts"] < worst
    expected = math.sqrt(sum(t["stderr_pts"] ** 2
                             for t in s["per_task"].values())) / len(TASKS)
    assert s["mean_stderr_pts"] == pytest.approx(expected, rel=1e-12)


def test_our_measured_numbers_reproduce_the_published_error_bar():
    """Pins the figure the gate quotes: +/-0.59 points on the 5-task mean."""
    s = summarize(_flat())
    assert s["mean_pts"] == pytest.approx(40.22, abs=0.05)
    assert s["mean_stderr_pts"] == pytest.approx(0.59, abs=0.02)


def test_openbookqa_is_flagged_as_the_coarse_column():
    """500 items means one question is 0.2 points, and it is the task most
    likely to move a mean for no reason."""
    out = render([summarize(_flat())], ["us"])
    assert "0.2 points" in out and "500 items" in out


def test_a_two_model_comparison_reports_sigmas_and_the_2_sigma_bar():
    a = summarize(_flat())
    b = summarize(_flat(hellaswag=0.312, arc_easy=0.401, piqa=0.621,
                        openbookqa=0.276, winogrande=0.498))
    out = render([a, b], ["us @0.5B", "GPT-2 124M"])
    assert "+1.9" in out and "σ" in out
    assert "1.65 points" in out          # 2 x the unpaired se
    assert "upper bound" in out          # the paired caveat, not buried


def test_the_paired_caveat_is_stated_rather_than_omitted():
    """Both models answer the same items, so the unpaired sigma overstates the
    error of the difference. Reporting it as if it were exact would make real
    differences look unresolved -- the opposite failure to the one this tool
    exists to prevent, and just as misleading."""
    out = render([summarize(_flat()), summarize(_flat())], ["a", "b"])
    assert "paired" in out and "upper bound" in out


def test_seed_variance_is_explicitly_out_of_scope():
    """The number this produces is benchmark sampling only. Presenting it as
    'the' error bar would understate the real uncertainty, since seed sigma at
    this scale is reported far larger and nothing here has measured it."""
    out = render([summarize(_flat())], ["us"])
    assert "seed variance" in out and "not bound" in out


def test_a_missing_task_is_dropped_rather_than_scored_as_zero():
    data = _flat()
    del data["piqa"], data["piqa_n"]
    s = summarize(data)
    assert "piqa" not in s["per_task"]
    assert s["mean_pts"] == pytest.approx(
        sum(t["acc_pts"] for t in s["per_task"].values()) / 4)
    assert "-" in render([s], ["partial"])


def test_our_nested_shape_and_the_flat_peer_shape_both_work():
    """`ours-*.json` wraps its numbers in `mean`; `peer-*.json` does not."""
    assert summarize({"mean": _flat()})["mean_pts"] == pytest.approx(
        summarize(_flat())["mean_pts"])


def test_an_empty_result_does_not_raise():
    s = summarize({})
    assert s["mean_pts"] is None
    assert "-" in render([s], ["nothing"])


def test_cli_writes_markdown(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_flat()))
    out = tmp_path / "noise.md"
    _cli([str(path), "--labels", "us", "--out", str(out)])
    assert "5-task mean" in out.read_text()
