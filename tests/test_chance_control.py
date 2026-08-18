"""Tests for scripts/eval_chance_control.py.

The "chance floor is 35.0" claim now appears in STATUS.md, README.md, the peer
table and the hero gate, and every reading of the 5-task mean is relative to it.
It was derived by hand from "how many choices does each task have". These tests
check that derivation against the tasks the harness actually loads, so a task
whose choice count differs from the assumption cannot silently shift the floor
the whole project is reported against.
"""
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import eval_chance_control as cc  # noqa: E402


def test_chance_table_covers_exactly_the_five_scored_tasks():
    import eval as ev
    assert set(cc.CHANCE) == set(ev.TASKS) if hasattr(ev, "TASKS") else True
    assert set(cc.CHANCE) == {"hellaswag", "arc_easy", "piqa",
                              "openbookqa", "winogrande"}


def test_chance_mean_is_the_35_0_the_project_reports():
    assert sum(cc.CHANCE.values()) / len(cc.CHANCE) == pytest.approx(35.0)


def test_binary_tasks_are_50_and_four_way_tasks_are_25():
    """Pins the shape of the claim, not just its average: a table that put
    PIQA at 25.0 and HellaSwag at 50.0 would still average 35.0."""
    assert cc.CHANCE["piqa"] == 50.0
    assert cc.CHANCE["winogrande"] == 50.0
    assert cc.CHANCE["hellaswag"] == 25.0
    assert cc.CHANCE["arc_easy"] == 25.0
    assert cc.CHANCE["openbookqa"] == 25.0


@pytest.mark.parametrize("task", ["piqa", "winogrande", "hellaswag",
                                  "openbookqa"])
def test_stated_chance_matches_the_real_choice_count(task):
    """The floor is 100/n_choices. Verified against the loaded examples rather
    than against memory of what these datasets look like.

    ARC-Easy is excluded on purpose: its questions carry 3, 4 or 5 choices, so
    it has no single chance value and 25.0 is the nominal one.
    """
    import eval as ev
    loader = getattr(ev, f"load_{task}", None)
    if loader is None:
        pytest.skip(f"no load_{task} in eval.py")
    try:
        examples = loader(limit=25)
    except Exception as e:  # dataset not cached on this box
        pytest.skip(f"{task} unavailable: {e}")
    if not examples:
        pytest.skip(f"{task} loaded no examples")
    counts = {len(ex.candidates) for ex in examples}
    assert len(counts) == 1, f"{task} has mixed choice counts {counts}"
    assert cc.CHANCE[task] == pytest.approx(100.0 / counts.pop())


def test_random_checkpoint_round_trips_through_the_real_load_path(tmp_path):
    """The control is only a control if it goes through the same loader a
    trained checkpoint does -- otherwise it tests a different code path."""
    from daedalus.config import PRESETS
    from daedalus.model import Daedalus
    from train import load_checkpoint

    path = str(tmp_path / "rand.pt")
    cc.write_random_checkpoint(path, "tiny", seed=7)

    model = Daedalus(PRESETS["tiny"])
    meta = load_checkpoint(path, model, map_location="cpu")
    assert meta["step"] == 0
    assert meta["tokens_seen"] == 0


def test_two_seeds_give_different_weights(tmp_path):
    """A control that silently ignored --seed would report one draw as though
    it were the distribution."""
    a, b = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    cc.write_random_checkpoint(a, "tiny", seed=1)
    cc.write_random_checkpoint(b, "tiny", seed=2)
    wa = torch.load(a, map_location="cpu", weights_only=False)["model"]
    wb = torch.load(b, map_location="cpu", weights_only=False)["model"]
    key = next(k for k, v in wa.items() if v.dtype.is_floating_point
               and v.numel() > 16)
    assert not torch.allclose(wa[key], wb[key])


def test_same_seed_is_reproducible(tmp_path):
    a, b = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    cc.write_random_checkpoint(a, "tiny", seed=3)
    cc.write_random_checkpoint(b, "tiny", seed=3)
    wa = torch.load(a, map_location="cpu", weights_only=False)["model"]
    wb = torch.load(b, map_location="cpu", weights_only=False)["model"]
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k
