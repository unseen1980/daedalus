"""Tests for scripts/peer_table.py.

This script renders the one table the whole project is judged by -- the bar in
README.md and the quality half of the hero gate are both read off it -- and it
had no tests at all. The failure mode that matters here is not a crash but a
*plausible wrong table*: a row that silently vanishes, or one that states no
token budget and so invites reading a 0.5B probe as the finished model.
"""
import glob
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import peer_table  # noqa: E402

TASKS = peer_table.TASKS


# The real full-split item counts eval.py records. Carried in the fixture
# because a fixture that omits them is not eval.py-shaped, and the omission hid
# that `ours_rows` was not propagating them while `load_rows` was.
TASK_N = {"hellaswag": 10042, "arc_easy": 2376, "piqa": 1838,
          "openbookqa": 500, "winogrande": 1267}


def _results(**scores):
    """An eval.py-shaped results.json (fractions, not percentages)."""
    mean = {t: scores.get(t, 0.30) for t in TASKS}
    mean.update({f"{t}_n": float(n) for t, n in TASK_N.items()})
    return {"per_checkpoint": [dict(mean, checkpoint="ckpt.pt")], "mean": mean}


def _write(tmp_path, name, **scores):
    path = tmp_path / name
    path.write_text(json.dumps(_results(**scores)))
    return str(path)


def _peer(tmp_path, slug, model_id=None, **scores):
    """A peer file. The row's display name comes from the JSON's `checkpoint`
    (the HF model id, as eval.py writes it), while the *published* lookup keys
    off the filename slug -- so the two differ on purpose."""
    path = tmp_path / f"peer-{slug}.json"
    r = _results(**scores)
    r["per_checkpoint"][0]["checkpoint"] = model_id or slug
    path.write_text(json.dumps(r))
    return path


# --- the silent-skip bug -----------------------------------------------------

def test_missing_ours_file_raises_instead_of_rendering_a_peer_only_table(tmp_path):
    """A typo'd --ours path used to render the peer table with no Daedalus row.

    That output is indistinguishable from a real table and reads as though we
    scored, so it must fail loudly rather than quietly drop the row.
    """
    with pytest.raises(FileNotFoundError):
        peer_table.ours_rows([str(tmp_path / "nope.json")], [], [])


def test_no_ours_flag_is_still_fine():
    assert peer_table.ours_rows([], [], []) == []


# --- the unlabelled-token-budget hazard --------------------------------------

def test_ours_row_without_tokens_renders_a_question_mark_not_a_dash(tmp_path):
    """'-' reads as 'not applicable'; '?' reads as 'unstated', which is true."""
    rows = peer_table.ours_rows([_write(tmp_path, "o.json")], [], [])
    assert rows[0]["tokens"] == "?"
    assert "| ? |" in peer_table.render(rows)


def test_ours_row_states_its_own_token_budget(tmp_path):
    rows = peer_table.ours_rows(
        [_write(tmp_path, "o.json")], ["Daedalus-150M @0.5B"], ["0.5B"])
    out = peer_table.render(rows)
    assert "**Daedalus-150M @0.5B**" in out
    assert "| 0.5B |" in out


def test_a_peer_row_still_gets_its_published_token_count(tmp_path):
    """The row-level override must not shadow PUBLISHED for the peers."""
    _peer(tmp_path, "eleutherai-pythia-160m")
    rows = peer_table.load_rows(str(tmp_path / "peer-*.json"))
    assert "| 300B |" in peer_table.render(rows)


# --- the slope: several of our own checkpoints in one table ------------------

def test_two_ours_files_produce_two_rows_in_order(tmp_path):
    """0.5B and 5B together are the point -- one dot cannot show a slope."""
    rows = peer_table.ours_rows(
        [_write(tmp_path, "a.json", hellaswag=0.27),
         _write(tmp_path, "b.json", hellaswag=0.31)],
        ["ours@0.5B", "ours@5B"], ["0.5B", "5B"])
    assert [r["name"] for r in rows] == ["**ours@0.5B**", "**ours@5B**"]
    assert rows[0]["hellaswag"] == pytest.approx(27.0)
    assert rows[1]["hellaswag"] == pytest.approx(31.0)
    assert rows[0]["key"] != rows[1]["key"]  # distinct, else PUBLISHED collides


def test_mismatched_label_count_is_rejected(tmp_path):
    """Positional pairing silently mislabels rows if the lists differ."""
    with pytest.raises(SystemExit):
        peer_table.ours_rows(
            [_write(tmp_path, "a.json"), _write(tmp_path, "b.json")],
            ["only-one"], [])


# --- arithmetic the bar is read off ------------------------------------------

def test_mean5_is_the_plain_mean_of_the_five_tasks(tmp_path):
    rows = peer_table.ours_rows(
        [_write(tmp_path, "o.json", hellaswag=0.2734, arc_easy=0.3868,
                piqa=0.5604, openbookqa=0.284, winogrande=0.5067)], [], [])
    # The real 0.5B probe: chance is 35.0, the bar is 42.2.
    assert peer_table.mean5(rows[0]) == pytest.approx(40.23, abs=0.02)


def test_ours_row_is_not_matched_against_a_published_peer(tmp_path):
    """Our key must not collide with PUBLISHED, or we would print somebody
    else's published average next to our score."""
    rows = peer_table.ours_rows([_write(tmp_path, "o.json")], [], [])
    assert rows[0]["key"] not in peer_table.PUBLISHED
    # The *row*, not the last line: render() now closes with a footnote
    # explaining the error bars, and anchoring on "last line" made this test a
    # hostage to anything appended below the table.
    line = next(l for l in peer_table.render(rows).splitlines()
                if l.startswith(f"| {rows[0]['name']} |"))
    assert line.endswith("| - | ? |")  # no published avg, tokens unstated


def test_delta_table_leaves_our_row_blank(tmp_path):
    """render_delta compares against PUBLISHED; ours has none, so it must show
    dashes rather than inventing a comparison."""
    rows = peer_table.ours_rows([_write(tmp_path, "o.json")], ["ours"], ["5B"])
    assert peer_table.render_delta(rows).splitlines()[-1].count("-") >= 5


# --- end to end through the CLI ----------------------------------------------

def test_cli_renders_peers_and_ours_together(tmp_path):
    _peer(tmp_path, "eleutherai-pythia-160m", hellaswag=0.304)
    ours = _write(tmp_path, "ours.json", hellaswag=0.273)
    out = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "peer_table.py"),
         "--pattern", str(tmp_path / "peer-*.json"),
         "--ours", ours, "--ours-label", "Daedalus-150M @0.5B",
         "--ours-tokens", "0.5B"],
        capture_output=True, text=True, check=True).stdout
    assert "pythia-160m" in out
    assert "**Daedalus-150M @0.5B**" in out
    assert "| 0.5B |" in out


def test_cli_fails_loudly_on_a_bad_ours_path(tmp_path):
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "peer_table.py"),
         "--pattern", str(tmp_path / "peer-*.json"),
         "--ours", str(tmp_path / "missing.json")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "missing.json" in r.stderr


def test_the_mean_carries_an_error_bar_when_item_counts_are_present(tmp_path):
    """The peers sit inside a 1.1-point band and this table is where that band
    is read, so a bare mean invites a 0.3-point gap being taken for a result."""
    rows = peer_table.ours_rows(
        [_write(tmp_path, "o.json", hellaswag=0.2734, arc_easy=0.3868,
                piqa=0.5604, openbookqa=0.284, winogrande=0.5067)], [], [])
    se = peer_table.mean5_stderr(rows[0])
    assert se == pytest.approx(0.59, abs=0.02)
    out = peer_table.render(rows)
    assert "± 0.59" in out
    assert "1.65 points" in out and "seed variance" in out


def test_a_row_without_item_counts_renders_a_bare_mean_not_a_fake_one(tmp_path):
    """An older results file has no `<task>_n`. Inventing an error bar from an
    assumed n would be worse than omitting it."""
    row = {"name": "old", "key": "old", "hellaswag": 30.0, "piqa": 60.0}
    assert peer_table.mean5_stderr(row) is None
    line = next(l for l in peer_table.render([row]).splitlines()
                if l.startswith("| old |"))
    assert "±" not in line


def test_the_error_bar_shrinks_with_a_bigger_split():
    small = {"name": "a", "key": "a", "hellaswag": 30.0, "hellaswag_n": 500}
    big = {"name": "b", "key": "b", "hellaswag": 30.0, "hellaswag_n": 10042}
    assert peer_table.mean5_stderr(big) < peer_table.mean5_stderr(small)


def test_the_reproduction_script_rebuilds_every_peer_the_table_reports():
    """README quotes `bash scripts/eval_peers.sh && python scripts/peer_table.py`
    as how the bar is reproduced, and `peer_table.py` renders whatever
    `runs/eval/peer-*.json` happens to be on disk. Those two facts only agree if
    the script scores every peer the glob picks up.

    They did not. `openai-community/gpt2` was scored by hand on 2026-08-09 and
    never added to the loop, so the documented command rebuilt five peers and
    left the sixth as a stale file rendered beside them -- and that sixth is the
    one measuring **42.2**, i.e. the peer that *sets* the success bar. A reader
    reproducing the table would have got a different bar than the one the
    project is judged against, with nothing anywhere saying so.
    """
    script = os.path.join(REPO, "scripts", "eval_peers.sh")
    body = open(script).read()
    # The slug eval_peers.sh derives for each model is exactly the peer-*.json
    # stem peer_table.py globs, so compare on that rather than on repo casing.
    scored = {m.lower().replace("/", "-")
              for m in body.split("for M in", 1)[1].split("; do", 1)[0].split()
              if "/" in m}

    reported = {os.path.basename(p)[len("peer-"):-len(".json")]
                for p in glob.glob(os.path.join(REPO, "runs", "eval",
                                                "peer-*.json"))}

    missing = reported - scored
    assert not missing, (
        f"peer-*.json files the table renders that eval_peers.sh does not "
        f"score: {sorted(missing)} -- the documented reproduction command "
        f"would leave them stale")
