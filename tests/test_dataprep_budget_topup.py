"""The resume contract for a *budget top-up*, not just a crash.

`run_dataprep(resume=True)` used to treat any manifest entry without an error
as finished. That is correct when resuming a crash -- the budgets are unchanged
and the only question is where to continue -- and silently wrong when a later
run asks for *more* of a source: the source was skipped on the strength of its
older, smaller entry, so raising `--source-budget` could never take effect.

Measured consequence (2026-08-10 18:03Z): the 60B corpus top-up ran to
completion in twelve seconds, wrote nothing, and exited 0. All ten sources were
short of their new budgets -- 3,499,030,510 tokens short in total, which was
precisely the amount the run existed to add -- and all ten were skipped as
"already recorded in manifest (resume)".

The contract now, implemented in `_demote_short_sources`:

  * a source is done when its recorded tokens reach the budget *this* run asks
    for (`spec.share * target_tokens`), not when it merely has an entry;
  * a short source with a resume point is continued, seeded so its shards
    accumulate rather than restart at `_00000`;
  * a short source with **no** resume point stays done and says so. It is the
    one case where doing less is right: restarting it would overwrite shards
    already on disk, and fineweb-edu alone is 3.75B tokens. Falling short can
    be fixed by running again; overwriting the corpus cannot.

The decision is tested directly rather than through `run_dataprep`, which
dispatches to a process pool -- these assert on which sources are handed to a
worker and with what resume state, which is the whole of the behaviour.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from daedalus import dataprep
from daedalus.dataprep import _demote_short_sources


def _spec(key, share):
    """A SourceSpec with a given key/share, built from a real one."""
    return replace(dataprep.MIXTURE[0], key=key, share=share,
                   near_dup_group=key, dataset=f"stub/{key}")


def _entry(key, tokens, n_seen=0, stream_state=None, error=None):
    e = {"key": key, "tokens": tokens, "n_seen": n_seen,
         "shards": [f"{key}_00000.bin"], "n_kept": n_seen}
    if stream_state is not None:
        e["stream_state"] = stream_state
    if error is not None:
        e["error"] = error
    return e


def _decide(specs, entries, target, monkeypatch, recovered=None):
    """Run the demotion and report what changed.

    `_recover_source_stats` is stubbed so nothing reads the repo's real
    `data/shards` -- these tests must never touch the 14.2B-token corpus the
    project has already paid for.
    """
    monkeypatch.setattr(dataprep, "_recover_source_stats",
                        lambda key, root: (recovered or {}).get(key, {}))
    manifest = {"sources": entries}
    done = {e["key"] for e in entries if not e.get("error")}
    resume_state: dict = {}
    logged: list = []
    _demote_short_sources(specs, manifest, target, "unused-root", done,
                          resume_state, log=logged.append)
    return done, resume_state, "\n".join(logged)


# --------------------------------------------------------------------------

def test_a_source_short_of_a_raised_budget_is_continued(monkeypatch):
    """The bug, in one test: 3.75B on disk, 5.625B asked for, must continue."""
    done, resume_state, log = _decide(
        [_spec("fineweb-edu", 1.0)],
        [_entry("fineweb-edu", 3_750_002_609, 3_527_006, {"pos": 42})],
        5_625_000_000, monkeypatch)

    assert "fineweb-edu" not in done, "short source still counted as finished"
    state = resume_state["fineweb-edu"]
    assert state["stream_state"] == {"pos": 42}, "no position -- would restart at row 0"
    assert state["resume_seed"]["tokens"] == 3_750_002_609, "prior tokens not carried"
    assert state["resume_skip"] == 3_527_006
    assert "continuing for 1,874,997,391 more" in log


def test_a_source_that_met_its_budget_is_still_skipped(monkeypatch):
    """The original behaviour has to survive: no needless re-streaming."""
    done, resume_state, _ = _decide(
        [_spec("finemath-3plus", 1.0)],
        [_entry("finemath-3plus", 1_350_008_857, 100, {"pos": 1})],
        1_350_008_857, monkeypatch)
    assert done == {"finemath-3plus"}
    assert resume_state == {}


def test_a_source_over_its_budget_is_skipped(monkeypatch):
    """Lowering a budget must not re-stream, and must not truncate anything."""
    done, resume_state, _ = _decide(
        [_spec("finephrase", 1.0)],
        [_entry("finephrase", 2_066_170_024, 100, {"pos": 1})],
        1_000_000_000, monkeypatch)
    assert done == {"finephrase"} and resume_state == {}


def test_a_short_source_without_a_resume_point_is_not_restarted(monkeypatch):
    """The data-loss case. Restarting would overwrite shards from _00000."""
    done, resume_state, log = _decide(
        [_spec("orphan", 1.0)],
        [_entry("orphan", 3_750_000_000, n_seen=0, stream_state=None)],
        5_000_000_000, monkeypatch)
    assert done == {"orphan"}, "restarted a source with no resume point -- data loss"
    assert resume_state == {}
    assert "no resume position" in log and "overwriting 3,750,000,000 tokens" in log


def test_a_position_recovered_from_disk_is_enough_to_continue(monkeypatch):
    """The manifest entry may predate stream_state; the shard dir still has it."""
    done, resume_state, _ = _decide(
        [_spec("finewiki-en", 1.0)],
        [_entry("finewiki-en", 410_000_206, n_seen=278_160, stream_state=None)],
        450_000_000, monkeypatch,
        recovered={"finewiki-en": {"stream_state": {"pos": 7}}})
    assert "finewiki-en" not in done
    assert resume_state["finewiki-en"]["stream_state"] == {"pos": 7}


def test_n_seen_only_source_is_continued_by_replay(monkeypatch):
    """stack-edu-python's shape: a position exists, it is just O(n) to reach."""
    done, resume_state, log = _decide(
        [_spec("stack-edu-python", 1.0)],
        [_entry("stack-edu-python", 1_210_964_651, 641_000, None)],
        1_350_000_000, monkeypatch)
    assert "stack-edu-python" not in done
    assert resume_state["stack-edu-python"]["resume_skip"] == 641_000
    assert "replaying 641,000 docs" in log


def test_an_errored_source_is_untouched(monkeypatch):
    """Errored entries were never in done_keys; the retry path still owns them."""
    done, resume_state, _ = _decide(
        [_spec("fineweb-edu", 1.0)],
        [_entry("fineweb-edu", 100, 10, {"p": 1}, error="WorkerMemoryExceeded")],
        5_000_000_000, monkeypatch)
    assert done == set() and resume_state == {}


def test_the_real_shortfall_continues_every_short_source(monkeypatch):
    """The live case, on the real 60B numbers: five short, five at budget."""
    on_disk = {
        "fineweb-edu": (3_750_002_609, 3_527_006, {"p": 1}),
        "dclm-baseline": (2_250_000_677, 1_780_740, {"p": 2}),
        "finepdfs-edu": (880_001_347, 210_890, {"p": 3}),
        "finewiki-en": (410_000_206, 278_160, {"p": 4}),
        "stack-edu-python": (1_210_964_651, 641_000, None),
        "finemath-3plus": (1_350_008_857, 10, {"p": 5}),
        "infiwebmath-3plus": (1_350_000_198, 10, {"p": 6}),
        "finephrase": (2_066_170_024, 10, {"p": 7}),
        "cosmopedia-v2": (950_000_576, 10, {"p": 8}),
        "everyday-conversations": (403_573, 10, {"p": 9}),
    }
    budgets = {"fineweb-edu": 5_625_000_000, "dclm-baseline": 3_375_000_000,
               "stack-edu-python": 1_350_000_000, "finepdfs-edu": 1_200_000_000,
               "finephrase": 2_066_170_024, "finemath-3plus": 1_350_008_857,
               "infiwebmath-3plus": 1_350_000_198, "cosmopedia-v2": 950_000_576,
               "finewiki-en": 450_000_000, "everyday-conversations": 403_573}
    total = sum(budgets.values())

    specs = [_spec(k, v / total) for k, v in budgets.items()]
    entries = [_entry(k, *v) for k, v in on_disk.items()]
    done, resume_state, _ = _decide(specs, entries, total, monkeypatch)

    short = {k for k, v in budgets.items() if on_disk[k][0] < v}
    assert short == {"fineweb-edu", "dclm-baseline", "finepdfs-edu",
                     "finewiki-en", "stack-edu-python"}
    assert set(resume_state) == short, f"continuing {sorted(resume_state)}, expected {sorted(short)}"
    assert done == set(budgets) - short
    # and the amount at stake is what the top-up was sized for
    assert sum(budgets[k] - on_disk[k][0] for k in short) == 3_499_030_510


def test_share_rounding_does_not_invent_a_shortfall(monkeypatch):
    """A source at exactly its budget must not be reopened by float drift.

    `share` is a ratio, so the budget is recomputed as share*target on every
    run. If that lands one token above the recorded count, the source is
    re-dispatched forever and every run re-streams for nothing.
    """
    budgets = {"a": 3_333_333_333, "b": 3_333_333_333, "c": 3_333_333_334}
    total = sum(budgets.values())
    specs = [_spec(k, v / total) for k, v in budgets.items()]
    entries = [_entry(k, int(round((v / total) * total)), 10, {"p": 1})
               for k, v in budgets.items()]
    done, resume_state, _ = _decide(specs, entries, total, monkeypatch)
    assert resume_state == {}, f"rounding reopened {sorted(resume_state)}"
    assert done == set(budgets)
