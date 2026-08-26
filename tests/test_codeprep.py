"""Tests for daedalus/codeprep.py -- the code corpus's decontamination index.

Phase 8's gates read "HumanEval+/MBPP+ pass@1 improves over untouched base".
An index that is written, digested and quoted in a manifest while filtering less
than it claims lets a corpus deliver that by memorisation, and nothing in the
gate can tell the difference. So these tests are about the ways a *reassuring*
code index gets produced: a benchmark that quietly failed to download, one that
came back short, an item too short to filter on at all, and -- the one with no
analogue in the general index -- an index built at an `n` the corpus filter will
never look up.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus import codeprep as CP
from daedalus.data import is_contaminated, ngram_set


# ------------------------------------------------------------------- fakes ---

BENCHMARKS = ("humaneval-plus", "mbpp-plus")


def _humaneval_problem(index: int) -> dict:
    """Shaped like the real thing: a long signature-and-docstring prompt, a body
    that continues it, and a `test` that defines and is expected to call
    `check`."""
    return {
        "task_id": f"HumanEval/{index}",
        "entry_point": f"solve_{index}",
        "prompt": (f'def solve_{index}(numbers: list, threshold: float) -> bool:\n'
                   f'    """ Check whether the given list of numbers {index} has '
                   f'any two elements closer to each other than the given '
                   f'threshold value supplied by the caller.\n'
                   f'    >>> solve_{index}([1.0, 2.0, 3.0], 0.5)\n'
                   f'    False\n'
                   f'    """\n'),
        "canonical_solution": (
            f'    for outer_index, outer in enumerate(numbers):\n'
            f'        for inner_index, inner in enumerate(numbers):\n'
            f'            if outer_index != inner_index and abs(outer - inner) '
            f'< threshold + {index}:\n'
            f'                return True\n'
            f'    return False\n'),
        "test": (f'def check(candidate):\n'
                 f'    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) '
                 f'== True, "case {index} a"\n'
                 f'    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) '
                 f'== False, "case {index} b"\n'),
        "base_input": [[[1.0, 2.0], 0.5]],
        "plus_input": [[[1.0], 0.5]],
        "atol": 0.0,
    }


def _mbpp_problem(index: int) -> dict:
    """MBPP+ ships no `test`; its assertion lives inside the prompt docstring."""
    return {
        "task_id": f"Mbpp/{index}",
        "entry_point": f"find_{index}",
        "prompt": (f'"""\nWrite a python function to find the minimum cost path '
                   f'to reach position {index} from the origin of the given '
                   f'cost matrix.\nassert find_{index}([[1, 2, 3], [4, 8, 2]], '
                   f'1, 2) == {index}\n"""\n'),
        "canonical_solution": (
            f'def find_{index}(cost, m, n):\n'
            f'    total = [[0 for _ in range(n + 1)] for _ in range(m + 1)]\n'
            f'    total[0][0] = cost[0][0] + {index}\n'
            f'    return total[m][n]\n'),
        "base_input": [[[[1, 2]], 0, 1]],
        "plus_input": [[[[1]], 0, 0]],
        "atol": 0.0,
    }


def _loader(counts=None, fail=(), empty=(), problems=None):
    """A stand-in for `code_eval.load_problems`, with its real failure shape: a
    benchmark that cannot be fetched *raises* rather than returning nothing."""
    counts = counts or {"humaneval-plus": 3, "mbpp-plus": 4}

    def load(name: str):
        if name in fail:
            raise RuntimeError(f"could not download {name}")
        if name in empty:
            return {}
        if problems and name in problems:
            return problems[name]
        make = _humaneval_problem if name == "humaneval-plus" else _mbpp_problem
        return {f"{name}/{i}": make(i) for i in range(counts.get(name, 0))}

    return load


def _build(**kw):
    kw.setdefault("loader", _loader())
    kw.setdefault("benchmarks", BENCHMARKS)
    kw.setdefault("expected_items", {})       # sized per test, not per benchmark
    kw.setdefault("now", lambda: "2026-08-26T00:00:00Z")
    kw.setdefault("version", lambda: "0.3.1")
    return CP.build_code_index(**kw)


# ------------------------------------------------------------ completeness ---

def test_a_complete_index_covers_every_benchmark_and_every_indexed_field():
    ngrams, prov = _build()

    assert prov["complete"] is True and prov["kind"] == "code"
    assert set(prov["benchmarks"]) == set(BENCHMARKS)
    assert prov["benchmarks"]["humaneval-plus"]["items"] == 3
    assert prov["benchmarks"]["mbpp-plus"]["items"] == 4
    assert prov["ngrams"] == len(ngrams) > 0
    assert prov["digest"] == CP.index_digest(ngrams)
    assert prov["evalplus"] == "0.3.1"


def test_humaneval_contributes_its_assertions_and_mbpp_does_not_pretend_to():
    """MBPP+ has no `test` key at all. A zero would read as "indexed, matched
    nothing"; an absent key says the dataset does not ship one."""
    _, prov = _build()

    assert set(prov["benchmarks"]["humaneval-plus"]["fields"]) == {
        "prompt", "solution", "reference", "test"}
    assert set(prov["benchmarks"]["mbpp-plus"]["fields"]) == {
        "prompt", "solution", "reference"}


def test_a_benchmark_that_failed_to_download_is_refused_not_warned():
    """The general index's lesson, in the place it costs more: a HumanEval+
    outage would yield an index that looks fine, filters nothing against the
    benchmark phase 8 is gated on, and leaves no trace in the corpus."""
    with pytest.raises(CP.IncompleteIndex) as exc:
        _build(loader=_loader(fail=("humaneval-plus",)))
    assert "humaneval-plus" in str(exc.value)


def test_a_benchmark_that_loaded_zero_items_is_refused():
    with pytest.raises(CP.IncompleteIndex) as exc:
        _build(loader=_loader(empty=("mbpp-plus",)))
    assert "mbpp-plus" in str(exc.value)


def test_a_truncated_benchmark_is_refused_even_though_it_loaded():
    """The failure no other guard catches: a partly-downloaded dataset comes
    back short rather than failing, and the index built from it is smaller,
    marked complete, digested, and used."""
    with pytest.raises(CP.IncompleteIndex) as exc:
        _build(expected_items={"humaneval-plus": 164})
    message = str(exc.value)
    assert "humaneval-plus" in message and "164" in message and "3" in message


def test_every_gap_is_reported_at_once():
    with pytest.raises(CP.IncompleteIndex) as exc:
        _build(loader=_loader(fail=("humaneval-plus",), empty=("mbpp-plus",)))
    assert len(exc.value.problems) == 2


def test_the_expected_counts_describe_the_benchmarks_actually_scored():
    """Pinned here so a change to a benchmark's size is a decision someone
    makes rather than a number a rebuild quietly adopts -- and tied to the eval
    harness, so a third dataset cannot be scored without reaching this index."""
    assert CP.CODE_EXPECTED_ITEMS == {"humaneval-plus": 164, "mbpp-plus": 378}
    from scripts.code_eval import DATASETS

    assert set(CP.CODE_EXPECTED_ITEMS) == set(DATASETS) == set(CP.CODE_BENCHMARKS)


# ------------------------------------------------------- what n=13 cannot do ---

def test_an_item_too_short_to_filter_on_at_all_is_refused():
    """A reference shorter than `n` whitespace tokens yields no n-grams, so that
    item is not decontaminated by anything in this index. Counting it would be
    the reassuring answer; there is no honest index that contains it."""
    stub = {"task_id": "Mbpp/999", "entry_point": "f",
            "prompt": '"""tiny"""\n', "canonical_solution": "def f(): return 1\n"}
    with pytest.raises(CP.IncompleteIndex) as exc:
        _build(loader=_loader(problems={"mbpp-plus": {"Mbpp/999": stub}}))
    message = str(exc.value)
    assert "Mbpp/999" in message and "cannot filter" in message


def test_a_short_solution_is_counted_rather_than_refused_or_hidden():
    """`return min(list1)` is three tokens that occur in every Python corpus
    ever built. It is unfilterable at 13-grams and lowering `n` to catch it
    would empty the corpus instead of cleaning it -- so the number is recorded
    and carried into the manifest, not fixed and not dropped."""
    problem = _mbpp_problem(7)
    problem["canonical_solution"] = "def find_7(x): return min(x)\n"
    _, prov = _build(loader=_loader(problems={"mbpp-plus": {"Mbpp/7": problem}}))

    meta = prov["benchmarks"]["mbpp-plus"]
    assert meta["short_fields"] == {"solution": 1}
    assert "solution" not in meta["fields"]
    # The prompt and the joined reference still cover the item.
    assert meta["fields"]["prompt"] > 0 and meta["fields"]["reference"] > 0


def test_short_field_counts_reach_the_manifest():
    problem = _mbpp_problem(7)
    problem["canonical_solution"] = "def find_7(x): return min(x)\n"
    _, prov = _build(loader=_loader(problems={"mbpp-plus": {"Mbpp/7": problem}}))

    record = CP.code_manifest_record(prov, path="data/decontam/code.gz")
    assert record["short_fields"]["mbpp-plus"] == {"solution": 1}
    assert record["digest"] == prov["digest"] and record["kind"] == "code"
    assert json.dumps(record)                 # must survive a manifest write


# ------------------------------------------------------------ the filter's n ---

def test_an_index_at_the_wrong_n_is_refused_by_coverage():
    """The silent failure with no analogue in the general index.

    `DedupState.keep` calls `is_contaminated(text, index)` at n=13 over one set,
    so a code corpus filters against `general | code`. An 8-gram code index
    loads without complaint, unions without complaint, and every lookup misses
    because the probe n-grams are a different length. Nothing anywhere reports
    a problem and the corpus manifest still names the index.
    """
    _, prov = _build(n=8)
    problems = CP.code_coverage_problems(prov, BENCHMARKS, {})
    assert any("8" in p and "13" in p for p in problems)

    _, right = _build(n=13)
    assert CP.code_coverage_problems(right, BENCHMARKS, {}) == []


def test_the_default_n_is_the_one_the_corpus_filter_looks_up():
    from daedalus.eval_index import DEFAULT_N

    assert CP.DEFAULT_CODE_N == DEFAULT_N == 13


def test_a_general_eval_index_does_not_pass_as_a_code_one():
    """Both are `eval_index` files with the same shape, so pointing
    `--code-index` at the general one is a plausible mistake that would filter
    the code corpus against no code benchmark whatsoever."""
    from daedalus.eval_index import SCHEMA

    general = {"schema": SCHEMA, "n": 13, "complete": True,
               "tasks": {"hellaswag": {"items": 10_042, "split": "validation"}}}
    problems = CP.code_coverage_problems(general, BENCHMARKS, {})
    assert any("not 'code'" in p for p in problems)
    assert any("humaneval-plus" in p for p in problems)


def test_coverage_flags_a_benchmark_that_has_since_grown():
    _, prov = _build()
    problems = CP.code_coverage_problems(prov, BENCHMARKS,
                                         {"humaneval-plus": 164})
    assert any("humaneval-plus" in p and "164" in p for p in problems)


# ---------------------------------------------------------- round tripping ---

def test_write_then_load_round_trips_the_index_and_its_provenance(tmp_path):
    ngrams, prov = _build()
    path = str(tmp_path / "code.txt.gz")
    CP.write_index(path, ngrams, prov)

    loaded, loaded_prov = CP.load_index(path)
    assert loaded == ngrams
    assert loaded_prov["digest"] == prov["digest"]
    assert loaded_prov["benchmarks"] == prov["benchmarks"]
    assert os.path.exists(CP.sidecar_path(path))


def test_loading_a_truncated_index_refuses_instead_of_filtering_less(tmp_path):
    import gzip

    ngrams, prov = _build()
    path = str(tmp_path / "code.txt.gz")
    CP.write_index(path, ngrams, prov)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(sorted(ngrams)[:5]) + "\n")

    with pytest.raises(CP.IndexDigestMismatch):
        CP.load_index(path)


def test_a_written_index_still_matches_the_benchmark_it_indexed(tmp_path):
    """The predicate has to survive the round trip, not just the set. This is
    the whole point of the artifact: a repository that copied a HumanEval+
    problem in must be dropped by `is_contaminated`, and ordinary code that
    merely looks like Python must not.
    """
    ngrams, prov = _build()
    path = str(tmp_path / "code.txt.gz")
    CP.write_index(path, ngrams, prov)
    loaded, _ = CP.load_index(path)

    problem = _humaneval_problem(1)
    copied_in = ("# solutions to the benchmark\n"
                 + problem["prompt"] + problem["canonical_solution"])
    assert is_contaminated(copied_in, loaded)
    # The assertions alone, as a harness repo would carry them.
    assert is_contaminated(problem["test"], loaded)
    # And an MBPP+ prompt lifted into a tutorial.
    assert is_contaminated(_mbpp_problem(2)["prompt"], loaded)

    ordinary = ("import os\n\n\ndef read_config(path):\n"
                "    with open(path) as handle:\n"
                "        return handle.read().splitlines()\n\n\n"
                "def main(argv=None):\n"
                "    for line in read_config(argv[1]):\n"
                "        print(line.strip())\n")
    assert not is_contaminated(ordinary, loaded)


def test_the_file_on_disk_is_a_function_of_the_set_alone(tmp_path):
    ngrams, prov = _build()
    a, b = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    CP.write_index(a, ngrams, prov)
    CP.write_index(b, sorted(ngrams, reverse=True), prov)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()


def test_ngrams_of_code_never_contain_a_newline():
    """What makes the line-delimited format lossless for code, which is mostly
    newlines and indentation. `ngram_set` splits on all whitespace."""
    grams = ngram_set(_humaneval_problem(0)["prompt"]
                      + _humaneval_problem(0)["canonical_solution"], 13)
    assert grams and all("\n" not in g and "\r" not in g for g in grams)


# ---------------------------------------------------------------- the cli ---

def _cli_build(tmp_path, monkeypatch, capsys, **kw):
    import scripts.codeprep as CLI

    monkeypatch.setattr(CLI, "build_code_index",
                        lambda **inner: _build(**{**inner, **kw}))
    # `verify` re-checks the index against the *real* benchmark sizes, which is
    # its job; the fixtures are three and four items. Only the sizes are
    # replaced, so the coverage logic under test is the shipped one.
    monkeypatch.setattr(CP, "CODE_EXPECTED_ITEMS",
                        {"humaneval-plus": 3, "mbpp-plus": 4})
    out = str(tmp_path / "code.txt.gz")
    rc = CLI._cli(["decontam", "build", "--out", out])
    return rc, out, capsys.readouterr()


def test_the_cli_builds_verifies_and_prints_the_digest_to_pin(
        tmp_path, monkeypatch, capsys):
    import scripts.codeprep as CLI

    rc, out, captured = _cli_build(tmp_path, monkeypatch, capsys)
    assert rc == 0
    prov = CP.read_provenance(out)
    assert prov["digest"] in captured.out
    assert "--code-index-digest" in captured.out

    json_out = str(tmp_path / "decontam.json")
    assert CLI._cli(["decontam", "verify", "--out", out,
                     "--expect-digest", prov["digest"],
                     "--json-out", json_out]) == 0
    with open(json_out) as f:
        assert json.load(f)["problems"] == []


def test_the_cli_verify_fails_and_still_records_why(tmp_path, monkeypatch,
                                                    capsys):
    """A failing verify that writes nothing leaves only an exit code, and the
    reason has to be reconstructed by re-running it -- which is exactly when the
    box is busy with something else."""
    import scripts.codeprep as CLI

    _, out, _ = _cli_build(tmp_path, monkeypatch, capsys, n=8)
    json_out = str(tmp_path / "decontam.json")

    assert CLI._cli(["decontam", "verify", "--out", out,
                     "--json-out", json_out]) == 3
    with open(json_out) as f:
        assert any("13" in p for p in json.load(f)["problems"])


def test_the_cli_refuses_to_verify_an_index_that_was_never_built(tmp_path,
                                                                 capsys):
    import scripts.codeprep as CLI

    rc = CLI._cli(["decontam", "verify", "--out", str(tmp_path / "absent.gz")])
    assert rc == 2
    assert "build it first" in capsys.readouterr().err


# --------------------------------------------------------- the real datasets ---

@pytest.mark.slow
def test_the_real_benchmarks_are_the_size_this_index_claims():
    """Every fixture above is written to the code. This one is written to the
    datasets, which is the half that was missing when MBPP+ turned out to ship
    no `test` key and the phase 8 baseline raised on its first problem.
    """
    pytest.importorskip("evalplus", reason="evalplus absent")
    from scripts.code_eval import load_problems

    for name, expected in CP.CODE_EXPECTED_ITEMS.items():
        assert len(load_problems(name)) == expected, name


@pytest.mark.slow
def test_every_real_item_contributes_something_this_index_can_filter_on():
    """The refusal, run against the real data rather than a stub: if any real
    HumanEval+ or MBPP+ item is unfilterable at n=13, the build must say so
    before a 65%-code corpus is built against it, not after."""
    pytest.importorskip("evalplus", reason="evalplus absent")

    ngrams, prov = CP.build_code_index(now=lambda: "2026-08-26T00:00:00Z")

    assert prov["ngrams"] == len(ngrams) > 0
    assert CP.code_coverage_problems(prov) == []
    for name, meta in prov["benchmarks"].items():
        assert meta["fields"].get("reference", 0) > 0, name
