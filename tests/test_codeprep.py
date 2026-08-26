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


# ======================================================= the admission gate ===
#
# Two properties phase 8's corpus has and the general corpus never needed: every
# document is permissively licensed, and no repository appears on both sides of
# the train/holdout split. Both are decided per row, before anything is
# tokenized, and both fail *quietly* -- a leaked repository makes the holdout
# read better, an unrecognised licence string makes the corpus undescribable,
# and neither raises. So these tests are about the quiet failures.


def _row(repo="octocat/hello-world", license_value="mit", code="print(1)\n",
         **extra) -> dict:
    """Shaped like a `codeparrot/github-code` row: the code, the repository it
    came from, and the licence GitHub reported for that repository."""
    return {"code": code, "repo_name": repo, "path": "src/main.py",
            "language": "Python", "license": license_value, **extra}


def _gate(want="train", **kw) -> "CP.RepositoryGate":
    kw.setdefault("holdout_frac", 0.25)   # coarse, so fixtures need few names
    return CP.RepositoryGate(want=want, **kw)


def _names(n: int, prefix: str = "org") -> list:
    return [f"{prefix}/repo-{i}" for i in range(n)]


# ------------------------------------------------------------- the licence ---

@pytest.mark.parametrize("license_value", sorted(CP.PERMISSIVE_LICENSES))
def test_every_permissive_licence_is_admitted(license_value):
    assert CP.license_verdict(license_value) == "permissive"


@pytest.mark.parametrize("license_value", sorted(CP.KNOWN_NON_PERMISSIVE_LICENSES))
def test_every_copyleft_licence_is_refused_by_name(license_value):
    """By name, not by falling through: a licence we know about and refuse is a
    different fact from one we have never seen, and only the second is news."""
    assert CP.license_verdict(license_value) == "non-permissive"


@pytest.mark.parametrize("license_value", [
    None,                       # the column exists and was never populated
    "",
    "   ",
    "other",                    # GitHub's own catch-all
    "elastic-2.0",              # source-available, not open source at all
    "gpl-4.0",                  # a value that does not exist yet
])
def test_an_unrecognised_licence_is_refused_and_counted_as_unknown(license_value):
    """The allow-list's whole purpose. A deny-list would admit every one of
    these, and the corpus would contain text nobody can describe the terms of.
    """
    assert CP.license_verdict(license_value) == "unknown"

    gate = _gate()
    assert gate(_row(license_value=license_value)) is False
    assert gate.refusals["unknown_license"] == 1
    assert gate.refusals["non_permissive"] == 0


def test_a_licence_is_normalised_before_it_is_looked_up():
    assert CP.license_verdict("  Apache-2.0 ") == "permissive"
    assert CP.license_verdict("MIT") == "permissive"
    # Multi-licensed rows have carried a list. Its first entry is kept whole
    # rather than joined, so what lands in the counters is something lookup-able.
    assert CP.license_verdict(["mit", "gpl-3.0"]) == "permissive"
    assert CP.normalize_license(["gpl-3.0"]) == "gpl-3.0"
    assert CP.normalize_license([]) == ""


def test_the_licence_histogram_keeps_refused_strings_verbatim():
    """The unknown counter says *how many*; only the verbatim string says what
    to do about it. Widening the allow-list on a guess is the alternative."""
    gate = _gate()
    for value in ("mit", "gpl-3.0", "gpl-3.0", "wtfpl"):
        gate(_row(repo="org/only", license_value=value))

    assert gate.manifest()["licenses"] == {"gpl-3.0": 2, "mit": 1, "wtfpl": 1}


# ---------------------------------------------------------- the repository ---

@pytest.mark.parametrize("key", CP.REPOSITORY_FIELDS)
def test_a_repository_is_found_under_every_field_that_has_carried_one(key):
    row = {"code": "x", "license": "mit", key: "Org/Name"}
    assert CP.repository_of(row) == "org/name"
    assert CP.repository_field_of(row) == key


def test_a_row_with_no_repository_is_refused_by_both_sides():
    """The only answer that keeps the two passes disjoint. Defaulting an
    unidentifiable row to a side means both passes default it the same way and
    the same document enters train *and* holdout -- the exact leak the split
    exists to prevent."""
    row = {"code": "print(1)", "license": "mit"}
    assert CP.repository_of(row) is None

    for want in CP.SPLITS:
        gate = _gate(want=want)
        assert gate(row) is False
        assert gate.refusals["no_repository"] == 1


def test_repository_case_is_not_part_of_its_identity():
    """GitHub treats `Owner/Repo` and `owner/repo` as one project; blake2b does
    not. Unlowercased, the two spellings hash to different buckets and one
    repository lands on both sides."""
    assert CP.repository_of(_row(repo="Octocat/Hello-World")) == \
        CP.repository_of(_row(repo="octocat/hello-world"))
    assert CP.repository_split("Torvalds/Linux".lower()) == \
        CP.repository_split("torvalds/linux")


# --------------------------------------------------------------- the split ---

def test_the_split_is_a_pure_function_of_the_name():
    names = _names(500)
    first = [CP.repository_split(n) for n in names]
    assert [CP.repository_split(n) for n in reversed(names)] == list(reversed(first))
    assert [CP.repository_split(n) for n in names] == first


def test_the_split_survives_a_different_interpreter_hash_seed():
    """The failure this pins is invisible in-process. `hash()` is randomized per
    interpreter, so a `hash()`-based split re-partitions every repository the
    moment a long build restarts -- and the resulting train/holdout overlap
    shows up only as a holdout that reads better than it should."""
    import subprocess
    import sys

    names = _names(64)
    program = (
        "import sys; sys.path.insert(0, %r);"
        "from daedalus.codeprep import repository_split;"
        "print(','.join(repository_split(n) for n in %r))"
        % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), names)
    )
    here = ",".join(CP.repository_split(n) for n in names)
    for seed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", program], check=True,
                             capture_output=True, text=True,
                             env={**os.environ, "PYTHONHASHSEED": seed})
        assert out.stdout.strip() == here, seed


def test_two_gates_partition_every_repository_exactly_once():
    """What makes the holdout a second independent stream rather than a second
    reading of the first."""
    train, holdout = _gate("train"), _gate("holdout")
    rows = [_row(repo=name) for name in _names(400)]

    admitted = {"train": {r["repo_name"] for r in rows if train(r)},
                "holdout": {r["repo_name"] for r in rows if holdout(r)}}

    assert admitted["train"] & admitted["holdout"] == set()
    assert admitted["train"] | admitted["holdout"] == {r["repo_name"] for r in rows}
    assert admitted["holdout"], "a holdout of nothing cannot be measured"


def test_a_repositorys_every_file_lands_on_one_side():
    """The property the plan asks for in its own words: split by repository, not
    by file or packed window. Two files from one project share helpers, idioms
    and licence headers, so one leaked repository inflates the holdout score the
    way training on it would."""
    files = [_row(repo="org/one", path=f"src/{i}.py") for i in range(50)]
    train, holdout = _gate("train"), _gate("holdout")

    verdicts = {"train": {train(f) for f in files},
                "holdout": {holdout(f) for f in files}}

    for side, calls in verdicts.items():
        assert calls in ({True}, {False}), f"{side} split one repository's files"
    # ...and it is on exactly one of them, not neither.
    assert verdicts["train"] != verdicts["holdout"]


def test_the_realised_holdout_is_near_the_fraction_asked_for():
    """A hash that clumps would satisfy every disjointness test above and still
    starve the holdout to nothing, or hand it a quarter of the corpus."""
    names = _names(20_000)
    for frac in (0.02, 0.1, 0.25):
        held = sum(CP.repository_split(n, holdout_frac=frac) == "holdout"
                   for n in names)
        assert abs(held / len(names) - frac) < 0.01, frac


def test_a_different_salt_is_a_different_split():
    """So a manifest that records the outcome without the salt records something
    nobody can re-derive."""
    names = _names(200)
    ours = [CP.repository_split(n) for n in names]
    theirs = [CP.repository_split(n, salt="something-else") for n in names]
    assert ours != theirs


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 2, 100])
def test_a_holdout_fraction_outside_the_unit_interval_is_refused(frac):
    """`2` for "2%" is the mistake, and it costs a whole build to find later."""
    with pytest.raises(ValueError, match="holdout_frac"):
        CP.repository_split("org/repo", holdout_frac=frac)
    with pytest.raises(ValueError, match="holdout_frac"):
        CP.RepositoryGate(want="train", holdout_frac=frac)


def test_an_unknown_split_side_is_refused_at_construction():
    with pytest.raises(ValueError, match="want"):
        CP.RepositoryGate(want="validation")


def test_split_is_disjoint_re_derives_both_sides_from_the_manifests_parameters():
    names = _names(300)
    audit = CP.split_is_disjoint(names, holdout_frac=0.25)

    assert audit["overlap"] == []
    assert sorted(audit["train"] + audit["holdout"]) == sorted(names)
    assert audit["split_salt"] == CP.SPLIT_SALT


# -------------------------------------------------------------- the record ---

def test_the_manifest_carries_every_refusal_reason_even_at_zero():
    """An absent counter and a zero one read identically, and "this build
    refused nothing for licence reasons" is a claim worth being able to make."""
    manifest = _gate().manifest()
    assert set(manifest["refused"]) == set(CP.REFUSAL_REASONS)
    assert all(count == 0 for count in manifest["refused"].values())


def test_the_manifest_records_what_the_split_cannot_be_re_derived_without():
    gate = _gate(want="holdout", holdout_frac=0.05)
    manifest = gate.manifest()

    assert manifest["split"] == "holdout"
    assert manifest["holdout_frac"] == 0.05
    assert manifest["split_salt"] == CP.SPLIT_SALT
    assert manifest["permissive_licenses"] == sorted(CP.PERMISSIVE_LICENSES)


def test_the_gate_counts_every_row_it_saw_and_the_repositories_it_admitted():
    gate = _gate("train")
    rows = ([_row(repo=n) for n in _names(100)]
            + [_row(repo="org/copyleft", license_value="gpl-3.0"),
               _row(repo="org/mystery", license_value="wtfpl"),
               {"code": "x", "license": "mit"}])

    kept = [r for r in rows if gate(r)]
    manifest = gate.manifest()

    assert manifest["rows_seen"] == len(rows)
    assert manifest["rows_admitted"] == len(kept)
    assert manifest["refused"]["non_permissive"] == 1
    assert manifest["refused"]["unknown_license"] == 1
    assert manifest["refused"]["no_repository"] == 1
    assert manifest["repositories"] == len(gate.repositories) == len(
        {r["repo_name"] for r in kept})
    assert manifest["repository_fields"] == {"repo_name": len(rows) - 1}


def test_a_refused_repository_is_not_in_the_shard_directorys_manifest():
    gate = _gate("train")
    gate(_row(repo="org/copyleft", license_value="gpl-3.0"))
    assert gate.repositories == []


def test_the_repository_list_is_bounded_and_says_when_it_stopped_recording():
    """Phase 8's code sources stream repositories in the millions. An unbounded
    set of names inside a `dataprep` worker is the growth its RSS caps exist to
    catch -- a provenance record that OOMs the build is a bad trade."""
    gate = CP.RepositoryGate(want="train", holdout_frac=0.25, max_repositories=10)
    admitted = [r["repo_name"] for r in
                (_row(repo=n) for n in _names(2_000)) if gate(r)]
    manifest = gate.manifest()

    assert len(manifest["repository_names"]) == 10 < len(admitted)
    assert manifest["repositories_truncated"] is True
    # The enumeration stops; the counts do not.
    assert manifest["rows_admitted"] == len(admitted)
    assert manifest["rows_seen"] == 2_000


def test_a_gate_under_its_cap_does_not_claim_to_be_truncated():
    gate = CP.RepositoryGate(want="train", holdout_frac=0.25, max_repositories=10)
    for name in _names(3):
        gate(_row(repo=name))
    assert gate.manifest()["repositories_truncated"] is False


# ------------------------------------------------- wired the way it is used ---

def test_the_gate_works_as_the_row_filter_dataprep_actually_calls(monkeypatch):
    """Pinned against `dataprep`'s own document stream rather than by calling
    the gate directly, because that is the seam the build uses: `filter_fn` is
    consulted before `text_fn`, so a gate that refused after tokenizing, or one
    whose signature did not match, would pass every test above and still admit
    the wrong text here."""
    from daedalus import dataprep as DP

    rows = [_row(repo="org/keep", license_value="mit", code="KEEP " * 100),
            _row(repo="org/drop", license_value="gpl-3.0", code="DROP " * 100),
            {"code": "NOREPO " * 100, "license": "mit"}]
    monkeypatch.setattr(DP, "_stream_rows", lambda s, *a, **k: iter(rows))

    gate = CP.RepositoryGate(want=CP.repository_split("org/keep"),
                             holdout_frac=CP.DEFAULT_HOLDOUT_FRAC)
    spec = DP.SourceSpec("code-python", "codeparrot/github-code", share=1.0,
                         text_fn=lambda r: r.get("code") or "", filter_fn=gate)

    documents = list(DP._documents(spec, max_docs=None))

    assert [d.split()[0] for d in documents] == ["KEEP"]
    assert gate.manifest()["rows_seen"] == 3
