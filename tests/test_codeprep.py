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


# ---------------------------------------------------------------- the probe ---
#
# Every row-shape assumption in the gate is a guess about a dataset until a row
# of it has been read, and each guess fails in the same silent direction: the
# gate refuses everything and the build writes an empty shard directory with a
# zero exit. The probe is what turns that into a two-minute answer, so these
# tests are about whether the probe would actually *notice*.


def _stream(rows, fail=()):
    def stream(config):
        if config in fail:
            raise FileNotFoundError(f"no such config {config!r}")
        return iter(rows(config) if callable(rows) else rows)
    return stream


def _github_rows(n=50, license_values=("mit", "gpl-3.0"), repo_key="repo_name"):
    return [{"code": f"print({i})\n" * 20, "path": f"src/{i}.py",
             "language": "Python", "size": 100 + i,
             repo_key: f"org-{i % 17}/repo-{i % 17}",
             "license": license_values[i % len(license_values)]}
            for i in range(n)]


def test_the_probe_reports_the_columns_and_field_a_real_row_actually_has():
    record = CP.probe_source("Python-all", rows=50,
                             stream=_stream(_github_rows()))

    assert record["resolved"] is True and record["rows_read"] == 50
    assert set(record["columns"]) == {"code", "path", "language", "size",
                                      "repo_name", "license"}
    assert record["repository_fields"] == {"repo_name": 50}
    assert record["licenses"] == {"mit": 25, "gpl-3.0": 25}
    assert record["refused"]["non_permissive"] == 25
    assert record["admitted"]["train"] + record["admitted"]["holdout"] == 25


def test_the_probe_names_the_failure_where_no_row_survives():
    """The one the build cannot tell from success. Every row arrives, every row
    is refused, and the shard directory is empty with a zero exit."""
    rows = _github_rows(repo_key="repository")      # a key the gate does not read
    record = CP.probe_source("Python-all", rows=50, stream=_stream(rows))

    assert record["admitted"] == {"train": 0, "holdout": 0}
    assert "admitted none" in record["problem"]
    assert record["refused"]["no_repository"] == 50


def test_a_config_that_does_not_resolve_is_reported_not_raised():
    """`GO-all` versus `Go-all`, or a `C++` that a path escapes. One misspelled
    directory must not hide the six that were right."""
    report = CP.probe_languages(["go", "java"], rows=10,
                                stream=_stream(_github_rows(10), fail=("GO-all",)))

    go = report["languages"]["go"]["configs"][0]
    java = report["languages"]["java"]["configs"][0]
    assert go["resolved"] is False and "GO-all" in go["error"]
    assert java["resolved"] is True
    assert any("go/GO-all did not resolve" in p for p in CP.probe_problems(report))


def test_the_probe_surfaces_licence_strings_the_gate_cannot_classify():
    """Refusing an unknown licence is the safe direction, and silently refusing
    most of a language is still how a bucket comes back empty."""
    rows = _github_rows(20, license_values=("mit", "wtfpl", "elastic-2.0"))
    report = CP.probe_languages(["python"], rows=20, stream=_stream(rows))

    problems = CP.probe_problems(report)
    assert any("does not classify" in p and "wtfpl" in p for p in problems)


def test_a_clean_probe_has_nothing_to_report():
    report = CP.probe_languages(["python"], rows=30,
                                stream=_stream(_github_rows(30, ("mit", "apache-2.0"))))
    assert CP.probe_problems(report) == []


def test_the_probe_covers_every_bucket_the_dataset_has_a_directory_for():
    """Every bucket, not only the ones the mixture still asks for: the probe
    measures the dataset, and the four buckets the user dropped by choice are
    the ones whose evidence has to stay reproducible after they leave the
    mixture. They are reported at share 0.0 rather than omitted."""
    report = CP.probe_languages(rows=5, stream=_stream(_github_rows(5)))

    assert set(report["languages"]) == set(CP.GITHUB_CODE_LANGUAGES)
    assert sum(e["share"] for e in report["languages"].values()) == pytest.approx(1.0)
    assert report["languages"]["go"]["share"] == 0.0
    assert report["languages"]["python"]["share"] == 0.55


def test_an_unknown_bucket_is_refused_rather_than_silently_skipped():
    with pytest.raises(ValueError, match="unknown code bucket"):
        CP.probe_languages(["cobol"], stream=_stream([]))


def test_the_probe_reports_which_languages_a_directory_actually_contains():
    """A directory's name says what it should hold; only the histogram says what
    it does. With four buckets' directories missing entirely, whether the
    interleaved `all-all` reaches a language at a usable rate is the question
    that decides the mixture."""
    rows = ([{"code": "x" * 50, "repo_name": f"o/r{i}", "license": "mit",
              "language": "Go"} for i in range(3)]
            + [{"code": "x" * 50, "repo_name": f"o/p{i}", "license": "mit",
                "language": "Python"} for i in range(17)])
    record = CP.probe_source("all-all", rows=20, stream=_stream(rows))

    assert record["languages"] == {"Python": 17, "Go": 3}


# ------------------------------------------ reading a bucket out of all-all ---
#
# Rust, Go, Shell and SQL -- 18% of the plan's code mixture -- have no directory
# on the pinned revision. The interleaved `all-all` carries them, and everything
# else besides, so a language filter is the only way to read one of those buckets
# at all. Whether that is *affordable* is a separate question, and the only
# honest answer to it is a measured rate.


def _mixed_rows(n=100, rust_every=10):
    """`all-all` in miniature: mostly Python, a little Rust, and copyleft rows
    interleaved so the licence gate has something to refuse in both."""
    return [{"code": "x" * (100 if i % rust_every else 400),
             "repo_name": f"org-{i}/repo-{i}",
             "license": "mit" if i % 3 else "gpl-3.0",
             "language": "Rust" if i % rust_every == 0 else "Python"}
            for i in range(n)]


def test_a_bucket_with_no_directory_is_read_out_of_the_interleaved_one():
    record = CP.probe_source("all-all", rows=100, languages=["Rust"],
                             stream=_stream(_mixed_rows()))

    kept = record["admitted"]["train"] + record["admitted"]["holdout"]
    assert 0 < kept <= 10
    assert record["languages_kept"] == ["rust"]
    # Every non-Rust row refused on language, and none of them counted against
    # the licence or repository vocabulary.
    assert record["refused"]["other_language"] == 90
    assert record["refused"]["no_repository"] == 0


def test_the_language_a_row_says_is_matched_however_the_dataset_spells_it():
    """`GO` is upper-case in this dataset's own vocabulary where `Rust` is not.
    An exact-match filter against the plan's spelling would refuse every row --
    the same failure the `GO-all` directory name already caused once, in a place
    where nothing raises."""
    rows = [{"code": "x" * 50, "repo_name": f"o/r{i}", "license": "mit",
             "language": "GO"} for i in range(8)]

    record = CP.probe_source("all-all", rows=8, languages=["Go"],
                             stream=_stream(rows))

    assert record["admitted"]["train"] + record["admitted"]["holdout"] == 8


def test_a_filtered_gate_manifests_its_own_languages_licences_not_the_directorys():
    """The shard is drawn from one language; a manifest that reported the whole
    interleaved directory's licence mix would describe a corpus that was never
    built."""
    rows = ([{"code": "x", "repo_name": f"o/r{i}", "license": "mit",
              "language": "Rust"} for i in range(5)]
            + [{"code": "x", "repo_name": f"o/p{i}", "license": "wtfpl",
                "language": "Python"} for i in range(20)])
    gate = CP.RepositoryGate(want="train", languages=["rust"])

    for row in rows:
        gate(row)

    manifest = gate.manifest()
    assert manifest["licenses"] == {"mit": 5}
    assert manifest["languages"] == ["rust"]
    assert manifest["rows_seen"] == 25 and manifest["refused"]["other_language"] == 20


def test_an_unfiltered_gate_still_says_so_in_its_manifest():
    assert CP.RepositoryGate(want="train").manifest()["languages"] is None


def test_a_language_allow_list_that_names_nothing_is_refused():
    """An empty allow-list refuses every row, which on a corpus build is
    indistinguishable from a language with no permissive code in it."""
    with pytest.raises(ValueError, match="pass None to keep every language"):
        CP.RepositoryGate(want="train", languages=[])


def test_the_probe_reports_what_streaming_a_rare_language_actually_costs():
    """The number the mixture decision divides by. A language reachable in
    principle is still unaffordable if a token of it costs a hundred rows of
    something else, and that rate cannot be read off a histogram taken over rows
    the licence gate had not yet refused."""
    record = CP.probe_source("all-all", rows=100, languages=["Rust"],
                             stream=_stream(_mixed_rows()))

    kept = record["admitted"]["train"] + record["admitted"]["holdout"]
    assert record["stream_amplification"] == pytest.approx(100 / kept, abs=0.01)
    assert sum(record["admitted_bytes"].values()) == 400 * kept


def test_one_pass_over_the_interleaved_directory_sizes_every_bucket_in_it():
    """Four buckets are drawn from `all-all`. Measuring them with four filtered
    probes would stream the same rows four times, and the rows are what costs;
    the gate's own per-language yield answers all four from one pass -- and
    answers it *after* the licence gate, which is the only side that can be
    budgeted from."""
    record = CP.probe_source("all-all", rows=100,
                             stream=_stream(_mixed_rows()))

    admitted = record["admitted_languages"]
    assert set(admitted) == {"python", "rust"}
    # Rows offered versus rows kept: two thirds of each language survive the
    # licence gate here, so the seen histogram alone would overcount both.
    assert record["languages"] == {"Python": 90, "Rust": 10}
    assert admitted["rust"]["rows"] < 10
    assert admitted["rust"]["bytes"] == 400 * admitted["rust"]["rows"]
    assert admitted["python"]["bytes"] == 100 * admitted["python"]["rows"]
    assert sum(entry["rows"] for entry in admitted.values()) == \
        record["admitted"]["train"] + record["admitted"]["holdout"]


def test_a_language_the_directory_does_not_carry_is_named_not_guessed_at():
    """Refusing every row looks identical whether the language is absent or its
    name is spelled the way the plan spells it. The counter distinguishes
    them."""
    record = CP.probe_source("all-all", rows=100, languages=["Cobol"],
                             stream=_stream(_mixed_rows()))

    assert record["admitted"] == {"train": 0, "holdout": 0}
    assert record["stream_amplification"] is None
    assert "no row of cobol survived" in record["problem"]
    assert "100 were refused on language alone" in record["problem"]
    assert any("all-all" in p for p in CP.probe_problems(
        {"languages": {"unbucketed": {"configs": [record]}}}))


def test_a_language_filter_over_a_buckets_own_directory_is_refused():
    """`Python-all` is already Python. A filter over it can only refuse rows it
    should have kept, and would do so silently."""
    with pytest.raises(ValueError, match="already the language"):
        CP.probe_languages(["python"], keep_languages=["Python"],
                           stream=_stream(_github_rows(5)))


def test_the_cli_probes_the_interleaved_directory_for_one_language(
        tmp_path, monkeypatch, capsys):
    import scripts.codeprep as CLI

    monkeypatch.setattr(
        CLI, "probe_languages",
        lambda languages, **kw: CP.probe_languages(
            languages, stream=_stream(_mixed_rows()), **kw))
    out = str(tmp_path / "probe.json")

    rc = CLI._cli(["corpus", "probe", "--config", "all-all",
                   "--keep-language", "Rust", "--rows", "100",
                   "--json-out", out])

    assert rc == 0
    assert "rows streamed per row kept" in capsys.readouterr().out
    written = json.load(open(out))
    record = written["languages"]["unbucketed"]["configs"][0]
    assert record["languages_kept"] == ["rust"]
    assert record["stream_amplification"] > 10


def test_a_named_directory_can_be_probed_without_being_given_a_share_first():
    report = CP.probe_languages(configs=["all-all"], rows=5,
                                stream=_stream(_github_rows(5)))

    assert set(report["languages"]) == {"unbucketed"}
    assert report["languages"]["unbucketed"]["configs"][0]["config"] == "all-all"
    assert report["languages"]["unbucketed"]["share"] == 0.0


def test_the_cli_refuses_to_probe_buckets_and_directories_at_once(capsys):
    import scripts.codeprep as CLI

    rc = CLI._cli(["corpus", "probe", "--config", "all-all",
                   "--language", "python"])

    assert rc == 2
    assert "one or the other" in capsys.readouterr().err


def test_the_probe_stops_at_the_row_count_it_was_given():
    """It runs before every build, so it has to stay a two-minute answer."""
    record = CP.probe_source("Python-all", rows=7,
                             stream=_stream(_github_rows(10_000)))
    assert record["rows_read"] == 7


# --------------------------------------------------------- the source plan ---

#: The revision as `corpus configs` found it: every directory carries ten
#: parquet files, and Go, Rust, Shell and SQL have none at all.
AVAILABLE = {"Python-all": 10, "JavaScript-all": 10, "TypeScript-all": 10,
             "C-all": 10, "C++-all": 10, "Java-all": 10, "all-all": 10}

#: The 200,000-row `all-all` yield in miniature, keeping the measured shape:
#: Rust is a third of a percent of what the licence gate admits, Go under three,
#: Shell and SQL together one -- against plan shares of 8%, 6% and 4%.
_ALLALL_BYTES = {"python": 5_500_000, "javascript": 1_000_000,
                 "typescript": 200_000, "c": 700_000, "c++": 300_000,
                 "java": 500_000, "go": 280_000, "shell": 60_000, "sql": 40_000,
                 "rust": 30_000, "html": 1_390_000}


def _allall(bytes_by_language=None, **overrides) -> dict:
    """A `probe_source` record for the interleaved directory."""
    measured = dict(_ALLALL_BYTES if bytes_by_language is None
                    else bytes_by_language)
    record = {
        "config": "all-all", "resolved": True, "rows_read": 200_000,
        "stream_amplification": 1.54,
        "admitted_languages": {name: {"rows": max(1, count // 1000),
                                      "bytes": count}
                               for name, count in measured.items()},
    }
    record.update(overrides)
    return record


def _plan(**kwargs) -> dict:
    """A plan over the *preregistered* seven buckets.

    These tests are about the fallback and drop logic -- what the interleaved
    directory can serve, and what it cannot -- and that logic only has anything
    to decide when the mixture asks for buckets with no directory of their own.
    The live `CODE_LANGUAGE_SHARES` no longer does: the user retargeted the code
    portion to Python and JavaScript/TypeScript, both of which have directories.
    Pinning these to the preregistered table keeps the measurement that proved
    rust unreachable exercised rather than deleted along with the buckets.
    """
    kwargs.setdefault("shares", CP.PREREGISTERED_CODE_LANGUAGE_SHARES)
    kwargs.setdefault("available", AVAILABLE)
    kwargs.setdefault("interleaved", _allall())
    return CP.source_plan(**kwargs)


def test_a_bucket_whose_directories_all_exist_is_served_from_them():
    plan = _plan()

    entry = plan["buckets"]["javascript-typescript"]
    assert entry["source"] == "directories"
    assert entry["configs"] == ["JavaScript-all", "TypeScript-all"]
    assert entry["missing_configs"] == []


def test_a_bucket_with_no_directory_is_capped_at_what_a_pass_actually_yields():
    """Go is reachable out of the interleaved directory and its 6% is not. A
    plan that kept the 6% would be a build that streams to exhaustion and comes
    up short, discovering here what this function is for."""
    plan = _plan()

    entry = plan["buckets"]["go"]
    assert entry["source"] == "interleaved"
    assert entry["source_config"] == "all-all"
    assert entry["languages"] == ["go"]
    assert entry["plan_share"] == 0.06
    assert entry["share"] == pytest.approx(0.028, abs=1e-6)
    # The budget that would have served it in full, recorded so the drop is a
    # decision about a price rather than a claim about the rows.
    assert entry["required_passes"] == pytest.approx(0.06 / 0.028, abs=1e-6)


def test_a_bucket_too_rare_to_reach_a_usable_share_is_dropped_by_name():
    """0.3% of the code mixture is a few million tokens of Rust: too little to
    teach it and enough for a model card to claim it."""
    plan = _plan()

    entry = plan["buckets"]["rust"]
    assert entry["source"] == "dropped"
    assert entry["share"] == 0.0
    assert entry["reachable_share"] == pytest.approx(0.003, abs=1e-6)
    assert entry["required_passes"] == pytest.approx(0.08 / 0.003, abs=1e-3)
    assert "under the 0.5% floor" in entry["reason"]
    assert "rust" not in plan["shares"]


def test_what_the_fallback_cannot_serve_is_redistributed_over_the_directories():
    """Following the general corpus's gated substitutions: the shortfall goes to
    the buckets that have a source, proportionally to their plan shares. It
    cannot go to the capped ones -- they are capped because the rows are not
    there."""
    plan = _plan()

    shortfall = 0.08 + (0.06 - 0.028) + (0.04 - 0.010)
    assert plan["redistributed"] == pytest.approx(shortfall, abs=1e-6)
    python, java = plan["buckets"]["python"], plan["buckets"]["java"]
    assert python["redistributed"] == pytest.approx(
        shortfall * 0.55 / 0.82, abs=1e-6)
    assert python["share"] == pytest.approx(0.55 + shortfall * 0.55 / 0.82,
                                            abs=1e-6)
    # Proportional, so the ratio the plan set between two served buckets is the
    # ratio it keeps -- to within the six decimal places the shares are rounded
    # to, which is 0.0001% of the mixture.
    assert python["share"] / java["share"] == pytest.approx(0.55 / 0.05,
                                                            rel=1e-4)
    assert sum(plan["shares"].values()) == pytest.approx(1.0, abs=1e-6)
    assert CP.plan_problems(plan) == []


def test_the_budget_is_a_parameter_so_one_measurement_serves_any_of_them():
    """Nothing here is unreachable in principle, only at a price. At a budget
    that pays Rust's price every bucket is served in full and nothing is
    redistributed -- from the same rows, without reading another."""
    plan = _plan(passes=27.0)

    assert plan["buckets"]["rust"]["source"] == "interleaved"
    assert plan["shares"] == {
        bucket: pytest.approx(share, abs=1e-6)
        for bucket, share in CP.PREREGISTERED_CODE_LANGUAGE_SHARES.items()}
    assert plan["redistributed"] == pytest.approx(0.0, abs=1e-9)


def test_a_fallback_language_the_directory_carries_none_of_is_a_problem():
    """A bucket at zero looks the same whether the language is absent or its
    name is spelled the way the plan spells it -- the failure that already cost
    four directories. A share of 0.0 would file both as a decision."""
    without_rust = {name: count for name, count in _ALLALL_BYTES.items()
                    if name != "rust"}

    plan = _plan(interleaved=_allall(without_rust))

    assert plan["buckets"]["rust"]["interleaved_rows"] == 0
    problems = CP.plan_problems(plan)
    assert any("check the spelling" in problem for problem in problems)
    assert not any("check the spelling" in problem
                   for problem in CP.plan_problems(_plan()))


def test_a_plan_whose_fallback_directory_is_missing_too_is_a_problem():
    plan = _plan(available={name: 10 for name in AVAILABLE if name != "all-all"})

    assert any("does not carry that directory" in problem
               for problem in CP.plan_problems(plan))


def test_the_fallback_share_is_measured_after_the_licence_gate_not_before():
    """`admitted_languages` counts rows kept; `languages` counts rows offered.
    The gate refuses about a third of this dataset and nothing makes it refuse
    at the same rate in every language, so budgeting from the offered histogram
    overstates every fallback bucket, unevenly."""
    offered_only = {"config": "all-all", "rows_read": 200_000,
                    "languages": {"Rust": 597, "GO": 3654}}

    with pytest.raises(ValueError, match="no per-language yield"):
        _plan(interleaved=offered_only)


def test_the_plan_reads_the_directory_out_of_a_report_probed_either_way():
    """The interleaved directory was probed with `--config`, which files it
    under `unbucketed` rather than under a bucket. The plan should not have to
    know which of the two ways its own evidence was gathered."""
    report = {"languages": {"unbucketed": {"configs": [_allall()]}}}

    assert CP.probe_record(report, "all-all")["rows_read"] == 200_000
    assert CP.probe_record(report, "Rust-all") is None


def test_the_cli_plans_from_the_two_measurements_already_on_disk(
        tmp_path, capsys):
    import scripts.codeprep as CLI

    configs_json = tmp_path / "configs.json"
    configs_json.write_text(json.dumps({"available": AVAILABLE}))
    probe_json = tmp_path / "probe.json"
    probe_json.write_text(json.dumps(
        {"languages": {"unbucketed": {"configs": [_allall()]}}}))
    out = str(tmp_path / "plan.json")

    rc = CLI._cli(["corpus", "plan", "--configs-json", str(configs_json),
                   "--probe-json", str(probe_json), "--json-out", out])

    assert rc == 0
    written = json.load(open(out))
    # The retargeted mixture: two buckets, both served from their own
    # directories, so nothing falls back and nothing is redistributed. The
    # fallback and drop logic is still exercised against the preregistered
    # seven-bucket table above -- see `_plan`.
    assert set(written["shares"]) == {"python", "javascript-typescript"}
    assert all(entry["source"] == "directories"
               for entry in written["buckets"].values())
    assert written["redistributed"] == pytest.approx(0.0, abs=1e-6)
    assert sum(written["shares"].values()) == pytest.approx(1.0, abs=1e-6)
    printed = capsys.readouterr().out
    assert "python" in printed and "javascript-typescript" in printed


# ------------------------------------------------------------ the supply ---

def _directory_probe(config, *, rows_read=2_000, admitted_bytes=8_000_000):
    """A `probe_source` record for one per-language directory."""
    return {"config": config, "resolved": True, "rows_read": rows_read,
            "admitted_bytes": {"train": admitted_bytes, "holdout": 0}}


def _supply_inputs(**row_overrides):
    plan = _plan()
    probes = {config: _directory_probe(config)
              for config in ("Python-all", "JavaScript-all", "TypeScript-all",
                             "C-all", "C++-all", "Java-all")}
    probes["all-all"] = _allall()
    rows = {config: {"files": 10, "files_read": 10, "rows": 500_000,
                     "errors": {}}
            for config in list(probes)}
    # The interleaved directory carries thirty languages, so it holds far more
    # rows than any one of them -- and a fallback bucket's supply is a slice of
    # it, not the whole thing.
    rows["all-all"] = {"files": 10, "files_read": 10, "rows": 20_000_000,
                       "errors": {}}
    rows.update(row_overrides)
    return plan, rows, probes


def test_a_buckets_supply_is_the_rate_it_admits_bytes_times_the_rows_there():
    """Two measured factors, no assumed ones: a probe gives the rate over a few
    thousand real rows, the footers give how many rows are actually there."""
    plan, rows, probes = _supply_inputs()

    supply = CP.bucket_supply(plan=plan, row_counts=rows, probes=probes,
                              bytes_per_token=2.862)

    # 8,000,000 admitted bytes per 2,000 rows read, over 500,000 rows.
    expected_bytes = 8_000_000 / 2_000 * 500_000
    assert supply["python"]["unique_bytes"] == expected_bytes
    assert supply["python"]["unique_tokens"] == int(expected_bytes / 2.862)
    assert "500,000 rows of Python-all" in supply["python"]["basis"]
    # Two directories, summed.
    assert supply["javascript-typescript"]["unique_bytes"] == 2 * expected_bytes


def test_an_interleaved_buckets_supply_counts_only_its_own_languages():
    """Go is 2.8% of the interleaved directory. A supply that credited it with
    the whole directory's admitted bytes would say the bucket has 35 times the
    tokens it has."""
    plan, rows, probes = _supply_inputs()

    supply = CP.bucket_supply(plan=plan, row_counts=rows, probes=probes)

    interleaved = supply["go"]
    assert interleaved["configs"] == ["all-all"]
    # 280,000 of the interleaved directory's 10,000,000 admitted bytes.
    assert interleaved["unique_bytes"] == int(280_000 / 200_000 * 20_000_000)
    assert supply["rust"]["unique_tokens"] == 0
    assert "not drawn from this revision" in supply["rust"]["basis"]


def test_a_fallback_capped_by_rate_can_still_be_short_of_rows():
    """`source_plan` caps Go at the *rate* one pass yields it. Whether the
    directory holds enough rows to deliver even that reduced share is a second
    question, and the plan cannot answer it -- a rate says nothing about how
    long the stream is."""
    from scripts.source_headroom import SHORT, Supply, epoch_curve

    plan, rows, probes = _supply_inputs(
        **{"all-all": {"files": 10, "files_read": 10, "rows": 400_000,
                       "errors": {}}})

    supply = CP.bucket_supply(plan=plan, row_counts=rows, probes=probes)
    curve = epoch_curve(
        {bucket: Supply(key=bucket, unique_tokens=entry["unique_tokens"],
                        realized_tokens=0, basis=entry["basis"])
         for bucket, entry in supply.items()},
        [(bucket, entry["share"], 1) for bucket, entry in supply.items()
         if entry["share"] > 0],
        budgets=[int(1_000_000_000 * CP.CORPUS_SHARES["code"])])

    totals = curve[0]["totals"]
    assert totals["verdict"] == SHORT
    assert totals["binding_source"] in ("go", "shell-sql-other")
    assert supply["python"]["unique_tokens"] > 0


def test_a_directory_counted_from_some_of_its_files_is_a_floor_not_a_number():
    """One 429 that outlasted its retries took the tenth file of `all-all` with
    it. The nine that were read are a floor, and a floor that *fails* the epoch
    cap may be failing on the file nobody counted -- so the two have to be
    distinguishable at the point the verdict is read."""
    plan, rows, probes = _supply_inputs(
        **{"all-all": {"files": 10, "files_read": 9, "rows": 547_000,
                       "errors": {"all-all/partial-train/0009.parquet": "429"}}})

    supply = CP.bucket_supply(plan=plan, row_counts=rows, probes=probes)

    assert supply["go"]["lower_bound"] is True
    assert supply["go"]["partial"] == ["all-all"]
    assert supply["go"]["basis"].startswith("lower bound -- ")
    assert "(9 of 10 files read)" in supply["go"]["basis"]
    # A directory read in full is not hedged.
    assert supply["python"]["lower_bound"] is False
    assert "of 10 files read" not in supply["python"]["basis"]


def test_a_bucket_missing_either_factor_reads_as_no_supply_not_as_a_number():
    """A directory whose footers could not be read is not a directory of zero
    rows, and the difference decides whether a build launches."""
    plan, rows, probes = _supply_inputs(
        **{"Python-all": {"files": 10, "files_read": 0, "rows": None,
                          "errors": {"Python-all/partial-train/0000.parquet":
                                     "HTTPError(504)"}}})

    supply = CP.bucket_supply(plan=plan, row_counts=rows, probes=probes)

    assert supply["python"]["unique_tokens"] == 0
    assert supply["python"]["missing"] == ["Python-all"]
    assert "unmeasured: Python-all" in supply["python"]["basis"]


def test_the_row_count_is_read_from_footers_and_a_bad_file_is_named():
    """Ten range requests rather than a pass over gigabytes -- and a file that
    could not be read is left out of the total instead of counted as empty."""
    listed = [f"Python-all/partial-train/{i:04d}.parquet" for i in range(3)]
    listed += ["Python-all/README.md", "Java-all/partial-train/0000.parquet"]

    class _Api:
        def list_repo_files(self, dataset, repo_type=None, revision=None):
            return listed

    def count(url):
        if url.endswith("0002.parquet"):
            raise OSError("range request refused")
        return 40_000

    counts = CP.config_row_counts(["Python-all", "Java-all"], api=_Api(),
                                  count_rows=count)

    assert counts["Python-all"] == {"files": 3, "files_read": 2, "rows": 80_000,
                                    "errors": {listed[2]: repr(OSError(
                                        "range request refused"))}}
    assert counts["Java-all"]["rows"] == 40_000


def test_a_directory_whose_every_footer_failed_is_unknown_not_empty():
    class _Api:
        def list_repo_files(self, dataset, repo_type=None, revision=None):
            return ["Python-all/partial-train/0000.parquet"]

    def count(url):
        raise OSError("no")

    counts = CP.config_row_counts(["Python-all"], api=_Api(), count_rows=count)

    assert counts["Python-all"]["rows"] is None
    assert counts["Python-all"]["files_read"] == 0


def test_the_headroom_cli_reuses_phase_sevens_epoch_cap_and_verdicts(
        tmp_path, capsys):
    """The question is phase 7's -- can this source fill its share without being
    re-read past the cap -- so the arithmetic and the verdict strings are phase
    7's too. What phase 8 supplies is the supply."""
    import scripts.codeprep as CLI

    plan, rows, probes = _supply_inputs()
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(json.dumps(plan))
    probe_json = tmp_path / "probes.json"
    probe_json.write_text(json.dumps({"languages": {"unbucketed": {
        "configs": list(probes.values())}}}))
    rows_json = tmp_path / "rows.json"
    rows_json.write_text(json.dumps(rows))
    out = str(tmp_path / "headroom.json")

    rc = CLI._cli(["corpus", "headroom", "--plan-json", str(plan_json),
                   "--probe-json", str(probe_json), "--rows-json",
                   str(rows_json), "--budget-tokens", "1000000000",
                   "--json-out", out])

    printed = capsys.readouterr().out
    assert "SUPPORTED" in printed
    assert rc == 0
    written = json.load(open(out))
    # 650M code tokens of a 1B budget, and the footers are read once and cached.
    assert written["budgets"][0]["budget"] == 650_000_000
    assert written["supply"]["python"]["unique_tokens"] > 0


def test_the_headroom_cli_fails_the_budget_a_directory_cannot_fill(
        tmp_path, capsys):
    """The failure this exists to catch: the plan moved 14 points onto Python,
    and a Python directory that cannot carry them turns a fixed shortfall into a
    hidden one."""
    import scripts.codeprep as CLI

    plan, rows, probes = _supply_inputs(
        **{"Python-all": {"files": 10, "files_read": 10, "rows": 1_000,
                          "errors": {}}})
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(json.dumps(plan))
    probe_json = tmp_path / "probes.json"
    probe_json.write_text(json.dumps({"languages": {"unbucketed": {
        "configs": list(probes.values())}}}))
    rows_json = tmp_path / "rows.json"
    rows_json.write_text(json.dumps(rows))

    rc = CLI._cli(["corpus", "headroom", "--plan-json", str(plan_json),
                   "--probe-json", str(probe_json), "--rows-json",
                   str(rows_json), "--budget-tokens", "1000000000"])

    assert rc == 3
    assert "python would be read" in capsys.readouterr().err


def test_the_headroom_cli_takes_the_budget_it_is_given_not_three_of_them(
        tmp_path, capsys):
    """A repeatable flag with a list default appends to it, so one
    `--budget-tokens` would quietly mean four."""
    import scripts.codeprep as CLI

    plan, rows, probes = _supply_inputs()
    for name, value in (("plan.json", plan), ("rows.json", rows)):
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "probes.json").write_text(json.dumps(
        {"languages": {"unbucketed": {"configs": list(probes.values())}}}))
    out = str(tmp_path / "headroom.json")

    CLI._cli(["corpus", "headroom", "--plan-json", str(tmp_path / "plan.json"),
              "--probe-json", str(tmp_path / "probes.json"),
              "--rows-json", str(tmp_path / "rows.json"),
              "--budget-tokens", "250000000", "--json-out", out])

    assert [b["budget"] for b in json.load(open(out))["budgets"]] == \
        [int(round(250_000_000 * CP.CORPUS_SHARES["code"]))]


def test_the_cli_refuses_a_plan_whose_evidence_names_no_such_directory(capsys):
    """The interleaved record is looked up by name; a report that does not
    contain it must not be read as an empty directory."""
    import scripts.codeprep as CLI
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        configs_json = os.path.join(directory, "configs.json")
        probe_json = os.path.join(directory, "probe.json")
        with open(configs_json, "w") as f:
            json.dump({"available": AVAILABLE}, f)
        with open(probe_json, "w") as f:
            json.dump({"languages": {"python": {"configs": [
                {"config": "Python-all", "admitted_languages": {}}]}}}, f)

        rc = CLI._cli(["corpus", "plan", "--configs-json", configs_json,
                       "--probe-json", probe_json])

    assert rc == 2
    assert "all-all" in capsys.readouterr().err


# ------------------------------------------------- the directories that exist ---
#
# The probe's first live run returned `DataFilesNotFoundError` for four of ten
# directories -- 18% of the plan's code mixture -- because they were named from
# the dataset's language list rather than read off the revision. The parquet
# branch is an auto-converted *subset*, so a name can be absent because it was
# misspelled or because it was never converted, and only the list distinguishes
# them.


class _FakeApi:
    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo_id, repo_type=None, revision=None):
        self.asked = {"repo_id": repo_id, "repo_type": repo_type,
                      "revision": revision}
        return list(self._files)


def test_the_available_directories_are_read_off_the_revision_not_guessed():
    api = _FakeApi(["Python-all/partial-train/0000.parquet",
                    "Python-all/partial-train/0001.parquet",
                    "Go-all/partial-train/0000.parquet",
                    "README.md",
                    ".gitattributes"])

    available = CP.github_code_configs(api=api)

    assert available == {"Go-all": 1, "Python-all": 2}
    assert api.asked["repo_type"] == "dataset"
    assert api.asked["revision"] == CP.GITHUB_CODE_REVISION


def test_a_bucket_naming_a_directory_that_does_not_exist_is_reported():
    available = {"Python-all": 3, "Java-all": 2}

    missing = CP.missing_configs(available)

    assert missing["rust"] == ["Rust-all"]
    assert set(missing["c-cpp"]) == {"C-all", "C++-all"}
    assert "python" not in missing and "java" not in missing


def test_a_directory_that_differs_only_in_case_is_offered_as_the_near_miss():
    """`GO-all` against a real `Go-all`: invisible to an exact lookup, and the
    single most likely reason a language came back empty."""
    available = {"Go-all": 4, "Shell-all": 2}

    assert CP.config_near_misses("GO-all", available) == ["Go-all"]
    assert CP.config_near_misses("Shell-all", available) == []
    assert CP.config_near_misses("Rust-all", available) == []


def test_the_cli_lists_the_directories_and_fails_naming_the_near_miss(
        tmp_path, monkeypatch, capsys):
    import scripts.codeprep as CLI

    monkeypatch.setattr(CLI, "github_code_configs",
                        lambda **kw: {"Go-all": 4, "Python-all": 9})
    out = str(tmp_path / "configs.json")

    rc = CLI._cli(["corpus", "configs", "--json-out", out])

    assert rc == 3
    err = capsys.readouterr().err
    assert "did you mean Go-all?" in err
    # The distinction the live run turned on: a name that is wrong versus a
    # config that was never converted at all. Only the first has a remedy.
    assert "never converted" in err
    written = json.load(open(out))
    assert written["available"] == {"Go-all": 4, "Python-all": 9}
    assert written["near_misses"]["GO-all"] == ["Go-all"]
    assert "python" not in written["missing"]


def test_listing_another_repository_does_not_report_this_datasets_buckets(
        tmp_path, monkeypatch, capsys):
    """`--dataset` exists to shop for a substitute source. Reporting that
    `bigcode/whatever` is missing `Python-all` would be noise about a layout it
    never claimed to have."""
    import scripts.codeprep as CLI

    monkeypatch.setattr(CLI, "github_code_configs",
                        lambda **kw: {"data/rust": 12, "data/go": 9})
    out = str(tmp_path / "configs.json")

    rc = CLI._cli(["corpus", "configs", "--dataset", "bigcode/the-stack-smol",
                   "--json-out", out])

    assert rc == 0
    assert json.load(open(out))["missing"] == {}


def test_the_cli_writes_the_probe_and_fails_on_what_it_found(
        tmp_path, monkeypatch, capsys):
    """The record is written *before* the verdict: what was actually in the rows
    is the whole output anyone needs to fix the assumption, and a non-zero exit
    with nothing on disk is not it."""
    import scripts.codeprep as CLI

    monkeypatch.setattr(
        CLI, "probe_languages",
        lambda languages, **kw: CP.probe_languages(
            languages, stream=_stream(_github_rows(20, ("mit", "wtfpl"))), **kw))
    out = str(tmp_path / "probe.json")

    rc = CLI._cli(["corpus", "probe", "--language", "python", "--rows", "20",
                   "--json-out", out])

    assert rc == 3
    written = json.load(open(out))
    assert any("wtfpl" in p for p in written["problems"])
    assert written["languages"]["python"]["configs"][0]["licenses"]["wtfpl"] == 10
    assert "does not classify" in capsys.readouterr().err


class _Recorder:
    """A stream that records being flushed, so the order can be asserted."""

    def __init__(self, events, name):
        self._events, self._name = events, name

    def flush(self):
        self._events.append(("flush", self._name))

    def write(self, text):        # pragma: no cover - not what is asserted
        return len(text)


def test_the_probe_leaves_with_its_verdict_instead_of_dying_in_teardown(
        monkeypatch):
    """Both live probes exited -6, after printing the report and writing the
    JSON, from a `PyGILState_Release` abort while the interpreter finalized
    underneath a `datasets` streaming thread.

    That is not a lost crash, it is a lost *verdict*: the exit code is what the
    controller ledger records, so "four of ten directories did not resolve" (3)
    and "nothing wrong with these rows" (0) were filed identically, and phase
    8's coverage measurement is recorded in `state.json` as a failure it never
    was. Flush, then leave -- `os._exit` does not flush, and the report is the
    output."""
    import scripts.codeprep as CLI

    events = []
    monkeypatch.setattr(CLI.sys, "stdout", _Recorder(events, "stdout"))
    monkeypatch.setattr(CLI.sys, "stderr", _Recorder(events, "stderr"))
    monkeypatch.setattr(CLI.os, "_exit", lambda code: events.append(("exit", code)))

    CLI._exit(3)

    assert events == [("flush", "stdout"), ("flush", "stderr"), ("exit", 3)]


# ---------------------------------------------------------------- the build ---
#
# The build is the expensive step: hours of streaming, and every mistake in it
# is a directory full of plausible shards. So what these test is the arithmetic
# and the wiring *before* a row is read -- which directory is asked for how many
# tokens, which side of the split each pass admits, what a resumed source's
# manifest says it contains, and whether a source that stopped short can be
# reported as a corpus that is whole.


def _plan_with_supply():
    plan = _plan()
    probes = {config: _directory_probe(config)
              for config in ("Python-all", "JavaScript-all", "TypeScript-all",
                             "C-all", "C++-all", "Java-all")}
    # TypeScript is a quarter the size of JavaScript here, which is the case an
    # even split gets wrong.
    probes["TypeScript-all"] = _directory_probe("TypeScript-all",
                                                admitted_bytes=2_000_000)
    probes["all-all"] = _allall()
    rows = {config: {"files": 10, "files_read": 10, "rows": 500_000, "errors": {}}
            for config in probes}
    rows["all-all"] = {"files": 10, "files_read": 10, "rows": 20_000_000,
                       "errors": {}}
    supply = CP.bucket_supply(plan=plan, row_counts=rows, probes=probes)
    return plan, supply


def test_the_two_directories_of_one_bucket_get_separate_shard_directories():
    """`C-all` and `C++-all` differ only in characters a slug throws away. Given
    the same key they would share a directory and a shard prefix, and the second
    source would be read back as an interrupted run of the first -- so the
    corpus would hold C++ files manifested as C, with nothing having failed."""
    plan, supply = _plan_with_supply()

    sources = CP.corpus_specs(plan=plan, split="train", supply=supply,
                              total_tokens=1_000_000_000)

    keys = {s.config: s.key for s in sources if s.bucket == "c-cpp"}
    assert keys == {"C-all": "code-c-cpp-c-all", "C++-all": "code-c-cpp-cpp-all"}


def test_a_bucket_with_one_directory_is_named_after_the_bucket():
    plan, supply = _plan_with_supply()

    sources = CP.corpus_specs(plan=plan, split="train", supply=supply,
                              total_tokens=1_000_000_000)

    assert [s.key for s in sources if s.bucket == "python"] == ["code-python"]


def test_the_budget_is_the_code_share_of_the_total_not_the_total():
    """65% of the mixture is code; the rest is technical prose and general
    replay, and no code directory supplies either. Budgeting the whole total
    would over-ask every directory by half again -- and the epoch-cap verdict
    that said the directories hold it was computed on the code share."""
    plan, supply = _plan_with_supply()

    sources = CP.corpus_specs(plan=plan, split="train", supply=supply,
                              total_tokens=1_000_000_000)

    total = sum(s.token_budget for s in sources)
    assert total == pytest.approx(650_000_000, rel=1e-6)
    python = next(s for s in sources if s.bucket == "python")
    assert python.token_budget == pytest.approx(
        650_000_000 * plan["buckets"]["python"]["share"], rel=1e-6)


def test_a_multi_directory_buckets_budget_is_split_on_measured_supply():
    """An even split asks `TypeScript-all` for tokens it does not hold. The
    shortfall would then be reported per directory and invisible per bucket."""
    plan, supply = _plan_with_supply()

    sources = CP.corpus_specs(plan=plan, split="train", supply=supply,
                              total_tokens=1_000_000_000)

    js = {s.config: s.token_budget for s in sources
          if s.bucket == "javascript-typescript"}
    assert js["JavaScript-all"] > js["TypeScript-all"] * 3
    assert sum(js.values()) == pytest.approx(
        650_000_000 * plan["buckets"]["javascript-typescript"]["share"], abs=2)
    basis = next(s.basis for s in sources
                 if s.bucket == "javascript-typescript")
    assert "measured unique tokens" in basis


def test_an_unmeasured_bucket_is_split_evenly_and_says_so():
    """A plan can be built before a headroom pass. Dividing evenly is then the
    only thing left to do -- and the manifest has to say that is what happened,
    or a later shortfall reads as a directory that came up short rather than a
    budget that was guessed."""
    plan, _ = _plan_with_supply()

    sources = CP.corpus_specs(plan=plan, split="train", supply=None,
                              total_tokens=1_000_000_000)

    js = {s.config: s.token_budget for s in sources
          if s.bucket == "javascript-typescript"}
    assert js["JavaScript-all"] == pytest.approx(js["TypeScript-all"], abs=1)
    assert "no measured supply" in next(
        s.basis for s in sources if s.bucket == "javascript-typescript")


def test_the_two_sides_name_the_same_sources_and_partition_the_rows():
    """The holdout is a second independent stream over the same directories,
    not a slice of the first: same keys, same directories, same language
    filters, opposite sides of a total split function."""
    plan, supply = _plan_with_supply()

    train = CP.corpus_specs(plan=plan, split="train", supply=supply,
                            total_tokens=1_000_000_000)
    holdout = CP.corpus_specs(plan=plan, split="holdout", supply=supply,
                              total_tokens=1_000_000_000)

    assert [(s.key, s.config) for s in train] == [(s.key, s.config) for s in holdout]
    for a, b in zip(train, holdout):
        assert a.gate.want == "train" and b.gate.want == "holdout"
        # In this source's own language, so a fallback bucket's filter is not
        # what refuses the row.
        language = (sorted(a.gate.languages)[0] if a.gate.languages else "Python")
        row = _row(repo="org/one", license_value="mit", language=language)
        # Exactly one side admits it, whichever side that is.
        assert a.gate(dict(row)) != b.gate(dict(row))


def test_a_fallback_bucket_carries_the_language_filter_that_reaches_it():
    """Go has no directory on this revision; it is read out of the interleaved
    one. Without the filter the bucket would be every language at once, and the
    per-language BPB holdout it exists for would measure nothing."""
    plan, supply = _plan_with_supply()

    sources = CP.corpus_specs(plan=plan, split="train", supply=supply,
                              total_tokens=1_000_000_000)

    go = next(s for s in sources if s.bucket == "go")
    assert go.config == "all-all"
    assert go.gate.languages == frozenset({"go"})
    assert go.gate.manifest()["languages"] == ["go"]
    python = next(s for s in sources if s.bucket == "python")
    assert python.gate.languages is None


def test_a_dropped_bucket_is_not_built_as_an_empty_directory():
    """Rust is 0.336% of the interleaved directory against an 8% share, so the
    plan dropped it by name. Building it at zero tokens would put an empty
    shard directory in the corpus, which reads as a source that failed."""
    plan, supply = _plan_with_supply()

    sources = CP.corpus_specs(plan=plan, split="train", supply=supply,
                              total_tokens=1_000_000_000)

    assert plan["buckets"]["rust"]["source"] == "dropped"
    assert not [s for s in sources if s.bucket == "rust"]


def test_the_holdout_is_capped_rather_than_sized_as_a_share_of_the_corpus():
    """Every holdout token costs ~50 tokens of streaming, because the pass reads
    the whole directory and keeps the 2% of repositories on its side. A holdout
    sized as a share of a 3B-token corpus is 50x more measurement than a BPB
    gate needs, at 2B tokens of streaming to collect it."""
    plan, supply = _plan_with_supply()

    holdout = CP.corpus_specs(plan=plan, split="holdout", supply=supply,
                              total_tokens=3_000_000_000, holdout_cap_tokens=2_000_000)

    python = next(s for s in holdout if s.bucket == "python")
    assert python.token_budget == 2_000_000
    # And it scales down rather than dominating a small build.
    small = CP.corpus_specs(plan=plan, split="holdout", supply=supply,
                            total_tokens=10_000_000, holdout_cap_tokens=2_000_000)
    assert next(s for s in small if s.bucket == "python").token_budget < 200_000


def test_a_build_is_refused_on_a_plan_that_cannot_be_built_as_written():
    plan, supply = _plan_with_supply()
    plan["buckets"]["go"]["interleaved_rows"] = 0

    with pytest.raises(ValueError, match="cannot be built as written"):
        CP.corpus_specs(plan=plan, split="train", supply=supply,
                        total_tokens=1_000_000_000)


def test_each_side_writes_under_its_own_root_with_the_same_key():
    plan, supply = _plan_with_supply()
    train = CP.corpus_specs(plan=plan, split="train", supply=supply,
                            total_tokens=1_000_000_000)[0]
    holdout = CP.corpus_specs(plan=plan, split="holdout", supply=supply,
                              total_tokens=1_000_000_000)[0]

    assert train.out_dir("data/code-shards").endswith(f"train/{train.key}")
    assert holdout.out_dir("data/code-shards").endswith(f"holdout/{holdout.key}")


# ------------------------------------------------- the resumed gate manifest ---

def _train_side(n: int, prefix: str = "org") -> list:
    """Repository names the split puts on the train side at `holdout_frac` 0.5,
    so a fixture's gate really does admit the names it was handed rather than
    quietly refusing half of them."""
    names, index = [], 0
    while len(names) < n:
        name = f"{prefix}/repo-{index}"
        if CP.repository_split(name, holdout_frac=0.5) == "train":
            names.append(name)
        index += 1
    return names


def _gate_manifest(repos=None, **overrides) -> dict:
    gate = CP.RepositoryGate(want="train", holdout_frac=0.5)
    for name in (_train_side(2) if repos is None else repos):
        gate(_row(repo=name, license_value="mit"))
    manifest = gate.manifest()
    assert manifest["rows_admitted"] == manifest["rows_seen"], \
        "fixture gate refused a row it was meant to admit"
    manifest.update(overrides)
    return manifest


def test_a_resumed_sources_gate_manifest_covers_every_attempt():
    """A gate lives in the process that streams the source. The shards and the
    stream position survive that process dying; the licence histogram does not
    -- so without a merge, a source that stopped at 90%% and finished on a
    second run manifests the licences of the last 10%% as the corpus's."""
    first, second = _train_side(2)
    prior = _gate_manifest(repos=(first,))
    current = _gate_manifest(repos=(second,))

    merged = CP.merge_gate_manifest(prior, current)

    assert merged["rows_seen"] == prior["rows_seen"] + current["rows_seen"]
    assert merged["rows_admitted"] == (prior["rows_admitted"]
                                       + current["rows_admitted"])
    assert merged["licenses"]["mit"] == 2
    assert merged["attempts"] == 2


def test_one_repository_met_by_both_attempts_is_still_one_repository():
    a, b, c = _train_side(3)
    prior = _gate_manifest(repos=(a, b))
    current = _gate_manifest(repos=(b, c))

    merged = CP.merge_gate_manifest(prior, current)

    assert merged["repository_names"] == sorted({a, b, c})
    assert merged["repositories"] == 3
    # The rows are added up even though the repository is not: four files were
    # read, from three projects.
    assert merged["rows_admitted"] == 4


def test_a_first_attempt_merges_to_itself():
    current = _gate_manifest()

    assert CP.merge_gate_manifest(None, current) == {**current, "attempts": 1}


@pytest.mark.parametrize("field,value", [
    ("split", "holdout"), ("holdout_frac", 0.25),
    ("split_salt", "other"), ("languages", ["go"])])
def test_gate_manifests_that_disagree_about_the_split_are_not_added_up(
        field, value):
    """The same directory under a different holdout fraction, salt or language
    filter is a different corpus. Adding those counters produces a manifest
    describing neither pass."""
    prior = _gate_manifest()
    current = _gate_manifest(**{field: value})

    with pytest.raises(ValueError, match="disagree about"):
        CP.merge_gate_manifest(prior, current)


# ------------------------------------------------------------- the build CLI ---

class _FakeTokenizer:
    """Two bytes per token, deterministically, so budgets are checkable."""

    name_or_path = "fake"

    def encode(self, text):
        return [1] * max(1, len(text) // 2)


def _build_argv(tmp_path, plan_path, **overrides):
    argv = ["corpus", "build", "--plan-json", str(plan_path),
            "--headroom-json", str(tmp_path / "missing-headroom.json"),
            "--out-root", str(tmp_path / "code-shards"),
            "--manifest", str(tmp_path / "code-shards" / "manifest.json"),
            "--total-tokens", "100000", "--eval-index", "",
            "--code-index", str(tmp_path / "code-index.txt.gz"),
            "--shard-tokens", "100000"]
    for flag, value in overrides.items():
        argv += [f"--{flag}"] if value is True else [f"--{flag}", str(value)]
    return argv


def _write_code_index(tmp_path):
    """A real index file, so the build's own digest check is exercised."""
    path = str(tmp_path / "code-index.txt.gz")
    ngrams = [" ".join(f"tok{i}{j}" for j in range(13)) for i in range(50)]
    provenance = {
        "schema": 1, "kind": "code", "n": CP.DEFAULT_CODE_N,
        "ngrams": len(ngrams), "complete": True, "limit": None,
        "evalplus": "0.3.1", "built_at": "2026-08-26T00:00:00Z",
        "benchmarks": {name: {"items": CP.CODE_EXPECTED_ITEMS[name],
                              "fields": {"reference": 10}, "short_fields": {}}
                       for name in CP.CODE_BENCHMARKS},
    }
    CP.write_index(path, ngrams, provenance)
    return path


def _one_bucket_plan(tmp_path):
    """A plan with a single directory-backed bucket, so a CLI test streams one
    source per side instead of eleven."""
    plan = {"buckets": {"python": {"configs": ["Python-all"], "plan_share": 1.0,
                                   "share": 1.0, "source": "directories",
                                   "missing_configs": [], "reason": "test"}},
            "interleaved": {"config": "all-all", "available": True},
            "shares": {"python": 1.0}, "redistributed": 0.0,
            "interleave_passes": 1.0, "min_bucket_share": 0.005}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def _code_rows(n=400):
    """Distinct documents, not one document repeated: the near-duplicate filter
    is real and 400 near-identical files would be dropped as the duplicates they
    are, which would make every budget below unreachable for the wrong reason."""
    return [_row(repo=f"org-{i}/repo-{i}", license_value="mit",
                 code="\n".join(
                     f"def transform_{i}_{j}(value_{i}_{j}, scale_{j}):\n"
                     f"    return value_{i}_{j} * scale_{j} + {i * 31 + j}"
                     for j in range(30)))
            for i in range(n)]


def _github_stream(monkeypatch, rows):
    from daedalus import dataprep as DP

    monkeypatch.setattr(DP, "_stream_rows", lambda s, *a, **k: iter(list(rows)))


class _StatefulRows:
    """A row stream that can report and restore its position, the way
    `datasets`' `IterableDataset.state_dict` does -- which is the path a real
    build resumes on, and the one where the gate has *not* re-read the prefix."""

    def __init__(self, rows, start=0):
        self._rows, self._index = rows, start
        self.resumed = start > 0

    def __iter__(self):
        while self._index < len(self._rows):
            row = self._rows[self._index]
            self._index += 1
            yield row

    def state(self):
        return {"index": self._index}


def _stateful_stream(monkeypatch, rows):
    from daedalus import dataprep as DP

    def stream(spec, stream_state=None, log=print):
        return _StatefulRows(list(rows), start=(stream_state or {}).get("index", 0))

    monkeypatch.setattr(DP, "_stream_rows", stream)


def test_the_build_streams_each_side_into_its_own_root_and_manifests_both(
        tmp_path, monkeypatch, capsys):
    import scripts.codeprep as CLI
    from daedalus import data as DATA

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, _code_rows())

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                              **{"holdout-frac": 0.5}))

    assert rc == 0, capsys.readouterr().err
    root = tmp_path / "code-shards"
    for split in ("train", "holdout"):
        gate = json.load(open(root / split / "code-python" / "gate.json"))
        assert gate["split"] == split
        assert gate["rows_admitted"] > 0
        assert gate["licenses"]["mit"] > 0
        assert (root / split / "code-python" / "manifest.json").exists()
    manifest = json.load(open(root / "manifest.json"))
    assert {entry["split"] for entry in manifest["sources"]} == {"train", "holdout"}
    # Disjoint by construction, and shown rather than asserted.
    names = [set(entry["gate"]["repository_names"]) for entry in manifest["sources"]]
    assert names[0] and names[1] and not (names[0] & names[1])


def test_the_build_records_the_index_it_filtered_against(tmp_path, monkeypatch,
                                                         capsys):
    """"Filtered against some index" is the claim the released corpus could not
    improve on. The digest is what makes it "filtered against *this* one"."""
    import scripts.codeprep as CLI
    from daedalus import data as DATA

    path = _write_code_index(tmp_path)
    digest = CP.read_provenance(path)["digest"]
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, _code_rows(8))

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                              **{"code-index-digest": digest,
                                 "max-docs-per-source": 5}))

    assert rc == 0, capsys.readouterr().err
    manifest = json.load(open(tmp_path / "code-shards" / "manifest.json"))
    assert manifest["decontam_index"]["code"]["digest"] == digest
    assert manifest["decontam_index"]["eval"] is None
    assert manifest["decontam_index"]["ngrams"] == 50


def test_the_build_refuses_an_index_that_is_not_the_one_it_was_pinned_to(
        tmp_path, monkeypatch, capsys):
    import scripts.codeprep as CLI

    _write_code_index(tmp_path)

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                              **{"code-index-digest": "sha256:not-this-one"}))

    assert rc == 2
    assert "REFUSE" in capsys.readouterr().err


def test_a_relaunch_continues_a_source_instead_of_rebuilding_it(
        tmp_path, monkeypatch, capsys):
    """The rule the phase 4 sweep lost 60.3M tokens to: a run relaunched beside
    its own output continues it. Here the first pass is stopped by a document
    cap; the second must pick up the shards, the counters and the stream
    position rather than starting at row zero and overwriting them."""
    import scripts.codeprep as CLI
    from daedalus import data as DATA

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _stateful_stream(monkeypatch, _code_rows())
    plan = _one_bucket_plan(tmp_path)

    assert CLI._cli(_build_argv(tmp_path, plan, **{"holdout-frac": 0.5,
                                                   "max-docs-per-source": 20})) == 0
    first = json.load(open(tmp_path / "code-shards" / "manifest.json"))
    first_train = next(e for e in first["sources"] if e["split"] == "train")
    assert first_train["tokens"] > 0
    assert first_train["gate"]["attempts"] == 1

    assert CLI._cli(_build_argv(tmp_path, plan, **{"holdout-frac": 0.5,
                                                   "max-docs-per-source": 20})) == 0
    second = json.load(open(tmp_path / "code-shards" / "manifest.json"))
    second_train = next(e for e in second["sources"] if e["split"] == "train")

    assert second_train["resumed"] is True
    assert second_train["tokens"] > first_train["tokens"]
    # Continued from the saved position, so the gate never saw the prefix
    # again and the two attempts' counters are the source's.
    assert second_train["gate"]["attempts"] == 2
    assert second_train["gate"]["rows_seen"] > first_train["gate"]["rows_seen"]


def test_a_replayed_resume_does_not_count_its_prefix_twice(
        tmp_path, monkeypatch, capsys):
    """Without a saved stream position the resume replays from row zero, and
    `_DocumentStream` consults the gate on every replayed row before it skips
    it. This attempt's manifest therefore already covers the whole source --
    folding the previous one onto it would report a licence histogram twice the
    size of the corpus it describes."""
    import scripts.codeprep as CLI
    from daedalus import data as DATA

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, _code_rows())            # no state() to restore
    plan = _one_bucket_plan(tmp_path)

    assert CLI._cli(_build_argv(tmp_path, plan, **{"holdout-frac": 0.5,
                                                   "max-docs-per-source": 20})) == 0
    assert CLI._cli(_build_argv(tmp_path, plan, **{"holdout-frac": 0.5,
                                                   "max-docs-per-source": 40})) == 0

    second = json.load(open(tmp_path / "code-shards" / "manifest.json"))
    train = next(e for e in second["sources"] if e["split"] == "train")
    assert train["resumed"] is True
    assert train["gate"]["attempts"] == 1
    assert train["gate"]["rows_seen"] <= len(_code_rows())


def test_a_source_that_stops_short_of_its_budget_fails_the_build(
        tmp_path, monkeypatch, capsys):
    """`headroom` said the directories hold their shares inside the epoch cap. A
    source that runs out anyway has falsified that, and the budget it fell short
    of is the one downstream training reads -- so this must not be a log line
    under a corpus used as if it were whole."""
    import scripts.codeprep as CLI
    from daedalus import data as DATA

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, [_row(repo="org/a", license_value="mit",
                                      code="print(1)\n" * 60)])

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path)))

    assert rc == 3
    assert "stopped short" in capsys.readouterr().err


def test_a_capped_smoke_does_not_manifest_the_directory_as_exhausted(
        tmp_path, monkeypatch, capsys):
    """`run_source` reads exhaustion off the stream ending, and under a document
    cap the stream ends at the cap. A 25-document live smoke manifested
    `Java-all` -- a directory with millions of rows left -- as having no more
    documents, which is a claim about the source made from a measurement of the
    cap."""
    import scripts.codeprep as CLI
    from daedalus import data as DATA

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, _code_rows(400))

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                              **{"holdout-frac": 0.5, "max-docs-per-source": 5}))

    assert rc == 0, capsys.readouterr().err
    manifest = json.load(open(tmp_path / "code-shards" / "manifest.json"))
    train = next(e for e in manifest["sources"] if e["split"] == "train")
    assert "exhausted" not in train
    assert train["stopped_by_doc_cap"] is True


def test_a_source_that_really_runs_out_still_says_so(tmp_path, monkeypatch,
                                                     capsys):
    """The other half of the same distinction: with no cap in force, a stream
    that ends *is* the source running out, and that is the finding."""
    import scripts.codeprep as CLI
    from daedalus import data as DATA

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, _code_rows(4))

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                              **{"holdout-frac": 0.5}))

    assert rc == 3
    manifest = json.load(open(tmp_path / "code-shards" / "manifest.json"))
    train = next(e for e in manifest["sources"] if e["split"] == "train")
    assert train["exhausted"] is True
    assert "stream exhausted" in capsys.readouterr().err


def test_one_source_tripping_the_memory_cap_stops_the_process(
        tmp_path, monkeypatch, capsys):
    """`_run_group_worker`'s rule. The eighth dataprep incident was a raw
    C-level malloc failure moments after one source's *graceful* cap trip, in a
    process that had gone on to the next source while still holding the memory
    -- and it took every other source's in-flight progress with it. Leaving
    exits non-zero, and the controller's next attempt is a fresh process that
    resumes from the shards."""
    import scripts.codeprep as CLI
    from daedalus import data as DATA
    from daedalus import dataprep as DP

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, _code_rows(400))

    seen = []

    def explode(limit_gb):
        raise DP.WorkerMemoryExceeded("worker RSS 9.1 GB exceeds the 8.0 GB cap")

    monkeypatch.setattr(DP, "_check_worker_rss", explode)
    monkeypatch.setattr(DP, "run_source", _counting(DP.run_source, seen))

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                              **{"holdout-frac": 0.5, "rss-check-every": 1}))

    assert rc == 3
    # The holdout side ran and tripped; the train side was never started.
    assert len(seen) == 1
    manifest = json.load(open(tmp_path / "code-shards" / "manifest.json"))
    train = next(e for e in manifest["sources"] if e["split"] == "train")
    assert "skipped" in train["error"]


def _counting(fn, seen):
    def wrapper(*args, **kwargs):
        seen.append(args[0].key)
        return fn(*args, **kwargs)
    return wrapper


def test_a_holdout_pass_checkpoints_even_though_it_yields_almost_nothing():
    """`run_source` counts its checkpoint cadence in *yielded* documents, and a
    holdout pass yields ~2% of what it streams. At the shared 50,000-document
    default the whole 2M-token Python holdout yields ~800 documents and never
    checkpoints once -- so the pass with the most streaming to lose is the one
    a fixed cadence never protects."""
    assert CP.checkpoint_every_for(2_000_000) == CP.MIN_CHECKPOINT_DOCUMENTS
    assert CP.checkpoint_every_for(2_000_000) < 50_000


def test_a_large_source_does_not_checkpoint_itself_into_hundreds_of_shards():
    """Every checkpoint closes the buffer as a new shard file, so the cadence
    cannot just be made small: a 500-document one over the 418M-token Python
    source would leave ~840 shards behind."""
    every = CP.checkpoint_every_for(418_827_500)

    documents = 418_827_500 / CP.CODE_TOKENS_PER_DOCUMENT
    assert documents / every == pytest.approx(CP.CHECKPOINTS_PER_SOURCE, abs=1)
    assert every > CP.MIN_CHECKPOINT_DOCUMENTS


def test_the_build_derives_the_cadence_per_source_unless_told_otherwise(
        tmp_path, monkeypatch, capsys):
    import scripts.codeprep as CLI
    from daedalus import data as DATA
    from daedalus import dataprep as DP

    _write_code_index(tmp_path)
    monkeypatch.setattr(DATA, "get_tokenizer", lambda name=None: _FakeTokenizer())
    _github_stream(monkeypatch, _code_rows(400))
    seen = []

    real_run_source = DP.run_source          # not DP.run_source at call time:
                                             # that is this wrapper, forever

    def record(spec, tokenizer, budget, out_dir, *args, **kwargs):
        seen.append((spec.key, kwargs["checkpoint_every"]))
        return real_run_source(spec, tokenizer, budget, out_dir, *args, **kwargs)

    monkeypatch.setattr(DP, "run_source", record)

    assert CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                                **{"holdout-frac": 0.5})) == 0
    assert {every for _, every in seen} == {CP.MIN_CHECKPOINT_DOCUMENTS}

    seen.clear()
    # A second root, because the first build met both budgets and a resumed
    # source that is already complete is skipped rather than re-run.
    assert CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                                **{"holdout-frac": 0.5, "checkpoint-every": 7,
                                   "out-root": tmp_path / "again",
                                   "manifest": tmp_path / "again" / "m.json"})) == 0
    assert {every for _, every in seen} == {7}


def test_a_dry_run_prints_the_budgets_without_reading_a_row(tmp_path, capsys):
    import scripts.codeprep as CLI

    rc = CLI._cli(_build_argv(tmp_path, _one_bucket_plan(tmp_path),
                              **{"dry-run": True}))

    assert rc == 0
    printed = capsys.readouterr().out
    assert "code-python" in printed and "nothing was streamed" in printed
    assert not (tmp_path / "code-shards").exists()


# --------------------------------------------------- the training mixture ----

def _mixture_source(key, split, bucket, budget, tokens=None):
    return {"key": key, "split": split, "bucket": bucket,
            "token_budget": budget,
            "tokens": budget if tokens is None else tokens}


def _code_manifest(**overrides) -> dict:
    """A finished two-bucket build's manifest, as `corpus build` writes one."""
    manifest = {
        "schema": 1, "total_tokens": 1_000_000_000, "code_tokens": 650_000_000,
        "corpus_shares": dict(CP.CORPUS_SHARES),
        "sources": [
            _mixture_source("code-python", "holdout", "python", 2_000_000),
            _mixture_source("code-python", "train", "python", 357_500_000),
            _mixture_source("code-javascript-typescript-javascript-all",
                            "train", "javascript-typescript", 125_727_256),
            _mixture_source("code-javascript-typescript-typescript-all",
                            "train", "javascript-typescript", 166_772_744),
        ],
    }
    manifest.update(overrides)
    return manifest


def _shard_dir(root, name, tokens=100_000_000):
    """A directory shaped the way `resolve_mixture` and `PackedTokenDataset`
    read one: the manifest is the file that has to be there."""
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"total_tokens": tokens, "shards": [], "eos_id": 0}))
    return path


def _general_roots(tmp_path, carved=("fineweb-edu", "dclm-baseline")):
    """The two general roots this box has: a holdout-disjoint carve of the
    sources with enough shards to split, and the original corpus for the rest."""
    shards, train, holdout = (tmp_path / "shards", tmp_path / "shards-train",
                              tmp_path / "holdout")
    for name in sorted(n for members in CP.replay_buckets().values()
                       for n in members):
        _shard_dir(shards, name)
    for name in carved:
        _shard_dir(train, name, tokens=98_000_000)
        _shard_dir(holdout, name, tokens=2_000_000)
    return shards, train, holdout


def test_the_three_buckets_are_the_shares_the_plan_preregistered():
    record = CP.training_mixture(
        code_sources=CP.code_train_sources(_code_manifest()))

    per_bucket = {bucket: sum(record["weights"][name] for name in members)
                  for bucket, members in record["buckets"].items()}
    assert per_bucket == pytest.approx({"code": 0.65, "technical": 0.15,
                                        "general-replay": 0.20}, abs=1e-9)
    assert sum(record["weights"].values()) == pytest.approx(1.0, abs=1e-12)


def test_the_code_bucket_is_divided_the_way_the_corpus_was_built():
    """By token budget, where python 55 / javascript-typescript 45 and the
    within-bucket split by measured supply are already encoded -- not by what
    happens to have landed on disk."""
    record = CP.training_mixture(
        code_sources=CP.code_train_sources(_code_manifest()))
    code = record["buckets"]["code"]

    assert code["code-python"] == pytest.approx(0.55, abs=1e-6)
    javascript = (code["code-javascript-typescript-javascript-all"]
                  + code["code-javascript-typescript-typescript-all"])
    assert javascript == pytest.approx(0.45, abs=1e-6)
    # The two JS/TS directories are not halves of each other: TypeScript-all
    # holds 1,443M unique tokens against JavaScript-all's 1,088M.
    assert (code["code-javascript-typescript-typescript-all"]
            > code["code-javascript-typescript-javascript-all"])


def test_general_replay_is_the_original_mixture_without_its_code_or_technical():
    """"Original general replay" is the distribution the released model was
    pretrained on, so it is read off `dataprep.MIXTURE` rather than retyped --
    minus the sources the other two buckets already serve."""
    from daedalus.dataprep import MIXTURE

    buckets = CP.replay_buckets()
    original = {spec.key: spec.share for spec in MIXTURE}

    assert set(buckets["technical"]) == set(CP.TECHNICAL_REPLAY_SOURCES)
    assert "stack-edu-python" not in buckets["general-replay"]
    assert set(buckets["general-replay"]) == (
        set(original) - set(CP.TECHNICAL_REPLAY_SOURCES)
        - set(CP.CODE_REPLAY_SOURCES))
    # Relative order and ratios are the original mixture's, only rescaled.
    ratio = original["fineweb-edu"] / original["dclm-baseline"]
    assert (buckets["general-replay"]["fineweb-edu"]
            / buckets["general-replay"]["dclm-baseline"]) == pytest.approx(ratio)


def test_replaying_the_original_code_source_would_overrun_the_code_share():
    """`stack-edu-python` is 9% of the original mixture and is GitHub code from
    the dataset the code bucket streams. Counted as general replay it would make
    a corpus whose manifest says 65% code 66.8% code."""
    from daedalus.dataprep import MIXTURE

    original = {spec.key: spec.share for spec in MIXTURE}
    general = {name: share for name, share in original.items()
               if name not in CP.TECHNICAL_REPLAY_SOURCES}
    leaked = (CP.CORPUS_SHARES["general-replay"]
              * general["stack-edu-python"] / sum(general.values()))

    assert CP.CORPUS_SHARES["code"] + leaked > 0.665


def test_a_source_missing_from_disk_leaves_its_buckets_share_where_it_was():
    """65/15/20 is the preregistered quantity. A directory that was never built
    moves share to the rest of *its* bucket, not from prose to code."""
    present = [name for members in CP.replay_buckets().values()
               for name in members if name != "finephrase"]

    record = CP.training_mixture(
        code_sources=CP.code_train_sources(_code_manifest()), present=present)

    assert record["absent"] == {"general-replay": ["finephrase"]}
    assert "finephrase" not in record["weights"]
    assert sum(record["weights"][name]
               for name in record["buckets"]["general-replay"]
               ) == pytest.approx(0.20, abs=1e-9)
    assert sum(record["weights"].values()) == pytest.approx(1.0, abs=1e-12)


def test_a_bucket_with_a_share_and_no_source_on_disk_is_refused():
    present = [name for name in CP.replay_buckets()["general-replay"]]

    with pytest.raises(ValueError, match="technical"):
        CP.training_mixture(
            code_sources=CP.code_train_sources(_code_manifest()),
            present=present)


def test_a_build_that_is_still_running_is_refused_rather_than_composed():
    """The failure this closes: the running build had finished Python and was
    9% into JavaScript. A mixture composed from that manifest trains on 100%
    Python at a weight that says 55%."""
    manifest = _code_manifest()
    for source in manifest["sources"]:
        if source["key"].endswith("javascript-all"):
            source["tokens"] = 11_593_381

    with pytest.raises(ValueError, match="still running or this source came up short"):
        CP.code_train_sources(manifest)


def test_a_source_the_build_has_not_reached_yet_is_refused_by_its_absence():
    """Found by running the dry run against the live corpus, which composed a
    mixture rather than refusing one. `corpus build` appends to its manifest as
    each source *finishes*, so the directory it has not started is absent rather
    than short -- and the bucket check passes anyway, because the bucket's other
    directory is there. What was composed was 74/26 python to javascript at
    weights that said 55/45."""
    manifest = _code_manifest()
    manifest["sources"] = [source for source in manifest["sources"]
                           if not source["key"].endswith("typescript-all")]

    with pytest.raises(ValueError, match="has not reached it"):
        CP.code_train_sources(manifest)


def test_the_gap_is_named_in_tokens_rather_than_in_directories():
    """The manifest cannot say which directory is missing -- there is no entry
    to name -- so it says how much of the code budget has no entry."""
    manifest = _code_manifest()
    manifest["sources"] = [source for source in manifest["sources"]
                           if not source["key"].endswith("typescript-all")]

    with pytest.raises(ValueError, match="166,772,744 tokens' worth"):
        CP.code_train_sources(manifest)


def test_a_finished_build_totals_its_own_code_budget():
    """The check above rests on the train budgets being a partition of
    `code_tokens`, which is what `config_budgets`' largest-remainder split
    makes them. If that ever stops holding, every finished build is refused."""
    manifest = _code_manifest()

    assert sum(source["token_budget"] for source in manifest["sources"]
               if source["split"] == "train") == manifest["code_tokens"]
    assert CP.code_train_sources(manifest)


def test_a_bucket_that_never_reached_the_shards_is_refused_by_name():
    manifest = _code_manifest()
    manifest["sources"] = [source for source in manifest["sources"]
                           if source["bucket"] != "javascript-typescript"]

    with pytest.raises(ValueError, match="javascript-typescript"):
        CP.code_train_sources(manifest)


def test_a_source_that_failed_is_refused_with_its_error():
    manifest = _code_manifest()
    manifest["sources"][1]["error"] = "WorkerMemoryExceeded(8.4)"

    with pytest.raises(ValueError, match="WorkerMemoryExceeded"):
        CP.code_train_sources(manifest)


def test_the_weights_are_the_flags_train_py_actually_accepts():
    """Twelve shares rounded to six places can miss 1.0 by more than
    `parse_mixture_weights` allows, and its refusal names the sum rather than
    the rounding."""
    from train import parse_mixture_weights

    record = CP.training_mixture(
        code_sources=CP.code_train_sources(_code_manifest()))
    flags = CP.mixture_weight_flags(record["weights"])

    assert flags[0] == "--mixture-weight"
    parsed = parse_mixture_weights(flags[1::2])
    assert parsed == pytest.approx(record["weights"], abs=1e-12)


def test_the_composed_root_is_the_mixture_the_trainer_resolves(tmp_path):
    """The real check on the composition: `resolve_mixture` over the composed
    root returns the shares that were asked for, over sources that live in two
    different corpora."""
    from train import resolve_mixture

    shards, train, _ = _general_roots(tmp_path)
    code_root = tmp_path / "code-shards"
    manifest = _code_manifest()
    code = CP.code_train_sources(manifest)
    for key, row in code.items():
        _shard_dir(code_root / "train", key, tokens=row["token_budget"])
    dirs, missing = CP.resolve_source_dirs(
        sorted(name for members in CP.replay_buckets().values()
               for name in members),
        roots=[str(train), str(shards)])
    assert missing == []
    dirs.update({key: str(code_root / "train" / key) for key in code})

    record = CP.training_mixture(code_sources=code, present=sorted(
        name for name in dirs if not name.startswith("code-")))
    CP.compose_mixture_root(out_root=str(tmp_path / "mix"), sources=dirs)

    names, target, probs, on_disk = resolve_mixture(
        str(tmp_path / "mix" / "train"), 250_000_000,
        weights=record["weights"], verbose=False)
    assert set(names) == set(record["weights"])
    assert target == pytest.approx(record["weights"], abs=1e-9)
    assert on_disk["code-python"] == 357_500_000


def test_the_carve_that_the_holdout_was_taken_from_is_preferred(tmp_path):
    """`data/shards` still holds the windows `data/holdout` scores, and the
    retention gate that reads them decides whether 1B tokens get spent."""
    shards, train, _ = _general_roots(tmp_path)

    dirs, missing = CP.resolve_source_dirs(
        ["fineweb-edu", "finephrase"], roots=[str(train), str(shards)])

    assert dirs["fineweb-edu"] == str(train / "fineweb-edu")
    # Never carved -- one shard, nothing to hold out -- so it resolves to the
    # original corpus and is scored by nothing.
    assert dirs["finephrase"] == str(shards / "finephrase")
    assert missing == []


def test_a_source_that_is_on_no_root_is_reported_not_guessed(tmp_path):
    shards, train, _ = _general_roots(tmp_path)

    dirs, missing = CP.resolve_source_dirs(
        ["fineweb-edu", "nemotron-cc"], roots=[str(train), str(shards)])

    assert missing == ["nemotron-cc"] and "nemotron-cc" not in dirs


def test_composition_refuses_a_directory_with_no_manifest(tmp_path):
    """`PackedTokenDataset` opens manifest.json, so an empty or mistyped path
    fails several layers inside a training run instead of here."""
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="no manifest.json"):
        CP.compose_mixture_root(out_root=str(tmp_path / "mix"),
                                sources={"fineweb-edu": str(tmp_path / "empty")})


def test_composition_never_replaces_a_real_directory(tmp_path):
    """The composed root is disposable and the corpora it points at are not."""
    source = _shard_dir(tmp_path / "shards", "fineweb-edu")
    real = tmp_path / "mix" / "train" / "fineweb-edu"
    real.mkdir(parents=True)
    (real / "keep.bin").write_bytes(b"0")

    with pytest.raises(ValueError, match="refusing to replace"):
        CP.compose_mixture_root(out_root=str(tmp_path / "mix"),
                                sources={"fineweb-edu": str(source)})
    assert (real / "keep.bin").exists()


def test_recomposing_onto_a_stale_link_repoints_it(tmp_path):
    """A composed root is a view. Re-running it after a corpus moves must follow
    the corpus rather than keep reading the old one."""
    first = _shard_dir(tmp_path / "old", "fineweb-edu", tokens=1_000)
    second = _shard_dir(tmp_path / "new", "fineweb-edu", tokens=2_000)
    out = str(tmp_path / "mix")

    CP.compose_mixture_root(out_root=out, sources={"fineweb-edu": str(first)})
    composed = CP.compose_mixture_root(out_root=out,
                                       sources={"fineweb-edu": str(second)})

    assert composed["fineweb-edu"]["relinked"] is True
    assert composed["fineweb-edu"]["tokens"] == 2_000
    assert os.path.realpath(composed["fineweb-edu"]["path"]) == str(second)


def _mixture_argv(tmp_path, **overrides):
    argv = ["corpus", "mixture",
            "--manifest", str(tmp_path / "code-shards" / "manifest.json"),
            "--code-root", str(tmp_path / "code-shards"),
            "--general-train-root", str(tmp_path / "shards-train"),
            "--general-root", str(tmp_path / "shards"),
            "--general-holdout-root", str(tmp_path / "holdout"),
            "--out-root", str(tmp_path / "code-mixture"),
            "--json-out", str(tmp_path / "train-mixture.json"),
            "--total-tokens", "250000000"]
    for flag, value in overrides.items():
        argv += [f"--{flag}"] if value is True else [f"--{flag}", str(value)]
    return argv


def _built_corpus(tmp_path):
    _general_roots(tmp_path)
    manifest = _code_manifest()
    code_root = tmp_path / "code-shards"
    for source in manifest["sources"]:
        _shard_dir(code_root / source["split"], source["key"],
                   tokens=source["tokens"])
    (code_root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_the_cli_composes_one_root_and_prints_what_to_train_on(
        tmp_path, capsys):
    import scripts.codeprep as CLI

    _built_corpus(tmp_path)

    assert CLI._cli(_mixture_argv(tmp_path)) == 0

    printed = capsys.readouterr().out
    assert "code-python" in printed and "fineweb-edu" in printed
    record = json.loads((tmp_path / "train-mixture.json").read_text())
    flags = record["train_flags"]
    assert flags[:2] == ["--data-dir", str(tmp_path / "code-mixture" / "train")]
    assert flags[2:4] == ["--val-dir", str(tmp_path / "code-mixture" / "holdout")]
    # Both corpora reachable under the one root `--data-dir` can name.
    train_root = tmp_path / "code-mixture" / "train"
    assert (train_root / "code-python" / "manifest.json").exists()
    assert (train_root / "fineweb-edu" / "manifest.json").exists()
    # And the holdout side carries the code sources plus the general holdout
    # every phase so far has been scored on.
    holdout_root = tmp_path / "code-mixture" / "holdout"
    assert sorted(p.name for p in holdout_root.iterdir()) == [
        "code-python", "dclm-baseline", "fineweb-edu"]
    assert record["preflight"]["total_run_tokens"] == 250_000_000


def test_the_cli_refuses_a_corpus_that_is_still_streaming(tmp_path, capsys):
    import scripts.codeprep as CLI

    manifest = _built_corpus(tmp_path)
    manifest["sources"][2]["tokens"] = 11_593_381
    (tmp_path / "code-shards" / "manifest.json").write_text(json.dumps(manifest))

    rc = CLI._cli(_mixture_argv(tmp_path))

    assert rc == 2
    assert "REFUSE" in capsys.readouterr().err
    assert not (tmp_path / "code-mixture").exists()


def test_the_cli_dry_run_prints_the_shares_without_linking(tmp_path, capsys):
    import scripts.codeprep as CLI

    _built_corpus(tmp_path)

    assert CLI._cli(_mixture_argv(tmp_path, **{"dry-run": True})) == 0
    assert "nothing was linked" in capsys.readouterr().out
    assert not (tmp_path / "code-mixture").exists()


def test_the_cli_fails_a_mixture_the_corpus_cannot_hold_at_the_budget(
        tmp_path, capsys):
    """The epoch cap moves shares, and a mixture far from the one asked for is a
    different experiment. `everyday-conversations` is 403,573 tokens on this box,
    so a large enough budget caps it and every share moves."""
    import scripts.codeprep as CLI

    _built_corpus(tmp_path)
    for name in ("everyday-conversations", "finewiki-en", "cosmopedia-v2",
                 "finephrase"):
        _shard_dir(tmp_path / "shards", name, tokens=400_000)

    rc = CLI._cli(_mixture_argv(tmp_path, **{"total-tokens": 3_000_000_000}))

    assert rc == 3
    assert "past the 5-point limit" in capsys.readouterr().err
    # Written before the verdict, so a failing compose still leaves the record.
    assert json.loads((tmp_path / "train-mixture.json").read_text())["preflight"]
