"""Tests for phase 8 step 6's code SFT admission gate.

The cases that matter are the ones separating this gate from the general SFT
filter it borrows its numbers from, because each of those is a way the corpus
comes back wrong while every count looks plausible:

  - a long code answer with short prose is *admitted* (the general 1,200-char
    cap applied to the whole assistant turn would drop it, and with it most of
    what this phase exists to add);
  - broken code in a *user* turn is admitted and broken code in an assistant
    turn is refused (checking every role drops debugging conversations);
  - a benchmark n-gram in a user turn is refused (checking only the assistant
    lets the item in through whoever pasted it);
  - an untagged block is never guessed at, so it lands in neither language
    bucket and is reported as unknown rather than as Python.
"""

import json
from pathlib import Path

import pytest

from daedalus.chatml import DEFAULT_MAX_ASSISTANT_CHARS
from daedalus.code_sft import (ADMISSION_REASONS, DEFAULT_CODE_SHARE,
                               DEFAULT_MAX_LEN, HOLDOUT_FILE, LANGUAGE_BUCKETS,
                               MANIFEST_FILE, OTHER_LANGUAGE,
                               RECORDED_ALTERNATIVES, SFT_HALVES, SFT_SOURCES,
                               SYNTAX_CHECKERS, TRAIN_FILE, UNDECLARED_LICENSE,
                               UNKNOWN_LANGUAGE, CodeBlock, CodeSFTError,
                               SFTSource, ShareAccumulator, build_dataset,
                               build_problems, contamination_hit,
                               dataset_license_verdict, group_sources,
                               half_shares, language_of, normalize_tag,
                               parse_markdown, parse_messages, probe_problems,
                               probe_source, probe_sources, row_messages,
                               share_report, shipped_test, source_group_key,
                               vet_example)
from daedalus.codeprep import CODE_LANGUAGE_SHARES
from daedalus.data import ngram_set


def chat(user: str, assistant: str):
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": assistant}]


def fenced(tag: str, body: str) -> str:
    return f"```{tag}\n{body}\n```"


# ------------------------------------------------------------- fence parsing ---

def test_parse_markdown_splits_prose_from_a_tagged_block():
    parsed = parse_markdown("before\n" + fenced("python", "x = 1") + "\nafter")
    assert parsed.prose == "before\nafter"
    assert [b.source for b in parsed.blocks] == ["x = 1"]
    assert parsed.blocks[0].tag == "python"
    assert parsed.blocks[0].language == "python"
    assert parsed.blocks[0].closed


def test_parse_markdown_closes_an_indented_fence():
    text = "  ```python\n  x = 1\n  ```\ntail"
    parsed = parse_markdown(text)
    assert parsed.blocks[0].closed
    assert parsed.blocks[0].source == "  x = 1"
    assert parsed.prose == "tail"


def test_parse_markdown_marks_an_unterminated_fence():
    parsed = parse_markdown("here you go\n```python\nx = 1\n")
    assert len(parsed.blocks) == 1
    assert not parsed.blocks[0].closed


def test_a_longer_fence_is_not_closed_by_a_shorter_one_inside_it():
    text = "````markdown\n```\nnested\n```\n````\nend"
    parsed = parse_markdown(text)
    assert len(parsed.blocks) == 1
    assert parsed.blocks[0].closed
    assert parsed.blocks[0].source == "```\nnested\n```"
    assert parsed.prose == "end"


def test_parse_messages_keeps_prose_assistant_only_and_blocks_from_every_role():
    messages = chat("look:\n" + fenced("python", "x ="),
                    "fixed:\n" + fenced("python", "x = 1"))
    prose, blocks = parse_messages(messages)
    assert prose.strip() == "fixed:"
    assert [b.role for b in blocks] == ["user", "assistant"]


def test_a_stray_closing_fence_reads_as_an_unterminated_block():
    """An assistant turn with an odd number of fences is broken markdown either
    way, and the conservative reading -- a block that never closed -- is the one
    that refuses it rather than training on half a code block as prose."""

    parsed = parse_markdown("here you go\n```\nx = 1")
    assert len(parsed.blocks) == 1 and not parsed.blocks[0].closed


# --------------------------------------------------------------- language ---

@pytest.mark.parametrize("info,expected", [
    ("python", "python"), ("py", "python"), ("Python3", "python"),
    ("js", "javascript"), ("jsx", "javascript"), ("ts", "typescript"),
    ("tsx", "typescript"), ("python {.line-numbers}", "python"),
    ("", ""), ("bash", "bash"),
])
def test_normalize_tag(info, expected):
    assert normalize_tag(info) == expected


def test_language_buckets_are_the_corpus_buckets():
    assert set(LANGUAGE_BUCKETS.values()) == set(CODE_LANGUAGE_SHARES)
    assert language_of("python") == "python"
    assert language_of("typescript") == "javascript-typescript"
    assert language_of("bash") == OTHER_LANGUAGE
    assert language_of("") == UNKNOWN_LANGUAGE


def test_an_untagged_block_is_unknown_rather_than_guessed_python():
    """`x = 1` parses as Python and as several other languages. Attribution by
    successful parse would inflate the one bucket both gate benchmarks live in."""

    verdict = vet_example(chat("q", "here:\n" + fenced("", "x = 1")))
    assert verdict.admitted
    assert verdict.language_bytes == {UNKNOWN_LANGUAGE: len("x = 1")}
    assert verdict.primary_language is None


# ----------------------------------------------------------------- refusals ---

def test_every_refusal_reason_is_registered():
    assert len(set(ADMISSION_REASONS)) == len(ADMISSION_REASONS)


def test_malformed_row_is_refused_rather_than_skipped():
    assert vet_example([{"role": "user"}]).reason == "malformed"
    assert vet_example([]).reason == "malformed"


def test_no_assistant_turn_is_refused():
    assert vet_example([{"role": "user", "content": "hi"}]).reason == "no_assistant"


def test_unterminated_assistant_fence_is_refused():
    verdict = vet_example(chat("q", "sure\n```python\nx = 1\n"))
    assert verdict.reason == "unterminated_code_fence"


def test_unterminated_fence_in_a_user_turn_is_not_the_assistants_problem():
    messages = chat("mine breaks:\n```python\nx =",
                    "use:\n" + fenced("python", "x = 1"))
    assert vet_example(messages).admitted


def test_assistant_syntax_error_is_refused():
    verdict = vet_example(chat("q", fenced("python", "def f(:\n    pass")))
    assert verdict.reason == "syntax_error"
    assert "SyntaxError" in verdict.detail


def test_user_syntax_error_is_admitted_because_debugging_is_the_point():
    messages = chat("why does this fail?\n" + fenced("python", "def f(:\n    pass"),
                    "a paren is unbalanced:\n" + fenced("python", "def f():\n    pass"))
    assert vet_example(messages).admitted


def test_unparseable_language_is_admitted_and_counted_unchecked():
    verdict = vet_example(chat("q", fenced("typescript", "const x: int = ;;;")))
    assert verdict.admitted, "no TypeScript parser on this box: unchecked, not passed"
    assert "typescript" not in SYNTAX_CHECKERS
    report = share_report([verdict])
    assert report["checked_code_bytes"] == 0
    assert report["unchecked_code_bytes"] > 0
    assert report["syntax_checked_share"] == 0.0


# ------------------------------------------------------------ contamination ---

def index_for(text: str, n: int = 13):
    return ngram_set(text, n)


ITEM = ("def has_close_elements ( numbers , threshold ) : "
        "for idx , elem in enumerate ( numbers ) : return False")


def test_contamination_in_an_assistant_turn_is_refused():
    verdict = vet_example(chat("q", ITEM), indexes={"code": index_for(ITEM)})
    assert verdict.reason == "contaminated"
    assert "'code'" in verdict.detail


def test_contamination_in_a_user_turn_is_refused_too():
    verdict = vet_example(chat(ITEM, "sure, here you go"),
                          indexes={"code": index_for(ITEM)})
    assert verdict.reason == "contaminated"


def test_contamination_hit_names_the_index_and_returns_none_when_clean():
    indexes = {"code": index_for(ITEM), "general": set()}
    assert contamination_hit(chat("q", ITEM), indexes) == "code"
    assert contamination_hit(chat("q", "unrelated prose"), indexes) is None


def test_contamination_is_checked_before_the_content_filters():
    """A contaminated *and* verbose example counts as contaminated. The
    contamination count is the one that must never be under-read."""

    verbose = ITEM + " " + "x" * (DEFAULT_MAX_ASSISTANT_CHARS + 1)
    verdict = vet_example(chat("q", verbose), indexes={"code": index_for(ITEM)})
    assert verdict.reason == "contaminated"


# --------------------------------------------------------- the reused caps ---

def test_long_code_with_short_prose_is_admitted():
    """The general cap applied to the whole assistant turn would refuse this,
    and a forty-line function is the payload phase 8 exists to add."""

    body = "\n".join(f"    x{i} = {i}" for i in range(400))
    answer = "here:\n" + fenced("python", f"def f():\n{body}")
    assert len(answer) > DEFAULT_MAX_ASSISTANT_CHARS
    verdict = vet_example(chat("q", answer))
    assert verdict.admitted
    assert verdict.prose_chars <= DEFAULT_MAX_ASSISTANT_CHARS


def test_long_prose_is_still_refused_at_the_preregistered_cap():
    verdict = vet_example(chat("q", "y" * (DEFAULT_MAX_ASSISTANT_CHARS + 1)))
    assert verdict.reason == "prose_too_long"
    assert str(DEFAULT_MAX_ASSISTANT_CHARS) in verdict.detail.replace(",", "")


def test_a_cot_marker_in_prose_is_refused():
    assert vet_example(chat("q", "Let me think about this.\nok")).reason == "long_cot"


def test_a_cot_marker_inside_a_code_comment_is_not_a_reasoning_trace():
    answer = "done:\n" + fenced("python", "# Step 1: parse the input\nx = 1")
    assert vet_example(chat("q", answer)).admitted


# ------------------------------------------------------------- token budget ---

class WordTokenizer:
    """Whitespace tokens, one id per distinct word. Enough for the encoder's
    contract: it only needs `encode(text, add_special_tokens=False)`."""

    def __init__(self):
        self.ids = {}

    def encode(self, text, add_special_tokens=False):
        out = []
        for word in text.split():
            out.append(self.ids.setdefault(word, len(self.ids) + 1))
        return out


def test_supervised_tokens_are_counted_when_a_tokenizer_is_supplied():
    verdict = vet_example(chat("a question", "an answer here"),
                          tokenizer=WordTokenizer())
    assert verdict.admitted
    assert verdict.supervised_tokens is not None
    assert 0 < verdict.supervised_tokens < verdict.total_tokens


def test_without_a_tokenizer_the_count_is_none_rather_than_zero():
    verdict = vet_example(chat("q", "a"))
    assert verdict.supervised_tokens is None
    assert share_report([verdict])["supervised_tokens"] is None


def test_an_example_over_the_token_budget_is_refused_with_a_reason():
    long_answer = " ".join(f"w{i}" for i in range(DEFAULT_MAX_LEN))
    verdict = vet_example(chat("q", long_answer), tokenizer=WordTokenizer(),
                          max_len=32, max_prose_chars=10**9)
    assert verdict.reason == "over_token_budget"


# ---------------------------------------------------------------- execution ---

def passing(*args, **kwargs):
    return {"status": "passed", "category": None, "detail": ""}


def failing(*args, **kwargs):
    return {"status": "failed", "category": "assertion_failed", "detail": "boom"}


def test_a_shipped_test_that_passes_admits_and_marks_the_example_executed():
    verdict = vet_example(chat("q", fenced("python", "def f():\n    return 1")),
                          test_code="assert f() == 1", execute=passing)
    assert verdict.admitted and verdict.executed
    assert verdict.execution["status"] == "passed"


def test_a_shipped_test_that_fails_is_refused_with_its_category():
    verdict = vet_example(chat("q", fenced("python", "def f():\n    return 2")),
                          test_code="assert f() == 1", execute=failing)
    assert verdict.reason == "execution_failed"
    assert "assertion_failed" in verdict.detail
    assert verdict.executed


def test_execution_is_refused_rather_than_guessed_when_the_block_is_ambiguous():
    answer = fenced("python", "def f():\n    return 1") + "\n" + \
        fenced("python", "print(f())")
    verdict = vet_example(chat("q", answer), test_code="assert f() == 1",
                          execute=passing)
    assert verdict.reason == "ambiguous_solution"
    assert not verdict.executed


def test_a_test_with_no_python_block_is_refused():
    verdict = vet_example(chat("q", fenced("javascript", "const x = 1")),
                          test_code="assert True", execute=passing)
    assert verdict.reason == "no_solution_block"


def test_without_a_test_the_example_is_admitted_syntax_checked_only():
    verdict = vet_example(chat("q", fenced("python", "def f():\n    return 1")))
    assert verdict.admitted
    assert not verdict.executed


def test_the_sandbox_is_not_reached_when_an_earlier_check_refuses():
    def explode(*args, **kwargs):               # pragma: no cover - must not run
        raise AssertionError("execution ran after a refusal")

    verdict = vet_example(chat("q", fenced("python", "def f(:")),
                          test_code="assert True", execute=explode)
    assert verdict.reason == "syntax_error"


# ------------------------------------------------------------------- report ---

def test_share_report_counts_reasons_and_publishes_the_honest_denominators():
    verdicts = [
        vet_example(chat("q", fenced("python", "def f():\n    return 1")),
                    tokenizer=WordTokenizer()),
        vet_example(chat("q", fenced("typescript", "const x = 1"))),
        vet_example(chat("q", fenced("python", "def f(:"))),
        vet_example([{"role": "user", "content": "hi"}]),
    ]
    report = share_report(verdicts)
    assert report["examples"] == 4
    assert report["admitted"] == 2
    assert report["refused"] == 2
    assert report["refusals"]["syntax_error"] == 1
    assert report["refusals"]["no_assistant"] == 1
    assert list(report["refusals"]) == list(ADMISSION_REASONS)
    assert set(report["code_language_shares"]) == {"python", "javascript-typescript"}
    assert sum(report["code_language_shares"].values()) == pytest.approx(1.0)
    assert 0.0 < report["syntax_checked_share"] < 1.0
    assert report["execution_tested_share"] == 0.0


def test_share_report_excludes_refused_examples_from_the_language_shares():
    """Otherwise the reported Python share is a property of the filter."""

    admitted = vet_example(chat("q", fenced("typescript", "const x = 1")))
    refused = vet_example(chat("q", fenced("python", "def f(:" + "\n" + " " * 4)))
    assert refused.reason == "syntax_error"
    report = share_report([admitted, refused])
    assert report["code_language_shares"] == {"javascript-typescript": 1.0}


def test_share_report_attributes_supervised_tokens_to_the_dominant_bucket():
    tokenizer = WordTokenizer()
    mostly_python = "ok\n" + fenced("python", "\n".join(f"a{i} = {i}" for i in range(20))) \
        + "\n" + fenced("javascript", "const b = 1")
    verdict = vet_example(chat("q", mostly_python), tokenizer=tokenizer)
    assert verdict.admitted
    assert verdict.primary_language == "python"
    report = share_report([verdict])
    assert report["supervised_tokens_by_language"] == {
        "python": verdict.supervised_tokens}
    assert report["supervised_tokens_counted_for"] == 1


def test_unknown_language_bytes_are_reported_separately_from_other():
    untagged = vet_example(chat("q", fenced("", "x = 1")))
    shell = vet_example(chat("q", fenced("bash", "ls -la")))
    report = share_report([untagged, shell])
    assert report["unknown_language_bytes"] == len("x = 1")
    assert report["code_language_bytes"][OTHER_LANGUAGE] == len("ls -la")
    assert report["code_language_shares"] == {}


def test_the_prose_cap_is_the_borrowed_one_plus_its_separators():
    """The claim is that this cap is `keep_example`'s, applied to prose. Both
    sum across every assistant turn, so removing blocks can only shrink the
    count -- the direction that makes the reuse more permissive on code, which
    is the whole point. The one deviation is the newline `parse_messages` joins
    turns with, and it is pinned here rather than left to be discovered."""

    from daedalus.chatml import assistant_char_count

    with_code = [{"role": "user", "content": "q"},
                 {"role": "assistant",
                  "content": "short\n" + fenced("python", "x = 1\n" * 200)}]
    prose, _ = parse_messages(with_code)
    assert len(prose) < assistant_char_count(with_code)

    turns = [{"role": "user", "content": "q"}]
    for _ in range(4):
        turns.append({"role": "assistant", "content": "answer"})
    prose, _ = parse_messages(turns)
    # Four assistant turns, three joining newlines, and no code to remove.
    assert len(prose) == assistant_char_count(turns) + 3


def test_a_deprecated_escape_is_admitted_and_prints_nothing(recwarn):
    """`"\\d"` in a non-raw string parses -- it is a deprecation, not a syntax
    error. Left unsuppressed, real code raises several per block and a build's
    log fills with warnings about conversations it accepted."""

    import ast
    import warnings

    verdict = vet_example(chat("q", fenced("python", 'p = "\\d+"')))
    assert verdict.admitted
    assert [w for w in recwarn if issubclass(w.category, SyntaxWarning)] == []

    # The test's own premise: this interpreter does warn on that source, so an
    # empty list above is the suppression working rather than the warning never
    # having existed.
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        ast.parse('p = "\\d+"')
    assert [w for w in raised if issubclass(w.category, SyntaxWarning)]


def test_code_block_bytes_are_utf8_not_characters():
    block = CodeBlock(tag="python", language="python", source="s = 'é'",
                      closed=True)
    assert block.bytes == len("s = 'é'".encode("utf-8")) == 8


# --------------------------------------------------------- dataset licences ---
#
# The gap this covers is that a *dataset* licence and a *source file* licence
# are different objects with overlapping vocabularies. `odc-by` is the only
# licence bigcode publishes under and never appears on a .py file; `mpl-2.0`
# appears on files constantly and is refused by both. Sharing one table would
# refuse the right sources for a reason that is not about them.

def test_attribution_only_data_licences_are_permissive():
    for value in ("apache-2.0", "mit", "odc-by", "cc-by-4.0", "cc0-1.0"):
        assert dataset_license_verdict(value) == "permissive", value


def test_share_alike_and_non_commercial_are_refused_by_name():
    for value in ("cc-by-sa-4.0", "cc-by-nc-4.0", "gpl-3.0", "other"):
        assert dataset_license_verdict(value) == "non-permissive", value


def test_an_absent_declaration_is_undeclared_not_unknown():
    """`HuggingFaceTB/smoltalk` carries no `license:` tag and no cardData
    entry, and that is a different finding from a string this table has never
    seen: one is fixed by widening the table, the other cannot be fixed here."""

    assert dataset_license_verdict(None) == UNDECLARED_LICENSE
    assert dataset_license_verdict("  ") == UNDECLARED_LICENSE
    # `HuggingFaceH4/CodeAlpaca_20K` declares the bare string `cc`, which names
    # no version and no clauses. Unknown, not non-permissive: nobody decided it
    # was refused, we just cannot tell what it is.
    assert dataset_license_verdict("cc") == "unknown"


def test_the_verdict_is_case_and_whitespace_insensitive():
    assert dataset_license_verdict(" Apache-2.0 ") == "permissive"


# ------------------------------------------------------------- source table ---

def test_the_live_table_is_permissive_and_covers_both_halves():
    """The module asserts this at import; asserting it again here is what makes
    a future edit to the table fail as a test rather than as an ImportError in
    whatever imported it first."""

    assert {s.half for s in SFT_SOURCES} == set(SFT_HALVES)
    assert all(s.license_verdict == "permissive" for s in SFT_SOURCES)
    assert len({s.key for s in SFT_SOURCES}) == len(SFT_SOURCES)


def test_the_two_halves_partition_one_dataset_rather_than_overlapping():
    """`self-oss-instruct` kept by the code half must be dropped by the general
    one. Left in both, the same rows are counted twice and the 65/35 the model
    card claims describes neither half."""

    code = [s for s in SFT_SOURCES if s.half == "code"]
    general = [s for s in SFT_SOURCES if s.half == "general"]
    for kept in code:
        for other in general:
            if other.dataset != kept.dataset:
                continue
            assert set(kept.keep_sources) <= set(other.drop_sources), \
                f"{other.key} does not drop what {kept.key} keeps"


def test_the_recorded_alternatives_carry_the_verdict_that_refused_them():
    by_name = {name: (declared, note) for name, declared, note
               in RECORDED_ALTERNATIVES}
    declared, note = by_name["HuggingFaceTB/smoltalk::self-oss-instruct"]
    assert declared == UNDECLARED_LICENSE
    assert "licence" in note
    # The upstream is usable and was measured; it is recorded as a choice, not
    # as a refusal, so nothing later reads it as unavailable.
    declared, _ = by_name["bigcode/self-oss-instruct-sc2-exec-filter-50k"]
    assert dataset_license_verdict(declared) == "permissive"


def test_a_source_refuses_both_filters_at_once():
    with pytest.raises(CodeSFTError):
        SFTSource(key="k", half="code", dataset="d", declared_license="mit",
                  note="", source_field="source", keep_sources=("a",),
                  drop_sources=("b",))


def test_a_source_filter_without_a_row_key_is_refused():
    with pytest.raises(CodeSFTError):
        SFTSource(key="k", half="code", dataset="d", declared_license="mit",
                  note="", keep_sources=("a",))


def test_a_source_half_must_be_one_the_plan_names():
    with pytest.raises(CodeSFTError):
        SFTSource(key="k", half="both", dataset="d", declared_license="mit",
                  note="")


# ------------------------------------------------------------- row adapters ---

CODE_SOURCE = SFTSource(key="code", half="code", dataset="d",
                        declared_license="mit", note="", source_field="source",
                        keep_sources=("self-oss-instruct",))
GENERAL_SOURCE = SFTSource(key="general", half="general", dataset="d",
                           declared_license="mit", note="",
                           source_field="source",
                           drop_sources=("self-oss-instruct",))


def test_the_source_filter_partitions_one_row_stream():
    row = {"source": "self-oss-instruct", "messages": chat("q", "a")}
    assert row_messages(CODE_SOURCE, row)[0] == chat("q", "a")
    assert row_messages(GENERAL_SOURCE, row) == (None, "other_source")

    other = {"source": "openhermes-50k", "messages": chat("q", "a")}
    assert row_messages(CODE_SOURCE, other) == (None, "other_source")
    assert row_messages(GENERAL_SOURCE, other)[0] == chat("q", "a")


def test_a_missing_source_key_is_its_own_refusal():
    """The silent-failure case: a row key the dataset does not have refuses
    every row, and `other_source` would report that as an ordinary low yield."""

    assert row_messages(CODE_SOURCE, {"messages": chat("q", "a")}) == \
        (None, "no_source_field")


def test_row_refusals_are_counted_apart_from_admission_refusals():
    """`other_source` is ~90% of the code half's stream by design. Folded into
    the admission counts it would report the gate as refusing nine rows in
    ten."""

    from daedalus.code_sft import ROW_REFUSALS

    assert not set(ROW_REFUSALS) & set(ADMISSION_REASONS)


def test_a_row_without_messages_is_refused_rather_than_raising():
    assert row_messages(CODE_SOURCE, {"source": "self-oss-instruct"}) == \
        (None, "no_messages")
    assert row_messages(CODE_SOURCE, {"source": "self-oss-instruct",
                                      "messages": [{"role": "user"}]}) == \
        (None, "no_messages")
    assert row_messages(CODE_SOURCE, "not a row") == (None, "not_a_mapping")


# ------------------------------------------------------------ shipped tests ---

ASSERTION = "assert f(2) == 4"


def test_a_user_turn_assertion_is_the_shipped_test():
    """`self-oss-instruct` states its test in the user turn -- "your code should
    pass the following assertion" -- and that block is the only execution
    evidence available: nothing here can manufacture a test."""

    messages = chat("write f\n" + fenced("python", ASSERTION),
                    fenced("python", "def f(x): return x * 2"))
    assert shipped_test(messages) == (ASSERTION, None)


def test_a_user_block_without_an_assertion_is_not_a_test():
    """It runs to completion whatever the assistant wrote, so admitting it
    would grow the execution-tested count with tests that cannot fail."""

    messages = chat("like this\n" + fenced("python", "f(2)"), "sure")
    assert shipped_test(messages) == (None, "no_assertion")


def test_two_user_blocks_are_ambiguous_rather_than_concatenated():
    messages = chat(fenced("python", "setup()") + "\n" + fenced("python", ASSERTION),
                    "ok")
    assert shipped_test(messages) == (None, "ambiguous_user_block")


def test_an_assistant_assertion_is_not_a_shipped_test():
    """The model writing its own assertion is not evidence about the model."""

    messages = chat("write f", fenced("python", "def f(x): return x\n" + ASSERTION))
    assert shipped_test(messages) == (None, "no_user_block")


def test_an_unterminated_user_fence_ships_no_test():
    messages = chat("```python\n" + ASSERTION, "ok")
    assert shipped_test(messages) == (None, "no_user_block")


# ------------------------------------------------------------------- probes ---

def rows_for(values):
    return [{"source": source, "messages": messages}
            for source, messages in values]


def stream_of(rows):
    return lambda source: list(rows)


def test_a_probe_reports_the_source_filter_separately_from_the_gate():
    rows = rows_for([
        ("self-oss-instruct", chat("q", fenced("python", "x = 1"))),
        ("self-oss-instruct", chat("q", fenced("python", "def f(:"))),
        ("openhermes-50k", chat("q", "prose")),
    ])
    record = probe_source(CODE_SOURCE, rows=10, stream=stream_of(rows))
    assert record["resolved"] and record["rows_offered"] == 3
    assert record["rows_kept"] == 2
    assert record["row_refusals"]["other_source"] == 1
    assert record["report"]["admitted"] == 1
    assert record["report"]["refusals"]["syntax_error"] == 1
    # The kept share is of *offered*, so the number that decides how much of a
    # dataset must be streamed to build a corpus is the one reported.
    assert record["kept_share"] == pytest.approx(2 / 3)


def test_a_probe_stops_at_the_row_budget():
    rows = rows_for([("self-oss-instruct", chat("q", "a"))] * 50)
    record = probe_source(CODE_SOURCE, rows=7, stream=stream_of(rows))
    assert record["rows_offered"] == 7


def test_a_probe_records_the_shipped_test_share_without_executing():
    """Execution costs a process per row at up to the timeout, so the probe
    publishes what *could* be run beside what was, and never lets the second
    read as a measurement when nothing ran."""

    tested = chat("run this\n" + fenced("python", ASSERTION),
                  fenced("python", "def f(x): return x * 2"))
    untested = chat("q", fenced("python", "x = 1"))
    rows = rows_for([("self-oss-instruct", tested),
                     ("self-oss-instruct", untested)])
    record = probe_source(CODE_SOURCE, rows=10, stream=stream_of(rows))
    assert record["executed"] is False
    assert record["shipped_tests"] == 1
    assert record["shipped_test_share"] == pytest.approx(0.5)
    assert record["report"]["execution_tested"] == 0
    assert record["no_test_reasons"]["no_user_block"] == 1


def test_the_shipped_test_count_and_its_share_carry_their_own_denominators():
    """`shipped_tests` is over kept rows, the share is over admitted ones. They
    differ by the test-carrying conversations the gate refused -- 18 of 556 on
    the live probe -- and reported as one number with the other's denominator
    that difference reads as an arithmetic slip."""

    tested = chat("run this\n" + fenced("python", ASSERTION),
                  fenced("python", "def f(x): return x * 2"))
    broken = chat("run this\n" + fenced("python", ASSERTION),
                  fenced("python", "def f(:"))
    rows = rows_for([("self-oss-instruct", tested),
                     ("self-oss-instruct", broken)])
    record = probe_source(CODE_SOURCE, rows=10, stream=stream_of(rows))
    assert record["shipped_tests"] == 2
    assert record["shipped_tests_admitted"] == 1
    assert record["report"]["admitted"] == 1
    assert record["shipped_test_share"] == pytest.approx(1.0)


def test_a_probe_executes_only_when_a_runner_is_supplied():
    calls = []

    def execute(solution, test_code, *, timeout_s, memory_mb):
        calls.append((solution, test_code))
        return {"status": "passed"}

    tested = chat("run this\n" + fenced("python", ASSERTION),
                  fenced("python", "def f(x): return x * 2"))
    record = probe_source(CODE_SOURCE, rows=10, execute=execute,
                          stream=stream_of(rows_for([("self-oss-instruct", tested)])))
    assert record["executed"] is True
    assert calls == [("def f(x): return x * 2", ASSERTION)]
    assert record["report"]["execution_tested"] == 1
    assert record["report"]["execution_tested_share"] == pytest.approx(1.0)


def test_a_probe_records_an_unreachable_source_rather_than_raising():
    def boom(source):
        raise OSError("404")

    record = probe_source(CODE_SOURCE, rows=3, stream=boom)
    assert record["resolved"] is False
    assert "OSError: 404" in record["error"]
    assert probe_problems({"sources": [record]})[0].startswith(
        "code did not resolve")


def test_a_filter_on_a_key_the_rows_lack_is_a_fatal_problem():
    """Every row refused, an empty build, a zero exit -- the failure the probe
    exists to catch, so it must not be reported as a low yield."""

    rows = [{"messages": chat("q", "a")} for _ in range(5)]
    record = probe_source(CODE_SOURCE, rows=10, stream=stream_of(rows))
    problems = probe_problems({"sources": [record]})
    assert any("'source' that its rows do not have" in problem
               for problem in problems)


def test_a_non_permissive_declaration_is_a_problem_even_when_rows_flow():
    source = SFTSource(key="k", half="code", dataset="d",
                       declared_license="cc-by-sa-4.0", note="")
    record = probe_source(source, rows=5, stream=stream_of(
        [{"messages": chat("q", fenced("python", "x = 1"))}]))
    problems = probe_problems({"sources": [record]})
    assert any("non-permissive" in problem for problem in problems)


def test_a_probe_of_one_half_reports_the_other_as_missing():
    """The plan asks for code *and* general SFT; a probe that resolved only one
    of them and said "every source resolved" would be the reassuring kind of
    success line."""

    rows = rows_for([("self-oss-instruct", chat("q", fenced("python", "x = 1")))])
    report = probe_sources([CODE_SOURCE], rows=5, stream=stream_of(rows))
    problems = probe_problems(report)
    assert any("general half kept no rows" in problem for problem in problems)


def test_a_clean_probe_of_both_halves_has_no_problems():
    rows = rows_for([
        ("self-oss-instruct", chat("q", fenced("python", "x = 1"))),
        ("openhermes-50k", chat("q", "an ordinary answer")),
    ])
    report = probe_sources([CODE_SOURCE, GENERAL_SOURCE], rows=5,
                           stream=stream_of(rows))
    assert probe_problems(report) == []
    assert [record["half"] for record in report["sources"]] == \
        ["code", "general"]
    assert report["alternatives"][0]["license_verdict"] == UNDECLARED_LICENSE


def test_a_probe_histograms_the_source_column_it_filters_on():
    """Which values a mixture actually carries is the measurement that decided
    the table, and a build cannot re-derive it after filtering them out."""

    rows = rows_for([("self-oss-instruct", chat("q", "a"))] * 2 +
                    [("openhermes-50k", chat("q", "a"))] * 3)
    record = probe_source(CODE_SOURCE, rows=10, stream=stream_of(rows))
    assert record["source_values"] == {"openhermes-50k": 3,
                                       "self-oss-instruct": 2}


def test_a_probe_passes_the_contamination_indexes_through_by_name():
    ngrams = ngram_set("a benchmark prompt that must never enter training data",
                       n=6)
    rows = rows_for([("self-oss-instruct",
                      chat("a benchmark prompt that must never enter training data",
                           fenced("python", "x = 1")))])
    record = probe_source(CODE_SOURCE, rows=5, stream=stream_of(rows),
                          indexes={"code": ngrams}, n=6)
    assert record["report"]["refusals"]["contaminated"] == 1


# -------------------------------------------------------------- the build ---
#
# The probe answers "would a build do what it says". These are the build, and
# the cases worth pinning are the ones where it writes a *plausible* set that
# is not the one its manifest describes: a mixture nobody chose because one
# half ran short, a holdout the trainer can reach, a conversation funding both
# halves, or an execution-tested share that was never executed.

CLEAN_INDEX = {"code": index_for(
    "a benchmark item that appears in no row of any fixture in this module")}


def code_row(i: int, *, tested: bool = True) -> dict:
    """One `self-oss-instruct`-shaped row: the test in the user turn."""

    user = f"please write f{i}"
    if tested:
        user += "\n" + fenced("python", f"assert f{i}(2) == 4")
    return {"source": "self-oss-instruct",
            "messages": chat(user, fenced("python",
                                          f"def f{i}(x):\n    return x * 2"))}


def general_row(i: int) -> dict:
    return {"source": "openhermes-50k",
            "messages": chat(f"question {i}", f"answer number {i}")}


def interleaved(code: int, general: int, *, tested: bool = True) -> list:
    rows = []
    for i in range(max(code, general)):
        if i < code:
            rows.append(code_row(i, tested=tested))
        if i < general:
            rows.append(general_row(i))
    return rows


def build(tmp_path, rows, **kwargs):
    """`build_dataset` over a fixture stream, with the two live halves' shape."""

    kwargs.setdefault("out_dir", tmp_path / "set")
    kwargs.setdefault("tokenizer", WordTokenizer())
    kwargs.setdefault("indexes", CLEAN_INDEX)
    kwargs.setdefault("execute", passing)
    kwargs.setdefault("holdout_examples", 2)
    kwargs.setdefault("supervised_tokens", 1_000)
    kwargs.setdefault("progress_every", 0)
    kwargs.setdefault("log", lambda line: None)
    return build_dataset([CODE_SOURCE, GENERAL_SOURCE],
                         stream=stream_of(rows), **kwargs)


def lines_of(path):
    return [json.loads(line) for line in
            Path(path).read_text().splitlines() if line]


def test_a_build_writes_a_training_set_a_holdout_and_a_manifest(tmp_path):
    manifest = build(tmp_path, interleaved(60, 60))
    out = tmp_path / "set"
    assert (out / MANIFEST_FILE).exists()

    train = lines_of(out / TRAIN_FILE)
    holdout = lines_of(out / HOLDOUT_FILE)
    assert len(train) == manifest["train_examples"]
    assert len(holdout) == manifest["holdout_examples"] == 4
    assert manifest["train_supervised_tokens"] == \
        sum(row["supervised_tokens"] for row in train)
    assert {row["half"] for row in train} == {"code", "general"}


def test_the_written_rows_are_the_shape_the_trainer_already_reads(tmp_path):
    """`post.iter_chat_examples` reads `row["messages"]` off whatever it is
    handed, so a built file is training data with no adapter between them. A
    record that needed one is a second rendering of the conversations, which is
    the reason the general half is read from the released model's own dataset."""

    from post import iter_chat_examples

    manifest = build(tmp_path, interleaved(40, 40))
    rows = lines_of(tmp_path / "set" / TRAIN_FILE)
    kept = list(iter_chat_examples(rows, max_assistant_chars=10 ** 9,
                                   drop_cot=False))
    assert len(kept) == manifest["train_examples"] > 0
    assert all(m["role"] in {"user", "assistant"}
               for messages in kept for m in messages)


def test_the_holdout_is_taken_first_and_the_trainer_cannot_reach_it(tmp_path):
    """Held out off the head of the same stream, so a held-out conversation is
    one no epoch can reach -- `post.take_eval_pairs`' property. Any other route
    relies on two reads of a streaming dataset agreeing about order."""

    build(tmp_path, interleaved(40, 40), holdout_examples=5)
    out = tmp_path / "set"
    train = lines_of(out / TRAIN_FILE)
    holdout = lines_of(out / HOLDOUT_FILE)

    def key(row):
        return json.dumps(row["messages"], sort_keys=True)

    assert not {key(row) for row in holdout} & {key(row) for row in train}
    for half in ("code", "general"):
        assert sum(1 for row in holdout if row["half"] == half) == 5


def test_the_holdout_does_not_spend_the_training_budget(tmp_path):
    """Otherwise the set is short by however much was held out, and the
    shortfall reads as a stream that ran out."""

    plain = build(tmp_path / "a", interleaved(400, 400), holdout_examples=0)
    held = build(tmp_path / "b", interleaved(400, 400), holdout_examples=5)
    assert plain["problems"] == held["problems"] == []
    assert held["holdout_examples"] == 10
    assert held["train_supervised_tokens"] == plain["train_supervised_tokens"]


def test_both_halves_of_one_dataset_are_served_by_a_single_pass(tmp_path):
    """Two reads of a streaming dataset are two chances for a Hub-side change
    between them to make the halves overlap. One pass makes the partition a
    property of the rows this build saw."""

    calls = []

    def stream(source):
        calls.append(source.key)
        return interleaved(30, 30)

    build_dataset([CODE_SOURCE, GENERAL_SOURCE], out_dir=tmp_path / "set",
                  tokenizer=WordTokenizer(), indexes=CLEAN_INDEX,
                  execute=passing, holdout_examples=1, supervised_tokens=500,
                  stream=stream, progress_every=0, log=lambda line: None)
    assert len(calls) == 1


def test_sources_group_by_the_rows_they_read(tmp_path):
    other = SFTSource(key="other", half="general", dataset="second",
                      declared_license="mit", note="")
    groups = group_sources([CODE_SOURCE, GENERAL_SOURCE, other])
    assert len(groups) == 2
    assert source_group_key(CODE_SOURCE) == source_group_key(GENERAL_SOURCE)


def test_a_row_two_sources_claim_is_fatal_rather_than_assigned(tmp_path):
    """Whichever half it was assigned to, the shares would then describe a
    corpus nobody built -- and the count that would say so is the one a
    best-effort assignment throws away."""

    greedy = SFTSource(key="greedy", half="general", dataset="d",
                       declared_license="mit", note="", source_field="source",
                       keep_sources=("self-oss-instruct",))
    manifest = build_dataset([CODE_SOURCE, greedy], out_dir=tmp_path / "set",
                             tokenizer=WordTokenizer(), indexes=CLEAN_INDEX,
                             execute=passing, holdout_examples=0,
                             supervised_tokens=500,
                             stream=stream_of(interleaved(10, 10)),
                             progress_every=0, log=lambda line: None)
    assert manifest["overlapping_rows"] == 10
    assert any("more than one source" in problem
               for problem in manifest["problems"])


def test_a_short_half_trims_the_other_rather_than_shipping_a_new_mixture(tmp_path):
    """The share is borrowed from a preregistered quantity and the budget is
    this build's own choice, so the budget is what gives way. A correctly sized
    set at a mixture nobody chose is the failure that would not announce
    itself."""

    manifest = build(tmp_path, interleaved(20, 400), code_share=0.5,
                     supervised_tokens=10 ** 7, holdout_examples=0)
    halves = manifest["halves"]
    assert halves["code"]["trimmed"] is False
    assert halves["general"]["trimmed"] is True
    assert halves["general"]["supervised_tokens"] < \
        halves["general"]["supervised_tokens_admitted"]
    assert manifest["realized_code_share"] == pytest.approx(0.5, abs=0.1)
    assert any("of 5,000,000 supervised tokens" in problem
               for problem in manifest["problems"])


def test_a_full_build_holds_the_share_it_was_asked_for(tmp_path):
    manifest = build(tmp_path, interleaved(400, 400), code_share=0.65,
                     supervised_tokens=2_000, holdout_examples=0)
    assert manifest["problems"] == []
    assert manifest["realized_code_share"] == pytest.approx(0.65, abs=0.05)
    for block in manifest["halves"].values():
        # The fill overshoots by up to one conversation and the trim then cuts
        # both halves to exactly proportional, so it is the fill that answers
        # "did this half's stream fund its share".
        assert block["supervised_tokens_admitted"] >= block["budget"]


def test_a_filled_half_stops_costing_the_pass(tmp_path):
    """Vetting runs a sandboxed process per shipped test, so a half that has
    met its budget must stop paying for rows it will not write. The count is
    kept apart from every refusal: nothing was wrong with those rows."""

    manifest = build(tmp_path, interleaved(400, 400), supervised_tokens=400,
                     holdout_examples=0)
    assert manifest["rows_offered"] < 800
    after = {record["key"]: record["rows_after_budget"]
             for record in manifest["sources"]}
    assert sum(after.values()) > 0
    for record in manifest["sources"]:
        assert record["rows_kept"] == \
            record["written_train"] + record["written_holdout"] + \
            record["report"]["refused"]


def test_the_shipped_tests_are_run_and_counted_as_execution_tested(tmp_path):
    """"Syntax-checked *and* execution-tested" is a measured fraction, not an
    assumption: the general half ships no tests at all."""

    manifest = build(tmp_path, interleaved(30, 30), holdout_examples=0)
    by_key = {record["key"]: record for record in manifest["sources"]}
    code, general = by_key["code"], by_key["general"]
    assert code["report"]["execution_tested"] == code["report"]["admitted"] > 0
    assert code["report"]["execution_tested_share"] == 1.0
    assert general["report"]["execution_tested"] == 0
    assert all(row["execution_tested"] for row in
               lines_of(tmp_path / "set" / TRAIN_FILE) if row["half"] == "code")


def test_a_shipped_test_that_fails_keeps_the_conversation_out(tmp_path):
    manifest = build(tmp_path, interleaved(30, 30), execute=failing,
                     holdout_examples=0)
    code = next(r for r in manifest["sources"] if r["key"] == "code")
    assert code["report"]["refusals"]["execution_failed"] == 30
    assert code["report"]["admitted"] == 0
    assert any("refused every one" in problem for problem in manifest["problems"])


def test_without_a_runner_the_tests_are_counted_but_never_run(tmp_path):
    """`execution_tested` must never read as a measurement when nothing ran, so
    what *could* have been executed is reported separately from what was."""

    manifest = build(tmp_path, interleaved(30, 30), execute=None,
                     holdout_examples=0)
    code = next(r for r in manifest["sources"] if r["key"] == "code")
    assert manifest["execution"]["enabled"] is False
    assert code["executed"] is False
    assert code["shipped_tests"] == 30
    assert code["report"]["execution_tested"] == 0


def test_the_training_file_is_shuffled_deterministically(tmp_path):
    """A stream grouped by source writes a file grouped by half, and a bounded
    shuffle buffer in the trainer cannot undo that -- the optimizer would see a
    curriculum nobody designed. The seed is what makes two arms comparable."""

    blocked = [code_row(i) for i in range(30)] + \
        [general_row(i) for i in range(30)]
    first = build(tmp_path / "a", blocked, holdout_examples=0, seed=0)
    again = build(tmp_path / "b", blocked, holdout_examples=0, seed=0)
    assert (tmp_path / "a" / "set" / TRAIN_FILE).read_bytes() == \
        (tmp_path / "b" / "set" / TRAIN_FILE).read_bytes()

    halves = [row["half"] for row in lines_of(tmp_path / "a" / "set" / TRAIN_FILE)]
    assert halves != sorted(halves, key=["code", "general"].index)
    assert first["train_examples"] == again["train_examples"]


def test_a_different_seed_reorders_the_same_conversations(tmp_path):
    rows = interleaved(30, 30)
    build(tmp_path / "a", rows, holdout_examples=0, seed=0)
    build(tmp_path / "b", rows, holdout_examples=0, seed=7)

    def written(name):
        return [json.dumps(row["messages"], sort_keys=True)
                for row in lines_of(tmp_path / name / "set" / TRAIN_FILE)]

    assert sorted(written("a")) == sorted(written("b"))
    assert written("a") != written("b")


def test_the_manifest_answers_the_problems_without_the_build(tmp_path):
    """`build_problems` is re-askable of the artifact by a reader who did not
    run the build -- `mixture_opt`'s rule for a rule that judges an artifact."""

    manifest = build(tmp_path, interleaved(20, 400), code_share=0.5,
                     supervised_tokens=10 ** 7, holdout_examples=0)
    on_disk = json.loads((tmp_path / "set" / MANIFEST_FILE).read_text())
    assert build_problems(on_disk) == manifest["problems"] != []


def test_the_manifest_records_what_filtered_the_set(tmp_path):
    provenance = {"code": {"path": "data/decontam/code.gz", "digest": "sha256:ab",
                           "n": 13, "ngrams": 34286}}
    manifest = build(tmp_path, interleaved(20, 20), holdout_examples=0,
                     index_provenance=provenance)
    assert manifest["decontam_indexes"] == ["code"]
    assert manifest["decontam_provenance"] == provenance
    assert manifest["tokenizer"]["vocab_size"] is None      # the fixture's


def test_a_contaminated_conversation_never_reaches_the_file(tmp_path):
    rows = interleaved(10, 10)
    rows.append({"source": "openhermes-50k", "messages": chat("q", ITEM)})
    manifest = build(tmp_path, rows, indexes={"code": index_for(ITEM)},
                     holdout_examples=0)
    general = next(r for r in manifest["sources"] if r["key"] == "general")
    assert general["report"]["refusals"]["contaminated"] == 1
    written = Path(tmp_path / "set" / TRAIN_FILE).read_text()
    assert "has_close_elements" not in written


# ------------------------------------------------------- the build refuses ---

def test_a_build_without_a_tokenizer_is_refused(tmp_path):
    """The mixture is measured in supervised tokens and `over_token_budget` is
    the only place an example the encoder would silently drop gets a reason."""

    with pytest.raises(CodeSFTError, match="tokenizer"):
        build(tmp_path, interleaved(5, 5), tokenizer=None)


def test_a_build_without_the_decontamination_indexes_is_refused(tmp_path):
    with pytest.raises(CodeSFTError, match="decontamination"):
        build(tmp_path, interleaved(5, 5), indexes={})


def test_a_build_weighting_a_half_it_has_no_source_for_is_refused(tmp_path):
    with pytest.raises(CodeSFTError, match="every half"):
        build_dataset([CODE_SOURCE], out_dir=tmp_path / "set",
                      tokenizer=WordTokenizer(), indexes=CLEAN_INDEX,
                      stream=stream_of(interleaved(5, 5)), progress_every=0,
                      log=lambda line: None)


def test_a_degenerate_share_is_refused_rather_than_streamed_and_discarded():
    for share in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(CodeSFTError):
            half_shares(share)
    assert half_shares(0.65) == {"code": 0.65, "general": pytest.approx(0.35)}
    assert half_shares()["code"] == DEFAULT_CODE_SHARE


def test_a_build_refuses_to_write_a_second_set_beside_the_first(tmp_path):
    build(tmp_path, interleaved(10, 10), holdout_examples=0)
    with pytest.raises(CodeSFTError, match="already exists"):
        build(tmp_path, interleaved(10, 10), holdout_examples=0)
    manifest = build(tmp_path, interleaved(10, 10), holdout_examples=0,
                     overwrite=True)
    assert manifest["train_examples"] > 0


def test_a_holdout_shorter_than_asked_for_is_a_problem(tmp_path):
    """A gate scored on whatever the stream happened to yield first is scored
    on a denominator nobody chose."""

    manifest = build(tmp_path, interleaved(3, 3), holdout_examples=10)
    assert any("held out" in problem for problem in manifest["problems"])


def test_an_unreachable_source_is_recorded_rather_than_raising(tmp_path):
    def boom(source):
        raise OSError("404")

    manifest = build_dataset([CODE_SOURCE, GENERAL_SOURCE],
                             out_dir=tmp_path / "set",
                             tokenizer=WordTokenizer(), indexes=CLEAN_INDEX,
                             stream=boom, progress_every=0,
                             log=lambda line: None)
    assert all("OSError: 404" in record["error"]
               for record in manifest["sources"])
    assert any("did not resolve" in problem for problem in manifest["problems"])


# ------------------------------------------------------ the shared counters ---

def test_the_streaming_counters_are_the_batch_report_s_own():
    """`share_report` is what every other test in this file pins, and a build
    that counted separately would be free to disagree with the probe that
    authorised it."""

    verdicts = [
        vet_example(chat("q", fenced("python", "def f():\n    return 1")),
                    tokenizer=WordTokenizer()),
        vet_example(chat("q", fenced("typescript", "const x = 1"))),
        vet_example(chat("q", fenced("python", "def f(:"))),
        vet_example(chat("q", fenced("python", "def f():\n    return 1")),
                    test_code="assert f() == 1", execute=passing),
    ]
    accumulator = ShareAccumulator()
    for verdict in verdicts:
        assert accumulator.add(verdict) is verdict
    assert accumulator.result() == share_report(verdicts)
