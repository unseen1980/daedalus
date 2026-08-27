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

import pytest

from daedalus.chatml import DEFAULT_MAX_ASSISTANT_CHARS
from daedalus.code_sft import (ADMISSION_REASONS, DEFAULT_MAX_LEN,
                               LANGUAGE_BUCKETS, OTHER_LANGUAGE,
                               SYNTAX_CHECKERS, UNKNOWN_LANGUAGE, CodeBlock,
                               contamination_hit, language_of, normalize_tag,
                               parse_markdown, parse_messages, share_report,
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


def test_code_block_bytes_are_utf8_not_characters():
    block = CodeBlock(tag="python", language="python", source="s = 'é'",
                      closed=True)
    assert block.bytes == len("s = 'é'".encode("utf-8")) == 8
