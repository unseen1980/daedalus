"""Phase 8 step 6's admission gate: which conversations may be code SFT data.

The manifest's wording is "run code/general SFT on **syntax-checked and
execution-tested** conversations. Track supervised-token counts and code-language
shares." This module is the filter that phrase names, written before any
candidate source has been read -- the same discipline `code_gates` was written
under, and for the same reason: a filter tuned after seeing which examples it
drops is a filter tuned to keep a corpus size.

Everything here is pure and CPU-only apart from one injected call-out to the
`code_eval` sandbox, so the parts with real correctness risk are testable without
a GPU, a network or a dataset.

What the gate refuses, and why each refusal exists
--------------------------------------------------

Each reason in `ADMISSION_REASONS` is a way a conversation makes the model
*worse* while looking like ordinary training data:

  - `malformed` -- a row without roles and contents. Skipping it silently is how
    a source that changed its schema becomes a corpus that trained on nothing.
  - `no_assistant` -- no supervised token at all; an all-`-100` row that costs a
    batch slot and returns a zero gradient (`chatml.keep_example`'s reasoning).
  - `unterminated_code_fence` -- an assistant turn whose ``` never closes. The
    answer was truncated by whoever built the source, and training on it teaches
    the model to open a block and run forever. Invisible in the loss.
  - `contaminated` -- any turn, from any role, carrying a benchmark n-gram.
  - `syntax_error` -- an assistant code block that does not parse.
  - `long_cot` / `prose_too_long` -- AGENT.md SS4's two content filters, applied
    to prose only. See "Reusing the general caps" below.
  - `over_token_budget` -- would not fit `max_len`, so `encode_sft_example`
    returns None. Dropped here, with a reason and a count, rather than
    disappearing inside the encoder.
  - `no_solution_block` / `ambiguous_solution` -- a source shipped a test but the
    assistant turn has no single Python block to run it against. Attributing an
    execution result to the wrong block is worse than dropping the example.
  - `execution_failed` -- the shipped test did not pass against the shipped
    answer.

Three design decisions that a plausible implementation gets wrong
----------------------------------------------------------------

**Only assistant blocks are syntax-checked.** A user turn legitimately contains
broken code -- "why does this raise?" is a debugging conversation, and those are
among the most valuable examples in a code SFT set. Checking every role would
drop exactly them, and the corpus would come back smaller in a way that looked
like a quality filter working.

**Contamination is checked on every role.** The mirror of the above: a HumanEval+
prompt pasted into a *user* turn is the benchmark item entering training, whoever
typed it. `is_contaminated` is asked of each message's content.

**Untagged code blocks are never guessed at.** An untagged block is recorded as
`UNKNOWN_LANGUAGE`, is not attributed to a language bucket and is not
syntax-checked, and its byte share is reported. The tempting alternative --
"`ast.parse` succeeds, so call it Python" -- over-attributes to the one bucket
where over-claiming is a live reporting hazard: `x = 1` parses as Python and as
half a dozen other languages, the corpus is 55% Python by design, and both
execution benchmarks are Python-only. A share that is honestly unknown is worth
more than a share that is confidently wrong, so the number is published instead
of removed.

Reusing the general caps rather than inventing code ones
--------------------------------------------------------

`chatml.keep_example` caps assistant *characters* at 1,200 and drops
chain-of-thought markers. Applied unchanged, that cap would drop nearly every
useful code answer -- a forty-line function is 1,200 characters on its own -- so
the filter meant to remove rambling would instead remove the payload this phase
exists to add. Applied not at all, there is no conciseness filter left.

The cap is therefore reused at its preregistered value and applied to
**prose**: assistant content with fenced blocks removed. That is what AGENT.md
SS4's number was about -- "biased toward concise replies. No long
chain-of-thought traces" is a statement about explanation, not about payload --
and it borrows an existing preregistered number instead of writing a new one,
which is the rule this program has followed every time a gate arrived without a
threshold. The CoT marker scan moves with it for the same reason and one of its
own: markers like `step 1:` and `reasoning:` appear in ordinary code comments,
so scanning inside blocks would drop well-commented answers as reasoning traces.

`max_len` is *not* reinterpreted: it stays the encoder's own token budget over
the whole rendered example, blocks included. The character cap bounds verbosity;
the token budget bounds what fits.

One difference from `assistant_char_count` is carried rather than hidden. Both
sum across *every* assistant turn, so the denominators match, and removing
fenced blocks can only shrink this one -- which is what makes the reuse strictly
more permissive on code. But `parse_messages` joins the per-turn prose with a
newline, so a conversation with `k` assistant turns is measured one character
per extra turn above the sum `keep_example` would take. It is left that way: the
separator is what stops a chain-of-thought marker being fabricated across a turn
boundary ("...step" + "1: ..."), the overshoot is at most a few characters
against a 1,200-character cap, and it can only ever refuse a conversation
sitting within `k-1` characters of the bound -- never admit one. Pinned by
`test_the_prose_cap_is_the_borrowed_one_plus_its_separators` so it stays a
recorded property rather than a later surprise.

Order of checks
---------------

A conversation can fail several ways at once, so the first matching reason in the
fixed `ADMISSION_REASONS` order is the one reported, and the counts in
`share_report` are exclusive by construction. Contamination is checked before the
content filters so that a contaminated example can never be counted as one
dropped for being verbose -- the contamination count is the one that must not be
under-read. Execution is last because it is the only check that costs a process.

Where the conversations come from
---------------------------------

`SFT_SOURCES` is the table the gate is pointed at, and like every other table in
this phase it names what to *ask* for rather than what exists:
`scripts/code_sft.py probe` resolves each entry against real rows before a
dataset is built from it. The measurements that chose the entries are in
`RECORDED_ALTERNATIVES` beside the ones that did not, because a table showing
only the current answer cannot be used to re-derive how the answer was reached.
"""

from __future__ import annotations

import ast
import collections.abc
import warnings
from dataclasses import dataclass, field
from typing import (Callable, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Tuple)

from daedalus.chatml import (DEFAULT_COT_MARKERS, DEFAULT_MAX_ASSISTANT_CHARS,
                             IGNORE_INDEX, encode_sft_example)
from daedalus.codeprep import CODE_LANGUAGE_SHARES, DEFAULT_CODE_N

SCHEMA = 1

#: post.py's own `--max-len` default. Named here so the gate and the trainer
#: cannot drift: an example admitted at one budget and encoded at a smaller one
#: is dropped inside the encoder, which returns None rather than truncating.
DEFAULT_MAX_LEN = 2048

#: A tagged block whose language is not one the code corpus carries -- bash,
#: json, sql, html. Not a defect and not attributed: the corpus is Python and
#: JavaScript/TypeScript, and a shell snippet inside a Python answer is context.
OTHER_LANGUAGE = "other"

#: A block with no info string at all. Deliberately distinct from
#: `OTHER_LANGUAGE`: "other" is a language we decided not to carry, "unknown" is
#: information the source did not provide, and only the second one is a
#: measurement gap worth reporting.
UNKNOWN_LANGUAGE = "unknown"

#: Fence info strings, normalised onto one canonical tag per language. Kept
#: separate from the bucket map below because the syntax checkers are per
#: *language* (a TypeScript checker is not a JavaScript one) while the corpus
#: shares are per *bucket*.
TAG_ALIASES: Dict[str, str] = {
    "py": "python", "python3": "python", "python2": "python",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "jsx": "javascript", "node": "javascript", "nodejs": "javascript",
    "ts": "typescript", "tsx": "typescript",
}

#: Canonical tag -> the code corpus's own bucket vocabulary
#: (`codeprep.CODE_LANGUAGE_SHARES`). Shared on purpose: "code-language shares"
#: is one table for the corpus and the SFT set, so the two can be compared
#: without a translation step that could disagree with itself.
LANGUAGE_BUCKETS: Dict[str, str] = {
    "python": "python",
    "javascript": "javascript-typescript",
    "typescript": "javascript-typescript",
}

assert set(LANGUAGE_BUCKETS.values()) <= set(CODE_LANGUAGE_SHARES), \
    "every SFT language bucket must be one the code corpus carries"


class CodeSFTError(ValueError):
    """Raised when the gate is asked something it must not answer."""


def _python_syntax(source: str) -> Tuple[bool, Optional[str]]:
    """`(ok, message)`. Parsed, never executed -- `code_eval.check_syntax`'s
    contract, reimplemented here so importing this module does not pull in the
    evaluation harness.

    `SyntaxWarning` is suppressed rather than surfaced. `"\\d"` in a non-raw
    string is a deprecation, not a syntax error: the block parses, it is
    admitted, and the warning says nothing about the decision. Left on, real
    code triggers several per block and a build over tens of thousands of
    conversations writes a log in which its own progress lines cannot be found
    -- the first live probe printed twelve of them before its first result.
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(source)
    except SyntaxError as error:
        return False, f"SyntaxError: {error.msg} (line {error.lineno})"
    except ValueError as error:                 # e.g. source with null bytes
        return False, f"{type(error).__name__}: {error}"
    return True, None


#: Canonical tag -> syntax checker. Python only, and the gap is deliberate
#: rather than hidden: this box has no bundled JavaScript or TypeScript parser,
#: so blocks in the 45% bucket are recorded as *unchecked* and their byte share
#: is published by `share_report`. Reporting them as checked would be the one
#: outcome worse than not checking them. A later slice can add a `node --check`
#: entry here without touching anything else.
SYNTAX_CHECKERS: Dict[str, Callable[[str], Tuple[bool, Optional[str]]]] = {
    "python": _python_syntax,
}

#: In the order they are checked; `share_report` counts them in this order too.
ADMISSION_REASONS = (
    "malformed",
    "no_assistant",
    "unterminated_code_fence",
    "contaminated",
    "syntax_error",
    "long_cot",
    "prose_too_long",
    "over_token_budget",
    "no_solution_block",
    "ambiguous_solution",
    "execution_failed",
)


@dataclass(frozen=True)
class CodeBlock:
    """One fenced block of an assistant or user turn."""

    tag: str                    #: canonical info-string tag, "" when untagged
    language: str               #: corpus bucket, `OTHER_` or `UNKNOWN_LANGUAGE`
    source: str                 #: block body, without the fence lines
    closed: bool                #: False when the fence never closed
    role: str = "assistant"

    @property
    def bytes(self) -> int:
        return len(self.source.encode("utf-8"))

    @property
    def checkable(self) -> bool:
        return self.tag in SYNTAX_CHECKERS


@dataclass(frozen=True)
class Parsed:
    """A message's prose and its fenced blocks, from one scan.

    Produced together rather than by two passes: the character cap is applied to
    the prose and the syntax check to the blocks, and two scanners that disagreed
    about where a fence ended would apply both to overlapping text.
    """

    prose: str
    blocks: Tuple[CodeBlock, ...]


def normalize_tag(info: str) -> str:
    """The canonical language tag of a fence info string, or "".

    Only the first whitespace-separated word is read -- ```` ```python
    {.line-numbers} ```` is Python -- and anything unrecognised is returned
    normalised but unmapped, so `LANGUAGE_BUCKETS` decides what is carried and
    this function only decides what was written.
    """

    word = (info or "").strip().split()[:1]
    if not word:
        return ""
    tag = word[0].strip().lower().lstrip("{.").rstrip("}")
    return TAG_ALIASES.get(tag, tag)


def language_of(tag: str) -> str:
    """The corpus bucket a tag belongs to, or `OTHER_`/`UNKNOWN_LANGUAGE`."""

    if not tag:
        return UNKNOWN_LANGUAGE
    return LANGUAGE_BUCKETS.get(tag, OTHER_LANGUAGE)


def parse_markdown(text: str, *, role: str = "assistant") -> Parsed:
    """Split one message into prose and fenced code blocks.

    A line scan rather than a regular expression, because the two cases that
    matter are exactly the ones a `(.*?)` pattern gets wrong: an *indented*
    fence, whose closing line is also indented, and an *unterminated* fence,
    which a lazy pattern silently treats as absent and a greedy one swallows the
    rest of the message into.

    CommonMark's rule is followed for the closing fence -- at least as many
    backticks as the opening one and nothing else on the line -- so a block
    opened with ```` ```` ```` is not closed by a ``` inside it.
    """

    lines = text.split("\n")
    prose: List[str] = []
    blocks: List[CodeBlock] = []
    body: List[str] = []
    fence_len = 0
    info = ""
    open_at = -1

    for index, line in enumerate(lines):
        stripped = line.strip()
        ticks = len(stripped) - len(stripped.lstrip("`"))
        if open_at < 0:
            if ticks >= 3 and "`" not in stripped[ticks:]:
                fence_len = ticks
                info = stripped[ticks:]
                open_at = index
                body = []
            else:
                prose.append(line)
            continue
        if ticks >= fence_len and not stripped[ticks:]:
            tag = normalize_tag(info)
            blocks.append(CodeBlock(tag=tag, language=language_of(tag),
                                    source="\n".join(body), closed=True,
                                    role=role))
            open_at = -1
            continue
        body.append(line)

    if open_at >= 0:
        tag = normalize_tag(info)
        blocks.append(CodeBlock(tag=tag, language=language_of(tag),
                                source="\n".join(body), closed=False, role=role))
    return Parsed(prose="\n".join(prose), blocks=tuple(blocks))


def parse_messages(messages: Sequence[Mapping[str, str]]
                   ) -> Tuple[str, Tuple[CodeBlock, ...]]:
    """`(assistant prose, every block of every role)`.

    The prose is assistant-only because that is what the character cap and the
    chain-of-thought scan are about; the blocks carry their `role` so the
    syntax check can select assistant ones and the contamination check does not
    have to.
    """

    prose: List[str] = []
    blocks: List[CodeBlock] = []
    for message in messages:
        parsed = parse_markdown(message.get("content") or "",
                                role=message.get("role") or "")
        blocks.extend(parsed.blocks)
        if message.get("role") == "assistant":
            prose.append(parsed.prose)
    return "\n".join(prose), tuple(blocks)


def has_long_cot_prose(prose: str,
                       markers: Sequence[str] = DEFAULT_COT_MARKERS) -> bool:
    lowered = prose.lower()
    return any(marker in lowered for marker in markers)


def contamination_hit(messages: Sequence[Mapping[str, str]],
                      indexes: Mapping[str, Set[str]],
                      n: int = DEFAULT_CODE_N) -> Optional[str]:
    """The name of the first index any message's content hits, or None.

    One `n` for every index, and 13 by default, because `codeprep` pins the code
    index's `n` to the general index's precisely so that one predicate can serve
    both -- an index built at a different width would need its own pass here and
    would silently match nothing at this one.
    """

    from daedalus.data import is_contaminated

    for name in sorted(indexes):
        ngrams = indexes[name]
        if not ngrams:
            continue
        for message in messages:
            if is_contaminated(message.get("content") or "", ngrams, n=n):
                return name
    return None


@dataclass
class Verdict:
    """One conversation's admission decision and everything measured getting it."""

    admitted: bool
    reason: Optional[str] = None
    detail: str = ""
    blocks: Tuple[CodeBlock, ...] = ()
    #: assistant code bytes per corpus bucket -- the supervised payload, which
    #: is what "code-language shares" of an SFT set means. User-turn blocks are
    #: context and are excluded.
    language_bytes: Dict[str, int] = field(default_factory=dict)
    prose_chars: int = 0
    supervised_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    executed: bool = False
    execution: Optional[dict] = None

    @property
    def primary_language(self) -> Optional[str]:
        """The bucket holding most of the assistant's code bytes, or None."""

        carried = {bucket: count for bucket, count in self.language_bytes.items()
                   if bucket in CODE_LANGUAGE_SHARES and count}
        if not carried:
            return None
        return max(sorted(carried), key=carried.__getitem__)


def _refuse(reason: str, detail: str = "", **measured) -> Verdict:
    if reason not in ADMISSION_REASONS:              # pragma: no cover - guard
        raise CodeSFTError(f"unknown admission reason {reason!r}")
    return Verdict(admitted=False, reason=reason, detail=detail, **measured)


def default_execute(solution: str, test_code: str, *, timeout_s: float,
                    memory_mb: int) -> dict:
    """The real `code_eval` sandbox. Public because `probe_source` takes the
    runner as an argument -- a caller asking for real execution should not have
    to import a private name to get the same one `vet_example` defaults to."""

    from scripts.code_eval import run_in_sandbox

    return run_in_sandbox(solution, test_code, timeout_s=timeout_s,
                          memory_mb=memory_mb)


def vet_example(
    messages: Sequence[Mapping[str, str]],
    *,
    indexes: Optional[Mapping[str, Set[str]]] = None,
    n: int = DEFAULT_CODE_N,
    max_prose_chars: int = DEFAULT_MAX_ASSISTANT_CHARS,
    drop_cot: bool = True,
    tokenizer=None,
    max_len: int = DEFAULT_MAX_LEN,
    test_code: Optional[str] = None,
    execute: Optional[Callable[..., dict]] = None,
    timeout_s: float = 30.0,
    memory_mb: int = 1024,
) -> Verdict:
    """Admit or refuse one conversation, with the reason and the measurements.

    `tokenizer` is optional and does two jobs when supplied: it is the only way
    to refuse an example that will not fit `max_len` -- otherwise the encoder
    drops it later without a reason or a count -- and it is where the plan's
    supervised-token counts come from. Without one, `supervised_tokens` is None
    and `share_report` says the count was not taken rather than reporting zero.

    `test_code` is optional in the same spirit. When a source ships a test, the
    single Python block of the assistant turn is executed against it and a
    failure is a refusal; when it does not, the example is admitted
    syntax-checked only and `executed` stays False, so the execution-tested
    share is a measured fraction of the corpus rather than an assumption about
    it. That distinction is the plan's "syntax-checked *and* execution-tested"
    read honestly: this box cannot manufacture tests for a source that has none.
    """

    for message in messages or ():
        if not isinstance(message, collections.abc.Mapping) \
                or "role" not in message or message.get("content") is None:
            return _refuse("malformed", "a message lacks a role or a content")
    if not messages:
        return _refuse("malformed", "no messages")

    if not any(m.get("role") == "assistant" and (m.get("content") or "").strip()
               for m in messages):
        return _refuse("no_assistant", "no assistant turn with content")

    prose, blocks = parse_messages(messages)
    assistant_blocks = tuple(b for b in blocks if b.role == "assistant")
    language_bytes: Dict[str, int] = {}
    for block in assistant_blocks:
        language_bytes[block.language] = \
            language_bytes.get(block.language, 0) + block.bytes
    measured = {"blocks": blocks, "language_bytes": language_bytes,
                "prose_chars": len(prose)}

    unterminated = [b for b in assistant_blocks if not b.closed]
    if unterminated:
        return _refuse("unterminated_code_fence",
                       f"{len(unterminated)} assistant code fence(s) never closed",
                       **measured)

    if indexes:
        hit = contamination_hit(messages, indexes, n=n)
        if hit is not None:
            return _refuse("contaminated", f"matched the {hit!r} index at n={n}",
                           **measured)

    for block in assistant_blocks:
        checker = SYNTAX_CHECKERS.get(block.tag)
        if checker is None:
            continue
        ok, message = checker(block.source)
        if not ok:
            return _refuse("syntax_error", f"{block.tag}: {message}", **measured)

    if drop_cot and has_long_cot_prose(prose):
        return _refuse("long_cot", "a chain-of-thought marker in assistant prose",
                       **measured)

    if len(prose) > max_prose_chars:
        return _refuse("prose_too_long",
                       f"{len(prose):,} prose characters over {max_prose_chars:,}",
                       **measured)

    supervised = total = None
    if tokenizer is not None:
        encoded = encode_sft_example(messages, tokenizer, max_len=max_len)
        if encoded is None:
            return _refuse("over_token_budget",
                           f"does not fit {max_len:,} tokens", **measured)
        ids, labels = encoded
        total = len(ids)
        supervised = sum(1 for label in labels if label != IGNORE_INDEX)
    measured["supervised_tokens"] = supervised
    measured["total_tokens"] = total

    if test_code is None:
        return Verdict(admitted=True, **measured)

    runnable = [b for b in assistant_blocks if b.tag == "python"]
    if not runnable:
        return _refuse("no_solution_block",
                       "a test was supplied but no Python block to run it against",
                       **measured)
    if len(runnable) > 1:
        return _refuse("ambiguous_solution",
                       f"a test was supplied but the assistant turn has "
                       f"{len(runnable)} Python blocks", **measured)

    runner = execute or default_execute
    outcome = runner(runnable[0].source, test_code, timeout_s=timeout_s,
                     memory_mb=memory_mb)
    measured["executed"] = True
    measured["execution"] = dict(outcome)
    if outcome.get("status") != "passed":
        return _refuse("execution_failed",
                       f"{outcome.get('category') or 'failed'}: "
                       f"{str(outcome.get('detail') or '')[:200]}", **measured)
    return Verdict(admitted=True, **measured)


def _shares(counts: Mapping[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {key: counts[key] / total for key in sorted(counts)}


def share_report(verdicts: Iterable[Verdict]) -> dict:
    """What the plan asks step 6 to track, over a whole candidate set.

    Language shares are byte shares of *admitted* examples only, in the code
    corpus's own bucket vocabulary, so the SFT set and the pretraining corpus can
    be read from one table. Refused examples are counted by reason and excluded
    from the shares: they are not training data, and folding them in would make
    the reported Python share a property of the filter rather than of the corpus.

    `unchecked_code_bytes` and `unknown_language_bytes` are published rather than
    buried. The first is code admitted without a syntax check because this box
    has no parser for its language; the second is code admitted without a
    language because the source did not tag it. Both are the honest denominators
    for "syntax-checked", and a reader who does not get them will assume 1.0.
    """

    reasons = {reason: 0 for reason in ADMISSION_REASONS}
    language_bytes: Dict[str, int] = {}
    supervised_by_language: Dict[str, int] = {}
    admitted = executed = 0
    total = 0
    supervised_tokens = 0
    supervised_counted = 0
    checked_bytes = unchecked_bytes = unknown_bytes = 0

    for verdict in verdicts:
        total += 1
        if not verdict.admitted:
            reasons[verdict.reason or "malformed"] += 1
            continue
        admitted += 1
        executed += 1 if verdict.executed else 0
        for bucket, count in verdict.language_bytes.items():
            language_bytes[bucket] = language_bytes.get(bucket, 0) + count
            if bucket == UNKNOWN_LANGUAGE:
                unknown_bytes += count
        for block in verdict.blocks:
            if block.role != "assistant":
                continue
            if block.checkable:
                checked_bytes += block.bytes
            else:
                unchecked_bytes += block.bytes
        if verdict.supervised_tokens is not None:
            supervised_counted += 1
            supervised_tokens += verdict.supervised_tokens
            primary = verdict.primary_language
            if primary is not None:
                supervised_by_language[primary] = \
                    supervised_by_language.get(primary, 0) + \
                    verdict.supervised_tokens

    carried = {bucket: count for bucket, count in language_bytes.items()
               if bucket in CODE_LANGUAGE_SHARES}
    return {
        "schema": SCHEMA,
        "examples": total,
        "admitted": admitted,
        "refused": total - admitted,
        "refusals": {reason: reasons[reason] for reason in ADMISSION_REASONS},
        "execution_tested": executed,
        # A fraction, named, because "syntax-checked and execution-tested" reads
        # as 1.0 unless the measured number is beside it.
        "execution_tested_share": (executed / admitted) if admitted else None,
        "code_language_bytes": dict(sorted(language_bytes.items())),
        "code_language_shares": _shares(carried),
        "unchecked_code_bytes": unchecked_bytes,
        "checked_code_bytes": checked_bytes,
        "syntax_checked_share": (checked_bytes / (checked_bytes + unchecked_bytes)
                                 if (checked_bytes + unchecked_bytes) else None),
        "unknown_language_bytes": unknown_bytes,
        # None, not 0: no tokenizer was supplied, so the count was never taken,
        # and a zero here would read as "admitted nothing worth supervising".
        "supervised_tokens": supervised_tokens if supervised_counted else None,
        "supervised_tokens_counted_for": supervised_counted,
        "supervised_tokens_by_language": dict(sorted(
            supervised_by_language.items())),
    }


# --------------------------------------------------------------- the sources ---

#: Dataset licences an SFT source may declare.
#:
#: A *second* allow-list rather than a reuse of `codeprep.PERMISSIVE_LICENSES`,
#: because the two are about different objects and sharing one would refuse the
#: right sources for the wrong reason. That list classifies the licence of a
#: *source file inside a repository*, where `odc-by` and `cc-by-4.0` never
#: appear; this one classifies the licence of a *dataset as a whole*, where they
#: are the two commonest permissive declarations. Merging them would have to
#: either admit data licences into the corpus gate or refuse the only licence
#: `bigcode` publishes under -- and a source refused for a reason that is not
#: about it is indistinguishable in a manifest from one refused correctly.
#:
#: The line is **attribution-only**. Everything here asks for credit and nothing
#: else: no reciprocal obligation, no field-of-use restriction, no downstream
#: licence condition. That is the same reading of "permissive" the corpus gate
#: uses when it puts `mpl-2.0` and `lgpl-*` on the refused side.
PERMISSIVE_DATASET_LICENSES = frozenset({
    "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0",
    "isc", "mit", "odc-by", "unlicense",
})

#: Declarations known to be refused, kept by name so "we know this one and it
#: does not qualify" stays distinguishable from "we have never seen this
#: string". Share-alike (`*-sa-*`) and non-commercial (`*-nc-*`) both carry the
#: condition permissive means the absence of; `other` and the model-derived
#: licences name terms that live somewhere else entirely.
KNOWN_NON_PERMISSIVE_DATASET_LICENSES = frozenset({
    "afl-3.0", "agpl-3.0", "bigscience-openrail-m", "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "gpl-2.0", "gpl-3.0",
    "llama2", "llama3", "openrail", "other",
})

#: A card with no licence field and no `license:` tag at all. Deliberately not
#: folded into `unknown`: "the card names something this module does not
#: classify" is a gap in *this* table and is fixed by widening it, while "the
#: card names nothing" is a gap in the *dataset* and cannot be fixed here at
#: all. The plan's hard-blocker list says ambiguous licence, and an absent
#: declaration is the most ambiguous case there is -- it is the finding that
#: refused `HuggingFaceTB/smoltalk`, whose `self-oss-instruct` config is
#: otherwise exactly the code half this phase wants.
UNDECLARED_LICENSE = "undeclared"


def dataset_license_verdict(value) -> str:
    """`permissive`, `non-permissive`, `unknown` or `undeclared`.

    Four answers rather than two, for `codeprep.license_verdict`'s reason: an
    allow-list that returned a bare boolean would collapse "refused because we
    know it" into "refused because we have never seen it", and only the second
    is news worth acting on.
    """

    if value is None:
        return UNDECLARED_LICENSE
    key = str(value).strip().lower()
    if not key:
        return UNDECLARED_LICENSE
    if key in PERMISSIVE_DATASET_LICENSES:
        return "permissive"
    if key in KNOWN_NON_PERMISSIVE_DATASET_LICENSES:
        return "non-permissive"
    return "unknown"


#: The two halves the plan names: "code and general SFT". Kept as a vocabulary
#: rather than a bool so a report can say which half a count belongs to without
#: a reader having to know which way round True meant.
SFT_HALVES = ("code", "general")


@dataclass(frozen=True)
class SFTSource:
    """One candidate conversation source, and how to read a row of it.

    `source_field` with `keep_sources`/`drop_sources` is how two halves are
    carved out of one dataset. That is not a convenience: it is what makes the
    code half reachable under a *declared* licence at all (see `SFT_SOURCES`),
    and it is the same shape `codeprep.RepositoryGate` uses to partition one
    directory into two disjoint streams.
    """

    key: str
    half: str
    dataset: str
    declared_license: str
    note: str
    split: str = "train"
    config: Optional[str] = None
    revision: Optional[str] = None
    #: Row key naming the sub-dataset a row came from, when the source is a
    #: mixture. None when the whole dataset is one source.
    source_field: Optional[str] = None
    #: Values of `source_field` to keep. Empty keeps everything not dropped.
    keep_sources: Tuple[str, ...] = ()
    #: Values to drop. Empty drops nothing.
    drop_sources: Tuple[str, ...] = ()
    messages_field: str = "messages"

    def __post_init__(self):
        if self.half not in SFT_HALVES:
            raise CodeSFTError(f"{self.key}: half must be one of {SFT_HALVES}")
        if self.keep_sources and self.drop_sources:
            # Both at once has two readings -- keep-then-drop and drop-then-keep
            # -- that differ on a value in both lists, and nothing about the
            # table says which. One list, so the partition is a fact rather
            # than an argument about precedence.
            raise CodeSFTError(
                f"{self.key}: keep_sources and drop_sources are two ways to "
                f"write one partition; pass one")
        if (self.keep_sources or self.drop_sources) and not self.source_field:
            raise CodeSFTError(
                f"{self.key}: a source filter needs the row key to read it from")

    @property
    def license_verdict(self) -> str:
        return dataset_license_verdict(self.declared_license)


#: What phase 8 step 6 actually trains on, pending the probe that resolves it.
#:
#: Both halves are views of **one** apache-2.0 dataset, `smol-smoltalk` -- the
#: source the released instruct model was itself post-trained on, which is the
#: borrow-rather-than-invent rule this program has followed every time a
#: decision arrived without one. Its `source` column carries the sub-dataset a
#: row came from, so the two halves are a partition of it rather than two
#: datasets that could disagree about tokenizer, rendering or turn structure.
#:
#: The code half is `self-oss-instruct`: StarCoder2-generated Python answers to
#: instructions seeded from permissively licensed repositories, execution
#: filtered upstream. Measured at **10.30%** of the dataset over 40,000 rows,
#: and 28.2% of those rows carry an executable assertion in the user turn --
#: which is where this phase's "execution-tested" comes from, since nothing on
#: this box can manufacture a test for a source that ships none.
#:
#: The general half is everything else, and `self-oss-instruct` is dropped from
#: it *by name*. Left in, the same rows would be counted in both halves: the
#: code-language shares would be a property of the general stream too, and the
#: 65/35 the model card will claim would describe neither.
SFT_SOURCES: Tuple[SFTSource, ...] = (
    SFTSource(
        key="code-self-oss-instruct",
        half="code",
        dataset="HuggingFaceTB/smol-smoltalk",
        declared_license="apache-2.0",
        source_field="source",
        keep_sources=("self-oss-instruct",),
        note="StarCoder2 self-alignment on permissively licensed seeds, "
             "rendered as chat and redistributed under the subset's own "
             "apache-2.0 declaration; 10.30% of the dataset, 28.2% of it "
             "carrying a user-turn assertion",
    ),
    SFTSource(
        key="general-smoltalk",
        half="general",
        dataset="HuggingFaceTB/smol-smoltalk",
        declared_license="apache-2.0",
        source_field="source",
        drop_sources=("self-oss-instruct",),
        note="the released instruct model's own SFT distribution, minus the "
             "code half above so the two are disjoint",
    ),
)

assert {s.key for s in SFT_SOURCES}.__len__() == len(SFT_SOURCES), \
    "SFT source keys must be unique"
assert all(s.license_verdict == "permissive" for s in SFT_SOURCES), \
    "every source in the live table must declare a permissive licence"
assert {s.half for s in SFT_SOURCES} == set(SFT_HALVES), \
    "the plan asks for code and general SFT; both halves must be represented"


#: Candidates measured and not used, with the measurement that decided each.
#: Kept for `codeprep.GITHUB_CODE_LANGUAGES`' reason: these are the evidence
#: that the pick above is a choice rather than the only thing anyone tried, and
#: a later slice that wants more code conversations starts from here rather than
#: from a fresh guess.
RECORDED_ALTERNATIVES: Tuple[Tuple[str, str, str], ...] = (
    ("HuggingFaceTB/smoltalk::self-oss-instruct", UNDECLARED_LICENSE,
     "the same conversations, already `messages`-shaped, and the most direct "
     "route to the code half -- refused because the card carries no licence "
     "field and no `license:` tag at all. Its apache-2.0 subset carries the "
     "same rows, so nothing is lost by refusing it"),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "odc-by",
     "the upstream of the code half, permissively declared and usable. Not "
     "used because it ships `instruction`/`response` rather than messages, so "
     "reading it would mean a second rendering of the same conversations that "
     "could drift from the one the released instruct model saw. Its "
     "instruction field carries a python block in 40.0% of rows and an "
     "assertion in 32.3%, matching the chat rendering's 37.9%/28.2%"),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", "mit",
     "declared MIT, but generated by a proprietary model whose terms are not "
     "the dataset's licence. Not refused by this table -- not read, either"),
    ("HuggingFaceH4/CodeAlpaca_20K", "cc",
     "declares the string `cc`, which names no version and no clauses. The "
     "`unknown` verdict this module returns for it is the correct one"),
)


def row_messages(source: SFTSource, row) -> Tuple[Optional[List[dict]],
                                                  Optional[str]]:
    """`(messages, refusal)` for one raw row of `source`.

    Separate from `vet_example` because the two refuse different things. This
    decides whether a row is *for this source at all*; the gate decides whether
    a conversation is fit to train on. Folding them together would count a row
    belonging to the other half as a quality drop.
    """

    if not isinstance(row, collections.abc.Mapping):
        return None, "not_a_mapping"
    if source.source_field is not None:
        if source.source_field not in row:
            # The silent-failure case, and the reason the probe exists: a
            # `source_field` the dataset does not have refuses every row, and
            # the build then writes an empty file and exits zero.
            return None, "no_source_field"
        value = row.get(source.source_field)
        if source.keep_sources and value not in source.keep_sources:
            return None, "other_source"
        if source.drop_sources and value in source.drop_sources:
            return None, "other_source"
    messages = row.get(source.messages_field)
    if not messages:
        return None, "no_messages"
    try:
        rendered = [{"role": m["role"], "content": m["content"]}
                    for m in messages]
    except (KeyError, TypeError):
        return None, "no_messages"
    return rendered, None


#: Why a row that carries user-turn code still has no test to run.
NO_TEST_REASONS = ("no_user_block", "ambiguous_user_block", "no_assertion")


def shipped_test(messages: Sequence[Mapping[str, str]]
                 ) -> Tuple[Optional[str], Optional[str]]:
    """`(test_code, why_not)` -- the test this conversation ships, if any.

    `self-oss-instruct` states its test in the *user* turn: "Your code should
    pass the following assertion", then a fenced python block. That block is a
    real test -- it names the function the assistant must define and asserts an
    output -- so it is the execution evidence this phase's "execution-tested"
    means, and it is the only such evidence available: this box cannot
    manufacture a test for a source that ships none.

    Two refusals rather than a best effort:

    **An assertion is required.** A user-turn block that only *demonstrates*
    usage runs to completion whatever the assistant wrote, so admitting it
    would grow `execution_tested` with tests that cannot fail. A gate that
    cannot return no is not a gate.

    **Exactly one block.** Two blocks in the user turn is the `ambiguous_solution`
    reasoning from the other side: concatenating them runs setup code as though
    it were the assertion, and picking one is a guess about which.
    """

    blocks = [b for message in messages
              for b in parse_markdown(message.get("content") or "",
                                      role=message.get("role") or "").blocks
              if (message.get("role") == "user" and b.tag == "python"
                  and b.closed)]
    if not blocks:
        return None, "no_user_block"
    if len(blocks) > 1:
        return None, "ambiguous_user_block"
    if "assert" not in blocks[0].source:
        return None, "no_assertion"
    return blocks[0].source, None


#: Why a row never reached the admission gate. Counted apart from
#: `ADMISSION_REASONS` so a source filter and a quality filter are never added
#: together: `other_source` is 90% of the code half's stream by design, and
#: folding it into the refusals would report that half as 10% admitted.
ROW_REFUSALS = ("not_a_mapping", "no_source_field", "other_source",
                "no_messages")


def _stream_hub(source: SFTSource):
    """Streaming rows of one source. Isolated so `probe_source` is testable
    without the Hub, matching `codeprep._stream_github_code`."""

    from datasets import load_dataset

    return load_dataset(source.dataset, source.config, split=source.split,
                        revision=source.revision, streaming=True)


def probe_source(
    source: SFTSource,
    *,
    rows: int = 2_000,
    stream: Optional[Callable[[SFTSource], Iterable]] = None,
    indexes: Optional[Mapping[str, Set[str]]] = None,
    n: int = DEFAULT_CODE_N,
    tokenizer=None,
    max_len: int = DEFAULT_MAX_LEN,
    execute: Optional[Callable[..., dict]] = None,
    timeout_s: float = 30.0,
    memory_mb: int = 1024,
) -> dict:
    """Read real rows of one source and report what the build would do with them.

    `execute` is opt-in and off by default. A probe that ran every shipped test
    would spend a process per row at up to `timeout_s` each -- minutes of
    sandboxing to answer a question about *fields*. So the record publishes
    `shipped_test_share` (how many admitted rows carry a runnable test) beside
    `share_report`'s `execution_tested_share` (how many were actually run), and
    when nothing was executed the second is zero with `executed` False rather
    than a share that reads as a measurement.
    """

    reader = stream or _stream_hub
    record: dict = {
        "key": source.key, "half": source.half, "dataset": source.dataset,
        "config": source.config, "split": source.split,
        "declared_license": source.declared_license,
        "license_verdict": source.license_verdict,
        "source_field": source.source_field,
        "keep_sources": list(source.keep_sources),
        "drop_sources": list(source.drop_sources),
        "rows_requested": rows, "rows_offered": 0, "rows_kept": 0,
        "resolved": False, "executed": execute is not None,
        "row_refusals": {reason: 0 for reason in ROW_REFUSALS},
        "fields": [], "source_values": {},
        "shipped_tests": 0, "shipped_tests_admitted": 0,
        "no_test_reasons": {reason: 0 for reason in NO_TEST_REASONS},
    }

    try:
        iterator = iter(reader(source))
    except Exception as error:                           # noqa: BLE001
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    fields: Set[str] = set()
    source_values: Dict[str, int] = {}
    verdicts: List[Verdict] = []
    for offered, row in enumerate(iterator):
        if offered >= rows:
            break
        record["rows_offered"] = offered + 1
        record["resolved"] = True
        if isinstance(row, collections.abc.Mapping):
            fields |= set(row)
            if source.source_field is not None:
                value = str(row.get(source.source_field))
                source_values[value] = source_values.get(value, 0) + 1
        messages, refusal = row_messages(source, row)
        if refusal is not None:
            record["row_refusals"][refusal] += 1
            continue
        record["rows_kept"] += 1
        test_code, why_not = shipped_test(messages)
        if test_code is None:
            record["no_test_reasons"][why_not] += 1
        else:
            record["shipped_tests"] += 1
        verdicts.append(vet_example(
            messages, indexes=indexes, n=n, tokenizer=tokenizer,
            max_len=max_len,
            test_code=test_code if execute is not None else None,
            execute=execute, timeout_s=timeout_s, memory_mb=memory_mb))

    record["fields"] = sorted(fields)
    record["source_values"] = dict(sorted(source_values.items(),
                                          key=lambda kv: (-kv[1], kv[0])))
    report = share_report(verdicts)
    record["report"] = report
    admitted = report["admitted"]
    # Two counts, because `shipped_tests` is over *kept* rows and the share has
    # to be over *admitted* ones to be comparable with
    # `execution_tested_share`. Reported as one number with the other's
    # denominator, they differ by however many test-carrying conversations the
    # gate refused -- a real quantity that would read as an arithmetic slip.
    record["shipped_tests_admitted"] = sum(
        1 for v in verdicts if v.admitted and shipped_test_available(v))
    record["shipped_test_share"] = None if not admitted else (
        record["shipped_tests_admitted"] / admitted)
    record["kept_share"] = (record["rows_kept"] / record["rows_offered"]
                            if record["rows_offered"] else None)
    return record


def shipped_test_available(verdict: Verdict) -> bool:
    """True when this admitted conversation carried a runnable user-turn test.

    Read off the verdict's own blocks rather than recomputed from the messages,
    so the share and the execution are answering over one parse.
    """

    blocks = [b for b in verdict.blocks
              if b.role == "user" and b.tag == "python" and b.closed]
    return len(blocks) == 1 and "assert" in blocks[0].source


def probe_sources(
    sources: Optional[Sequence[SFTSource]] = None, **kwargs
) -> dict:
    """`probe_source` over the table, plus what the alternatives measured."""

    chosen = list(sources if sources is not None else SFT_SOURCES)
    return {
        "schema": SCHEMA,
        "sources": [probe_source(source, **kwargs) for source in chosen],
        "alternatives": [{"dataset": name, "declared_license": declared,
                          "license_verdict": dataset_license_verdict(
                              None if declared == UNDECLARED_LICENSE else declared),
                          "note": note}
                         for name, declared, note in RECORDED_ALTERNATIVES],
    }


def probe_problems(report: Mapping) -> List[str]:
    """Everything in a probe that would make a build quietly produce nothing."""

    problems: List[str] = []
    halves = {half: 0 for half in SFT_HALVES}
    for record in report.get("sources") or ():
        key = record.get("key")
        half = record.get("half")
        if half in halves:
            halves[half] += record.get("rows_kept", 0)
        if record.get("error"):
            problems.append(f"{key} did not resolve: {record['error']}")
            continue
        if not record.get("resolved"):
            problems.append(f"{key} did not resolve: it yielded no rows at all")
            continue
        if record.get("license_verdict") != "permissive":
            problems.append(
                f"{key} declares {record.get('declared_license')!r}, which this "
                f"module reads as {record.get('license_verdict')}")
        refusals = record.get("row_refusals") or {}
        if refusals.get("no_source_field"):
            # Fatal rather than reported: the filter names a key the rows do
            # not have, so every row is refused and the build's own counters
            # look like an unremarkable low yield.
            problems.append(
                f"{key} filters on a row key {record.get('source_field')!r} "
                f"that its rows do not have; {refusals['no_source_field']:,} of "
                f"{record.get('rows_offered', 0):,} rows lack it")
        if not record.get("rows_kept"):
            problems.append(
                f"{key} kept none of {record.get('rows_offered', 0):,} offered "
                f"rows, so a build from it would write nothing")
        elif not (record.get("report") or {}).get("admitted"):
            problems.append(
                f"{key} kept {record['rows_kept']:,} rows and the admission "
                f"gate refused every one")
    for half, kept in sorted(halves.items()):
        if half in SFT_HALVES and not kept:
            problems.append(
                f"the {half} half kept no rows; the plan asks for code *and* "
                f"general SFT and this probe found only one of them")
    return problems
