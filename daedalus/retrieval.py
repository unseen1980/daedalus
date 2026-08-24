"""Deterministic retrieval tasks: passkey, multi-query associative recall, and
the synthetic controls that prove the harness itself is sound.

Why this exists. Daedalus keeps a KV cache in only 6 of its 18 blocks; the other
12 are gated short convolutions whose state is fixed-width. That is the whole
architectural bet, and it is precisely the design that can lose long-range
*retrieval* while leaving perplexity and the five cloze tasks untouched -- a
model can predict the next token well everywhere and still be unable to fetch a
specific value it saw 1,500 tokens ago. Phase 3's QAT recovery and Phase 8's
code adaptation both carry a hard retrieval-retention gate (1 and 2 points), so
this measurement has to exist before either can be trusted.

Three properties make the numbers usable:

  - **Deterministic.** Keys, values, filler order and needle placement all come
    from one seeded `random.Random`. Re-running an evaluation at temperature
    zero must be bit-identical, or a paired before/after comparison is measuring
    the generator's noise instead of the model's.

  - **Held out.** Pass keys are drawn from a digit range the filler never
    contains, and every key/value in an item is unique. Without that, a model
    that emits *any* number could score, and a wrong recall could accidentally
    match the right answer.

  - **Controlled.** `OracleBackend` answers from the prompt text alone. If a
    generated item is well formed -- needle present, question answerable -- the
    oracle scores it 1. So a control run that is not 100% is a formatter bug,
    reported as such, independently of whether any model can do the task. That
    separation matters most exactly when the real scores are low, which at 150M
    they will be.

The prompts are plain completion text, not chat: the base checkpoints this
phase scores are not instruction-tuned, and wrapping them in ChatML would
measure template compliance rather than retrieval.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# The four context depths the plan names. 2048 is the trained
# `max_position_embeddings`, so it is the deepest honest measurement; anything
# beyond would measure extrapolation, not retention.
DEPTHS = (256, 512, 1024, 2048)

# Needle placements as a fraction of the filler, from "first thing said" to
# "last thing before the question". A single mid-context placement would hide
# the failure mode this is built to catch: fixed-width conv state degrades with
# distance, so the depth *curve* is the signal, not any one point.
DEFAULT_DEPTH_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Filler is fixed, dull, and digit-free. Digit-free is not cosmetic: the passkey
# answer is a digit run, and `extract_answer` takes the first one it sees, so a
# number anywhere in the filler could be scored as the model's answer.
_FILLER_SENTENCES = (
    "The grass is green and the sky above is a pale washed blue.",
    "The sun rises over the hill and the long shadows retreat.",
    "A slow river runs past the mill and turns the old wheel.",
    "Here we go again around the garden and back to the gate.",
    "The wind moves through the tall pines and carries the salt air.",
    "There and back again along the same worn path by the wall.",
    "The bread cools on the sill while the kettle begins to sing.",
    "Dust settles on the shelf where the empty jars are kept.",
)

_PASSKEY_HEADER = (
    "There is an important piece of information hidden somewhere in the text "
    "below. Read the text and remember the information."
)
_PASSKEY_QUESTION = "\nWhat is the pass key? The pass key is"
_PASSKEY_NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key."

_MQAR_HEADER = (
    "Below is a list of keys and their values. Read the list and then answer "
    "the queries at the end with the matching value."
)
_MQAR_QUERY_HEADER = "\nQueries:"

_COPY_HEADER = "Repeat the phrase below exactly.\n"
_COPY_QUESTION = "\nThe phrase is:"

# Keys and values are drawn from *disjoint* pools. Overlapping pools would let a
# model score by echoing a key it just read, which is copying, not recall -- and
# they also make key==value collisions possible, which have to be patched up
# afterwards in a way that quietly breaks value uniqueness.
_KEY_WORDS = (
    "amber", "birch", "cobalt", "dune", "ember", "fjord", "granite", "harbor",
    "indigo", "juniper", "kelp", "lantern", "marble", "nectar", "opal", "prairie",
    "quartz", "russet", "saffron", "thicket", "umber", "violet", "willow", "xenon",
    "yarrow", "zephyr", "alcove", "bramble", "cinder", "dapple", "esker", "fennel",
    "gable", "hollow", "ivory", "jasper", "kestrel", "lichen", "mistral", "nimbus",
)
_VALUE_WORDS = (
    "anchor", "beacon", "canyon", "delta", "eagle", "falcon", "glacier", "hearth",
    "island", "jetty", "keystone", "ledger", "meadow", "needle", "orchard", "pillar",
    "quiver", "ridge", "summit", "tundra", "upland", "valley", "warren", "xystus",
    "yeoman", "zenith", "abbey", "burrow", "cavern", "delve", "estuary", "furrow",
    "gorge", "hamlet", "inlet", "junction", "knoll", "lagoon", "moor", "narrows",
)
_WORDS = _KEY_WORDS


@dataclass
class RetrievalItem:
    """One scored retrieval question, carrying its exact prompt."""

    id: str
    task: str                 # "passkey" | "mqar" | "copy-control"
    depth: int                # requested context depth, in tokens
    prompt: str
    answer: str
    needle_text: str          # the exact substring that carries the answer
    needle_depth_frac: float
    prompt_tokens: int
    meta: Dict[str, object] = field(default_factory=dict)


# ------------------------------------------------------------------- filler ---

def _token_len(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:                       # tokenizers without the kwarg
        return len(tokenizer.encode(text))


def _fill_to_tokens(tokenizer, budget: int, rng: random.Random) -> str:
    """Deterministic filler of at most `budget` tokens.

    Sentences are appended whole, so the result lands within one sentence of the
    budget rather than exactly on it. That is the right trade: truncating
    mid-sentence would leave a token fragment whose retokenisation depends on
    the vocabulary, and the depth curve does not need single-token precision.
    """

    if budget <= 0:
        return ""
    parts: List[str] = []
    used = 0
    while True:
        sentence = _FILLER_SENTENCES[rng.randrange(len(_FILLER_SENTENCES))]
        cost = _token_len(tokenizer, (" " if parts else "") + sentence)
        if used + cost > budget:
            break
        parts.append(sentence)
        used += cost
        if not parts:                       # a budget smaller than one sentence
            break
    return " ".join(parts)


def _assemble(tokenizer, header: str, question: str, needle: str,
              depth: int, depth_frac: float, rng: random.Random) -> str:
    """Header, filler split around the needle at `depth_frac`, then question."""

    scaffold = _token_len(tokenizer, header + " " + needle + question)
    if scaffold > depth:
        raise ValueError(
            f"requested depth {depth} is below this task's scaffold "
            f"({scaffold} tokens); use a depth of at least {scaffold}")
    filler_budget = depth - scaffold
    before_budget = int(round(filler_budget * depth_frac))
    before = _fill_to_tokens(tokenizer, before_budget, rng)
    after = _fill_to_tokens(tokenizer, filler_budget - before_budget, rng)
    body = " ".join(part for part in (header, before, needle, after) if part)
    return body + question


# ------------------------------------------------------------------ passkey ---

def make_passkey_items(tokenizer, *, depths: Sequence[int] = DEPTHS,
                       per_depth: int = 10, seed: int = 20260824,
                       depth_fractions: Sequence[float] = DEFAULT_DEPTH_FRACTIONS
                       ) -> List[RetrievalItem]:
    """Classic passkey retrieval: one hidden number, recalled after N tokens."""

    rng = random.Random(seed)
    used_keys = set()
    items: List[RetrievalItem] = []
    for depth in depths:
        for index in range(per_depth):
            while True:
                # 5 digits: long enough that a lucky guess is ~1/90000, short
                # enough to stay one or two tokens in a BPE vocabulary.
                key = str(rng.randrange(10000, 100000))
                if key not in used_keys:
                    used_keys.add(key)
                    break
            needle = _PASSKEY_NEEDLE.format(key=key)
            frac = depth_fractions[index % len(depth_fractions)]
            prompt = _assemble(tokenizer, _PASSKEY_HEADER, _PASSKEY_QUESTION,
                               needle, depth, frac, rng)
            items.append(RetrievalItem(
                id=f"passkey-d{depth}-{index}",
                task="passkey", depth=depth, prompt=prompt, answer=key,
                needle_text=needle, needle_depth_frac=frac,
                prompt_tokens=_token_len(tokenizer, prompt),
                meta={"key": key},
            ))
    return items


# --------------------------------------------------------------------- mqar ---

def make_mqar_items(tokenizer, *, depths: Sequence[int] = DEPTHS,
                    per_depth: int = 10, seed: int = 20260824,
                    n_queries: int = 4) -> List[RetrievalItem]:
    """Multi-query associative recall over a key/value table.

    Harder than passkey in the way that matters here: the context holds many
    interchangeable pairs, so the model cannot succeed by noticing that one span
    looks unlike its surroundings. It must bind a *specific* key to a *specific*
    value while holding many such bindings -- the capacity a fixed-width
    recurrent state trades away.

    Format. The queries are a block of `key: value` lines whose last line is
    left open, so the item is an unambiguous continuation for a base model:

        Queries:
        delta: needle
        omega: ledger
        alpha:

    The earlier lines are *demonstrations* -- their values are themselves
    recalled from the table, so they establish the format without revealing the
    scored answer. Only the final key is scored, which keeps `correct` binary
    and therefore keeps McNemar pairing valid.

    The rejected alternative was to leave every query open and parse N answer
    lines from one completion. On a 150M base model that measures whether the
    model emits a well-formed N-line list, not whether it can recall -- and a
    parse failure would be indistinguishable from a retrieval failure, which is
    the exact confusion the controls exist to prevent.
    """

    rng = random.Random(seed ^ 0x5151)
    items: List[RetrievalItem] = []
    for depth in depths:
        for index in range(per_depth):
            # ~3 tokens a pair line; keep the table a minority of the context so
            # the padding, not the table, sets the distance to the query.
            n_pairs = min(len(_KEY_WORDS), max(n_queries + 4, depth // 24))
            keys = rng.sample(_KEY_WORDS, n_pairs)
            values = rng.sample(_VALUE_WORDS, n_pairs)
            pairs = dict(zip(keys, values))
            queried = rng.sample(keys, min(n_queries, n_pairs))
            demonstrations, scored_key = queried[:-1], queried[-1]

            table = "\n".join(f"{key}: {value}" for key, value in pairs.items())
            demo_block = "".join(f"\n{key}: {pairs[key]}" for key in demonstrations)
            head = f"{_MQAR_HEADER}\n{table}"
            tail = f"{_MQAR_QUERY_HEADER}{demo_block}\n{scored_key}:"

            budget = depth - _token_len(tokenizer, head + tail)
            if budget < 0:
                raise ValueError(
                    f"requested depth {depth} is below this task's scaffold "
                    f"({_token_len(tokenizer, head + tail)} tokens)")
            padding = _fill_to_tokens(tokenizer, budget, rng)
            prompt = f"{head}\n{padding}{tail}" if padding else head + tail

            items.append(RetrievalItem(
                id=f"mqar-d{depth}-{index}",
                task="mqar", depth=depth, prompt=prompt,
                answer=pairs[scored_key],
                needle_text=f"{scored_key}: {pairs[scored_key]}",
                needle_depth_frac=0.0,
                prompt_tokens=_token_len(tokenizer, prompt),
                meta={
                    "pairs": pairs,
                    "queried_keys": queried,
                    "demonstration_keys": demonstrations,
                    "scored_key": scored_key,
                    "answers": [pairs[key] for key in queried],
                    "n_pairs": n_pairs,
                },
            ))
    return items


# ---------------------------------------------------------------- controls ---

def make_copy_control_items(tokenizer, *, per_item: int = 8,
                            seed: int = 20260824) -> List[RetrievalItem]:
    """The easiest possible retrieval: the answer is the previous line.

    Not a measure of the model -- a measure of the *harness*. If this scores
    zero while the prompts look fine, the fault is in prompt assembly, answer
    extraction, or the backend's stop condition, not in long-range recall.
    """

    rng = random.Random(seed ^ 0x0C09)
    items: List[RetrievalItem] = []
    for index in range(per_item):
        phrase = " ".join(rng.sample(_WORDS, 3))
        prompt = f"{_COPY_HEADER}{phrase}{_COPY_QUESTION}"
        items.append(RetrievalItem(
            id=f"copy-control-{index}",
            task="copy-control", depth=0, prompt=prompt, answer=phrase,
            needle_text=phrase, needle_depth_frac=1.0,
            prompt_tokens=_token_len(tokenizer, prompt),
            meta={"phrase": phrase},
        ))
    return items


class OracleBackend:
    """Answers from the prompt alone, to validate items without a model.

    It reads the prompt the way the task defines the answer to be recoverable:
    a pass key from its needle sentence, a recalled value from the `key: value`
    line the query names, a copied phrase from the line above the question. When
    the prompt is well formed this is exact; when the formatter dropped the
    needle or asked for an undefined key, it returns nothing and the control
    fails loudly.
    """

    def generate(self, item: RetrievalItem) -> str:
        if item.task == "passkey":
            match = re.search(r"The pass key is (\d+)\.", item.prompt)
            return match.group(1) if match else ""
        if item.task == "mqar":
            # The scored key's `key: value` line exists only in the table -- it
            # is deliberately excluded from the demonstration block -- so this
            # finds the binding the item claims to test, and finds nothing when
            # the table row is missing.
            match = re.search(rf"^{re.escape(item.meta['scored_key'])}: (\w+)$",
                              item.prompt, flags=re.MULTILINE)
            return match.group(1) if match else ""
        if item.task == "copy-control":
            match = re.search(rf"{re.escape(_COPY_HEADER)}(.+?)"
                              rf"{re.escape(_COPY_QUESTION)}",
                              item.prompt, flags=re.DOTALL)
            return match.group(1).strip() if match else ""
        raise ValueError(f"no oracle for task {item.task!r}")

    def generate_all(self, items: Sequence[RetrievalItem]) -> List[str]:
        return [self.generate(item) for item in items]


# ------------------------------------------------------------------ scoring ---

_PUNCTUATION = " \t\r\n.,;:!?\"'`()[]{}*_-"


def normalize_answer(text: str) -> str:
    """Case-fold, strip wrapping punctuation, collapse internal whitespace."""

    return re.sub(r"\s+", " ", text.strip().strip(_PUNCTUATION)).lower()


def extract_answer(item: RetrievalItem, response: str) -> str:
    """Pull the answer out of a completion, per the task's answer shape.

    Extraction is deliberately generous about *surroundings* and strict about
    the answer: a base model will happily continue past its answer, and
    penalising it for chatter would measure formatting compliance rather than
    retrieval. The comparison afterwards is exact.
    """

    if item.task == "passkey":
        match = re.search(r"\d+", response)
        return match.group(0) if match else ""
    expected_lines = len(item.answer.splitlines()) or 1
    lines = [line.strip() for line in response.strip().splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines[:expected_lines])


def score_items(items: Sequence[RetrievalItem],
                responses: Sequence[str]) -> List[dict]:
    """Per-item outcomes, in the shape the scorecard sidecar stores."""

    if len(items) != len(responses):
        raise ValueError(
            f"expected one response per item, got {len(responses)} responses "
            f"for {len(items)} items")
    records = []
    for item, response in zip(items, responses):
        extracted = extract_answer(item, response)
        expected_lines = item.answer.splitlines() or [item.answer]
        got_lines = extracted.splitlines() or [extracted]
        hits = sum(1 for expected, got in zip(expected_lines, got_lines)
                   if normalize_answer(expected) == normalize_answer(got))
        records.append({
            "id": item.id,
            "task": item.task,
            "depth": item.depth,
            "needle_depth_frac": item.needle_depth_frac,
            "prompt_tokens": item.prompt_tokens,
            "correct": int(normalize_answer(extracted) ==
                           normalize_answer(item.answer)),
            "query_accuracy": hits / len(expected_lines),
            "expected": item.answer,
            "extracted": extracted,
            "response": response,
        })
    return records


def summarize(items: Sequence[RetrievalItem],
              records: Sequence[dict]) -> Dict[str, float]:
    """Headline exact match plus the per-depth curve the gates read."""

    if not records:
        return {"exact_match": float("nan"), "n": 0}
    metrics: Dict[str, float] = {
        "exact_match": sum(record["correct"] for record in records) / len(records),
        "query_accuracy": (sum(record["query_accuracy"] for record in records)
                           / len(records)),
        "n": float(len(records)),
    }
    depths = sorted({record["depth"] for record in records})
    for depth in depths:
        at_depth = [record for record in records if record["depth"] == depth]
        metrics[f"exact_match_d{depth}"] = (
            sum(record["correct"] for record in at_depth) / len(at_depth))
        metrics[f"n_d{depth}"] = float(len(at_depth))
    return metrics


def make_all_items(tokenizer, *, depths: Sequence[int] = DEPTHS,
                   per_depth: int = 10, seed: int = 20260824,
                   n_queries: int = 4,
                   control_items: int = 8) -> Dict[str, List[RetrievalItem]]:
    """Every retrieval task, keyed by name, from one seed."""

    return {
        "passkey": make_passkey_items(tokenizer, depths=depths,
                                      per_depth=per_depth, seed=seed),
        "mqar": make_mqar_items(tokenizer, depths=depths, per_depth=per_depth,
                                seed=seed, n_queries=n_queries),
        "copy-control": make_copy_control_items(tokenizer, per_item=control_items,
                                                seed=seed),
    }
