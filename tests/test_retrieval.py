"""Tests for the deterministic retrieval task generators and their controls.

The controls are the load-bearing part. A retrieval score for a 150M model is
low and noisy by nature, so a formatter bug -- a needle that never made it into
the prompt, a question that asks for a key the context never defined -- would
look exactly like "the model cannot do retrieval". The oracle backend exists to
tell those two apart *without* consulting a model, which is why it must score
100% on every generated item.
"""

import pytest

from daedalus.retrieval import (
    DEPTHS,
    OracleBackend,
    RetrievalItem,
    extract_answer,
    make_copy_control_items,
    make_mqar_items,
    make_passkey_items,
    normalize_answer,
    score_items,
    summarize,
)


class WhitespaceTokenizer:
    """A stand-in with the only two methods the generators use.

    Deliberately not the real SmolLM2 tokenizer: these tests pin generator
    behaviour (determinism, depth targeting, needle placement), none of which
    should depend on a 49k-vocab download.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


# ------------------------------------------------------------- determinism ---

def test_passkey_items_are_identical_for_the_same_seed(tokenizer):
    first = make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=7)
    second = make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=7)

    assert [item.prompt for item in first] == [item.prompt for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]
    assert [item.id for item in first] == [item.id for item in second]


def test_passkey_items_differ_for_a_different_seed(tokenizer):
    first = make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=7)
    second = make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=8)

    assert [item.answer for item in first] != [item.answer for item in second]


def test_default_depths_are_the_four_the_plan_names():
    assert DEPTHS == (256, 512, 1024, 2048)


# ------------------------------------------------------------------ passkey ---

def test_passkey_prompt_reaches_the_requested_token_depth(tokenizer):
    for depth in (256, 512, 1024):
        items = make_passkey_items(tokenizer, depths=(depth,), per_depth=2, seed=1)
        for item in items:
            measured = len(tokenizer.encode(item.prompt))
            assert depth - 32 <= measured <= depth, (depth, measured)
            assert item.prompt_tokens == measured


def test_passkey_answer_occurs_only_inside_the_needle(tokenizer):
    # The needle names the key twice on purpose -- that is the standard passkey
    # phrasing and it helps a 150M model notice the span. What must never happen
    # is the key appearing *outside* the needle, which would let a model score
    # by reading some other part of the prompt.
    for item in make_passkey_items(tokenizer, depths=(512,), per_depth=6, seed=3):
        assert item.prompt.count(item.needle_text) == 1
        assert item.prompt.replace(item.needle_text, "").count(item.answer) == 0


def test_passkey_needle_sits_at_the_requested_depth_fraction(tokenizer):
    items = make_passkey_items(tokenizer, depths=(1024,), per_depth=5, seed=11,
                               depth_fractions=(0.0, 0.25, 0.5, 0.75, 1.0))

    fractions = [item.needle_depth_frac for item in items]
    assert fractions == [0.0, 0.25, 0.5, 0.75, 1.0]

    positions = [item.prompt.index(item.answer) / len(item.prompt) for item in items]
    assert positions == sorted(positions)
    assert positions[0] < positions[-1]


def test_passkey_answers_never_occur_in_the_filler(tokenizer):
    for item in make_passkey_items(tokenizer, depths=(256,), per_depth=8, seed=5):
        without_needle = item.prompt.replace(item.needle_text, "")
        assert item.answer not in without_needle


def test_passkey_answers_are_held_out_across_items(tokenizer):
    items = make_passkey_items(tokenizer, depths=(256, 512), per_depth=8, seed=2)

    answers = [item.answer for item in items]
    assert len(set(answers)) == len(answers)


def test_item_ids_encode_task_depth_and_index(tokenizer):
    items = make_passkey_items(tokenizer, depths=(256,), per_depth=3, seed=1)

    assert [item.id for item in items] == [
        "passkey-d256-0", "passkey-d256-1", "passkey-d256-2"]


def test_a_depth_below_the_scaffold_is_refused(tokenizer):
    with pytest.raises(ValueError, match="depth"):
        make_passkey_items(tokenizer, depths=(8,), per_depth=1, seed=1)


# --------------------------------------------------------------------- mqar ---

def test_mqar_queries_are_answerable_from_the_context(tokenizer):
    for item in make_mqar_items(tokenizer, depths=(512,), per_depth=4, seed=4):
        for key, value in zip(item.meta["queried_keys"], item.meta["answers"]):
            assert f"{key}: {value}" in item.prompt


def test_mqar_scored_binding_appears_exactly_once(tokenizer):
    # Once, in the table. If it also appeared in the demonstration block the
    # item would be answerable by copying the line above the question.
    for item in make_mqar_items(tokenizer, depths=(512,), per_depth=4, seed=4):
        assert item.prompt.count(item.needle_text) == 1
        assert item.meta["scored_key"] not in item.meta["demonstration_keys"]


def test_mqar_keys_and_values_are_unique_and_drawn_from_disjoint_pools(tokenizer):
    for item in make_mqar_items(tokenizer, depths=(512,), per_depth=4, seed=6):
        pairs = item.meta["pairs"]
        values = list(pairs.values())
        assert len(set(values)) == len(values)
        assert len(set(pairs)) == len(pairs)
        # Disjoint pools, so echoing a key can never score as a recalled value.
        assert not set(pairs).intersection(values)


def test_mqar_shows_earlier_queries_as_demonstrations_and_leaves_the_last_open(
        tokenizer):
    items = make_mqar_items(tokenizer, depths=(512,), per_depth=2, seed=9,
                            n_queries=3)

    for item in items:
        assert len(item.meta["queried_keys"]) == 3
        assert len(item.meta["demonstration_keys"]) == 2
        assert item.answer == item.meta["pairs"][item.meta["scored_key"]]
        assert item.prompt.endswith(f"\n{item.meta['scored_key']}:")
        for key in item.meta["demonstration_keys"]:
            assert f"\n{key}: {item.meta['pairs'][key]}" in item.prompt


def test_mqar_prompt_reaches_the_requested_depth(tokenizer):
    for item in make_mqar_items(tokenizer, depths=(1024,), per_depth=2, seed=1):
        measured = len(tokenizer.encode(item.prompt))
        assert 1024 - 48 <= measured <= 1024


# ----------------------------------------------------------------- controls ---

def test_copy_control_places_the_needle_next_to_the_question(tokenizer):
    for item in make_copy_control_items(tokenizer, per_item=4, seed=1):
        tail = item.prompt[item.prompt.index(item.answer):]
        assert len(tokenizer.encode(tail)) <= 24


def test_oracle_backend_scores_every_generated_item_correct(tokenizer):
    items = (
        make_passkey_items(tokenizer, depths=DEPTHS, per_depth=2, seed=1)
        + make_mqar_items(tokenizer, depths=(512,), per_depth=2, seed=1)
        + make_copy_control_items(tokenizer, per_item=4, seed=1)
    )

    scored = score_items(items, OracleBackend().generate_all(items))

    assert all(record["correct"] == 1 for record in scored)
    assert summarize(items, scored)["exact_match"] == 1.0


def test_oracle_backend_detects_a_formatter_that_drops_the_needle(tokenizer):
    items = make_passkey_items(tokenizer, depths=(256,), per_depth=4, seed=1)
    broken = [
        RetrievalItem(**{**item.__dict__,
                         "prompt": item.prompt.replace(item.needle_text, "")})
        for item in items
    ]

    scored = score_items(broken, OracleBackend().generate_all(broken))

    assert summarize(broken, scored)["exact_match"] == 0.0


def test_oracle_backend_detects_a_query_whose_binding_is_absent(tokenizer):
    items = make_mqar_items(tokenizer, depths=(512,), per_depth=3, seed=1,
                            n_queries=2)
    broken = [
        RetrievalItem(**{**item.__dict__,
                         "prompt": item.prompt.replace(
                             f"\n{item.needle_text}", "", 1)})
        for item in items
    ]

    scored = score_items(broken, OracleBackend().generate_all(broken))

    assert summarize(broken, scored)["exact_match"] == 0.0


def test_extract_answer_stops_at_the_first_line_for_recall_after_generation(
        tokenizer):
    item = make_mqar_items(tokenizer, depths=(512,), per_depth=1, seed=1,
                           n_queries=2)[0]

    assert extract_answer(item, f" {item.answer}\nnext: chatter") == item.answer


# ------------------------------------------------------- scoring primitives ---

def test_normalize_answer_ignores_case_padding_and_wrapping_punctuation():
    assert normalize_answer('  "Fizz".  ') == normalize_answer("fizz")
    assert normalize_answer("4821\n") == "4821"


def test_extract_answer_takes_the_first_digit_run_for_passkey(tokenizer):
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    assert extract_answer(item, " 4821 and then 9999") == "4821"
    assert extract_answer(item, "The pass key is 4821.") == "4821"


def test_extract_answer_stops_at_the_first_line_for_recall(tokenizer):
    item = make_mqar_items(tokenizer, depths=(512,), per_depth=1, seed=1,
                           n_queries=2)[0]

    assert extract_answer(item, " zebra\nunrelated chatter\n") == "zebra"


def test_scoring_is_exact_match_not_substring(tokenizer):
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]
    longer = item.answer + "0"

    scored = score_items([item], [longer])

    assert scored[0]["correct"] == 0


def test_score_items_records_the_response_and_extracted_answer(tokenizer):
    item = make_passkey_items(tokenizer, depths=(256,), per_depth=1, seed=1)[0]

    scored = score_items([item], ["  " + item.answer + " trailing"])

    assert scored[0]["id"] == item.id
    assert scored[0]["depth"] == 256
    assert scored[0]["extracted"] == item.answer
    assert scored[0]["response"] == "  " + item.answer + " trailing"
    assert scored[0]["correct"] == 1


def test_summarize_reports_a_per_depth_curve(tokenizer):
    items = make_passkey_items(tokenizer, depths=(256, 512), per_depth=2, seed=1)
    responses = [items[0].answer, "wrong", items[2].answer, items[3].answer]

    metrics = summarize(items, score_items(items, responses))

    assert metrics["exact_match"] == pytest.approx(0.75)
    assert metrics["exact_match_d256"] == pytest.approx(0.5)
    assert metrics["exact_match_d512"] == pytest.approx(1.0)
    assert metrics["n_d256"] == 2


def test_score_items_refuses_a_response_count_mismatch(tokenizer):
    items = make_passkey_items(tokenizer, depths=(256,), per_depth=2, seed=1)

    with pytest.raises(ValueError, match="response"):
        score_items(items, ["only-one"])
