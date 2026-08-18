"""Tests for daedalus/chatml.py -- ChatML rendering and SFT label masking.

The two properties worth testing here are the ones that fail silently: label
alignment against Daedalus's internal shift, and which tokens are supervised.
Both produce a converging loss and a bad model rather than an error.
"""
import pytest
import torch

from daedalus.chatml import (
    IGNORE_INDEX,
    IM_END,
    IM_START,
    assistant_char_count,
    encode_sft_example,
    has_long_cot,
    keep_example,
    pad_batch,
    render_messages,
    render_prompt,
)
from daedalus.config import PRESETS
from daedalus.data import get_tokenizer
from daedalus.model import Daedalus

CHAT = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]


# ------------------------------------------------------------- rendering ---

def test_render_messages_is_chatml():
    text = render_messages(CHAT)
    assert text == (f"{IM_START}user\nWhat is the capital of France?{IM_END}\n"
                    f"{IM_START}assistant\nParis.{IM_END}\n")


def test_render_prompt_stops_where_the_model_continues():
    """Training and inference must use the same renderer; a prompt that ends
    anywhere else silently trains/queries a different format."""
    assert render_prompt(CHAT[:1]).endswith(f"{IM_START}assistant\n")
    assert render_prompt(CHAT[:1]).startswith(render_messages(CHAT[:1]))


# ---------------------------------------------------------------- filters ---

def test_keep_example_rejects_missing_or_empty_assistant_turn():
    assert not keep_example([{"role": "user", "content": "hi"}])
    assert not keep_example([{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "   "}])
    assert keep_example(CHAT)


def test_keep_example_enforces_the_conciseness_cap():
    long_chat = [{"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "x" * 5000}]
    assert not keep_example(long_chat, max_assistant_chars=1200)
    assert keep_example(long_chat, max_assistant_chars=10_000)


def test_keep_example_drops_chain_of_thought():
    cot = [{"role": "user", "content": "2+2?"},
           {"role": "assistant", "content": "Let me think step by step. It is 4."}]
    assert has_long_cot(cot)
    assert not keep_example(cot)
    assert keep_example(cot, drop_cot=False)  # opt out, for an ablation


def test_cot_detection_ignores_the_user_turn():
    """A user who says 'let me think' must not disqualify a clean answer."""
    chat = [{"role": "user", "content": "Let me think... what is 2+2?"},
            {"role": "assistant", "content": "4."}]
    assert not has_long_cot(chat)
    assert keep_example(chat)


def test_assistant_char_count_counts_only_assistant():
    assert assistant_char_count(CHAT) == len("Paris.")


# ------------------------------------------------------------------ masking ---

@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


def test_segmented_encoding_matches_whole_string(tok):
    """encode_sft_example tokenizes each segment separately and concatenates.
    That is only safe if it reproduces tokenizing the rendered string in one
    go -- otherwise training text differs from inference text by a token here
    and there, which is invisible and degrades quality."""
    ids, _ = encode_sft_example(CHAT, tok)
    whole = tok.encode(render_messages(CHAT), add_special_tokens=False)
    assert ids == whole


def test_only_assistant_content_is_supervised(tok):
    ids, labels = encode_sft_example(CHAT, tok)
    assert len(ids) == len(labels)

    supervised = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
    assert supervised, "something must be supervised"
    # Labels are unshifted: wherever supervised, the label IS the input token.
    for i in supervised:
        assert labels[i] == ids[i]

    # The supervised span decodes to exactly the assistant's reply plus its
    # terminator -- not the "<|im_start|>assistant\n" header the model is
    # handed at inference.
    decoded = tok.decode([ids[i] for i in supervised])
    assert decoded.startswith("Paris.")
    assert IM_END in decoded
    assert "assistant" not in decoded


def test_the_stop_token_is_supervised(tok):
    """Without <|im_end|> in the labels the model never learns to stop, and
    generation runs to max_tokens every time."""
    ids, labels = encode_sft_example(CHAT, tok)
    im_end_id = tok.convert_tokens_to_ids(IM_END)
    supervised_ids = [l for l in labels if l != IGNORE_INDEX]
    assert im_end_id in supervised_ids


def test_prompt_tokens_are_masked(tok):
    ids, labels = encode_sft_example(CHAT, tok)
    user_ids = tok.encode("What is the capital of France?", add_special_tokens=False)
    # every position holding a user-content token must be ignored
    for i, t in enumerate(ids):
        if t in user_ids and labels[i] != IGNORE_INDEX:
            pytest.fail(f"user token {t} at {i} was supervised")


def test_multi_turn_supervises_every_assistant_turn(tok):
    chat = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Goodbye."},
    ]
    ids, labels = encode_sft_example(chat, tok)
    decoded = tok.decode([l for l in labels if l != IGNORE_INDEX])
    assert "Hello." in decoded and "Goodbye." in decoded
    assert "Hi" not in decoded and "Bye" not in decoded


def test_oversized_example_is_dropped_not_truncated(tok):
    """A truncated example ends with no <|im_end|>, teaching the model to run
    past its stop token -- a failure that only appears at inference."""
    big = [{"role": "user", "content": "hi"},
           {"role": "assistant", "content": "word " * 5000}]
    assert encode_sft_example(big, tok, max_len=128) is None


def test_example_with_no_supervised_tokens_is_dropped(tok):
    assert encode_sft_example([{"role": "user", "content": "hi"}], tok) is None


# ------------------------------------------------------------------ padding ---

def test_pad_batch_pads_labels_with_ignore_index():
    ids, labels = pad_batch([([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, 5])],
                            pad_id=0)
    assert ids == [[1, 2, 3], [4, 5, 0]]
    assert labels == [[-100, 2, 3], [-100, 5, IGNORE_INDEX]]


# -------------------------------------------------- alignment against model ---

def test_model_accepts_the_mask_and_honours_ignore_index(tok):
    """What this actually proves: the model consumes an unshifted, -100-masked
    label tensor of the same width as the input, and -100 positions really are
    excluded from the loss.

    It does NOT prove label alignment. A label vector shifted by one would
    still land inside the supervised span here and still move the loss, so
    this perturbation cannot detect it. The alignment property is pinned
    directly by `labels[i] == ids[i]` in
    test_only_assistant_content_is_supervised -- a shifted encoder would
    produce labels[i] == ids[i+1] and fail there.
    """
    cfg = PRESETS["tiny"]
    torch.manual_seed(0)
    model = Daedalus(cfg)
    model.eval()

    ids = list(range(1, 33))
    labels = [IGNORE_INDEX] * 16 + ids[16:]
    x = torch.tensor([ids])
    y = torch.tensor([labels])
    with torch.no_grad():
        _, base_loss, _ = model(x, targets=y)
    assert torch.isfinite(base_loss)

    # Corrupt one supervised label; loss must move. If labels were shifted by
    # one here, the model would be scoring a different position and this
    # perturbation would land on an ignored slot.
    bad = list(labels)
    bad[20] = (bad[20] + 7) % cfg.vocab_size
    with torch.no_grad():
        _, bad_loss, _ = model(x, targets=torch.tensor([bad]))
    assert not torch.isclose(base_loss, bad_loss)

    # Masking that same position out must also move the loss, and the number
    # of contributing positions is what changes -- a sanity check that -100
    # is honoured rather than silently trained on.
    masked = list(labels)
    masked[20] = IGNORE_INDEX
    with torch.no_grad():
        _, masked_loss, _ = model(x, targets=torch.tensor([masked]))
    assert not torch.isclose(base_loss, masked_loss)
