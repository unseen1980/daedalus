"""ChatML rendering and SFT label masking for `post` (AGENT.md SS4).

Scope: turning a chat example into `(input_ids, labels)`. The training driver
lives in post.py; everything here is pure and CPU-only so the parts with real
correctness risk can be tested without a GPU.

Two things here can silently produce a worse model rather than an error, which
is the class this project has been told to hunt:

  - **Label alignment.** Daedalus shifts internally: `model.forward` compares
    `logits[:, :-1]` against `targets[:, 1:]` (model.py's forward docstring and
    the `targets[:, 1:]` reshape). So `labels` is the same length as
    `input_ids` and *unshifted* -- `labels[i]` is the token at position i, and
    the model learns to predict it from position i-1. Shifting again here
    would train the model to predict the token it was just given, which
    converges to a plausible-looking loss and a useless model.

  - **Which tokens are supervised.** Only assistant content, plus the
    `<|im_end|>` that terminates it -- without that the model never learns to
    stop. Prompt tokens are -100. Supervising the prompt trains the model to
    generate the user's turns too, which shows up as rambling
    self-conversation at inference rather than as a bad number in training.

The SmolLM2 tokenizer already carries `<|im_start|>` (1) and `<|im_end|>` (2)
as real special tokens, so no vocabulary change is needed and the blueprint's
"tokenizer byte-identical" constraint is untouched.
"""
from typing import Dict, List, Optional, Sequence, Tuple

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IGNORE_INDEX = -100

# AGENT.md SS4: "biased toward concise replies. No long chain-of-thought
# traces -- they measurably hurt models under 3B."
DEFAULT_MAX_ASSISTANT_CHARS = 1200
DEFAULT_COT_MARKERS = (
    "<think>", "</think>",
    "let me think", "let's think step by step", "let me work through",
    "step 1:", "step 1.", "first, let's", "chain of thought",
    "reasoning:", "let me break this down",
)


def render_messages(messages: Sequence[Dict[str, str]]) -> str:
    """The exact text form the model is trained on and must be prompted with.

    A mismatch between training and inference formatting is invisible at
    training time and shows up only as a model that ignores its prompt, so
    post.py and export/inference must both go through this function.
    """
    return "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
                   for m in messages)


def render_prompt(messages: Sequence[Dict[str, str]]) -> str:
    """Same rendering, but ending at the point the assistant should continue
    from -- for generation and for eval."""
    return render_messages(messages) + f"{IM_START}assistant\n"


def assistant_char_count(messages: Sequence[Dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages if m["role"] == "assistant")


def has_long_cot(messages: Sequence[Dict[str, str]],
                 markers: Sequence[str] = DEFAULT_COT_MARKERS) -> bool:
    """Marker-based, deliberately. Length alone cannot distinguish a long
    useful answer from a long reasoning trace, and this filter runs alongside
    a length cap that already handles the former."""
    for m in messages:
        if m["role"] != "assistant":
            continue
        lowered = m["content"].lower()
        if any(marker in lowered for marker in markers):
            return True
    return False


def keep_example(messages: Sequence[Dict[str, str]],
                 max_assistant_chars: int = DEFAULT_MAX_ASSISTANT_CHARS,
                 drop_cot: bool = True) -> bool:
    """AGENT.md SS4's two content filters, plus the structural minimum: an
    example with no assistant turn contributes no supervised tokens at all and
    would otherwise become an all -100 row that wastes a slot and contributes
    a zero gradient."""
    if not any(m["role"] == "assistant" and m["content"].strip()
               for m in messages):
        return False
    if assistant_char_count(messages) > max_assistant_chars:
        return False
    if drop_cot and has_long_cot(messages):
        return False
    return True


def encode_sft_example(messages: Sequence[Dict[str, str]], tokenizer,
                       max_len: int = 2048,
                       ) -> Optional[Tuple[List[int], List[int]]]:
    """`(input_ids, labels)` for one chat example, or None if it does not fit.

    Returns None rather than truncating. A truncated example ends mid-sentence
    with no `<|im_end|>`, which teaches the model to run past its stop token --
    the failure is at inference, long after the loss looked fine.

    Segments are tokenized separately and concatenated. That is safe here
    because every boundary falls on a special token or a newline immediately
    after one; `test_segmented_encoding_matches_whole_string` pins that
    against the tokenizer rather than trusting it.
    """
    input_ids: List[int] = []
    labels: List[int] = []
    for m in messages:
        header = tokenizer.encode(f"{IM_START}{m['role']}\n",
                                  add_special_tokens=False)
        body = tokenizer.encode(f"{m['content']}{IM_END}\n",
                                add_special_tokens=False)
        input_ids += header + body
        # The header is prompt scaffolding even for the assistant turn: the
        # model is *given* "<|im_start|>assistant\n" at inference, so training
        # it to produce those tokens teaches it to open turns it was handed.
        labels += [IGNORE_INDEX] * len(header)
        labels += body if m["role"] == "assistant" else [IGNORE_INDEX] * len(body)

    if not input_ids or len(input_ids) > max_len:
        return None
    if all(l == IGNORE_INDEX for l in labels):
        return None
    return input_ids, labels


def pad_batch(examples: Sequence[Tuple[List[int], List[int]]],
              pad_id: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Right-pad to the longest example. Padding is -100 in `labels`, so it
    contributes nothing to the loss and the pad id's own embedding never gets
    a gradient from it."""
    width = max(len(ids) for ids, _ in examples)
    padded_ids, padded_labels = [], []
    for ids, labels in examples:
        gap = width - len(ids)
        padded_ids.append(list(ids) + [pad_id] * gap)
        padded_labels.append(list(labels) + [IGNORE_INDEX] * gap)
    return padded_ids, padded_labels
