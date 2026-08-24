"""Contracts for the Phase 4 tokenizer lab library.

Everything here is a property a candidate vocabulary must have *before* it is
allowed to be measured, or a property of a measurement that decides between
vocabularies. The two are kept in one file because the second is worthless
without the first: a bytes-per-token table over a tokenizer that cannot
round-trip its own input is a table of numbers about a broken artifact.
"""

import json
import math
from pathlib import Path

import pytest

from daedalus.tokenizer_train import (
    CANDIDATE_VOCAB_SIZES,
    ENDOFTEXT,
    IM_END,
    IM_START,
    INCUMBENT_VOCAB_SIZE,
    PINNED_SPECIAL_IDS,
    Q6_K_BITS_PER_WEIGHT,
    RoundTripError,
    bytes_per_token,
    byte_alphabet_coverage,
    embedding_cost,
    identifier_fragmentation,
    kv_bytes_per_context_token,
    longest_tokens,
    special_token_isolation,
    throughput,
    train_bpe,
    verify_round_trip,
    whitespace_behaviour,
)


CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 40,
    "def compute_total_price(items, tax_rate):\n    return sum(i.price for i in items) * (1 + tax_rate)\n" * 40,
    "Let $x = 3$ and $y = 4$; then $\\sqrt{x^2 + y^2} = 5$.\n" * 40,
    "user: how are you\nassistant: I am well, thank you.\n" * 40,
    "Ελληνικά, 中文, العربية, русский, 🚀 emoji, tabs\tand\nnewlines.\n" * 40,
]


@pytest.fixture(scope="module")
def small_tokenizer(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok") / "v300"
    train_bpe(CORPUS, vocab_size=300, out_dir=out)
    from daedalus.tokenizer_train import load_tokenizer
    return load_tokenizer(out)


# ------------------------------------------------------------- vocabulary ----

def test_trained_vocabulary_is_exactly_the_requested_size(small_tokenizer):
    """A vocabulary that lands near its target is not the vocabulary that was
    measured. The whole phase compares three exact sizes, and an off-by-a-few
    vocab would move both the embedding parameter count and the projected Q6_K
    bytes that the selection rule reads."""
    assert small_tokenizer.vocab_size == 300


def test_special_tokens_hold_their_pinned_ids(small_tokenizer):
    """`<|endoftext|>` at 0 is not cosmetic: `daedalus/data.py`'s DEFAULT_EOS_ID,
    `config.py`'s bos/eos/pad and every packed shard's document separator are
    all the literal integer 0, and `daedalus/chatml.py` documents 1/2 for the
    ChatML pair. A tokenizer that renumbers them silently repacks every shard
    against a different separator."""
    for token, expected in PINNED_SPECIAL_IDS.items():
        assert small_tokenizer.convert_tokens_to_ids(token) == expected, token
    assert PINNED_SPECIAL_IDS == {ENDOFTEXT: 0, IM_START: 1, IM_END: 2}


def test_every_byte_value_is_representable(small_tokenizer):
    """Byte-level BPE only has complete byte fallback if all 256 byte-level
    characters are actually in the vocabulary. Trained on a small corpus they
    are not, unless the trainer is seeded with the full initial alphabet."""
    coverage = byte_alphabet_coverage(small_tokenizer)
    assert coverage["missing"] == []
    assert coverage["covered"] == 256


def test_arbitrary_bytes_survive_a_round_trip(small_tokenizer):
    """The gate the phase brief states: a tokenizer that cannot round-trip
    arbitrary bytes is rejected before it is measured."""
    report = verify_round_trip(small_tokenizer)
    assert report["passed"] is True
    assert report["cases"] >= 8
    # latin-1 is a bijection onto 0..255, so this case *is* "arbitrary bytes".
    assert "all-256-bytes" in report["case_names"]


def test_round_trip_verification_rejects_a_lossy_tokenizer(small_tokenizer):
    """The verifier must fail loudly rather than return a report nobody reads."""

    class Lossy:
        """Drops non-ASCII, the classic silent tokenizer defect."""

        def __getattr__(self, name):
            return getattr(small_tokenizer, name)

        def decode(self, ids, **kwargs):
            return small_tokenizer.decode(ids, **kwargs).encode(
                "ascii", "ignore").decode("ascii")

    with pytest.raises(RoundTripError):
        verify_round_trip(Lossy())


def test_special_tokens_encode_as_single_ids(small_tokenizer):
    """A ChatML turn marker that splits into pieces is trained as text rather
    than as a boundary, which is the failure `daedalus/chatml.py` describes:
    invisible during training, visible only as a model that runs past its stop
    token."""
    isolation = special_token_isolation(small_tokenizer)
    for token in PINNED_SPECIAL_IDS:
        assert isolation[token]["n_ids"] == 1, token
        assert isolation[token]["isolated_in_context"] is True


# ------------------------------------------------------------ measurement ----

def test_bytes_per_token_is_bytes_over_tokens(small_tokenizer):
    """Stated as an identity so a later refactor cannot quietly switch it to
    characters per token, which differs from bytes per token by a factor of
    three on the CJK and emoji rows this corpus contains."""
    text = "hello world, 世界 🚀"
    measured = bytes_per_token(small_tokenizer, {"probe": [text]})
    n_tokens = len(small_tokenizer.encode(text, add_special_tokens=False))
    assert measured["probe"]["bytes"] == len(text.encode("utf-8"))
    assert measured["probe"]["tokens"] == n_tokens
    assert measured["probe"]["bytes_per_token"] == pytest.approx(
        len(text.encode("utf-8")) / n_tokens)


def test_bytes_per_token_reports_every_domain_separately(small_tokenizer):
    """The selection rule has a per-domain floor ("no domain regresses by more
    than 5%"), so an aggregate alone cannot decide it."""
    measured = bytes_per_token(small_tokenizer, {
        "general": [CORPUS[0]], "code": [CORPUS[1]], "math": [CORPUS[2]]})
    assert set(measured) == {"general", "code", "math", "__all__"}
    assert measured["__all__"]["bytes"] == sum(
        measured[d]["bytes"] for d in ("general", "code", "math"))


def test_identifier_fragmentation_counts_pieces_per_identifier(small_tokenizer):
    frag = identifier_fragmentation(small_tokenizer,
                                    ["compute_total_price", "tax_rate"])
    assert frag["n_identifiers"] == 2
    assert frag["mean_pieces"] >= 1.0
    assert frag["worst"]["identifier"] in {"compute_total_price", "tax_rate"}


def test_whitespace_behaviour_reports_indentation_and_newlines(small_tokenizer):
    """Indentation is most of Python's bytes; a tokenizer that spends one token
    per space costs more on code than its overall fertility suggests."""
    ws = whitespace_behaviour(small_tokenizer)
    assert ws["indent_4_spaces"]["n_ids"] >= 1
    assert ws["newline"]["n_ids"] >= 1
    assert "tokens_per_indent_level" in ws


def test_longest_tokens_are_reported_for_pathology_review(small_tokenizer):
    longest = longest_tokens(small_tokenizer, n=5)
    assert len(longest) == 5
    lengths = [entry["n_bytes"] for entry in longest]
    assert lengths == sorted(lengths, reverse=True)


def test_throughput_reports_both_directions(small_tokenizer):
    rates = throughput(small_tokenizer, CORPUS[:2])
    assert rates["encode_mb_per_s"] > 0
    assert rates["decode_mb_per_s"] > 0


# --------------------------------------------------------------- artifacts ----

def test_embedding_cost_uses_the_shipped_q6_k_lattice():
    """`token_embd.weight` ships Q6_K, not Q8_0 -- measured in
    `runs/preflight/token-embd-quant-grid.md` and pinned by
    `tests/test_export.py`. Projecting a vocabulary's artifact cost against
    Q8_0 would overstate the saving by 22%."""
    cost = embedding_cost(vocab_size=32768, hidden_size=768)
    assert cost["parameters"] == 32768 * 768
    assert Q6_K_BITS_PER_WEIGHT == pytest.approx(210 * 8 / 256)
    assert cost["q6_k_bytes"] == pytest.approx(
        32768 * 768 * Q6_K_BITS_PER_WEIGHT / 8)


def test_embedding_cost_falls_with_vocabulary_size():
    sizes = list(CANDIDATE_VOCAB_SIZES) + [INCUMBENT_VOCAB_SIZE]
    costs = [embedding_cost(v, 768)["q6_k_bytes"] for v in sorted(sizes)]
    assert costs == sorted(costs)


def test_candidate_sizes_are_the_three_the_phase_preregistered():
    assert CANDIDATE_VOCAB_SIZES == (24576, 32768, 40960)
    assert INCUMBENT_VOCAB_SIZE == 49152


def test_every_candidate_size_is_k_quant_blockable():
    """Every candidate must keep `vocab_size % 256 == 0`: `DaedalusConfig`
    asserts it, and llama.cpp falls back off k-quants for tensors it cannot
    block cleanly (`runs/eval/decode-vs-smollm2.md`)."""
    for size in CANDIDATE_VOCAB_SIZES + (INCUMBENT_VOCAB_SIZE,):
        assert size % 256 == 0


def test_kv_cache_is_neutral_to_vocabulary_size():
    """The KV cache is attention-shaped, so a vocabulary change moves neither
    its size nor its bandwidth. Stated as a measurement rather than as prose
    because "KV-neutral decode impact" is one of the phase's required
    readings, and a reader is entitled to the number rather than the claim."""
    small = kv_bytes_per_context_token(vocab_size=24576)
    large = kv_bytes_per_context_token(vocab_size=49152)
    assert small["kv_bytes_per_context_token"] == large["kv_bytes_per_context_token"]
    assert small["depends_on_vocab"] is False
