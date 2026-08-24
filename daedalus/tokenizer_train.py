"""Train and measure candidate V2 vocabularies (Phase 4).

The phase asks one question -- would a 24,576, 32,768 or 40,960-token
vocabulary be a better foundation for a future V2 than the 49,152-token
SmolLM2 vocabulary V1 reuses -- and the answer has to survive two traps that
make tokenizer comparisons routinely wrong.

**Perplexity per token is not comparable across vocabularies.** A larger
vocabulary packs more bytes into each token, so its per-token likelihood is
better *by construction*, with no improvement in the model. Every quality
number this module produces is therefore bits per *byte*, and the comparison
helper refuses a perplexity field outright rather than converting one.

**A tokenizer that cannot round-trip its input is not a candidate.** Byte-level
BPE has complete byte coverage only when the trainer is seeded with the full
256-character byte alphabet; trained on a finite corpus without that seed it
silently omits byte values it never saw, and the omission shows up much later
as a model that cannot reproduce a rare character. `verify_round_trip` is a
precondition, run before any measurement, and it raises.

Two facts about the shipped artifact set the units the artifact-cost numbers
are quoted in:

  - `token_embd.weight` ships **Q6_K**, not the blueprint's Q8_0 -- measured in
    `runs/preflight/token-embd-quant-grid.md`. It is tied, so that one tensor
    is both the input embedding and the output projection, about 25% of the
    model. Projecting a vocabulary's saving against Q8_0 would overstate it.
  - The KV cache is attention-shaped and **vocabulary-neutral**. That is worth
    a measured zero rather than an assertion, because "smaller vocabulary,
    smaller memory" is exactly the kind of claim that gets extended to the
    wrong quantity.

Nothing here touches the released V1 weights or Daedalus-Code. A trained
tokenizer cannot be transplanted into a trained model -- every embedding row
and every output logit is indexed by the vocabulary the model was trained
under -- so the deliverable is a migration report for a future from-scratch
run, not a new checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

# The three ids the rest of the codebase hard-codes as integers. `data.py`'s
# DEFAULT_EOS_ID is the literal 0 that separates documents in every packed
# shard, `config.py`'s to_hf_dict writes bos/eos/pad as 0, and `chatml.py`
# records 1/2 for the ChatML pair. A candidate vocabulary that renumbers any of
# them is not a drop-in for the pipeline, so the trainer pins them instead of
# discovering them.
ENDOFTEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PINNED_SPECIAL_IDS: Dict[str, int] = {ENDOFTEXT: 0, IM_START: 1, IM_END: 2}
SPECIAL_TOKENS = tuple(PINNED_SPECIAL_IDS)

CANDIDATE_VOCAB_SIZES = (24576, 32768, 40960)
INCUMBENT_VOCAB_SIZE = 49152          # SmolLM2, what V1 ships
INCUMBENT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"

# ggml's `block_q6_K` is 210 bytes per 256 weights: ql[128] + qh[64] +
# scales[16] + d (one f16). 6.5625 bits per weight.
Q6_K_BITS_PER_WEIGHT = 210 * 8 / 256

# The shipped attention shape, from `DaedalusConfig`'s defaults. Restated as
# numbers rather than imported so this module stays importable without torch.
_SHIPPED_KV_HEADS = 4
_SHIPPED_HEAD_DIM = 64
_SHIPPED_ATTN_LAYERS = 6
_KV_CACHE_DTYPE_BYTES = 2             # f16 K and V, llama.cpp's default


class RoundTripError(ValueError):
    """Raised when a candidate cannot reproduce its own input."""


# ------------------------------------------------------------------ training ---

def train_bpe(texts: Iterable[str], *, vocab_size: int, out_dir,
              min_frequency: int = 2, show_progress: bool = False,
              chat_template: Optional[str] = None) -> Path:
    """Train one byte-level BPE vocabulary and save it as an HF tokenizer dir.

    Three settings are load-bearing and none of them is the default:

    - `initial_alphabet=ByteLevel.alphabet()` seeds all 256 byte characters, so
      byte coverage is complete whether or not the sample happened to contain a
      given byte. Without it `byte_alphabet_coverage` finds holes.
    - `special_tokens` are passed first, which is what gives them ids 0/1/2.
      The trainer assigns special ids before the alphabet and before any merge,
      so the pinning is a consequence of ordering rather than a rename
      afterwards (renaming afterwards would leave the merges referring to the
      old ids).
    - `add_prefix_space=False` matches SmolLM2's configuration, so a candidate
      differs from the incumbent in vocabulary size and merges only. Changing
      two things at once is how a tokenizer comparison stops being one.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    if vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise ValueError(
            f"vocab_size {vocab_size} leaves no room for merges above the "
            f"{len(SPECIAL_TOKENS)} specials and the 256-byte alphabet")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                       use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=min_frequency,
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    trained = tokenizer.get_vocab_size()
    if trained != vocab_size:
        raise ValueError(
            f"trainer produced {trained} tokens, not the requested "
            f"{vocab_size}; the sample is too small or too repetitive to "
            f"support this many merges, and a vocabulary that is not the size "
            f"it claims would misreport both embedding parameters and "
            f"projected Q6_K bytes")

    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=ENDOFTEXT, eos_token=ENDOFTEXT,
        unk_token=ENDOFTEXT, pad_token=ENDOFTEXT,
        additional_special_tokens=[IM_START, IM_END],
        clean_up_tokenization_spaces=False,
    )
    if chat_template is not None:
        fast.chat_template = chat_template

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))

    saved = load_tokenizer(out_dir)
    for token, expected in PINNED_SPECIAL_IDS.items():
        got = saved.convert_tokens_to_ids(token)
        if got != expected:
            raise ValueError(
                f"{token} landed at id {got}, not the pinned {expected}; every "
                f"packed shard separates documents with the literal integer "
                f"{PINNED_SPECIAL_IDS[ENDOFTEXT]}")
    if saved.vocab_size != vocab_size:
        raise ValueError(
            f"saved vocabulary is {saved.vocab_size}, not {vocab_size}")
    return out_dir


def load_tokenizer(path_or_name):
    """An HF tokenizer from a local lab directory or a Hub name."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path_or_name))


# -------------------------------------------------------------- verification ---

def byte_alphabet_coverage(tokenizer) -> dict:
    """Which of the 256 byte-level characters are in the vocabulary.

    ByteLevel BPE represents raw bytes as a fixed 256-character alphabet, so
    "complete byte fallback" is exactly "all 256 of those characters are
    tokens". Anything missing is a byte the tokenizer can never emit.
    """
    from tokenizers import pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = tokenizer.get_vocab()
    missing = sorted(ch for ch in alphabet if ch not in vocab)
    return {"expected": len(alphabet), "covered": len(alphabet) - len(missing),
            "missing": missing}


# One case per way a tokenizer is usually lossy. `all-256-bytes` is the literal
# "arbitrary bytes" requirement: latin-1 is a bijection onto 0..255, so the
# string below carries every byte value exactly once.
_ROUND_TRIP_CASES: Dict[str, str] = {
    "ascii": "".join(chr(c) for c in range(32, 127)),
    "all-256-bytes": bytes(range(256)).decode("latin-1"),
    "control-and-whitespace": "\t\n\r\x0b\x0c \u00a0\u2009\u3000",
    "indentation": "def f():\n    if x:\n        return [\n            1,\n        ]\n",
    "cjk": "私は日本語を話します。中文简体與繁體。한국어도 됩니다.",
    "rtl-and-combining": "العربية עברית नमस्ते é e\u0301 ﬁ",
    "emoji-zwj": "🚀 👩\u200d💻 👨\u200d👩\u200d👧\u200d👦 ✅ 🇬🇧",
    "math-and-symbols": "∀x∈ℝ: x²≥0 — ∑ᵢ₌₁ⁿ i = n(n+1)/2 ‰ ¤ ﷽",
    "code-punctuation": "a=b->c[d]{e}(f)|g&h^i~j`k'l\"m\\n/o?p!q@r#s$t%u",
    "repeated-runs": "aaaaaaaaaaaaaaaa" + "\n" * 12 + " " * 24 + "\t" * 8,
}


def verify_round_trip(tokenizer) -> dict:
    """Refuse a candidate that cannot reproduce arbitrary bytes.

    Run before any measurement, and raises rather than returning a verdict,
    because a bytes-per-token table over a lossy tokenizer is a table of
    numbers about a broken artifact.
    """
    coverage = byte_alphabet_coverage(tokenizer)
    failures = []
    if coverage["missing"]:
        failures.append({
            "case": "byte-alphabet",
            "reason": f"{len(coverage['missing'])} of 256 byte characters are "
                      f"absent from the vocabulary",
        })

    for name, text in _ROUND_TRIP_CASES.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded != text:
            failures.append({
                "case": name,
                "reason": "decode(encode(x)) != x",
                "input_bytes": len(text.encode("utf-8")),
                "output_bytes": len(decoded.encode("utf-8")),
            })

    if failures:
        raise RoundTripError(
            "tokenizer failed round-trip verification and is rejected before "
            "measurement: " + json.dumps(failures))

    return {
        "passed": True,
        "cases": len(_ROUND_TRIP_CASES),
        "case_names": sorted(_ROUND_TRIP_CASES),
        "byte_alphabet": coverage,
    }


def special_token_isolation(tokenizer) -> dict:
    """Each pinned special must encode to exactly one id, alone and in context.

    In context matters separately: a marker that is one token on its own but
    merges with neighbouring text when a turn actually renders it is still
    broken, and that is the arrangement `chatml.py` produces.
    """
    report = {}
    for token, expected_id in PINNED_SPECIAL_IDS.items():
        alone = tokenizer.encode(token, add_special_tokens=False)
        context = tokenizer.encode(f"hello{token}world",
                                   add_special_tokens=False)
        report[token] = {
            "id": expected_id,
            "n_ids": len(alone),
            "ids": list(alone),
            "isolated_in_context": expected_id in list(context),
        }
    return report


# -------------------------------------------------------------- measurement ---

def bytes_per_token(tokenizer, domains: Mapping[str, Sequence[str]]) -> dict:
    """Fertility per domain, plus the pooled `__all__` row.

    Bytes, not characters: a CJK codepoint is three UTF-8 bytes and an emoji
    four, so characters-per-token would flatter a vocabulary on exactly the
    text where it is weakest. Pooled rather than averaged, so a domain with a
    handful of long documents cannot outvote one with many short ones.
    """
    result: Dict[str, dict] = {}
    total_bytes = total_tokens = 0
    for domain, texts in domains.items():
        n_bytes = n_tokens = n_docs = 0
        for text in texts:
            n_bytes += len(text.encode("utf-8"))
            n_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            n_docs += 1
        result[domain] = {
            "bytes": n_bytes,
            "tokens": n_tokens,
            "documents": n_docs,
            "bytes_per_token": (n_bytes / n_tokens) if n_tokens else float("nan"),
        }
        total_bytes += n_bytes
        total_tokens += n_tokens
    result["__all__"] = {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "documents": sum(r["documents"] for r in result.values()),
        "bytes_per_token": (total_bytes / total_tokens) if total_tokens
        else float("nan"),
    }
    return result


def identifier_fragmentation(tokenizer, identifiers: Sequence[str]) -> dict:
    """How many pieces a code identifier costs.

    Fertility over whole files hides this: a file is mostly punctuation and
    keywords, which every vocabulary handles, while the identifiers are where a
    code-weak vocabulary actually spends its budget.
    """
    pieces = []
    for identifier in identifiers:
        n = len(tokenizer.encode(identifier, add_special_tokens=False))
        pieces.append({"identifier": identifier, "pieces": n,
                       "bytes": len(identifier.encode("utf-8"))})
    if not pieces:
        return {"n_identifiers": 0, "mean_pieces": float("nan"), "worst": None}
    worst = max(pieces, key=lambda p: (p["pieces"], p["identifier"]))
    single = sum(1 for p in pieces if p["pieces"] == 1)
    return {
        "n_identifiers": len(pieces),
        "mean_pieces": sum(p["pieces"] for p in pieces) / len(pieces),
        "single_token_frac": single / len(pieces),
        "bytes_per_piece": (sum(p["bytes"] for p in pieces)
                            / sum(p["pieces"] for p in pieces)),
        "worst": worst,
    }


def whitespace_behaviour(tokenizer) -> dict:
    """Indentation and newline cost, which dominate code fertility.

    Python indents four spaces a level and nests three or four levels deep, so
    a vocabulary that has no multi-space tokens pays for the whole left margin
    of every file.
    """
    def n_ids(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    levels = {level: n_ids(" " * (4 * level)) for level in (1, 2, 3, 4)}
    return {
        "indent_4_spaces": {"n_ids": levels[1]},
        "indent_8_spaces": {"n_ids": levels[2]},
        "indent_16_spaces": {"n_ids": levels[4]},
        "tab": {"n_ids": n_ids("\t")},
        "newline": {"n_ids": n_ids("\n")},
        "double_newline": {"n_ids": n_ids("\n\n")},
        "newline_then_indent": {"n_ids": n_ids("\n    ")},
        "tokens_per_indent_level": {str(k): v for k, v in levels.items()},
    }


def longest_tokens(tokenizer, n: int = 20) -> List[dict]:
    """The longest tokens, for pathology review.

    A vocabulary trained on scraped text reliably learns a few absurd merges --
    a licence header, a base64 run, a repeated separator bar. They are wasted
    rows in the highest-leverage tensor in the model, and they are only visible
    if someone looks.
    """
    from tokenizers import pre_tokenizers

    # ByteLevel stores bytes as printable stand-ins, so token *string* length
    # over-counts multi-byte characters and under-counts nothing. Decode each
    # piece back to real bytes before measuring.
    byte_decoder = {ch: i for i, ch in
                    enumerate(pre_tokenizers.ByteLevel.alphabet())}
    entries = []
    for token, index in tokenizer.get_vocab().items():
        if token in PINNED_SPECIAL_IDS:
            continue
        try:
            raw = bytes(byte_decoder[ch] for ch in token)
        except KeyError:
            raw = token.encode("utf-8")
        entries.append({"id": index, "token": token, "n_bytes": len(raw),
                        "text": raw.decode("utf-8", errors="replace")})
    entries.sort(key=lambda e: (-e["n_bytes"], e["id"]))
    return entries[:n]


def throughput(tokenizer, texts: Sequence[str], repeats: int = 1) -> dict:
    """Encode and decode rate in MB/s.

    Reported because a vocabulary is also a runtime cost, and because a
    pathologically slow tokenizer is a symptom worth catching -- but it is not
    in the selection rule: at these rates tokenization is never the bottleneck
    beside a forward pass.
    """
    payload = list(texts) * max(1, repeats)
    n_bytes = sum(len(t.encode("utf-8")) for t in payload)

    start = time.perf_counter()
    encoded = [tokenizer.encode(t, add_special_tokens=False) for t in payload]
    encode_sec = time.perf_counter() - start

    start = time.perf_counter()
    for ids in encoded:
        tokenizer.decode(ids, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False)
    decode_sec = time.perf_counter() - start

    n_tokens = sum(len(ids) for ids in encoded)
    return {
        "bytes": n_bytes,
        "tokens": n_tokens,
        "encode_sec": encode_sec,
        "decode_sec": decode_sec,
        "encode_mb_per_s": (n_bytes / 1e6) / encode_sec if encode_sec else float("inf"),
        "decode_mb_per_s": (n_bytes / 1e6) / decode_sec if decode_sec else float("inf"),
    }


# ----------------------------------------------------------- artifact cost ---

def embedding_cost(vocab_size: int, hidden_size: int = 768) -> dict:
    """What one vocabulary costs in the tied embedding tensor.

    Quoted at Q6_K because that is what `token_embd.weight` actually ships as
    (`runs/preflight/token-embd-quant-grid.md`), and at f16 for the training
    checkpoint. Tied embeddings mean this single tensor is both the input table
    and the output projection, so the saving is counted once, not twice.
    """
    parameters = vocab_size * hidden_size
    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "parameters": parameters,
        "f16_bytes": parameters * 2,
        "q6_k_bytes": parameters * Q6_K_BITS_PER_WEIGHT / 8,
        "q6_k_bits_per_weight": Q6_K_BITS_PER_WEIGHT,
        "k_quant_blockable": vocab_size % 256 == 0 and hidden_size % 256 == 0,
    }


def kv_bytes_per_context_token(vocab_size: int,
                               kv_heads: int = _SHIPPED_KV_HEADS,
                               head_dim: int = _SHIPPED_HEAD_DIM,
                               attn_layers: int = _SHIPPED_ATTN_LAYERS,
                               dtype_bytes: int = _KV_CACHE_DTYPE_BYTES) -> dict:
    """The KV cache cost per context token -- and the fact it does not move.

    `vocab_size` is accepted and deliberately unused in the arithmetic: the
    point of the reading is that the answer is the same for every candidate, so
    "smaller vocabulary" may be claimed for the embedding tensor and must not
    be extended to decode-time memory or bandwidth.
    """
    per_token = 2 * kv_heads * head_dim * attn_layers * dtype_bytes
    return {
        "vocab_size": vocab_size,
        "kv_bytes_per_context_token": per_token,
        "depends_on_vocab": False,
        "note": "the KV cache is attention-shaped; a vocabulary change moves "
                "the embedding tensor and nothing in this cache",
    }
