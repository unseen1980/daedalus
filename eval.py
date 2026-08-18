"""Daedalus evaluation (AGENT.md SS3, item 3): held-out bits-per-byte plus
loglikelihood cloze-format accuracy on HellaSwag, ARC-Easy, PIQA, OpenBookQA,
WinoGrande. ARC-Challenge and exact-match GSM8K are deliberately excluded --
too noisy at this scale (AGENT.md SS4). Report mean over the last 3-5
checkpoints, per spec.

Cloze scoring follows lm-evaluation-harness **exactly**, because every
published number we compare against comes from it and small formatting
differences move these scores by points:

  - the prompt templates are the harness's own (`Question: {q}\nAnswer:` for
    ARC-Easy/PIQA, the `activity_label: ctx_a ctx_b` rebuild plus bracket
    cleanup for HellaSwag, bare `question_stem` for OpenBookQA), and each
    continuation is joined with the harness's `target_delimiter` -- a single
    leading space (`lm_eval/api/task.py`: `(ctx, f"{target_delimiter}{cont}")`);
  - `acc` is the argmax of the **sum** continuation log-probability;
  - `acc_norm` is the argmax of that sum divided by the **character length of
    the choice string** (`completion_len = [float(len(i)) for i in choices]`),
    not by its token count.

Both are reported. Published tables for this model class quote `acc_norm` for
HellaSwag/ARC/PIQA/OpenBookQA and `acc` for WinoGrande, which is what
`headline_metric()` selects.

WinoGrande is the harness's `multiple_input` case: its blank sits mid-sentence,
so each candidate substitutes into the *context* (`sentence[:idx] + option`)
and the shared continuation is the stripped text after the blank. Because the
continuation is identical across candidates, only `acc` is meaningful there --
and the harness's own task config lists only `acc`.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from daedalus.config import PRESETS
from daedalus.model import Daedalus


def _git_short_sha() -> str:
    """Best-effort short SHA for the W&B run name (AGENT.md SS5.1's
    `<run>-<sha>` convention); never raises."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------- cloze ---

TARGET_DELIMITER = " "   # lm-evaluation-harness's default; joins ctx to choice


@dataclass
class ClozeExample:
    candidates: List[Tuple[str, str]]   # (context, continuation) per candidate
    label: int
    choice_lengths: Optional[List[int]] = None
    """Character length of each *choice* string (before the delimiter space),
    the denominator lm-evaluation-harness uses for `acc_norm`. `None` means the
    task reports `acc` only -- WinoGrande, whose candidates share one
    continuation, so normalising by its length cannot change the argmax."""


def score_ids(model, context_ids: Sequence[int], continuation_ids: Sequence[int],
             device: str = "cpu") -> float:
    """Sum log P(continuation_ids | context_ids).

    Deliberately *not* length-normalised: the harness scores with the raw sum
    and applies its own character-length denominator afterwards for `acc_norm`
    (see the module docstring). Normalising here by token count would be a
    third, non-comparable metric.
    """
    if not context_ids or not continuation_ids:
        return float("-inf")
    ids = torch.tensor([list(context_ids) + list(continuation_ids)], device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        logits, _, _ = model(ids, targets=None, return_logits=True)
    model.train(was_training)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    start = len(context_ids) - 1
    return sum(logprobs[start + j, tok].item()
              for j, tok in enumerate(continuation_ids))


def score_text(model, tokenizer, context: str, continuation: str,
              device: str = "cpu") -> float:
    """Tokenizes `context + continuation` jointly and splits at the context's
    token count -- the standard cloze-scoring approach (lm-evaluation-harness's
    `_encode_pair`), which avoids the failure mode of separately tokenizing each
    half and concatenating ids (BPE isn't concatenation-invariant at the
    boundary).

    Trailing whitespace is moved from the context onto the continuation first,
    exactly as `_encode_pair` does: a context ending in a space would otherwise
    tokenize differently on its own than it does inside the joined string.
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    # `add_special_tokens=False`, explicitly, matching lm-evaluation-harness's
    # HFLM default for causal models (`add_bos_token=False`). Not cosmetic and
    # not uniform across models if left to the tokenizer: measured here,
    # facebook/opt-125m's tokenizer prepends BOS (id 2) by default while
    # pythia-160m, gpt-neo-125m and SmolLM2's do not. Leaving it implicit would
    # have scored one peer in the comparison under a different convention from
    # the other four -- and from the published numbers, which are all
    # BOS-free -- while looking like a model difference.
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    cont_ids = full_ids[len(ctx_ids):]
    return score_ids(model, full_ids[:len(ctx_ids)], cont_ids, device)


def score_example(model, tokenizer, example: ClozeExample,
                 device: str = "cpu") -> List[float]:
    """Raw sum log-likelihood per candidate."""
    return [score_text(model, tokenizer, ctx, cont, device)
           for ctx, cont in example.candidates]


def predict(model, tokenizer, example: ClozeExample, device: str = "cpu") -> int:
    """`acc`: argmax of the unnormalized sum log-likelihood."""
    return int(np.argmax(score_example(model, tokenizer, example, device)))


def predict_norm(model, tokenizer, example: ClozeExample,
                device: str = "cpu") -> int:
    """`acc_norm`: argmax of sum log-likelihood / choice character length."""
    scores = score_example(model, tokenizer, example, device)
    if not example.choice_lengths:
        return int(np.argmax(scores))
    lengths = np.array(example.choice_lengths, dtype=float)
    return int(np.argmax(np.array(scores) / np.maximum(lengths, 1.0)))


def evaluate_cloze_task(model, tokenizer, examples: Sequence[ClozeExample],
                        device: str = "cpu") -> dict:
    """Returns both harness metrics, scoring each example once.

    `acc_norm` is omitted (None) for tasks whose examples carry no
    `choice_lengths` -- WinoGrande -- rather than silently reporting `acc`
    twice under two names.
    """
    n = len(examples)
    has_norm = bool(examples) and all(ex.choice_lengths for ex in examples)
    correct = correct_norm = 0
    # Per-item outcomes, so two models can be compared as a *paired* test.
    # Without them the only available error bar is the unpaired one, which
    # overstates the error of a difference because every model answers the
    # same items -- and this project's success definition is written in
    # ~1-point margins against an unpaired sigma of +/-0.83.
    items, items_norm = [], []
    for ex in examples:
        scores = score_example(model, tokenizer, ex, device)
        hit = int(int(np.argmax(scores)) == ex.label)
        correct += hit
        items.append(hit)
        if has_norm:
            lengths = np.maximum(np.array(ex.choice_lengths, dtype=float), 1.0)
            hit_norm = int(int(np.argmax(np.array(scores) / lengths)) == ex.label)
            correct_norm += hit_norm
            items_norm.append(hit_norm)
    return {
        "n": n,
        "correct": correct,
        "acc": correct / n if n else float("nan"),
        "acc_norm": (correct_norm / n if n else float("nan")) if has_norm else None,
        # kept so existing callers/tests that read "accuracy" keep working
        "accuracy": correct / n if n else float("nan"),
        "items_acc": items,
        "items_acc_norm": items_norm if has_norm else None,
    }


def item_digest(examples: Sequence[ClozeExample]) -> str:
    """A fingerprint of *which* items were scored, in order.

    Pairing two runs item-by-item is only valid if they saw the same items in
    the same order. A `--task-limit` on one side, or a dataset revision that
    reorders rows, would otherwise pair item 7 against a different question
    entirely and produce a confident, meaningless McNemar result. Cheap to
    compute, and it turns that into a loud mismatch.
    """
    h = hashlib.sha256()
    for ex in examples:
        h.update(str(ex.label).encode())
        h.update(b"\x00")
        h.update(str(len(ex.candidates)).encode())
        h.update(b"\x00")
        h.update((ex.candidates[0][0] if ex.candidates else "").encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def per_item_record(task_examples: dict, results_by_task: dict) -> dict:
    """The sidecar written next to `--out`: outcomes plus their fingerprint.

    Kept out of the main results JSON on purpose -- 10,042 HellaSwag outcomes
    would bury the six numbers a human reads there.
    """
    out = {}
    for name, res in results_by_task.items():
        if res.get("items_acc") is None:
            continue
        out[name] = {
            "n": res["n"],
            "digest": item_digest(task_examples[name]),
            "headline": HEADLINE_METRIC.get(name, "acc"),
            "items_acc": res["items_acc"],
            "items_acc_norm": res.get("items_acc_norm"),
        }
    return out


# Published tables for the ~125-160M class quote acc_norm for the multiple-
# choice-over-varying-length-answers tasks and acc for WinoGrande (whose own
# harness config lists only acc). Selecting per task here keeps our headline
# numbers on the same footing as theirs.
HEADLINE_METRIC = {
    "hellaswag": "acc_norm",
    "arc_easy": "acc_norm",
    "piqa": "acc_norm",
    "openbookqa": "acc_norm",
    "winogrande": "acc",
}


def headline_metric(task: str, result: dict) -> float:
    """The metric a published table would quote for `task`."""
    key = HEADLINE_METRIC.get(task, "acc")
    value = result.get(key)
    return result["acc"] if value is None else value


def task_record(name: str, result: dict) -> dict:
    """One task's contribution to a results record.

    Shared by both scoring paths -- our checkpoints and HF peers -- because
    they had drifted: the peer path recorded `<task>_n` and the checkpoint
    path did not. `n` is what makes a peer comparison checkable. The peer
    table was built over full validation splits (hellaswag n=10042); scoring
    our own model over a `--task-limit` subset and printing the two side by
    side would look like a fair comparison and not be one, with nothing in
    either artifact to show it.

    `record[name]` is the headline number a published table would quote
    (acc_norm for the MC tasks, acc for WinoGrande); both raw metrics sit
    alongside it.
    """
    record = {name: headline_metric(name, result),
              f"{name}_acc": result["acc"],
              f"{name}_n": result["n"]}
    if result.get("acc_norm") is not None:
        record[f"{name}_acc_norm"] = result["acc_norm"]
    return record


# --------------------------------------------------------------- task data ---

def _load_with_fallback(candidates, split):
    """Try each (repo_id, config) candidate in turn, and for each, try the
    normal loading path then a parquet-conversion fallback -- `datasets` 5.x
    dropped script-based loading, which several of these older benchmark
    repos still use at their canonical name (e.g. plain 'piqa'). Mirrors
    AGENT.md's "if a dataset is gated or renamed, substitute the closest
    equivalent and continue" policy from the dataprep job.
    """
    from datasets import load_dataset
    last_err = None
    for repo, config in candidates:
        for revision in (None, "refs/convert/parquet"):
            try:
                kwargs = {"split": split}
                if revision:
                    kwargs["revision"] = revision
                ds = (load_dataset(repo, config, **kwargs) if config
                     else load_dataset(repo, **kwargs))
                return ds, repo
            except Exception as e:
                last_err = e
    raise RuntimeError(f"no candidate dataset could be loaded: {candidates}; "
                       f"last error: {last_err}")


def _example(context: str, choices: Sequence[str], label: int) -> ClozeExample:
    """Build a ClozeExample the way the harness builds its requests: one shared
    context, each choice joined by TARGET_DELIMITER, and the *unjoined* choice
    length kept for acc_norm."""
    return ClozeExample(
        candidates=[(context, TARGET_DELIMITER + c) for c in choices],
        label=label,
        choice_lengths=[len(c) for c in choices],
    )


def _hellaswag_preprocess(text: str) -> str:
    """lm_eval/tasks/hellaswag/utils.py::preprocess, verbatim -- the bracket
    artifacts are from HellaSwag's WikiHow portion and every published
    HellaSwag number is computed after this cleanup."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_hellaswag(split: str = "validation", limit: Optional[int] = None) -> List[ClozeExample]:
    ds, _ = _load_with_fallback([("Rowan/hellaswag", None), ("hellaswag", None)], split)
    out = []
    for row in ds:
        # harness process_docs: query = preprocess(activity_label + ": " +
        # ctx_a + " " + ctx_b.capitalize()) -- NOT the raw `ctx` column.
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        query = _hellaswag_preprocess(row["activity_label"] + ": " + ctx)
        out.append(_example(query,
                            [_hellaswag_preprocess(e) for e in row["endings"]],
                            int(row["label"])))
        if limit and len(out) >= limit:
            break
    return out


def load_arc_easy(split: str = "test", limit: Optional[int] = None) -> List[ClozeExample]:
    """`test`, not `validation` -- see TASK_SPLITS."""
    ds, _ = _load_with_fallback([("allenai/ai2_arc", "ARC-Easy")], split)
    # harness arc_easy.yaml: doc_to_text: "Question: {{question}}\nAnswer:"
    return _load_lettered_choice(
        ds, lambda r: f"Question: {r['question']}\nAnswer:", limit)


def load_openbookqa(split: str = "test", limit: Optional[int] = None) -> List[ClozeExample]:
    """`test`, not `validation` -- see TASK_SPLITS."""
    ds, _ = _load_with_fallback(
        [("allenai/openbookqa", "main"), ("openbookqa", "main")], split)
    # harness openbookqa.yaml: doc_to_text is the bare question_stem
    return _load_lettered_choice(ds, lambda r: r["question_stem"], limit)


def _load_lettered_choice(ds, context_fn, limit: Optional[int]) -> List[ClozeExample]:
    """Shared loader for ARC-Easy/OpenBookQA's {question, choices{text,label},
    answerKey} schema. `context_fn(row)` supplies the task's prompt template."""
    out = []
    for row in ds:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        answer_key = row["answerKey"].lstrip()   # harness: answerKey.lstrip()
        if answer_key not in labels:
            continue
        out.append(_example(context_fn(row), texts, labels.index(answer_key)))
        if limit and len(out) >= limit:
            break
    return out


def load_piqa(split: str = "validation", limit: Optional[int] = None) -> List[ClozeExample]:
    ds, _ = _load_with_fallback(
        [("ybisk/piqa", None), ("baber/piqa", None), ("piqa", None)], split)
    out = []
    for row in ds:
        # harness piqa.yaml: doc_to_text: "Question: {{goal}}\nAnswer:"
        out.append(_example(f"Question: {row['goal']}\nAnswer:",
                            [row["sol1"], row["sol2"]], int(row["label"])))
        if limit and len(out) >= limit:
            break
    return out


def load_winogrande(split: str = "validation", limit: Optional[int] = None) -> List[ClozeExample]:
    ds, _ = _load_with_fallback(
        [("allenai/winogrande", "winogrande_xl"), ("winogrande", "winogrande_xl")], split)
    out = []
    for row in ds:
        sentence = row["sentence"]
        if "_" not in sentence:
            continue
        # harness preprocess_winogrande.py: the choices are the *contexts*
        # (sentence up to the blank + the option) and the shared continuation
        # is the stripped text after the blank. No acc_norm for this task.
        idx = sentence.index("_")
        continuation = TARGET_DELIMITER + sentence[idx + 1:].strip()
        label = int(row["answer"]) - 1  # "1"/"2" -> 0/1
        out.append(ClozeExample(
            candidates=[(sentence[:idx] + row["option1"], continuation),
                        (sentence[:idx] + row["option2"], continuation)],
            label=label, choice_lengths=None))
        if limit and len(out) >= limit:
            break
    return out


TASK_LOADERS = {
    "hellaswag": load_hellaswag,
    "arc_easy": load_arc_easy,
    "piqa": load_piqa,
    "openbookqa": load_openbookqa,
    "winogrande": load_winogrande,
}

# Which split each task is scored on, and why it differs per task.
#
# lm-evaluation-harness scores a task's **test** split whenever the YAML
# declares one, falling back to validation otherwise. ARC-Easy and OpenBookQA
# declare `test_split: test` (arc/arc_easy.yaml, openbookqa/openbookqa.yaml);
# HellaSwag, PIQA and WinoGrande have held-out test labels, so their configs
# score validation.
#
# This is not cosmetic. Scoring ARC-Easy and OpenBookQA on `validation` -- which
# is what this file did originally -- measures a different, smaller set of
# examples (ARC-Easy: 570 validation vs 2,376 test) than every published number
# we are compared against. It showed up as exactly those two tasks disagreeing
# with the published Pythia-160M row by 3-5 points while the other three landed
# within 0.7, which is what sent us looking. `test` here is a correctness fix,
# not a preference.
TASK_SPLITS = {
    "hellaswag": "validation",
    "arc_easy": "test",
    "piqa": "validation",
    "openbookqa": "test",
    "winogrande": "validation",
}


def load_all_tasks(limit: Optional[int] = None) -> Dict[str, List[ClozeExample]]:
    """Loads every task, skipping (with a warning) any that fail entirely --
    a benchmark being temporarily unavailable must not crash the whole eval."""
    out = {}
    for name, loader in TASK_LOADERS.items():
        # Only override the split for tasks TASK_SPLITS knows about; the
        # loaders' own defaults already agree with it (pinned by
        # test_each_loader_defaults_to_its_harness_split), and key parity
        # between the two dicts is pinned by
        # test_task_splits_match_the_harness_yaml -- so a task added without a
        # split entry fails the suite rather than silently scoring the wrong
        # split here.
        kwargs = {"limit": limit}
        if name in TASK_SPLITS:
            kwargs["split"] = TASK_SPLITS[name]
        try:
            out[name] = loader(**kwargs)
        except Exception as e:
            print(f"WARNING: could not load task '{name}' ({e}); skipping")
    return out


# --------------------------------------------------------------------- bpb ---

def evaluate_bpb(model, shard_dir: str, seq_len: int, tokenizer, device: str = "cpu",
                 batch_size: int = 8, max_batches: Optional[int] = None) -> float:
    """Held-out bits-per-byte over packed shards (see daedalus/data.py)."""
    from daedalus.data import make_loader
    from train import bits_per_byte

    import time

    # stride=seq_len: non-overlapping windows, so a full pass (max_batches=None)
    # visits each held-out token once instead of every sliding-window offset
    # (~seq_len x more "batches" than the shard actually contains).
    loader = make_loader(shard_dir, seq_len, batch_size, shuffle=False, num_workers=0,
                         stride=seq_len)
    was_training = model.training
    model.eval()
    # The model's training loss is CE + z_loss * mean(logsumexp^2). z-loss is a
    # stability regulariser, not part of the likelihood -- leaving it in would
    # inflate reported bits-per-byte by a fraction of a percent and make our
    # numbers incomparable to any published bpb. Zeroed for the duration of the
    # measurement and restored afterwards, like `model.eval()` above.
    saved_z_loss = model.cfg.z_loss
    model.cfg.z_loss = 0.0
    total_nats, total_tokens, total_bytes = 0.0, 0, 0
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    log_every = max(1, n_batches // 20)  # ~20 progress lines regardless of scale
    t0 = time.monotonic()
    try:
        with torch.no_grad(), torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            for i, xb in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                if i > 0 and i % log_every == 0:
                    elapsed = time.monotonic() - t0
                    eta = elapsed / i * (n_batches - i)
                    print(f"  val_bpb progress: {i}/{n_batches} batches "
                          f"({100 * i / n_batches:.0f}%), elapsed {elapsed / 60:.1f}m, "
                          f"eta {eta / 60:.1f}m", flush=True)
                xb = xb.to(device)
                _, loss, _ = model(xb, targets=xb)
                # model.py predicts T-1 positions per row (its internal shift, see
                # smoke.py's net(x, targets=x)); byte count uses the full decoded
                # row as an approximation (off by one token's bytes per row).
                n_pred = xb.numel() - xb.size(0)
                total_nats += loss.item() * n_pred
                total_tokens += n_pred
                for row in xb:
                    total_bytes += len(tokenizer.decode(row.tolist()).encode("utf-8"))
    finally:
        # cfg is a shared PRESETS instance -- restore it even on an exception,
        # or a failed eval silently disables z-loss for every later caller.
        model.cfg.z_loss = saved_z_loss
        model.train(was_training)
    if total_tokens == 0:
        return float("nan")
    avg_nats_per_token = total_nats / total_tokens
    return bits_per_byte(avg_nats_per_token, total_tokens, total_bytes)


def evaluate_bpb_mixture(model, root: str, seq_len: int, tokenizer,
                         device: str = "cpu", batch_size: int = 8,
                         max_batches: Optional[int] = None,
                         weights: Optional[Dict[str, float]] = None) -> dict:
    """Held-out BPB over *either* a single shard dir or a `dataprep` mixture
    root (one subdirectory per source, each with its own manifest.json).

    `evaluate_bpb` handles only the single-dir case -- `make_loader` opens
    `<dir>/manifest.json`, which a mixture root does not have. That mattered:
    `hero.py` carves a per-source holdout with `make_mixture_holdout_split` and
    passes the *root* as `train.py --val-dir`, where `Trainer._val_bpb`
    swallows every exception by design. The four-day run would have logged
    `val_bpb: null` at every eval interval behind a WARNING line -- no
    validation curve at all, and nothing for the watchdog to act on.

    `weights` maps source name -> the probability training actually draws it
    with (`MixtureBatchSource.probs`, i.e. blueprint mixture shares after the
    epoch cap). With it, val_bpb estimates BPB under the training
    distribution. Without it, sources fall back to their holdout token counts
    -- which is *not* the mixture: `make_holdout_split` reserves whole shard
    files, so a source's holdout share is set by the arbitrary size of its
    trailing partial shard. Measured on the real 9-source corpus, that weights
    stack-edu-python at 1.71x its training share and fineweb-edu at 0.65x.
    Missing/extra names are ignored and the rest renormalized, matching
    `MixtureBatchSource`'s tolerance for a partially-built mixture.
    """
    if os.path.exists(os.path.join(root, "manifest.json")):
        bpb = evaluate_bpb(model, root, seq_len, tokenizer, device,
                           batch_size=batch_size, max_batches=max_batches)
        return {"val_bpb": bpb, "per_source_val_bpb": {}}

    sources = sorted(
        name for name in os.listdir(root)
        if os.path.exists(os.path.join(root, name, "manifest.json")))
    if not sources:
        raise ValueError(f"no source under {root!r} has a manifest.json")

    per_source: Dict[str, dict] = {}
    for name in sources:
        source_dir = os.path.join(root, name)
        with open(os.path.join(source_dir, "manifest.json")) as f:
            n_tokens = int(json.load(f)["total_tokens"])
        bpb = evaluate_bpb(model, source_dir, seq_len, tokenizer, device,
                           batch_size=batch_size, max_batches=max_batches)
        per_source[name] = {"val_bpb": bpb, "tokens": n_tokens}

    if weights:
        raw = {n: weights[n] for n in sources if n in weights}
        if not raw:
            raise ValueError(
                f"none of the sources under {root!r} ({sources}) appear in "
                f"`weights` ({sorted(weights)}); refusing to guess")
        dropped = sorted(set(sources) - set(raw))
        if dropped:
            print(f"evaluate_bpb_mixture: {dropped} carry no training weight; "
                  "excluded from val_bpb")
    else:
        raw = {n: float(per_source[n]["tokens"]) for n in sources}

    total_weight = sum(raw.values())
    weighted_sum = 0.0
    for name, w in raw.items():
        share = w / total_weight if total_weight else 0.0
        per_source[name]["weight"] = share
        weighted_sum += per_source[name]["val_bpb"] * share
    for name in sources:
        per_source[name].setdefault("weight", 0.0)

    val_bpb = weighted_sum if total_weight else float("nan")
    return {"val_bpb": val_bpb, "per_source_val_bpb": per_source}


# ------------------------------------------------------------- peer models ---

class HFCausalLMAdapter(torch.nn.Module):
    """Wraps a HuggingFace causal LM in `Daedalus.forward`'s signature so the
    *same* cloze scorer measures peers and Daedalus.

    Why this exists: the whole project is judged on beating published ~125-160M
    models, and quoting their numbers out of a paper compares two different
    harnesses as much as two different models -- prompt templates, the
    `acc_norm` denominator, and whether a leading space lands on the context or
    the continuation each move these scores by points. Re-measuring the peers
    through this file's own `score_text`/`evaluate_cloze_task` removes that
    confound: whatever this harness gets wrong, it gets wrong for everyone
    equally, so the *comparison* stays valid even where an absolute number
    drifts from a published table.

    Each peer is scored with **its own** tokenizer (returned alongside the
    model), as the harness does -- scoring GPT-2 through SmolLM2's vocabulary
    would measure a tokenizer mismatch, not a model.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, input_ids, targets=None, return_logits: bool = True):
        # Daedalus returns (logits, loss, aux); only logits are used by
        # score_ids, and only when return_logits is True.
        out = self.hf(input_ids=input_ids)
        return out.logits, None, None


def load_peer_model(name: str, device: str = "cpu", dtype=torch.float32):
    """Load a published peer model + its own tokenizer, ready for `score_text`.

    fp32 by default, deliberately. bf16 looked like a free saving and is not:
    measured on pythia-160m, bf16 moved ARC-Easy `acc` by +0.7 and `acc_norm` by
    +1.2 points against fp32, and PIQA by -0.5. Those are the same magnitude as
    the gaps between the peers in README's bar, so a numeric-precision choice
    would have been indistinguishable from a real quality difference. These
    models are ~125-160M parameters; fp32 costs seconds here.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype if device.startswith("cuda") else torch.float32)
    model = HFCausalLMAdapter(hf).to(device)
    model.eval()
    return model, tok


# ---------------------------------------------------------- checkpoint eval ---

def evaluate_checkpoint(ckpt_path: str, cfg_name: str, tokenizer,
                        task_examples: Dict[str, List[ClozeExample]],
                        shard_dir: Optional[str] = None, seq_len: int = 2048,
                        device: str = "cpu", bpb_max_batches: Optional[int] = 100,
                        bpb_batch_size: int = 8,
                        raw_out: Optional[dict] = None,
                        bpb_weights: Optional[Dict[str, float]] = None) -> dict:
    """bpb_max_batches defaults to a bounded sample (100 batches), not the
    whole held-out shard set -- a full-shard-directory pass here would make
    periodic during-training eval (every checkpoint) prohibitively slow (a
    single fineweb-edu-scale shard is tens of thousands of seq_len windows).
    Pass None to score every window for a final, thorough report instead.
    """
    from train import load_checkpoint

    cfg = PRESETS[cfg_name]
    model = Daedalus(cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    results = {}
    if shard_dir:
        # `evaluate_bpb_mixture`, not `evaluate_bpb`. `hero.py` carves a
        # per-source holdout (`make_mixture_holdout_split`) whose *root* has no
        # manifest.json, and the single-dir function raises FileNotFoundError
        # there. That fired before the task loop below, so the process died
        # with `--out` never written: the after-run chain's eval step produced
        # no val_bpb *and* no 5-task mean -- the two numbers the project is
        # judged on -- after a ~6-day run. The mixture function handles the
        # single-dir case too (it checks for a top-level manifest.json first),
        # so single-source callers are unaffected.
        #
        # Kept as a float under `val_bpb` so W&B and `mean_over_checkpoints`
        # both see it; the per-source breakdown rides alongside.
        bpb = evaluate_bpb_mixture(model, shard_dir, seq_len, tokenizer, device,
                                   batch_size=bpb_batch_size,
                                   max_batches=bpb_max_batches,
                                   weights=bpb_weights)
        results["val_bpb"] = bpb["val_bpb"]
        results["per_source_val_bpb"] = bpb["per_source_val_bpb"]
    for name, examples in task_examples.items():
        res = evaluate_cloze_task(model, tokenizer, examples, device)
        # The caller keeps the raw result only to write the per-item sidecar;
        # `task_record` still decides what lands in the results JSON, so the
        # two scoring paths cannot drift apart again.
        if raw_out is not None:
            raw_out[name] = res
        results.update(task_record(name, res))
    return results


def mean_over_checkpoints(per_checkpoint_results: List[dict]) -> dict:
    """Average each *numeric* metric across checkpoints, ignoring missing/NaN
    entries and any non-numeric fields (e.g. a "checkpoint" path string the
    caller stashed alongside the metrics) (AGENT.md: report mean over the
    last 3-5 checkpoints)."""
    if not per_checkpoint_results:
        return {}
    keys = set()
    for r in per_checkpoint_results:
        keys |= set(r.keys())
    mean = {}
    for k in keys:
        vals = [r[k] for r in per_checkpoint_results
                if isinstance(r.get(k), (int, float)) and not (isinstance(r[k], float) and math.isnan(r[k]))]
        if not vals:
            continue  # every value for k was missing/NaN/non-numeric -- nothing to report
        mean[k] = sum(vals) / len(vals)
    return mean


# --------------------------------------------------------------------- cli ---

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="daedalus-150m")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="last 3-5 checkpoint paths to average over")
    p.add_argument("--hf-model", default=None,
                   help="score a published HuggingFace causal LM instead of a "
                        "Daedalus checkpoint (e.g. EleutherAI/pythia-160m), "
                        "through this same harness -- see HFCausalLMAdapter for "
                        "why peers are re-measured rather than quoted. Uses the "
                        "peer's own tokenizer.")
    p.add_argument("--shard-dir", default=None, help="held-out packed shards for val_bpb")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--bpb-max-batches", type=int, default=100,
                   help="bound val_bpb to this many batches (pass -1 for the "
                        "full held-out shard set -- slow, use for a final report)")
    p.add_argument("--bpb-batch-size", type=int, default=8)
    p.add_argument("--mixture-weights-from", default=None,
                   help="training shard root. Weights val_bpb by the per-source "
                        "probabilities the sampler actually draws with, which is "
                        "how train.py computes the val_bpb curve -- without it "
                        "sources are weighted by holdout token counts, which is "
                        "not the training mixture (fineweb-edu lands at 0.65x "
                        "its share) and the final number would not be "
                        "comparable to the curve it continues")
    p.add_argument("--mixture-total-tokens", type=float, default=None,
                   help="the run's token budget, so the epoch cap is applied "
                        "the same way training applied it")
    p.add_argument("--task-limit", type=int, default=None,
                   help="examples per task. Default (unset, or 0) is the full "
                        "validation split, which is what a published-table "
                        "comparison needs and how runs/eval/peer-*.json were "
                        "produced -- a 500-example subset carries about +/-2 "
                        "points of sampling noise, wider than the gaps between "
                        "the peers we are measured against. Set it only for a "
                        "smoke run.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/eval/results.json")
    p.add_argument("--per-item", default=None,
                   help="write per-item correctness here (default: alongside "
                        "--out as <out stem>.items.json). Set empty to skip.")
    p.add_argument("--run-name", default=None,
                   help="W&B run name; defaults to eval-<config>-<git sha>")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    bpb_max_batches = None if args.bpb_max_batches < 0 else args.bpb_max_batches
    if bool(args.checkpoints) == bool(args.hf_model):
        p.error("pass exactly one of --checkpoints or --hf-model")
    # Derived rather than required, deliberately: the runs that most need a
    # paired comparison are the unattended ones (tonight's 5B arm-1 eval fires
    # from a waiter script), and a sidecar that only appears when someone
    # remembers a flag is a sidecar that will not exist when the claim is made.
    # `--per-item ""` opts out.
    if args.per_item is None:
        args.per_item = re.sub(r"\.json$", "", args.out) + ".items.json"

    import os
    from daedalus.wandb_logger import WandbLogger

    run_name = args.run_name or (
        f"eval-peer-{args.hf_model.split('/')[-1]}-{_git_short_sha()}"
        if args.hf_model else f"eval-{args.config}-{_git_short_sha()}")
    wb = WandbLogger(
        project=args.wandb_project or os.environ.get("WANDB_PROJECT", "daedalus"),
        entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        name=run_name, config={"eval_args": vars(args)},
        tags=["eval", *(["peer"] if args.hf_model else [])],
        enabled=not args.no_wandb,
    )

    task_examples = None   # loaded after the tokenizer, below

    per_ckpt = []
    per_item: Dict[str, dict] = {}
    if args.hf_model:
        # A peer is scored with its own tokenizer, and has no Daedalus-format
        # shards to take bits-per-byte over -- only the cloze tasks, which are
        # the comparable half anyway.
        model, tokenizer = load_peer_model(args.hf_model, device=args.device)
        task_examples = load_all_tasks(limit=args.task_limit)
        r = {}
        raw = {}
        for name, examples in task_examples.items():
            res = evaluate_cloze_task(model, tokenizer, examples, args.device)
            raw[name] = res
            r.update(task_record(name, res))
            print(f"  {name}: {r[name]:.4f} (n={res['n']})")
        per_item[args.hf_model] = per_item_record(task_examples, raw)
        r["checkpoint"] = args.hf_model
        per_ckpt.append(r)
        wb.log({k: v for k, v in r.items() if isinstance(v, (int, float))}, step=0)
        print(json.dumps(r, indent=2))

    # Local import: `abl_arch` is a job script, and `train.py` imports this
    # module lazily at every val interval -- it must not pay for it.
    bpb_weights = None
    if args.mixture_weights_from:
        from abl_arch import mixture_sampling_weights
        bpb_weights = mixture_sampling_weights(
            args.mixture_weights_from,
            int(args.mixture_total_tokens) if args.mixture_total_tokens else None)
        print(f"val_bpb weighted by the training sampler over "
              f"{args.mixture_weights_from}: "
              f"{ {k: round(v, 4) for k, v in sorted(bpb_weights.items())} }")

    from daedalus.data import get_tokenizer
    for i, ckpt in enumerate(args.checkpoints or []):
        if task_examples is None:
            tokenizer = get_tokenizer()
            task_examples = load_all_tasks(limit=args.task_limit)
        print(f"evaluating {ckpt} ...")
        raw = {}
        r = evaluate_checkpoint(ckpt, args.config, tokenizer, task_examples,
                                shard_dir=args.shard_dir, seq_len=args.seq_len,
                                device=args.device, bpb_max_batches=bpb_max_batches,
                                bpb_batch_size=args.bpb_batch_size, raw_out=raw,
                                bpb_weights=bpb_weights)
        per_item[ckpt] = per_item_record(task_examples, raw)
        r["checkpoint"] = ckpt
        print(json.dumps(r, indent=2))
        per_ckpt.append(r)
        wb.log({k: v for k, v in r.items() if isinstance(v, (int, float))}, step=i)

    mean = mean_over_checkpoints(per_ckpt)
    wb.log({f"mean_{k}": v for k, v in mean.items()})
    wb.finish()

    out = {"per_checkpoint": per_ckpt, "mean": mean}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    # Per-item outcomes go beside the results, not inside them: 10,042
    # HellaSwag rows would bury the six numbers a human reads there. They are
    # what makes a *paired* comparison possible, which is the difference
    # between "we beat Pythia-160M" being a result and being a coin flip --
    # the peer group sits inside a 1.1-point band and the unpaired error of a
    # difference is +/-0.83.
    if per_item and args.per_item:
        items_path = args.per_item
        os.makedirs(os.path.dirname(items_path) or ".", exist_ok=True)
        with open(items_path, "w") as f:
            json.dump({"models": per_item, "task_limit": args.task_limit}, f)
        print(f"wrote {items_path}")

    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    _cli()
