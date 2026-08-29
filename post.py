"""`post`: SFT on smol-smoltalk with ChatML (AGENT.md SS4, plan step 10).

Runs after `hero`, taking its checkpoint and producing an instruction-following
model: SFT on smol-smoltalk, then (with `--dpo`) one light DPO round on
UltraFeedback. The final GGUF export is export.py's job, not this file's.

Design:
  - **`--init-from`, not `--resume`.** Hero's checkpoint supplies weights and
    nothing else: SFT starts at step 0 with fresh optimizers and its own WSD
    schedule. Handing it to train.py as `resume` -- which this file did until
    the end-to-end smoke below -- also restores step=610000 and
    tokens_seen=40e9, so `fit()` breaks at the top of its first iteration and
    the entire SFT stage does nothing while printing a success line. `--resume`
    is still accepted, and means what it means everywhere else: continue an
    interrupted *post* run. It wins over `--init-from` when it exists, so a
    supervisor can relaunch one command line either way.
  - **Reuses train.py's Trainer.** SFT differs from pretraining only in where
    the batches come from and which positions are supervised, so the
    optimizers, WSD schedule, gradient accumulation, checkpointing, resume,
    metrics and W&B all come from the same code that ran `hero`. The Trainer
    accepts either `x` or `(x, y)` from a batch source; SFTBatchSource yields
    the latter, with -100 on prompt and pad positions.
  - **Streams the dataset.** smol-smoltalk is ~1M conversations. Encoding
    them all up front would hold hundreds of MB to GB of Python ints --
    exactly what ADDENDUM 2 rule 2 forbids on a box whose RAM is scarcer than
    its VRAM. Examples are pulled, filtered and encoded on demand through a
    bounded shuffle buffer, so peak memory is set by the buffer, not by the
    corpus.
  - **No seq ramp.** post.py pins seq_start == seq_end; the 1024->2048 ramp is
    a pretraining device and a moving seq_len would repack every batch.

Formatting comes from daedalus/chatml.py and must not be duplicated here: a
mismatch between the SFT format and the inference format is invisible at
training time and shows up only as a model that ignores its prompt.
"""
import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import torch

from daedalus.chatml import (
    IGNORE_INDEX,
    encode_sft_example,
    keep_example,
    pad_batch,
)
from daedalus.dpo import freeze_reference

DEFAULT_SFT_DATASET = "HuggingFaceTB/smol-smoltalk"


def iter_chat_examples(dataset: Iterable, max_assistant_chars: int,
                       drop_cot: bool) -> Iterator[List[dict]]:
    """Yield message lists that pass AGENT.md SS4's content filters.

    Accepts any iterable of rows carrying a `messages` field, so tests can
    pass a plain list and the CLI can pass a streaming HF dataset.
    """
    for row in dataset:
        messages = row.get("messages") if isinstance(row, dict) else None
        if not messages:
            continue
        messages = [{"role": m["role"], "content": m["content"]}
                    for m in messages]
        if keep_example(messages, max_assistant_chars=max_assistant_chars,
                        drop_cot=drop_cot):
            yield messages


def iter_encoded(examples: Iterable[List[dict]], tokenizer, max_len: int
                 ) -> Iterator[Tuple[List[int], List[int]]]:
    """Encode lazily, dropping anything that does not fit (chatml.py returns
    None rather than truncating, since a truncated example has no <|im_end|>
    and teaches the model to run past its stop token)."""
    for messages in examples:
        encoded = encode_sft_example(messages, tokenizer, max_len=max_len)
        if encoded is not None:
            yield encoded


def shuffle_buffered(it: Iterable, buffer_size: int, seed: int = 0) -> Iterator:
    """Reservoir-style local shuffle over a stream.

    A streaming dataset arrives in whatever order it was written; smol-smoltalk
    is grouped by source, so consecutive batches would otherwise be
    single-domain and the optimizer would see a curriculum nobody designed.
    Bounded on purpose -- `buffer_size` examples, not the corpus.
    """
    rng = random.Random(seed)
    buffer = []
    for item in it:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        i = rng.randrange(buffer_size)
        yield buffer[i]
        buffer[i] = item
    rng.shuffle(buffer)
    yield from buffer


class SFTBatchSource:
    """Padded, label-masked batches for train.py's Trainer.

    `get_batch` ignores its `seq_len` argument: SFT batches are ragged and
    padded to their own longest member, and post.py pins the Trainer's seq
    ramp flat so nothing else varies it either. Returning `(x, y)` is what
    tells train_step to use masked labels.
    """

    def __init__(self, examples: Iterable[Tuple[List[int], List[int]]],
                 micro_batch: int, device: str, pad_id: int,
                 loop: bool = True):
        self._make_iter = lambda: iter(examples)
        self._it = self._make_iter()
        self.micro_batch = micro_batch
        self.device = device
        self.pad_id = pad_id
        self.loop = loop
        self.epochs = 0
        self.examples_seen = 0
        self.supervised_tokens = 0
        self.padded_tokens = 0

    def _next_example(self):
        try:
            return next(self._it)
        except StopIteration:
            if not self.loop:
                raise
            # A materialised sequence can be re-iterated; a one-shot generator
            # cannot, and silently training on a shorter run than requested is
            # worse than saying so.
            self.epochs += 1
            self._it = self._make_iter()
            try:
                return next(self._it)
            except StopIteration:
                raise RuntimeError(
                    "SFT example source is exhausted and cannot be re-iterated. "
                    "Pass a materialised sequence (e.g. a list) if you need "
                    "more steps than it has examples.")

    def get_batch(self, seq_len: int):
        batch = [self._next_example() for _ in range(self.micro_batch)]
        self.examples_seen += len(batch)
        ids, labels = pad_batch(batch, pad_id=self.pad_id)
        x = torch.tensor(ids, dtype=torch.long, device=self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        self.supervised_tokens += int((y != IGNORE_INDEX).sum())
        self.padded_tokens += int(y.numel())
        return x, y


def build_sft_source(dataset, tokenizer, micro_batch: int, device: str,
                     max_len: int = 2048, max_assistant_chars: int = 1200,
                     drop_cot: bool = True, shuffle_buffer: int = 10_000,
                     limit: Optional[int] = None, seed: int = 0,
                     materialise: bool = False) -> SFTBatchSource:
    """Wire the stream: filter -> encode -> local shuffle -> batches."""
    chats = iter_chat_examples(dataset, max_assistant_chars, drop_cot)
    if limit is not None:
        chats = itertools.islice(chats, limit)
    encoded = iter_encoded(chats, tokenizer, max_len)
    stream = shuffle_buffered(encoded, shuffle_buffer, seed=seed)
    examples = list(stream) if materialise else stream
    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return SFTBatchSource(examples, micro_batch, device, pad_id=pad_id,
                          loop=materialise)


# ------------------------------------------------------- a locally built set ---
# Phase 8 step 6 trains on conversations that were selected on disk rather than
# streamed: `daedalus/code_sft.py` runs each one through an admission gate that
# syntax-checks its code, executes the tests that ship with it, filters it
# against both decontamination indexes and measures it in supervised tokens,
# then writes what it admits as `train.jsonl` beside a manifest of those counts.
#
# The rows it writes are already `iter_chat_examples`' shape, which makes
# `--dataset <that file>` look like the whole of the wiring. It is the one thing
# that must not happen. That path re-applies `chatml.keep_example` at its
# default 1,200-character cap over the *whole* assistant turn, while the build
# applies the same cap to assistant **prose** with fenced code removed -- so a
# forty-line function, the payload this phase exists to add, passes the build
# and is dropped by the reader. Nothing would say so: the set would simply come
# back smaller, reading exactly like a stream that ran short.
#
# So a built set gets a reader of its own, and it applies no content filter at
# all. Everything below is about proving the file on disk is the set its
# manifest describes, since that is the assumption the missing filter rests on.

BUILT_MANIFEST_FILE = "manifest.json"


class BuiltSFTSetError(RuntimeError):
    """Raised when a built SFT set is not the set its manifest describes."""


def read_built_manifest(directory, *, max_len: Optional[int] = None,
                        tokenizer=None) -> dict:
    """The manifest of a built set, or a refusal naming what is wrong with it.

    Four refusals, each for a way of training on something other than what the
    manifest describes, and none of them visible in a loss curve.
    """

    path = Path(directory) / BUILT_MANIFEST_FILE
    try:
        with path.open() as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise BuiltSFTSetError(
            f"{path} does not exist. A built set is a directory carrying its "
            f"manifest beside its files: the manifest is the only record of "
            f"which gate admitted these conversations, which tokenizer measured "
            f"them and at what budget, and this reader applies no filter of its "
            f"own because that gate already ran. A bare .jsonl could have come "
            f"from anywhere.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BuiltSFTSetError(f"{path} is not readable as JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BuiltSFTSetError(f"{path} does not decode to an object")
    missing = [key for key in ("problems", "files") if key not in manifest]
    if missing:
        raise BuiltSFTSetError(
            f"{path} is not a built-set manifest: no {', '.join(missing)} "
            f"block, so nothing here can tell whether the build finished or "
            f"what it thinks of what it wrote.")

    problems = manifest.get("problems") or []
    if problems:
        # The build's own verdict on what it wrote. It is reported rather than
        # raised at build time so a short half still lands on disk to be looked
        # at -- which makes training on it the mistake this refusal covers.
        raise BuiltSFTSetError(
            f"{path} reports {len(problems)} problem(s) with the set it "
            f"describes: " + "; ".join(str(problem) for problem in problems) +
            ". The build is saying this is not the corpus its own counts claim, "
            "so a run over it trains a mixture nobody chose.")

    built_max_len = manifest.get("max_len")
    if not isinstance(built_max_len, int) or built_max_len <= 0:
        raise BuiltSFTSetError(
            f"{path} records no token budget, so there is no way to tell "
            f"whether this run's --max-len would silently drop what the build "
            f"admitted.")
    if max_len is not None and max_len < built_max_len:
        raise BuiltSFTSetError(
            f"this set was admitted at max_len {built_max_len:,} and would be "
            f"trained at {max_len:,}. `encode_sft_example` returns None rather "
            f"than truncating and the encoder drops what it returns, so every "
            f"conversation between the two budgets would leave the set without "
            f"a count or a reason -- and the long ones are the code payload the "
            f"set exists to add. A larger budget is fine: it admits everything "
            f"the build admitted.")

    if tokenizer is not None:
        built_tokenizer = manifest.get("tokenizer") or {}
        built_vocab = built_tokenizer.get("vocab_size")
        vocab = getattr(tokenizer, "vocab_size", None)
        if built_vocab is not None and vocab is not None \
                and int(built_vocab) != int(vocab):
            raise BuiltSFTSetError(
                f"this set was measured with a {int(built_vocab):,}-token "
                f"tokenizer ({built_tokenizer.get('name_or_path')!r}) and would "
                f"be trained with a {int(vocab):,}-token one "
                f"({getattr(tokenizer, 'name_or_path', None)!r}). Its supervised-"
                f"token counts, its half shares and its max_len refusals are all "
                f"measurements of the first; re-tokenized they describe a corpus "
                f"that was never built.")
    return manifest


def built_train_path(directory, manifest) -> Path:
    """The training file this manifest names, or a refusal.

    Read out of the manifest rather than composed from a constant: the file the
    manifest's counts describe is the one it names, and a reader that assumes
    the name would happily train on a leftover from an earlier build.
    """

    files = manifest.get("files")
    name = files.get("train") if isinstance(files, dict) else None
    if not name:
        raise BuiltSFTSetError(
            f"the manifest in {directory} names no training file, so its counts "
            f"describe nothing this can open")
    path = Path(directory) / str(name)
    if not path.exists():
        raise BuiltSFTSetError(
            f"{path} does not exist, though the manifest beside it says it does. "
            f"A build publishes its files and its manifest together, so this is "
            f"a set that was moved or half-copied rather than one that was built.")
    return path


def iter_built_examples(path) -> Iterator[List[dict]]:
    """Message lists from a built set, with **no** content filter applied.

    Deliberately not `iter_chat_examples`: see the note above this section. The
    conversations in this file passed a strictly stronger gate than the one that
    function applies, and re-applying it drops the code it was built to carry.

    A malformed line raises rather than being skipped, which is the opposite of
    the streaming reader's stance and right for the same reason: the producer of
    this file is known, so a line it cannot have written means the file is not
    the built set, and skipping would train on an unrecorded subset of it.
    """

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuiltSFTSetError(f"{path} line {number} is not JSON: {exc}")
            messages = row.get("messages") if isinstance(row, dict) else None
            if not messages:
                raise BuiltSFTSetError(
                    f"{path} line {number} carries no messages. Every line a "
                    f"build writes is a conversation; a line that is not means "
                    f"this file is not the set the manifest describes.")
            try:
                yield [{"role": message["role"], "content": message["content"]}
                       for message in messages]
            except (KeyError, TypeError) as exc:
                raise BuiltSFTSetError(
                    f"{path} line {number} has a message without a role or a "
                    f"content: {exc}")


def encode_built(examples: Iterable[List[dict]], tokenizer, max_len: int
                 ) -> Iterator[Tuple[List[int], List[int]]]:
    """Encode a built set, refusing rather than dropping what does not fit.

    `iter_encoded` drops silently, which is right for a stream whose rows nobody
    vetted. Here the build already refused `over_token_budget` with this same
    encoder, so a conversation that does not fit means the set and the tokenizer
    disagree with the manifest that was just checked -- and the honest answer to
    that is to stop rather than to train on the remainder. The shuffle buffer
    fills before the first batch is yielded, so this surfaces at startup rather
    than hours in.
    """

    for number, messages in enumerate(examples, start=1):
        encoded = encode_sft_example(messages, tokenizer, max_len=max_len)
        if encoded is None:
            raise BuiltSFTSetError(
                f"conversation {number} of the built set does not fit "
                f"{max_len:,} tokens, though the build admitted it at its own "
                f"budget with this same encoder. The set, the tokenizer and the "
                f"manifest do not describe the same corpus.")
        yield encoded


class _Reiterable:
    """A source `SFTBatchSource` can start over, without holding the corpus.

    `iter()` on a generator hands back the same exhausted generator, which is
    why the streaming path can only loop when it materialises. A built set is a
    file and can simply be read again, so this rebuilds the whole
    read -> encode -> shuffle pipeline per epoch and peak memory stays the
    shuffle buffer rather than the set.

    The shuffle seed is fixed, so every epoch sees the same order -- which is
    what looping a materialised list already does today.
    """

    def __init__(self, factory):
        self._factory = factory

    def __iter__(self):
        return iter(self._factory())


def build_built_sft_source(directory, tokenizer, micro_batch: int, device: str,
                           max_len: int = 2048, shuffle_buffer: int = 10_000,
                           limit: Optional[int] = None, seed: int = 0
                           ) -> Tuple[SFTBatchSource, dict]:
    """Wire a built set: read -> encode -> local shuffle -> batches.

    Returns the source and the manifest it was checked against, because the
    manifest is what the run has to report -- the supervised-token counts and
    the code-language shares step 6 is asked to track are measurements of this
    file, not of the batches that come out of it.
    """

    manifest = read_built_manifest(directory, max_len=max_len,
                                   tokenizer=tokenizer)
    train_path = built_train_path(directory, manifest)

    def pipeline():
        chats = iter_built_examples(train_path)
        if limit is not None:
            chats = itertools.islice(chats, limit)
        return shuffle_buffered(encode_built(chats, tokenizer, max_len),
                                shuffle_buffer, seed=seed)

    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return SFTBatchSource(_Reiterable(pipeline), micro_batch, device,
                          pad_id=pad_id, loop=True), manifest


def refuse_built_set_as_dataset(dataset: Optional[str]) -> None:
    """Refuse a built set handed to `--dataset`.

    The trap this whole section exists for, caught at the one place it is
    reachable. A built row is already `iter_chat_examples`' shape, so the wrong
    flag trains without complaint on most of the general half and a fraction of
    the code half.
    """

    if not dataset:
        return
    path = Path(dataset)
    looks_built = (path / BUILT_MANIFEST_FILE).exists() or \
        (path.suffix == ".jsonl" and path.exists())
    if looks_built:
        raise SystemExit(
            f"--dataset {dataset} looks like a locally built SFT set. It would "
            f"be filtered a second time on the way in -- `keep_example` at the "
            f"whole-turn character cap, over conversations a build already "
            f"admitted by measuring their prose alone -- and most of the code "
            f"would be dropped without a count. Use --built-dataset, which "
            f"applies no filter and checks the manifest instead.")


DEFAULT_DPO_DATASET = "HuggingFaceH4/ultrafeedback_binarized"


def iter_preference_pairs(dataset, tokenizer, max_len: int,
                          max_assistant_chars: int) -> "Iterator[dict]":
    """Yield `{chosen: (ids, labels), rejected: (ids, labels)}` per row.

    UltraFeedback rows carry `chosen` and `rejected` as full message lists
    sharing a prompt. Both sides go through the same chatml encoder as SFT, so
    only the response tokens are scored and the shared prompt cancels out of
    the margin.

    A pair is dropped unless *both* sides encode: scoring a pair where one
    side was silently truncated compares a complete response against a
    fragment, and DPO would learn that fragments are preferable.
    """
    for row in dataset:
        chosen, rejected = row.get("chosen"), row.get("rejected")
        if not chosen or not rejected:
            continue
        pair = {}
        for name, messages in (("chosen", chosen), ("rejected", rejected)):
            msgs = [{"role": m["role"], "content": m["content"]}
                    for m in messages]
            # No CoT filter here: UltraFeedback's value is the preference, and
            # dropping a pair because the *rejected* side rambles would bias
            # the comparison toward easy pairs.
            if not keep_example(msgs, max_assistant_chars=max_assistant_chars,
                                drop_cot=False):
                break
            encoded = encode_sft_example(msgs, tokenizer, max_len=max_len)
            if encoded is None:
                break
            pair[name] = encoded
        if len(pair) == 2:
            yield pair


def take_eval_pairs(pairs: Iterable[dict], n: int) -> Tuple[List[dict], Iterator[dict]]:
    """Split a preference stream into a held-out head and the training tail.

    Returns `(held_out, remaining)`. Disjoint *by construction*: both come from
    one iterator, so a pair handed to the evaluation set is one `run_dpo` can
    never reach. Holding out by any other route -- a second `load_dataset` call,
    a `--limit`, a seed -- relies on two streams agreeing about order, and when
    they quietly disagree the gate is scored on pairs the model trained on and
    reports a number that only says the round memorised its own data.

    The head is not a random sample: a streaming dataset arrives in file order.
    That is fine for a delta measured on the same pairs before and after, which
    is all the gate asks, but it is why `--dpo-eval-split` exists for datasets
    that ship a real held-out split.
    """
    it = iter(pairs)
    return (list(itertools.islice(it, n)) if n > 0 else []), it


def evaluate_preference_gate(reference, policy, eval_pairs, *, pad_id: int,
                             micro_batch: int = 2, device: str = "cpu") -> dict:
    """Held-out preference accuracy before and after the DPO round.

    `reference` is the before-model, and it costs nothing to say so late: it is
    a frozen snapshot of the policy taken before the first DPO step and it never
    moves, so scoring it after the round returns the pre-DPO answer. Measuring
    "before" up front instead would mean a second forward pass over the same
    pairs and one more chance to score the two models on different data.

    `accuracy_improved` is *half* of phase 8's rule for keeping the DPO model.
    The other half is execution pass@1, which is scripts/code_eval.py's job on
    an exported checkpoint, out of band. A caller that reads this key alone is
    reading a preference gate, not the gate.
    """
    from daedalus.dpo import preference_metrics

    before = preference_metrics(reference, eval_pairs, pad_id=pad_id,
                                micro_batch=micro_batch, device=device)
    after = preference_metrics(policy, eval_pairs, pad_id=pad_id,
                               micro_batch=micro_batch, device=device)
    keys = ("accuracy", "margin", "accuracy_len_norm", "margin_len_norm")
    delta = {k: after[k] - before[k] for k in keys
             if before[k] is not None and after[k] is not None}
    return {
        "n": before["n"],
        "before": before,
        "after": after,
        "delta": delta,
        # Strictly greater: a round that moved nothing has not improved
        # anything, and `>=` would pass an unchanged model.
        "accuracy_improved": bool(delta.get("accuracy", 0.0) > 0.0),
        "gate": "phase 8 step 7 also requires execution pass@1 to improve; "
                "that is measured out of band by scripts/code_eval.py",
    }


def run_dpo(policy, reference, pairs, optimizers, device: str,
            beta: float = 0.1, max_steps: int = 500, micro_batch: int = 2,
            pad_id: int = 0, log_every: int = 20, logger=None,
            step_offset: int = 0) -> dict:
    """One light DPO round (AGENT.md SS4).

    `policy` trains; `reference` is frozen by the caller. Both are scored on
    the same pairs, and the loss is driven by the *difference* of differences,
    so the shared prompt and any bias common to both models cancel.

    Returns the last metrics dict. `accuracy` is the number to read, not the
    loss: DPO's loss falls monotonically whether or not the ranking improves.

    `step_offset` continues the SFT run's W&B x-axis. W&B drops any log at a
    step it has already passed, so restarting the count at 1 after an SFT loop
    that reached step N silently discards the whole DPO round from the
    dashboard.
    """
    from daedalus.dpo import dpo_loss, sequence_logprob

    freeze_reference(reference)
    it = iter(pairs)
    last: dict = {}
    for step in range(1, max_steps + 1):
        batch = list(itertools.islice(it, micro_batch))
        if len(batch) < micro_batch:
            print(f"[dpo] preference stream exhausted at step {step}", flush=True)
            break

        scores = {}
        for side in ("chosen", "rejected"):
            ids, labels = pad_batch([p[side] for p in batch], pad_id=pad_id)
            x = torch.tensor(ids, dtype=torch.long, device=device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            scores[f"policy_{side}"] = sequence_logprob(policy, x, y)
            with torch.no_grad():
                scores[f"ref_{side}"] = sequence_logprob(reference, x, y)

        loss, metrics = dpo_loss(scores["policy_chosen"],
                                 scores["policy_rejected"],
                                 scores["ref_chosen"], scores["ref_rejected"],
                                 beta=beta)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        last = {"step": step, **metrics}
        if logger is not None:
            logger.log({f"dpo_{k}" if k != "step" else k: v
                        for k, v in last.items()}, step=step_offset + step)
        if step % log_every == 0 or step == 1:
            print(f"[dpo] step {step:5d}  loss {metrics['dpo_loss']:.4f}  "
                  f"margin {metrics['margin']:+.3f}  "
                  f"acc {metrics['accuracy']:.3f}", flush=True)
    return last


def _run_dpo_stage(args, tokenizer, trainer, device: str) -> dict:
    """Wire the DPO round onto the just-SFT'd model.

    The reference is a snapshot of the policy *after* SFT, not the hero
    checkpoint. DPO's margin is measured against whatever the reference
    prefers, so referencing the pre-SFT model would make the round chase the
    SFT gain a second time instead of the preference signal.
    """
    import copy

    from datasets import load_dataset
    from daedalus.dpo import dpo_batch_memory_gb
    from daedalus.muon import build_optimizers

    est = dpo_batch_memory_gb(args.dpo_micro_batch, args.dpo_max_len,
                              trainer.cfg.vocab_size)
    print(f"[dpo] ~{est:.1f} GB of logits per step "
          f"(batch {args.dpo_micro_batch} x seq {args.dpo_max_len} x 4 forwards)",
          flush=True)

    policy = trainer.model
    reference = copy.deepcopy(policy)
    freeze_reference(reference)

    muon, adamw, _ = build_optimizers(policy, muon_lr=args.muon_lr / 10,
                                      adam_lr=args.adam_lr / 10)
    dataset = load_dataset(args.dpo_dataset, split=args.dpo_split,
                           streaming=True)
    pairs = iter_preference_pairs(dataset, tokenizer, args.dpo_max_len,
                                  args.max_assistant_chars)
    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    if args.dpo_eval_split:
        eval_pairs, _ = take_eval_pairs(
            iter_preference_pairs(
                load_dataset(args.dpo_dataset, split=args.dpo_eval_split,
                             streaming=True),
                tokenizer, args.dpo_max_len, args.max_assistant_chars),
            args.dpo_eval_pairs)
    else:
        # Off the *training* iterator, before run_dpo sees it, so the two sets
        # cannot overlap however the stream is ordered.
        eval_pairs, pairs = take_eval_pairs(pairs, args.dpo_eval_pairs)
    if eval_pairs:
        print(f"[dpo] holding out {len(eval_pairs)} pairs for the gate"
              f"{f' from split {args.dpo_eval_split}' if args.dpo_eval_split else ''}",
              flush=True)

    last = run_dpo(policy, reference, pairs, [muon, adamw], device,
                   beta=args.dpo_beta, max_steps=args.dpo_steps,
                   micro_batch=args.dpo_micro_batch, pad_id=pad_id,
                   logger=getattr(trainer, "wandb", None),
                   step_offset=trainer.step)

    if eval_pairs:
        gate = evaluate_preference_gate(
            reference, policy, eval_pairs, pad_id=pad_id,
            micro_batch=args.dpo_micro_batch, device=device)
        path = os.path.join(trainer.run_dir, "dpo-eval.json")
        with open(path, "w") as handle:
            json.dump(gate, handle, indent=2, sort_keys=True)
        print(f"[dpo] held-out preference accuracy "
              f"{gate['before']['accuracy']:.3f} -> {gate['after']['accuracy']:.3f} "
              f"on {gate['n']} pairs "
              f"(length-normalised {gate['before']['accuracy_len_norm']:.3f} -> "
              f"{gate['after']['accuracy_len_norm']:.3f}); "
              f"{'improved' if gate['accuracy_improved'] else 'NOT improved'} "
              f"-> {path}", flush=True)
        logger = getattr(trainer, "wandb", None)
        if logger is not None:
            logger.log({f"dpo_heldout_{k}": v
                        for k, v in gate["after"].items() if v is not None},
                       step=trainer.step + args.dpo_steps)
        last = {**last, "heldout": gate}
    return last


def save_final(trainer, args) -> str:
    """Write `runs/<run-name>/final.pt` -- the weights export.py should ship.

    Optimizer state is omitted deliberately: this artifact is the end of the
    pipeline, and post's optimizers are rebuilt from scratch anyway (SFT's by
    `--init-from`, DPO's inside `_run_dpo_stage`). `checkpoint.pt` remains the
    resumable one.
    """
    from train import save_checkpoint

    path = os.path.join(trainer.run_dir, "final.pt")
    return save_checkpoint(
        path, trainer.model, trainer.muon, trainer.adamw,
        trainer.step, trainer.tokens_seen, trainer.cfg,
        save_optimizer=False,
        extra={"stage": "post", "dpo": bool(args.dpo),
               "init_from": args.init_from})


def _cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-from", required=True,
                   help="checkpoint to fine-tune, normally hero's. Loads "
                        "weights only: SFT starts at step 0 with fresh "
                        "optimizers and its own LR schedule.")
    p.add_argument("--resume", default=None,
                   help="continue an interrupted *post* run (not hero's "
                        "checkpoint -- that is --init-from). Takes precedence "
                        "over --init-from when it exists, so a supervisor can "
                        "relaunch the same command line unchanged.")
    p.add_argument("--total-tokens", type=int, default=500_000_000,
                   help="budget the WSD schedule decays over. Must be a real "
                        "estimate: train.py's 5B pretraining default would "
                        "leave SFT ending at ~full LR because the stream runs "
                        "out roughly 10x earlier.")
    p.add_argument("--run-name", default="post-sft")
    p.add_argument("--config", default="daedalus-150m")
    p.add_argument("--dataset", default=None,
                   help=f"Hub dataset to stream (default {DEFAULT_SFT_DATASET}). "
                        f"Mutually exclusive with --built-dataset")
    p.add_argument("--built-dataset", default=None,
                   help="train on a locally built SFT set instead: a directory "
                        "holding train.jsonl and the manifest of the gate that "
                        "admitted it. No content filter is applied on the way "
                        "in -- those conversations already passed a stronger "
                        "one -- so --max-assistant-chars and --keep-cot have "
                        "nothing to do here")
    p.add_argument("--split", default="train")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--max-assistant-chars", type=int, default=1200)
    p.add_argument("--keep-cot", action="store_true",
                   help="disable the chain-of-thought filter (ablation only; "
                        "AGENT.md SS4 says long CoT hurts models under 3B)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap examples pulled from the stream")
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--muon-lr", type=float, default=2e-3,
                   help="an order below pretraining: SFT on a converged model "
                        "at pretraining lr erases what hero learned")
    p.add_argument("--adam-lr", type=float, default=3e-5)
    p.add_argument("--device", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dpo", action="store_true",
                   help="run the DPO round after SFT (AGENT.md SS4)")
    p.add_argument("--dpo-dataset", default=DEFAULT_DPO_DATASET)
    p.add_argument("--dpo-split", default="train_prefs")
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--dpo-steps", type=int, default=500)
    p.add_argument("--dpo-micro-batch", type=int, default=2,
                   help="DPO materialises logits for 4 forwards per step; see "
                        "daedalus.dpo.dpo_batch_memory_gb before raising this")
    p.add_argument("--dpo-max-len", type=int, default=1024)
    p.add_argument("--dpo-eval-pairs", type=int, default=128,
                   help="preference pairs held out of the DPO round to score "
                        "the gate on. Taken off the head of the training "
                        "stream, so the round trains on the pairs after them; "
                        "0 disables the measurement entirely, which leaves the "
                        "only reported accuracy the one run_dpo computes on "
                        "its own training pairs relative to a reference it "
                        "starts out identical to")
    p.add_argument("--dpo-eval-split", default=None,
                   help="take the held-out pairs from this split of "
                        "--dpo-dataset instead of off the training stream. "
                        "Preferred when the dataset ships a real held-out "
                        "split; the default costs no such assumption")
    args = p.parse_args(argv)

    if args.built_dataset and args.dataset:
        raise SystemExit(
            "--built-dataset and --dataset name two different sets and only one "
            "of them can be trained on. The built set is a directory on this "
            "box; --dataset streams from the Hub.")
    if args.built_dataset and args.keep_cot:
        raise SystemExit(
            "--keep-cot has nothing to disable under --built-dataset: the "
            "chain-of-thought filter ran at build time, over assistant prose "
            "with fenced code removed, and its refusals are counted in the "
            "manifest. Passing it here would look like an ablation that was "
            "never applied.")
    refuse_built_set_as_dataset(args.dataset)

    from daedalus.data import get_tokenizer
    from train import TrainArgs, Trainer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    manifest = None
    if args.built_dataset:
        # No `datasets` import on this path: a built set is read off local disk
        # and must not need the Hub library to train on.
        source, manifest = build_built_sft_source(
            args.built_dataset, tokenizer, micro_batch=args.micro_batch,
            device=device, max_len=args.max_len,
            shuffle_buffer=args.shuffle_buffer, limit=args.limit)
        halves = manifest.get("halves") or {}
        print(f"built SFT set {args.built_dataset}: "
              f"{manifest.get('train_examples')} conversations, "
              f"{manifest.get('train_supervised_tokens')} supervised tokens, "
              f"realized code share {manifest.get('realized_code_share')}, "
              f"halves " + ", ".join(
                  f"{half} {block.get('supervised_tokens')}"
                  for half, block in sorted(halves.items())))
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset or DEFAULT_SFT_DATASET,
                               split=args.split, streaming=True)
        source = build_sft_source(
            dataset, tokenizer, micro_batch=args.micro_batch, device=device,
            max_len=args.max_len, max_assistant_chars=args.max_assistant_chars,
            drop_cot=not args.keep_cot, shuffle_buffer=args.shuffle_buffer,
            limit=args.limit)

    train_args = TrainArgs(
        run_name=args.run_name, config=args.config,
        # init_from, NOT resume: hero's checkpoint carries step=610000 and
        # tokens_seen=40e9, and restoring those into this run makes fit() exit
        # before its first step. See TrainArgs.init_from.
        init_from=args.init_from, resume=args.resume,
        total_tokens=args.total_tokens,
        max_steps=args.max_steps, micro_batch=args.micro_batch,
        # Flat seq schedule: the 1024->2048 ramp is a pretraining device, and
        # a moving seq_len would repack every batch for no benefit.
        seq_start=args.max_len, seq_end=args.max_len,
        tok_start=args.micro_batch * args.max_len,
        tok_end=args.micro_batch * args.max_len,
        # 5 min, not TrainArgs' 1800 s default. This job has been killed
        # repeatedly at 10-17 min, so a 30 min checkpoint interval meant every
        # restart resumed from step 1 and threw away all progress. Cheap
        # insurance: the checkpoint is 1.4 GB written to local disk.
        ckpt_every_sec=300.0,
        muon_lr=args.muon_lr, adam_lr=args.adam_lr, device=device,
        wandb_enabled=not args.no_wandb,
        # A run over a built set is a different experiment from one over the
        # Hub stream and has to be findable as such afterwards.
        tags=["post", "sft"] + (["built-sft"] if args.built_dataset else []),
        # DPO runs after fit() returns, in this same process, and must be able
        # to log to the same run. post.py closes it below instead.
        finish_wandb=False)

    trainer = Trainer(train_args)
    trainer.batch_source = source
    trainer.fit()
    print(f"SFT done: {source.examples_seen} examples, "
          f"{source.supervised_tokens} supervised of {source.padded_tokens} "
          f"padded tokens "
          f"({100 * source.supervised_tokens / max(source.padded_tokens, 1):.1f}%)")

    if args.dpo:
        _run_dpo_stage(args, tokenizer, trainer, device)

    # The artifact export.py consumes, written last so it includes the DPO
    # round. `fit()`'s forced checkpoint.pt is the *crash-resume* file and is
    # written before DPO starts -- until this existed, run_dpo updated the
    # weights in memory and nothing ever wrote them out, so the entire DPO
    # stage was computed and discarded and export.py would have shipped the
    # SFT-only model. Kept separate from checkpoint.pt so a crash-restart
    # still resumes SFT rather than re-running it over DPO'd weights.
    final_path = save_final(trainer, args)
    print(f"post done: final weights -> {final_path}")
    trainer.wandb.finish()


if __name__ == "__main__":
    _cli()
    # Exit deterministically instead of letting CPython finalize.
    #
    # post.py leaves two HF streaming iterators half-consumed (the SFT stream
    # when the token budget is reached, the preference stream when --dpo-steps
    # is), and each keeps a parquet worker thread alive. During finalization
    # those threads touch modules whose globals are already None; with CUDA
    # also tearing down, this aborted the process:
    #
    #   Fatal Python error: PyGILState_Release: thread state ... must be
    #   current when releasing                       -> exit code 134
    #
    # Everything durable is already on disk by here -- final.pt written, W&B
    # finished, the Trainer's finally block run -- so the only thing left to
    # lose is the exit status, and a successful `post` reporting 134 would
    # make any supervisor treat it as a crash and retry a job that succeeded.
    # Deliberately *not* inside _cli(): the tests call that directly, and
    # os._exit there would kill the pytest process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
