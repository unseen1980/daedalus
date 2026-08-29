"""Tests for post.py's SFT data path (AGENT.md SS4, plan step 10).

The HF dataset is replaced by a plain list of rows, so these run offline and
in milliseconds. What they pin is the wiring: filters actually applied,
batches actually masked, and the stream actually bounded.
"""
import itertools
import json
import os

import pytest
import torch

import post
from daedalus.chatml import IGNORE_INDEX
from daedalus.data import get_tokenizer


def _row(user, assistant):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


# ------------------------------------------------------------------ filters ---

def test_iter_chat_examples_applies_the_content_filters():
    rows = [
        _row("hi", "hello"),                                  # keep
        _row("hi", "x" * 5000),                               # too long
        _row("hi", "Let me think step by step. 4."),          # chain of thought
        _row("hi", "   "),                                    # empty assistant
        {"messages": []},                                     # malformed
        {"nothing": 1},                                       # malformed
    ]
    kept = list(post.iter_chat_examples(rows, max_assistant_chars=1200,
                                        drop_cot=True))
    assert len(kept) == 1
    assert kept[0][1]["content"] == "hello"


def test_keep_cot_is_opt_in():
    rows = [_row("hi", "Let me think step by step. 4.")]
    assert len(list(post.iter_chat_examples(rows, 1200, drop_cot=False))) == 1
    assert len(list(post.iter_chat_examples(rows, 1200, drop_cot=True))) == 0


# ------------------------------------------------------------------ streaming ---

def test_iter_chat_examples_is_lazy():
    """ADDENDUM 2 rule 2: never accumulate the corpus. An infinite source must
    be consumable, which it is not if the filter builds a list first."""
    infinite = ({"messages": [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "yes"}]}
                for _ in itertools.count())
    got = list(itertools.islice(post.iter_chat_examples(infinite, 1200, True), 3))
    assert len(got) == 3


def test_shuffle_buffered_is_bounded_and_lossless():
    items = list(range(500))
    out = list(post.shuffle_buffered(iter(items), buffer_size=32, seed=1))
    assert sorted(out) == items          # nothing dropped or duplicated
    assert out != items                  # and it actually shuffled


def test_shuffle_buffered_holds_only_buffer_size():
    """The buffer, not the corpus, sets peak memory."""
    seen_max = 0

    def probe():
        nonlocal seen_max
        for i in itertools.count():
            seen_max = i + 1
            if i >= 999:
                return
            yield i

    gen = post.shuffle_buffered(probe(), buffer_size=16, seed=0)
    next(gen)
    # After one output the source has been pulled at most buffer_size + 1
    # times -- it has not been drained to build a full-corpus shuffle.
    assert seen_max <= 17


# -------------------------------------------------------------------- batches ---

def test_batch_source_yields_masked_pairs(tok):
    rows = [_row(f"question {i}", f"answer {i}") for i in range(8)]
    src = post.build_sft_source(rows, tok, micro_batch=4, device="cpu",
                                max_len=128, shuffle_buffer=4,
                                materialise=True)
    x, y = src.get_batch(2048)

    assert x.shape == y.shape == (4, x.shape[1])
    assert x.dtype == y.dtype == torch.long
    # Something supervised, and strictly less than everything -- the prompt
    # and the padding must not be.
    n_sup = int((y != IGNORE_INDEX).sum())
    assert 0 < n_sup < y.numel()
    # Wherever supervised, the label is the input token (labels are unshifted;
    # the model does the shift).
    sup = y != IGNORE_INDEX
    assert torch.equal(x[sup], y[sup])


def test_batch_source_pads_to_the_longest_member(tok):
    rows = [_row("a", "b"), _row("a much longer question here",
                                 "a considerably longer answer as well")]
    src = post.build_sft_source(rows, tok, micro_batch=2, device="cpu",
                                max_len=128, shuffle_buffer=2,
                                materialise=True)
    x, y = src.get_batch(2048)
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    # the shorter row is padded, and every pad position is ignored in labels
    pads = x == pad_id
    assert pads.any()
    assert bool((y[pads] == IGNORE_INDEX).all())


def test_batch_source_counts_supervision_density(tok):
    rows = [_row(f"q{i}", f"a{i}") for i in range(4)]
    src = post.build_sft_source(rows, tok, micro_batch=2, device="cpu",
                                max_len=128, shuffle_buffer=2,
                                materialise=True)
    src.get_batch(2048)
    assert src.examples_seen == 2
    assert 0 < src.supervised_tokens < src.padded_tokens


def test_batch_source_loops_a_materialised_set(tok):
    rows = [_row("q", "a")]
    src = post.build_sft_source(rows, tok, micro_batch=2, device="cpu",
                                max_len=128, shuffle_buffer=2,
                                materialise=True)
    src.get_batch(2048)
    src.get_batch(2048)          # past the end of a 1-example set
    assert src.epochs >= 1


def test_batch_source_refuses_to_silently_shorten_a_one_shot_stream(tok):
    """A generator cannot be re-iterated. Training fewer steps than asked for,
    quietly, is worse than stopping."""
    src = post.SFTBatchSource(iter([([1, 2], [-100, 2])]), micro_batch=1,
                              device="cpu", pad_id=0, loop=True)
    src.get_batch(8)
    with pytest.raises(RuntimeError, match="cannot be re-iterated"):
        src.get_batch(8)


# --------------------------------------------------------- built SFT sets ---
# Phase 8 step 6 trains on a set `daedalus/code_sft.py` selected on disk. Its
# rows are already `iter_chat_examples`' shape, which is exactly what makes
# reading them through the streaming path so easy to do by accident -- and that
# path re-applies a filter the build already applied more strictly, to prose
# alone, so it drops the code payload without a count.

#: Short prose, long payload: the shape the streaming reader gets wrong.
_LONG_FUNCTION = ("Here you go.\n\n```python\n"
                  + "\n".join(f"    step_{i} = {i} * 2" for i in range(120))
                  + "\n```")


def _built_row(user, assistant, half="code"):
    """One line as `code_sft.build_record` writes it."""
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}],
            "half": half, "source": f"{half}-source",
            "dataset": "HuggingFaceTB/smol-smoltalk",
            "supervised_tokens": 32, "total_tokens": 64,
            "primary_language": "python" if half == "code" else None,
            "ships_test": False, "execution_tested": False}


def _built_set(tmp_path, rows, *, vocab_size, max_len=2048, problems=(),
               name="built", train_file="train.jsonl", write_train=True):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    if write_train:
        with (directory / train_file).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    manifest = {
        "schema": 1, "seed": 0, "max_len": max_len,
        "supervised_token_budget": 8_000_000, "code_share": 0.65,
        "train_examples": len(rows), "train_supervised_tokens": 32 * len(rows),
        "realized_code_share": 0.65,
        "halves": {"code": {"supervised_tokens": 21},
                   "general": {"supervised_tokens": 11}},
        "tokenizer": {"name_or_path": "HuggingFaceTB/SmolLM2-135M",
                      "vocab_size": vocab_size},
        "files": {"train": train_file, "holdout": "holdout.jsonl"},
        "problems": list(problems),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_a_built_set_is_read_without_a_second_content_filter(tmp_path, tok):
    """The trap this reader exists for. `keep_example` counts the whole
    assistant turn; the build counted its prose with the code removed, so the
    streaming path drops exactly the payload phase 8 is adding."""
    row = _built_row("write it", _LONG_FUNCTION)
    assert len(_LONG_FUNCTION) > 1200

    through_the_stream = list(post.iter_chat_examples(
        [row], max_assistant_chars=1200, drop_cot=True))
    assert through_the_stream == [], (
        "the streaming reader is expected to drop this; if it stops doing so "
        "the contrast this reader exists for has changed")

    directory = _built_set(tmp_path, [row], vocab_size=tok.vocab_size)
    kept = list(post.iter_built_examples(directory / "train.jsonl"))
    assert len(kept) == 1
    assert kept[0][1]["content"] == _LONG_FUNCTION


def test_a_built_set_refuses_a_manifest_that_reports_problems(tmp_path, tok):
    """`build_problems` is the build's own verdict on what it wrote, reported
    rather than raised so a short half still lands on disk to be looked at."""
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size,
                           problems=["the code half came up short"])
    with pytest.raises(post.BuiltSFTSetError, match="came up short"):
        post.read_built_manifest(directory)


def test_a_built_set_refuses_a_smaller_budget_than_it_was_admitted_at(tmp_path, tok):
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size, max_len=2048)
    with pytest.raises(post.BuiltSFTSetError, match="admitted at max_len"):
        post.read_built_manifest(directory, max_len=1024)


def test_a_larger_budget_than_the_build_used_is_accepted(tmp_path, tok):
    """It admits everything the build admitted, so nothing is dropped."""
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size, max_len=1024)
    manifest = post.read_built_manifest(directory, max_len=2048)
    assert manifest["max_len"] == 1024


def test_a_built_set_refuses_a_tokenizer_it_was_not_measured_with(tmp_path, tok):
    """Its token counts, its half shares and its max_len refusals are all
    measurements of one tokenizer."""
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size + 1)
    with pytest.raises(post.BuiltSFTSetError, match="tokenizer"):
        post.read_built_manifest(directory, tokenizer=tok)


def test_a_built_set_refuses_a_missing_manifest(tmp_path):
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "train.jsonl").write_text("{}\n")
    with pytest.raises(post.BuiltSFTSetError, match="does not exist"):
        post.read_built_manifest(tmp_path / "loose")


def test_a_built_set_refuses_a_training_file_that_is_not_there(tmp_path, tok):
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size, write_train=False)
    manifest = post.read_built_manifest(directory)
    with pytest.raises(post.BuiltSFTSetError, match="half-copied"):
        post.built_train_path(directory, manifest)


def test_a_built_line_that_is_not_a_conversation_stops_the_run(tmp_path, tok):
    """The opposite of the streaming reader's stance, for the same reason: the
    producer of this file is known, so skipping trains on an unrecorded subset."""
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size)
    with (directory / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"half": "code"}) + "\n")
    with pytest.raises(post.BuiltSFTSetError, match="line 2 carries no messages"):
        list(post.iter_built_examples(directory / "train.jsonl"))


def test_encode_built_refuses_rather_than_dropping_what_does_not_fit(tok):
    """`iter_encoded` drops silently, which is right for rows nobody vetted and
    wrong for a set whose manifest says every one of them fits."""
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "a " * 500}]
    assert list(post.iter_encoded([messages], tok, max_len=8)) == []
    with pytest.raises(post.BuiltSFTSetError, match="does not fit"):
        list(post.encode_built([messages], tok, max_len=8))


def test_a_built_source_loops_without_materialising_the_set(tmp_path, tok):
    """A file can be read again, so the source loops epochs while peak memory
    stays the shuffle buffer."""
    rows = [_built_row(f"q{i}", f"a{i}") for i in range(4)]
    directory = _built_set(tmp_path, rows, vocab_size=tok.vocab_size,
                           max_len=128)
    source, manifest = post.build_built_sft_source(
        directory, tok, micro_batch=2, device="cpu", max_len=128,
        shuffle_buffer=2)
    assert manifest["train_examples"] == 4
    assert not isinstance(source._make_iter(), list)
    for _ in range(4):                       # 8 examples out of a 4-example set
        source.get_batch(2048)
    assert source.epochs >= 1
    assert source.examples_seen == 8


def test_a_built_source_drives_a_real_train_step(tmp_path, tok):
    """End to end through train.py's Trainer, as the streaming source is."""
    from train import TrainArgs, Trainer

    rows = [_built_row(f"q{i}", f"a{i}") for i in range(16)]
    directory = _built_set(tmp_path, rows, vocab_size=tok.vocab_size,
                           max_len=64)
    source, _ = post.build_built_sft_source(
        directory, tok, micro_batch=2, device="cpu", max_len=64,
        shuffle_buffer=4)
    args = TrainArgs(run_name="built-smoke", config="tiny", max_steps=1,
                     micro_batch=2, seq_start=64, seq_end=64,
                     tok_start=128, tok_end=128, compile=False, device="cpu",
                     run_dir=str(tmp_path / "built-run"), wandb_enabled=False)
    t = Trainer(args)
    t.batch_source = _ClampedSource(source, args)
    stats = t.train_step()
    assert not stats["skipped"]
    assert torch.isfinite(torch.tensor(stats["loss"]))


# ------------------------------------------------------------ trainer wiring ---

def test_source_drives_a_real_train_step(tmp_path, tok):
    """End to end through train.py's Trainer: the (x, y) pair reaches the loss
    and produces a finite gradient step."""
    from train import TrainArgs, Trainer

    rows = [_row(f"q{i}", f"a{i}") for i in range(16)]
    src = post.build_sft_source(rows, tok, micro_batch=2, device="cpu",
                                max_len=64, shuffle_buffer=4,
                                materialise=True)
    args = TrainArgs(run_name="sft-smoke", config="tiny", max_steps=1,
                     micro_batch=2, seq_start=64, seq_end=64,
                     tok_start=128, tok_end=128, compile=False, device="cpu",
                     run_dir=str(tmp_path / "sft"), wandb_enabled=False)
    t = Trainer(args)
    # `tiny` has a 512-token vocab; the real tokenizer emits larger ids.
    t.batch_source = _ClampedSource(src, args)
    stats = t.train_step()
    assert not stats["skipped"]
    assert torch.isfinite(torch.tensor(stats["loss"]))


class _ClampedSource:
    """Maps real-tokenizer ids into the tiny preset's vocab so the wiring can
    be exercised without instantiating a 150M model on CPU."""

    def __init__(self, inner, args):
        self.inner = inner
        from daedalus.config import PRESETS
        self.vocab = PRESETS["tiny"].vocab_size

    def get_batch(self, seq_len):
        x, y = self.inner.get_batch(seq_len)
        mask = y != IGNORE_INDEX
        x = x % self.vocab
        y = torch.where(mask, y % self.vocab, torch.full_like(y, IGNORE_INDEX))
        return x, y


# --------------------------------------------------------------------- DPO ---

def _pref_row(prompt, good, bad):
    return {"chosen": [{"role": "user", "content": prompt},
                       {"role": "assistant", "content": good}],
            "rejected": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": bad}]}


def test_preference_pairs_encode_both_sides(tok):
    rows = [_pref_row("2+2?", "4.", "Fish.")]
    pairs = list(post.iter_preference_pairs(rows, tok, max_len=128,
                                            max_assistant_chars=1200))
    assert len(pairs) == 1
    for side in ("chosen", "rejected"):
        ids, labels = pairs[0][side]
        assert len(ids) == len(labels)
        assert any(l != IGNORE_INDEX for l in labels)


def test_preference_pair_dropped_when_one_side_does_not_fit(tok):
    """Scoring a complete response against a truncated one would teach DPO
    that fragments are preferable."""
    rows = [_pref_row("hi", "short", "word " * 5000)]
    assert list(post.iter_preference_pairs(rows, tok, max_len=64,
                                           max_assistant_chars=100000)) == []


def test_preference_rows_missing_a_side_are_skipped(tok):
    rows = [{"chosen": [{"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"}]},
            {"rejected": []}, {}]
    assert list(post.iter_preference_pairs(rows, tok, 128, 1200)) == []


def test_run_dpo_steps_the_policy_and_leaves_the_reference_alone(tok):
    """End to end on the tiny preset: the policy must move, the reference
    must not, and the metrics must come back."""
    import copy
    from daedalus.config import PRESETS
    from daedalus.model import Daedalus
    from daedalus.muon import build_optimizers

    torch.manual_seed(0)
    policy = Daedalus(PRESETS["tiny"])
    reference = copy.deepcopy(policy)
    ref_before = [p.detach().clone() for p in reference.parameters()]
    pol_before = [p.detach().clone() for p in policy.parameters()]

    V = PRESETS["tiny"].vocab_size
    def pair(seed):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(1, V, (12,), generator=g).tolist()
        labels = [IGNORE_INDEX] * 6 + ids[6:]
        return {"chosen": (ids, labels), "rejected": (ids[::-1], labels)}

    muon, adamw, _ = build_optimizers(policy, muon_lr=1e-3, adam_lr=1e-4)
    last = post.run_dpo(policy, reference, (pair(i) for i in range(20)),
                        [muon, adamw], device="cpu", beta=0.1, max_steps=3,
                        micro_batch=2, pad_id=0, log_every=100)

    assert last["step"] == 3
    assert 0.0 <= last["accuracy"] <= 1.0
    assert any(not torch.equal(a, b)
               for a, b in zip(pol_before, policy.parameters())), "policy must move"
    for a, b in zip(ref_before, reference.parameters()):
        assert torch.equal(a, b), "reference must not move"


def test_run_dpo_stops_cleanly_when_pairs_run_out(tok):
    """Better than looping on a short stream and reporting steps it never
    actually took."""
    import copy
    from daedalus.config import PRESETS
    from daedalus.model import Daedalus
    from daedalus.muon import build_optimizers

    torch.manual_seed(0)
    policy = Daedalus(PRESETS["tiny"])
    reference = copy.deepcopy(policy)
    muon, adamw, _ = build_optimizers(policy, muon_lr=1e-3, adam_lr=1e-4)
    ids = [1, 2, 3, 4, 5, 6]
    labels = [IGNORE_INDEX] * 3 + ids[3:]
    one = [{"chosen": (ids, labels), "rejected": (ids[::-1], labels)}]

    last = post.run_dpo(policy, reference, iter(one), [muon, adamw],
                        device="cpu", max_steps=50, micro_batch=2, pad_id=0)
    assert last == {}  # never completed a full batch


# --------------------------------------------------------------- CLI wiring ---
# post.py fine-tunes hero's checkpoint. It used to hand it to train.py as
# `--resume`, which restores hero's step (610000) and tokens_seen (40e9) as
# well as its weights -- so fit() broke at the top of its first iteration and
# the entire SFT stage silently trained on nothing. See TrainArgs.init_from.

class _StubTrainer:
    """Captures the TrainArgs post.py builds, without touching a GPU."""
    last_args = None

    def __init__(self, args):
        _StubTrainer.last_args = args
        self.batch_source = None
        self.fitted = False
        self.wandb = type("W", (), {"finish": lambda self: None,
                                    "log": lambda self, *a, **k: None})()

    def fit(self):
        self.fitted = True


@pytest.fixture
def captured_train_args(monkeypatch, tok):
    """These pin the TrainArgs post.py builds; save_final needs a real Trainer
    and is covered separately below."""
    import train
    monkeypatch.setattr(train, "Trainer", _StubTrainer)
    monkeypatch.setattr(post, "save_final", lambda trainer, args: "final.pt")
    monkeypatch.setattr("daedalus.data.get_tokenizer", lambda *a, **k: tok)
    import datasets
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda *a, **k: [_row("hi", "hello")] * 4)
    _StubTrainer.last_args = None
    return lambda argv: (post._cli(argv), _StubTrainer.last_args)[1]


def test_cli_passes_the_base_checkpoint_as_init_from_not_resume(captured_train_args):
    args = captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                                "--limit", "2", "--micro-batch", "1",
                                "--max-len", "128"])
    assert args.init_from == "/ckpt/hero.pt"
    assert args.resume is None, (
        "hero's checkpoint must not arrive as `resume`: that restores step "
        "610000/tokens_seen 40e9 and fit() exits before its first step")


def test_cli_keeps_resume_for_restarting_the_post_run_itself(captured_train_args):
    args = captured_train_args(["--init-from", "/ckpt/hero.pt",
                                "--resume", "runs/post-sft/checkpoint.pt",
                                "--no-wandb", "--limit", "2",
                                "--micro-batch", "1", "--max-len", "128"])
    assert args.init_from == "/ckpt/hero.pt"
    assert args.resume == "runs/post-sft/checkpoint.pt"


def test_cli_sets_a_real_token_budget_for_the_lr_schedule(captured_train_args):
    """train.py's 5B pretraining default would leave SFT ending at ~full LR:
    the stream runs out roughly 10x earlier, so the WSD decay never arrives."""
    args = captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                                "--limit", "2", "--micro-batch", "1",
                                "--max-len", "128"])
    assert args.total_tokens == 500_000_000
    assert args.total_tokens < 5_000_000_000

    override = captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                                    "--total-tokens", "123456789", "--limit", "2",
                                    "--micro-batch", "1", "--max-len", "128"])
    assert override.total_tokens == 123_456_789


def test_cli_pins_the_seq_ramp_flat(captured_train_args):
    args = captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                                "--limit", "2", "--micro-batch", "1",
                                "--max-len", "128"])
    assert args.seq_start == args.seq_end == 128
    assert args.tok_start == args.tok_end == 128


def test_cli_still_streams_the_default_dataset(captured_train_args, monkeypatch):
    """The built path is additive: with neither flag, nothing moves."""
    asked = {}
    import datasets

    def record(name, **kwargs):
        asked["name"] = name
        return [_row("hi", "hello")] * 4

    monkeypatch.setattr(datasets, "load_dataset", record)
    args = captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                                "--limit", "2", "--micro-batch", "1",
                                "--max-len", "128"])
    assert asked["name"] == post.DEFAULT_SFT_DATASET
    assert "built-sft" not in args.tags


def test_cli_trains_on_a_built_set_without_touching_the_hub(
        captured_train_args, monkeypatch, tmp_path, tok):
    """A set on local disk must not need the Hub library to train on."""
    import datasets

    def refuse(*a, **k):
        raise AssertionError("the built path must not call load_dataset")

    monkeypatch.setattr(datasets, "load_dataset", refuse)
    directory = _built_set(tmp_path, [_built_row(f"q{i}", f"a{i}")
                                      for i in range(8)],
                           vocab_size=tok.vocab_size, max_len=128)
    args = captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                                "--built-dataset", str(directory),
                                "--micro-batch", "1", "--max-len", "128"])
    assert args.init_from == "/ckpt/hero.pt"
    assert "built-sft" in args.tags


def test_cli_refuses_a_built_set_handed_to_dataset(tmp_path, tok):
    """The trap: a built row is already the streaming reader's shape, so the
    wrong flag trains on a filtered fraction of the set without complaint."""
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size)
    with pytest.raises(SystemExit, match="--built-dataset"):
        post._cli(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                   "--dataset", str(directory)])
    with pytest.raises(SystemExit, match="--built-dataset"):
        post._cli(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                   "--dataset", str(directory / "train.jsonl")])


def test_cli_refuses_two_sets_at_once(tmp_path, tok):
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size)
    with pytest.raises(SystemExit, match="only one"):
        post._cli(["--init-from", "/ckpt/hero.pt", "--no-wandb",
                   "--built-dataset", str(directory),
                   "--dataset", post.DEFAULT_SFT_DATASET])


def test_cli_refuses_an_ablation_flag_that_would_never_be_applied(tmp_path, tok):
    """`--keep-cot` under a built set reads as an ablation nobody ran: the
    chain-of-thought filter ran at build time and its refusals are counted."""
    directory = _built_set(tmp_path, [_built_row("q", "a")],
                           vocab_size=tok.vocab_size)
    with pytest.raises(SystemExit, match="keep-cot"):
        post._cli(["--init-from", "/ckpt/hero.pt", "--no-wandb", "--keep-cot",
                   "--built-dataset", str(directory)])


# ------------------------------------------------ the DPO round must survive ---
# run_dpo updates the policy in memory. Until save_final existed, nothing ever
# wrote those weights out: fit()'s forced checkpoint.pt is written *before* DPO
# starts, so the whole round was computed and thrown away and export.py would
# have shipped the SFT-only model.

def _post_trainer(tmp_path, tok, steps=2):
    import train
    args = train.TrainArgs(
        run_name="p", config="tiny", device="cpu", compile=False,
        micro_batch=1, seq_start=16, seq_end=16, tok_start=16, tok_end=16,
        wandb_enabled=False, finish_wandb=False, max_steps=steps,
        run_dir=str(tmp_path / "p"), ckpt_every_sec=1e9, push_every_sec=1e9,
        total_tokens=1024)
    t = train.Trainer(args)
    t.batch_source = _FixedSource(t.cfg.vocab_size)
    t.fit()
    return t


class _FixedSource:
    def __init__(self, vocab):
        self.vocab = vocab

    def get_batch(self, seq_len):
        return torch.randint(0, self.vocab, (1, 16))


class _Args:
    dpo = True
    init_from = "/ckpt/hero.pt"


def test_save_final_captures_weights_changed_after_the_sft_checkpoint(tmp_path, tok):
    """Simulates what DPO does: mutate the policy after fit() has already
    written checkpoint.pt."""
    import train

    t = _post_trainer(tmp_path, tok)
    resume_ckpt = torch.load(t.ckpt_path, map_location="cpu", weights_only=False)

    with torch.no_grad():                      # stand in for the DPO updates
        for p in t.model.parameters():
            p.add_(0.5)

    final_path = post.save_final(t, _Args())
    final = torch.load(final_path, map_location="cpu", weights_only=False)

    key = next(iter(final["model"]))
    assert not torch.allclose(final["model"][key], resume_ckpt["model"][key]), \
        "final.pt did not capture the post-checkpoint (DPO) weight updates"
    for k, v in t.model.state_dict().items():
        assert torch.allclose(final["model"][k], v)


def test_save_final_is_a_separate_file_from_the_resume_checkpoint(tmp_path, tok):
    """A crash-restart must resume SFT, not re-run it over DPO'd weights."""
    t = _post_trainer(tmp_path, tok)
    final_path = post.save_final(t, _Args())
    assert os.path.basename(final_path) == "final.pt"
    assert os.path.realpath(final_path) != os.path.realpath(t.ckpt_path)
    assert os.path.exists(t.ckpt_path)


def test_save_final_records_how_the_model_was_made(tmp_path, tok):
    t = _post_trainer(tmp_path, tok)
    final = torch.load(post.save_final(t, _Args()), map_location="cpu",
                       weights_only=False)
    assert final["extra"]["stage"] == "post"
    assert final["extra"]["dpo"] is True
    assert final["extra"]["init_from"] == "/ckpt/hero.pt"


def test_final_is_loadable_by_export(tmp_path, tok):
    """export.py calls train.load_checkpoint on whatever it is given."""
    from train import load_checkpoint
    from daedalus.model import Daedalus
    from daedalus.config import PRESETS

    t = _post_trainer(tmp_path, tok)
    path = post.save_final(t, _Args())
    fresh = Daedalus(PRESETS["tiny"])
    load_checkpoint(path, fresh, map_location="cpu")   # strict load, must not raise
    for a, b in zip(t.model.parameters(), fresh.parameters()):
        assert torch.allclose(a, b)


# ------------------------------------------------------- DPO stays on the chart ---

class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def log(self, record, step=None):
        self.calls.append((step, record))


def test_dpo_logs_continue_the_sft_step_axis(tok):
    """W&B drops a log at a step it has already passed, so restarting DPO's
    count at 1 after an SFT loop that reached step N discards the entire round
    from the operator's dashboard."""
    logger = _RecordingLogger()
    pairs = [{"chosen": ([1, 2, 3], [1, 2, 3]),
              "rejected": ([1, 2, 4], [1, 2, 4])} for _ in range(6)]

    from daedalus.config import PRESETS
    from daedalus.model import Daedalus
    import copy
    policy = Daedalus(PRESETS["tiny"])
    reference = copy.deepcopy(policy)
    opts = []

    post.run_dpo(policy, reference, pairs, opts, "cpu", max_steps=3,
                 micro_batch=1, pad_id=0, logger=logger, step_offset=1000)

    steps = [s for s, _ in logger.calls]
    assert steps == [1001, 1002, 1003], steps
    assert all(any(k.startswith("dpo_") for k in rec) for _, rec in logger.calls)


# ------------------------------------------------ the held-out preference gate ---
# Phase 8 step 7 keeps the DPO model over the SFT one only if held-out
# preference accuracy improves. run_dpo's `accuracy` cannot decide that: it is
# measured on the pairs it just trained on, and relative to a reference that
# starts out as a copy of the policy, so it begins at 0.0 and rises on any
# movement at all.

def _stream(n, start=0):
    """Preference pairs whose first token is their index, so a test can say
    exactly which ones DPO was handed."""
    return ({"chosen": ([i, 2, 3, 4], [IGNORE_INDEX, 2, 3, 4]),
             "rejected": ([i, 5, 6, 7], [IGNORE_INDEX, 5, 6, 7])}
            for i in range(start, start + n))


def test_held_out_pairs_are_disjoint_from_the_pairs_dpo_trains_on():
    """Both sets come off one iterator, so overlap is not possible however the
    stream is ordered. Two `load_dataset` calls would rely on them agreeing."""
    held, remaining = post.take_eval_pairs(_stream(10), 3)
    assert [p["chosen"][0][0] for p in held] == [0, 1, 2]
    assert [p["chosen"][0][0] for p in remaining] == [3, 4, 5, 6, 7, 8, 9]


def test_holding_out_nothing_leaves_the_training_stream_whole():
    held, remaining = post.take_eval_pairs(_stream(4), 0)
    assert held == []
    assert len(list(remaining)) == 4


def test_holding_out_more_pairs_than_exist_takes_what_there_is():
    held, remaining = post.take_eval_pairs(_stream(2), 50)
    assert len(held) == 2
    assert list(remaining) == []


def test_gate_reports_no_improvement_when_the_round_changed_nothing():
    """The case the relative metric cannot express. policy == reference is a
    DPO round that did nothing, and `dpo_loss` calls that accuracy 0.0 -> any
    movement. Here it is correctly a delta of zero and not an improvement."""
    import copy
    from daedalus.config import PRESETS
    from daedalus.model import Daedalus

    torch.manual_seed(0)
    reference = Daedalus(PRESETS["tiny"]).eval()
    policy = copy.deepcopy(reference)
    pairs = [{"chosen": ([1, 2, 3, 4], [IGNORE_INDEX, 2, 3, 4]),
              "rejected": ([1, 5, 6, 7], [IGNORE_INDEX, 5, 6, 7])}
             for _ in range(3)]

    gate = post.evaluate_preference_gate(reference, policy, pairs, pad_id=0,
                                         micro_batch=2)
    assert gate["n"] == 3
    assert gate["delta"]["accuracy"] == 0.0
    assert gate["accuracy_improved"] is False
    assert gate["before"]["accuracy"] == gate["after"]["accuracy"]


def test_gate_reads_the_before_model_off_the_frozen_reference():
    """`reference` never moves, so scoring it after the round still answers
    "what did the SFT model do" -- which is the model the gate keeps if DPO
    fails to beat it."""
    import copy
    from daedalus.config import PRESETS
    from daedalus.model import Daedalus

    torch.manual_seed(0)
    reference = Daedalus(PRESETS["tiny"]).eval()
    policy = copy.deepcopy(reference)
    with torch.no_grad():                       # stand in for the DPO updates
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    pairs = [{"chosen": ([1, 2, 3, 4], [IGNORE_INDEX, 2, 3, 4]),
              "rejected": ([1, 5, 6, 7], [IGNORE_INDEX, 5, 6, 7])}
             for _ in range(4)]

    gate = post.evaluate_preference_gate(reference, policy, pairs, pad_id=0,
                                         micro_batch=2)
    from daedalus.dpo import preference_metrics
    assert gate["before"] == preference_metrics(reference, pairs, pad_id=0,
                                                micro_batch=2)
    assert gate["after"] != gate["before"], "a moved policy must score apart"
    assert set(gate["delta"]) == {"accuracy", "margin", "accuracy_len_norm",
                                  "margin_len_norm"}


def test_gate_says_it_is_only_half_of_phase_8_step_7():
    """The other half is execution pass@1, out of band. A caller that reads
    accuracy_improved alone must not think it read the gate."""
    import copy
    from daedalus.config import PRESETS
    from daedalus.model import Daedalus

    torch.manual_seed(0)
    reference = Daedalus(PRESETS["tiny"]).eval()
    gate = post.evaluate_preference_gate(
        reference, copy.deepcopy(reference),
        [{"chosen": ([1, 2], [IGNORE_INDEX, 2]),
          "rejected": ([1, 3], [IGNORE_INDEX, 3])}], pad_id=0)
    assert "pass@1" in gate["gate"]


def test_cli_holds_out_gate_pairs_by_default(captured_train_args, monkeypatch):
    """Off by default would leave run_dpo's training-pair accuracy as the only
    reported number, and that is the one that cannot fail."""
    seen = {}
    monkeypatch.setattr(
        post, "_run_dpo_stage",
        lambda args, tokenizer, trainer, device: seen.update(args=args) or {})
    captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb", "--dpo",
                         "--limit", "2", "--micro-batch", "1", "--max-len", "128"])
    assert seen["args"].dpo_eval_pairs == 128
    assert seen["args"].dpo_eval_split is None


def test_cli_can_take_the_holdout_from_a_real_split(captured_train_args,
                                                    monkeypatch):
    seen = {}
    monkeypatch.setattr(
        post, "_run_dpo_stage",
        lambda args, tokenizer, trainer, device: seen.update(args=args) or {})
    captured_train_args(["--init-from", "/ckpt/hero.pt", "--no-wandb", "--dpo",
                         "--dpo-eval-split", "test_prefs",
                         "--dpo-eval-pairs", "32",
                         "--limit", "2", "--micro-batch", "1", "--max-len", "128"])
    assert seen["args"].dpo_eval_split == "test_prefs"
    assert seen["args"].dpo_eval_pairs == 32


def test_dpo_stage_scores_the_gate_on_pairs_it_did_not_train_on(tmp_path, tok,
                                                                monkeypatch):
    """End to end through _run_dpo_stage: the holdout comes off the training
    stream, the round trains on what is left, and dpo-eval.json lands beside
    the run."""
    import json

    t = _post_trainer(tmp_path, tok)
    trained_on = []

    real_run_dpo = post.run_dpo

    def spy(policy, reference, pairs, *a, **kw):
        pairs = list(pairs)
        trained_on.extend(p["chosen"][0][0] for p in pairs)
        return real_run_dpo(policy, reference, iter(pairs), *a, **kw)

    monkeypatch.setattr(post, "run_dpo", spy)
    monkeypatch.setattr(post, "iter_preference_pairs",
                        lambda *a, **k: _stream(12))
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [])

    class _A:
        dpo_dataset, dpo_split, dpo_eval_split = "d", "train", None
        dpo_eval_pairs, dpo_max_len, max_assistant_chars = 4, 32, 1200
        dpo_beta, dpo_steps, dpo_micro_batch = 0.1, 2, 2
        muon_lr, adam_lr = 1e-3, 1e-4

    out = post._run_dpo_stage(_A(), tok, t, "cpu")

    assert trained_on and min(trained_on) >= 4, (
        f"DPO trained on held-out pairs {sorted(set(trained_on))}")
    gate = json.load(open(os.path.join(t.run_dir, "dpo-eval.json")))
    assert gate["n"] == 4
    assert out["heldout"] == gate


def test_dpo_stage_takes_the_holdout_from_the_eval_split_when_asked(tmp_path, tok,
                                                                    monkeypatch):
    """With --dpo-eval-split the holdout comes from that split and the training
    stream is left whole. Swapping the two branches would silently go on
    carving the holdout out of training while the operator asked for a real
    split, and nothing downstream would say so."""
    import json

    t = _post_trainer(tmp_path, tok)
    trained_on = []

    real_run_dpo = post.run_dpo

    def spy(policy, reference, pairs, *a, **kw):
        pairs = list(pairs)
        trained_on.extend(p["chosen"][0][0] for p in pairs)
        return real_run_dpo(policy, reference, iter(pairs), *a, **kw)

    monkeypatch.setattr(post, "run_dpo", spy)
    import datasets
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda name, split=None, **k: ("ds", split))
    monkeypatch.setattr(
        post, "iter_preference_pairs",
        lambda dataset, *a, **k: (_stream(8) if dataset == ("ds", "train")
                                  else _stream(3, start=100)))

    class _A:
        dpo_dataset, dpo_split, dpo_eval_split = "d", "train", "test_prefs"
        dpo_eval_pairs, dpo_max_len, max_assistant_chars = 3, 32, 1200
        dpo_beta, dpo_steps, dpo_micro_batch = 0.1, 2, 2
        muon_lr, adam_lr = 1e-3, 1e-4

    post._run_dpo_stage(_A(), tok, t, "cpu")

    assert sorted(trained_on) == list(range(8)), (
        "the training split must arrive whole when the holdout has its own split")
    assert json.load(open(os.path.join(t.run_dir, "dpo-eval.json")))["n"] == 3
