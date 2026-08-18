"""Tests for eval.py. CPU-only, offline: no HF dataset downloads --
`_load_with_fallback` is monkeypatched with fixed rows so the task-schema
transforms are exercised without a network call.

Run: python -m pytest tests/test_eval.py -v
"""
import json
import math
import os

import numpy as np
import pytest
import torch

import eval as eval_module
from daedalus.config import PRESETS
from daedalus.data import ShardWriter
from daedalus.model import Daedalus
from eval import (
    ClozeExample,
    evaluate_bpb,
    evaluate_cloze_task,
    load_all_tasks,
    load_arc_easy,
    load_hellaswag,
    load_openbookqa,
    load_piqa,
    load_winogrande,
    mean_over_checkpoints,
    predict,
    score_ids,
    score_text,
)


# ---------------------------------------------------------------- score_ids ---

def test_score_ids_matches_manual_computation():
    cfg = PRESETS["tiny"]
    torch.manual_seed(0)
    model = Daedalus(cfg)
    model.eval()
    ctx_ids, cont_ids = [5, 6, 7], [8, 9]

    score = score_ids(model, ctx_ids, cont_ids, device="cpu")

    ids = torch.tensor([ctx_ids + cont_ids])
    with torch.no_grad():
        logits, _, _ = model(ids, targets=None, return_logits=True)
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    # sum, not mean: the harness scores with the raw sum and applies its own
    # character-length denominator for acc_norm (see eval.py's docstring)
    expected = logprobs[2, 8].item() + logprobs[3, 9].item()
    assert score == pytest.approx(expected, abs=1e-5)


def test_score_ids_restores_training_mode():
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)
    model.train()
    score_ids(model, [1, 2], [3], device="cpu")
    assert model.training is True


def test_score_ids_empty_inputs_return_neg_inf():
    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)
    assert score_ids(model, [], [1], device="cpu") == float("-inf")
    assert score_ids(model, [1], [], device="cpu") == float("-inf")


# --------------------------------------------------------------- score_text ---

class _CharTokenizer:
    """Deterministic, concatenation-invariant tokenizer for testing the
    context/continuation split logic in isolation from real BPE quirks."""

    def encode(self, s, add_special_tokens=None):
        # accepts add_special_tokens like every real HF tokenizer -- score_text
        # passes it explicitly (see test_score_text_disables_special_tokens)
        return [ord(c) for c in s]


def test_score_text_splits_at_context_token_count(monkeypatch):
    captured = {}

    def fake_score_ids(model, context_ids, continuation_ids, device="cpu"):
        captured["ctx"], captured["cont"] = context_ids, continuation_ids
        return 0.0

    monkeypatch.setattr(eval_module, "score_ids", fake_score_ids)
    tok = _CharTokenizer()
    score_text(model=None, tokenizer=tok, context="ab", continuation="cd", device="cpu")
    assert captured["ctx"] == tok.encode("ab")
    assert captured["cont"] == tok.encode("cd")


# ------------------------------------------------------------------ predict ---

def test_predict_selects_argmax_score(monkeypatch):
    scores = iter([0.1, 0.9, 0.3])
    monkeypatch.setattr(eval_module, "score_text", lambda *a, **k: next(scores))
    ex = ClozeExample(candidates=[("c", "a"), ("c", "b"), ("c", "c")], label=1)
    assert predict(model=None, tokenizer=None, example=ex, device="cpu") == 1


def test_evaluate_cloze_task_computes_accuracy(monkeypatch):
    # candidate 1 always wins on the raw sum
    scores = iter([[0.0, 1.0]] * 4)
    monkeypatch.setattr(eval_module, "score_example", lambda *a, **k: next(scores))
    examples = [ClozeExample(candidates=[("c", " a"), ("c", " b")], label=l,
                             choice_lengths=[1, 1])
               for l in [1, 1, 1, 0]]
    result = evaluate_cloze_task(None, None, examples)
    assert result["n"] == 4
    assert result["acc"] == 0.75
    assert result["acc_norm"] == 0.75
    assert result["accuracy"] == 0.75  # back-compat alias


def test_evaluate_cloze_task_acc_and_acc_norm_can_disagree(monkeypatch):
    """acc takes the best raw sum; acc_norm takes the best per-character rate,
    which favours a longer choice whose total is worse only because it is
    longer. That divergence is the whole reason acc_norm exists, and it is why
    quoting the wrong one against a published table is not a rounding error."""
    # short choice: -2.0 over 10 chars = -0.20/char
    # long  choice: -3.0 over 100 chars = -0.03/char
    monkeypatch.setattr(eval_module, "score_example",
                        lambda *a, **k: [-2.0, -3.0])
    ex = ClozeExample(candidates=[("c", " " + "x" * 10), ("c", " " + "y" * 100)],
                      label=1, choice_lengths=[10, 100])
    result = evaluate_cloze_task(None, None, [ex])
    assert result["acc"] == 0.0        # raw sum picks the short choice
    assert result["acc_norm"] == 1.0   # per-character picks the long one


def test_evaluate_cloze_task_omits_acc_norm_without_choice_lengths(monkeypatch):
    """WinoGrande: candidates share one continuation, so acc_norm is
    meaningless and the harness's own config reports acc only."""
    monkeypatch.setattr(eval_module, "score_example", lambda *a, **k: [1.0, 0.0])
    ex = ClozeExample(candidates=[("a", " c"), ("b", " c")], label=0,
                      choice_lengths=None)
    result = evaluate_cloze_task(None, None, [ex])
    assert result["acc"] == 1.0
    assert result["acc_norm"] is None


def test_headline_metric_picks_the_published_convention():
    from eval import headline_metric
    r = {"acc": 0.30, "acc_norm": 0.42}
    assert headline_metric("hellaswag", r) == 0.42
    assert headline_metric("piqa", r) == 0.42
    assert headline_metric("winogrande", r) == 0.30
    # a task with no acc_norm falls back to acc rather than returning None
    assert headline_metric("winogrande", {"acc": 0.51, "acc_norm": None}) == 0.51


def test_evaluate_cloze_task_empty_examples():
    result = evaluate_cloze_task(None, None, [])
    assert result["n"] == 0
    assert math.isnan(result["accuracy"])


# ---------------------------------------------------------------- task data ---

def _hellaswag_row(i=0, label="2"):
    return {"activity_label": f"Yoga{i}", "ctx_a": "A man is sitting.",
            "ctx_b": "he", "endings": ["a", "b", "c", "d"], "label": label}


def test_load_hellaswag_matches_harness_query_construction(monkeypatch):
    """harness process_docs: activity_label + ": " + ctx_a + " " +
    ctx_b.capitalize(), then bracket cleanup -- NOT the raw `ctx` column."""
    rows = [_hellaswag_row()]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    examples = load_hellaswag()
    assert len(examples) == 1
    ctx, cont = examples[0].candidates[0]
    assert ctx == "Yoga0: A man is sitting. He"
    assert cont == " a"            # single delimiter space, as the harness joins
    assert examples[0].label == 2
    assert examples[0].choice_lengths == [1, 1, 1, 1]


def test_load_hellaswag_applies_bracket_preprocessing(monkeypatch):
    rows = [{"activity_label": "Cooking", "ctx_a": "Add salt [title] then stir",
             "ctx_b": "now", "endings": ["[header] boil it", "wait"], "label": "0"}]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    ex = load_hellaswag()[0]
    ctx, cont = ex.candidates[0]
    assert "[title]" not in ctx and "[header]" not in cont
    assert ctx == "Cooking: Add salt. then stir Now"
    # the harness's own preprocess leaves the space the stripped bracket sat
    # on, so the joined continuation really does start with two spaces. We
    # reproduce it verbatim rather than "tidying" it -- every published
    # HellaSwag number was produced with this quirk in place.
    assert cont == "  boil it"
    assert ex.choice_lengths == [len(" boil it"), len("wait")]


def test_load_hellaswag_respects_limit(monkeypatch):
    rows = [_hellaswag_row(i, "0") for i in range(5)]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    assert len(load_hellaswag(limit=2)) == 2


def test_load_arc_easy_transforms_and_skips_bad_answer_key(monkeypatch):
    rows = [
        {"question": "Q1", "choices": {"text": ["a", "b", "c", "d"],
                                       "label": ["A", "B", "C", "D"]}, "answerKey": "C"},
        {"question": "Q2", "choices": {"text": ["a", "b"], "label": ["A", "B"]},
        "answerKey": "Z"},  # not in labels -> skipped
    ]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    examples = load_arc_easy()
    assert len(examples) == 1
    assert examples[0].label == 2
    # harness arc_easy.yaml: "Question: {{question}}\nAnswer:" + " {choice}"
    assert examples[0].candidates[2] == ("Question: Q1\nAnswer:", " c")
    assert examples[0].choice_lengths == [1, 1, 1, 1]


def test_load_openbookqa_transforms_rows(monkeypatch):
    rows = [{"question_stem": "Q", "choices": {"text": ["x", "y"], "label": ["A", "B"]},
            "answerKey": "B"}]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    examples = load_openbookqa()
    assert examples[0].label == 1
    # harness openbookqa.yaml: doc_to_text is the bare question_stem
    assert examples[0].candidates == [("Q", " x"), ("Q", " y")]


def test_load_piqa_transforms_rows(monkeypatch):
    rows = [{"goal": "G", "sol1": "s1", "sol2": "s2", "label": "1"}]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    examples = load_piqa()
    # harness piqa.yaml: "Question: {{goal}}\nAnswer:" + " {sol}"
    assert examples[0].candidates == [("Question: G\nAnswer:", " s1"),
                                      ("Question: G\nAnswer:", " s2")]
    assert examples[0].label == 1
    assert examples[0].choice_lengths == [2, 2]


def test_load_winogrande_splits_at_blank_and_substitutes(monkeypatch):
    rows = [{"sentence": "Sarah was better than Maria so _ won.",
            "option1": "Sarah", "option2": "Maria", "answer": "2"}]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    examples = load_winogrande()
    ex = examples[0]
    assert ex.label == 1
    # harness preprocess_winogrande: context = sentence[:idx] + option,
    # continuation = sentence[idx+1:].strip(), joined by one delimiter space
    assert ex.candidates[0] == ("Sarah was better than Maria so Sarah", " won.")
    assert ex.candidates[1] == ("Sarah was better than Maria so Maria", " won.")
    assert ex.choice_lengths is None   # acc only, per the harness task config


def test_load_winogrande_skips_rows_without_blank(monkeypatch):
    rows = [{"sentence": "no blank here", "option1": "a", "option2": "b", "answer": "1"}]
    monkeypatch.setattr(eval_module, "_load_with_fallback",
                        lambda candidates, split: (rows, "fake"))
    assert load_winogrande() == []


def test_load_all_tasks_skips_failing_loader(monkeypatch):
    def ok_loader(split="validation", limit=None):
        return [ClozeExample(candidates=[("c", "a"), ("c", "b")], label=0)]

    def bad_loader(split="validation", limit=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(eval_module, "TASK_LOADERS", {"good": ok_loader, "bad": bad_loader})
    result = load_all_tasks(limit=5)
    assert "good" in result and "bad" not in result


# ------------------------------------------------------------------- bpb ---

class _DecodeOnlyTokenizer:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)  # deterministic byte length


def test_evaluate_bpb_runs_and_returns_finite_positive(tmp_path):
    cfg = PRESETS["tiny"]
    tokens = list(np.random.randint(0, cfg.vocab_size, size=200))
    w = ShardWriter(str(tmp_path), shard_tokens=200)
    w.write(tokens)
    w.close()
    w.write_manifest({"eos_id": 0})

    model = Daedalus(cfg)
    bpb = evaluate_bpb(model, str(tmp_path), seq_len=16, tokenizer=_DecodeOnlyTokenizer(),
                       device="cpu", batch_size=4, max_batches=2)
    assert math.isfinite(bpb)
    assert bpb > 0


def test_evaluate_bpb_restores_training_mode(tmp_path):
    cfg = PRESETS["tiny"]
    tokens = list(np.random.randint(0, cfg.vocab_size, size=100))
    w = ShardWriter(str(tmp_path), shard_tokens=100)
    w.write(tokens)
    w.close()
    w.write_manifest({"eos_id": 0})

    model = Daedalus(cfg)
    model.train()
    evaluate_bpb(model, str(tmp_path), seq_len=16, tokenizer=_DecodeOnlyTokenizer(),
                device="cpu", batch_size=4)
    assert model.training is True


def test_evaluate_bpb_excludes_z_loss_and_restores_it(tmp_path):
    """bpb must measure the likelihood only. z-loss is a stability
    regulariser the model adds to its *training* loss; leaving it in inflates
    reported bits-per-byte and makes it incomparable to any published number.
    """
    import copy

    cfg = copy.deepcopy(PRESETS["tiny"])
    tokens = list(np.random.randint(0, cfg.vocab_size, size=200))
    w = ShardWriter(str(tmp_path), shard_tokens=200)
    w.write(tokens)
    w.close()
    w.write_manifest({"eos_id": 0})

    torch.manual_seed(0)
    model = Daedalus(cfg)
    kw = dict(seq_len=16, tokenizer=_DecodeOnlyTokenizer(), device="cpu",
              batch_size=4, max_batches=2)

    cfg.z_loss = 0.0
    baseline = evaluate_bpb(model, str(tmp_path), **kw)

    cfg.z_loss = 5.0  # absurdly large: would dominate bpb if it leaked through
    with_z = evaluate_bpb(model, str(tmp_path), **kw)

    assert with_z == pytest.approx(baseline, rel=1e-9)
    assert cfg.z_loss == 5.0  # restored, not left zeroed for later callers


# ------------------------------------------------------------- aggregation ---

def test_mean_over_checkpoints_averages_ignoring_missing():
    results = [
        {"hellaswag": 0.3, "val_bpb": 1.0},
        {"hellaswag": 0.5, "val_bpb": 1.2},
        {"hellaswag": 0.4},  # missing val_bpb
    ]
    mean = mean_over_checkpoints(results)
    assert mean["hellaswag"] == pytest.approx(0.4)
    assert mean["val_bpb"] == pytest.approx(1.1)


def test_mean_over_checkpoints_empty_list():
    assert mean_over_checkpoints([]) == {}


def test_mean_over_checkpoints_ignores_nan():
    results = [{"x": 1.0}, {"x": float("nan")}, {"x": 3.0}]
    mean = mean_over_checkpoints(results)
    assert mean["x"] == pytest.approx(2.0)


def test_mean_over_checkpoints_skips_non_numeric_fields():
    """A "checkpoint" path string stashed alongside metrics (as the CLI does)
    must not blow up sum() -- found live when the CLI crashed on this."""
    results = [
        {"hellaswag": 0.3, "checkpoint": "/runs/a/ckpt.pt"},
        {"hellaswag": 0.5, "checkpoint": "/runs/b/ckpt.pt"},
    ]
    mean = mean_over_checkpoints(results)
    assert mean["hellaswag"] == pytest.approx(0.4)
    assert "checkpoint" not in mean


def test_mean_over_checkpoints_omits_key_when_all_values_missing():
    results = [{"x": 1.0}, {"x": 2.0}]  # "val_bpb" never present at all
    mean = mean_over_checkpoints(results)
    assert "val_bpb" not in mean


# ------------------------------------------------------- evaluate_checkpoint ---

def test_evaluate_checkpoint_bounds_bpb_by_default(tmp_path, monkeypatch):
    """A real held-out shard directory can be tens of thousands of seq_len
    windows -- evaluate_checkpoint must default to a bounded sample, not an
    unbounded full pass, or periodic during-training eval becomes far too
    slow (this was discovered live: an unbounded call looked like a hang)."""
    from daedalus.muon import build_optimizers
    from eval import evaluate_checkpoint

    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model)
    ckpt_path = eval_module_save_checkpoint(tmp_path, model, muon, adamw, cfg)

    captured = {}

    def fake_mixture(model, shard_dir, seq_len, tokenizer, device,
                     batch_size, max_batches, weights=None):
        captured["max_batches"] = max_batches
        captured["batch_size"] = batch_size
        return {"val_bpb": 1.23, "per_source_val_bpb": {}}

    monkeypatch.setattr(eval_module, "evaluate_bpb_mixture", fake_mixture)
    r = evaluate_checkpoint(str(ckpt_path), "tiny", tokenizer=None, task_examples={},
                            shard_dir="fake_dir")
    assert captured["max_batches"] == 100  # bounded, not None/unbounded
    assert r["val_bpb"] == 1.23


def test_evaluate_checkpoint_scores_a_mixture_root_per_source(tmp_path, monkeypatch):
    """`evaluate_checkpoint` must route a mixture root through
    `evaluate_bpb_mixture`, not the single-dir `evaluate_bpb`.

    Found live on 2026-08-11 by running the after-run chain's own eval command
    against the real hero holdout: `hero.py` carves a per-source holdout whose
    *root* has no manifest.json, so `evaluate_bpb` raised FileNotFoundError --
    and it fired before the task loop, so the process died with `--out` never
    written. The `[DONE] hero` issue would have reported neither val_bpb nor
    the 5-task mean after a ~6-day run. `evaluate_bpb_mixture`'s own docstring
    had warned about exactly this for `train.py`; the fix was never applied to
    this caller.

    Asserting on the *paths* rather than just "it did not raise" is what makes
    this a regression test: under the old code `evaluate_bpb` is called once
    with the root, so `seen` reads ["holdout"] instead of the two sources.
    """
    from daedalus.muon import build_optimizers
    from eval import evaluate_checkpoint

    cfg = PRESETS["tiny"]
    model = Daedalus(cfg)
    muon, adamw, _ = build_optimizers(model)
    ckpt_path = eval_module_save_checkpoint(tmp_path, model, muon, adamw, cfg)

    root = tmp_path / "holdout"
    for name in ("source-a", "source-b"):
        (root / name).mkdir(parents=True)
        (root / name / "manifest.json").write_text(json.dumps({"total_tokens": 100}))
    assert not (root / "manifest.json").exists()   # the shape that broke it

    seen = []

    def fake_evaluate_bpb(model, shard_dir, *a, **k):
        seen.append(shard_dir)
        return 2.0

    monkeypatch.setattr(eval_module, "evaluate_bpb", fake_evaluate_bpb)
    r = evaluate_checkpoint(str(ckpt_path), "tiny", tokenizer=None,
                            task_examples={}, shard_dir=str(root))

    assert sorted(os.path.basename(p) for p in seen) == ["source-a", "source-b"]
    # A float, not the mixture dict, so W&B and mean_over_checkpoints both see
    # it; the per-source breakdown rides alongside rather than replacing it.
    assert r["val_bpb"] == pytest.approx(2.0)
    assert set(r["per_source_val_bpb"]) == {"source-a", "source-b"}


def eval_module_save_checkpoint(tmp_path, model, muon, adamw, cfg):
    from train import save_checkpoint
    return save_checkpoint(str(tmp_path / "ckpt.pt"), model, muon, adamw,
                           step=0, tokens_seen=0, cfg=cfg)


# ---------------------------------------------------------------- wandb ---

def test_git_short_sha_returns_string_in_a_repo():
    sha = eval_module._git_short_sha()
    assert isinstance(sha, str) and sha  # this repo is a git checkout


def test_git_short_sha_survives_subprocess_failure(monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise RuntimeError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert eval_module._git_short_sha() == "unknown"


# ------------------------------------------------------------- peer models ---

class _FakeHFOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeHFModel(torch.nn.Module):
    """Stands in for a HuggingFace causal LM: keyword-only `input_ids`, and a
    return object carrying `.logits` rather than Daedalus's 3-tuple."""

    def __init__(self, vocab_size=11):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen = []
        self.lin = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids=None, **kw):
        self.seen.append(input_ids)
        return _FakeHFOutput(self.lin(input_ids).float())


def test_hf_adapter_presents_the_daedalus_forward_signature():
    """score_ids() calls model(ids, targets=None, return_logits=True) and
    unpacks a 3-tuple; a peer must arrive through that same path or the
    comparison is measuring two different scorers."""
    from eval import HFCausalLMAdapter
    hf = _FakeHFModel()
    model = HFCausalLMAdapter(hf)
    ids = torch.tensor([[1, 2, 3]])
    logits, loss, aux = model(ids, targets=None, return_logits=True)
    assert logits.shape == (1, 3, hf.vocab_size)
    assert loss is None and aux is None
    assert torch.equal(hf.seen[0], ids)      # passed as a keyword, not positionally


def test_hf_adapter_scores_through_the_shared_cloze_path():
    """End to end through score_ids: a peer and Daedalus must produce
    log-likelihoods on the same scale, or acc/acc_norm are not comparable."""
    from eval import HFCausalLMAdapter
    model = HFCausalLMAdapter(_FakeHFModel())
    s = score_ids(model, [1, 2], [3], device="cpu")
    assert isinstance(s, float) and s < 0        # a log-probability
    assert math.isfinite(s)


def test_hf_adapter_is_eval_mode_safe():
    """score_ids flips train/eval around the forward pass; the adapter must
    forward that to the wrapped model rather than shadowing it."""
    from eval import HFCausalLMAdapter
    hf = _FakeHFModel()
    model = HFCausalLMAdapter(hf)
    model.train()
    assert hf.training is True
    model.eval()
    assert hf.training is False


# ------------------------------------------------------------------ splits ---

def test_task_splits_match_the_harness_yaml():
    """lm-evaluation-harness scores a task's `test` split when its YAML
    declares one. ARC-Easy and OpenBookQA do; HellaSwag/PIQA/WinoGrande have
    held-out test labels and score validation.

    Pinned because getting this wrong is silent and expensive: scoring
    ARC-Easy on validation measures 570 examples instead of the 2,376 every
    published number uses, and the only symptom was a few points of
    disagreement that looks exactly like a model being worse."""
    from eval import TASK_LOADERS, TASK_SPLITS
    assert TASK_SPLITS == {
        "hellaswag": "validation",
        "arc_easy": "test",
        "piqa": "validation",
        "openbookqa": "test",
        "winogrande": "validation",
    }
    assert set(TASK_SPLITS) == set(TASK_LOADERS)


def test_each_loader_defaults_to_its_harness_split():
    """A caller reaching for `load_arc_easy()` directly must get the same split
    `load_all_tasks` would give it."""
    import inspect
    from eval import TASK_LOADERS, TASK_SPLITS
    for name, loader in TASK_LOADERS.items():
        default = inspect.signature(loader).parameters["split"].default
        assert default == TASK_SPLITS[name], f"{name} defaults to {default!r}"


def test_load_all_tasks_requests_the_declared_split(monkeypatch):
    """load_all_tasks must pass the split through, not rely on the default
    happening to agree."""
    from eval import TASK_SPLITS
    seen = {}

    def _fake(name):
        def loader(split=None, limit=None):
            seen[name] = split
            return []
        return loader

    monkeypatch.setattr(eval_module, "TASK_LOADERS",
                        {n: _fake(n) for n in TASK_SPLITS})
    eval_module.load_all_tasks(limit=1)
    assert seen == TASK_SPLITS


def test_score_text_disables_special_tokens():
    """lm-evaluation-harness scores causal models with add_special_tokens=False
    (HFLM's add_bos_token defaults to False). Left implicit, this is not even
    uniform across the peers we compare against: opt-125m's tokenizer prepends
    BOS by default while pythia/gpt-neo/SmolLM2's do not, so one model in the
    table would be scored under a different convention than the rest."""
    calls = []

    class _Tok:
        def encode(self, text, add_special_tokens=None):
            calls.append(add_special_tokens)
            return [1] * (len(text.split()) + 1)

    class _M(torch.nn.Module):
        def forward(self, ids, targets=None, return_logits=True):
            return torch.zeros(1, ids.shape[1], 5), None, None

    score_text(_M(), _Tok(), "a b", " c", device="cpu")
    assert calls and all(c is False for c in calls), calls


def test_score_text_still_splits_context_and_continuation_correctly():
    """The BOS change must not shift where the continuation starts."""
    class _Tok:
        def encode(self, text, add_special_tokens=None):
            return [ord(c) % 97 for c in text if not c.isspace()]

    seen = {}

    class _M(torch.nn.Module):
        def forward(self, ids, targets=None, return_logits=True):
            seen["n"] = ids.shape[1]
            return torch.zeros(1, ids.shape[1], 128), None, None

    score_text(_M(), _Tok(), "abc", " de", device="cpu")
    assert seen["n"] == 5      # "abc" (3) + "de" (2), whitespace dropped by _Tok


# ------------------------------------------------- mixture-aware val_bpb ---

def test_evaluate_bpb_mixture_passes_a_single_source_dir_straight_through(
        tmp_path, monkeypatch):
    """A dir with its own manifest.json is one source, not a mixture root --
    `sweep` and any single-source `train.py --val-dir` still take this path."""
    import eval as eval_mod

    d = tmp_path / "holdout"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"total_tokens": 10}))
    monkeypatch.setattr(eval_mod, "evaluate_bpb", lambda *a, **k: 1.25)

    out = eval_mod.evaluate_bpb_mixture(object(), str(d), 16, object())
    assert out == {"val_bpb": 1.25, "per_source_val_bpb": {}}


def test_evaluate_bpb_mixture_excludes_sources_with_no_training_weight(
        tmp_path, monkeypatch, capsys):
    """A source held out but never sampled (e.g. one dropped by the epoch cap)
    must not pull the average toward a distribution training never saw."""
    import eval as eval_mod

    root = tmp_path / "holdout"
    for name in ("kept", "unweighted"):
        (root / name).mkdir(parents=True)
        (root / name / "manifest.json").write_text(json.dumps({"total_tokens": 100}))

    bpb = {"kept": 2.0, "unweighted": 9.0}
    monkeypatch.setattr(
        eval_mod, "evaluate_bpb",
        lambda model, shard_dir, *a, **k: bpb[os.path.basename(shard_dir)])

    out = eval_mod.evaluate_bpb_mixture(object(), str(root), 16, object(),
                                        weights={"kept": 1.0})
    assert out["val_bpb"] == pytest.approx(2.0)
    assert out["per_source_val_bpb"]["unweighted"]["weight"] == 0.0
    assert "unweighted" in capsys.readouterr().out


def test_evaluate_bpb_mixture_refuses_to_guess_when_no_weight_matches(
        tmp_path, monkeypatch):
    """Silently falling back to holdout sizes when the weight map misses every
    source would hide a wiring mistake behind a plausible-looking number."""
    import eval as eval_mod

    root = tmp_path / "holdout"
    (root / "source-a").mkdir(parents=True)
    (root / "source-a" / "manifest.json").write_text(json.dumps({"total_tokens": 100}))
    monkeypatch.setattr(eval_mod, "evaluate_bpb", lambda *a, **k: 2.0)

    with pytest.raises(ValueError, match="refusing to guess"):
        eval_mod.evaluate_bpb_mixture(object(), str(root), 16, object(),
                                      weights={"some-other-source": 1.0})


# ------------------------------------------------- peer/checkpoint parity ---
# The two scoring paths -- our checkpoints and HF peers -- had drifted: the
# peer path recorded `<task>_n` and the checkpoint path did not. `n` is what
# makes a peer comparison checkable, and the peer table was built over full
# validation splits while --task-limit defaulted to 500.

def test_task_record_includes_the_sample_count():
    rec = eval_module.task_record("hellaswag",
                         {"acc": 0.3, "acc_norm": 0.35, "n": 10042})
    assert rec["hellaswag_n"] == 10042
    assert rec["hellaswag_acc"] == 0.3
    assert rec["hellaswag_acc_norm"] == 0.35
    assert rec["hellaswag"] == 0.35            # acc_norm is the headline


def test_task_record_omits_acc_norm_where_it_is_meaningless():
    rec = eval_module.task_record("winogrande", {"acc": 0.51, "acc_norm": None, "n": 1267})
    assert "winogrande_acc_norm" not in rec
    assert rec["winogrande"] == 0.51 and rec["winogrande_n"] == 1267


def test_checkpoint_and_peer_paths_produce_the_same_keys():
    """Whatever the two paths report, they must be comparable field for field
    -- the headline table is built by putting them side by side."""
    for name in ("hellaswag", "arc_easy", "piqa", "openbookqa"):
        res = {"acc": 0.3, "acc_norm": 0.35, "n": 500}
        assert set(eval_module.task_record(name, res)) == {
            name, f"{name}_acc", f"{name}_acc_norm", f"{name}_n"}


def test_task_limit_defaults_to_the_full_split(monkeypatch):
    """runs/eval/peer-*.json were produced over full splits (hellaswag
    n=10042). A 500-example default would have scored our own model on a
    subset and printed the two side by side as if comparable."""
    import sys
    captured = {}
    monkeypatch.setattr(eval_module, "load_all_tasks",
                        lambda limit=None: captured.setdefault("limit", limit) or {})
    monkeypatch.setattr(sys, "argv",
                        ["eval.py", "--hf-model", "sshleifer/tiny-gpt2",
                         "--no-wandb", "--out", "/tmp/_unused_eval.json"])
    monkeypatch.setattr(eval_module, "load_peer_model",
                        lambda *a, **k: (object(), object()))
    monkeypatch.setattr(eval_module, "mean_over_checkpoints", lambda r: {})
    try:
        eval_module._cli()
    except SystemExit:
        pass
    assert captured.get("limit") is None


def test_task_limit_zero_and_unset_produce_the_same_items():
    """The after-run chain pairs two sides scored under *different flags*: step
    7b passes `--task-limit 0` for the peers, step 5 leaves it unset for hero.
    `mcnemar.compare` refuses any task whose item digest differs between the two
    sides, so if these two conventions ever diverged, every task would land in
    "Not compared" -- and the paired file would still be written, still look
    well-formed, and contain no verdict at all.

    It holds today only because `0` is falsy in each loader's
    `if limit and len(out) >= limit`. That is exactly the kind of thing an
    explicit `if limit is not None` "cleanup" would break silently.
    """
    from eval import load_openbookqa, item_digest
    zero, unset = load_openbookqa(limit=0), load_openbookqa(limit=None)
    assert len(zero) == len(unset) > 0
    assert item_digest(zero) == item_digest(unset)
