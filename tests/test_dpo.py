"""Tests for daedalus/dpo.py (AGENT.md SS4: one light DPO round).

DPO's loss falls monotonically whether or not the ranking actually improves,
so a plausible curve proves very little. These pin the things that would
otherwise be inferred from that curve: the sign of the margin, which tokens
are scored, and that the reference model is genuinely frozen.
"""
import math

import pytest
import torch

from daedalus.config import PRESETS
from daedalus.dpo import (
    IGNORE_INDEX,
    dpo_batch_memory_gb,
    dpo_loss,
    freeze_reference,
    preference_metrics,
    sequence_logprob,
)
from daedalus.model import Daedalus


def _t(*vals):
    return torch.tensor(list(vals), dtype=torch.float32)


# --------------------------------------------------------------------- loss ---

def test_zero_margin_is_log_two():
    """Policy identical to reference: loss = -log sigmoid(0) = log 2."""
    z = _t(0.0, 0.0)
    loss, m = dpo_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert m["margin"] == pytest.approx(0.0)


def test_loss_falls_as_the_margin_grows():
    ref_c, ref_r = _t(0.0), _t(0.0)
    small, _ = dpo_loss(_t(1.0), _t(0.0), ref_c, ref_r, beta=0.1)
    large, _ = dpo_loss(_t(5.0), _t(0.0), ref_c, ref_r, beta=0.1)
    assert large.item() < small.item()


def test_preferring_the_rejected_response_costs_more_than_log_two():
    """The sign check. If the margin were inverted this would be the cheap
    direction and DPO would confidently train the wrong preference."""
    loss, m = dpo_loss(_t(0.0), _t(3.0), _t(0.0), _t(0.0), beta=0.1)
    assert loss.item() > math.log(2)
    assert m["margin"] < 0
    assert m["accuracy"] == 0.0


def test_accuracy_counts_pairs_ranked_correctly():
    policy_c, policy_r = _t(1.0, 0.0, 2.0), _t(0.0, 1.0, 0.0)
    zeros = torch.zeros(3)
    _, m = dpo_loss(policy_c, policy_r, zeros, zeros, beta=0.1)
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_reference_shifts_the_margin():
    """DPO is relative to the reference: an improvement the reference already
    had is not an improvement."""
    _, same = dpo_loss(_t(2.0), _t(0.0), _t(2.0), _t(0.0), beta=0.1)
    assert same["margin"] == pytest.approx(0.0)


def test_large_negative_margin_stays_finite():
    """log(sigmoid(x)) underflows to -inf here; logsigmoid does not. This is
    the early-training regime, so a NaN would surface as a dead run rather
    than as a large loss."""
    loss, _ = dpo_loss(_t(-500.0), _t(500.0), _t(0.0), _t(0.0), beta=1.0)
    assert torch.isfinite(loss)
    assert loss.item() > 100


def test_beta_scales_the_loss():
    a, _ = dpo_loss(_t(1.0), _t(0.0), _t(0.0), _t(0.0), beta=0.1)
    b, _ = dpo_loss(_t(1.0), _t(0.0), _t(0.0), _t(0.0), beta=0.5)
    assert b.item() < a.item()  # stronger beta, more reward for the same margin


def test_loss_is_differentiable_towards_the_chosen_response():
    pc = torch.tensor([0.0], requires_grad=True)
    pr = torch.tensor([0.0], requires_grad=True)
    loss, _ = dpo_loss(pc, pr, _t(0.0), _t(0.0), beta=0.1)
    loss.backward()
    # raising the chosen logprob must reduce the loss, and vice versa
    assert pc.grad.item() < 0
    assert pr.grad.item() > 0


# ---------------------------------------------------------------- logprobs ---

@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = Daedalus(PRESETS["tiny"])
    m.eval()
    return m


def test_sequence_logprob_ignores_masked_positions(model):
    x = torch.randint(1, PRESETS["tiny"].vocab_size, (2, 12))
    all_sup = x.clone()
    half = x.clone()
    half[:, :6] = IGNORE_INDEX

    with torch.no_grad():
        full = sequence_logprob(model, x, all_sup)
        partial = sequence_logprob(model, x, half)
    # fewer supervised tokens -> strictly greater (less negative) total
    assert torch.all(partial > full)


def test_sequence_logprob_of_nothing_is_zero(model):
    x = torch.randint(1, PRESETS["tiny"].vocab_size, (2, 8))
    with torch.no_grad():
        out = sequence_logprob(model, x, torch.full_like(x, IGNORE_INDEX))
    assert torch.allclose(out, torch.zeros(2))


def test_sequence_logprob_is_a_sum_not_a_mean(model):
    """DPO compares whole-sequence likelihoods; length-normalising here would
    quietly change the objective."""
    x = torch.randint(1, PRESETS["tiny"].vocab_size, (1, 16))
    y = x.clone()
    y[:, :8] = IGNORE_INDEX
    with torch.no_grad():
        total = sequence_logprob(model, x, y)
    n_sup = int((y[:, 1:] != IGNORE_INDEX).sum())
    # a mean would sit in [-log V, 0]; a sum over n tokens is n times larger
    assert total.item() < -1.0
    assert total.item() == pytest.approx(total.item())
    assert n_sup > 1
    assert total.item() < -n_sup * 0.1


def test_sequence_logprob_matches_manual_gather(model):
    """Pins the shift convention: logits[:, :-1] against labels[:, 1:], the
    same pairing Daedalus.forward uses. Off by one here and DPO would score
    the wrong token at every position."""
    x = torch.randint(1, PRESETS["tiny"].vocab_size, (1, 10))
    y = x.clone()
    y[:, :4] = IGNORE_INDEX

    with torch.no_grad():
        got = sequence_logprob(model, x, y)
        logits, _, _ = model(x, return_logits=True)

    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    expected = sum(lp[i, y[0, i + 1]].item()
                   for i in range(x.shape[1] - 1)
                   if y[0, i + 1] != IGNORE_INDEX)
    assert got.item() == pytest.approx(expected, abs=1e-4)


def test_sequence_logprob_is_differentiable(model):
    x = torch.randint(1, PRESETS["tiny"].vocab_size, (1, 8))
    out = sequence_logprob(model, x, x.clone())
    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    model.zero_grad(set_to_none=True)


# --------------------------------------------------------------- reference ---

def test_freeze_reference_stops_gradients_and_training():
    ref = Daedalus(PRESETS["tiny"])
    freeze_reference(ref)
    assert not ref.training
    assert all(not p.requires_grad for p in ref.parameters())

    x = torch.randint(1, PRESETS["tiny"].vocab_size, (1, 8))
    out = sequence_logprob(ref, x, x.clone())
    assert not out.requires_grad, "a trainable reference is a moving baseline"


# ------------------------------------------------- held-out preference gate ---
# Phase 8 step 7 keeps the DPO model only if held-out preference accuracy
# improves. `dpo_loss`'s `accuracy` cannot answer that: it is relative to the
# reference, on the pairs just trained on, and the reference starts out as a
# copy of the policy -- so it begins at exactly 0.0 and any movement at all
# reads as an improvement.


class _FixedLogits:
    """A model with one fixed next-token distribution, so the ranking a test
    asks about is arithmetic rather than a property of random init."""

    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, input_ids, return_logits=False):
        b, t = input_ids.shape
        return self._logits.view(1, 1, -1).expand(b, t, -1).clone(), None, None


def _pair(chosen_ids, rejected_ids, n_prompt=1):
    def side(ids):
        return (list(ids), [IGNORE_INDEX] * n_prompt + list(ids[n_prompt:]))
    return {"chosen": side(chosen_ids), "rejected": side(rejected_ids)}


def _model_favouring(token: int, vocab: int = 8) -> _FixedLogits:
    logits = torch.zeros(vocab)
    logits[token] = 6.0
    return _FixedLogits(logits)


def test_preference_accuracy_is_absolute_not_relative_to_a_reference():
    """The defect this metric exists for.

    With policy == reference every DPO margin is identically zero, so
    `dpo_loss` reports accuracy 0.0 for a model that in fact ranks every pair
    correctly. Read as "held-out preference accuracy", 0.0 is not merely
    imprecise -- it is the wrong number about the wrong model.
    """
    model = _model_favouring(3)
    pairs = [_pair([1, 3, 3, 3], [1, 5, 5, 5]) for _ in range(4)]

    got = preference_metrics(model, pairs, pad_id=0, micro_batch=2)
    assert got["n"] == 4
    assert got["accuracy"] == 1.0
    assert got["margin"] > 0

    # the same four pairs through the relative metric, policy == reference
    zero = torch.zeros(4)
    _, relative = dpo_loss(zero, zero, zero, zero)
    assert relative["accuracy"] == 0.0


def test_preference_accuracy_falls_when_the_model_prefers_the_rejected_side():
    """It has to be able to say no, or it is not a gate."""
    model = _model_favouring(5)
    pairs = [_pair([1, 3, 3, 3], [1, 5, 5, 5]) for _ in range(3)]
    got = preference_metrics(model, pairs, pad_id=0, micro_batch=2)
    assert got["accuracy"] == 0.0
    assert got["margin"] < 0


def test_length_normalised_accuracy_separates_a_preference_from_a_length_shift():
    """`sequence_logprob` is a sum, so every extra token makes it smaller and a
    longer response is penalised for its length alone. Here the chosen side is
    better per token and longer; the sum prefers the short rejected one and the
    length-normalised control does not."""
    logits = torch.zeros(8)
    logits[3] = 0.5                                   # chosen token, mildly liked
    logits[5] = 0.0                                   # rejected token
    model = _FixedLogits(logits)
    pairs = [_pair([1] + [3] * 12, [1] + [5] * 2) for _ in range(3)]

    got = preference_metrics(model, pairs, pad_id=0, micro_batch=3)
    assert got["accuracy"] == 0.0, "the sum should be dragged down by length"
    assert got["margin"] < 0
    assert got["accuracy_len_norm"] == 1.0, "per token, chosen is preferred"
    assert got["margin_len_norm"] > 0


def test_preference_metrics_scores_both_models_on_the_same_pairs():
    """`pairs` is consumed into a list, so a generator can be handed to two
    models without the second one silently scoring nothing."""
    model_a, model_b = _model_favouring(3), _model_favouring(5)
    pairs = (_pair([1, 3, 3], [1, 5, 5]) for _ in range(3))
    materialised = list(pairs)

    a = preference_metrics(model_a, iter(materialised), pad_id=0)
    b = preference_metrics(model_b, iter(materialised), pad_id=0)
    assert a["n"] == b["n"] == 3
    assert a["accuracy"] == 1.0 and b["accuracy"] == 0.0


def test_preference_metrics_on_no_pairs_reports_nothing_rather_than_zero():
    """0.0 accuracy over no pairs is a measurement that never happened wearing
    a failing gate's clothes."""
    got = preference_metrics(_model_favouring(3), [], pad_id=0)
    assert got["n"] == 0
    assert got["accuracy"] is None and got["margin"] is None


def test_preference_metrics_leaves_the_model_as_it_found_it():
    """It is called mid-run on the live policy; leaving it in eval would drop
    dropout for the rest of the round."""
    model = _model_favouring(3).train()
    preference_metrics(model, [_pair([1, 3, 3], [1, 5, 5])], pad_id=0)
    assert model.training, "training mode must be restored"

    frozen = _model_favouring(3).eval()
    preference_metrics(frozen, [_pair([1, 3, 3], [1, 5, 5])], pad_id=0)
    assert not frozen.training, "and eval mode must not be turned into train"


def test_preference_metrics_does_not_move_or_grad_the_model():
    torch.manual_seed(0)
    model = Daedalus(PRESETS["tiny"])
    before = [p.detach().clone() for p in model.parameters()]
    v = PRESETS["tiny"].vocab_size
    pairs = [_pair([1, 2, 3, 4], [4, 3, 2, 1]) for _ in range(2)]

    got = preference_metrics(model, pairs, pad_id=0, micro_batch=2)
    assert 0.0 <= got["accuracy"] <= 1.0
    assert v > 4
    for a, b in zip(before, model.parameters()):
        assert torch.equal(a, b)
    assert all(p.grad is None for p in model.parameters())


# ------------------------------------------------------------------ memory ---

def test_batch_memory_estimate_is_honest_about_the_ceiling():
    """DPO here materialises logits, unlike the training path. The estimate
    exists so a caller picks a batch size deliberately."""
    # one forward at batch 4 / seq 1024 is ~0.8 GB; DPO does four of them
    gb = dpo_batch_memory_gb(batch_size=4, seq_len=1024, vocab_size=49152)
    assert 3.0 < gb < 3.5                      # ~3.2 GB, fine on 32 GB
    assert dpo_batch_memory_gb(batch_size=4, seq_len=1024, vocab_size=49152,
                               n_forwards=1) == pytest.approx(gb / 4)
    big = dpo_batch_memory_gb(batch_size=8, seq_len=2048, vocab_size=49152)
    assert big > 12                            # and this is not fine
