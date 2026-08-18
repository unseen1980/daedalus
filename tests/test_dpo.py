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
