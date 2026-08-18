"""Equivalence tests for the chunked (fused) loss head.

The chunked path in `Daedalus.forward` must be a drop-in replacement for the
original single-shot path: same loss value, same gradients. These tests pin that
down on CPU with the tiny preset so they run anywhere, including in CI without a
GPU.

Run: python -m pytest tests/test_loss_chunking.py -v
"""
import copy

import pytest
import torch

from daedalus.config import PRESETS
from daedalus.model import Daedalus


def _tiny_model(seed=0, **overrides):
    cfg = copy.deepcopy(PRESETS["tiny"])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    torch.manual_seed(seed)
    model = Daedalus(cfg)
    model.train()
    return model, cfg


def _batch(cfg, batch=2, seq=64, seed=1):
    torch.manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (batch, seq))


# --------------------------------------------------------------- loss value ---

@pytest.mark.parametrize("chunk", [1, 7, 64, 4096])
@pytest.mark.parametrize("z_loss,softcap", [(1e-4, 0.0), (0.0, 0.0),
                                            (1e-4, 30.0), (0.0, 30.0)])
def test_chunked_loss_matches_full(chunk, z_loss, softcap):
    """Chunked loss equals the single-shot loss across chunk sizes and knobs.

    Chunk sizes deliberately include 1 (degenerate), 7 (ragged — does not divide
    the token count evenly), and values at/above the total token count.
    """
    model, cfg = _tiny_model(z_loss=z_loss, logit_softcap=softcap,
                             loss_chunk_size=chunk)
    ids = _batch(cfg)

    _, loss_full, _ = model(ids, targets=ids, return_logits=True)
    _, loss_chunked, _ = model(ids, targets=ids, return_logits=False)

    torch.testing.assert_close(loss_chunked, loss_full, rtol=1e-5, atol=1e-6)


def test_chunked_path_returns_no_logits():
    """The memory win depends on logits never being materialised."""
    model, cfg = _tiny_model()
    ids = _batch(cfg)

    logits, loss, _ = model(ids, targets=ids)
    assert logits is None, "training path must not materialise logits"
    assert loss is not None

    logits, _, _ = model(ids, targets=ids, return_logits=True)
    assert logits is not None and logits.shape == (2, 64, cfg.vocab_size)


def test_inference_path_unchanged():
    """With no targets, logits are still returned and loss is None."""
    model, cfg = _tiny_model()
    ids = _batch(cfg)

    logits, loss, _ = model(ids)
    assert loss is None
    assert logits.shape == (2, 64, cfg.vocab_size)


# ----------------------------------------------------------------- gradients ---

@pytest.mark.parametrize("chunk", [7, 64])
def test_gradients_match_full(chunk):
    """Gradients through the chunked head match the single-shot head.

    This is the test that actually protects training: activation checkpointing
    recomputes the chunk logits in backward, and a mistake there would show up
    as wrong gradients while the forward loss still looked correct.
    """
    ids = _batch(PRESETS["tiny"])

    model_a, _ = _tiny_model(loss_chunk_size=chunk)
    model_b = copy.deepcopy(model_a)

    _, loss_a, _ = model_a(ids, targets=ids, return_logits=True)
    loss_a.backward()

    _, loss_b, _ = model_b(ids, targets=ids, return_logits=False)
    loss_b.backward()

    grads_a = {n: p.grad for n, p in model_a.named_parameters() if p.grad is not None}
    grads_b = {n: p.grad for n, p in model_b.named_parameters() if p.grad is not None}

    assert grads_a.keys() == grads_b.keys(), "same parameters must receive grads"
    assert grads_a, "expected at least one gradient"

    for name in grads_a:
        torch.testing.assert_close(
            grads_b[name], grads_a[name], rtol=1e-4, atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch for {n}:\n{m}",
        )


def test_embedding_grad_flows_through_tied_head():
    """Tied embeddings receive gradient from the chunked head.

    The output weight is a closure variable inside the checkpointed function
    rather than an explicit input, so this guards the case where autograd could
    silently drop it.
    """
    model, cfg = _tiny_model(loss_chunk_size=7)
    assert cfg.tie_word_embeddings and model.lm_head is None
    ids = _batch(cfg)

    _, loss, _ = model(ids, targets=ids, return_logits=False)
    loss.backward()

    grad = model.embed_tokens.weight.grad
    assert grad is not None, "tied embedding weight received no gradient"
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "tied embedding gradient is all zeros"


# -------------------------------------------------------------- edge cases ---

def test_ignore_index_respected():
    """Ignored positions are excluded from the mean, matching cross_entropy."""
    model, cfg = _tiny_model(loss_chunk_size=7, z_loss=0.0)
    ids = _batch(cfg)
    targets = ids.clone()
    targets[:, 1::2] = -100

    _, loss_full, _ = model(ids, targets=targets, return_logits=True)
    _, loss_chunked, _ = model(ids, targets=targets, return_logits=False)

    torch.testing.assert_close(loss_chunked, loss_full, rtol=1e-5, atol=1e-6)


def test_all_ignored_is_finite():
    """An all-ignored batch yields 0.0 rather than the NaN the old path gave.

    Documented, deliberate divergence from the single-shot path.
    """
    model, cfg = _tiny_model(loss_chunk_size=7, z_loss=0.0)
    ids = _batch(cfg)
    targets = torch.full_like(ids, -100)

    _, loss, _ = model(ids, targets=targets, return_logits=False)
    assert torch.isfinite(loss), "all-ignored batch must not produce NaN"
    assert loss.item() == pytest.approx(0.0)


# ------------------------------------------------- block checkpointing ---

@pytest.mark.parametrize("chunk", [7, 1024])
def test_grad_checkpointing_matches(chunk):
    """Block-level checkpointing changes memory, never the loss or gradients."""
    ids = _batch(PRESETS["tiny"])

    model_a, _ = _tiny_model(loss_chunk_size=chunk, gradient_checkpointing=False)
    model_b = copy.deepcopy(model_a)
    model_b.cfg.gradient_checkpointing = True

    _, loss_a, _ = model_a(ids, targets=ids)
    loss_a.backward()
    _, loss_b, _ = model_b(ids, targets=ids)
    loss_b.backward()

    torch.testing.assert_close(loss_b, loss_a, rtol=1e-5, atol=1e-6)

    grads_a = {n: p.grad for n, p in model_a.named_parameters() if p.grad is not None}
    grads_b = {n: p.grad for n, p in model_b.named_parameters() if p.grad is not None}
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        torch.testing.assert_close(
            grads_b[name], grads_a[name], rtol=1e-4, atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch for {n}:\n{m}",
        )


def test_grad_checkpointing_skipped_in_eval_and_no_grad():
    """Checkpointing must not engage for inference or incremental decoding.

    Recomputation is pointless without a backward pass, and `checkpoint` under
    `no_grad` would silently waste a forward.
    """
    model, cfg = _tiny_model(gradient_checkpointing=True)
    ids = _batch(cfg)

    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(ids)
    assert loss is None and logits.shape == (2, 64, cfg.vocab_size)

    out = model.generate(ids[:, :4], max_new_tokens=3)
    assert out.shape[1] == 7


def test_chunking_disabled_falls_back():
    """loss_chunk_size=0 restores the single-shot path, logits included."""
    model, cfg = _tiny_model(loss_chunk_size=0)
    ids = _batch(cfg)

    logits, loss, _ = model(ids, targets=ids)
    assert logits is not None, "chunking disabled must fall back to full logits"
    assert torch.isfinite(loss)


# --- z-loss follows the cross-entropy mask ---------------------------------
#
# z-loss used to be averaged over EVERY position (z_sum / n_tokens), including
# ones the cross-entropy ignores. Pretraining never sets a -100 target, so
# this was invisible there -- and it stays exactly invisible, which is what
# the first test pins. It matters for `post`: an SFT batch is mostly prompt
# and padding, and regularising log Z at padding positions penalises logits at
# slots carrying neither supervision nor meaning.

def _manual_loss(model, x, targets, z_over_all_positions):
    """Reference loss computed outside the model, with the z-loss denominator
    chosen by the caller."""
    import torch.nn.functional as F
    with torch.no_grad():
        logits, _, _ = model(x, return_logits=True)
    lg = logits[:, :-1].float()
    tgt = targets[:, 1:].reshape(-1)
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt, ignore_index=-100)
    lse = torch.logsumexp(lg, dim=-1).reshape(-1)
    if z_over_all_positions:
        z = (lse ** 2).mean()
    else:
        mask = (tgt != -100).to(lse.dtype)
        z = ((lse ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return ce + model.cfg.z_loss * z


def test_zloss_masking_is_a_noop_for_pretraining():
    """No target is ever -100 in pretraining, so masked and unmasked z-loss
    must agree bit for bit. This is what makes the change safe for sweep,
    abl-arch and hero."""
    model, cfg = _tiny_model()
    model.eval()
    x = torch.randint(1, cfg.vocab_size, (3, 24))
    with torch.no_grad():
        _, actual, _ = model(x, targets=x)
    old = _manual_loss(model, x, x, z_over_all_positions=True)
    new = _manual_loss(model, x, x, z_over_all_positions=False)
    assert torch.equal(old, new)
    assert torch.allclose(actual, new, atol=1e-6)


def test_zloss_ignores_masked_positions_under_sft():
    model, cfg = _tiny_model()
    model.eval()
    x = torch.randint(1, cfg.vocab_size, (3, 24))
    y = x.clone()
    y[:, :12] = -100  # prompt/padding half

    with torch.no_grad():
        _, actual, _ = model(x, targets=y)
    masked = _manual_loss(model, x, y, z_over_all_positions=False)
    unmasked = _manual_loss(model, x, y, z_over_all_positions=True)

    # The model computes the masked form.
    assert torch.allclose(actual, masked, atol=1e-6)

    # Non-vacuity, asserted on the z term itself rather than on the total.
    # At random init log Z is nearly uniform across positions, so masking
    # moves the *total* loss by only ~6e-7 -- inside torch.allclose's default
    # rtol, which is why comparing totals here would have looked equal and
    # passed for the wrong reason. The gap grows as the model sharpens and as
    # the padded fraction of an SFT batch rises.
    import torch.nn.functional as F
    with torch.no_grad():
        logits, _, _ = model(x, return_logits=True)
    lse = torch.logsumexp(logits[:, :-1].float(), dim=-1).reshape(-1)
    tgt = y[:, 1:].reshape(-1)
    m = (tgt != -100).to(lse.dtype)
    z_masked = ((lse ** 2) * m).sum() / m.sum()
    z_all = (lse ** 2).mean()
    assert z_masked.item() != z_all.item()
    assert masked.item() != unmasked.item()


def test_all_masked_batch_is_exactly_zero():
    """Used by the SFT loop as the cheapest proof that `targets`, not
    `input_ids`, reaches the loss. With z-loss over all positions this
    returned a small non-zero number instead."""
    model, cfg = _tiny_model()
    model.eval()
    x = torch.randint(1, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        _, loss, _ = model(x, targets=torch.full_like(x, -100))
    assert loss.item() == 0.0
