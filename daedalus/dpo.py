"""Direct Preference Optimization for `post` (AGENT.md SS4: "one light DPO
round on UltraFeedback").

Rafailov et al. 2023 (arXiv 2305.18290). The pieces here are pure functions so
the parts that are easy to get subtly wrong -- the sign of the margin, which
tokens are scored, whether the reference model is really frozen -- are pinned
by tests rather than inferred from a loss curve that looks plausible either
way.

A note on memory. `sequence_logprob` materialises logits, unlike the training
path's `_chunked_loss`. That is a deliberate, bounded choice: DPO here is a
short round over chat-length sequences: at seq 1024 / batch 4 one forward's
logits are ~0.8 GB in fp32, and DPO does four of them, so ~3.2 GB against
32 GB of VRAM. It does NOT scale to pretraining lengths -- the same shape at
seq 2048 / batch 8 is ~12.9 GB -- and `dpo_batch_memory_gb` exists so a caller
can check rather than discover that at runtime.
"""
from typing import Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def sequence_logprob(model, input_ids: torch.Tensor, labels: torch.Tensor
                     ) -> torch.Tensor:
    """Summed log P(token) over supervised positions, one value per sequence.

    Uses the model's own shift convention -- `logits[:, :-1]` against
    `labels[:, 1:]`, the same pairing `Daedalus.forward` uses -- so a label
    tensor built by daedalus.chatml drops straight in. Positions marked
    IGNORE_INDEX (prompt and padding) contribute nothing, which is what makes
    this a score of *the response* rather than of the prompt the two
    candidates share.

    Returns [B]. Sum, not mean: DPO compares whole-sequence likelihoods, and
    length-normalising here would silently turn it into a different objective.
    """
    logits, _, _ = model(input_ids, return_logits=True)
    logits = logits[:, :-1].float()
    targets = labels[:, 1:]

    logprobs = torch.log_softmax(logits, dim=-1)
    mask = targets != IGNORE_INDEX
    # gather needs a real index everywhere, including at masked slots
    safe = targets.masked_fill(~mask, 0)
    picked = logprobs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(dim=-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1) -> Tuple[torch.Tensor, dict]:
    """DPO loss and its diagnostics.

        L = -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

    The bracketed term is the margin: how much more the policy prefers the
    chosen response than the reference did, minus the same for the rejected
    one. Loss falls as the margin grows.

    Computed through `logsigmoid` rather than `log(sigmoid(...))`, which
    underflows to -inf once the margin goes sufficiently negative -- precisely
    the early-training regime where a mis-signed margin would otherwise show
    up as a NaN rather than as a large loss.

    Returns (loss, metrics). `accuracy` -- the fraction of pairs the policy
    already ranks correctly relative to the reference -- is the number to
    watch: DPO's loss falls monotonically whether or not the ranking improves,
    so the loss alone does not say whether it is working.
    """
    pi_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    margin = pi_logratio - ref_logratio
    loss = -F.logsigmoid(beta * margin).mean()

    with torch.no_grad():
        metrics = {
            "dpo_loss": loss.item(),
            "margin": margin.mean().item(),
            "accuracy": (margin > 0).float().mean().item(),
            "chosen_reward": beta * (policy_chosen - ref_chosen).mean().item(),
            "rejected_reward": beta * (policy_rejected - ref_rejected).mean().item(),
        }
    return loss, metrics


def preference_metrics(model, pairs, *, pad_id: int = 0, micro_batch: int = 2,
                       device: str = "cpu") -> dict:
    """Does *this* model rank the chosen response above the rejected one?

    Deliberately not the `accuracy` `dpo_loss` returns, and the difference is
    what makes this a gate rather than a formality. That one is a *relative*
    quantity -- `(pi_c - pi_r) - (ref_c - ref_r) > 0` -- measured on the pairs
    the step just trained on. Against a reference that is a snapshot of the
    policy taken at step 0 it begins at exactly 0.0, because the two models are
    the same model and every margin is identically zero. So "did held-out
    preference accuracy improve" is answered `yes` by any policy that moved at
    all, in any direction, and phase 8's rule for keeping the DPO model instead
    of the SFT one is decided by a number that cannot say no.

    This is the absolute one: the fraction of pairs where the model itself puts
    more probability on the chosen response than on the rejected one. One model,
    no reference, so it can be read for the SFT model and the DPO model
    separately and compared -- which is the comparison the gate is written as.

    Two accuracies, because `sequence_logprob` is a sum and a sum is
    length-sensitive: every additional token subtracts from it, so a shorter
    response scores higher for being shorter. UltraFeedback's chosen responses
    are typically the longer ones, so the sum-based accuracy can sit below 0.5
    on a perfectly reasonable model. That bias is constant across before/after
    on the same pairs and cancels in the delta, which is why `accuracy` stays
    primary -- it is the quantity DPO actually optimises. `accuracy_len_norm`
    divides each side by its own supervised-token count and is the control: a
    gain in one and not the other is a length shift wearing a preference gain's
    clothes.

    `pairs` is consumed into a list, so a generator works and the *same* pairs
    are scored for both models -- scoring two models on two draws from one
    stream would compare them on different data.
    """
    from daedalus.chatml import pad_batch

    pairs = list(pairs)
    if not pairs:
        return {"n": 0, "accuracy": None, "margin": None,
                "accuracy_len_norm": None, "margin_len_norm": None}

    step = max(micro_batch, 1)
    sums = {"chosen": [], "rejected": []}
    lengths = {"chosen": [], "rejected": []}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), step):
                batch = pairs[start:start + step]
                for side in ("chosen", "rejected"):
                    ids, labels = pad_batch([p[side] for p in batch],
                                            pad_id=pad_id)
                    x = torch.tensor(ids, dtype=torch.long, device=device)
                    y = torch.tensor(labels, dtype=torch.long, device=device)
                    sums[side].append(sequence_logprob(model, x, y).cpu())
                    # `[:, 1:]` to match the shift sequence_logprob scores over,
                    # so the divisor counts exactly the positions the numerator
                    # summed and not one more.
                    lengths[side].append(
                        (y[:, 1:] != IGNORE_INDEX).sum(dim=-1).cpu())
    finally:
        model.train(was_training)

    chosen = torch.cat(sums["chosen"]).double()
    rejected = torch.cat(sums["rejected"]).double()
    # clamp(min=1): a pair with no supervised tokens on one side scores 0.0
    # there either way, and dividing it by zero would make the whole batch's
    # mean NaN rather than that one pair uninformative.
    n_chosen = torch.cat(lengths["chosen"]).double().clamp(min=1.0)
    n_rejected = torch.cat(lengths["rejected"]).double().clamp(min=1.0)

    margin = chosen - rejected
    margin_norm = chosen / n_chosen - rejected / n_rejected
    return {
        "n": len(pairs),
        "accuracy": (margin > 0).double().mean().item(),
        "margin": margin.mean().item(),
        "accuracy_len_norm": (margin_norm > 0).double().mean().item(),
        "margin_len_norm": margin_norm.mean().item(),
    }


def freeze_reference(model) -> None:
    """Put a model permanently in eval mode with grads off.

    The reference must not train and must not drift. If it did, the margin
    would be measured against a moving baseline and DPO would optimise a
    quantity that changes underneath it -- which does not raise, and does not
    look wrong in the loss.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def dpo_batch_memory_gb(batch_size: int, seq_len: int, vocab_size: int,
                        n_forwards: int = 4, bytes_per_elem: int = 4) -> float:
    """Rough peak logit memory for one DPO step.

    Four forward passes (policy/reference x chosen/rejected), though only the
    two policy ones are held for backward. Callers use this to pick a batch
    size deliberately instead of finding the ceiling by hitting it.
    """
    per_forward = batch_size * seq_len * vocab_size * bytes_per_elem
    return n_forwards * per_forward / 1e9
