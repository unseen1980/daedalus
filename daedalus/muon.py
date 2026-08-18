"""Muon: momentum + Newton-Schulz orthogonalization for 2D weight matrices.

Muon is the single largest training lever available at our scale. The rigorous
tuned-baseline study (arXiv 2509.02046, 11 optimizers x 4 scales) found matrix-
preconditioned methods top out at ~1.4x over well-tuned AdamW -- and that the
maximum occurs at ~130M params, i.e. essentially exactly Daedalus's size.

Reference: Keller Jordan, https://kellerjordan.github.io/posts/muon/
Production validation: Moonlight (3B/16B), Kimi K2 (1.04T), GLM-5.

Split rule (universal in every production Muon run): 2D hidden matrices go to
Muon; embeddings, the LM head, norms, and all 1D params go to AdamW.
"""

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonalization UV^T.

    The coefficients are tuned so the iteration converges fast in the region
    that matters rather than to machine precision -- exact orthogonality is not
    the goal, a good-enough update direction is.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D matrices. Use AdamW for everything else.

    Args:
        lr: 0.02 is the tuned starting point at 100-200M scale (sweep 0.01/0.02/0.04)
        momentum: 0.95 with nesterov; warm up 0.85 -> 0.95 over ~300 steps
        weight_decay: 0.1 -- deliberately high; weight decay is what keeps Muon
            ahead of AdamW in the heavily-overtrained regime (Moonlight finding)
        ns_steps: 5 Newton-Schulz iterations (adds <1% wall-clock at our scale)
        adjust_lr: scale the update by sqrt(max(1, fan_out/fan_in)) so one LR
            transfers across layer shapes -- this is why Muon needs little
            per-width retuning.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.1, ns_steps=5, adjust_lr=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps,
                        adjust_lr=adjust_lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, f"Muon expects 2D params, got {tuple(g.shape)}"
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                if group["adjust_lr"]:
                    upd = upd * max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"] > 0:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(upd, alpha=-lr)
        return loss


def is_conv_projection(name: str, p) -> bool:
    """Is this the `in_proj`/`out_proj` of a `ShortConv` block?

    Named rather than inlined because two callers must agree on it exactly:
    the optimizer split below, and any diagnostic that reports how many
    parameters sit in the no-decay group. `conv.conv.weight` (the depthwise
    kernel) is 3D and so is *not* matched here -- it goes to AdamW.
    """
    return p.ndim == 2 and ".conv." in name


def build_optimizers(model, muon_lr=0.02, adam_lr=3e-4, muon_wd=0.1,
                     adam_wd=0.01, betas=(0.9, 0.95), conv_proj_wd=None):
    """Split params the canonical way and return (muon, adamw).

    Embeddings and the LM head get AdamW: orthogonalizing a vocab-sized matrix
    is both expensive and wrong (its rows are lookups, not a linear map).
    Norms and any 1D tensor also go to AdamW, with weight decay disabled --
    decaying norm gains destabilizes training (SmolLM3 finding).

    `conv_proj_wd` is the conv-channel-death fix (issue #7) and is **off by
    default**: `None` reproduces the shipped single-group behaviour exactly,
    down to the optimizer `state_dict` layout. Set it (to 0.0) to put the
    `ShortConv` `in_proj`/`out_proj` matrices in a second Muon group with their
    own weight decay.

    Why it is a group split and not a second optimizer: `Muon` reads
    `weight_decay` per group, so the conv projections keep the same lr,
    momentum schedule and Newton-Schulz treatment as everything else. Only the
    decay term differs -- which is the one variable the experiment isolated.

    **Setting this changes the number of Muon param groups, so an optimizer
    `state_dict` saved with one setting will not load with the other.** That is
    deliberate and PyTorch raises on it: the fix cannot be half-applied to a run
    already in progress, it can only start a new one.
    """
    muon_params, conv_params, adam_decay, adam_nodecay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = "embed_tokens" in name or "lm_head" in name
        if p.ndim == 2 and not is_embed:
            if conv_proj_wd is not None and is_conv_projection(name, p):
                conv_params.append(p)
            else:
                muon_params.append(p)
        elif p.ndim >= 2:
            adam_decay.append(p)          # embeddings / head / depthwise kernel
        else:
            adam_nodecay.append(p)        # norms, biases

    if conv_proj_wd is None:
        muon = Muon(muon_params, lr=muon_lr, weight_decay=muon_wd)
    else:
        # A silent empty group would make the fix a no-op that looks applied --
        # the failure mode that would cost a whole run to discover.
        if not conv_params:
            raise ValueError(
                "conv_proj_wd was set but no conv projection matrices were "
                "found; the fix would silently do nothing")
        muon = Muon([{"params": muon_params, "weight_decay": muon_wd},
                     {"params": conv_params, "weight_decay": conv_proj_wd}],
                    lr=muon_lr, weight_decay=muon_wd)
    adamw = torch.optim.AdamW(
        [{"params": adam_decay, "weight_decay": adam_wd},
         {"params": adam_nodecay, "weight_decay": 0.0}],
        lr=adam_lr, betas=betas, eps=1e-10,
    )
    stats = {
        "muon_tensors": len(muon_params) + len(conv_params),
        "muon_params": sum(p.numel() for p in muon_params + conv_params),
        "adam_tensors": len(adam_decay) + len(adam_nodecay),
        "adam_params": sum(p.numel() for p in adam_decay + adam_nodecay),
        "conv_proj_wd": conv_proj_wd,
        "conv_proj_tensors": len(conv_params),
        "conv_proj_params": sum(p.numel() for p in conv_params),
    }
    return muon, adamw, stats


# ------------------------------------------------------------- schedules ----

def decay_start_step(total: int, decay_frac: float = 0.45) -> int:
    """The step at which WSD leaves the stable phase and begins decaying.

    Exists so the schedule and the milestone-checkpoint trigger cannot drift
    apart: `train.py` writes its end-of-stable-phase branch point at exactly
    this step, and a branch point taken anywhere else is not the artifact the
    model card claims it is.
    """
    return int(total * (1 - decay_frac))


def wsd_lr(step: int, total: int, warmup: int = 300, decay_frac: float = 0.45,
           floor: float = 0.0) -> float:
    """Warmup-Stable-Decay with *linear decay to zero*.

    D2Z (arXiv 2502.15938) showed linear-to-zero beats cosine-to-10%: a 610M
    model at 80 tokens/param with D2Z beat the same model at 200 tokens/param
    with a 10x-decay schedule. Decaying the last ~45% is the nanochat-retuned
    value.
    """
    if step < warmup:
        return step / max(warmup, 1)
    decay_start = decay_start_step(total, decay_frac)
    if step < decay_start:
        return 1.0
    prog = (step - decay_start) / max(total - decay_start, 1)
    return max(floor, 1.0 - prog)


def momentum_warmup(step: int, warmup: int = 300,
                    start: float = 0.85, end: float = 0.95) -> float:
    """Muon momentum warmup, straight from the speedrun record scripts."""
    if step >= warmup:
        return end
    return start + (end - start) * step / max(warmup, 1)
