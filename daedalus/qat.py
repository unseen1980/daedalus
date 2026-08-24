"""Quantization-aware training against llama.cpp's *exact* Q4_0 grid.

Q4_0 is this project's ship format (blueprint Part II: it is the only format
with automatic runtime repacking into CPU-optimal layouts, ~2.3x decode over
unpacked, and it beats Q4_K_M by 15-20% on x86 decode). Post-training RTN into
4 bits costs real quality; Google's Gemma-3-QAT playbook recovers most of it by
spending the last few percent of training with the forward pass already on the
quantized grid, and reports a **54% cut in Q4_0 perplexity damage**.

The whole idea only works if the grid we train against is the grid
`llama-quantize` will later land on. If they differ even slightly, QAT
optimizes for one lattice and shipping snaps to another, which is worse than
not doing it at all. So the reference below is transcribed from
`ggml-quants.c::quantize_row_q4_0_ref`, not approximated:

    amax/max: the element of largest magnitude, keeping its **sign**
    d  = max / -8                      (hence the negative divisor)
    id = 1/d                           computed in **fp32**, before rounding d
    q  = MIN(15, (int8_t)(x*id + 8.5)) (C cast truncates; x*id+8.5 > 0 always)
    stored scale = fp16(d), so dequant is (q - 8) * fp16(d)

Two details are easy to get wrong and both change the lattice: `d` is derived
from the *signed* absmax (so a block whose extreme value is positive gets a
negative scale), and the reciprocal is taken from the fp32 `d` while the
*stored* scale is the fp16-rounded one. `test_qat.py` cross-checks this
implementation against real `llama-quantize` output rather than against my
reading of the C.

Straight-through estimator: the forward sees `quant(w)`, the backward sees
gradient w.r.t. `w` unchanged. Implemented as `w + (quant(w) - w).detach()`.

**No imatrix.** The blueprint is explicit that imatrix quantization uses a
different scale search, which breaks grid matching -- `export.py` already
asserts `llama-quantize` is never handed an imatrix flag.

Applied via `torch.nn.utils.parametrize`, which keeps the trainable float
master weight as `...parametrizations.weight.original`. That matters for three
reasons: the optimizer keeps the *same* `Parameter` objects (so Muon/AdamW
state survives switching QAT on mid-run), the master weight stays full
precision (quantizing it in place would destroy it after one step), and
`disable_qat` restores the plain module bit-exactly so checkpoints keep their
normal `state_dict` keys.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import Tensor

QK4_0 = 32   # llama.cpp block size for Q4_0
QK8_0 = 32   # ... and for Q8_0


# `d.half().float()` is not decoration -- it is what puts the scale on
# llama.cpp's lattice, which stores `d` as fp16. Written plainly, though,
# `torch.compile` **deletes it**: inductor folds the fp32->fp16->fp32 pair away
# as a redundant round trip and the fused kernel keeps the full-precision `d`.
# Measured on this box (torch 2.12.0+cu130, sm_120): 88% of elements come back
# on a different lattice from eager, with the recovered scale equal to the fp32
# `d` rather than the fp16 one. `torch._inductor.config.emulate_precision_casts`
# does not prevent it.
#
# That matters because `train.py` compiles the model *once* at construction and
# registers these parametrizations ~90 hours later, so the entire QAT phase --
# the whole point of which is that the training grid IS the shipping grid --
# would run against a grid `llama-quantize` never lands on. The error is small
# (fp16 carries ~5e-4 relative on the scale, ~0.4% of one Q4_0 level) but it is
# in exactly the direction QAT exists to remove, and it silently voids the
# bit-exactness `test_q4_0_matches_real_llama_cpp_bit_for_bit` certifies.
#
# An opaque custom op is the fix that costs nothing: inductor cannot see inside
# it, so it cannot fold the cast, and unlike `torch.compiler.disable` it stays
# *in* the graph rather than breaking it once per parametrized module.
@torch.library.custom_op("daedalus::round_fp16", mutates_args=())
def _round_fp16_op(x: Tensor) -> Tensor:
    return x.half().float()


@_round_fp16_op.register_fake
def _(x: Tensor) -> Tensor:
    return torch.empty_like(x)


def _round_fp16(x: Tensor) -> Tensor:
    """Round to fp16 precision and back, in a way `torch.compile` respects.

    `q4_0_qdq`/`q8_0_qdq` are only ever consumed inside the straight-through
    estimator's `.detach()` (see `FakeQuant.forward`) or under `no_grad`, so
    the op needs no backward formula; detaching here states that rather than
    leaving it to a backward that would raise.
    """
    return torch.ops.daedalus.round_fp16(x.detach())


def _blocks(w: Tensor, qk: int) -> Tuple[Tensor, Tuple[int, ...]]:
    """Reshape to (..., n_blocks, qk) along the last dim."""
    if w.shape[-1] % qk != 0:
        raise ValueError(
            f"last dim {w.shape[-1]} is not divisible by the block size {qk}; "
            f"llama.cpp would fall back to a different quantization type for "
            f"this tensor, so training against a {qk}-block grid would be a lie"
        )
    return w.reshape(*w.shape[:-1], w.shape[-1] // qk, qk), w.shape


# The smallest `|d|` whose reciprocal is still a finite fp32 number. Below
# this, `1/d` overflows to inf -- and inf is not a harmless placeholder here,
# because every *exactly zero* element of the same block then computes
# `0 * inf = NaN`.
_MIN_INVERTIBLE_SCALE = 1.0 / torch.finfo(torch.float32).max


def _safe_reciprocal(d: Tensor) -> Tensor:
    """`1/d` where that is representable, 0 elsewhere, **without poisoning the
    gradient**.

    llama.cpp writes this as `const float id = d ? 1.0f/d : 0.0f;` and the
    obvious transcription is `torch.where(d != 0, 1.0 / d, 0)`. That is correct
    in the forward pass and silently catastrophic in the backward one: the
    unselected branch is still *evaluated*, so `1.0 / 0` produces `inf`, and
    `where`'s backward multiplies it by a zero mask -- `0 * inf = NaN`. The NaN
    lands on `d`, flows back to the weight, and `clip_grad_norm_` then multiplies
    **every** parameter in the model by a NaN total-norm.

    A block is all-zero exactly when a channel is dead, and dead channels do not
    stay at ~1e-11: Muon's weight decay drives them as `0.02 * 0.998^step`, which
    underflows fp32 to *exactly* 0.0 at around step 49,700. `hero` turns QAT on at
    step ~118,252. So this would have fired ~5 days and ~$55 into a 6-day run, at
    95% completion, and taken the whole run with it.

    **`d == 0` is not the whole of that story, which cost Phase 3 its first
    smoke run.** A channel on its way to zero passes through a window where the
    block absmax is denormal-small but not zero. There `d` is representable and
    `1/d` is not: it overflows fp32 to inf, and the block's exactly-zero
    elements -- of which a dying channel has many -- compute `0 * inf = NaN`.
    `floor(NaN + 8.5)` stays NaN and `clamp` propagates it, so one such block
    poisons the tensor. Measured on the released checkpoint: three FFN tensors,
    3,095 NaNs between them, with every stored weight finite.

    So the mask is "is the reciprocal representable", not "is `d` nonzero",
    which subsumes the original condition. It matches the C reference wherever
    the C reference is defined: for a denormal `d` under the flush-to-zero mode
    these kernels run in, `d ? 1.0f/d : 0.0f` takes the zero branch too. And it
    cannot move any lattice that matters -- a block reaching this branch has an
    absmax below 2.4e-38, so `fp16(d)` is 0 and the block dequantizes to zero
    whichever way `q` lands.

    Masking `d` before the division rather than after is what keeps the
    gradient clean: the division never sees a value it cannot invert, so there
    is no inf for `where`'s backward to multiply by zero.
    """
    invertible = d.abs() > _MIN_INVERTIBLE_SCALE
    safe_d = torch.where(invertible, d, torch.ones_like(d))
    return torch.where(invertible, 1.0 / safe_d, torch.zeros_like(d))


def q4_0_qdq(w: Tensor) -> Tensor:
    """Quantize-dequantize through llama.cpp's exact Q4_0 grid.

    Round-trips in fp32 regardless of the input dtype, because the reference
    computes `id` in fp32; doing it in bf16 would land on a visibly different
    lattice.
    """
    orig_dtype = w.dtype
    x, shape = _blocks(w.float(), QK4_0)

    amax_idx = x.abs().argmax(dim=-1, keepdim=True)
    signed_absmax = torch.gather(x, -1, amax_idx)          # keeps the sign
    d = signed_absmax / -8.0
    # `id` from the fp32 d, but the *stored* scale is fp16 -- see module docstring.
    inv_d = _safe_reciprocal(d)
    d_fp16 = _round_fp16(d)

    q = torch.clamp(torch.floor(x * inv_d + 8.5), 0, 15)
    out = (q - 8.0) * d_fp16
    return out.reshape(shape).to(orig_dtype)


def q8_0_qdq(w: Tensor) -> Tensor:
    """Quantize-dequantize through llama.cpp's exact Q8_0 grid.

    `quantize_row_q8_0_ref`: `d = amax/127` from the *unsigned* absmax (unlike
    Q4_0), `q = round(x/d)`, stored scale fp16.
    """
    orig_dtype = w.dtype
    x, shape = _blocks(w.float(), QK8_0)

    amax = x.abs().amax(dim=-1, keepdim=True)
    d = amax / 127.0
    inv_d = _safe_reciprocal(d)
    d_fp16 = _round_fp16(d)

    q = torch.clamp(torch.round(x * inv_d), -128, 127)
    out = q * d_fp16
    return out.reshape(shape).to(orig_dtype)


_QDQ = {"q4_0": q4_0_qdq, "q8_0": q8_0_qdq}


def master_weight(mod: nn.Module) -> Tensor:
    """The trainable float weight, whether or not QAT is currently active.

    Under `parametrize`, `mod.weight` is *recomputed on every access* and is
    already quantized -- reading it to measure quantization error would report
    ~0 by construction, and using its `id()` to detect tied tensors would never
    match. The master lives at `parametrizations.weight.original`.
    """
    if parametrize.is_parametrized(mod, "weight"):
        return mod.parametrizations.weight.original
    return mod.weight


class FakeQuant(nn.Module):
    """Straight-through fake quantization, for `parametrize`.

    Forward returns the quantized weight; backward is the identity, so the
    float master weight receives the gradient of the quantized forward. This
    is the standard STE and is what Gemma-3-QAT does.
    """

    def __init__(self, kind: str):
        super().__init__()
        if kind not in _QDQ:
            raise ValueError(f"unknown quantization kind {kind!r}; "
                             f"expected one of {sorted(_QDQ)}")
        self.kind = kind

    def forward(self, w: Tensor) -> Tensor:
        return w + (_QDQ[self.kind](w) - w).detach()


def plan_qat(model: nn.Module,
             linear_kind: str = "q4_0",
             embed_kind: Optional[str] = "q8_0") -> List[Tuple[str, nn.Module, str]]:
    """Decide which tensors QAT touches, without touching them.

    - Every `nn.Linear` in the backbone -> `linear_kind` (Q4_0). These are the
      tensors `llama-quantize` actually converts and where all the quality
      damage is.
    - `embed_tokens` / `lm_head` -> `embed_kind` (Q8_0), matching the
      blueprint's "keep token_embd/output at Q8_0". With tied embeddings these
      are the same tensor and it is registered once.
    - `nn.Conv1d` depthwise kernels and every norm/1D gain are **left alone**:
      they ship as F32 in the GGUF, so fake-quantizing them would train against
      a grid that inference never uses.

    Returned as a plan so the choice is inspectable and testable separately
    from the mutation.
    """
    if embed_kind is not None and embed_kind not in _QDQ:
        raise ValueError(f"unknown embed_kind {embed_kind!r}")
    plan: List[Tuple[str, nn.Module, str]] = []
    seen: set = set()
    for name, mod in model.named_modules():
        is_embed = name.endswith("embed_tokens") or name.endswith("lm_head")
        if isinstance(mod, nn.Embedding) or is_embed:
            if embed_kind is None or not hasattr(mod, "weight"):
                continue
            kind = embed_kind
        elif isinstance(mod, nn.Linear):
            kind = linear_kind
        else:
            continue
        # Tied embeddings make embed_tokens and lm_head the same Parameter;
        # registering twice would stack two parametrizations on one tensor.
        if id(master_weight(mod)) in seen:
            continue
        seen.add(id(master_weight(mod)))
        plan.append((name, mod, kind))
    return plan


def enable_qat(model: nn.Module, linear_kind: str = "q4_0",
               embed_kind: Optional[str] = "q8_0") -> Dict[str, str]:
    """Turn QAT on. Returns {module_name: kind} for what was registered.

    Idempotent: a module already carrying a `weight` parametrization is left
    alone, so calling this every step (as the training loop does once it
    crosses into the QAT phase) does not stack transforms.
    """
    applied: Dict[str, str] = {}
    for name, mod, kind in plan_qat(model, linear_kind, embed_kind):
        if parametrize.is_parametrized(mod, "weight"):
            continue
        parametrize.register_parametrization(mod, "weight", FakeQuant(kind))
        applied[name] = kind
    return applied


def disable_qat(model: nn.Module) -> List[str]:
    """Turn QAT off, restoring plain `nn.Parameter` weights bit-exactly.

    `leave_parametrized=False` puts the untouched float master weight back, so
    a checkpoint written after this has the model's normal `state_dict` keys
    and values -- no QAT residue in the artifact, and resume is unaffected.
    """
    removed = []
    for name, mod in model.named_modules():
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight",
                                                leave_parametrized=False)
            removed.append(name)
    return removed


def is_qat_active(model: nn.Module) -> bool:
    return any(parametrize.is_parametrized(m, "weight")
               for m in model.modules())


def qat_active_at(progress: float, qat_frac: float) -> bool:
    """Whether QAT should be on at `progress` in [0, 1] of the run.

    Blueprint: QAT over the final ~5%, i.e. `qat_frac=0.05`. `qat_frac <= 0`
    disables QAT entirely, which is the default everywhere except `hero`.
    """
    if qat_frac <= 0:
        return False
    return progress >= (1.0 - min(qat_frac, 1.0))


def grid_id(linear_kind: str = "q4_0",
            embed_kind: Optional[str] = "q8_0") -> str:
    """A short, stable name for the lattice a quantized forward is running on.

    Logged beside every quantized number so it can never be compared against a
    float one by accident. Phase 3 runs QAT from step 0, so *every* validation
    figure a recovery run produces is on this grid rather than in fp32 -- a
    distinction that is invisible in the number itself and changes what the
    number means.
    """
    return linear_kind if embed_kind is None else f"{linear_kind}/{embed_kind}"


def quantization_error(model: nn.Module, linear_kind: str = "q4_0",
                       embed_kind: Optional[str] = "q8_0") -> Dict[str, float]:
    """Mean relative error the ship-format grid would inflict right now.

    Logged during the QAT phase: `qat_rel_rmse` should fall toward zero as the
    weights settle onto the grid, which is the direct evidence QAT is doing its
    job. Cheap enough to call every few hundred steps.

    `qat_tensors` counts *tensors* on the grid and `qat_elements` counts the
    weights inside them. Both are reported because they answer different
    questions: a coverage regression (a plan that stopped matching some module
    class, or a tie that collapsed two entries into one) moves the tensor count
    while barely touching the element count.
    """
    total_sq, total_ref, elements, tensors = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for _, mod, kind in plan_qat(model, linear_kind, embed_kind):
            w = master_weight(mod).detach().float()
            err = (_QDQ[kind](w) - w)
            total_sq += float(err.pow(2).sum())
            total_ref += float(w.pow(2).sum())
            elements += w.numel()
            tensors += 1
    return {
        "qat_rel_rmse": (total_sq / total_ref) ** 0.5 if total_ref > 0 else 0.0,
        "qat_tensors": float(tensors),
        "qat_elements": float(elements),
    }


_QAT_SUFFIX = ".parametrizations.weight.original"


def strip_qat_state_dict(sd: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Rewrite a QAT-active `state_dict` into the model's normal key layout.

    Under `parametrize`, `foo.weight` disappears and `foo.parametrizations.
    weight.original` takes its place. A checkpoint written mid-QAT would
    therefore be unloadable by a plain model -- which matters because QAT turns
    on in the *last 5%* of the hero run, exactly where an interruption is most
    expensive and resume must still work.

    The mapping is total and lossless (the parametrization holds no state of
    its own), so this is preferable to toggling QAT off and on around every
    checkpoint: that would re-register ~50 modules and force a
    `torch.compile` recompile every 30 minutes.
    """
    out: Dict[str, Tensor] = {}
    for k, v in sd.items():
        out[k[:-len(_QAT_SUFFIX)] + ".weight" if k.endswith(_QAT_SUFFIX) else k] = v
    return out
