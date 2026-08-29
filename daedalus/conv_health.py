"""Functional health of the `ShortConv` channels.

The shipped ruler for channel death measures **one tensor**: the mean `|w|` down
each column of `conv.out_proj`, called dead below 1% of the layer's own p95
(`runs/preflight/conv-channel-death.md`). It found 47.9% of `hero`'s 9,216
channels dead and it was right -- zeroing the 4,417 it flagged moved held-out
loss by exactly 0.0.

It is still the wrong instrument to select a *fix* with, and
`runs/preflight/conv-death-fix-validated.md` says why in its own words: the
criterion is about weights, and every candidate fix is a change to weight decay.
An arm can read 0.00% dead because nothing shrank rather than because nothing
died, and those two are indistinguishable to a magnitude proxy. That note bought
its way out with a matched ablation control per arm. This module makes the
instrument itself carry the property, so the ablation confirms a reading instead
of standing in for one.

**What "dead" has to mean here is multiplicative.** `ShortConv` computes
`out_proj(C * conv(B * x))`, so channel `j` reaches the residual stream through
five slices in series:

    B[j] = in_proj.weight[j]        C[j] = in_proj.weight[h+j]
    x[j] = in_proj.weight[2h+j]     k[j] = conv.weight[j]
    o[j] = out_proj.weight[:, j]

The contribution is a product, so **any one** of the five collapsing takes the
channel out, whatever the other four are doing. The shipped proxy watches only
`o[j]`, which makes it sound but incomplete: a channel whose `B` row has
collapsed contributes nothing while its `out_proj` column still looks healthy,
and the proxy scores it alive. Under the shipped constant decay that gap is
mostly harmless because decay shrinks every slice together. Under the phase 5
arms -- where decay on these projections is exactly what varies -- it is the gap
that could certify a fix that does not work.

So a channel is dead here when its **weakest** factor falls below the threshold,
each factor read relative to that factor's own p95 across the layer. Two
properties follow, and both are tested:

- restricted to `o[j]` alone this reproduces the shipped ruler exactly, so the
  phase 5 "dead fraction < 1%" gate is read on the same scale as the 47.9%;
- the flagged set is a strict **superset** of the proxy's, so nothing the
  shipped instrument catches is lost.

**A layer whose scale is undefined is refused, not scored.** `out_proj` is
zero-initialised (`model.py:204`), so at step 0 every factor p95 is 0 and every
relative scale is 0/0. Reporting that as "100% dead" or as NaN would put a
number no one computed into a gate -- the phase 4 scorer emitted a NaN BPB cell
and the aggregate absorbed it, which is the failure this refuses by
construction.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


#: Fraction of a factor's p95 below which that factor counts as collapsed.
#: Inherited from the shipped weight proxy so both rulers are read on one scale.
DEAD_THRESHOLD = 0.01

#: The five per-channel slices a `ShortConv` channel passes through, in the
#: order the forward pass uses them. Named because the report, the ablation and
#: the tests must agree on the set exactly; adding a sixth here is the only
#: place a new coupled slice needs to be declared.
FACTORS: Tuple[str, ...] = ("gate_B", "gate_C", "value_x", "kernel", "out")

#: p95 rather than max: one outlier channel must not set the bar for a layer.
#: Matches `conv-channel-death.md`, which also showed the count is insensitive
#: to the threshold here because death is binary (largest dead 2.195e-11 against
#: a smallest alive many orders above it).
_P95 = 0.95


class ChannelHealthUndefined(RuntimeError):
    """A layer's scale is degenerate, so no channel fraction exists for it.

    Raised rather than returned so a gate cannot absorb it the way an aggregate
    absorbs a NaN. `model_health(..., strict=False)` returns the per-layer
    detail instead, for diagnosis.
    """


@dataclass(frozen=True)
class LayerHealth:
    """One conv layer's channel health. Tensors are float64 and CPU-resident."""

    layer_index: int
    hidden_size: int
    defined: bool
    undefined_reason: Optional[str] = None
    #: Per-factor mean |w| for each channel, before normalisation.
    factor_scale: Optional[Dict[str, torch.Tensor]] = None
    #: p95 of each factor's scale across the layer's channels.
    factor_p95: Optional[Dict[str, float]] = None
    #: Weakest factor's relative scale, per channel. The dead criterion.
    relative: Optional[torch.Tensor] = None
    #: log of the product of all five relative factors. Ranks channels by
    #: contribution; only meaningful as an ordering, not as a magnitude.
    log_contribution: Optional[torch.Tensor] = None
    dead: Optional[torch.Tensor] = None
    proxy_dead: Optional[torch.Tensor] = None
    threshold: float = DEAD_THRESHOLD

    @property
    def dead_count(self) -> int:
        if not self.defined:
            raise ChannelHealthUndefined(
                f"layer {self.layer_index}: {self.undefined_reason}")
        return int(self.dead.sum())

    @property
    def proxy_dead_count(self) -> int:
        if not self.defined:
            raise ChannelHealthUndefined(
                f"layer {self.layer_index}: {self.undefined_reason}")
        return int(self.proxy_dead.sum())

    @property
    def dead_fraction(self) -> float:
        return self.dead_count / self.hidden_size

    def to_json(self) -> dict:
        """A JSON-safe summary. Never emits a fraction it could not compute."""
        payload = {
            "layer_index": self.layer_index,
            "hidden_size": self.hidden_size,
            "threshold": self.threshold,
            "defined": self.defined,
        }
        if not self.defined:
            payload["undefined_reason"] = self.undefined_reason
            return payload
        payload.update({
            "dead_channels": self.dead_count,
            "dead_fraction": self.dead_fraction,
            "proxy_dead_channels": self.proxy_dead_count,
            "proxy_dead_fraction": self.proxy_dead_count / self.hidden_size,
            "factor_p95": dict(self.factor_p95),
        })
        return payload


@dataclass(frozen=True)
class ModelHealth:
    """Aggregate channel health across every conv layer in a model."""

    layers: List[LayerHealth]
    threshold: float = DEAD_THRESHOLD

    @property
    def undefined_layers(self) -> List[int]:
        return [layer.layer_index for layer in self.layers if not layer.defined]

    @property
    def total_channels(self) -> int:
        return sum(layer.hidden_size for layer in self.layers)

    @property
    def dead_channels(self) -> int:
        return sum(layer.dead_count for layer in self.layers)

    @property
    def dead_fraction(self) -> float:
        total = self.total_channels
        if total == 0:
            raise ChannelHealthUndefined("model has no conv layers to score")
        return self.dead_channels / total

    @property
    def proxy_dead_channels(self) -> int:
        return sum(layer.proxy_dead_count for layer in self.layers)

    def to_json(self) -> dict:
        payload = {
            "threshold": self.threshold,
            "conv_layers": len(self.layers),
            "total_channels": self.total_channels,
            "per_layer": [layer.to_json() for layer in self.layers],
        }
        undefined = self.undefined_layers
        if undefined:
            # Same rule as the per-layer summary: the aggregate is the number a
            # gate reads, so it must be absent rather than partial.
            payload["defined"] = False
            payload["undefined_layers"] = undefined
            return payload
        payload.update({
            "defined": True,
            "dead_channels": self.dead_channels,
            "dead_fraction": self.dead_fraction,
            "proxy_dead_channels": self.proxy_dead_channels,
            "proxy_dead_fraction": self.proxy_dead_channels / self.total_channels,
        })
        return payload


def conv_layers(model: nn.Module) -> List[Tuple[int, nn.Module]]:
    """Every `ShortConv` in the model, paired with its block index.

    Indexed by *block* rather than by position among the conv layers, so a
    per-layer reading lines up with `cfg.layer_types` and with the per-layer
    table in `conv-channel-death.md`.
    """
    found: List[Tuple[int, nn.Module]] = []
    blocks = getattr(model, "layers", None)
    if blocks is None:
        return found
    for index, block in enumerate(blocks):
        conv = getattr(block, "conv", None)
        if conv is not None and hasattr(conv, "in_proj") and hasattr(conv, "out_proj"):
            found.append((index, conv))
    return found


def channel_factor_scales(layer: nn.Module) -> Dict[str, torch.Tensor]:
    """Mean `|w|` of each of the five coupled slices, per channel.

    Mean absolute rather than an L2 norm because that is what the shipped proxy
    measures; keeping the `out` factor bit-identical to it is what lets the two
    rulers be compared and what makes the superset property exact rather than
    approximate.
    """
    weight = layer.out_proj.weight
    hidden = weight.shape[0]
    in_weight = layer.in_proj.weight.detach().to(torch.float64)
    if in_weight.shape[0] != 3 * hidden:
        raise ValueError(
            f"in_proj emits {in_weight.shape[0]} rows, expected {3 * hidden} "
            f"for hidden size {hidden}")
    kernel = layer.conv.weight.detach().to(torch.float64).reshape(hidden, -1)
    return {
        # B, C, x = in_proj(u).chunk(3) -- the row order the forward pass uses.
        "gate_B": in_weight[0:hidden].abs().mean(dim=1),
        "gate_C": in_weight[hidden:2 * hidden].abs().mean(dim=1),
        "value_x": in_weight[2 * hidden:3 * hidden].abs().mean(dim=1),
        "kernel": kernel.abs().mean(dim=1),
        # Down each column: column j is what channel j writes to the residual.
        "out": weight.detach().to(torch.float64).abs().mean(dim=0),
    }


def layer_health(layer: nn.Module, layer_index: int = 0,
                 threshold: float = DEAD_THRESHOLD) -> LayerHealth:
    """Score one `ShortConv`'s channels, or refuse if its scale is degenerate."""
    scales = channel_factor_scales(layer)
    hidden = int(layer.out_proj.weight.shape[0])
    p95: Dict[str, float] = {}
    for name in FACTORS:
        value = float(torch.quantile(scales[name], _P95))
        if not (value > 0.0):
            # Zero p95 means at least 95% of the layer's channels are exactly
            # zero in this factor -- true of `out_proj` on an untrained model,
            # where `relative` would be 0/0. There is no fraction to report.
            return LayerHealth(
                layer_index=layer_index,
                hidden_size=hidden,
                defined=False,
                undefined_reason=(
                    f"factor {name!r} has p95 {value!r}, so no relative scale "
                    f"exists; an untrained model reaches this because "
                    f"conv.out_proj is zero-initialised"),
                threshold=threshold,
            )
        p95[name] = value

    relative_by_factor = {name: scales[name] / p95[name] for name in FACTORS}
    stacked = torch.stack([relative_by_factor[name] for name in FACTORS])
    # Weakest link: the product is what reaches the residual stream, so the
    # smallest factor bounds the channel however healthy the others look.
    relative = stacked.min(dim=0).values
    # Ranking only. Clamped so an exactly-zero slice sorts last instead of
    # poisoning the ordering with NaN.
    log_contribution = torch.log(stacked.clamp_min(torch.finfo(torch.float64).tiny)).sum(dim=0)
    return LayerHealth(
        layer_index=layer_index,
        hidden_size=hidden,
        defined=True,
        factor_scale=scales,
        factor_p95=p95,
        relative=relative,
        log_contribution=log_contribution,
        dead=relative < threshold,
        # The shipped ruler, restricted to the one tensor it reads.
        proxy_dead=relative_by_factor["out"] < threshold,
        threshold=threshold,
    )


def model_health(model: nn.Module, threshold: float = DEAD_THRESHOLD,
                 strict: bool = True) -> ModelHealth:
    """Score every conv layer. Raises when strict and any layer is undefined."""
    health = ModelHealth(
        layers=[layer_health(layer, index, threshold)
                for index, layer in conv_layers(model)],
        threshold=threshold,
    )
    if strict and health.undefined_layers:
        raise ChannelHealthUndefined(
            f"conv layers {health.undefined_layers} have no defined channel "
            f"scale, so no dead fraction exists for this model")
    return health


def rank_channels(health: LayerHealth, ascending: bool = True) -> torch.Tensor:
    """Channel indices ordered by contribution, weakest first by default."""
    if not health.defined:
        raise ChannelHealthUndefined(
            f"layer {health.layer_index}: {health.undefined_reason}")
    order = torch.argsort(health.log_contribution, descending=not ascending)
    return order


def weakest_alive(health: LayerHealth, k: int) -> List[int]:
    """The `k` weakest channels the instrument still calls alive.

    This is the matched control from `conv-death-fix-validated.md`, and it is
    the load-bearing measurement on a *clean* arm: a fix arm flags nothing, so
    ablating its flagged set ablates nothing and proves nothing. Ablating the
    same *number* of its weakest live channels is what shows the 0% is real
    capacity rather than a metric that never fired.
    """
    order = rank_channels(health, ascending=True)
    alive = [int(index) for index in order if not bool(health.dead[index])]
    return alive[:k]


def strongest_alive(health: LayerHealth, k: int) -> List[int]:
    """The `k` strongest channels. The other end of the ablation ladder."""
    order = rank_channels(health, ascending=False)
    alive = [int(index) for index in order if not bool(health.dead[index])]
    return alive[:k]


def dead_channels(health: LayerHealth) -> List[int]:
    """The flagged set, as indices."""
    if not health.defined:
        raise ChannelHealthUndefined(
            f"layer {health.layer_index}: {health.undefined_reason}")
    return [int(index) for index in torch.nonzero(health.dead).flatten()]


def _channel_slices(layer: nn.Module, channel: int) -> List[Tuple[torch.Tensor, tuple]]:
    """Every (tensor, index) pair that belongs to one channel.

    One list so zeroing and restoring cannot disagree about the set, which is
    the way an ablation quietly stops being an ablation.
    """
    hidden = int(layer.out_proj.weight.shape[0])
    pairs: List[Tuple[torch.Tensor, tuple]] = [
        (layer.in_proj.weight, (channel,)),
        (layer.in_proj.weight, (hidden + channel,)),
        (layer.in_proj.weight, (2 * hidden + channel,)),
        (layer.conv.weight, (channel,)),
        (layer.out_proj.weight, (slice(None), channel)),
    ]
    if getattr(layer.in_proj, "bias", None) is not None:
        for offset in (0, hidden, 2 * hidden):
            pairs.append((layer.in_proj.bias, (offset + channel,)))
    if getattr(layer.conv, "bias", None) is not None:
        pairs.append((layer.conv.bias, (channel,)))
    # out_proj.bias is per *output* unit, not per channel: no channel owns it.
    return pairs


@contextlib.contextmanager
def ablated(layer: nn.Module, channels: Sequence[int]) -> Iterator[None]:
    """Zero every slice of `channels` for the duration of the block.

    Restores on the way out, including on an exception, so a scoring pass
    cannot leave a model quietly mutilated for whatever runs next.
    """
    saved: List[Tuple[torch.Tensor, tuple, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for channel in channels:
                for tensor, index in _channel_slices(layer, int(channel)):
                    saved.append((tensor, index, tensor[index].detach().clone()))
                    tensor[index] = 0.0
        yield
    finally:
        with torch.no_grad():
            for tensor, index, original in reversed(saved):
                tensor[index] = original


@contextlib.contextmanager
def ablated_model(model: nn.Module,
                  channels_by_layer: Dict[int, Sequence[int]]) -> Iterator[None]:
    """`ablated`, across several layers of one model, keyed by block index."""
    layers = dict(conv_layers(model))
    missing = sorted(set(channels_by_layer) - set(layers))
    if missing:
        raise KeyError(f"no conv layer at block index {missing}")
    with contextlib.ExitStack() as stack:
        for index, channels in channels_by_layer.items():
            stack.enter_context(ablated(layers[index], channels))
        yield


def projection_norms(layer: nn.Module,
                     health: Optional[LayerHealth] = None) -> Dict[str, float]:
    """Mean `|w|` of the conv projections, over alive channels only.

    Phase 5 gates a candidate decay schedule on projection norms staying within
    2x the alive-channel baseline, and that comparison is only meaningful
    alive-only: a baseline arm's mean is dragged down by its own ~1,030
    near-zero columns, which flatters exactly the arm that is failing. The
    correction is called out in `conv-death-fix-validated.md`; here it is the
    default rather than a footnote.
    """
    health = health or layer_health(layer)
    if not health.defined:
        raise ChannelHealthUndefined(
            f"layer {health.layer_index}: {health.undefined_reason}")
    alive = ~health.dead
    if not bool(alive.any()):
        raise ChannelHealthUndefined(
            f"layer {health.layer_index}: every channel is dead, so there is "
            f"no alive-channel norm to report")
    scales = health.factor_scale
    return {
        "in_proj": float(torch.stack([scales["gate_B"], scales["gate_C"],
                                      scales["value_x"]])[:, alive].mean()),
        "out_proj": float(scales["out"][alive].mean()),
        "kernel": float(scales["kernel"][alive].mean()),
        "alive_channels": int(alive.sum()),
    }
