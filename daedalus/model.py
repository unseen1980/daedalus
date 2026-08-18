"""Daedalus: an LFM2-class conv/attention hybrid.

Tensor names deliberately mirror HuggingFace's `Lfm2ForCausalLM` so that
(a) we can prove equivalence against their reference implementation and
(b) `convert_hf_to_gguf.py` works on our checkpoints with zero patches.

Block layout per `config.layer_types`:
    conv block:      x + ShortConv(operator_norm(x));  x + FFN(ffn_norm(x))
    attention block: x + GQA(operator_norm(x));        x + FFN(ffn_norm(x))
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import DaedalusConfig


# ------------------------------------------------------------------ norms ----

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# ------------------------------------------------------------------- rope ----

def build_rope_cache(head_dim: int, max_pos: int, theta: float, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(t, inv)                      # [T, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)          # [T, hd]
    return emb.cos(), emb.sin()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, T, hd];  cos/sin: [T, hd]
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ------------------------------------------------------------- short conv ----

class ShortConv(nn.Module):
    """LFM2's double-gated short convolution.

    B, C, x = in_proj(u).chunk(3)
    y       = C * depthwise_causal_conv(B * x)
    out     = out_proj(y)

    Cost is O(1) per token and there is no KV cache, which is why this block is
    so much cheaper than attention on a CPU.
    """

    def __init__(self, cfg: DaedalusConfig):
        super().__init__()
        h, L = cfg.hidden_size, cfg.conv_L_cache
        self.L = L
        self.in_proj = nn.Linear(h, 3 * h, bias=cfg.conv_bias)
        self.conv = nn.Conv1d(h, h, kernel_size=L, groups=h,
                              padding=0, bias=cfg.conv_bias)
        self.out_proj = nn.Linear(h, h, bias=cfg.conv_bias)

    def forward(self, u: torch.Tensor,
                state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # u: [B, T, H]
        B_, C_, x_ = self.in_proj(u).chunk(3, dim=-1)
        Bx = (B_ * x_).transpose(1, 2)                    # [B, H, T]
        if state is None:
            state = Bx.new_zeros(Bx.shape[0], Bx.shape[1], self.L - 1)
        padded = torch.cat([state, Bx], dim=-1)           # causal left-pad
        new_state = padded[..., -(self.L - 1):] if self.L > 1 else state
        y = self.conv(padded).transpose(1, 2)             # [B, T, H]
        return self.out_proj(C_ * y), new_state


# -------------------------------------------------------------- attention ----

class Attention(nn.Module):
    """GQA + QK-norm + RoPE. Order matches HF Lfm2: proj -> qk_norm -> rope."""

    def __init__(self, cfg: DaedalusConfig):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.rep = self.nh // self.nkv
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.out_proj = nn.Linear(self.nh * self.hd, h, bias=False)
        self.q_layernorm = RMSNorm(self.hd, cfg.norm_eps)
        self.k_layernorm = RMSNorm(self.hd, cfg.norm_eps)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q = self.q_layernorm(self.q_proj(x).view(B, T, self.nh, self.hd)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).view(B, T, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        off = 0 if kv_cache is None else kv_cache[0].shape[2]
        q, k = apply_rope(q, k, cos[off:off + T], sin[off:off + T])

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None and T > 1))
        y = y.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.out_proj(y), new_cache


# -------------------------------------------------------------------- ffn ----

class FeedForward(nn.Module):
    """SwiGLU: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, cfg: DaedalusConfig):
        super().__init__()
        h, ff = cfg.hidden_size, cfg.block_ff_dim
        self.w1 = nn.Linear(h, ff, bias=False)
        self.w3 = nn.Linear(h, ff, bias=False)
        self.w2 = nn.Linear(ff, h, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ------------------------------------------------------------------ block ----

class Block(nn.Module):
    def __init__(self, cfg: DaedalusConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.operator_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if layer_type == "conv":
            self.conv = ShortConv(cfg)
        else:
            self.self_attn = Attention(cfg)
        self.feed_forward = FeedForward(cfg)

    def forward(self, x, cos, sin, cache=None):
        h = self.operator_norm(x)
        if self.layer_type == "conv":
            out, new_cache = self.conv(h, cache)
        else:
            out, new_cache = self.self_attn(h, cos, sin, cache)
        x = x + out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache


# ------------------------------------------------------------------ model ----

class Daedalus(nn.Module):
    def __init__(self, cfg: DaedalusConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Block(cfg, t) for t in cfg.layer_types])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        cos, sin = build_rope_cache(cfg.head_dim,
                                    cfg.max_position_embeddings * 4,
                                    cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # zero-init every residual-output projection (speedrun-proven stability)
        for blk in self.layers:
            nn.init.zeros_(blk.feed_forward.w2.weight)
            if blk.layer_type == "conv":
                nn.init.zeros_(blk.conv.out_proj.weight)
            else:
                nn.init.zeros_(blk.self_attn.out_proj.weight)

    def _init(self, m):
        std = self.cfg.initializer_range
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, std)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, std)

    def _apply_head(self, hidden, w):
        """Project hidden states to logits and apply the soft cap, if enabled."""
        logits = F.linear(hidden, w)
        if self.cfg.logit_softcap > 0:
            cap = self.cfg.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits

    def _chunked_loss(self, hidden, targets, w):
        """Cross-entropy (+ z-loss) without materialising the full logit tensor.

        `hidden` is [N, H] and `targets` is [N], already shifted by the caller.
        Logits are produced one chunk at a time under activation checkpointing,
        so peak loss-head memory is `loss_chunk_size * vocab_size` rather than
        `N * vocab_size`. The chunk logits are freed after each forward pass and
        recomputed during backward.

        Numerically equivalent to the single-shot path: cross-entropy is summed
        over non-ignored targets and divided once at the end, and the z-loss term
        is averaged over the same non-ignored positions. An all-ignored batch
        yields 0.0 rather than NaN.

        z-loss follows the cross-entropy's mask rather than covering every
        position. For pretraining the two are identical -- no target is ever
        -100 there -- so this changes nothing about `sweep`, `abl-arch` or
        `hero`. It matters for `post`: an SFT batch is mostly prompt and
        padding, and regularising log Z at padding positions penalises logits
        at slots that carry no supervision and no meaning.

        Args:
            hidden: [N, H] final-norm hidden states for the predicting positions.
            targets: [N] int64 target ids; -100 marks ignored positions.
            w: [vocab, H] output projection weight (tied embeddings or lm_head).

        Returns:
            Scalar loss tensor.
        """
        cap = self.cfg.logit_softcap
        z_coef = self.cfg.z_loss

        def chunk_terms(h_chunk, t_chunk):
            logits = F.linear(h_chunk, w)
            if cap > 0:
                logits = cap * torch.tanh(logits / cap)
            logits = logits.float()
            ce = F.cross_entropy(logits, t_chunk, ignore_index=-100,
                                 reduction="sum")
            if z_coef > 0:
                # Same positions as the cross-entropy. Multiplying by a mask
                # rather than indexing keeps the shape static for
                # torch.compile and for the activation checkpoint.
                z_per = torch.logsumexp(logits, dim=-1) ** 2
                z = (z_per * (t_chunk != -100).to(z_per.dtype)).sum()
            else:
                z = logits.new_zeros(())
            return ce, z

        n_tokens = hidden.size(0)
        chunk = self.cfg.loss_chunk_size
        ce_sum = hidden.new_zeros((), dtype=torch.float32)
        z_sum = hidden.new_zeros((), dtype=torch.float32)

        for i in range(0, n_tokens, chunk):
            h_chunk = hidden[i:i + chunk]
            t_chunk = targets[i:i + chunk]
            if torch.is_grad_enabled() and hidden.requires_grad:
                ce, z = checkpoint(chunk_terms, h_chunk, t_chunk,
                                   use_reentrant=False)
            else:
                ce, z = chunk_terms(h_chunk, t_chunk)
            ce_sum = ce_sum + ce
            z_sum = z_sum + z

        n_valid = (targets != -100).sum().clamp(min=1)
        loss = ce_sum / n_valid
        if z_coef > 0:
            loss = loss + z_coef * (z_sum / n_valid)
        return loss

    def forward(self, input_ids, targets=None, caches=None, return_logits=None):
        """Run the model, optionally computing the training loss.

        Args:
            input_ids: [B, T] token ids.
            targets: [B, T] next-token targets, or None for pure inference.
                Position t is predicted from position t-1, matching the original
                shift (`logits[:, :-1]` against `targets[:, 1:]`).
            caches: per-layer incremental decoding caches, or None.
            return_logits: whether to materialise and return the [B, T, vocab]
                logit tensor. Defaults to None, meaning "only when no targets are
                given" — during training the fused chunked loss head is used
                instead and the returned logits are None, which is what keeps
                large batches inside 16 GB. Pass True to force the full tensor
                (needed for eval paths that inspect logits directly); this
                restores the original memory profile.

        Returns:
            (logits, loss, new_caches). `logits` is None when the chunked loss
            path is taken.
        """
        x = self.embed_tokens(input_ids)
        # The RoPE cache is built once for `max_position_embeddings * 4`.
        # Attention slices it as `cos[off:off + T]`, and a slice past the end
        # silently yields a *shorter* tensor rather than raising -- which
        # surfaces as a broadcast error from inside torch.compile's fake-tensor
        # tracing ("Attempting to broadcast a dimension of length 1024 at -2"),
        # naming neither the config nor the sequence length. Say what is
        # actually wrong instead.
        off = 0 if caches is None or caches[0] is None else caches[0][0].shape[-2]
        need = off + input_ids.shape[1]
        if need > self.rope_cos.shape[0]:
            raise ValueError(
                f"RoPE cache covers {self.rope_cos.shape[0]} positions but this "
                f"forward needs {need} (offset {off} + seq {input_ids.shape[1]}). "
                f"max_position_embeddings={self.cfg.max_position_embeddings} "
                f"gives a {self.cfg.max_position_embeddings * 4}-position cache; "
                "raise it or lower the sequence length.")
        new_caches: List = []
        # Checkpointing is a training-time memory trade only: it is skipped for
        # incremental decoding (caches present) and under torch.no_grad().
        ckpt_blocks = (self.cfg.gradient_checkpointing and self.training
                       and caches is None and torch.is_grad_enabled())
        for i, blk in enumerate(self.layers):
            c = None if caches is None else caches[i]
            if ckpt_blocks:
                x, nc = checkpoint(blk, x, self.rope_cos, self.rope_sin, c,
                                   use_reentrant=False)
            else:
                x, nc = blk(x, self.rope_cos, self.rope_sin, c)
            new_caches.append(nc)
        x = self.norm(x)

        w = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

        if return_logits is None:
            return_logits = targets is None

        if targets is not None and not return_logits and self.cfg.loss_chunk_size > 0:
            hidden = x[:, :-1].reshape(-1, x.size(-1))
            tgt = targets[:, 1:].reshape(-1)
            return None, self._chunked_loss(hidden, tgt, w), new_caches

        logits = self._apply_head(x, w)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            if self.cfg.z_loss > 0:
                tgt = targets[:, 1:].reshape(-1)
                lse = torch.logsumexp(logits[:, :-1].float(), dim=-1).reshape(-1)
                mask = (tgt != -100).to(lse.dtype)
                n_valid = mask.sum().clamp(min=1)
                loss = loss + self.cfg.z_loss * ((lse ** 2) * mask).sum() / n_valid
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.8, top_k=50):
        caches = None
        for _ in range(max_new_tokens):
            inp = idx if caches is None else idx[:, -1:]
            logits, _, caches = self(inp, caches=caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
        return n
