# `dense-150m` verified trainable before the chain reaches it

`abl-arch`'s second arm has never been trained — the earlier GGUF dry run
exported it from random init. It first trains unattended at ~04:30Z, and a
non-memory failure there (optimizer split, init) would make
`preflight_batch.sh` report "no" for every candidate batch, aborting the chain
and idling the box until noticed. Checked on CPU, ~0 cost, 2026-08-09 ~20:50Z.

## Optimizer split covers both arms completely

| config | params | Muon | AdamW | uncovered |
|---|---|---|---|---|
| daedalus-150m | 160.49M | 102 tensors / 122.68M | 62 / 37.81M | none |
| dense-150m | 161.25M | 168 tensors / 129.76M | 98 / 31.49M | none |

Muon + AdamW sum exactly to the parameter count in both cases, so nothing is
silently left un-optimized or double-stepped. One real forward/backward/step
leaves every parameter finite in both arms.

## Initial loss is `hidden * init_std`, not `ln(vocab)` — expected, not a bug

Measured CE at init (z-loss zeroed): **15.20** hybrid, **12.80** dense, against
`ln(49152) = 10.80`. Fully explained in closed form:

- every residual-output projection is zero-initialized, so at step 0 each block
  is the identity and the final hidden state *is* the token embedding;
- embeddings are tied, so RMSNorm scales `E[t]` by `1/rms(E[t]) = 1/init_std`
  and the self logit becomes `||E[t]||^2 / init_std = hidden_size * init_std`.

Predicted `768 * 0.02 = 15.36` and `640 * 0.02 = 12.80`; measured 15.20 and
12.80. The model starts by confidently predicting *the current* token and
unlearns it within the first few hundred steps of a ~1e5-step run.

Two consequences worth having in hand:

1. **The arms start at different losses (15.36 vs 12.80) because they are
   different widths (768 vs 640)** — not seed noise and not an architecture
   effect. The first few hundred steps of the two loss curves are not
   comparable; the comparison is the held-out BPB at the end.
2. Not changed. It is a deliberate init ("speedrun-proven stability") meeting
   tied embeddings, it self-corrects in well under 1% of the run, and altering
   init scale on the eve of a ~$55 sequence on a hunch is the kind of unforced
   risk this project cannot absorb. Pinned by a test instead, so a later
   "fix" to the loss head cannot change it silently.

`logit_softcap` is 0.0 in both presets, so the large init logits are not
being saturated by tanh; `z_loss` at 1e-4 contributes 0.02 nats at init.
