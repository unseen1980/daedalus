# Phase 6 stage-a: attention x KV-head screen (stagea)

Full-pass held-out BPB on `fineweb-edu`, 100,663,296 tokens at 2048 context. Control `a8-kv4` = 1.2616 BPB.

| arm | attn | kv | KV B/tok | KV saved | params | BPB | vs ctrl % | param surplus % | credited % | floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `a2-kv1` | 2 | 1 | 512 | +94% | 106.1M | 1.2823 | +1.64 | +1.13 | +2.03 | FAIL |
| `a3-kv1` | 3 | 1 | 768 | +91% | 105.6M | 1.2762 | +1.16 | +0.69 | +1.40 | FAIL |
| `a2-kv2` | 2 | 2 | 1024 | +88% | 106.2M | 1.2744 | +1.02 | +1.26 | +1.44 | FAIL |
| `a4-kv1` | 4 | 1 | 1024 | +88% | 105.2M | 1.2770 | +1.22 | +0.26 | +1.31 | FAIL |
| `a3-kv2` | 3 | 2 | 1536 | +81% | 105.8M | 1.2714 | +0.78 | +0.88 | +1.08 | FAIL |
| `a6-kv1` | 6 | 1 | 1536 | +81% | 104.3M | 1.2747 | +1.04 | -0.62 | +0.83 | FAIL |
| `a2-kv4` | 2 | 4 | 2048 | +75% | 106.5M | 1.2711 | +0.76 | +1.51 | +1.27 | FAIL |
| `a4-kv2` | 4 | 2 | 2048 | +75% | 105.4M | 1.2707 | +0.72 | +0.51 | +0.89 | FAIL |
| `a8-kv1` | 8 | 1 | 2048 | +75% | 103.3M | 1.2742 | +1.00 | -1.50 | +0.49 | FAIL |
| `a3-kv4` | 3 | 4 | 3072 | +62% | 106.2M | 1.2674 | +0.46 | +1.26 | +0.89 | pass |
| `a6-kv2` | 6 | 2 | 3072 | +62% | 104.6M | 1.2690 | +0.59 | -0.25 | +0.50 | FAIL |
| `a4-kv4` | 4 | 4 | 4096 | +50% | 106.0M | 1.2635 | +0.15 | +1.00 | +0.49 | pass |
| `a8-kv2` | 8 | 2 | 4096 | +50% | 103.9M | 1.2671 | +0.44 | -1.00 | +0.10 | pass |
| `a6-kv4` | 6 | 4 | 6144 | +25% | 105.4M | 1.2625 | +0.07 | +0.50 | +0.24 | pass |
| `a8-kv4` (control) | 8 | 4 | 8192 | +0% | 104.9M | 1.2616 | +0.00 | +0.00 | +0.00 | pass |

## Stage B selection

- rule: raw delta <= 0.5% of control, then Pareto frontier on (kv_bytes down, bpb down), cheapest cache first, capped at 3
- eligible: ['a6-kv4', 'a4-kv4', 'a8-kv2', 'a3-kv4']
- frontier: ['a3-kv4', 'a4-kv4', 'a6-kv4']
- **selected: ['a3-kv4', 'a4-kv4', 'a6-kv4']**
- verdict: `advance`

## Caveats

- The grid is parameter-matched only to +/-1.5%, and the residual favours attention-sparse arms: a conv block is dearer than an attention block, so cutting attention adds parameters. credited_bpb_delta_pct discounts that surplus at Chinchilla's 0.34 exponent on N, which is an upper bound and therefore conservative against exactly the arms this phase hopes to advance.
- A ranking measured on 105M-parameter proxies over 101M tokens is a ranking at that scale, and nothing here should be quoted as a property of a larger successor.
- BPB is the only measured column. Retrieval by depth, GGUF export/load and decode shape are preregistered stage-6 gates that this pass does not measure; no arm is recommended on BPB alone.
- Scored on fineweb-edu alone, the one source the arms trained on. Transfer to held-out code and other web text is unmeasured at stage A.
