# Phase 6 stage-b: attention x KV-head screen (stageb)

Full-pass held-out BPB on `fineweb-edu`, 251,658,240 tokens at 2048 context. Control `a8-kv4` = 1.0939 BPB.

| arm | attn | kv | KV B/tok | KV saved | params | BPB | vs ctrl % | param surplus % | credited % | floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `a3-kv4` | 3 | 4 | 3072 | +62% | 162.9M | 1.0959 | +0.18 | +2.48 | +1.03 | pass |
| `a4-kv4` | 4 | 4 | 4096 | +50% | 162.1M | 1.0968 | +0.26 | +1.98 | +0.94 | pass |
| `a6-kv4` | 6 | 4 | 6144 | +25% | 160.5M | 1.0946 | +0.07 | +0.99 | +0.41 | pass |
| `a8-kv4` (control) | 8 | 4 | 8192 | +0% | 158.9M | 1.0939 | +0.00 | +0.00 | +0.00 | pass |

## Advancement

- verdict: `not-applicable`

> stage-b advances arms to no stage this module schedules, so it selects none. The admission rule here is the *next* stage's, and run over stage-b's own rows it would answer which of these arms advances to the stage they are already in. What stage-b hands on is the recommendation gate over all five preregistered columns, not an arm list.

## Caveats

- The grid is parameter-matched only to +/-2.2%, and the residual favours attention-sparse arms: a conv block is dearer than an attention block, so cutting attention adds parameters. credited_bpb_delta_pct discounts that surplus at Chinchilla's 0.34 exponent on N, which is an upper bound and therefore conservative against exactly the arms this phase hopes to advance.
- A ranking measured on 159M-parameter proxies over 252M tokens is a ranking at that scale, and nothing here should be quoted as a property of a larger successor.
- BPB is the only measured column. Retrieval by depth, GGUF export/load and decode shape are preregistered stage-6 gates that this pass does not measure; no arm is recommended on BPB alone.
- Scored on fineweb-edu alone, the one source the arms trained on. Transfer to held-out code and other web text is unmeasured at stage-b.
