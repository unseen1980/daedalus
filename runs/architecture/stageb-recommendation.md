# Phase 6 stage-b: recommendation gate (stageb)

Control `a8-kv4`. Five preregistered columns, all required: BPB within 0.5%, retrieval within 2 points at every trained depth, KV bytes at or under 6,144 (preferred 4,096), stock llama.cpp export and load, and artifact size with decode inside 5%/5%.

| arm | BPB | retrieval | KV | export | decode | verdict |
| --- | :---: | :---: | :---: | :---: | :---: | :--- |
| `a8-kv4` (control) | pass | n/p | **FAIL** | pass | pass | blocked |
| `a6-kv4` | pass | **FAIL** | pass | pass | pass | blocked |
| `a4-kv4` | pass | **FAIL** | pass | pass | pass | blocked |
| `a3-kv4` | pass | **FAIL** | pass | pass | pass | blocked |

`--` is unmeasured and `n/p` is measured-without-power. Neither is a pass: an arm is recommended only when every column is demonstrated.

## Outcome

- **Pareto set: none**
- unproven: none
- blocked: ['a8-kv4', 'a6-kv4', 'a4-kv4', 'a3-kv4']
- verdict: `no-recommendation`

> no shape clears every preregistered column, so this phase recommends none. That is a statement about the evidence, not about the shapes: see `unproven` for the columns still to be measured.

## Why each arm landed where it did

- `a8-kv4` -- blocked: failed ['kv'] against the preregistered gate; ['retrieval'] also unproven
  - bpb: pass -- BPB +0.00% against the control, within the 0.5% floor
  - retrieval: no-power -- 2 of 8 task/depth cells could not carry the 2-point gate; retention is not demonstrated at this scale
  - kv: fail -- 8,192 KV bytes per context token, over the 6,144 ceiling
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +0.0%, decode +0.0% at depth 0 and +0.0% at 2048; the ratio to the control moves +0.0% with depth
- `a6-kv4` -- blocked: failed ['retrieval'] against the preregistered gate
  - bpb: pass -- BPB +0.07% against the control, within the 0.5% floor
  - retrieval: fail -- passkey at depth 1024 is 20.0 points under the control, past the 2-point gate
  - kv: pass -- 6,144 KV bytes per context token, at or under the 6,144 ceiling
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +0.9%, decode +0.8% at depth 0 and +4.1% at 2048; the ratio to the control moves +3.2% with depth
- `a4-kv4` -- blocked: failed ['retrieval'] against the preregistered gate
  - bpb: pass -- BPB +0.26% against the control, within the 0.5% floor
  - retrieval: fail -- passkey at depth 256 is 42.0 points under the control, past the 2-point gate
  - kv: pass -- 4,096 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +1.8%, decode +1.6% at depth 0 and +9.0% at 2048; the ratio to the control moves +7.3% with depth
- `a3-kv4` -- blocked: failed ['retrieval'] against the preregistered gate
  - bpb: pass -- BPB +0.18% against the control, within the 0.5% floor
  - retrieval: fail -- passkey at depth 256 is 38.0 points under the control, past the 2-point gate
  - kv: pass -- 3,072 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +2.2%, decode +3.4% at depth 0 and +2.3% at 2048; the ratio to the control moves -1.0% with depth

## Caveats

- A ranking measured on 159M-parameter proxies over 252M tokens is a ranking at that scale. Depth, attention fraction and KV-head choices do not extrapolate cleanly to a materially larger successor; the deliverable is a set to sanity-check a decision, not a configuration to copy.
- Parameter and byte accounting and KV bytes per context token transfer more reliably than the quality ranking does: the first is arithmetic and the second is a deployment constraint that binds harder as the model grows.
- Apple Silicon decode is pending the Mac run. The decode column here is this box's CPU, which fixes the shape of the curve and not the number a user would feel.
