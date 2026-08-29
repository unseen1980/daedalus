# Phase 6 stage-a: recommendation gate (stagea)

Control `a8-kv4`. Five preregistered columns, all required: BPB within 0.5%, retrieval within 2 points at every trained depth, KV bytes at or under 6,144 (preferred 4,096), stock llama.cpp export and load, and artifact size with decode inside 5%/5%.

| arm | BPB | retrieval | KV | export | decode | verdict |
| --- | :---: | :---: | :---: | :---: | :---: | :--- |
| `a8-kv4` (control) | pass | n/p | **FAIL** | pass | pass | blocked |
| `a6-kv4` | pass | n/p | pass | pass | pass | unproven |
| `a4-kv4` | pass | n/p | pass | pass | pass | unproven |
| `a8-kv2` | pass | -- | pass | -- | -- | unproven |
| `a3-kv4` | pass | **FAIL** | pass | pass | pass | blocked |
| `a6-kv2` | **FAIL** | -- | pass | -- | -- | blocked |
| `a2-kv4` | **FAIL** | -- | pass | -- | -- | blocked |
| `a4-kv2` | **FAIL** | -- | pass | -- | -- | blocked |
| `a8-kv1` | **FAIL** | -- | pass | -- | -- | blocked |
| `a3-kv2` | **FAIL** | -- | pass | -- | -- | blocked |
| `a6-kv1` | **FAIL** | -- | pass | -- | -- | blocked |
| `a2-kv2` | **FAIL** | -- | pass | -- | -- | blocked |
| `a4-kv1` | **FAIL** | -- | pass | -- | -- | blocked |
| `a3-kv1` | **FAIL** | -- | pass | -- | -- | blocked |
| `a2-kv1` | **FAIL** | -- | pass | -- | -- | blocked |

`--` is unmeasured and `n/p` is measured-without-power. Neither is a pass: an arm is recommended only when every column is demonstrated.

## Outcome

- **Pareto set: none**
- unproven: ['a6-kv4', 'a4-kv4', 'a8-kv2']
- blocked: ['a8-kv4', 'a3-kv4', 'a6-kv2', 'a2-kv4', 'a4-kv2', 'a8-kv1', 'a3-kv2', 'a6-kv1', 'a2-kv2', 'a4-kv1', 'a3-kv1', 'a2-kv1']
- verdict: `no-recommendation`

> no shape clears every preregistered column, so this phase recommends none. That is a statement about the evidence, not about the shapes: see `unproven` for the columns still to be measured.

## Why each arm landed where it did

- `a8-kv4` -- blocked: failed ['kv'] against the preregistered gate; ['retrieval'] also unproven
  - bpb: pass -- BPB +0.00% against the control, within the 0.5% floor
  - retrieval: no-power -- 2 of 8 task/depth cells could not carry the 2-point gate; retention is not demonstrated at this scale
  - kv: fail -- 8,192 KV bytes per context token, over the 6,144 ceiling
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +0.0%, decode +0.0% at depth 0 and +0.0% at 2048; the ratio to the control moves +0.0% with depth
- `a6-kv4` -- unproven: ['retrieval'] not demonstrated, so this shape cannot be recommended on the ['bpb', 'decode', 'export', 'kv'] it does clear
  - bpb: pass -- BPB +0.07% against the control, within the 0.5% floor
  - retrieval: no-power -- 2 of 8 task/depth cells could not carry the 2-point gate; retention is not demonstrated at this scale
  - kv: pass -- 6,144 KV bytes per context token, at or under the 6,144 ceiling
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +0.4%, decode -0.2% at depth 0 and +1.5% at 2048; the ratio to the control moves +1.6% with depth
- `a4-kv4` -- unproven: ['retrieval'] not demonstrated, so this shape cannot be recommended on the ['bpb', 'decode', 'export', 'kv'] it does clear
  - bpb: pass -- BPB +0.15% against the control, within the 0.5% floor
  - retrieval: no-power -- 2 of 8 task/depth cells could not carry the 2-point gate; retention is not demonstrated at this scale
  - kv: pass -- 4,096 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +0.9%, decode +1.0% at depth 0 and +11.6% at 2048; the ratio to the control moves +10.5% with depth
- `a8-kv2` -- unproven: ['retrieval', 'export', 'decode'] not demonstrated, so this shape cannot be recommended on the ['bpb', 'kv'] it does clear
  - bpb: pass -- BPB +0.44% against the control, within the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 4,096 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a3-kv4` -- blocked: failed ['retrieval'] against the preregistered gate
  - bpb: pass -- BPB +0.46% against the control, within the 0.5% floor
  - retrieval: fail -- passkey at depth 256 is 8.0 points under the control, past the 2-point gate
  - kv: pass -- 3,072 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: pass -- stock llama.cpp loaded and generated from ['gguf-q4_0']
  - decode: pass -- artifact +1.1%, decode +0.9% at depth 0 and +10.1% at 2048; the ratio to the control moves +9.1% with depth
- `a6-kv2` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +0.59% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 3,072 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a2-kv4` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +0.76% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 2,048 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a4-kv2` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +0.72% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 2,048 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a8-kv1` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +1.00% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 2,048 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a3-kv2` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +0.78% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 1,536 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a6-kv1` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +1.04% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 1,536 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a2-kv2` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +1.02% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 1,024 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a4-kv1` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +1.22% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 1,024 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a3-kv1` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +1.16% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 768 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required
- `a2-kv1` -- blocked: failed ['bpb'] against the preregistered gate; ['retrieval', 'export', 'decode'] also unproven
  - bpb: fail -- BPB +1.64% against the control, outside the 0.5% floor
  - retrieval: unmeasured -- no retrieval scorecard for this arm
  - kv: pass -- 512 KV bytes per context token, at or under the 6,144 ceiling and at or under the preferred 4,096
  - export: unmeasured -- nothing has been scored through a GGUF artifact for this arm
  - decode: unmeasured -- no decode pass measures this arm and the control together at depth [0, 2048]; absolutes from separate invocations are not comparable, so a pass containing both is required

## Caveats

- A ranking measured on 105M-parameter proxies over 101M tokens is a ranking at that scale. Depth, attention fraction and KV-head choices do not extrapolate cleanly to a materially larger successor; the deliverable is a set to sanity-check a decision, not a configuration to copy.
- Parameter and byte accounting and KV bytes per context token transfer more reliably than the quality ranking does: the first is arithmetic and the second is a deployment constraint that binds harder as the model grows.
- Apple Silicon decode is pending the Mac run. The decode column here is this box's CPU, which fixes the shape of the curve and not the number a user would feel.
