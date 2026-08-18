# Dry-running the gate assembly, ~11 h before it has to work

2026-08-10 ~20:10 UTC. Cost: nothing — no GPU, no llama-bench, synthetic inputs.

## Why

At ~07:05Z on the 11th the chain hands me `runs/abl-arch/results.json` and I
have ~15 minutes to turn it into the `[ASK HUMAN] ready for hero` gate. That is
the first time `scripts/abl_table.py` would ever have run against a two-arm
result. A crash or a wrong branch there is discovered at exactly the moment the
box is idle and the operator is waiting.

So both branches were exercised now, on a synthetic arm 2 built from arm 1's
real `arm1-entry.json` (same schema, `val_bpb` +0.012, decode scaled 0.95/0.72/
0.48 across the three depths — the shape a dense twin should show).

## Both branches work, and they differ in the way that matters

**Without `decode-paired.json`** — the state the chain was actually in until
19:59Z today:

> ⚠ Each arm measured itself during its own export, ~12 h apart. Absolute
> llama-bench numbers move with box load, so this ratio is **indicative**; run
> `scripts/rebench_arms.py` for an alternating measurement of both arms in one
> pass.

**With the sidecar present:**

> Both arms measured in **one alternating pass** at matched thread counts
> (`scripts/rebench_arms.py`), not in their separate export steps ~12 h apart.

The paired numbers override the per-arm ones in the table body, not just in the
prose — the depth-2048 ratio moved 2.08× → 2.09× on synthetic inputs chosen to
differ. That is the whole point of the sidecar and it is now confirmed rather
than assumed.

## What this establishes

1. `abl_table.py` runs end to end on a two-arm `results.json` and emits the full
   gate section: quality verdict, the depth table, the decision rule, and the
   Q4_0 delta caveat.
2. The "indicative" warning is real and fires on exactly the state the gate
   would have been assembled in. That is the independent confirmation that
   arming `scripts/rebench_when_quiet.sh` closed a live gap rather than a
   theoretical one — the warning is what the operator would have read.
3. The decision rule fires correctly: hybrid wins by 1.32% against a
   pre-registered 0.5% floor (`runs/preflight/abl-arch-decision-rule.md`).

**The synthetic numbers above are not predictions.** `val_bpb` +0.012 and the
decode scalings were chosen to exercise the formatting, not to guess arm 2's
result. Arm 2's real numbers land at ~06:45Z.

Isolated under `/tmp/abltable-test` with its own tree; nothing in `runs/` was
written or read except arm 1's entry, read-only.
