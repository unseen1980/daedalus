# The gate's headline paste, dry-run against the real schemas

*2026-08-11 ~02:35Z, while arm 2 trains. No GPU used.*

At ~07:47Z the gate's `### abl-arch result` section is filled by pasting

```
python scripts/abl_table.py --results runs/abl-arch/results.json
```

Three programs have to agree for that paste to say anything, and they are
written by three different scripts running unattended hours apart. This checks
they do, before the paste is the only thing standing between the operator and a
$59.85 approval.

## What was checked

`runs/abl-arch/results.json` does not exist yet — arm 2 is at 63%. So the check
drives `abl_table.py` with a **synthetic** results.json built from the *real*
`runs/abl-arch/arm1-entry.json` plus a plausible dense twin, in the exact shape
`scripts/finish_dense_arm.py:209-217` writes.

**The numbers below are invented and must never be quoted as results.** Only the
plumbing is being tested.

| link | writer | reader | agree? |
|---|---|---|---|
| combined results | `finish_dense_arm.py` → `{"runs": {config: entry}}` | `abl_table.py:516` | **yes** |
| paired decode sidecar path | `rebench_arms.py` → `runs/abl-arch/decode-paired.json` | `abl_table.load_paired` reads it *by convention* beside results.json | **yes** |
| paired per-model keys | `decode_bench.bench` → `mean` / `stdev` | `abl_table.apply_paired:362-363` reads `mean` / `stdev` | **yes** |

The third is the one worth having checked. `stdev` has one `d` on both sides,
and the first attempt at this dry run wrote `stddev` in its fixture and silently
lost every sigma from the depth table — the failure a real mismatch would
produce, arriving from the fixture instead. Whether the sigma renders is not
cosmetic: it is how a reader judges whether the headline ratio is separated.

## Rendered output, both ways

Without the paired sidecar, `abl_table.py` uses each arm's own export-time
benchmark and marks the ratio indicative:

> ⚠ Each arm measured itself during its own export, ~12 h apart. Absolute
> llama-bench numbers move with box load, so this ratio is indicative; run
> `scripts/rebench_arms.py` for an alternating measurement of both arms in one
> pass.

With `decode-paired.json` present, the depth table switches to the paired
numbers, the sigmas render, and the caveat is replaced by the alternating-pass
note. That is the behaviour the ~20-minute re-bench at ~07:37Z is scheduled to
buy, and it is now known to work rather than assumed.

## Size, which is the reason this mattered today

| | chars |
|---|---:|
| paste, no paired sidecar | 2,065 |
| paste, with paired sidecar | 2,104 |
| headroom in the postable gate body | **5,863** |

The gate body is capped at 60,000 characters against GitHub's 65,536-character
hard limit (`scripts/gate_body.py`, and the finding that produced it). The paste
fits with ~3,700 characters to spare, so filling the last `<<TODO>>` cannot make
the gate unpostable. Re-run `python scripts/gate_body.py --check` after filling
it anyway — that is what the check is for.

## What this does not prove

The synthetic dense entry copies arm 1's export block, so it exercises the
*shape* of an export, not a real one. If arm 2's export fails, `entry` carries
`export_error` instead and the table degrades — that path is covered by
`tests/test_abl_table.py`, not by this dry run.
