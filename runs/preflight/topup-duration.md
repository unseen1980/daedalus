# The 60B corpus top-up takes ~6.8 h, not 4.2 h — and it sits on `hero`'s critical path

Measured 2026-08-10 16:20Z, while `abl-arch` arm 1 finishes. Nothing here
changes *whether* the top-up runs; issue #5 settled that. It changes the two
numbers the operator will read in the gate, and one line of the script.

## What the old estimate did

`scripts/topup_for_60b.sh` and issue #5 both priced 3.50B tokens at **"the
measured 0.83B tok/h"** → 4.2 h, ~$1.89. That rate is an **aggregate across 3–6
concurrent group workers**. The script then runs **one** worker, deliberately.
Dividing an aggregate by a rate measured at a different concurrency is the same
class of mistake as the 100 h sequential-dataprep bug: the arithmetic is fine,
the rate belongs to a different configuration.

## The measurement

Rates are recovered from the shard files themselves — every full shard is
exactly 100M tokens (200 MB of `uint16`), so consecutive `mtime`s are a
throughput trace of the 2026-08-09 build that nothing had to be instrumented to
collect. "Active" time sums only the consecutive gaps under 45 min, so a source
sitting idle behind another group does not count against it.

| source | shards in window | tokens | active | **rate** |
|---|---:|---:|---:|---:|
| fineweb-edu | 68 | 3.393B | 5.38 h | **0.629 B/h** |
| dclm-baseline | 37 | 2.250B | 3.65 h | **0.599 B/h** |
| finephrase | 7 | 0.690B | 0.95 h | 0.624 B/h |
| cosmopedia-v2 | 24 | 0.824B | 1.87 h | 0.437 B/h |
| finepdfs-edu | 23 | 0.504B | 1.92 h | **0.255 B/h** |
| finewiki-en | 48 | 0.240B | 1.53 h | **0.153 B/h** |
| stack-edu-python | (earlier build) | | | **1.211 B/h** |

Two cross-checks that the number is a *rate* and not an artefact of bucketing:

- Aggregate over the whole 13 h window was **0.62 B/h** — i.e. essentially the
  same as a single source running alone (0.629 for fineweb-edu). **Concurrency
  bought almost nothing.** Whatever the bottleneck is, it is shared; four
  workers did not deliver four times the tokens.
- Hours in which exactly one source emitted shards read 0.35–0.69 B/h, which
  brackets the per-source table above.

That first point is the load-bearing one, and it cuts both ways: it is why the
old 0.83 B/h aggregate cannot be attributed to one worker, and it is why moving
from 1 worker to 2 buys much less than 2×.

## What the top-up therefore costs

`fineweb-edu` and `dclm-baseline` share one `near_dup_group`
(`daedalus/dataprep.py:104,143,146`), so they run **in one process, serially,
at any `--max-workers`**. That group is the floor on wall-clock and no amount of
concurrency moves it.

| work | tokens | at | time |
|---|---:|---|---:|
| fineweb-edu 3.750B → 5.625B | +1.875B | 0.629 B/h | 2.98 h |
| dclm-baseline 2.250B → 3.375B | +1.125B | 0.599 B/h | 1.88 h |
| **web group (serial, irreducible)** | **+3.000B** | | **4.86 h** |
| finepdfs-edu 0.880B → 1.200B | +0.320B | 0.255 B/h | 1.25 h |
| stack-edu-python 1.211B → 1.350B | +0.139B | 1.211 B/h | 0.11 h + replay |
| finewiki-en 0.410B → 0.450B | +0.040B | 0.153 B/h | 0.26 h |

- **1 worker (as configured):** 4.86 + 1.25 + ~0.4 + 0.26 ≈ **6.8 h → ~$3.05**
- **2 workers:** the three small groups hide inside the web group's 4.86 h →
  **~5.0 h → ~$2.25**
- **4 workers:** no better than 2 — there are only four groups to top up and
  the largest is one of them.

Band **5–9 h**. The rates were measured *under contention*, so they are lower
bounds on speed; against that, resume costs are not in the table (below).

## One source resumes the slow way

Every source the top-up touches carries an O(1) saved stream position except
one:

| source | `stream_state` | resume |
|---|---|---|
| fineweb-edu | yes (epoch 0) | O(1) |
| dclm-baseline | yes | O(1) |
| finepdfs-edu | yes | O(1) |
| finewiki-en | yes | O(1) |
| **stack-edu-python** | **no** | **replays 641,000 docs** |

`stack-edu-python` completed on the first pass and so was never resumed, which
is exactly why it has no position (`data/manifest.json` carries its `n_seen`,
and `dataprep.py:1213` turns that into a `resume_skip` replay). It is a
one-off, its group is small, and `_RowStream` logs the replay explicitly so it
will not read as a hang — but it is why stack's row above says "+ replay".

## What changes

1. `scripts/topup_for_60b.sh` → **`--max-workers 2`**, saving ~1.8 h. ADDENDUM 2
   rule 5 arithmetic, stated before launching:

   ```
   ckpt uploader daemon      1.0 GB
   dataprep parent          ~1.5 GB
   2 workers x 6.0 GB cap   12.0 GB
   jupyter/portal/etc       ~0.5 GB
   -------------------------------
   total                    15.0 GB     ceiling 20.0 GB
   ```

   The 6.0 GB figure is the per-worker `RLIMIT_AS`, not an expected peak (the
   13B build ran under 4.0 GB caps), so this is the conservative direction. The
   `--min-available-gb 6.0` self-stop is unchanged and is the backstop, not the
   plan. Peak concurrency is also short-lived: the three small groups finish
   inside the first ~1.5 h and the web group then runs alone.

   Not 4 workers: there are only four groups with work, the biggest is
   irreducible, and the aggregate-vs-single measurement above says the extra
   two would buy ~nothing for 12 GB more of a 20 GB ceiling.

2. The gate stops claiming the top-up's wall-clock is free. It is free of
   *rent* — the box is already waiting for a reply — but it is **not free of
   time**: `hero` cannot start until it finishes, whenever the operator
   answers. Gate-open ~06:45Z + ~5.0 h ⇒ **`hero` cannot start before ~11:45Z
   on the 11th**, and that is now stated in the ask instead of implied away.

## Why this was worth 20 minutes

The error is small in money (~$0.36 at 2 workers) and real in sequencing: the
gate told the operator a prerequisite took 4.2 h when it takes ~5–7, in the one
document where they decide whether to spend $61.89. And the correction came
from a trace that already existed on disk — no run, no GPU, no cost.
