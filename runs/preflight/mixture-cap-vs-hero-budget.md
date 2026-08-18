# The 4-epoch cap silently reshapes `hero`'s mixture, and a DNS blip made it worse

2026-08-10 ~00:10Z. Written before `sweep` starts, because it changes what
`dataprep` has to finish and it is an input to the `[ASK HUMAN] ready for hero`
gate.

## What happened

At 21:51Z one DNS failure killed `dclm-baseline` mid-stream:

```
'[Errno -3] Temporary failure in name resolution' thrown while requesting GET
  https://huggingface.co/.../dclm-baseline-1.0/.../shard_00000013_processed.jsonl.zst
Retrying in 1s [Retry 1/5].
FAILED source dclm-baseline: RuntimeError('Cannot send a request, as the client
  has been closed.'); 1,570,031,774 tokens already flushed will still be manifested
```

It is the only network failure in the whole run (`grep -c "name resolution"` = 1).
dclm stopped at **1.57B of its approved 2.05B budget**. The shards and the
resume position both survived — `data/shards/dclm-baseline/manifest.json` carries
`stream_state.shard_idx=13, shard_example_idx=79535, n_seen=1,234,191` — so the
remainder resumes in O(1) rather than re-streaming 1.23M documents. The issue #3
within-source checkpoint did its job on a failure mode it was not written for.

The retry logic deserves a note: it *did* retry (`Retry 1/5`), but by then the
`httpx` client had been closed, so all five retries failed the same way and the
source was marked FAILED. A transient DNS blip should not be able to permanently
kill a source. Not fixed here — the top-up below is the repair for this run, and
the resume path makes the cost bounded — but it is a real robustness gap and
belongs on the list.

## Why 480M missing tokens cost far more than 480M tokens

`MixtureBatchSource` samples at the blueprint `MIXTURE` shares, clamped by
`cap_weights_by_epochs` (`train.py:316`) so no source is repeated more than
4 epochs. The cap is `4 x on_disk / total_run_tokens`. So a source's *effective
share* depends on the size of the run, and freed mass is water-filled onto
whatever else has headroom.

Measured with the real capping code against the real manifests
(`fineweb-edu` counted at its 3.75B finish):

| corpus state | hero 20B | hero 30B | hero 40B | web% @40B |
|---|---|---|---|---|
| dclm 1.57B (as the DNS blip left it) | L1 3.98 | L1 7.12 | **L1 17.59** | 53.20 |
| dclm 2.05B (the approved budget) | L1 3.98 | L1 3.99 | L1 7.99 | 58.00 |
| dclm 2.25B (what is now running) | L1 3.98 | L1 3.99 | **L1 3.99** | 60.74 |

`L1` is the total absolute deviation from the target mixture, in percentage
points. Target is web 60.00, fineweb-edu 37.50, dclm 22.50, code 9.00.

Left alone, a 40B `hero` would have trained on **53.2% web instead of 60%**, with
the missing 6.8 points pushed onto code (11.40% vs 9.00%) and finephrase (8.87%
vs 7.00%). Nothing would have raised. The operator's standing note is that
"corpus mixture balance matters more than corpus size", and this is exactly that
failure, arriving quietly.

Note the run-size dependence: at 20B or 30B tokens the cap barely binds and the
corpus is fine as-is. **This is a 40B-specific problem.** Reducing `hero`'s token
budget is an alternative repair to building more data, and a cheaper one — that
belongs in the gate issue, not in a unilateral decision here.

## What is running

`scripts/topup_dclm_then_chain.sh`, launched 00:09Z as PID 283789. It waits for
the live `dataprep` to exit, runs a **dclm-only top-up to 2.25B** with one worker
(1 x 8.0 GB + ~1 GB parent ≈ 9 GB, well inside ADDENDUM 2's 20 GB ceiling), then
`exec`s the existing `/tmp/chain-live3.sh`.

Sequential, not raced. The chain fires `sweep` after seeing no dataprep for
3 x 60s; slipping a top-up into that window risks losing the race and putting
`dataprep` and a training probe on the box together — measured to drive available
memory from 18 GB to 4.65 GB, through dataprep's 6.0 GB floor and into a repeat of
the 2026-08-08 wedge. So the old chain was killed and the top-up now owns the
handoff. It refuses to start if any training job is already running, and skips
the top-up (going straight to the chain) if under 12 GB is available.

Cost: ~700M tokens at the measured ~180K tok/s tail rate ≈ **~65 min ≈ $0.49**.
It delays `abl-arch`'s finish from ~06:00Z to ~07:05Z on the 11th.

## The +200M deviation, stated plainly

The approved plan said `dclm-baseline +2.05B`. This builds **2.25B**, +200M over
it, ~20 min and ~$0.15 of the cost above.

A source escapes the cap at a T-token run iff `4 x disk >= share x T`. At T=40B
that threshold is exactly `0.225 x 40e9 / 4 = 2.25B` for dclm. Every other source
already clears its own bar:

| source | needs @40B | has |
|---|---|---|
| fineweb-edu | 3.75B | 3.75B (exactly) |
| dclm-baseline | 2.25B | 1.57B → **2.25B** |
| stack-edu-python | 0.90B | 1.21B |
| finepdfs-edu | 0.80B | 0.88B |
| finephrase | 0.70B | 2.07B |
| cosmopedia-v2 | 0.50B | 0.95B |
| finemath / infiwebmath | 0.30B each | 1.35B each |
| finewiki-en | 0.30B | 0.41B |
| everyday-conversations | 0.20B | 0.0004B — **permanently capped** |

So +200M makes dclm the last *avoidable* distortion to disappear. The residual
L1 3.99 is entirely `everyday-conversations`: its 2.00 points cannot be supplied
(the whole dataset is 0.4M tokens, the deviation already recorded in STATUS.md),
and redistributing them costs another 1.99 across the sources that absorb them.
**3.99 is the floor**, not a number that more data can improve.

Resulting mixture at a 40B `hero`, corpus 14.22B:

| source | disk | target% | effective% | epochs |
|---|---|---|---|---|
| fineweb-edu | 3.75B | 37.50 | 37.50 | 4.00 |
| dclm-baseline | 2.25B | 22.50 | 22.50 | 4.00 |
| stack-edu-python | 1.21B | 9.00 | 9.47 | 3.13 |
| finepdfs-edu | 0.88B | 8.00 | 8.42 | 3.83 |
| finephrase | 2.07B | 7.00 | 7.37 | 1.43 |
| cosmopedia-v2 | 0.95B | 5.00 | 5.26 | 2.22 |
| finemath-3plus | 1.35B | 3.00 | 3.16 | 0.94 |
| infiwebmath-3plus | 1.35B | 3.00 | 3.16 | 0.94 |
| finewiki-en | 0.41B | 3.00 | 3.16 | 3.08 |
| everyday-conversations | 0.0004B | 2.00 | 0.00 | 4.00 |

I made this call rather than gating it because it is $0.15 against a $94.66
balance, it is the same source the operator already approved, and it is what
makes the *approved* `hero` size actually reach the *approved* mixture. If `hero`
lands at 30B instead, dclm only needed 1.69B and the surplus is harmless — more
real data is never wasted.

## What this does not prove

- The L1 metric weights every point of every source equally. It says the mixture
  is closer to the blueprint's; it does not say by how much benchmarks improve.
  Nothing here is measured in eval terms.
- Both `fineweb-edu` and `dclm-baseline` sit at exactly 4.00 epochs at 40B. That
  is the boundary the Muennighoff et al. (2305.16264) result is quoted for, not
  comfortably inside it. At 30B they are at 3.0 and 3.0.
- The top-up worker starts with an empty near-dup filter, so the last 680M tokens
  are not deduped against the 1.57B already written for dclm. That is the
  existing behaviour at every respawn, not a new regression, but it is real.
