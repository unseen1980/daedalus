# The upload wedge recurred, and the recovery path ran for the first time

**Date:** 2026-08-10 15:18–15:45Z, during `abl-arch` arm 1. No GPU cost.

`hero` precondition #1 is *"weights-only checkpoints upload to a Hub model repo on
a ~2 h cadence, out-of-band"*. Until today it was proven only in the **success**
case: six uploads landed, each on its first attempt. The 900 s deadline added at
11:30Z after the morning's wedge had **never fired**, so the thing that makes the
cadence survive a hang was untested in production.

It fired today, on its own, and the full cycle is now on the record.

## Timeline, measured

| time | event | evidence |
|---|---|---|
| 15:13:0xZ | trainer stages `weights-step000009265.pt` (321,035,201 B) | outbox mtime |
| 15:18:1xZ | daemon 795860 spawns pass child 862610 | `ps` etime |
| ~15:19Z | **wedge**: socket `CLOSE-WAIT` (server sent FIN), 0.3% CPU | `ss -tnp`: `CLOSE-WAIT 25 0 172.17.0.2:36350 3.168.178.31:443 fd=3` |
| 15:19–15:33Z | `wchar` moves **800 bytes per 25 s** — log lines, not payload | `/proc/862610/io` sampled 6× |
| 15:33:0xZ | deadline kills the pass at 900 s | `/tmp/abl-arch.log:1700` — *"pass exceeded 900s and was killed; payloads stay pending and the next pass retries"* |
| 15:38:1xZ | retry child 864708 starts after the 300 s interval | `ps` etime 162 s at 15:40:52Z |
| 15:40–15:41Z | **healthy**: 5 `ESTAB` sockets, `wchar` +49.6 MB in 20 s (~20 Mbit/s/stream) | `/proc/864708/io`, `ss -tnp` |
| ~15:43:30Z | **upload #7 lands**, outbox drains to empty | `/tmp/abl-arch.log:1709` — *"rolling/abl-arch-daedalus-150m/weights.pt@rolling step 9265 (321 MB)"* |

**End to end: 25 minutes from wedge to delivered**, unattended, with no
intervention and nothing lost. That is the number precondition #1 is actually
worth.

The wedge signature is identical to 11:01Z: the socket goes `CLOSE-WAIT`, the
process stops burning CPU, no bytes move, and **no exception is ever raised** —
which is why `upload_once`'s catch-everything and `watch()`'s belt-and-braces
`except` both see nothing. Only a wall-clock deadline in a killable child can see
it. That is the whole argument for `daedalus/ckpt_uploader.py:277-330`, and it now
has an execution behind it rather than only a design note.

## What it says about `hero`

**The wedge is transient, and retry is the correct response.** Every payload that
has wedged has later uploaded on a retry, including the 1,435 MB milestone. It is
not a size limit — the largest payload we have has succeeded, and today's failure
was on the smaller 321 MB one.

**Rate, honestly:** 3 wedges in 9 attempts today (11:01Z milestone, its immediate
reproduction, 15:18Z rolling) against 6 first-attempt successes — call it ~30%,
on a sample far too small to be a rate. Each costs 900 s of deadline plus 300 s of
interval = **20 minutes of added lag**, and nothing is lost: the payload stays in
the outbox and the staged file is untouched.

Priced onto `hero` (142 GPU-h, ~71 rolling uploads at the 2 h cadence): if ~30%
wedge once each, that is ~21 × 20 min of extra lag spread over 5.9 days, moving
the *effective* cadence from 2.0 h to about **2.3 h**. The quantity that matters
is how much training a lost instance would cost, and that goes from ≤2.0 h to
≤2.3 h of work. Acceptable; no change warranted.

**Deliberately not tightening the deadline.** 900 s is ~18× a healthy 321 MB
transfer, which looks wasteful, but the same constant has to cover the 1,435 MB
milestone (measured at 217 s at 53 Mbit/s) and this box's uplink has been seen at
20 Mbit/s as well as 53. A deadline tight enough to catch a 321 MB wedge quickly
is tight enough to kill a healthy milestone on a slow evening, and that failure
mode is a **livelock** — every attempt killed, nothing ever uploaded, precondition
#1 silently dead — which is far worse than 20 minutes of lag. Left alone.

## The one thing still uncovered, and it is arm 1 only

Arm 1's `train.py` started 05:13Z and holds the **pre-fix** code in memory, so its
end-of-run drain at ~16:34Z calls the *unbounded* `upload_once` synchronously. A
wedge there hangs the trainer, and `abl_arch.py` waits on the subprocess with no
timeout — the GPU idle at $10.78/day with nothing raising.

That is what `scripts/guard_exit_drain.py` (pid 800742, since 12:18:19Z) exists
for, and its four conditions must all hold at once — target reached, process
alive, GPU idle, no halt marker. If it fires, the payload stays in the outbox and
**daemon 795860 — which has the bounded code — picks it up on its next pass**, so
the kill costs nothing durable. Arm 2 and `hero` launch fresh and get
`upload_once_bounded` (`train.py:1295`) properly.

## Incidental: a second `--once` child appeared and was not a bug

At 15:30:48Z a second `ckpt_uploader ... --once` process existed alongside the
wedged one, which should be impossible — the daemon runs one pass at a time under
a blocking `subprocess.run`. It was **my own `pytest` run**, which had started at
15:27Z and spawns the module as a subprocess. Recorded because "two uploaders on
one outbox" is exactly the alarm `acquire_lock` (`ckpt_uploader.py:396`) exists to
raise, and I nearly reported a production race that was my test suite.
