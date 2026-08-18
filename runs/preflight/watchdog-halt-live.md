# The watchdog halt path, verified live with real processes

**2026-08-10 03:15Z.** The unit tests inject the runner, the sleeper and the
watchdog process, which is what makes them fast and what makes them unable to
prove the one thing that matters here: that a real `watchdog.py`, spawned the
way `abl_arch.py` and `hero.py` spawn it, actually kills a real training
process and actually stops a real supervisor from resuming it.

Both checks ran against throwaway run dirs under `/tmp` with `--status-path`
pointed at a temp file, so the repo's own `STATUS.md` and `runs/` were never
touched.

## 1. A supervised crash must not end the watch

A live "trainer" (`sleep 120`) with a pidfile and recent metrics, watched with
`--supervised`, then killed.

| check | result |
|---|---|
| still polling after 3 cycles | **yes** |
| survived the trainer's death | **yes** |
| wrote a `WATCHDOG HALT` to STATUS | **no** — correct, the supervisor owns this |
| dropped a halt marker | **no** — correct, a marker would make every blip terminal |
| stopped cleanly on SIGTERM | **yes**, rc −15 |

Output:

```
watchdog: watching /tmp/wd-live-.../runs/probe (stall threshold 30.0 min, supervised)
watchdog: crash_supervised -- training process (pid 419295) is no longer running
          and the run did not reach its token target
```

Said **once**, not once per 2-second poll — the supervisor's backoff can be 15
minutes, which is ~450 polls at `hero`'s cadence.

This is the case that was silently broken before: the watchdog used to exit
here, so a recoverable crash on day one left days two to four of `hero`
unwatched.

## 2. A divergence must be terminal, and must stick

A live trainer, a metrics file ending in a diverged loss (3.0 → 42.0), the same
spawn.

| check | result |
|---|---|
| watchdog exited on its own | **yes**, rc 0 |
| the real trainer process was killed | **yes** |
| halt marker | `divergence` — *"loss diverged at step 10: 42.0000 > 2.0x running mean 3.0000 (over last 10 points)"* |
| STATUS heading | `# WATCHDOG HALT: \`probe\`` |
| `run_with_resume` given that marker | **stopped after 1 attempt**, `halt["kind"] == "divergence"` |

The last row is the fix. Before it, that same SIGTERM exit (143) was read as a
crash and resumed — from the diverged checkpoint, with the watchdog already
gone — and on `hero` that is three more days and ~$30 spent training a model
that had already failed, finishing with exit 0 and "finished in 2 attempt(s)".

## 3. Read-only against the live `sweep`

Also run against the real `runs/sweep-wsdfix-lr0.02` while it trained, with
`do_halt=False`: pidfile held the true training PID (416312, alive), 15 metric
records, no divergence, no stall (metrics 0.42 min old against a 30 min
threshold), `check_once` returned `None` and wrote nothing.

That is the check that the halt path has something real to aim at: a watchdog
whose `read_pidfile` came back `None` would detect divergence perfectly and
then halt nothing at all.
