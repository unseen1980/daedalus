# The $63 training run was the only critical process on this box with no owner

2026-08-11, 19:20–19:35Z. `hero` at step 17,020, 3.94B tokens, healthy throughout.

## The finding

Three things can stop `hero`. Two were covered:

| how it stops | what brings it back |
|---|---|
| `train.py` crashes | `daedalus.supervise.run_with_resume` retries from the checkpoint, up to `max_attempts=10` |
| the box reboots | supervisord starts `daedalus_resume` → `scripts/boot_resume.py` |
| **the launcher process dies, box stays up** | **nothing** |

The retry loop *is* a loop inside the launcher process, so it cannot survive that
process. And the boot service is `autorestart=unexpected` with `exitcodes=0`: it
ran once at 05:36, decided, exited 0, and supervisord has left it `EXITED` ever
since. It will not fire again until the next boot.

Read straight off the live box:

```
$ supervisorctl status
daedalus_after_hero    RUNNING   5:47:02
daedalus_conv_death    RUNNING   3:57:44
daedalus_credit_watch  RUNNING   0:42:54
daedalus_heartbeat     RUNNING   6:30:47
daedalus_resume        EXITED    Aug 11 05:36 AM
$ ps -o pid,ppid,cmd -p 1133711
   PID   PPID CMD
1133711      1 /venv/main/bin/python …          <- ppid 1: no owner
```

Every *reporting* job here has a supervisord owner. The thing being reported on
did not. That is the same defect found and fixed at 18:35Z for the credit watch
("nothing restarted it"), applied to the most expensive process on the box, and
its symptom is identical: silence, which reads exactly like health.

Cost if it fires: the GPU rents at $0.449/h with nothing on it. `HEARTBEAT.md`
would show `gpu=0%`, but the agent loop has died unnoticed once already
(issue #3), and `hero` has ~$31.90 of slack against a 5.9-day run.

## The hazard that makes the naive fix worse than the bug

"No `train.py` → resume" is wrong. `run_with_resume` backs off between attempts
to `max_backoff_sec` = 900 s, and during that sleep **a healthy run has no
`train.py` at all**. A poller that acted on the first silence would start a
second trainer next to the one the live launcher is about to start. Two of these
do not fit in 32.6 GB of VRAM, and the survivor would be writing over the other's
checkpoint.

So an absence is acted on only when both hold:

1. **The launcher is provably gone.** `supervise.supervisor_is_live` compares
   `(pid, start_ticks)` from `/proc/<pid>/stat` against the marker. A pid alone
   is not an identity — Linux reuses them within seconds — and `start_ticks`
   makes the pair unique for the life of the boot.
2. **The absence has persisted for 30 minutes.** That clears the 900 s backoff
   ceiling *plus* the ~2 min a restarted trainer spends building the model and
   the data pipeline before it writes `train.pid` (`train.py:1646`). ~13 min of
   margin, at a cost of $0.04 of extra idle time in the case it exists for.

A launcher that is **alive** while the trainer stays absent past the threshold is
a wedge, not a death: that is reported (`[STALL]` issue, once) and never acted
on.

## The live run was the one case this could not see

`hero` was launched at 10:22:07Z, hours before `supervisor_pid` existed. So the
one run that matters was the one run whose launcher liveness was unknown — and
unknown falls through to the timer, which would have relaunched beside a wedged
launcher.

Fixed by deriving it rather than guessing: **the launcher is the parent of the
running `train.py`**, which `run_with_resume` guarantees because it starts the
trainer as a direct child. Self-verifying, and it keeps the file free of any
launcher name (`tests/test_hero_gate_safety.py`).

Run against the live marker:

```
[train-watch] backfill runs/hero: launcher is pid 1133711 (parent of trainer 1133734)
supervisor_pid        : 1133711
supervisor_start_ticks: 2480689897
supervisor_is_live    : True
```

`ps` independently shows pid 1133711 as the parent of train.py 1133734. The
marker is otherwise **byte-identical** to its backup — diffed with the two new
keys removed.

The backfill refuses a **shell** parent: a trainer started by hand has no
supervisor, and recording the shell would make an unsupervised run look
supervised for as long as that shell lives — i.e. it would disable the rescue.

## What it may and may not do

It decides *when*; `scripts/boot_resume.py` decides *whether* and *what*. Every
refusal there applies unchanged — completed, halted, no command, nothing ever
trained here, the `--max-boot-resumes` cap. A run that spent all 10 attempts or
hit a watchdog halt has `completed: true` written by `mark_inflight_done`, so it
is never restarted: those are failures to diagnose, not gaps to fill.

Two settings deliberately differ from every other watch here:
`stopasgroup`/`killasgroup` are **false**, because this service can be
supervising a six-day run by proxy and `supervisorctl restart` must never be a
way to kill it. The child is also `start_new_session=True`.

## Evidence

**22 tests**, using *real* processes rather than a patched `trainer_is_live` —
the whole question is what `/proc` says about a pid. Four guards verified
load-bearing by deleting each and watching the suite fail:

| guard removed | test that fails |
|---|---|
| the launcher check (`alive = False`) | `test_a_live_launcher_is_never_relaunched_beside` |
| pid-only liveness (drop `start_ticks`) | `test_pid_reuse_cannot_pass_as_a_live_launcher` |
| the quiet threshold | `test_a_short_absence_is_a_gap_not_a_death` |
| `supervisor_pid` in `write_inflight` | `test_write_inflight_records_the_process_that_will_retry` |
| the shell-parent refusal | `test_backfill_refuses_a_shell_parent` |

`test_the_default_threshold_clears_the_backoff_ceiling` reads *both* numbers out
of the shipped code — `--quiet-min`'s default and `run_with_resume`'s
`max_backoff_sec` — so they cannot drift apart unnoticed.

`test_it_hands_the_decision_to_boot_resume_which_still_refuses` runs the real CLI
end to end: the watcher acts, and `boot_resume` refuses anyway because nothing
ever trained in that directory. Both halves have to hold — a watcher that never
acts is useless, one that acts without the refusals is dangerous.

`test_it_is_a_no_op_against_the_live_run` runs the shipped command against the
real `runs/` and asserts it never reports RELAUNCH while a trainer is alive.

**322 pass** across the eight affected suites (`train_watch`, `boot_resume`,
`supervise`, `hero`, `hero_gate_safety`, `after_hero`, `credit_watch`,
`instance_safety`).

The service was verified **by killing it**, not by reading the conf: `SIGKILL` →
supervisord restarted it within `startsecs` → it re-took the flock and resumed
polling. `hero` was untouched (step 17,300, 125,181 tok/s, 20.4 GB after).

## Interaction with the after-run chain

`scripts/after_hero.sh` waits out a quiet period of 4 × 900 s before believing a
run has ended, precisely because "no process matches" is not "the run is over".
A rescue at ~30 min lands inside that window, so the chain sees the job come back
and returns to waiting — which is the behaviour its own header already documents
for a reboot.
