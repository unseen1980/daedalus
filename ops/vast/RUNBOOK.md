# Vast program runbook

The program runs unattended. This is what to do on the rare occasions it asks
for a human, written for someone who did not build it.

## What is running

| Service | Role | Restart policy |
| --- | --- | --- |
| `daedalus_progress` | Publishes the five-minute heartbeat to `vast/progress-20260824` | Restarted on unexpected exit |
| `daedalus_session_keeper` | Keeps one bounded Claude engineering session alive for the active phase | Restarted on unexpected exit; a clean stop or a hard blocker is *not* unexpected |
| `daedalus_resume` | One-shot boot resume for an interrupted training run | Runs once per boot |

`scripts/vast_program.py` owns phase state, leases, and the deadline.
`ops/vast/run-approved`, installed as `/usr/local/bin/daedalus-approved`, is the
only shell, Git, PR, and evaluation surface an engineering session may use.

## Is it healthy?

Read `STATUS.md` on the progress branch. It leads with an action banner when a
human is needed and always carries elapsed hours, hours until the finalization
window, and the deadline stage. A heartbeat older than about ten minutes means
the publisher, not the program, needs attention.

```bash
supervisorctl status
tail -20 /var/log/portal/daedalus_session_keeper.log
```

## Clearing a blocker

The keeper records a hard blocker rather than relaunching forever, and it exits
cleanly when it does. **Supervisord treats that exit as expected and will not
restart it.** So clearing a blocker is always two steps, and forgetting the
second leaves the box idle:

```bash
# 1. Fix the recorded cause, then clear the status.
daedalus-approved phase transition --phase <phase> --status running \
    --details-json '{"reason": "what the operator did"}'

# 2. Start the keeper again.
supervisorctl start daedalus_session_keeper
```

The blocker itself is in `runs/vast-program/state.json` under `details.blocker`,
with the full history in `runs/vast-program/events.jsonl`.

## Why phases are detached

An engineering session runs in its own process group so the keeper can reap the
whole tree when it ends a turn. Anything the session launches inherits that
group, so a phase started in-session dies whenever the session ends -- on a
normal exit as readily as on a timeout -- and takes the trainer with it. The
phase 4 sweep lost its second arm that way and left the box idle until the next
session noticed.

`run-phase --detach` starts the phase controller in a fresh session, so a turn
can end without reaping it, and `run-phase` now refuses an undetached phase
estimated at 0.25h or more. The detached controller takes the same single-owner
lease, so detaching never adds a second writer.

```bash
daedalus-approved phase run-phase --phase <name> --estimated-hours <h> --detach \
    --log runs/<area>/<name>.log -- <command>
```

A detached phase is orphaned to init by design; `runs/vast-program/state.json`
and `controller.lock` say whether one is still running.

## Changing repository files by hand

An engineering session is a child process of the keeper, not a supervised
service, so stopping the keeper leaves the session running. That is deliberate:
it lets an operator land a fix without killing work in flight.

```bash
supervisorctl stop daedalus_session_keeper   # the running session continues
# edit, test, and commit with path-scoped commits so a concurrent session's
# staged files are never swept into your commit:
git add -- <paths> && git commit -m "..." -- <paths>
supervisorctl start daedalus_session_keeper  # waits for the session to exit
```

The keeper waits for any running session before starting, so the last step is
safe immediately and needs no sequencing by hand.

## After changing an approved-wrapper or keeper file

```bash
bash ops/vast/install_supervisor.sh
```

A committed change to `ops/vast/run-approved` does nothing until it is
installed: sessions call `/usr/local/bin/daedalus-approved`, not the repository
copy. That mismatch has already blocked the program once.

## Failure drills

Both should pass before any phase spends real GPU hours.

```bash
# A killed session resumes with its own identifier and the failure context.
pkill -f "^/root/.local/bin/claude -p"
tail -f /var/log/portal/daedalus_session_keeper.log

# A killed keeper restarts without adding a second writer.
supervisorctl pid daedalus_session_keeper | xargs kill -9
supervisorctl status daedalus_session_keeper
```

## What this container cannot do

`unshare -n` fails with `Operation not permitted`: there is no `CAP_SYS_ADMIN`
here, so nothing may be built on network namespaces. Sandboxing has to be done
with privilege dropping and resource limits instead.

## Ending the run

The keeper opens the finalization phase on its own at T+136h and stops
supervising at T+144h. The instance is never destroyed automatically; that stays
a human decision after the reports are reviewed.
