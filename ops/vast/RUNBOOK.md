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

## Running a second pass beside the phase that owns the box

The schedule pairs work that does not contend for the same device -- phase 6's
evidence pass is stock `llama-cli` on the CPU and runs beside stage B's ten GPU
hours; hours 28-50 pair the corpus rebuild with QAT probes the same way. One
lease and one `phase`/`status` pair could not express that: the second job
either took the lease and was refused by the run in progress, or took no lease
and left no trace in the ledger.

`--lane <name>` gives it its own lane. A lane takes its own lease
(`controller-<name>.lock`) and records under `lanes.<name>` in the state file,
leaving the top-level phase -- which is what the progress branch, the resume
guard and the deadline check all mean by "the phase" -- to whatever owns the
box. Everything else is unchanged: the lane checks the same deadline, stops on
the same terminal state, and shows up in `STATUS.md` and in the attention
banner when it fails.

```bash
daedalus-approved phase --lane evidence run-phase --phase <name> \
    --estimated-hours <h> --detach --log runs/<area>/<name>.log -- <command>
```

`--lane` is a controller option, so it goes before `run-phase`. Use it only for
a pass that genuinely does not contend: two GPU phases in two lanes is two
trainers on one card, which nothing here prevents.

Read the state file as JSON, not with `grep`. The snapshot is written with
sorted keys, so `lanes` now sorts *before* `status` -- and
`grep '"status"' state.json | head -1` answers with some lane's status rather
than the program's. Every reader in the repository parses it properly; the trap
is only for a check typed at the prompt.

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
copy. That mismatch has already blocked the program once. The copy is
deliberate -- an approved wrapper that changed whenever a branch edited a file
would let a session widen its own permissions by editing the repository -- so
the install stays an operator step rather than something a session can do to
itself.

The same gap exists one level up, for the *code a service is running*. The
publisher and the keeper are long-lived processes, so they keep running the
module they imported at start: a committed change to `scripts/github_progress.py`
changes what the next publisher does and nothing about the one publishing now.
Once the wrapper above is installed, a session can reload either of this
program's own services (and only those):

```bash
daedalus-approved reload-service daedalus_progress
```

The same install also unblocks `branch`, which is how phase 8 starts. Everything
downstream already works from whichever source branch the checkout is on --
`commit-push` pushes `HEAD:` that branch and `pr-draft` opens from it — so the
only missing capability was getting there, and a session cannot run
`git checkout` itself:

```bash
daedalus-approved branch vast/daedalus-code-20260824   # prints the parent SHA
daedalus-approved pr-draft "Daedalus-Code" vast/daedalus-improvements-20260824
```

It moves only between the two source branches, creates only the code branch,
and refuses while a tracked file is modified — a modification follows the
checkout, which is how work meant for one branch lands in a commit on the other.

The same install also unblocks `pr-find`, which is how a session learns the pull
request number it needs for `pr-edit`. Without it, `runs/vast-program/pr-body.md`
is kept current by session after session and never applied, because the number
lives on the progress branch and guessing one is a write to somebody else's pull
request:

```bash
daedalus-approved pr-find     # "<number> open main <url>" for this branch
daedalus-approved pr-edit <number> runs/vast-program/pr-body.md
```

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
