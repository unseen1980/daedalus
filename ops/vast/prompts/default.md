# Standing engineering turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

Read the versioned plan, the current controller state in
`runs/vast-program/state.json`, the latest progress snapshot, and the nearby
owning code before acting. Implement the next preregistered slice of the active
phase as one coherent, tested behaviour.

Do not ask for routine implementation approval. Choose the smallest
evidence-backed, reversible change consistent with repository patterns, stock
llama.cpp compatibility, branch boundaries, experiment gates, and deadline
policy, then record the decision in the phase log.

On a failure, reproduce it narrowly, state a falsifiable hypothesis, add or
update the focused regression test, implement the smallest repair, and rerun the
focused check followed by the affected suite.

Run the focused checks before every commit. All shell, test, Git, PR, hash,
phase, and log actions must go through `/usr/local/bin/daedalus-approved`. Push
every tested commit to the active source branch immediately. Keep all artifact
publication private until explicit approval.

Start any phase that outruns a single turn with `--detach`, which hands it to a
controller in its own session:

```bash
daedalus-approved phase run-phase --phase <name> --estimated-hours <h> --detach \
    --log runs/<area>/<name>.log -- <command>
```

Your session runs in its own process group and the keeper kills that group when
it ends a turn, so a phase started in-session dies with the turn and takes the
trainer with it. `run-phase` refuses an undetached phase estimated at 0.25h or
more. Check on a detached phase through `runs/vast-program/state.json` and its
log; do not restart one that already holds the controller lease.

## Long runs must outlive the turn that starts them

A multi-hour training or evaluation command launched through the controller is
currently a child of this session, so it dies when the turn ends. That has
already cost one probe arm: a session exited cleanly mid-sweep, its sweep died
with it, and the next session restarted the arm from scratch beside a checkpoint
it never read.

The program's own rule is that a stopped session must not interrupt an
already-launched data or training job. Two changes make that true, and both
belong in the launcher rather than in each caller:

- Start a long phase command detached, in its own session (`setsid`), with its
  output redirected to the run directory. The controller then tracks it by its
  in-flight marker rather than by process parentage, which is what the marker
  exists for, and the keeper's supervised-job back-pressure begins working as
  designed instead of never firing.
- On relaunch, resume from the run's checkpoint when one exists rather than
  starting a fresh attempt. A restart that ignores a 600MB checkpoint beside it
  is throwing away the exact thing that makes the run restart-safe.

Phases 5 through 8 have longer runs than phase 4. Fix this before starting one.
