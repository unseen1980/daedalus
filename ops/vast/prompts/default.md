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
