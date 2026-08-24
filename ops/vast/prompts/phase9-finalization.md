# Phase 9 finalization turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

The reserved window has opened. No new experiment may start. Rescore, upload,
report, and leave the program auditable, then stop.

## Order of work

Re-run every headline metric from immutable final artifacts rather than from
transient rolling checkpoints. Write `runs/final/improvement-report.json` as the
machine-readable source of truth and `runs/final/improvement-report.md` for the
user, plus `runs/final/v2-recommendation.md` and
`runs/final/daedalus-code-next.md`. Record immutable artifact manifests with
SHA-256 hashes, Hub repository, revision and path, local path, config,
tokenizer, data manifest, seed, and producing commit.

Separate released-model gains, code-model gains, proxy evidence, and V2
projections. Report negative results and stopped branches as first-class
findings; a proxy result is never presented as a full-model gain.

## Final validation

Run the entire suite before finalization. Verify private Hub downloads into a
fresh temporary directory and hash-match them. Verify both source branches are
fully pushed, the progress heartbeat is current, both draft pull requests exist
with correct bases and complete descriptions, the default branch is unchanged,
no secret-like file is tracked, and `.env` remains ignored. Verify no `wip:`
recovery commit remains unresolved.

Mark a pull request ready for review only when its own focused and full tests
and artifact gates pass; otherwise leave it a draft and state the blocker
precisely. Never merge.

Finish by writing the completion or halt status with its reason and the exact
final source SHAs and pull request URLs, then transition the controller to
`complete` (or `halted` with the blocker recorded) so the keeper stops cleanly
and the heartbeat can publish the final snapshot.

## Working rules

All shell, test, Git, PR, hash, phase, and log actions must go through
`/usr/local/bin/daedalus-approved`. Keep every artifact private.
