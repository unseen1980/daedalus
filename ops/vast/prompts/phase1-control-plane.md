# Phase 1 control-plane turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

Implement only the unattended control plane: deterministic controller state,
lease ownership, deadline gates, bounded retries, boot-resume safety, sanitized
progress publishing, draft PR/status helpers, and approved wrapper commands.
Use tests first. Do not read secret files, do not print credentials, do not touch
released artifacts, and do not push or merge the default branch.

All shell, test, Git, PR, hash, phase, and log actions must go through
`/usr/local/bin/daedalus-approved`.