# Phase 7 corpus and mixture turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

## Read this first: report a curve, not a single budget

The operator trains on a single RTX 5090 and has not fixed a successor size.
Measured throughput makes the reachable envelope roughly 60B to 200B tokens in a
month, depending on parameter count, so a single assumed budget would be wrong
almost however it is chosen.

Report unique-token headroom as a **curve across budgets** -- at least 30B, 60B,
100B, 200B, and 500B -- so any target can be read off later without re-running
this phase. State, for each budget, the implied epoch count per source and the
shortfall against the four-epoch bar.

This is the phase's headline deliverable. Unique data, not compute, is what
limits a successor once the budget passes a few tens of billions of tokens, and
a shortfall found here is free while the same shortfall found partway through a
month of training is not. Report the numbers even when unwelcome.

## Required work

Run the epoch-headroom simulation against a **1T-token** budget, not the plan's
original one. Report per-source unique tokens, the implied epoch count at 1T,
and the shortfall to a four-epoch ceiling. Name the sources that would have to
grow and by how much.

Freeze complete evaluation contamination indexes for every scored item and
split, closing the earlier 2,000-item HellaSwag and split-coverage gaps. Make
exact normalized hashes persistent across the whole build rather than
periodically forgotten, and expand shared near-duplicate groups across
overlapping web, educational, math, and code sources while keeping memory
bounded and recording reset and coverage statistics.

Add source revision, license, filter, document count, unique token,
duplicate-drop, and contamination-drop provenance to every manifest. Remove the
tiny everyday-conversations source from general pretraining or cap it near zero;
dialogue belongs in SFT. Build source-specific holdouts by whole document or
repository before packing.

Derive candidate mixture weights from tiny-model per-source excess loss and
compare at least a baseline, a quality-heavy, and a derived mixture under equal
compute. Select on aggregate BPB **plus domain floors**, never aggregate loss
alone, preserving a general-web backbone and explicit math and code floors.

## Acceptance

No known exact evaluation n-gram hits survive in the filtered corpus. Scored
split coverage is complete and recorded. Mixture L1 skew stays within 5 points
with no all-capped fallback. Every source and transformation reproduces from
revision-pinned manifests.

The four-epoch condition is reported against 1T rather than asserted: if the
corpus cannot support it, say so with the numbers.

## Working rules

Upload completed source shards incrementally to a private dataset repository and
delete safe local intermediates; disk is finite and this phase is the largest
consumer. Long runs must outlive the turn that starts them. Run focused checks
before every commit. All shell, test, Git, PR, hash, phase, and log actions go
through `/usr/local/bin/daedalus-approved`.
