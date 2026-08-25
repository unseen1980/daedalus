# Phase 7 corpus and mixture turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

## Read this first: the target changed

The operator intends a successor at **500M parameters over 1T tokens**, against
V1's 150M over 59.9B. Timing is undecided; the analysis does not depend on it.

That makes **unique data, not compute, the likely binding constraint**, and it
makes this phase the most valuable one remaining. V1's corpus of roughly 60B
tokens implies about **17 epochs** to reach 1T, against the plan's bar of no
source exceeding four. Clearing that bar needs on the order of **250B unique
tokens**, roughly four times what exists today.

So the headline deliverable is not a mixture. It is an honest answer to: how
many unique tokens can this pipeline actually produce, and what is the shortfall
against 1T at four epochs? Report the number even if it is unwelcome. Discovering
a shortfall here costs nothing; discovering it partway through a rented
multi-GPU run costs thousands.

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
