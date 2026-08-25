# Phase 6 architecture Pareto turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

## Read this first: the target changed

The operator intends a successor at **500M parameters over 1T tokens**, against
V1's 150M over 59.9B. Generate and score candidates around **500M**, keeping the
shipped 18x768 / 6-attention / 4-KV configuration as the control so the
comparison stays anchored to something measured.

Be candid about what a proxy at this budget can and cannot say. Depth, attention
fraction, and KV-head choices measured on small models do not extrapolate
cleanly to 3.3x the parameters. The deliverable is a Pareto set to sanity-check
a decision, not a configuration to copy. Say so in the report.

Two things transfer more reliably than quality ranking and deserve weight:
parameter and byte accounting, which is arithmetic, and KV bytes per context
token, which binds harder at 500M than at 150M and is a deployment constraint
rather than a training one.

## Method

Generate stock-LFM2-compatible candidates varying depth, attention layers, KV
heads, and FFN width, checking head divisibility, quantization-friendly
dimensions, parameter counts, KV bytes per context token, and exportability
before any training.

Use successive halving. Stage A trains scaled variants across attention
fractions and KV head counts. Stage B trains the best parameter-matched
candidates for 250M tokens. Stage C trains at most two finalists, and only if
the deadline reserve and the discrimination achieved so far justify it.

Include the corrected 24x640 depth comparison at a genuinely parameter-matched
FFN dimension. The existing under-parameterized deep preset is not evidence and
must not be presented as such.

Evaluate full-pass BPB, the five-task sample where it is powered, retrieval by
depth, artifact size, KV traffic, GGUF export and load, and CPU decode shape.
Mark Apple Silicon speed as pending until the Mac run.

## Gate for a recommended successor candidate

BPB no worse than the control by more than 0.5%. Retrieval no worse by more than
2 points at any trained depth. KV bytes per context token at or under 6,144 and
preferably 4,096. Stock llama.cpp export and load succeed. Artifact size and
depth-zero decode do not erase the long-context benefit.

Select a Pareto set rather than a single quality-only winner, and state the
proxy scale beside every quality claim.

## Working rules

Extend `daedalus/config.py` with generated, validated presets rather than
hand-written names. Long runs must outlive the turn that starts them. Run
focused checks before every commit. All shell, test, Git, PR, hash, phase, and
log actions go through `/usr/local/bin/daedalus-approved`.
