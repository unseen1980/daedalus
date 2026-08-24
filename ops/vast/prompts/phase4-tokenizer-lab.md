# Phase 4 tokenizer lab turn

Run non-interactively with `--permission-mode dontAsk`, Opus, and xhigh effort.

Train and compare 24,576-, 32,768-, and 40,960-token vocabularies for a future
V2. Nothing here touches the released V1 weights or Daedalus-Code: a tokenizer
cannot be transplanted into a trained model, and the deliverable is a migration
report, not a new checkpoint.

## Corpus and training

Build a deterministic, source-balanced raw-text sample with immutable row and
revision hashes, covering general text, math, technical prose, dialogue, and the
planned code-language distribution. Train byte-level BPE at each of the three
exact k-quant-friendly sizes.

Pin special-token strings and IDs, including `<|endoftext|>` at ID 0 and the
ChatML tokens. Require complete byte fallback and UTF-8 round trips; a tokenizer
that cannot round-trip arbitrary bytes is rejected before it is measured.

## Measurements

Bytes per token overall and per domain, code identifier fragmentation,
indentation and newline behaviour, special-token isolation, longest-token
pathologies, encode and decode throughput, embedding parameter count, projected
Q6_K bytes, and KV-neutral decode impact.

Then train identical tiny models under equal compute and equal *bytes* for each
tokenizer and compare held-out BPB. Compare BPB, never token-level perplexity:
perplexity per token is not comparable across vocabularies, and reporting it
would make the largest vocabulary look best by construction.

## Preregistered selection

A candidate is selected only if no domain regresses bytes per token by more than
5%, code improves or ties, tiny-model BPB improves or stays within 0.5%, and
projected embedding bytes fall materially. Write this rule down with the
measured values beside it; do not adjust it after seeing the numbers.

The expected winner is 32,768, but the measured gate decides. Record a negative
result plainly if none of the three clears the rule.

## Working rules

Modify `daedalus/data.py`, `daedalus/config.py`, and `export.py` to accept an
explicit tokenizer path while preserving the SmolLM2 defaults, so no existing
run changes behaviour. Run focused checks before every commit. All shell, test,
Git, PR, hash, phase, and log actions go through `/usr/local/bin/daedalus-approved`.
