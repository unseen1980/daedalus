#!/bin/bash
# Measure every peer in README's bar through *our own* harness, full validation
# splits, fp32. See eval.py's HFCausalLMAdapter for why they are re-measured
# rather than quoted: the published table (MobileLLM arXiv:2402.14905 Table 3)
# comes from a different evaluation setup, and we reproduce it only to within
# 1-3 points per task -- comparable to the gaps between the peers themselves.
# A bar we measured ourselves is the only one our own number can be compared
# against honestly.
#
# Sequential on purpose: one model resident at a time keeps this inside the RAM
# budget it shares with a live dataprep (ADDENDUM 2).
set -u
cd "${DAEDALUS_ROOT:-/workspace/daedalus}"
set -a; . ./.env; set +a
mkdir -p runs/eval
# openai-community/gpt2 belongs in this loop even though `peer_table.py`
# deliberately has no token-budget entry for it (OpenAI never disclosed GPT-2's
# budget). It measured **42.2** here -- the highest of the four beatable peers,
# i.e. the number the success bar *is*. It was scored by hand on 2026-08-09 and
# never added here, so the reproduction command README quotes
# (`bash scripts/eval_peers.sh && python scripts/peer_table.py`) rebuilt every
# peer except the one that sets the bar, and `peer_table.py`'s glob quietly
# rendered the stale file alongside five fresh ones.
for M in EleutherAI/pythia-160m facebook/opt-125m EleutherAI/gpt-neo-125m \
         openai-community/gpt2 HuggingFaceTB/SmolLM2-135M \
         facebook/MobileLLM-125M; do
  SLUG=$(echo "$M" | tr '/' '-' | tr 'A-Z' 'a-z')
  echo "=== $M ==="
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$DAEDALUS_PYTHON" eval.py \
    --hf-model "$M" --task-limit 0 --device cuda --no-wandb \
    --out "runs/eval/peer-${SLUG}.json" || echo "FAILED: $M"
done
