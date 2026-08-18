#!/bin/bash
# Find the largest micro-batch that BOTH abl-arch arms can train at, and print
# it. Exits non-zero if none of the candidates fit.
#
# Why this exists
# ---------------
# `abl-arch` trains daedalus-150m then dense-150m, ~12 h each. Only the hybrid
# has ever been benchmarked: runs/smoke/results.json measured it at 25.31 GB
# peak at batch 16, of 33.7 GB. The dense twin is param-matched but not
# activation-matched -- 24 attention layers against the hybrid's 6, and FF
# 24x2304 against 18x2048, which is ~50% more FF activation and ~33% more
# overall. Scaling the hybrid's ~23 GB of activations by that lands at ~31 GB,
# plus ~2 GB of weights and optimizer state: essentially on the OOM line,
# before torch.compile's autotune workspace (which is what OOM'd the hybrid at
# batch 24).
#
# So dense-150m at batch 16 is a coin flip, and it would land ~36 h into an
# unattended run, after the hybrid arm had already burned 12 h. abl_arch.py
# catches the failure per-config, so results.json would survive with one arm
# -- but the arms must share a micro-batch (it sets `accum`, which sets
# tokens/step, which sets the step count, and "same steps" is what makes the
# comparison publishable), so recovering means retraining BOTH. ~$11 and a day.
#
# Two minutes of measurement now instead of guessing a "safe" number, which is
# exactly the "try it and see" that ADDENDUM 2 rule 3 forbids.
#
# Uses smoke.py, which trains on synthetic random tokens -- no data pipeline,
# so it stays well clear of the system-RAM ceiling that matters on this box.
# Writes to a scratch --out so the recorded baseline in runs/smoke/results.json
# (cited by issue #1) is not clobbered.
set -u
cd "${DAEDALUS_ROOT:-$(dirname "$0")/..}"
set -a; . ./.env; set +a

CANDIDATES="${1:-16,12,8}"
CONFIGS="${2:-daedalus-150m,dense-150m}"
SEQ="${3:-2048}"
OUT_DIR=runs/preflight
mkdir -p "$OUT_DIR"

# Descending, so the first batch that clears every config wins.
for bs in $(echo "$CANDIDATES" | tr ',' ' '); do
  ok=1
  for cfg in $(echo "$CONFIGS" | tr ',' ' '); do
    out="$OUT_DIR/${cfg}-b${bs}.json"
    # A separate process per (config, batch): a CUDA OOM leaves the allocator
    # fragmented and the compile cache warm, so reusing one process would make
    # later trials read pessimistically low.
    "$DAEDALUS_PYTHON" smoke.py --config "$cfg" --batches "$bs" --seq "$SEQ" \
      --steps 5 --out "$out" >> "$OUT_DIR/preflight.log" 2>&1
    fits=$("$DAEDALUS_PYTHON" -c "
import json
try:
    r = json.load(open('$out'))['results']['$bs']
    print('yes' if 'tok_per_sec' in r else 'no')
except Exception:
    print('no')
" 2>/dev/null || echo no)
    echo "$(date -u +%FT%TZ) preflight ${cfg} batch ${bs}: ${fits}" >> "$OUT_DIR/preflight.log"
    [ "$fits" = "yes" ] || { ok=0; break; }
  done
  if [ "$ok" = "1" ]; then
    echo "$bs"
    exit 0
  fi
done

echo "$(date -u +%FT%TZ) preflight: no candidate in ${CANDIDATES} fits every config" \
  >> "$OUT_DIR/preflight.log"
exit 1
