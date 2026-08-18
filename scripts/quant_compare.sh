#!/usr/bin/env bash
# Compare 4-bit quantisation recipes for the Daedalus instruct model, on CPU.
#
# Why this exists: the shipped Q4_0 costs ~6.2% perplexity against f16, well
# above the ~2.5% the project measured at 5B tokens. Quantisation-aware training
# was meant to close that gap but crashed and never ran, so the released model
# carries the unmitigated penalty.
#
# Two levers remain, both free and neither needing a GPU or any retraining:
#   * Q4_K_M    -- better error characteristics at the same bit width
#   * imatrix   -- weight quantisation by which channels actually matter,
#                  measured by running f16 over calibration text. export.py
#                  explicitly did "Q4_0, no imatrix", so this was never tried.
#
# Quality alone does not decide it. Q4_0 was chosen for its ARM dot-product
# kernel speed, and a quality win that costs the speed claim is not a win, so
# this measures decode throughput too.

set -euo pipefail

MODEL_DIR="${HOME}/daedalus"
WORK="${MODEL_DIR}/quant-compare"
CALIB="$(cd "$(dirname "$0")/.." && pwd)/data/eval/ppl-finewiki-150k.txt"
F16="${MODEL_DIR}/gguf/instruct-f16.gguf"
THREADS="${THREADS:-8}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:-40}"

command -v llama-quantize >/dev/null || { echo "need llama-quantize (brew install llama.cpp)"; exit 1; }
[ -f "${CALIB}" ] || { echo "calibration text missing: ${CALIB}"; exit 1; }

mkdir -p "${WORK}"

if [ ! -f "${F16}" ]; then
  echo "==> fetching f16 (323 MB, one time)"
  hf download Unseen1980/daedalus-checkpoints gguf/instruct-f16.gguf \
    --local-dir "${MODEL_DIR}"
fi

# --- build the candidates ----------------------------------------------------
# Q4_0 is rebuilt from the same f16 rather than reusing the shipped file, so
# every number below comes from one identical source.
for q in Q4_0 Q4_K_M Q5_K_M; do
  out="${WORK}/instruct-${q}.gguf"
  [ -f "${out}" ] || { echo "==> quantising ${q}"; llama-quantize "${F16}" "${out}" "${q}"; }
done

# imatrix: measure which channels matter, then quantise Q4_0 informed by it.
IMAT="${WORK}/instruct.imatrix"
if [ ! -f "${IMAT}" ]; then
  echo "==> computing importance matrix (a few minutes)"
  llama-imatrix -m "${F16}" -f "${CALIB}" -o "${IMAT}" -t "${THREADS}" --chunks 32
fi
for q in Q4_0 Q4_K_M; do
  out="${WORK}/instruct-${q}-imat.gguf"
  [ -f "${out}" ] || { echo "==> quantising ${q} + imatrix"; \
    llama-quantize --imatrix "${IMAT}" "${F16}" "${out}" "${q}"; }
done

# --- measure -----------------------------------------------------------------
echo
echo "=============================================================="
printf "%-24s %10s %10s %12s\n" "model" "size MB" "PPL" "tok/s"
echo "=============================================================="

measure() {
  local label="$1" path="$2"
  [ -f "${path}" ] || return 0
  local mb ppl tps
  mb=$(( $(stat -f%z "${path}") / 1000000 ))
  ppl=$(llama-perplexity -m "${path}" -f "${CALIB}" -c "${CTX}" --chunks "${CHUNKS}" \
        -t "${THREADS}" 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  # -ngl 0 forces CPU: the project's claim is about CPU decode, and on a Mac
  # llama.cpp silently offloads to Metal otherwise.
  tps=$(llama-bench -m "${path}" -t "${THREADS}" -ngl 0 -p 0 -n 64 2>/dev/null \
        | grep -oE "[0-9.]+ ± [0-9.]+" | tail -1 | cut -d' ' -f1)
  printf "%-24s %10s %10s %12s\n" "${label}" "${mb}" "${ppl:-?}" "${tps:-?}"
}

measure "f16 (reference)"    "${F16}"
measure "Q4_0 (shipped)"     "${WORK}/instruct-Q4_0.gguf"
measure "Q4_0 + imatrix"     "${WORK}/instruct-Q4_0-imat.gguf"
measure "Q4_K_M"             "${WORK}/instruct-Q4_K_M.gguf"
measure "Q4_K_M + imatrix"   "${WORK}/instruct-Q4_K_M-imat.gguf"
measure "Q5_K_M"             "${WORK}/instruct-Q5_K_M.gguf"

echo "=============================================================="
cat <<'NOTE'

Reading it:
  * PPL vs the f16 row is the quality cost. Shipped Q4_0 measured ~6.2% on the
    box; anything near ~2-3% is a real recovery.
  * tok/s is CPU-only decode. If a better-quality row is also slower, that is a
    genuine trade, not a free win -- the project leads with decode speed.
  * Q5_K_M is included as an upper bound: better quality, bigger file. Worth
    knowing what you are giving up by insisting on 4 bits.

Pick the row with acceptable PPL at the best tok/s, then upload it as the new
instruct/model-q4_0.gguf. No retraining, no GPU, no box.
NOTE
