#!/bin/bash
set -euo pipefail

: "${WORKSPACE:=/workspace}"
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

config_dir=${DAEDALUS_CONFIG_DIR:-/root/.config/daedalus}
venv_dir=${DAEDALUS_VENV:-/venv/main}
source "${config_dir}/claude.env"
source "${venv_dir}/bin/activate"

source_repo="${WORKSPACE}/daedalus"
cd "${source_repo}"
export PYTHONPATH="${source_repo}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -u scripts/session_keeper.py \
  --repo "${source_repo}" \
  --state "${source_repo}/runs/vast-program/state.json" \
  --keeper-state "${source_repo}/runs/vast-program/keeper.json" \
  --prompt-dir "${source_repo}/ops/vast/prompts" \
  --default-prompt "${source_repo}/ops/vast/prompts/default.md" \
  --runs-root "${source_repo}/runs" \
  --busy-poll-sec "${DAEDALUS_KEEPER_BUSY_POLL_SEC:-900}" \
  --plan-hashes "${config_dir}/plan-hashes.txt" \
  --plan-context "${config_dir}/claude-plan-context.md" \
  --claude-bin "${DAEDALUS_CLAUDE_BIN:-/root/.local/bin/claude}" \
  --model opus \
  --effort xhigh \
  --permission-mode dontAsk \
  --session-timeout-sec "${DAEDALUS_SESSION_TIMEOUT_SEC:-7200}" \
  --poll-interval-sec "${DAEDALUS_KEEPER_POLL_SEC:-60}"
