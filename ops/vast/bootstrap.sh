#!/bin/bash
set -euo pipefail

: "${WORKSPACE:=/workspace}"
repo=${DAEDALUS_REPO:-${WORKSPACE}/daedalus}
config_dir=${DAEDALUS_CONFIG_DIR:-/root/.config/daedalus}

mkdir -p "${config_dir}"
chmod 700 "${config_dir}"

cd "${repo}"
base_sha=$(git rev-parse origin/main 2>/dev/null || git rev-parse main)
python scripts/vast_program.py \
  --state "${repo}/runs/vast-program/state.json" \
  --base-sha "${base_sha}" \
  init --phase bootstrap --status running

bash "${repo}/ops/vast/install_supervisor.sh"