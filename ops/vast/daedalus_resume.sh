#!/bin/bash
set -euo pipefail

: "${WORKSPACE:=/workspace}"
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

config_dir=${DAEDALUS_CONFIG_DIR:-/root/.config/daedalus}
venv_dir=${DAEDALUS_VENV:-/venv/main}
source "${config_dir}/runtime.env"
source "${venv_dir}/bin/activate"

source_repo="${WORKSPACE}/daedalus"
cd "${source_repo}"
exec python -u scripts/boot_resume.py --runs-root "${source_repo}/runs"
