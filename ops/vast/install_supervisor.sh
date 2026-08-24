#!/bin/bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "install_supervisor.sh must run as root" >&2
    exit 1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
install -m 0755 "${repo_root}/ops/vast/daedalus_progress.sh" \
    /opt/supervisor-scripts/daedalus_progress.sh
install -m 0755 "${repo_root}/ops/vast/daedalus_resume.sh" \
    /opt/supervisor-scripts/daedalus_resume.sh
install -m 0755 "${repo_root}/ops/vast/run-approved" \
    /usr/local/bin/daedalus-approved
install -m 0644 "${repo_root}/ops/vast/supervisord.conf" \
    /etc/supervisor/conf.d/daedalus.conf

supervisorctl reread
supervisorctl update
supervisorctl status daedalus_progress
supervisorctl status daedalus_resume || true
