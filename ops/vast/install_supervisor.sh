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
install -m 0755 "${repo_root}/ops/vast/daedalus_session_keeper.sh" \
    /opt/supervisor-scripts/daedalus_session_keeper.sh
install -m 0755 "${repo_root}/ops/vast/run-approved" \
    /usr/local/bin/daedalus-approved
install -m 0644 "${repo_root}/ops/vast/supervisord.conf" \
    /etc/supervisor/conf.d/daedalus.conf

# Permission rules in .claude/settings.json are ignored until the workspace is
# trusted, and the deny rules protecting credentials are among them.
python3 "${repo_root}/ops/vast/trust_workspace.py" \
    --workspace "${repo_root}"

supervisorctl reread
supervisorctl update
supervisorctl status daedalus_progress
supervisorctl status daedalus_resume || true
supervisorctl status daedalus_session_keeper
