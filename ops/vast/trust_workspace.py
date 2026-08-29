"""Mark the program workspace trusted so its Claude permission rules apply.

Claude Code ignores a project's `.claude/settings.json` until the workspace is
trusted, and it says so only in a stderr warning that a non-interactive session
never shows anyone. The entries being ignored are not conveniences: they are the
`deny` rules that keep an engineering session away from `.env`, the runtime
credential directory, and the SSH material. An untrusted workspace therefore
looks exactly like a configured one while enforcing none of it.

Trust is recorded per project in the user-level `~/.claude.json`, which is why
this cannot live in the repository's own settings file.
"""

import argparse
import json
import os
from pathlib import Path


def trust_workspace(config_path, workspace: str) -> bool:
    """Record `workspace` as trusted. Returns True when the file changed."""

    path = Path(config_path)
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        config = {}
    except json.JSONDecodeError as error:
        raise SystemExit(f"refusing to rewrite unparsable {path}: {error}")
    if not isinstance(config, dict):
        raise SystemExit(f"refusing to rewrite {path}: expected an object")

    projects = config.setdefault("projects", {})
    entry = projects.setdefault(workspace, {})
    if entry.get("hasTrustDialogAccepted") is True:
        return False
    entry["hasTrustDialogAccepted"] = True

    # The file holds account state, so it is rewritten atomically at mode 0600.
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return True


def is_trusted(config_path, workspace: str) -> bool:
    try:
        config = json.loads(Path(config_path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    project = (config.get("projects") or {}).get(workspace) or {}
    return project.get("hasTrustDialogAccepted") is True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path.home() / ".claude.json"))
    parser.add_argument("--workspace", default="/workspace/daedalus")
    args = parser.parse_args(argv)

    changed = trust_workspace(args.config, args.workspace)
    print(f"{args.workspace}: {'trusted now' if changed else 'already trusted'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
