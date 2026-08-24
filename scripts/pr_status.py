"""Render safe GitHub draft-PR commands for the Vast program.

The script only prepares `gh pr create --draft` / `gh pr edit` commands and body
text. It deliberately has no merge, close, force-push, or default-branch write
operation.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def build_body(*, status_path, progress_url: str = "", extra: str = "") -> str:
    status_text = Path(status_path).read_text() if status_path else ""
    parts = ["# Daedalus Vast Program", "", "This PR is a draft until final gates pass."]
    if progress_url:
        parts.extend(["", f"Progress branch: {progress_url}"])
    if status_text:
        parts.extend(["", "## Current public status", "", status_text.strip()])
    if extra:
        parts.extend(["", "## Notes", "", extra.strip()])
    return "\n".join(parts) + "\n"


def draft_command(*, title: str, base: str, head: str, body_file: str) -> list[str]:
    return [
        "gh", "pr", "create", "--draft", "--base", base, "--head", head,
        "--title", title, "--body-file", body_file,
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True)
    parser.add_argument("--body-out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--base", default="main")
    parser.add_argument("--head", default="vast/daedalus-improvements-20260824")
    parser.add_argument("--progress-url", default="")
    parser.add_argument("--extra", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    body = build_body(status_path=args.status, progress_url=args.progress_url, extra=args.extra)
    Path(args.body_out).write_text(body)
    command = draft_command(title=args.title, base=args.base, head=args.head, body_file=args.body_out)
    if args.json:
        print(json.dumps({"body": args.body_out, "command": command}, indent=2))
    else:
        print(" ".join(shlex.quote(part) for part in command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())