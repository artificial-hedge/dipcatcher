#!/usr/bin/env python3
"""Append the audit chain head to a separate anchors file. Stdlib only.

The head is the SHA-256 hex of the last raw non-empty line in the chain,
which is the ``prev`` the next audit line would store. An empty chain uses
the SHA-256 of empty bytes. The anchors file is append-only and unsigned.
Signing that digest is an owner step, the same as ``gate_pins.sig``.

``scripts/prod_backup.py`` copies this file with the chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def generations(path: Path) -> list[Path]:
    """Oldest rotated sibling first, then the active file."""
    found: list[Path] = []
    generation = 1
    while True:
        candidate = path.with_name(f"{path.name}.{generation}")
        if not candidate.is_file():
            break
        found.append(candidate)
        generation += 1
    found.reverse()
    if path.is_file():
        found.append(path)
    return found


def last_raw_line(path: Path) -> str | None:
    """Last non-empty line, ``""`` when the chain exists and is empty, else ``None``."""
    files = generations(path)
    if not files:
        return None
    last = ""
    found = False
    for file in files:
        for line in file.read_text(encoding="ascii").splitlines():
            if line:
                last = line
                found = True
    if not found:
        return ""
    return last


def head_digest(line: str) -> str:
    """SHA-256 hex of one raw audit line, or of empty bytes when ``line`` is empty."""
    return hashlib.sha256(line.encode("ascii")).hexdigest()


def append_anchor(anchors: Path, digest: str, now: datetime | None = None) -> None:
    """Append one JSON line and fsync it. The file mode is ``0600``."""
    moment = now if now is not None else datetime.now(UTC)
    record = {
        "head": digest,
        "time": moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(anchors, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(anchors, 0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anchor an admin audit chain head.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--anchors", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        line = last_raw_line(args.path)
    except (OSError, UnicodeError):
        print("audit anchor unavailable")
        return 1
    if line is None:
        print("audit anchor missing")
        return 1
    try:
        append_anchor(args.anchors, head_digest(line))
    except (OSError, UnicodeError):
        print("audit anchor unavailable")
        return 1
    print("audit anchor ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
