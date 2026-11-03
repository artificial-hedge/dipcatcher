#!/usr/bin/env python3
"""Copy an audit chain and its anchors file. Stdlib only.

The copy includes the active log, every rotated sibling ``NAME.N``, and the
anchors file written by ``scripts/anchor_audit_head.py``. This is the
prod-backup set for those files. The anchors stay unsigned. Signing one is
an owner step.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def backup_sources(log: Path, anchors: Path) -> list[Path]:
    """Chain files plus the anchors file. Names must be unique."""
    files = generations(log)
    if not files:
        raise FileNotFoundError(log)
    if not anchors.is_file():
        raise FileNotFoundError(anchors)
    files.append(anchors)
    names = [item.name for item in files]
    if len(names) != len(set(names)):
        raise ValueError("backup names collide")
    return files


def copy_tree(log: Path, anchors: Path, dest: Path) -> list[str]:
    """Write each source into ``dest`` as mode ``0600``. Returns the names."""
    sources = backup_sources(log, anchors)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sources:
        _copy_file(source, dest / source.name)
        copied.append(source.name)
    return copied


def _copy_file(source: Path, dest: Path) -> None:
    payload = source.read_bytes()
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(dest, 0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up an audit chain and its anchors.")
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        names = copy_tree(args.audit_log, args.anchors, args.dest)
    except (OSError, UnicodeError, ValueError):
        print("prod backup missing")
        return 1
    print("prod backup ok")
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
