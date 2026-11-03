#!/usr/bin/env python3
"""Verify an admin audit hash chain. Stdlib only.

Each JSON line's ``prev`` is the SHA-256 hex of the previous raw line,
without the trailing newline. The first line's ``prev`` is the SHA-256 of
empty bytes. A non-JSON line is a link: the next JSON line hashes that
text. Rotated siblings ``PATH.N`` are read from the highest N down through
``PATH``.

Exit 0 prints ``audit chain ok``. Exit 1 prints ``audit chain break``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GENESIS = hashlib.sha256(b"").hexdigest()


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


def chain_errors(path: Path) -> list[str]:
    """Return one short error per broken link. The line text stays out."""
    files = generations(path)
    if not files:
        return [f"missing {path.name}"]
    errors: list[str] = []
    previous: str | None = None
    for file in files:
        errors.extend(_file_errors(file, previous))
        previous = _advance(file, previous)
    return errors


def _file_errors(file: Path, previous: str | None) -> list[str]:
    errors: list[str] = []
    cursor = previous
    try:
        text = file.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        return [f"{file.name}: {type(exc).__name__}"]
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        if not line.startswith("{"):
            cursor = line
            continue
        problem = _json_link(line, cursor)
        if problem:
            errors.append(f"{file.name}:{number} {problem}")
        cursor = line
    return errors


def _advance(file: Path, previous: str | None) -> str | None:
    try:
        lines = [line for line in file.read_text(encoding="ascii").splitlines() if line]
    except (OSError, UnicodeError):
        return previous
    if not lines:
        return previous
    return lines[-1]


def _json_link(line: str, previous: str | None) -> str:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return "not json"
    if not isinstance(row, dict):
        return "not an object"
    prev = row.get("prev")
    if not isinstance(prev, str):
        return "missing prev"
    expected = GENESIS if previous is None else hashlib.sha256(previous.encode("ascii")).hexdigest()
    if prev != expected:
        return "prev mismatch"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify an admin audit hash chain.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    errors = chain_errors(args.path)
    if errors:
        print("audit chain break")
        for error in errors:
            print(error)
        return 1
    print("audit chain ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
