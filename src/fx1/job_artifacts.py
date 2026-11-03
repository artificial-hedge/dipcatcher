"""Fail closed when a fine-tune result file cannot be read."""

from __future__ import annotations

from pathlib import Path


def read_result_artifact(name: str, path: str | Path) -> bytes:
    """Return artifact bytes. An unreadable path is a job failure, not a skip."""
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read fine-tune result artifact {name!r}: {path}") from exc
