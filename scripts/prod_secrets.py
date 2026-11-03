#!/usr/bin/env python3
"""Write production secrets to files for docker secrets.

The shell still holds ``QUANT_API_KEY`` and ``RESEARCH_API_KEY``. This
script copies them into ``DIPCATCHER_QUANT_API_KEY_FILE`` and
``DIPCATCHER_RESEARCH_API_KEY_FILE``. The alert webhook secret is copied
when ``DIPCATCHER_ALERT_WEBHOOK_SECRET`` is set. Otherwise an existing
token file is kept, or a new token is written. Values are not printed.
The directory is mode ``0700``. Each file is mode ``0644`` because Compose
bind-mounts the host inode and ignores a secret ``mode``; the API user is
uid 10001 and cannot read a mode ``0600`` file owned by the host user.
"""

from __future__ import annotations

import os
import secrets
import stat
import sys
from pathlib import Path

_DEFAULT_DIR = Path("/tmp/dipcatcher-ops7-secrets")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-secrets takes no arguments\n")
        return 1
    quant = os.environ.get("QUANT_API_KEY", "").strip()
    research = os.environ.get("RESEARCH_API_KEY", "").strip()
    if not quant or not research:
        sys.stderr.write("QUANT_API_KEY and RESEARCH_API_KEY are required\n")
        return 1
    if not _single_token(quant) or not _single_token(research):
        sys.stderr.write("QUANT_API_KEY and RESEARCH_API_KEY must be one token\n")
        return 1
    quant_path = _path("DIPCATCHER_QUANT_API_KEY_FILE", "quant_api_key")
    research_path = _path("DIPCATCHER_RESEARCH_API_KEY_FILE", "research_api_key")
    webhook_path = _path("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE", "alert_webhook_secret")
    _write(quant_path, _preserved(quant_path, quant))
    _write(research_path, _preserved(research_path, research))
    webhook = _webhook_text(webhook_path)
    if webhook is None:
        sys.stderr.write("DIPCATCHER_ALERT_WEBHOOK_SECRET must contain one token\n")
        return 1
    if webhook:
        _write(webhook_path, webhook)
    else:
        _publish(webhook_path)
    return 0


def _path(name: str, leaf: str) -> Path:
    raw = os.environ.get(name, "").strip()
    if raw:
        return Path(raw)
    return _DEFAULT_DIR / leaf


def _webhook_text(path: Path) -> str | None:
    """Return text to write, ``""`` to keep the file, or ``None`` when invalid."""
    plain = os.environ.get("DIPCATCHER_ALERT_WEBHOOK_SECRET", "").strip()
    if plain:
        if not _single_token(plain):
            return None
        return plain
    existing = _read(path)
    if existing is not None and _overlap_text(existing):
        return ""
    return secrets.token_hex(32)


def _preserved(path: Path, incoming: str) -> str:
    """Keep both lines when the shell token is already one of them."""
    existing = _read(path)
    if existing is None or not _overlap_text(existing) or "\n" not in existing:
        return incoming
    if incoming in existing.split("\n"):
        return existing
    return incoming


def _overlap_text(value: str) -> bool:
    parts = [part for part in value.split("\n") if part]
    if not parts or len(parts) > 2:
        return False
    return all(_single_token(part) for part in parts)


def _read(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if text.endswith("\n"):
        text = text[:-1]
    return text


def _single_token(value: str) -> bool:
    if not value or any(char.isspace() for char in value):
        return False
    return not any(char in value for char in "\"'{}#")


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        os.write(descriptor, (value + "\n").encode("utf-8"))
        os.fchmod(descriptor, 0o644)
    finally:
        os.close(descriptor)
    _publish(path)


def _publish(path: Path) -> None:
    """Make an existing secret readable by uid 10001 without widening the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    os.chmod(path, 0o644)
    mode = stat.S_IMODE(path.stat().st_mode)
    parent = stat.S_IMODE(path.parent.stat().st_mode)
    if mode != 0o644 or parent & 0o077:
        raise OSError(f"{path.name} mode")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
