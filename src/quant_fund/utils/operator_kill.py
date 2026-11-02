"""One operator kill file for readiness, paper orders, and the fx-1 drain.

The file holds a single state token. Readers never return the body. A missing
file is ``ENABLED``. A symlink, directory, unreadable file, or unknown token
fails closed. ``latch_drain`` sets a one-way event and does not clear it.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

ENABLED = "ENABLED"
HALT_NEW_ORDERS = "HALT_NEW_ORDERS"
CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
FLATTEN_OPTIONAL = "FLATTEN_OPTIONAL"
STATES = frozenset({ENABLED, HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL})
ENV_PATH = "DIPCATCHER_KILL_SWITCH_PATH"
_MAX_BYTES = 64


class OperatorKillError(ValueError):
    """The kill file could not be written. The message names a code, not the body."""


@dataclass(frozen=True)
class OperatorKillReading:
    state: str | None
    blocking: bool
    problem: str | None


def read_operator_kill(path: Path) -> OperatorKillReading:
    """Read ``path`` without following a symlink."""
    if path.is_symlink():
        return OperatorKillReading(None, True, "kill_switch_symlink")
    if path.is_dir():
        return OperatorKillReading(None, True, "kill_switch_directory")
    if not path.exists():
        return OperatorKillReading(ENABLED, False, None)
    try:
        if path.stat().st_size > _MAX_BYTES:
            return OperatorKillReading(None, True, "kill_switch_invalid")
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return OperatorKillReading(None, True, "kill_switch_unreadable")
    except UnicodeError:
        return OperatorKillReading(None, True, "kill_switch_invalid")
    if token not in STATES:
        return OperatorKillReading(None, True, "kill_switch_invalid")
    return OperatorKillReading(token, token != ENABLED, None)


def orders_blocked(path: Path | None) -> str | None:
    """A stable block code, or ``None`` when new orders are allowed."""
    if path is None:
        return None
    reading = read_operator_kill(path)
    if reading.problem:
        return reading.problem
    if reading.blocking and reading.state:
        return f"kill_switch:{reading.state}"
    return None


def orders_blocked_by_env() -> str | None:
    raw = os.environ.get(ENV_PATH, "").strip()
    if not raw:
        return None
    return orders_blocked(Path(raw))


def write_operator_kill(path: Path, state: str) -> None:
    """Atomically replace ``path`` with ``state``. Refuses to follow a symlink."""
    if state not in STATES:
        raise OperatorKillError("invalid kill switch state")
    if path.is_symlink():
        raise OperatorKillError("kill_switch_symlink")
    if path.is_dir():
        raise OperatorKillError("kill_switch_directory")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise OperatorKillError("kill_switch_directory")
    temporary = parent / f".{path.name}.tmp"
    if temporary.is_symlink():
        raise OperatorKillError("kill_switch_symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        os.write(descriptor, f"{state}\n".encode())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def publish_operator_kill(root: Path, state: str, explicit: Path | None) -> Path:
    """Write ``explicit``, else the env path, else ``root / kill_switch``."""
    raw = os.environ.get(ENV_PATH, "").strip()
    if explicit is not None:
        path = explicit
    elif raw:
        path = Path(raw)
    else:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "kill_switch"
    write_operator_kill(path, state)
    return path


def latch_drain(event: threading.Event, path: Path | None = None) -> bool:
    """Set ``event`` when the kill file blocks. Never clears it."""
    if event.is_set():
        return False
    blocked = orders_blocked_by_env() if path is None else orders_blocked(path)
    if blocked is None:
        return False
    event.set()
    return True
