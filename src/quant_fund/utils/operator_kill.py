"""One operator kill file for readiness, paper orders, and the fx-1 drain.

The file holds a single state token. Readers never return the body. A missing
file is ``ENABLED``. A symlink, directory, unreadable file, or unknown token
fails closed. ``latch_drain`` sets a one-way event and does not clear it.
``DrainLatch`` is the harness watcher: it sets the same event, then clears
only a latch it owns after ``ENABLED`` has held for the debounce.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from quant_fund.utils.write_owner import WriteOwnerError, require_write_owner

ENABLED = "ENABLED"
HALT_NEW_ORDERS = "HALT_NEW_ORDERS"
CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
FLATTEN_OPTIONAL = "FLATTEN_OPTIONAL"
FLATTEN = "FLATTEN"
STATES = frozenset({ENABLED, HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL, FLATTEN})
ENV_PATH = "DIPCATCHER_KILL_SWITCH_PATH"
_MAX_BYTES = 64
_DEBOUNCE_ENV = "DIPCATCHER_KILL_DRAIN_DEBOUNCE_S"
_DEFAULT_DEBOUNCE_S = 1.0
_MAX_DEBOUNCE_S = 300.0


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


def _require_kill_writer() -> None:
    try:
        require_write_owner()
    except WriteOwnerError as exc:
        raise OperatorKillError(str(exc)) from exc


def write_operator_kill(path: Path, state: str) -> None:
    """Atomically replace ``path`` with ``state``. Refuses to follow a symlink."""
    if state not in STATES:
        raise OperatorKillError("invalid kill switch state")
    _require_kill_writer()
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


def debounce_seconds() -> float:
    """Debounce for ``DrainLatch``. Invalid values never clear the latch."""
    raw = os.environ.get(_DEBOUNCE_ENV, "").strip()
    if not raw:
        return _DEFAULT_DEBOUNCE_S
    try:
        value = float(raw)
    except ValueError:
        return math.inf
    if not math.isfinite(value) or value < 0.0 or value > _MAX_DEBOUNCE_S:
        return math.inf
    return value


def _emit(event: str, **fields: str) -> None:
    payload = {"event": event, "service": "dipcatcher", **fields}
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stderr.flush()


class DrainLatch:
    """Set the harness drain from the kill file and clear only that latch.

    ``POST /harness/drain`` stays one-way: an event this watcher did not set
    is left alone. A blocking file latches immediately. ``ENABLED`` clears
    the owned latch only after ``debounce_s`` of continuous permission.
    """

    def __init__(
        self, debounce_s: float = _DEFAULT_DEBOUNCE_S, clock: Callable[[], float] | None = None
    ) -> None:
        self.debounce_s = float(debounce_s)
        self._clock = time.monotonic if clock is None else clock
        self._enabled_since: float | None = None
        self._owned = False
        self._invalid_logged = False

    @classmethod
    def from_env(cls, clock: Callable[[], float] | None = None) -> DrainLatch:
        seconds = debounce_seconds()
        latch = cls(seconds, clock)
        if math.isinf(seconds):
            latch._log_invalid_debounce()
        return latch

    def _log_invalid_debounce(self) -> None:
        if self._invalid_logged:
            return
        self._invalid_logged = True
        _emit("operator_kill_drain_debounce_invalid")

    def poll(self, event: threading.Event, path: Path | None = None) -> str:
        blocked = orders_blocked_by_env() if path is None else orders_blocked(path)
        if blocked is not None:
            return self._latch(event, blocked)
        return self._release(event)

    def _latch(self, event: threading.Event, code: str) -> str:
        self._enabled_since = None
        if event.is_set():
            return "held"
        event.set()
        self._owned = True
        _emit("operator_kill_drain_latched", code=code)
        return "latched"

    def _release(self, event: threading.Event) -> str:
        if not self._owned or not event.is_set():
            self._owned = False
            self._enabled_since = None
            return "open"
        now = self._clock()
        started = self._enabled_since
        if started is None:
            self._enabled_since = now
            started = now
            if self.debounce_s > 0.0:
                return "waiting"
        if now - started < self.debounce_s:
            return "waiting"
        event.clear()
        self._owned = False
        self._enabled_since = None
        _emit("operator_kill_drain_cleared")
        return "cleared"
