"""Operator kill-switch file.

The file holds one state token from ``quant_fund.monitoring.kill_switch``.
It is not a secret and the reader never returns the raw body. A missing file
means ``ENABLED``. Any other problem fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_fund.monitoring.kill_switch import ENABLED, VALID_STATES

_MAX_BYTES = 64


@dataclass(frozen=True)
class KillSwitchReading:
    state: str | None
    blocking: bool
    problem: str | None


def inspect_kill_switch(path: Path) -> KillSwitchReading:
    """Read ``path``. Does not follow a symlink."""
    if path.is_symlink():
        return KillSwitchReading(None, True, "kill_switch_symlink")
    if path.is_dir():
        return KillSwitchReading(None, True, "kill_switch_directory")
    if not path.exists():
        return KillSwitchReading(ENABLED, False, None)
    try:
        if path.stat().st_size > _MAX_BYTES:
            return KillSwitchReading(None, True, "kill_switch_invalid")
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return KillSwitchReading(None, True, "kill_switch_unreadable")
    if token not in VALID_STATES:
        return KillSwitchReading(None, True, "kill_switch_invalid")
    return KillSwitchReading(token, token != ENABLED, None)


def validate_kill_switch_path(path: Path) -> str | None:
    """Startup check. ``None`` means the path is usable."""
    if len(str(path)) > 512:
        return "DIPCATCHER_KILL_SWITCH_PATH is invalid"
    if path.is_symlink():
        return "DIPCATCHER_KILL_SWITCH_PATH must not be a symlink"
    if path.is_dir():
        return "DIPCATCHER_KILL_SWITCH_PATH must be a file"
    if path.exists():
        reading = inspect_kill_switch(path)
        if reading.problem:
            return f"DIPCATCHER_KILL_SWITCH_PATH is not usable ({reading.problem})"
        return None
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        return "DIPCATCHER_KILL_SWITCH_PATH directory does not exist"
    return None
