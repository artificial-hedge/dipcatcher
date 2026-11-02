"""Operator kill-switch file.

The file holds one state token from ``quant_fund.monitoring.kill_switch``.
It is not a secret and the reader never returns the raw body. A missing file
means ``ENABLED``. Any other problem fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_fund.utils.operator_kill import read_operator_kill


@dataclass(frozen=True)
class KillSwitchReading:
    state: str | None
    blocking: bool
    problem: str | None


def inspect_kill_switch(path: Path) -> KillSwitchReading:
    """Read ``path``. Does not follow a symlink."""
    reading = read_operator_kill(path)
    return KillSwitchReading(reading.state, reading.blocking, reading.problem)


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
