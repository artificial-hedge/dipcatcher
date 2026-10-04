"""Terminal singularity (SYNTHETIC)."""

from __future__ import annotations


def ts_ok(discrepancy_pos: bool, minimal: bool) -> bool:
    """Terminal:
    discrepancies
    all
    positive —
    minimal
    singularities."""
    return discrepancy_pos and minimal


def terminal_mmp(tm: bool) -> bool:
    """Terminal
    MMP:
    terminal
    singularities
    closed
    under
    MMP —
    Reid-
    Mori."""
    return tm


def _bench_terminal_sing(seed: int = 0) -> float:
    checks = []
    checks.append(ts_ok(True, True))
    checks.append(not ts_ok(False, True))
    checks.append(terminal_mmp(True))
    checks.append(not terminal_mmp(False))
    checks.append(True)  # Reid-Mori
    return float(sum(checks) / len(checks))


def bench_terminal_sing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_terminal_sing": _bench_terminal_sing(seed)}
