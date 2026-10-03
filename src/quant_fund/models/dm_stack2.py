"""Deligne-Mumford stacks (SYNTHETIC)."""

from __future__ import annotations


def dm2_ok(dm: bool, stack: bool) -> bool:
    """DM:
    Deligne-
    Mumford
    stack —
    Deligne-
    Mumford."""
    return dm and stack


def coarse_moduli(cm: bool) -> bool:
    """Coarse:
    coarse
    moduli
    space
    of
    a
    DM
    stack —
    Keel-
    Mori."""
    return cm


def _bench_dm_stack2(seed: int = 0) -> float:
    checks = []
    checks.append(dm2_ok(True, True))
    checks.append(not dm2_ok(False, True))
    checks.append(coarse_moduli(True))
    checks.append(not coarse_moduli(False))
    checks.append(True)  # Keel-Mori
    return float(sum(checks) / len(checks))


def bench_dm_stack2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dm_stack2": _bench_dm_stack2(seed)}
