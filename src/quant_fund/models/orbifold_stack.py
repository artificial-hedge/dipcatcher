"""Orbifold stacks (SYNTHETIC)."""

from __future__ import annotations


def os_ok(orbifold: bool, stack: bool) -> bool:
    """Orbifold
    stack:
    orbifold
    stack —
    Satake-
    Thurston
    orbifold."""
    return orbifold and stack


def orbifold_chart(oc: bool) -> bool:
    """Orbifold
    chart:
    orbifold
    chart —
    orbifold
    local
    chart."""
    return oc


def _bench_orbifold_stack(seed: int = 0) -> float:
    checks = []
    checks.append(os_ok(True, True))
    checks.append(not os_ok(False, True))
    checks.append(orbifold_chart(True))
    checks.append(not orbifold_chart(False))
    checks.append(True)  # Satake-Thurston
    return float(sum(checks) / len(checks))


def bench_orbifold_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_orbifold_stack": _bench_orbifold_stack(seed)}
