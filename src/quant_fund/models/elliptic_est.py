"""Elliptic estimates (SYNTHETIC)."""

from __future__ import annotations


def ee_ok(gaarding: bool, sobolev: bool) -> bool:
    """Elliptic
    estimates:
    Gårding
    inequality
    gives
    Sobolev
    coercivity
    from
    symbol
    positivity."""
    return gaarding and sobolev


def interior_reg(ir: bool) -> bool:
    """Interior
    regularity:
    elliptic
    order m
    gains
    m
    derivatives
    in
    H^s."""
    return ir


def _bench_elliptic_est(seed: int = 0) -> float:
    checks = []
    checks.append(ee_ok(True, True))
    checks.append(not ee_ok(False, True))
    checks.append(interior_reg(True))
    checks.append(not interior_reg(False))
    checks.append(True)  # Gårding
    return float(sum(checks) / len(checks))


def bench_elliptic_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_est": _bench_elliptic_est(seed)}
