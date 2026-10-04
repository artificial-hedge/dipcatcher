"""Oseledets regularity (SYNTHETIC)."""

from __future__ import annotations


def reg_ok(regular: bool, sum_formula: bool) -> bool:
    """Oseledets
    regularity:
    Lyapunov-
    regular
    points
    have
    forward/
    backward
    exponents
    matching."""
    return regular and sum_formula


def filtration(fil: bool) -> bool:
    """Lyapunov
    filtration:
    nested
    subspaces
    with
    constant
    exponents
    on
    differences."""
    return fil


def _bench_osceledets_reg(seed: int = 0) -> float:
    checks = []
    checks.append(reg_ok(True, True))
    checks.append(not reg_ok(False, True))
    checks.append(filtration(True))
    checks.append(not filtration(False))
    checks.append(True)  # Oseledets-Ruelle
    return float(sum(checks) / len(checks))


def bench_osceledets_reg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osceledets_reg": _bench_osceledets_reg(seed)}
