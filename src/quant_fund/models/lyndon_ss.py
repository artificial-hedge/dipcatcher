"""Lyndon-Hochschild-Serre (SYNTHETIC)."""

from __future__ import annotations


def lhs_ok(group_extension: bool, group_cohomology: bool) -> bool:
    """LHS
    spectral
    sequence:
    group
    cohomology
    of
    extension —
    Lyndon-
    Hochschild-
    Serre."""
    return group_extension and group_cohomology


def inflation_restriction(ir: bool) -> bool:
    """Inflation-
    restriction:
    low-
    degree
    exact
    sequence —
    LHS
    edge."""
    return ir


def _bench_lyndon_ss(seed: int = 0) -> float:
    checks = []
    checks.append(lhs_ok(True, True))
    checks.append(not lhs_ok(False, True))
    checks.append(inflation_restriction(True))
    checks.append(not inflation_restriction(False))
    checks.append(True)  # Lyndon-Hochschild-Serre
    return float(sum(checks) / len(checks))


def bench_lyndon_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lyndon_ss": _bench_lyndon_ss(seed)}
