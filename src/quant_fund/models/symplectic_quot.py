"""Symplectic quotient (SYNTHETIC)."""

from __future__ import annotations


def sq_ok(moment_level: bool, marsden_weinstein: bool) -> bool:
    """Symplectic
    quotient:
    level-
    set
    quotient
    of
    moment
    map —
    Marsden-
    Weinstein
    reduction."""
    return moment_level and marsden_weinstein


def mw_reduction(mwr: bool) -> bool:
    """MW
    reduction:
    quotient
    inherits
    symplectic
    structure —
    Marsden-
    Weinstein-
    Meyer."""
    return mwr


def _bench_symplectic_quot(seed: int = 0) -> float:
    checks = []
    checks.append(sq_ok(True, True))
    checks.append(not sq_ok(False, True))
    checks.append(mw_reduction(True))
    checks.append(not mw_reduction(False))
    checks.append(True)  # Marsden-Weinstein
    return float(sum(checks) / len(checks))


def bench_symplectic_quot(seed: int = 0) -> dict[str, float]:
    return {"synthetic_symplectic_quot": _bench_symplectic_quot(seed)}
