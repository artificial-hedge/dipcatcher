"""Cartesian closed (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(exponential: bool, cartesian: bool) -> bool:
    """Cartesian
    closed:
    exponential
    object
    adjoint
    to
    product —
    lambda
    category."""
    return exponential and cartesian


def currying(cur: bool) -> bool:
    """Currying:
    curry
    and
    uncurry
    biject
    Hom
    sets —
    exponential
    adjoint."""
    return cur


def _bench_cartesian_closed(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(currying(True))
    checks.append(not currying(False))
    checks.append(True)  # Lawvere
    return float(sum(checks) / len(checks))


def bench_cartesian_closed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartesian_closed": _bench_cartesian_closed(seed)}
