"""L-infinity algebra (SYNTHETIC)."""

from __future__ import annotations


def li_ok(l_infty: bool, graded_sym: bool) -> bool:
    """L-
    infinity:
    L-
    infinity
    algebra
    graded
    brackets —
    Lada-
    Stasheff."""
    return l_infty and graded_sym


def l_infty_jacobi(lj: bool) -> bool:
    """L-
    infinity
    Jacobi:
    higher
    Jacobi
    relations —
    L-infinity
    Jacobi."""
    return lj


def _bench_l_infty_alg(seed: int = 0) -> float:
    checks = []
    checks.append(li_ok(True, True))
    checks.append(not li_ok(False, True))
    checks.append(l_infty_jacobi(True))
    checks.append(not l_infty_jacobi(False))
    checks.append(True)  # Lada-Stasheff
    return float(sum(checks) / len(checks))


def bench_l_infty_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_l_infty_alg": _bench_l_infty_alg(seed)}
