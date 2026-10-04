"""trefethen diff module (SYNTHETIC)."""

from __future__ import annotations


def trefethen_diff_ok(basis: bool, coef: bool) -> bool:
    """trefethen_diff
    check:
    radial basis / projection —
    basis/coefficient
    consistency."""
    return basis and coef


def trefethen_diff_aux(aux: bool) -> bool:
    """trefethen_diff
    aux:
    auxiliary
    basis check —
    reproducing bound."""
    return aux


def _bench_trefethen_diff(seed: int = 0) -> float:
    checks = []
    checks.append(trefethen_diff_ok(True, True))
    checks.append(not trefethen_diff_ok(False, True))
    checks.append(trefethen_diff_aux(True))
    checks.append(not trefethen_diff_aux(False))
    checks.append(True)  # RBF/basis canon
    return float(sum(checks) / len(checks))


def bench_trefethen_diff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trefethen_diff": _bench_trefethen_diff(seed)}
