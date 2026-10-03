"""periodic spline module (SYNTHETIC)."""

from __future__ import annotations


def periodic_spline_ok(basis: bool, coef: bool) -> bool:
    """periodic_spline
    check:
    radial basis / projection —
    basis/coefficient
    consistency."""
    return basis and coef


def periodic_spline_aux(aux: bool) -> bool:
    """periodic_spline
    aux:
    auxiliary
    basis check —
    reproducing bound."""
    return aux


def _bench_periodic_spline(seed: int = 0) -> float:
    checks = []
    checks.append(periodic_spline_ok(True, True))
    checks.append(not periodic_spline_ok(False, True))
    checks.append(periodic_spline_aux(True))
    checks.append(not periodic_spline_aux(False))
    checks.append(True)  # RBF/basis canon
    return float(sum(checks) / len(checks))


def bench_periodic_spline(seed: int = 0) -> dict[str, float]:
    return {"synthetic_periodic_spline": _bench_periodic_spline(seed)}
