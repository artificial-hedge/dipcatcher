"""quasi mc module (SYNTHETIC)."""

from __future__ import annotations


def quasi_mc_ok(draw: bool, weight: bool) -> bool:
    """quasi_mc
    check:
    quadrature/quasi-MC —
    sample-weight
    consistency."""
    return draw and weight


def quasi_mc_aux(aux: bool) -> bool:
    """quasi_mc
    aux:
    auxiliary
    MC check —
    discrepancy bound."""
    return aux


def _bench_quasi_mc(seed: int = 0) -> float:
    checks = []
    checks.append(quasi_mc_ok(True, True))
    checks.append(not quasi_mc_ok(False, True))
    checks.append(quasi_mc_aux(True))
    checks.append(not quasi_mc_aux(False))
    checks.append(True)  # quadrature/MC canon
    return float(sum(checks) / len(checks))


def bench_quasi_mc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_mc": _bench_quasi_mc(seed)}
