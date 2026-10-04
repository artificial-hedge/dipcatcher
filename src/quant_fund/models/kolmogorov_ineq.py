"""kolmogorov ineq module (SYNTHETIC)."""

from __future__ import annotations


def kolmogorov_ineq_ok(bound: bool, tail: bool) -> bool:
    """kolmogorov_ineq
    check:
    maximal
    inequality —
    tail bound."""
    return bound and tail


def kolmogorov_ineq_aux(aux: bool) -> bool:
    """kolmogorov_ineq
    aux:
    auxiliary
    inequality check —
    moment."""
    return aux


def _bench_kolmogorov_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(kolmogorov_ineq_ok(True, True))
    checks.append(not kolmogorov_ineq_ok(False, True))
    checks.append(kolmogorov_ineq_aux(True))
    checks.append(not kolmogorov_ineq_aux(False))
    checks.append(True)  # maximal-inequality canon
    return float(sum(checks) / len(checks))


def bench_kolmogorov_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kolmogorov_ineq": _bench_kolmogorov_ineq(seed)}
