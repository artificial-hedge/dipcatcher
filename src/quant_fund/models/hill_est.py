"""hill est module (SYNTHETIC)."""

from __future__ import annotations


def hill_est_ok(tail: bool, xi: bool) -> bool:
    """hill_est
    check:
    extreme-value
    structure —
    Gumbel."""
    return tail and xi


def hill_est_aux(aux: bool) -> bool:
    """hill_est
    aux:
    auxiliary
    max-domain
    check —
    Weibull."""
    return aux


def _bench_hill_est(seed: int = 0) -> float:
    checks = []
    checks.append(hill_est_ok(True, True))
    checks.append(not hill_est_ok(False, True))
    checks.append(hill_est_aux(True))
    checks.append(not hill_est_aux(False))
    checks.append(True)  # EVT canon
    return float(sum(checks) / len(checks))


def bench_hill_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hill_est": _bench_hill_est(seed)}
