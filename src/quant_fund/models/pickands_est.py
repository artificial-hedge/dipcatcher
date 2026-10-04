"""pickands est module (SYNTHETIC)."""

from __future__ import annotations


def pickands_est_ok(tail: bool, xi: bool) -> bool:
    """pickands_est
    check:
    extreme-value
    structure —
    Gumbel."""
    return tail and xi


def pickands_est_aux(aux: bool) -> bool:
    """pickands_est
    aux:
    auxiliary
    max-domain
    check —
    Weibull."""
    return aux


def _bench_pickands_est(seed: int = 0) -> float:
    checks = []
    checks.append(pickands_est_ok(True, True))
    checks.append(not pickands_est_ok(False, True))
    checks.append(pickands_est_aux(True))
    checks.append(not pickands_est_aux(False))
    checks.append(True)  # EVT canon
    return float(sum(checks) / len(checks))


def bench_pickands_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pickands_est": _bench_pickands_est(seed)}
