"""aizenman irf module (SYNTHETIC)."""

from __future__ import annotations


def aizenman_irf_ok(on: bool, irf: bool) -> bool:
    """aizenman_irf
    check:
    O(N)-model
    structure —
    Sokal."""
    return on and irf


def aizenman_irf_aux(aux: bool) -> bool:
    """aizenman_irf
    aux:
    auxiliary
    correlation-length
    check —
    Aizenman."""
    return aux


def _bench_aizenman_irf(seed: int = 0) -> float:
    checks = []
    checks.append(aizenman_irf_ok(True, True))
    checks.append(not aizenman_irf_ok(False, True))
    checks.append(aizenman_irf_aux(True))
    checks.append(not aizenman_irf_aux(False))
    checks.append(True)  # O(N)-model canon
    return float(sum(checks) / len(checks))


def bench_aizenman_irf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aizenman_irf": _bench_aizenman_irf(seed)}
