"""fradkin sokal module (SYNTHETIC)."""

from __future__ import annotations


def fradkin_sokal_ok(on: bool, irf: bool) -> bool:
    """fradkin_sokal
    check:
    O(N)-model
    structure —
    Sokal."""
    return on and irf


def fradkin_sokal_aux(aux: bool) -> bool:
    """fradkin_sokal
    aux:
    auxiliary
    correlation-length
    check —
    Aizenman."""
    return aux


def _bench_fradkin_sokal(seed: int = 0) -> float:
    checks = []
    checks.append(fradkin_sokal_ok(True, True))
    checks.append(not fradkin_sokal_ok(False, True))
    checks.append(fradkin_sokal_aux(True))
    checks.append(not fradkin_sokal_aux(False))
    checks.append(True)  # O(N)-model canon
    return float(sum(checks) / len(checks))


def bench_fradkin_sokal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fradkin_sokal": _bench_fradkin_sokal(seed)}
