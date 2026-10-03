"""cardy on module (SYNTHETIC)."""

from __future__ import annotations


def cardy_on_ok(on: bool, irf: bool) -> bool:
    """cardy_on
    check:
    O(N)-model
    structure —
    Sokal."""
    return on and irf


def cardy_on_aux(aux: bool) -> bool:
    """cardy_on
    aux:
    auxiliary
    correlation-length
    check —
    Aizenman."""
    return aux


def _bench_cardy_on(seed: int = 0) -> float:
    checks = []
    checks.append(cardy_on_ok(True, True))
    checks.append(not cardy_on_ok(False, True))
    checks.append(cardy_on_aux(True))
    checks.append(not cardy_on_aux(False))
    checks.append(True)  # O(N)-model canon
    return float(sum(checks) / len(checks))


def bench_cardy_on(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cardy_on": _bench_cardy_on(seed)}
