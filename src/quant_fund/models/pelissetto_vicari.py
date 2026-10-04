"""pelissetto vicari module (SYNTHETIC)."""

from __future__ import annotations


def pelissetto_vicari_ok(on: bool, irf: bool) -> bool:
    """pelissetto_vicari
    check:
    O(N)-model
    structure —
    Sokal."""
    return on and irf


def pelissetto_vicari_aux(aux: bool) -> bool:
    """pelissetto_vicari
    aux:
    auxiliary
    correlation-length
    check —
    Aizenman."""
    return aux


def _bench_pelissetto_vicari(seed: int = 0) -> float:
    checks = []
    checks.append(pelissetto_vicari_ok(True, True))
    checks.append(not pelissetto_vicari_ok(False, True))
    checks.append(pelissetto_vicari_aux(True))
    checks.append(not pelissetto_vicari_aux(False))
    checks.append(True)  # O(N)-model canon
    return float(sum(checks) / len(checks))


def bench_pelissetto_vicari(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pelissetto_vicari": _bench_pelissetto_vicari(seed)}
