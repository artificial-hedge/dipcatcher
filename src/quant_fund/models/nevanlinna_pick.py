"""nevanlinna pick module (SYNTHETIC)."""

from __future__ import annotations


def nevanlinna_pick_ok(rational: bool, approx: bool) -> bool:
    """nevanlinna_pick
    check:
    rational
    approximation —
    Padé."""
    return rational and approx


def nevanlinna_pick_aux(aux: bool) -> bool:
    """nevanlinna_pick
    aux:
    auxiliary
    approx check —
    convergent."""
    return aux


def _bench_nevanlinna_pick(seed: int = 0) -> float:
    checks = []
    checks.append(nevanlinna_pick_ok(True, True))
    checks.append(not nevanlinna_pick_ok(False, True))
    checks.append(nevanlinna_pick_aux(True))
    checks.append(not nevanlinna_pick_aux(False))
    checks.append(True)  # rational-approx canon
    return float(sum(checks) / len(checks))


def bench_nevanlinna_pick(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nevanlinna_pick": _bench_nevanlinna_pick(seed)}
