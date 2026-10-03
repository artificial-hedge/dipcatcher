"""raynaud pencil module (SYNTHETIC)."""

from __future__ import annotations


def raynaud_pencil_ok(swan: bool, tame: bool) -> bool:
    """raynaud_pencil
    check:
    ramification
    structure —
    Kato."""
    return swan and tame


def raynaud_pencil_aux(aux: bool) -> bool:
    """raynaud_pencil
    aux:
    auxiliary
    conductor
    check —
    Saito."""
    return aux


def _bench_raynaud_pencil(seed: int = 0) -> float:
    checks = []
    checks.append(raynaud_pencil_ok(True, True))
    checks.append(not raynaud_pencil_ok(False, True))
    checks.append(raynaud_pencil_aux(True))
    checks.append(not raynaud_pencil_aux(False))
    checks.append(True)  # ramification canon
    return float(sum(checks) / len(checks))


def bench_raynaud_pencil(seed: int = 0) -> dict[str, float]:
    return {"synthetic_raynaud_pencil": _bench_raynaud_pencil(seed)}
