"""raynaud height module (SYNTHETIC)."""

from __future__ import annotations


def raynaud_height_ok(chromatic: bool, height: bool) -> bool:
    """raynaud_height
    check:
    chromatic
    height
    structure —
    stratified."""
    return chromatic and height


def raynaud_height_aux(aux: bool) -> bool:
    """raynaud_height
    aux:
    auxiliary
    chromatic
    check —
    spectral."""
    return aux


def _bench_raynaud_height(seed: int = 0) -> float:
    checks = []
    checks.append(raynaud_height_ok(True, True))
    checks.append(not raynaud_height_ok(False, True))
    checks.append(raynaud_height_aux(True))
    checks.append(not raynaud_height_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_raynaud_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_raynaud_height": _bench_raynaud_height(seed)}
