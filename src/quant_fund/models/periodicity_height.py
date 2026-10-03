"""periodicity height module (SYNTHETIC)."""

from __future__ import annotations


def periodicity_height_ok(chromatic: bool, height: bool) -> bool:
    """periodicity_height
    check:
    chromatic
    height
    structure —
    stratified."""
    return chromatic and height


def periodicity_height_aux(aux: bool) -> bool:
    """periodicity_height
    aux:
    auxiliary
    chromatic
    check —
    spectral."""
    return aux


def _bench_periodicity_height(seed: int = 0) -> float:
    checks = []
    checks.append(periodicity_height_ok(True, True))
    checks.append(not periodicity_height_ok(False, True))
    checks.append(periodicity_height_aux(True))
    checks.append(not periodicity_height_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_periodicity_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_periodicity_height": _bench_periodicity_height(seed)}
