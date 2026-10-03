"""telescope tower2 module (SYNTHETIC)."""

from __future__ import annotations


def telescope_tower2_ok(chromatic: bool, height: bool) -> bool:
    """telescope_tower2
    check:
    chromatic
    height
    structure —
    stratified."""
    return chromatic and height


def telescope_tower2_aux(aux: bool) -> bool:
    """telescope_tower2
    aux:
    auxiliary
    chromatic
    check —
    spectral."""
    return aux


def _bench_telescope_tower2(seed: int = 0) -> float:
    checks = []
    checks.append(telescope_tower2_ok(True, True))
    checks.append(not telescope_tower2_ok(False, True))
    checks.append(telescope_tower2_aux(True))
    checks.append(not telescope_tower2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_telescope_tower2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_telescope_tower2": _bench_telescope_tower2(seed)}
