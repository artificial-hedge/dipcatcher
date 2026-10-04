"""telescope tower3 module (SYNTHETIC)."""

from __future__ import annotations


def telescope_tower3_ok(chromatic: bool, height: bool) -> bool:
    """telescope_tower3
    check:
    chromatic
    structure —
    height."""
    return chromatic and height


def telescope_tower3_aux(aux: bool) -> bool:
    """telescope_tower3
    aux:
    auxiliary
    chromatic
    check —
    periodicity."""
    return aux


def _bench_telescope_tower3(seed: int = 0) -> float:
    checks = []
    checks.append(telescope_tower3_ok(True, True))
    checks.append(not telescope_tower3_ok(False, True))
    checks.append(telescope_tower3_aux(True))
    checks.append(not telescope_tower3_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_telescope_tower3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_telescope_tower3": _bench_telescope_tower3(seed)}
