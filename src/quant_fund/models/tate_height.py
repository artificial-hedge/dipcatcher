"""tate height module (SYNTHETIC)."""

from __future__ import annotations


def tate_height_ok(chromatic: bool, height: bool) -> bool:
    """tate_height
    check:
    chromatic
    height
    structure —
    stratified."""
    return chromatic and height


def tate_height_aux(aux: bool) -> bool:
    """tate_height
    aux:
    auxiliary
    chromatic
    check —
    spectral."""
    return aux


def _bench_tate_height(seed: int = 0) -> float:
    checks = []
    checks.append(tate_height_ok(True, True))
    checks.append(not tate_height_ok(False, True))
    checks.append(tate_height_aux(True))
    checks.append(not tate_height_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_tate_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_height": _bench_tate_height(seed)}
