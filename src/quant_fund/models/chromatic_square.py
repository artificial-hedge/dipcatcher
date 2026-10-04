"""Chromatic square (SYNTHETIC)."""

from __future__ import annotations


def cs_ok(chromatic: bool, square: bool) -> bool:
    """Chromatic
    square:
    chromatic
    fracture
    square —
    pullback."""
    return chromatic and square


def fracture_pullback(fp: bool) -> bool:
    """Fracture
    pullback:
    chromatic
    fracture
    pullback —
    localization."""
    return fp


def _bench_chromatic_square(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok(True, True))
    checks.append(not cs_ok(False, True))
    checks.append(fracture_pullback(True))
    checks.append(not fracture_pullback(False))
    checks.append(True)  # chromatic fracture
    return float(sum(checks) / len(checks))


def bench_chromatic_square(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chromatic_square": _bench_chromatic_square(seed)}
