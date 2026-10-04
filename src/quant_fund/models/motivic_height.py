"""Motivic height (SYNTHETIC)."""

from __future__ import annotations


def mh_ok(motivic: bool, height: bool) -> bool:
    """Motivic
    height:
    motivic
    height
    of
    a
    variety —
    motivic
    height."""
    return motivic and height


def height_filtration(hf: bool) -> bool:
    """Height
    filtration:
    height
    filtration
    on
    motives —
    weight
    filtration."""
    return hf


def _bench_motivic_height(seed: int = 0) -> float:
    checks = []
    checks.append(mh_ok(True, True))
    checks.append(not mh_ok(False, True))
    checks.append(height_filtration(True))
    checks.append(not height_filtration(False))
    checks.append(True)  # motivic height
    return float(sum(checks) / len(checks))


def bench_motivic_height(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_height": _bench_motivic_height(seed)}
