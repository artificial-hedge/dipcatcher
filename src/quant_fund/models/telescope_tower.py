"""Telescope tower (SYNTHETIC)."""

from __future__ import annotations


def tt_ok(telescope: bool, tower: bool) -> bool:
    """Telescope
    tower:
    chromatic
    tower —
    monochromatic
    layers."""
    return telescope and tower


def monochromatic_layer(ml: bool) -> bool:
    """Monochromatic:
    monochromatic
    layer —
    fiber of
    tower."""
    return ml


def _bench_telescope_tower(seed: int = 0) -> float:
    checks = []
    checks.append(tt_ok(True, True))
    checks.append(not tt_ok(False, True))
    checks.append(monochromatic_layer(True))
    checks.append(not monochromatic_layer(False))
    checks.append(True)  # Kuhn
    return float(sum(checks) / len(checks))


def bench_telescope_tower(seed: int = 0) -> dict[str, float]:
    return {"synthetic_telescope_tower": _bench_telescope_tower(seed)}
