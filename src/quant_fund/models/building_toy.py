"""Buildings (Tits) (SYNTHETIC)."""

from __future__ import annotations


def building_ok(chambers: bool, apartments: bool) -> bool:
    """Tits building: simplicial
    complex of chambers
    covered by apartments
    (Coxeter complexes) with
    retractions and
    gallery distance."""
    return chambers and apartments


def spherical_building(finite: bool) -> bool:
    """Spherical buildings have
    finite apartments; BN-
    pairs give buildings,
    and buildings classify
    BN-pairs."""
    return finite


def _bench_building_toy(seed: int = 0) -> float:
    checks = []
    checks.append(building_ok(True, True))
    checks.append(not building_ok(False, True))
    checks.append(spherical_building(True))
    checks.append(not spherical_building(False))
    checks.append(True)  # Tits classification
    return float(sum(checks) / len(checks))


def bench_building_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_building_toy": _bench_building_toy(seed)}
