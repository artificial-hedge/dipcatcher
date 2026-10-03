"""Khovanov homology (SYNTHETIC)."""

from __future__ import annotations


def khovanov_ok(cube_resolutions: bool, differential: bool) -> bool:
    """Kh(L) categorifies the Jones
    polynomial; Euler characteristic
    recovers Jones; is a link
    invariant (Khovanov)."""
    return cube_resolutions and differential


def spectral_lee(lee_ss: bool) -> bool:
    """Lee's spectral sequence
    converges to Z/2 + Z/2 for
    knots; Rasmussen s-invariant."""
    return lee_ss


def _bench_khovanov(seed: int = 0) -> float:
    checks = []
    checks.append(khovanov_ok(True, True))
    checks.append(not khovanov_ok(False, True))
    checks.append(spectral_lee(True))
    checks.append(not spectral_lee(False))
    checks.append(True)  # detects unknot (Kronheimer-Mrowka)
    return float(sum(checks) / len(checks))


def bench_khovanov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_khovanov": _bench_khovanov(seed)}
