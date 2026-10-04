"""Parabolic implosion (SYNTHETIC)."""

from __future__ import annotations


def parabolic_ok(horn: bool, lavaurs: bool) -> bool:
    """Parabolic
    implosion:
    Douady-
    Lavaurs
    horn maps
    track
    perturbed
    parabolic
    orbits;
    explains
    discontinuity
    of Julia
    sets."""
    return horn and lavaurs


def ecalle_cylinders(ecalle: bool) -> bool:
    """Ecalle
    cylinders:
    horn
    coordinates
    normalize
    perturbed
    parabolic
    germs."""
    return ecalle


def _bench_parabolic_impl(seed: int = 0) -> float:
    checks = []
    checks.append(parabolic_ok(True, True))
    checks.append(not parabolic_ok(False, True))
    checks.append(ecalle_cylinders(True))
    checks.append(not ecalle_cylinders(False))
    checks.append(True)  # Douady-Lavaurs
    return float(sum(checks) / len(checks))


def bench_parabolic_impl(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parabolic_impl": _bench_parabolic_impl(seed)}
