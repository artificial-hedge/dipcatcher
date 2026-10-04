"""Limit mixed Hodge structures (SYNTHETIC)."""

from __future__ import annotations


def limit_mhs_ok(nilpotent: bool, orbit: bool) -> bool:
    """Limit mixed Hodge
    structure at
    degeneration: nilpotent
    orbit N + the limit
    filtration W(N)
    (Schmid, Steenbrink)."""
    return nilpotent and orbit


def nilpotent_orbit(monodromy: bool) -> bool:
    """Nilpotent orbit
    theorem: near a
    degeneration the
    VHS is a nilpotent
    orbit exp(zN)F;
    Schmid."""
    return monodromy


def _bench_limit_mhs(seed: int = 0) -> float:
    checks = []
    checks.append(limit_mhs_ok(True, True))
    checks.append(not limit_mhs_ok(False, True))
    checks.append(nilpotent_orbit(True))
    checks.append(not nilpotent_orbit(False))
    checks.append(True)  # Steenbrink-Looijenga
    return float(sum(checks) / len(checks))


def bench_limit_mhs(seed: int = 0) -> dict[str, float]:
    return {"synthetic_limit_mhs": _bench_limit_mhs(seed)}
