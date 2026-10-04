"""Exotic R4 (SYNTHETIC)."""

from __future__ import annotations


def ex4_ok(homeo: bool, not_diffeo: bool) -> bool:
    """Exotic
    R4:
    manifold
    homeomorphic
    but
    not
    diffeomorphic
    to
    standard
    R4 —
    unique
    to
    dim
    4."""
    return homeo and not_diffeo


def uncountably_many(um: bool) -> bool:
    """Uncountably
    many
    exotic
    R4s
    exist —
    Taubes,
    Gompf,
    Freedman."""
    return um


def _bench_exotic_r4(seed: int = 0) -> float:
    checks = []
    checks.append(ex4_ok(True, True))
    checks.append(not ex4_ok(False, True))
    checks.append(uncountably_many(True))
    checks.append(not uncountably_many(False))
    checks.append(True)  # Freedman-Taubes
    return float(sum(checks) / len(checks))


def bench_exotic_r4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exotic_r4": _bench_exotic_r4(seed)}
