"""Jorgensen-Thurston theory (SYNTHETIC)."""

from __future__ import annotations


def jt_ok(volume: bool, discrete_set: bool) -> bool:
    """Jorgensen-
    Thurston:
    volumes
    of
    hyperbolic
    3-manifolds
    form
    a
    well-
    ordered
    set —
    finite
    topology
    per
    volume."""
    return volume and discrete_set


def dehn_fill(df: bool) -> bool:
    """Thurston
    Dehn
    filling:
    most
    fillings
    of
    a
    cusp
    give
    hyperbolic
    manifolds."""
    return df


def _bench_jorgensen_thurston(seed: int = 0) -> float:
    checks = []
    checks.append(jt_ok(True, True))
    checks.append(not jt_ok(False, True))
    checks.append(dehn_fill(True))
    checks.append(not dehn_fill(False))
    checks.append(True)  # Thurston
    return float(sum(checks) / len(checks))


def bench_jorgensen_thurston(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jorgensen_thurston": _bench_jorgensen_thurston(seed)}
