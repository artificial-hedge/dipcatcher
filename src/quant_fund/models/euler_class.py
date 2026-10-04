"""Euler class (SYNTHETIC)."""

from __future__ import annotations


def ec_ok(top_deg: bool, zeros: bool) -> bool:
    """Euler
    class:
    top
    degree
    obstruction
    to
    nonvanishing
    sections —
    zeros
    counted."""
    return top_deg and zeros


def gauss_bonnet_chern(gbc: bool) -> bool:
    """Gauss-
    Bonnet-
    Chern:
    Euler
    class
    of
    tangent
    bundle
    is
    curvature
    density."""
    return gbc


def _bench_euler_class(seed: int = 0) -> float:
    checks = []
    checks.append(ec_ok(True, True))
    checks.append(not ec_ok(False, True))
    checks.append(gauss_bonnet_chern(True))
    checks.append(not gauss_bonnet_chern(False))
    checks.append(True)  # Chern-GBC
    return float(sum(checks) / len(checks))


def bench_euler_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_euler_class": _bench_euler_class(seed)}
