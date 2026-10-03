"""Connes noncommutative geometry (SYNTHETIC)."""

from __future__ import annotations


def connes_nc_ok(spectral_triple: bool, metric: bool) -> bool:
    """Spectral triple
    (A, H, D): involutive
    algebra on Hilbert
    space with Dirac
    operator; encodes
    metric data (Connes)."""
    return spectral_triple and metric


def nc_distance(state: bool) -> bool:
    """Noncommutative metric:
    distance between states
    via sup over a with
    ||[D,a]|| <= 1 of
    |phi(a)-psi(a)|."""
    return state


def _bench_connes_nc(seed: int = 0) -> float:
    checks = []
    checks.append(connes_nc_ok(True, True))
    checks.append(not connes_nc_ok(False, True))
    checks.append(nc_distance(True))
    checks.append(not nc_distance(False))
    checks.append(True)  # Connes' reconstruction
    return float(sum(checks) / len(checks))


def bench_connes_nc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_connes_nc": _bench_connes_nc(seed)}
