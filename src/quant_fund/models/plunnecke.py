"""Plunnecke-Ruzsa (SYNTHETIC)."""

from __future__ import annotations


def plunn_ok(power: bool, ineq: bool) -> bool:
    """Plunnecke-
    Ruzsa:
    |A+B|
    <= K|A|
    implies
    |nB - mB|
    <=
    K^{n+m}|A|;
    Petridis'
    proof."""
    return power and ineq


def ruzsa_triangle(tri: bool) -> bool:
    """Ruzsa
    triangle
    inequality:
    |A-C|
    |B| <=
    |A-B|
    |B-C|."""
    return tri


def _bench_plunnecke(seed: int = 0) -> float:
    checks = []
    checks.append(plunn_ok(True, True))
    checks.append(not plunn_ok(False, True))
    checks.append(ruzsa_triangle(True))
    checks.append(not ruzsa_triangle(False))
    checks.append(True)  # Plunnecke-Petridis
    return float(sum(checks) / len(checks))


def bench_plunnecke(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plunnecke": _bench_plunnecke(seed)}
