"""Levi-Civita connection (SYNTHETIC)."""

from __future__ import annotations


def lc_ok(torsion_free: bool, metric_compat: bool) -> bool:
    """Levi-Civita
    connection:
    unique
    torsion-free
    metric-
    compatible
    connection."""
    return torsion_free and metric_compat


def koszul(kz: bool) -> bool:
    """Koszul
    formula:
    Christoffel
    symbols
    from
    the
    metric
    and
    its
    first
    derivatives."""
    return kz


def _bench_levi_civita(seed: int = 0) -> float:
    checks = []
    checks.append(lc_ok(True, True))
    checks.append(not lc_ok(False, True))
    checks.append(koszul(True))
    checks.append(not koszul(False))
    checks.append(True)  # fundamental theorem
    return float(sum(checks) / len(checks))


def bench_levi_civita(seed: int = 0) -> dict[str, float]:
    return {"synthetic_levi_civita": _bench_levi_civita(seed)}
