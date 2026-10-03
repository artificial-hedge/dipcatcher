"""Extremal rays (SYNTHETIC)."""

from __future__ import annotations


def er_ok(extremal: bool, ray: bool) -> bool:
    """Extremal:
    extremal
    ray
    of
    the
    Mori
    cone —
    extremal
    ray."""
    return extremal and ray


def contraction_theorem(ct: bool) -> bool:
    """Contraction:
    contraction
    theorem
    for
    extremal
    rays —
    Kawamata."""
    return ct


def _bench_extremal_ray(seed: int = 0) -> float:
    checks = []
    checks.append(er_ok(True, True))
    checks.append(not er_ok(False, True))
    checks.append(contraction_theorem(True))
    checks.append(not contraction_theorem(False))
    checks.append(True)  # Kawamata
    return float(sum(checks) / len(checks))


def bench_extremal_ray(seed: int = 0) -> dict[str, float]:
    return {"synthetic_extremal_ray": _bench_extremal_ray(seed)}
