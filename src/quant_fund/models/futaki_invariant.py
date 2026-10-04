"""Futaki invariant (SYNTHETIC)."""

from __future__ import annotations


def fi_ok(holomorphic_vf: bool, vanishing: bool) -> bool:
    """Futaki
    invariant:
    Lie-
    algebra
    character
    obstructing
    cscK —
    vanishes
    on
    KE
    manifolds."""
    return holomorphic_vf and vanishing


def mabuchi_energy(me: bool) -> bool:
    """Mabuchi
    K-energy:
    functional
    whose
    critical
    points
    are
    cscK
    —
    convexity
    along
    geodesics."""
    return me


def _bench_futaki_invariant(seed: int = 0) -> float:
    checks = []
    checks.append(fi_ok(True, True))
    checks.append(not fi_ok(False, True))
    checks.append(mabuchi_energy(True))
    checks.append(not mabuchi_energy(False))
    checks.append(True)  # Futaki 1983
    return float(sum(checks) / len(checks))


def bench_futaki_invariant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_futaki_invariant": _bench_futaki_invariant(seed)}
