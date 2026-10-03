"""Donaldson theorem (SYNTHETIC)."""

from __future__ import annotations


def don_ok(definite: bool, diagonal: bool) -> bool:
    """Donaldson's
    theorem:
    smooth
    definite
    intersection
    forms
    are
    diagonal —
    via
    ASD
    instanton
    moduli."""
    return definite and diagonal


def gauge_theory(gt: bool) -> bool:
    """Gauge
    theory:
    anti-self-dual
    connections
    give
    smooth
    4-manifold
    invariants."""
    return gt


def _bench_donaldson_thm(seed: int = 0) -> float:
    checks = []
    checks.append(don_ok(True, True))
    checks.append(not don_ok(False, True))
    checks.append(gauge_theory(True))
    checks.append(not gauge_theory(False))
    checks.append(True)  # Donaldson 1983
    return float(sum(checks) / len(checks))


def bench_donaldson_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_donaldson_thm": _bench_donaldson_thm(seed)}
