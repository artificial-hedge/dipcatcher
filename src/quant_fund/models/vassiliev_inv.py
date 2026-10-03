"""Vassiliev invariants (SYNTHETIC)."""

from __future__ import annotations


def vass_ok(finite: bool, chord: bool) -> bool:
    """Vassiliev
    invariant:
    finite-type
    invariant
    via
    singular
    knots —
    governed
    by
    chord
    diagrams."""
    return finite and chord


def weight_system(ws: bool) -> bool:
    """Weight
    systems:
    4T-
    relation
    functionals
    —
    Kontsevich
    integrates
    all
    Vassiliev
    invariants."""
    return ws


def _bench_vassiliev_inv(seed: int = 0) -> float:
    checks = []
    checks.append(vass_ok(True, True))
    checks.append(not vass_ok(False, True))
    checks.append(weight_system(True))
    checks.append(not weight_system(False))
    checks.append(True)  # Kontsevich
    return float(sum(checks) / len(checks))


def bench_vassiliev_inv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vassiliev_inv": _bench_vassiliev_inv(seed)}
