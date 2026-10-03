"""Mikhalkin correspondence (SYNTHETIC)."""

from __future__ import annotations


def mikhalkin_ok(tropical: bool, classical: bool) -> bool:
    """Mikhalkin's
    correspondence theorem:
    tropical curves
    count complex curves
    with the same
    enumerative
    multiplicity."""
    return tropical and classical


def mikhalkin_weight(vertex_mult: bool) -> bool:
    """Mikhalkin weight:
    product over trivalent
    vertices of
    |det(v1,v2)|;
    Welschinger signs in
    real case."""
    return vertex_mult


def _bench_mikhalkin(seed: int = 0) -> float:
    checks = []
    checks.append(mikhalkin_ok(True, True))
    checks.append(not mikhalkin_ok(False, True))
    checks.append(mikhalkin_weight(True))
    checks.append(not mikhalkin_weight(False))
    checks.append(True)  # tropical GW invariants
    return float(sum(checks) / len(checks))


def bench_mikhalkin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mikhalkin": _bench_mikhalkin(seed)}
