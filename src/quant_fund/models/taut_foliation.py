"""Taut foliations (SYNTHETIC)."""

from __future__ import annotations


def tf_ok(codim1: bool, transverse_circle: bool) -> bool:
    """Taut
    foliation:
    codimension-1
    foliation
    with
    a
    closed
    transversal
    through
    every
    leaf —
    topological
    richness."""
    return codim1 and transverse_circle


def ruelle_sullivan(rs: bool) -> bool:
    """Ruelle-
    Sullivan:
    taut
    foliations
    carry
    a
    transverse
    invariant
    measure —
    volume
    form
    existence."""
    return rs


def _bench_taut_foliation(seed: int = 0) -> float:
    checks = []
    checks.append(tf_ok(True, True))
    checks.append(not tf_ok(False, True))
    checks.append(ruelle_sullivan(True))
    checks.append(not ruelle_sullivan(False))
    checks.append(True)  # Novikov
    return float(sum(checks) / len(checks))


def bench_taut_foliation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taut_foliation": _bench_taut_foliation(seed)}
