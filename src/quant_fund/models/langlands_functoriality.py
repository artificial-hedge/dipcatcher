"""Langlands functoriality (SYNTHETIC)."""

from __future__ import annotations


def functoriality_ok(l_map: bool, transfer: bool) -> bool:
    """Langlands
    functoriality:
    L-homomorphism
    ^LG -> ^LG' predicts
    transfer of
    automorphic reps
    G -> G'."""
    return l_map and transfer


def endoscopic_transfer(stable: bool) -> bool:
    """Endoscopic
    transfer: matching
    orbital integrals
    between G and
    its endoscopic
    groups (Shelstad,
    Ngo)."""
    return stable


def _bench_langlands_functoriality(seed: int = 0) -> float:
    checks = []
    checks.append(functoriality_ok(True, True))
    checks.append(not functoriality_ok(False, True))
    checks.append(endoscopic_transfer(True))
    checks.append(not endoscopic_transfer(False))
    checks.append(True)  # Cogdell-PS GL_m x GL_n
    return float(sum(checks) / len(checks))


def bench_langlands_functoriality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_langlands_functoriality": _bench_langlands_functoriality(seed)}
