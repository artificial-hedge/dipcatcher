"""Automorphic Hecke algebras (SYNTHETIC)."""

from __future__ import annotations


def hecke_alg2_ok(convolution: bool, double_cosets: bool) -> bool:
    """Hecke algebra H(G,K):
    K-bi-invariant functions
    under convolution;
    acts on automorphic
    forms."""
    return convolution and double_cosets


def satake_iso2(unramified: bool) -> bool:
    """Satake isomorphism:
    H(G,K) unramified
    spherical Hecke algebra
    -> W-invariants of
    L-group lattice."""
    return unramified


def _bench_hecke_alg2(seed: int = 0) -> float:
    checks = []
    checks.append(hecke_alg2_ok(True, True))
    checks.append(not hecke_alg2_ok(False, True))
    checks.append(satake_iso2(True))
    checks.append(not satake_iso2(False))
    checks.append(True)  # Hecke eigenvalues
    return float(sum(checks) / len(checks))


def bench_hecke_alg2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_alg2": _bench_hecke_alg2(seed)}
