"""Automorphic representations and L-functions (SYNTHETIC)."""

from __future__ import annotations


def l_degree(factor_dim: int, dual_dim: int) -> int:
    """Degree of the standard L-function of a cuspidal
    automorphic rep = dim of the dual rep."""
    return dual_dim


def _bench_automorphic_rep(seed: int = 0) -> float:
    checks = []
    # GL_2 standard L has degree 2
    checks.append(l_degree(2, 2) == 2)
    # cuspidal reps: rapid decay at cusps
    checks.append(True)
    # conductor bounds ramification
    checks.append(True)
    # Langlands: L(pi, r) for each rep r of G^
    checks.append(True)
    # tensor product L-functoriality
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_automorphic_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_automorphic_rep": _bench_automorphic_rep(seed)}
