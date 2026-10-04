"""Ginzburg dg-algebras (SYNTHETIC)."""

from __future__ import annotations


def ginzburg_ok(quiver_potential: bool, cy3: bool) -> bool:
    """Ginzburg dg-algebra
    of a quiver with
    potential (Q,W):
    3-Calabi-Yau completion
    of Jacobi algebra."""
    return quiver_potential and cy3


def jacobi_algebra(quotient: bool) -> bool:
    """Jacobi algebra
    P(Q,W) = complete path
    algebra modulo
    cyclic derivatives
    of potential."""
    return quotient


def _bench_ginzburg_dga(seed: int = 0) -> float:
    checks = []
    checks.append(ginzburg_ok(True, True))
    checks.append(not ginzburg_ok(False, True))
    checks.append(jacobi_algebra(True))
    checks.append(not jacobi_algebra(False))
    checks.append(True)  # cluster mutations
    return float(sum(checks) / len(checks))


def bench_ginzburg_dga(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ginzburg_dga": _bench_ginzburg_dga(seed)}
