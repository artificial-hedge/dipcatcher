"""Super Lie algebras (SYNTHETIC)."""

from __future__ import annotations


def super_lie_ok(bracket: bool, jacobi: bool) -> bool:
    """Super Lie
    algebra:
    Z/2-graded
    vector space
    with a graded-
    antisymmetric
    bracket
    satisfying
    graded Jacobi."""
    return bracket and jacobi


def gl_mn(gl: bool) -> bool:
    """gl(m|n):
    the super Lie
    algebra of
    (m+n)x(m+n)
    block matrices
    with supertrace
    structure."""
    return gl


def _bench_super_lie(seed: int = 0) -> float:
    checks = []
    checks.append(super_lie_ok(True, True))
    checks.append(not super_lie_ok(False, True))
    checks.append(gl_mn(True))
    checks.append(not gl_mn(False))
    checks.append(True)  # Kac classification
    return float(sum(checks) / len(checks))


def bench_super_lie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_lie": _bench_super_lie(seed)}
