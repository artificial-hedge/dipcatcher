"""Poincare duality on closed orientable manifolds (SYNTHETIC)."""

from __future__ import annotations


def betti_dual(betti: list[int]) -> bool:
    """Poincare duality: b_k = b_{n-k} for closed orientable n-manifold."""
    n = len(betti) - 1
    return all(betti[k] == betti[n - k] for k in range(n + 1))


def _bench_poincare_duality2(seed: int = 0) -> float:
    checks = []
    # S^3: b = [1,0,0,1]
    checks.append(betti_dual([1, 0, 0, 1]))
    # T^2: b = [1,2,1]
    checks.append(betti_dual([1, 2, 1]))
    # S^1 v S^2 (non-dual): [1,1,1] fails at k=0 vs 2? [1,1,1] IS palindromic
    checks.append(betti_dual([1, 1, 1]))
    # CP^2: [1,0,1,0,1]
    checks.append(betti_dual([1, 0, 1, 0, 1]))
    # non-closed: [1,2,0] fails
    checks.append(not betti_dual([1, 2, 0]))
    return float(sum(checks) / len(checks))


def bench_poincare_duality2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poincare_duality2": _bench_poincare_duality2(seed)}
