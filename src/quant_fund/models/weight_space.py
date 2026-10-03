"""Weight-space decomposition of sl2 modules (SYNTHETIC)."""

from __future__ import annotations


def weights(n: int) -> list[int]:
    """Weights of the (n+1)-dim irrep: n, n-2, ..., -n."""
    return [n - 2 * i for i in range(n + 1)]


def _bench_weight_space(seed: int = 0) -> float:
    checks = []
    # standard rep: weights +1, -1
    checks.append(weights(1) == [1, -1])
    # adjoint: 2, 0, -2
    checks.append(weights(2) == [2, 0, -2])
    # each weight space 1-dim in an irrep
    checks.append(len(weights(4)) == 5)
    # symmetric about 0
    w = weights(3)
    checks.append(w == [-x for x in reversed(w)])
    # highest weight determines the irrep
    checks.append(max(weights(5)) == 5)
    return float(sum(checks) / len(checks))


def bench_weight_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weight_space": _bench_weight_space(seed)}
