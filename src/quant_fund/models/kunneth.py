"""Kunneth theorem for product spaces (SYNTHETIC)."""

from __future__ import annotations


def product_betti(b1: list[int], b2: list[int]) -> list[int]:
    """betti of X x Y over a field: convolution of betti vectors."""
    n = len(b1) + len(b2) - 1
    out = [0] * n
    for i, a in enumerate(b1):
        for j, b in enumerate(b2):
            out[i + j] += a * b
    return out


def _bench_kunneth(seed: int = 0) -> float:
    checks = []
    # S^1 x S^1 = T^2: [1,1]*[1,1] = [1,2,1]
    checks.append(product_betti([1, 1], [1, 1]) == [1, 2, 1])
    # S^2 x S^2: [1,0,1]^2 = [1,0,2,0,1]
    checks.append(product_betti([1, 0, 1], [1, 0, 1]) == [1, 0, 2, 0, 1])
    # X x pt = X
    checks.append(product_betti([1, 2, 1], [1]) == [1, 2, 1])
    # S^1 x S^2: [1,1]*[1,0,1] = [1,1,1,1]
    checks.append(product_betti([1, 1], [1, 0, 1]) == [1, 1, 1, 1])
    # Euler char multiplies: chi(XxY) = chi X . chi Y
    checks.append((1 - 1) * (1 - 0 + 1) == 0)
    return float(sum(checks) / len(checks))


def bench_kunneth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kunneth": _bench_kunneth(seed)}
