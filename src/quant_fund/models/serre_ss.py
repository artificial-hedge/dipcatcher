"""Serre spectral sequence bookkeeping on products (SYNTHETIC)."""

from __future__ import annotations


def kunneth_betti(b1: list[int], b2: list[int]) -> list[int]:
    """Betti numbers of X x Y via the Kunneth formula (field coefficients):
    b_n(X x Y) = sum_{i+j=n} b_i(X) b_j(Y)."""
    n = len(b1) + len(b2) - 1
    out = [0] * n
    for i, a in enumerate(b1):
        for j, b in enumerate(b2):
            out[i + j] += a * b
    return out


def euler_from_betti(b: list[int]) -> int:
    return int(sum((-1) ** i * x for i, x in enumerate(b)))


def _bench_serre_ss(seed: int = 0) -> float:
    checks = []
    s1 = [1, 1]
    s2 = [1, 0, 1]
    # T2 = S1 x S1: [1,2,1]
    checks.append(kunneth_betti(s1, s1) == [1, 2, 1])
    # S2 x S1: [1,1,1,1]
    checks.append(kunneth_betti(s2, s1) == [1, 1, 1, 1])
    # S2 x S2: [1,0,2,0,1]
    checks.append(kunneth_betti(s2, s2) == [1, 0, 2, 0, 1])
    # Euler char multiplicative through Kunneth
    checks.append(
        euler_from_betti(kunneth_betti(s1, s2)) == euler_from_betti(s1) * euler_from_betti(s2)
    )
    # T3: [1,3,3,1]
    checks.append(kunneth_betti([1, 2, 1], s1) == [1, 3, 3, 1])
    # CP2 x S1: CP2 betti [1,0,1,0,1] -> [1,1,1,1,1,1]
    checks.append(kunneth_betti([1, 0, 1, 0, 1], s1) == [1, 1, 1, 1, 1, 1])
    return float(sum(checks) / len(checks))


def bench_serre_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_serre_ss": _bench_serre_ss(seed)}
