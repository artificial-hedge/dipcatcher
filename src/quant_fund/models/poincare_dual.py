"""Poincare duality on closed surfaces: b_k = b_{n-k} (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.spectral_seq_toy import product_betti


def _bench_poincare_dual(seed: int = 0) -> float:
    checks = []
    # S^2: (1,0,1) symmetric
    s2 = [1, 0, 1]
    checks.append(s2 == s2[::-1])
    # T^2: (1,2,1) symmetric
    t2 = [1, 2, 1]
    checks.append(t2 == t2[::-1])
    # genus g orientable: chi even and b0 = b2 = 1
    for g in (0, 1, 2, 3):
        b = [1, 2 * g, 1]
        checks.append(b[0] == b[2])
        checks.append(sum((-1) ** i * v for i, v in enumerate(b)) == 2 - 2 * g)
    # T^3 = T^2 x S^1: (1,3,3,1) symmetric
    t3 = product_betti(t2, [1, 1])
    checks.append(t3 == t3[::-1])
    # S^1 x S^2 = (1,1,1,1): 3-manifold closed -> symmetric
    s1s2 = product_betti([1, 1], s2)
    checks.append(s1s2 == s1s2[::-1])
    # non-closed counterexample: disc b=(1,0,0) not symmetric -> excluded by
    # hypothesis; verify asymmetry
    checks.append([1, 0, 0][::-1] != [1, 0, 0])
    return float(sum(checks) / len(checks))


def bench_poincare_dual(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poincare_dual": _bench_poincare_dual(seed)}
