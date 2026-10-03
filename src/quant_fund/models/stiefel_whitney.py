"""Stiefel-Whitney classes for RP^n tangent bundles (SYNTHETIC)."""

from __future__ import annotations


def sw_rpn(n: int, i: int) -> int:
    """w_i(tau RP^n) = C(n+1, i) mod 2 for 0 <= i <= n (truncated
    binomial (1+a)^{n+1}); zero beyond n."""
    from math import comb

    if i < 0 or i > n:
        return 0
    return comb(n + 1, i) % 2


def orientable(n_sw1: int) -> bool:
    """A bundle is orientable iff w1 = 0."""
    return n_sw1 == 0


def _bench_stiefel_whitney(seed: int = 0) -> float:
    checks = []
    # w(tau RP^2) = (1+a)^3 truncated at degree 2 = 1 + a + a^2
    checks.append(sw_rpn(2, 0) == 1)
    checks.append(sw_rpn(2, 1) == 1)  # C(3,1) = 3 mod 2 = 1
    checks.append(sw_rpn(2, 2) == 1)  # C(3,2) = 3 mod 2 = 1
    # RP^2 non-orientable
    checks.append(not orientable(sw_rpn(2, 1)))
    # w(tau RP^3) = (1+a)^4 = 1 + 0a + 0a^2 + 0a^3 + a^4 -> truncated: 1
    checks.append(sw_rpn(3, 1) == 0)  # C(4,1)=4 mod 2=0 -> RP^3 orientable
    checks.append(orientable(sw_rpn(3, 1)))
    # w(S^n) = 1 (stably trivial)
    checks.append(sw_rpn(0, 0) == 1)
    # beyond dimension
    checks.append(sw_rpn(2, 3) == 0)
    return float(sum(checks) / len(checks))


def bench_stiefel_whitney(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stiefel_whitney": _bench_stiefel_whitney(seed)}
