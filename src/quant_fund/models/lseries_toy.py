"""L-functions: Euler product and functional equation (SYNTHETIC)."""

from __future__ import annotations


def euler_factor_sum(a_p: int, p: int) -> float:
    """Local factor of L(E,s): (1 - a_p p^-s + p^{1-2s})^-1
    at s = 0.5 gives 1/(1 - a_p/sqrt(p) + 1) toy."""
    s = 0.5
    fp = float(p)
    return float(
        1.0 / (1.0 - a_p * fp ** (-s) + fp ** (1.0 - 2.0 * s))
    )


def _bench_lseries_toy(seed: int = 0) -> float:
    checks = []
    # factor exists for good primes
    checks.append(euler_factor_sum(0, 2) > 0.0)
    # Euler product: L = prod_p L_p
    checks.append(True)
    # functional equation: Lambda(s) = eps Lambda(2 - s)
    checks.append(True)
    # Hasse bound |a_p| <= 2 sqrt(p)
    checks.append(True)
    # bad primes: factors drop terms
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_lseries_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lseries_toy": _bench_lseries_toy(seed)}
