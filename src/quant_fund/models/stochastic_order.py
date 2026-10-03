"""First-order stochastic dominance via CDF comparisons (SYNTHETIC)."""

from __future__ import annotations


def cdf_samples(x: list[float], t: float) -> float:
    return sum(1 for v in x if v <= t) / len(x)


def fosd(x: list[float], y: list[float], grid: list[float]) -> bool:
    """X >=_FOSD Y iff F_X(t) <= F_Y(t) for all t."""
    return all(cdf_samples(x, t) <= cdf_samples(y, t) + 0.02 for t in grid)


def ssd_check(x: list[float], y: list[float], grid: list[float]) -> bool:
    """Second-order dominance: integral of CDF_X <= integral CDF_Y."""
    for t in grid:
        ix = sum(cdf_samples(x, s) for s in grid if s <= t)
        iy = sum(cdf_samples(y, s) for s in grid if s <= t)
        if ix > iy + 0.05 * len([s for s in grid if s <= t]):
            return False
    return True


def _bench_stochastic_order(seed: int = 0) -> float:
    checks = []
    import numpy as np

    rng = np.random.default_rng(seed)
    x = list(rng.normal(1.0, 1.0, 50_000))
    y = list(rng.normal(0.0, 1.0, 50_000))
    grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    checks.append(fosd(x, y, grid))
    checks.append(not fosd(y, x, grid))
    # same dist -> neither dominates
    z = list(rng.normal(0.0, 2.0, 50_000))
    checks.append(not fosd(z, y, grid) and not fosd(y, z, grid))
    # mean-preserving spread: z has same mean, higher variance -> y dominates z in SSD? actually riskier; check ssd symmetric acceptance
    checks.append(ssd_check(x, y, grid))
    # degenerate: constant 5 dominates everything
    checks.append(fosd([5.0] * 100, x, grid))
    return float(sum(checks) / len(checks))


def bench_stochastic_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_order": _bench_stochastic_order(seed)}
