"""Kolmogorov consistency of finite-dimensional measures (SYNTHETIC)."""

from __future__ import annotations


def marginal(
    dist: dict[tuple[int, ...], float], keep: tuple[int, ...]
) -> dict[tuple[int, ...], float]:
    """Marginalize a joint distribution over coordinates `keep`."""
    out: dict[tuple[int, ...], float] = {}
    for x, p in dist.items():
        k = tuple(x[i] for i in keep)
        out[k] = out.get(k, 0.0) + p
    return out


def consistent(dist2: dict[tuple[int, ...], float], dist1: dict[tuple[int, ...], float]) -> bool:
    """Kolmogorov consistency: the 1-d marginal of the 2-d law equals
    the given 1-d law at each coordinate."""
    m0 = marginal(dist2, (0,))
    m1 = marginal(dist2, (1,))
    return m0 == dist1 and m1 == dist1


def _bench_kolmogorov_01(seed: int = 0) -> float:
    checks = []
    # iid fair coin joint is consistent with the 1-d Bernoulli(1/2)
    d1: dict[tuple[int, ...], float] = {(0,): 0.5, (1,): 0.5}
    d2: dict[tuple[int, ...], float] = {
        (0, 0): 0.25,
        (0, 1): 0.25,
        (1, 0): 0.25,
        (1, 1): 0.25,
    }
    checks.append(consistent(d2, d1))
    # a biased 1-d law is inconsistent with the fair joint
    biased: dict[tuple[int, ...], float] = {(0,): 0.6, (1,): 0.4}
    checks.append(not consistent(d2, biased))
    # perfectly correlated pair is consistent with d1
    corr: dict[tuple[int, ...], float] = {(0, 0): 0.5, (1, 1): 0.5}
    checks.append(consistent(corr, d1))
    # marginal returns a proper distribution
    m = marginal(d2, (0,))
    checks.append(abs(sum(m.values()) - 1.0) < 1e-12)
    # 2-d marginal of itself is identity
    checks.append(marginal(d2, (0, 1)) == d2)
    return float(sum(checks) / len(checks))


def bench_kolmogorov_01(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kolmogorov_01": _bench_kolmogorov_01(seed)}
