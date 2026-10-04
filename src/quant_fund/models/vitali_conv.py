"""Vitali convergence theorem bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def conv_in_prob(vals_xn: list[list[float]], vals_x: list[float], eps: float) -> bool:
    """Convergence in probability on a finite probability space:
    every term |X_n - X| < eps eventually? For our finite model we ask
    that the worst index error drops below eps for all late n."""
    n = min(len(vals_xn), len(vals_x))
    late = range(n // 2, n)
    return all(abs(vals_xn[i][0] - vals_x[i]) < eps for i in late)


def l1_conv(vals_xn: list[list[float]], vals_x: list[float]) -> float:
    """Mean absolute deviation between X_n samples and X (model of
    L1 distance with equal weights)."""
    n = min(len(vals_xn), len(vals_x))
    if n == 0:
        return 0.0
    return sum(abs(vals_xn[i][0] - vals_x[i]) for i in range(n)) / n


def _bench_vitali_conv(seed: int = 0) -> float:
    checks = []
    # X_n -> X in L1: errors shrinking
    xs = [0.0] * 10
    xn = [[1.0 / (i + 1)] for i in range(10)]
    checks.append(l1_conv(xn, xs) < 0.4)
    # convergent in probability model check
    checks.append(conv_in_prob(xn, xs, 0.51))
    checks.append(not conv_in_prob(xn, xs, 0.1))
    # constant sequences converge trivially
    xn0 = [[0.0] for _ in range(10)]
    checks.append(l1_conv(xn0, xs) == 0.0)
    # diverging sequence fails prob convergence at small eps
    xb = [[float(i)] for i in range(10)]
    checks.append(not conv_in_prob(xb, xs, 1.0))
    return float(sum(checks) / len(checks))


def bench_vitali_conv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vitali_conv": _bench_vitali_conv(seed)}
