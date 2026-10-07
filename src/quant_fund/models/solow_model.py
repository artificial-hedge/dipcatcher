"""Solow growth: k' = s·k^α - (n+g+δ)k → exact steady state (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 755


def solow_steady(s: float, alpha: float, breakeven: float) -> float:
    """k* = (s/breakeven)^(1/(1-α))."""
    return float((s / breakeven) ** (1.0 / (1.0 - alpha)))


def solow_sim(k0: float, s: float, alpha: float, be: float, steps: int) -> np.ndarray:
    k = np.zeros(steps)
    k[0] = k0
    for t in range(1, steps):
        k[t] = k[t - 1] + s * k[t - 1] ** alpha - be * k[t - 1]
    return k


def bench_solow_model(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    s, alpha, be = 0.2, 0.33, 0.08
    k_star = solow_steady(s, alpha, be)
    path = solow_sim(0.5 * k_star, s, alpha, be, 500)
    conv = float(abs(path[-1] - k_star) / k_star < 0.01)
    return {"synthetic_solow_converges": conv}
