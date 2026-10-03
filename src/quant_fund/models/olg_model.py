"""2-period OLG: savings → capital; law of motion consistency."""

import numpy as np

_SEED = 20261231 + 756


def olg_path(k0: float, alpha: float, s_share: float, steps: int) -> np.ndarray:
    """k_{t+1} = s·w(k) with w = (1-α)k^α (competitive wage)."""
    k = np.zeros(steps)
    k[0] = k0
    for t in range(1, steps):
        w = (1 - alpha) * k[t - 1] ** alpha
        k[t] = s_share * w
    return k


def bench_olg_model(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    path = olg_path(0.5, 0.33, 0.3, 200)
    # steady state solves k = s(1-α)k^α → k* = (s(1-α))^(1/(1-α))
    k_star = (0.3 * 0.67) ** (1.0 / 0.67)
    ok = float(abs(path[-1] - k_star) / k_star < 0.02)
    return {"synthetic_olg_steady": ok}
