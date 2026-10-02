"""Ski rental — the canonical online-competitive problem.

Deterministic: rent until cumulative rent reaches buy price, then buy
(2-competitive). Randomized: buy at day threshold drawn from the
e/(e-1)-competitive distribution. Bench sweeps horizons and reports
the worst competitive ratio vs the offline optimum.
"""

import numpy as np


def _det_cost(T: int, buy: float) -> float:
    # rent each day while day < buy; buy at day B if still skiing
    if buy > T:
        return T * 1.0
    return (buy - 1.0) * 1.0 + buy


def _rand_cost(T: int, buy: float, rng: np.random.Generator) -> float:
    # buy at day ~ density f(t) = (e^{t/b})/(b(e-1)), t in [0, b)
    u = rng.random()
    t = buy * np.log(1.0 + u * (np.e - 1.0))
    return T if t > T else t + buy


def bench_ski_rental(seed: int = 5401) -> dict[str, float]:
    buy = 10.0
    rng = np.random.default_rng(seed)
    worst_det = 0.0
    worst_rand = 0.0
    for t in range(1, 40):
        opt = min(t * 1.0, buy)
        worst_det = max(worst_det, _det_cost(t, buy) / opt)
        rand = np.mean([_rand_cost(t, buy, rng) for _ in range(1500)])
        worst_rand = max(worst_rand, rand / opt)
    return {
        "synthetic_ski_det_ratio": worst_det,
        "synthetic_ski_rand_ratio": worst_rand,
        "synthetic_ski_det_bound": 2.0 - 1.0 / buy,
        "synthetic_ski_rand_bound": np.e / (np.e - 1.0),
    }
