"""Wagner–Whitin (1958) dynamic lot-sizing DP for deterministic (SYNTHETIC)
time-varying demand — optimal plan cost vs naive policies.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._inv_synth import H_COST, K_ORDER


def _ww_cost(d: np.ndarray) -> float:
    T = len(d)
    f = np.full(T + 1, np.inf)
    f[T] = 0.0
    for t in range(T - 1, -1, -1):
        hold = 0.0
        for j in range(t, T):
            f[t] = min(f[t], K_ORDER + hold + f[j + 1])
            if j + 1 < T:
                hold += H_COST * (j + 1 - t) * d[j + 1]
    return float(f[0])


def bench_wagner_whitin(seed: int = 3029) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    T = 12
    d = rng.poisson(20, T).astype(float)
    c_opt = _ww_cost(d)
    c_lfl = float(K_ORDER * T)
    hold = sum(H_COST * float(d[2 * i + 1]) for i in range(T // 2) if 2 * i + 1 < T)
    c_alt = float(K_ORDER * (T // 2) + hold)
    return {
        "synthetic_ww_cost": c_opt,
        "synthetic_lfl_cost": c_lfl,
        "synthetic_alt_cost": c_alt,
        "synthetic_ww_gain_lfl": float(c_lfl - c_opt),
        "synthetic_ww_gain_alt": float(c_alt - c_opt),
        "synthetic_torch_available": 0.0,
    }
