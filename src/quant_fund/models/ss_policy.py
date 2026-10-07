"""(s, S) continuous-review policy (Scarf 1960): reorder at s, order up
to S. Grid-searched (s,S) vs fixed-quantity baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._inv_synth import simulate


def bench_ss_policy(seed: int = 3025) -> dict[str, float]:
    best_cost, best_fill = 1e18, 0.0
    best_par = (0.0, 0.0)
    for s in np.arange(5, 30, 5):
        for S in np.arange(s + 20, s + 100, 20):
            c, f, _ = simulate(lambda ip, w, s=s, S=S: S - ip if ip <= s else 0.0, seed)
            if c < best_cost:
                best_cost, best_fill = c, f
                best_par = (float(s), float(S))
    c_n, f_n, _ = simulate(lambda ip, w: 80.0 if ip <= 20 else 0.0, seed)
    return {
        "synthetic_ss_cost": float(best_cost),
        "synthetic_naive_cost": float(c_n),
        "synthetic_ss_gain": float(c_n - best_cost),
        "synthetic_ss_fill": float(best_fill),
        "synthetic_ss_s": best_par[0],
        "synthetic_ss_S": best_par[1],
        "synthetic_torch_available": 0.0,
    }
