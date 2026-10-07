"""Clark–Scarf (1960) serial two-echelon inventory: echelon base-stock
levels via sequential newsvendor recursion vs myopic policies.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def _echelon_levels(d_rate: float, h1: float, h2: float, p: float) -> tuple[float, float]:
    lam1 = float(poisson.mean(d_rate * 2))
    lam2 = float(poisson.mean(d_rate * 3))
    c1 = p / (p + h1)
    c2 = (p + h1) / (p + h1 + h2)
    s1 = float(poisson.ppf(c1, lam1))
    s2 = float(poisson.ppf(c2, lam2))
    return s1, max(s2, s1)


def bench_clark_scarf(seed: int = 3037) -> dict[str, float]:
    d_rate, h1, h2, p = 20.0, 0.5, 0.2, 5.0
    s1, s2 = _echelon_levels(d_rate, h1, h2, p)
    rng = np.random.default_rng(seed)
    weeks = 156
    dem = rng.poisson(d_rate, weeks)

    def run(myp_s1: float, myp_s2: float) -> tuple[float, float]:
        inv1, inv2 = 0.0, 0.0
        p1: list[tuple[int, float]] = []
        tot = filled = 0.0
        for w, d in enumerate(dem):
            for aw, q in p1:
                if aw <= w:
                    inv1 += q
                    inv2 -= q
            p1 = [(aw, q) for aw, q in p1 if aw > w]
            sell = min(inv1, float(d))
            inv1 -= sell
            filled += sell
            tot += 5.0 * (float(d) - sell) + h1 * inv1 + h2 * max(inv2, 0.0)
            ip1 = inv1 + sum(q for _, q in p1)
            q1 = min(max(myp_s1 - ip1, 0.0), max(inv2, 0.0))
            if q1 > 0:
                p1.append((w + 1, q1))
            ip2 = inv2 - sum(q for _, q in p1)
            q2 = max(myp_s2 - ip2 - float(dem[max(0, w - 1)]), 0.0) if w else 0.0
            inv2 += q2 * 0.5
        return tot / weeks, filled / max(float(dem.sum()), 1.0)

    c_e, f_e = run(s1, s2)
    s_my = float(poisson.ppf(p / (p + h1), d_rate * 2))
    c_m, f_m = run(s_my, s_my)
    return {
        "synthetic_cs_cost": float(c_e),
        "synthetic_myopic_cost": float(c_m),
        "synthetic_cs_gain": float(c_m - c_e),
        "synthetic_cs_fill": float(f_e),
        "synthetic_myopic_fill": float(f_m),
        "synthetic_cs_s1": s1,
        "synthetic_cs_s2": s2,
        "synthetic_torch_available": 0.0,
    }
