"""Base-stock (order-up-to) policy: optimal S = F^{-1}(p/(p+h)) for
lead-time demand vs conservative baseline.
"""

from __future__ import annotations

from scipy.stats import poisson

from quant_fund.models._inv_synth import D_RATE, H_COST, S_COST, simulate


def bench_base_stock(seed: int = 3033) -> dict[str, float]:
    lam_ltd = 2.0 * D_RATE
    crit = S_COST / (S_COST + H_COST)
    s_star = float(poisson.ppf(crit, lam_ltd))

    def bs(ip: float, w: int) -> float:
        return s_star - ip if ip < s_star else 0.0

    def cons(ip: float, w: int) -> float:
        s_c = s_star * 1.4
        return s_c - ip if ip < s_c else 0.0

    c_bs, f_bs, _ = simulate(bs, seed)
    c_c, f_c, _ = simulate(cons, seed)
    return {
        "synthetic_bs_sstar": s_star,
        "synthetic_bs_cost": float(c_bs),
        "synthetic_bs_fill": float(f_bs),
        "synthetic_cons_cost": float(c_c),
        "synthetic_cons_fill": float(f_c),
        "synthetic_bs_gain": float(c_c - c_bs),
        "torch_available": 0.0,
    }
