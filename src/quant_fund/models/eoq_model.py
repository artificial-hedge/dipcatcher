"""EOQ (Harris 1913): Q* = sqrt(2·K·D/h). Simulated-cost advantage vs
naive periodic restock on Poisson demand.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._inv_synth import D_RATE, H_COST, K_ORDER, simulate


def bench_eoq_model(seed: int = 3017) -> dict[str, float]:
    q_eoq = float(np.sqrt(2 * K_ORDER * D_RATE / H_COST))

    def eoq_pol(ip: float, w: int) -> float:
        return q_eoq if ip <= 0.5 * D_RATE else 0.0

    def naive(ip: float, w: int) -> float:
        return 4 * D_RATE if ip <= D_RATE else 0.0

    c_eoq, f_eoq, _ = simulate(eoq_pol, seed)
    c_n, f_n, _ = simulate(naive, seed)
    tc_star = float(np.sqrt(2 * K_ORDER * D_RATE * H_COST))
    return {
        "synthetic_eoq_cost": float(c_eoq),
        "synthetic_naive_cost": float(c_n),
        "synthetic_eoq_gain": float(c_n - c_eoq),
        "synthetic_eoq_fill": float(f_eoq),
        "synthetic_naive_fill": float(f_n),
        "synthetic_eoq_analytic_cost": tc_star,
        "torch_available": 0.0,
    }
