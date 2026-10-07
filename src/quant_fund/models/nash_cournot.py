"""N-firm Cournot-Nash equilibrium with asymmetric marginal costs (SYNTHETIC).

Firm i picks q_i to maximize (a - b*Q)*q_i - c_i*q_i. The equilibrium is the
fixed point q_i = (a - c_i - b*Q_{-i})/(2b); bench reports the best-response
residual plus the monopoly/collusive output for comparison.
"""

import numpy as np

from quant_fund.models._mfg_synth import COU_A, COU_B, COU_C


def _equilibrium(c: np.ndarray) -> np.ndarray:
    n = len(c)
    # analytic: q_i = (a - 2c_i + mean_c_adj)/(b(n+1)) with mean_c = avg(c)
    mean_c = float(np.mean(c))
    return np.maximum(0.0, (COU_A - (n + 1) * c + n * mean_c) / (COU_B * (n + 1)))


def _best_response_resid(q: np.ndarray, c: np.ndarray) -> float:
    resid = 0.0
    for i in range(len(q)):
        qm = q.sum() - q[i]
        br = max(0.0, (COU_A - c[i] - COU_B * qm) / (2 * COU_B))
        resid = max(resid, abs(br - q[i]))
    return resid


def bench_nash_cournot(seed: int = 4205) -> dict[str, float]:
    del seed
    q = _equilibrium(COU_C)
    resid = _best_response_resid(q, COU_C)
    qtot = float(q.sum())
    # monopoly output with average cost
    qmono = max(0.0, (COU_A - float(np.mean(COU_C))) / (2 * COU_B))
    price = COU_A - COU_B * qtot
    profit = (price - COU_C) @ q
    return {
        "synthetic_cournot_resid": resid,
        "synthetic_cournot_q": qtot,
        "synthetic_cournot_monopoly_q": float(qmono),
        "synthetic_cournot_price": float(price),
        "synthetic_cournot_profit": float(profit),
    }
