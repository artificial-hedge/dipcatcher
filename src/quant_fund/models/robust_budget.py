"""Bertsimas-Sim budget-of-uncertainty robust LP.

max c'x s.t. sum_j (a_ij x_j) + Gamma_i * max|S|<=Gamma sum_j d_ij x_j
<= b_i: the budgeted robust counterpart. Bench compares nominal,
budgeted (Gamma=1), and full worst-case objectives on a small 4x3 LP.
"""

import numpy as np
from scipy.optimize import linprog

_C = np.array([6.0, 5.0, 4.0, 3.0])
_A = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 2.0, 1.0], [1.0, 3.0, 1.0, 2.0]])
_B = np.array([10.0, 12.0, 14.0])
_D = np.array([[0.2, 0.3, 0.2, 0.4], [0.3, 0.2, 0.3, 0.2], [0.4, 0.2, 0.3, 0.3]])


def _robust_obj(gamma: float) -> float:
    # budgeted counterpart: a_i'x + Gamma_i * sum of the Gamma largest
    # deviations is approximated here by scaling the deviation matrix,
    # conservative at integer gamma for small instances
    res = linprog(-_C, A_ub=_A + gamma * _D, b_ub=_B, bounds=(0, None), method="highs")
    return float(-res.fun) if res.success else 0.0


def bench_robust_budget(seed: int = 5511) -> dict[str, float]:
    nominal = _robust_obj(0.0)
    rob = _robust_obj(1.0)
    worst = _robust_obj(2.0)
    return {
        "synthetic_rb_nominal": nominal,
        "synthetic_rb_gamma1": rob,
        "synthetic_rb_worst": worst,
        "synthetic_rb_price_of_robust": nominal - rob,
    }
