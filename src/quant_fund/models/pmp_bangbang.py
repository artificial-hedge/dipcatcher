"""PMP bang-bang minimum-time control: double integrator, |u|<=1,
minimize time to reach target. Optimal switching curve
v = -sign(x)·sqrt(2|x|). Time to reach vs PD baseline time.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._oc_synth import DT, dyn


def _reach_time(x0: np.ndarray, policy, target: float = 1.0, tol: float = 0.03) -> float:
    x = np.array(x0, dtype=np.float64)
    t = 0.0
    for _ in range(2000):
        if abs(x[0] - target) < tol and abs(x[1]) < 0.2:
            return t
        x = dyn(x, policy(x, target))
        t += DT
    return float("inf")


def bench_pmp_bangbang(seed: int = 2921) -> dict[str, float]:
    def bang(x, target):
        # PMP switching curve: u = -sign(err + v|v|/2)
        err = x[0] - target
        return float(-np.sign(err + 0.5 * x[1] * abs(x[1])))

    t_bb = _reach_time(np.array([0.0, 0.0]), bang)

    def pd(x, target):
        return np.clip(0.6 * (target - x[0]) - 1.2 * x[1], -1, 1)

    t_pd = _reach_time(np.array([0.0, 0.0]), pd)
    return {
        "synthetic_bangbang_time": float(t_bb),
        "synthetic_pd_time": float(t_pd),
        "synthetic_bangbang_speedup": float(t_pd - t_bb),
        "synthetic_torch_available": 0.0,
    }
