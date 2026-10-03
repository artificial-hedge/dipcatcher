"""CTMC availability of a repairable unit (2-state birth-death).

Steady-state availability A = mu/(lam + mu); mean time between failures
MTBF = 1/lam and mean downtime MDT = 1/mu. Bench: transient CTMC
occupancy at horizon T vs the stationary limit, plus a Monte-Carlo uptime
fraction as the simulation check.
"""

import numpy as np

from quant_fund.models._queue_synth import AVAIL_LAM, AVAIL_MU


def _transient(t: float) -> float:
    rate = AVAIL_LAM + AVAIL_MU
    return float(AVAIL_MU / rate + (AVAIL_LAM / rate) * np.exp(-rate * t))


def _mc_uptime(seed: int, horizon: float = 5000.0) -> float:
    rng = np.random.default_rng(seed)
    t, up = 0.0, 0.0
    while t < horizon:
        dur = rng.exponential(1.0 / AVAIL_MU)  # start up? draw dwell in up
        dn = rng.exponential(1.0 / AVAIL_LAM)
        up += min(dn, horizon - t)
        t += dn + dur
    return up / horizon


def bench_ctmc_availability(seed: int = 4407) -> dict[str, float]:
    a_inf = AVAIL_MU / (AVAIL_LAM + AVAIL_MU)
    a_t = _transient(50.0)
    a_mc = _mc_uptime(seed)
    return {
        "synthetic_avail_stat": a_inf,
        "synthetic_avail_transient": a_t,
        "synthetic_avail_mc": a_mc,
        "synthetic_avail_mc_err": abs(a_mc - a_inf),
        "synthetic_avail_trans_gap": abs(a_t - a_inf),
        "synthetic_avail_mtbf": 1.0 / AVAIL_LAM,
        "synthetic_avail_mdt": 1.0 / AVAIL_MU,
    }
