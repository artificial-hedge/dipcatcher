"""M/G/1 queue with server vacations (Fuhrmann-Cooper decomposition).

With exhaustive service and vacations V, mean waiting time decomposes as
E[W] = E[W_MG1] + E[V_e] where E[W_MG1] is the Pollaczek-Khinchine value
and E[V_e] = E[V^2]/(2E[V]) is the residual vacation. Bench: simulated
M/M/1 + Exp(vac) mean number in system vs Little's-law-implied theory, and
the decomposition uplift over plain PK.
"""

import numpy as np

from quant_fund.models._queue_synth import VQ_LAM, VQ_MU, VQ_VAC


def _pk_wait(lam: float, mu: float) -> float:
    rho = lam / mu
    return float(rho / (mu - lam))  # M/M/1: E[W] = rho/(mu-lam)


def _sim(seed: int, horizon: float = 60000.0) -> float:
    """Event-driven M/M/1 with exhaustive exponential vacations.

    Multiple vacations: when the queue empties the server takes repeated
    Exp(VQ_VAC) vacations until customers are waiting on its return —
    the regime where the Fuhrmann-Cooper decomposition holds exactly.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    n = 0
    next_arr = rng.exponential(1.0 / VQ_LAM)
    next_dep = np.inf
    vac_end = np.inf
    area, t_last = 0.0, 0.0
    while t < horizon:
        nxt = min(next_arr, next_dep, vac_end)
        area += n * (nxt - t_last)
        t, t_last = nxt, nxt
        if nxt == next_arr:
            n += 1
            next_arr = t + rng.exponential(1.0 / VQ_LAM)
            if n == 1 and not np.isfinite(vac_end) and not np.isfinite(next_dep):
                next_dep = t + rng.exponential(1.0 / VQ_MU)
        elif nxt == next_dep:
            n -= 1
            if n > 0:
                next_dep = t + rng.exponential(1.0 / VQ_MU)
            else:
                next_dep = np.inf
                vac_end = t + rng.exponential(1.0 / VQ_VAC)
        else:  # vacation ends
            if n > 0:
                vac_end = np.inf
                next_dep = t + rng.exponential(1.0 / VQ_MU)
            else:
                vac_end = t + rng.exponential(1.0 / VQ_VAC)  # take another
    return area / t


def bench_vacation_queue(seed: int = 4411) -> dict[str, float]:
    rho = VQ_LAM / VQ_MU
    w_pk = _pk_wait(VQ_LAM, VQ_MU)
    res_vac = 1.0 / VQ_VAC  # E[V^2]/(2E[V]) for Exp(VQ_VAC)
    w_theory = w_pk + res_vac
    n_theory = VQ_LAM * w_theory + rho  # Little's law on the system
    n_sim = _sim(seed)
    return {
        "synthetic_vq_pk_wait": w_pk,
        "synthetic_vq_res_vac": res_vac,
        "synthetic_vq_n_theory": n_theory,
        "synthetic_vq_n_sim": n_sim,
        "synthetic_vq_n_err": abs(n_sim - n_theory),
        "synthetic_vq_uplift_vs_pk": w_theory - w_pk,
    }
