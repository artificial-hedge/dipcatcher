"""Renewal-reward theorem: long-run reward rate = E[R]/E[T] (SYNTHETIC).

For renewal inter-arrival times T_i ~ Exp(lam) and cycle rewards
R_i ~ Exp(mean r), the long-run rate converges to r*lam. Bench: cumulative
reward rate from a simulated renewal process vs theory at several horizons
(shows O(1/sqrt(t)) convergence) plus a regenerative-cycle estimator.
"""

import numpy as np

from quant_fund.models._queue_synth import RR_LAM, RR_MEAN


def _sim(seed: int, n_cycles: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.exponential(1.0 / RR_LAM, n_cycles)
    r = rng.exponential(RR_MEAN, n_cycles)
    ctime = np.cumsum(t)
    crew = np.cumsum(r)
    return ctime, crew


def bench_renewal_reward(seed: int = 4409) -> dict[str, float]:
    theory = RR_MEAN * RR_LAM
    ct, cr = _sim(seed, 4000)
    rate_early = cr[99] / ct[99]
    rate_late = cr[-1] / ct[-1]
    # regenerative estimator: mean(R)/mean(T)
    regen = (cr[-1] / len(cr)) / (ct[-1] / len(cr))
    return {
        "synthetic_rr_theory": theory,
        "synthetic_rr_rate_early": rate_early,
        "synthetic_rr_rate_late": rate_late,
        "synthetic_rr_regen": regen,
        "synthetic_rr_err_early": abs(rate_early - theory),
        "synthetic_rr_err_late": abs(rate_late - theory),
    }
