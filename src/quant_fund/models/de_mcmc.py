"""DE-MCMC (ter Braak 2006): population MCMC where proposals are
scaled differences of other chains — self-tuning jump distribution.
Correlated-Gaussian recovery vs MH.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._mc2_synth import errors, logp


def bench_de_mcmc(seed: int = 2967, k: int = 8, steps: int = 500) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((k, 2)) * 0.5
    gamma = 2.38 / np.sqrt(2)
    traj: list[np.ndarray] = []
    for _ in range(steps):
        for i in range(k):
            j1, j2 = rng.choice([j for j in range(k) if j != i], 2, replace=False)
            y = X[i] + gamma * (X[j1] - X[j2]) + 0.01 * rng.standard_normal(2)
            if np.log(rng.uniform()) < logp(y) - logp(X[i]):
                X[i] = y
        traj.append(X.copy())
    samp = np.asarray(traj[len(traj) // 2 :]).reshape(-1, 2)
    me, ce = errors(samp)
    from quant_fund.models._mc2_synth import indep_mh

    mh = indep_mh(k * steps // 2, rng)
    me_b, ce_b = errors(mh)
    return {
        "synthetic_demc_mean_err": me,
        "synthetic_demc_cov_err": ce,
        "synthetic_mh_mean_err": me_b,
        "synthetic_mh_cov_err": ce_b,
        "torch_available": 0.0,
    }
