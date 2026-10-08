"""DRAM (Haario et al. 2006): delayed-rejection adaptive Metropolis — (SYNTHETIC)
rejected proposals get a second-stage smaller proposal; covariance
adapts online. Better ESS than plain RWM at same budget.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._mc2_synth import errors, ess, logp


def bench_dram(seed: int = 2971, n: int = 1500, warmup: int = 500) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = np.zeros(2)
    S = np.eye(2) * 0.5
    samples = []
    accept2 = 0
    for i in range(n):
        L = np.linalg.cholesky(S)
        y = x + L @ rng.standard_normal(2)
        if np.log(rng.uniform()) < logp(y) - logp(x):
            x = y
        else:
            # delayed rejection: smaller second proposal
            y2 = x + 0.3 * L @ rng.standard_normal(2)
            if np.log(rng.uniform()) < logp(y2) - logp(x):
                x = y2
                accept2 += 1
        samples.append(x.copy())
        if i > warmup:
            d = x - np.asarray(samples).mean(0)
            S = 0.9 * S + 0.1 * (np.outer(d, d) + 1e-3 * np.eye(2))
            S = (S + S.T) / 2 + 1e-6 * np.eye(2)
    samp = np.asarray(samples[warmup:])
    me, ce = errors(samp)
    e = ess(samp[:, 0]) / len(samp)
    return {
        "synthetic_dram_mean_err": me,
        "synthetic_dram_cov_err": ce,
        "synthetic_dram_ess_frac": float(e),
        "synthetic_dram_second_accept": float(accept2 / n),
        "synthetic_torch_available": 0.0,
    }
