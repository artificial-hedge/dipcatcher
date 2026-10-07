"""Affine-invariant ensemble sampler (Goodman & Weare emcee stretch
move): K walkers each proposing along the line to a randomly paired
partner — no tuning, robust to correlation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._mc2_synth import errors, ess, logp


def bench_emcee_stretch(seed: int = 2961, k: int = 10, steps: int = 400) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((k, 2))
    traj: list[np.ndarray] = []
    for _ in range(steps):
        for i in range(k):
            j = int(rng.integers(k - 1))
            if j >= i:
                j += 1
            z = ((2.0 - 1.0) * rng.uniform() + 1.0) ** 2 / 2.0
            y = X[j] + z * (X[i] - X[j])
            lp = logp(y)
            if np.log(rng.uniform()) < np.log(z) + lp - logp(X[i]):
                X[i] = y
        traj.append(X.copy())
    samp = np.asarray(traj[len(traj) // 2 :]).reshape(-1, 2)
    me, ce = errors(samp)
    e = ess(samp[:, 0])
    # single-chain MH comparison at same budget
    from quant_fund.models._mc2_synth import indep_mh

    mh = indep_mh(k * steps, rng)
    me_b, ce_b = errors(mh)
    return {
        "synthetic_emcee_mean_err": me,
        "synthetic_emcee_cov_err": ce,
        "synthetic_emcee_ess_frac": float(e / len(samp)),
        "synthetic_mh_mean_err": me_b,
        "synthetic_mh_cov_err": ce_b,
        "synthetic_torch_available": 0.0,
    }
