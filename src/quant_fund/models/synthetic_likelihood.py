"""Bayesian synthetic likelihood — Wood (2010) / Price et al. (2018).

For simulators with intractable likelihoods, BSL fits a multivariate
normal to the summary statistics of n_sim synthetic datasets
simulated at θ:

    logp_BSL(s_obs | θ) = log N(s_obs; m̂(θ), Σ̂(θ))

and plugs that surrogate likelihood into MCMC (random-walk
Metropolis-Hastings). Unlike ABC, no tolerance kernel is needed and
the estimator is asymptotically exact whenever the summary
distribution is truly Gaussian; the shrinkage-robust version
(Price et al.) regularizes Σ̂.

References
----------
- Wood, S.N. (2010). "Statistical inference for noisy nonlinear
  ecological dynamic systems." *Nature* 466.
- Price, L.F., Drovandi, C.C., Lee, A., Nott, D.J. (2018).
  "Bayesian synthetic likelihood." *JCGS* 27(1).
- Frazier, D.T., Maneesoonthorn, O., Martin, G.M., McCabe, B.P.M.
  (2019). "Approximate Bayesian forecasting." *IJF* 35(2).
- Sisson, S.A., Fan, Y., Beaumont, M.A., eds. (2018). *Handbook of
  Approximate Bayesian Computation* — BSL vs ABC comparison.

Honesty
-------
SYNTHETIC targets only; bench verifies BSL recovers the exact
Gaussian posterior when the simulator really is Gaussian — not a
live inference claim.

Composition
-----------
Called by ``quant_fund.research.benches_w67.bench_synth_lik``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

FloatArray = NDArray[np.float64]
SimulatorFn = Callable[[FloatArray, np.random.Generator], FloatArray]
SummaryFn = Callable[[FloatArray], FloatArray]


def synthetic_loglik(
    theta: FloatArray,
    s_obs: FloatArray,
    simulate: SimulatorFn,
    summarize: SummaryFn,
    n_sim: int = 50,
    rng: np.random.Generator | None = None,
    shrink: float = 0.0,
) -> float:
    """Gaussian synthetic log-likelihood at ``theta``.

    Simulates n_sim datasets, summarizes each, fits N(m,S) and
    evaluates log N(s_obs; m,S). ``shrink`` blends S toward its
    diagonal for stability in high dimensions.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    theta = np.asarray(theta, dtype=float).reshape(-1)
    sims = np.stack([summarize(simulate(theta, rng)) for _ in range(n_sim)])
    m = sims.mean(axis=0)
    s = np.cov(sims.T)
    if s.ndim == 0:
        s = s.reshape(1, 1)
    s = np.asarray(s, dtype=float)
    if shrink > 0:
        diag = np.diag(np.diag(s))
        s = (1.0 - shrink) * s + shrink * diag
    # tiny ridge for numerical PSD
    s = s + 1e-8 * np.eye(s.shape[0])
    try:
        return float(multivariate_normal.logpdf(s_obs, m, s))
    except (np.linalg.LinAlgError, ValueError):
        return -np.inf


def bsl_mcmc(
    theta0: FloatArray,
    s_obs: FloatArray,
    simulate: SimulatorFn,
    summarize: SummaryFn,
    n_iter: int = 2000,
    n_sim: int = 40,
    step: float = 0.15,
    burn: int = 500,
    seed: int = 0,
) -> FloatArray:
    """Random-walk MH chain on the synthetic likelihood surface."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta0, dtype=float).reshape(-1)
    chain = np.zeros((n_iter, theta.size))
    cur = synthetic_loglik(theta, s_obs, simulate, summarize, n_sim, rng)
    for i in range(n_iter):
        cand = theta + rng.standard_normal(theta.size) * step
        lp = synthetic_loglik(cand, s_obs, simulate, summarize, n_sim, rng)
        if np.log(rng.random()) < lp - cur:
            theta, cur = cand, lp
        chain[i] = theta
    return chain[burn:]


def bench_synthetic_likelihood(seed: int = 20261231 + 392) -> dict[str, float]:
    """SYNTHETIC check — BSL recovers the exact Gaussian posterior."""
    rng = np.random.default_rng(seed)
    theta_true = 1.7
    sd_obs = 0.5
    n_data = 25
    data = theta_true + sd_obs * rng.standard_normal(n_data)

    def simulate(theta: FloatArray, r: np.random.Generator) -> FloatArray:
        return np.asarray(theta[0] + sd_obs * r.standard_normal(n_data), dtype=np.float64)

    def summarize(dset: FloatArray) -> FloatArray:
        return np.array([dset.mean()])

    s_obs = summarize(data)
    # Exact posterior for theta | mean(y): N(ybar, sd^2/n) with
    # flat prior.
    exact_mean = float(s_obs[0])
    exact_sd = float(sd_obs / np.sqrt(n_data))
    chain = bsl_mcmc(
        np.array([0.0]),
        s_obs,
        simulate,
        summarize,
        n_iter=3000,
        n_sim=60,
        step=0.12,
        burn=800,
        seed=seed + 1,
    )
    post_mean = float(chain.mean())
    post_sd = float(chain.std())
    mean_err = abs(post_mean - exact_mean)
    sd_err = abs(post_sd - exact_sd) / exact_sd
    if mean_err > 0.1 or sd_err > 0.5:
        raise ValueError("BSL posterior recovery failed")
    # BSL loglik at truth > at clearly-wrong theta.
    ll_true = synthetic_loglik(np.array([theta_true]), s_obs, simulate, summarize, 100, rng)
    ll_bad = synthetic_loglik(np.array([theta_true + 1.5]), s_obs, simulate, summarize, 100, rng)
    if not (ll_true > ll_bad):
        raise ValueError("BSL ordering failed")
    return {
        "synthetic_bsl_mean_err": mean_err,
        "synthetic_bsl_sd_err": sd_err,
        "synthetic_bsl_llr": float(ll_true - ll_bad),
        "score": 1.0,
    }
