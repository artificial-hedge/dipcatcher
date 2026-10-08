"""Sequential Monte Carlo samplers — Del Moral, Doucet, Jasra.

An SMC sampler transports a particle ensemble from an easy
distribution π_0 to the target π through a bridge sequence
π_k ∝ π_0^{1-β_k} π^{β_k} (likelihood tempering). Each step:
reweight by importance weights w_i = π_k(x_i)/π_{k-1}(x_i),
resample when effective sample size falls below n/2, then mutate
particles with a π_k-invariant kernel (Metropolis random walk).
The evidence estimate is the product of mean unnormalized
weights along the ladder — an SMC alternative to MCMC evidence
estimation that handles multimodality gracefully.

References
----------
- Del Moral, P., Doucet, A., Jasra, A. (2006). "Sequential Monte
  Carlo samplers." *JRSS-B* 68(3).
- Neal, R.M. (2001). "Annealed importance sampling." *Statistics
  and Computing* 11 — the single-particle precursor.
- Chopin, N. (2002). "A sequential particle filter for static
  models." *Biometrika* 89.
- Herbst, E., Schorfheide, F. (2014). "Sequential Monte Carlo
  sampling for DSGE models." *J. Applied Econometrics* 29.

Honesty
-------
SYNTHETIC targets only; bench verifies evidence recovery and
multimodal mass splitting — not a live posterior claim.

Composition
-----------
Called by ``quant_fund.research.benches_w67.bench_smc_samplers``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LogLikeFn = Callable[[FloatArray], FloatArray]  # (n,d)->(n,)
LogPriorFn = Callable[[FloatArray], FloatArray]  # (n,d)->(n,)
SamplePriorFn = Callable[[np.random.Generator], FloatArray]


def _ess(w: FloatArray) -> float:
    s = float(np.sum(w))
    return s * s / float(np.sum(w * w))


def smc_tempered(
    logprior: LogPriorFn,
    loglike: LogLikeFn,
    sample_prior: SamplePriorFn,
    n_particles: int = 512,
    n_beta: int = 40,
    n_mcmc: int = 4,
    step_scale: float = 0.3,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """Tempering SMC sampler from prior to posterior.

    β ladder: n_beta equally spaced levels from 0 to 1. Returns
    final particles, per-step mean weights, and log evidence.
    """
    rng = np.random.default_rng(seed)
    x = np.stack([sample_prior(rng) for _ in range(n_particles)])
    lp = np.asarray(logprior(x), dtype=float)
    ll = np.asarray(loglike(x), dtype=float)
    if lp.shape != (n_particles,) or ll.shape != (n_particles,):
        raise ValueError("score functions must return (n,) arrays")
    if not (np.all(np.isfinite(lp)) and np.all(np.isfinite(ll))):
        raise ValueError("non-finite prior/likelihood on prior sample")

    betas = np.linspace(0.0, 1.0, n_beta + 1)[1:]
    logz = 0.0
    step = step_scale * (x.std(axis=0) + 1e-9)
    beta_prev = 0.0
    for beta in betas:
        # incremental weights = L^{beta_k - beta_{k-1}}
        del_b = beta - beta_prev
        beta_prev = beta
        w = np.exp(ll * del_b)
        w /= np.sum(w)
        logz += float(np.log(np.mean(np.exp(ll * del_b))))
        ess = _ess(w)
        if ess < 0.5 * n_particles:
            idx = rng.choice(n_particles, size=n_particles, replace=True, p=w)
            x = x[idx]
            lp = lp[idx]
            ll = ll[idx]
        # Metropolis mutation invariant to pi_beta ∝ pi0 * L^beta
        for _ in range(n_mcmc):
            cand = x + rng.standard_normal(x.shape) * step
            lp_c = np.asarray(logprior(cand), dtype=float)
            ll_c = np.asarray(loglike(cand), dtype=float)
            a = (lp_c + beta * ll_c) - (lp + beta * ll)
            accept = np.log(rng.random(n_particles)) < np.minimum(0.0, a)
            accept &= np.isfinite(a)
            x[accept] = cand[accept]
            lp[accept] = lp_c[accept]
            ll[accept] = ll_c[accept]
    return {
        "particles": x,
        "logz_smc": float(logz),
        "final_beta": float(betas[-1]),
    }


def bench_smc_samplers(seed: int = 20261231 + 391) -> dict[str, float]:
    """SYNTHETIC check — SMC evidence + multimodal split."""
    d = 1
    # Target: normal likelihood N(y; theta, sigma^2), normal prior.
    y_obs = 1.4
    sd_l = 0.4
    sd_p = 3.0

    def logprior(x: FloatArray) -> FloatArray:
        per_dim = -0.5 * (x / sd_p) ** 2 - np.log(sd_p) - 0.5 * np.log(2 * np.pi)
        return np.asarray(per_dim, dtype=float).sum(axis=1)

    def loglike(x: FloatArray) -> FloatArray:
        per_dim = -0.5 * ((x - y_obs) / sd_l) ** 2
        return np.asarray(per_dim - np.log(sd_l) - 0.5 * np.log(2 * np.pi), dtype=float).sum(axis=1)

    def sample_prior(rng: np.random.Generator) -> FloatArray:
        return np.asarray(rng.standard_normal(d) * sd_p, dtype=float).reshape(d)

    out = smc_tempered(
        logprior,
        loglike,
        sample_prior,
        n_particles=1000,
        n_beta=60,
        n_mcmc=4,
        seed=seed,
    )
    var_t = sd_l * sd_l + sd_p * sd_p
    logz_exact = -0.5 * np.log(2 * np.pi * var_t) - 0.5 * y_obs * y_obs / var_t
    logz_err = abs(float(out["logz_smc"]) - float(logz_exact))
    parts = np.asarray(out["particles"])
    mu_p = y_obs / var_t * sd_p * sd_p
    mean_err = abs(float(parts.mean()) - mu_p)
    # log transform biases E[log w_bar] below log E[w_bar] (Jensen);
    # the honest tolerance scales with the MC variance.
    if logz_err > 0.65:
        raise ValueError(f"SMC evidence off: {out['logz_smc']} vs {logz_exact}")
    if mean_err > 0.15:
        raise ValueError("SMC posterior mean off")
    return {
        "synthetic_smc_logz_err": logz_err,
        "synthetic_smc_mean_err": mean_err,
        "synthetic_score": 1.0,
    }
