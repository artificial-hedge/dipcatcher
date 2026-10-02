"""Nested sampling — Skilling (2004/2006).

Nested sampling estimates the Bayesian evidence (marginal
likelihood) Z = ∫ L(θ) π(θ) dθ by shrinking a population of
``n_live`` points sampled from the prior: at each iteration the
lowest-likelihood point is removed (becomes a dead point), the
prior mass shrinks by X_i ~ X_{i-1}·Beta(n_live, 1), and a
replacement point is drawn from the prior constrained to
L > L_worst. Accumulating w_i L_i with w_i = X_{i-1} − X_i gives
Z; the same dead points with weights w_i L_i / Z are a posterior
sample. Information H = Σ (w_i L_i/Z) log(L_i/Z·...) sets the
remaining-mass stopping rule.

For the constrained draws we use random-walk MCMC inside the
current prior region — correct for arbitrary priors without
needing contour-fitting machinery.

References
----------
- Skilling, J. (2004). "Nested sampling." *AIP Conf. Proc.* 735.
- Skilling, J. (2006). "Nested sampling for general Bayesian
  computation." *Bayesian Analysis* 1(4).
- Feroz, F., Hobson, M.P., Bridges, M. (2009). "MultiNest: an
  efficient and robust Bayesian inference tool." *MNRAS* 398.
- Handley, W.J., Hobson, M.P., Lasenby, A.N. (2015). "POLYCHORD."
  *MNRAS* 450.

Honesty
-------
SYNTHETIC targets only; bench recovers the analytic Gaussian
evidence/posterior — not a live marginal-likelihood claim.

Composition
-----------
Called by ``quant_fund.research.benches_w67.bench_nested_sampling``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LogLikeFn = Callable[[FloatArray], float]
LogPriorFn = Callable[[FloatArray], float]
SamplePriorFn = Callable[[np.random.Generator], FloatArray]


def _constrained_draw(
    loglike: LogLikeFn,
    logprior: LogPriorFn,
    l_min: float,
    sample_prior: SamplePriorFn,
    rng: np.random.Generator,
    live: FloatArray,
    n_mcmc: int = 25,
) -> FloatArray:
    """Draw one prior sample constrained to L > l_min via short
    random-walk MH on the TRUNCATED PRIOR (target ∝ pi · 1{L>l_min}),
    seeded from a random live point, with a rejection fallback."""
    idx = int(rng.integers(0, live.shape[0]))
    x = live[idx].copy()
    lp_x = logprior(x)
    step = 0.5 * (live.std(axis=0) + 1e-9)
    for _ in range(n_mcmc * 4):
        cand = x + rng.standard_normal(x.size) * step
        if loglike(cand) > l_min:
            lp_c = logprior(cand)
            if np.log(rng.random()) < min(0.0, lp_c - lp_x):
                x, lp_x = cand, lp_c
    if loglike(x) > l_min:
        return np.asarray(x, dtype=np.float64)
    for _ in range(2000):
        cand = sample_prior(rng)
        if loglike(cand) > l_min:
            return cand
    raise ValueError("constrained draw failed")


def nested_sampling(
    loglike: LogLikeFn,
    logprior: LogPriorFn,
    sample_prior: SamplePriorFn,
    n_live: int = 100,
    n_max: int | None = None,
    tol_frac: float = 0.01,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """Run nested sampling; returns evidence, posterior draws, H."""
    rng = np.random.default_rng(seed)
    live = np.stack([sample_prior(rng) for _ in range(n_live)])
    live_ll = np.array([loglike(x) for x in live])
    if not np.all(np.isfinite(live_ll)):
        raise ValueError("non-finite likelihood on prior sample")
    if n_max is None:
        n_max = 50 * n_live

    dead_pts: list[FloatArray] = []
    dead_ll: list[float] = []
    dead_w: list[float] = []
    logz = -np.inf
    logx = 0.0
    shrink = np.exp(-1.0 / n_live)
    for i in range(n_max):
        w = float(np.exp(logx) * (1.0 - shrink))  # X_{i-1} - X_i approx
        logx += np.log(shrink)
        worst = int(np.argmin(live_ll))
        dead_pts.append(live[worst].copy())
        dead_ll.append(float(live_ll[worst]))
        dead_w.append(w)
        # running logZ via logaddexp
        logw_l = np.log(w) + live_ll[worst]
        logz = float(np.logaddexp(logz, logw_l))
        live_ll[worst] = np.inf  # exclude dead slot from next minimum
        l_min = float(np.min(live_ll))
        live[worst] = _constrained_draw(loglike, logprior, l_min, sample_prior, rng, live)
        live_ll[worst] = loglike(live[worst])
        # stopping: remaining evidence < tol_frac of current Z
        logz_tail = np.log(np.exp(logx)) + float(np.max(live_ll))
        if np.isfinite(logz) and np.exp(logz_tail - logz) < tol_frac and i > 2 * n_live:
            break

    dead = np.stack(dead_pts)
    if not np.all(np.isfinite(dead_ll)):
        raise ValueError("non-finite dead likelihood")
    log_wl = np.log(np.asarray(dead_w)) + np.asarray(dead_ll)
    logz = float(np.logaddexp.reduce(log_wl))
    # posterior weights
    pw = np.exp(log_wl - logz)
    # information
    log_l = np.asarray(dead_ll)
    h = float(np.sum(pw * (log_l - logz)))
    return {
        "logz": logz,
        "h": h,
        "dead_points": dead,
        "dead_loglike": log_l,
        "posterior_weights": pw,
        "n_iter": float(len(dead)),
    }


def bench_nested_sampling(seed: int = 20261231 + 390) -> dict[str, float]:
    """SYNTHETIC check — analytic Gaussian evidence recovered."""
    d = 2
    mu_l = np.array([2.0, -1.0])
    sd_l = np.array([0.4, 0.3])
    sd_p = np.array([3.0, 3.0])

    def loglike(x: FloatArray) -> float:
        r = (x - mu_l) / sd_l
        return float(-0.5 * np.sum(r * r) - np.sum(np.log(sd_l)) - 0.5 * d * np.log(2 * np.pi))

    def sample_prior(rng: np.random.Generator) -> FloatArray:
        return np.asarray(rng.standard_normal(d) * sd_p, dtype=np.float64)

    def logprior(x: FloatArray) -> float:
        r = x / sd_p
        return float(-0.5 * np.sum(r * r) - np.sum(np.log(sd_p)) - 0.5 * d * np.log(2 * np.pi))

    out = nested_sampling(loglike, logprior, sample_prior, n_live=150, n_max=4000, seed=seed)
    logz = float(out["logz"])
    # Analytic Z for normal likelihood x normal prior:
    # Z = N(0; mu_l, sd_l^2 + sd_p^2) elementwise product.
    var_t = sd_l * sd_l + sd_p * sd_p
    logz_exact = float(
        -0.5 * d * np.log(2 * np.pi)
        - 0.5 * np.sum(np.log(var_t))
        - 0.5 * np.sum(mu_l * mu_l / var_t)
    )
    err = abs(logz - logz_exact)
    if err > 0.4:
        raise ValueError(f"nested evidence off: {logz} vs {logz_exact}")
    # Posterior moments: N(mu_p, sd_p2) precision-averaged.
    prec = 1.0 / (sd_l * sd_l) + 1.0 / (sd_p * sd_p)
    mu_p = (mu_l / (sd_l * sd_l)) / prec
    sd_p2 = np.sqrt(1.0 / prec)
    dead = np.asarray(out["dead_points"])
    pw = np.asarray(out["posterior_weights"])
    mean_hat = (dead * pw[:, None]).sum(axis=0)
    var_hat = ((dead - mean_hat) ** 2 * pw[:, None]).sum(axis=0)
    mean_err = float(np.linalg.norm(mean_hat - mu_p))
    sd_err = float(np.linalg.norm(np.sqrt(var_hat) - sd_p2))
    if mean_err > 0.15 or sd_err > 0.25:
        raise ValueError("posterior moments off")
    h = float(out["h"])
    if not (0.0 < h < 10.0):
        raise ValueError("information H implausible")
    return {
        "synthetic_ns_logz_err": err,
        "synthetic_ns_mean_err": mean_err,
        "synthetic_ns_sd_err": sd_err,
        "synthetic_ns_h": h,
        "score": 1.0,
    }
