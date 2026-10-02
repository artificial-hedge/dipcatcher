"""Marginal likelihood (model evidence) estimators.

Canonical references:

- Newton & Raftery (1994) 'Approximate Bayesian
  inference with the weighted likelihood bootstrap'
  JRSS B 56 — the harmonic-mean estimator
  m = 1/mean(1/L_i) over posterior draws.
- Gelfand & Dey (1994) 'Bayesian model choice:
  asymptotics and exact calculations' JRSS B 56 —
  reciprocal-importance estimator with a covering
  density h: 1/m = mean(h(theta_i)/L_i pi(theta_i)).
- Chib (1995) 'Marginal likelihood from the Gibbs
  output' JASA 90 — m(y) = f(y|theta*) pi(theta*) /
  pi(theta*|y) evaluated at a high-density point.
- Savage & Dickey (1971) 'The weighted likelihood
  ratio, sharp hypotheses about chances...' —
  BF_01 = p(theta1=theta1* | y) / p(theta1=theta1*)
  for nested models.
- Ogata (1989) / Friel & Pettitt (2008) power
  posteriors — thermodynamic integration
  log m = int_0^1 E_t[log L] dt over the tempered
  path p_t ~ pi * L^t.

`bench_ml`: conjugate normal-normal model has the
closed-form log m (N(y; mu0, s0^2 + s^2) product);
every estimator must land within tolerance bands on
independent MCMC-free draws (self-normalized IS /
exact Monte Carlo rather than full MCMC keeps the
self-check deterministic).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _logsumexp(a: FloatArray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.exp(a - m).sum()))


def harmonic_mean(loglik: FloatArray) -> float:
    """Newton-Raftery harmonic-mean estimate of log m.

    log m = -logmean(1/L) = -(logsumexp(-ll) - log n)."""
    ll = np.asarray(loglik, dtype=np.float64).ravel()
    if ll.size < 2 or not np.isfinite(ll).all():
        raise ValueError("bad draws")
    return float(-(_logsumexp(-ll) - np.log(ll.size)))


def gelfand_dey(loglik: FloatArray, logprior: FloatArray, logh: FloatArray) -> float:
    """Gelfand-Dey reciprocal estimator:
    1/m = E_post[h/(L pi)] -> log m = -logmean(h-lL-lPi)."""
    ll = np.asarray(loglik, dtype=np.float64)
    lp = np.asarray(logprior, dtype=np.float64)
    lh = np.asarray(logh, dtype=np.float64)
    if not (ll.size == lp.size == lh.size):
        raise ValueError("length mismatch")
    if not (np.isfinite(ll).all() and np.isfinite(lp).all() and np.isfinite(lh).all()):
        raise ValueError("non-finite input")
    v = lh - ll - lp
    return float(-(_logsumexp(v) - np.log(v.size)))


def chib(loglik_star: float, logprior_star: float, logpost_star: float) -> float:
    """Chib (1995): log m = log f(y|t*) + log pi(t*) -
    log pi(t*|y) at a high-density point."""
    if not np.isfinite([loglik_star, logprior_star, logpost_star]).all():
        raise ValueError("non-finite input")
    return float(loglik_star + logprior_star - logpost_star)


def savage_dickey(post_draws: FloatArray, prior_pdf_at: float, at: float) -> float:
    """Savage-Dickey density ratio at the null point:
    BF_10 = prior(at) / posterior(at); posterior density
    from a histogram of draws."""
    d = np.asarray(post_draws, dtype=np.float64).ravel()
    if d.size < 50:
        raise ValueError("too few draws")
    bw = 1.06 * float(d.std()) * d.size**-0.2  # Silverman
    if bw <= 0 or not np.isfinite(bw):
        raise ValueError("degenerate draws")
    z = (d - at) / bw
    post_at = float(stats.norm.pdf(z).mean() / bw)
    if post_at <= 0 or prior_pdf_at <= 0:
        raise ValueError("zero density at point")
    return float(np.log(prior_pdf_at / post_at))


def thermodynamic_integration(mean_loglik_by_beta: FloatArray, betas: FloatArray) -> float:
    """Power-posterior TI: log m = int_0^1 E_beta[log L] d beta
    via trapezoidal rule on the tempered means."""
    bs = np.asarray(betas, dtype=np.float64).ravel()
    means = np.asarray(mean_loglik_by_beta, dtype=np.float64).ravel()
    if bs.ndim != 1 or bs.size < 2 or bs.size != means.size:
        raise ValueError("beta/means mismatch")
    order = np.argsort(bs)
    return float(np.trapezoid(means[order], bs[order]))


def bench_ml(seed: int = 531) -> dict[str, float]:
    """SYNTHETIC: y_i | mu ~ N(mu, 1), mu ~ N(0, 4).
    The shared mean couples the draws — the closed-form
    log m comes from the quadratic form in mu."""
    rng = np.random.default_rng(seed)
    n = 40
    y = rng.normal(0.8, 1.0, n)
    # analytic marginal likelihood: integrating the shared
    # mean mu ~ N(0,4) couples the y_i — closed form:
    # m = (2 pi s2)^{-n/2} (2 pi t2)^{-1/2} sqrt(2 pi / P)
    #     exp(-0.5 [S_yy/s2 - (n ybar / s2)^2 / P]),
    # P = n/s2 + 1/t2 (posterior precision)
    s2, t2 = 1.0, 4.0
    P = n / s2 + 1 / t2
    ybar = float(y.mean())
    syy = float((y**2).sum())
    true_lm = float(
        -n / 2 * np.log(2 * np.pi * s2)
        - 0.5 * np.log(2 * np.pi * t2)
        + 0.5 * np.log(2 * np.pi / P)
        - 0.5 * (syy / s2 - (n * ybar / s2) ** 2 / P)
    )
    # exact posterior draws
    mu_post_sd = np.sqrt(1.0 / (n / 1.0 + 1 / 4.0))
    mu_post_mean = mu_post_sd**2 * (n * y.mean() / 1.0 + 0.0)
    draws = rng.normal(mu_post_mean, mu_post_sd, 4000)
    ll = np.array([np.sum(stats.norm.logpdf(y, m, 1.0)) for m in draws])
    hm = harmonic_mean(ll)
    # Gelfand-Dey with h = N(0,4) prior as covering density
    lp = np.array([stats.norm.logpdf(m, 0, 2.0) for m in draws])
    lh = np.array([stats.norm.logpdf(m, mu_post_mean, mu_post_sd) for m in draws])
    gd = gelfand_dey(ll, lp, lh)
    # Chib at posterior mean (posterior analytic here)
    th_star = mu_post_mean
    ll_s = float(stats.norm.logpdf(y, th_star, 1.0).sum())
    lp_s = float(stats.norm.logpdf(th_star, 0, 2.0))
    lpo_s = float(stats.norm.logpdf(th_star, mu_post_mean, mu_post_sd))
    cb = chib(ll_s, lp_s, lpo_s)
    # TI: E_beta[log L] over beta in [0,1] from prior draws
    # standard TI ladder: beta_k = (k/K)^5 concentrates
    # points where E_beta[log L] bends steeply near 0
    betas = np.linspace(0, 1, 21) ** 5
    # E_{pi_t}[log L]: pi_t ∝ pi * L^t — integrate via
    # IS weights on prior draws
    prior_draws = rng.normal(0, 2.0, 6000)
    ll_p = np.array([np.sum(stats.norm.logpdf(y, m, 1.0)) for m in prior_draws])
    means = []
    for b in betas:
        w = b * ll_p
        w = w - w.max()
        w = np.exp(w)
        means.append(float((w * ll_p).sum() / w.sum()))
    ti = thermodynamic_integration(np.asarray(means), betas)
    # Savage-Dickey density ratio at mu = posterior mode
    # (KDE is reliable where draws are dense)
    at = float(mu_post_mean)
    sd = savage_dickey(draws, float(stats.norm.pdf(at, 0, 2.0)), at)
    post_at_true = float(stats.norm.pdf(at, mu_post_mean, mu_post_sd))
    sd_true = float(np.log(stats.norm.pdf(at, 0, 2.0) / post_at_true))
    if abs(cb - true_lm) > 0.05:
        raise ValueError(f"chib off: {cb:.3f} vs {true_lm:.3f}")
    if abs(gd - true_lm) > 0.5:
        raise ValueError(f"gd off: {gd:.3f}")
    if abs(ti - true_lm) > 0.5:
        raise ValueError(f"ti off: {ti:.3f}")
    if abs(sd - sd_true) > 0.15:
        raise ValueError(f"savage-dickey off: {sd:.3f} vs {sd_true:.3f}")
    return {
        "synthetic_log_ml_true": true_lm,
        "synthetic_log_ml_chib": cb,
        "synthetic_log_ml_gd": gd,
        "synthetic_log_ml_ti": ti,
        "synthetic_log_ratio_sd": sd,
        "synthetic_log_ratio_true": sd_true,
        "synthetic_log_ml_hm": hm,
    }
