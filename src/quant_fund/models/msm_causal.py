"""Robins marginal structural models via stabilized IPTW (SYNTHETIC).

Two-visit marginal structural model for time-varying
treatment with treatment-confounder feedback. For
continuous Gaussian treatments the stabilized weights are

    sw_i = prod_t  f(A_t | A_{t-1}) /
                   f(A_t | A_{t-1}, L_t, baseline)

with conditional densities estimated by OLS (mean +
homoscedastic residual sd), truncated at a quantile cap.
Binary treatments use logistic propensities instead. The
MSM estimand is recovered by a weighted outcome
regression with a Huber-White sandwich SE.

Honesty: the bench simulates a DGP with known cumulative
exposure coefficient and treatment-confounder feedback
where naive regression is biased; it checks the MSM
estimate lands closer to the truth than the naive slope —
a recoverability diagnostic, not a real-treatment claim.

References
----------
* Robins, Hernan & Brumback (2000) "Marginal structural
  models and causal inference in epidemiology",
  Epidemiology 11(5), 550-560.
* Hernan, Brumback & Robins (2000) "Marginal structural
  models to estimate the causal effect of zidovudine...",
  JASA 96, 561-570.
* Cole & Hernan (2008) "Constructing inverse probability
  weights for marginal structural models", AJE 168(6).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _ols_moments(x: FloatArray, a: FloatArray) -> tuple[FloatArray, float]:
    """E[A|X] and residual sd from OLS."""
    n, p = x.shape
    xb = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(xb, a, rcond=None)[0]
    resid = a - xb @ beta
    sd = float(resid.std(ddof=min(n - 1, p + 1)))
    return np.asarray(xb @ beta, dtype=np.float64), max(sd, 1e-3)


def _logit_fit(x: FloatArray, a: FloatArray) -> FloatArray:
    """P(A=1 | X) via logistic regression (Newton, 30 iters)."""
    n, p = x.shape
    xb = np.column_stack([np.ones(n), x])
    beta = np.zeros(p + 1)
    for _ in range(30):
        mu = 1.0 / (1.0 + np.exp(-xb @ beta))
        w = np.clip(mu * (1.0 - mu), 1e-6, None)
        g = xb.T @ (a - mu)
        h = xb.T @ (xb * w[:, None])
        step = np.linalg.solve(h + 1e-8 * np.eye(p + 1), g)
        beta += step
        if float(np.abs(step).max()) < 1e-8:
            break
    mu = 1.0 / (1.0 + np.exp(-xb @ beta))
    return np.asarray(np.clip(mu, 1e-4, 1 - 1e-4), dtype=np.float64)


def _dens(x: FloatArray, a: FloatArray, binary: bool) -> FloatArray:
    """Conditional density/probability of A given X."""
    if binary:
        p = _logit_fit(x, a)
        return np.asarray(np.where(a > 0.5, p, 1.0 - p), dtype=np.float64)
    mu, sd = _ols_moments(x, a)
    z = (a - mu) / sd
    d = np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))
    return np.asarray(np.clip(d, 1e-10, None), dtype=np.float64)


def stabilized_weights(
    a1: FloatArray,
    l1: FloatArray,
    a2: FloatArray,
    l2: FloatArray,
    baseline: FloatArray,
    trunc: float = 0.99,
) -> FloatArray:
    """Two-visit stabilized weights.

    Numerator: P(A1) P(A2|A1) (marginal, so the weighted
    population has treatment independent of the whole
    covariate history).  Denominator:
    P(A1|L1,base) P(A2|A1,L2,base).
    Truncated at the ``trunc`` quantile.
    """
    aa1 = np.asarray(a1).ravel().astype(np.float64)
    aa2 = np.asarray(a2).ravel().astype(np.float64)
    ll1 = np.asarray(l1, dtype=np.float64)
    ll2 = np.asarray(l2, dtype=np.float64)
    bb = np.asarray(baseline, dtype=np.float64)
    if ll1.ndim == 1:
        ll1 = ll1[:, None]
    if ll2.ndim == 1:
        ll2 = ll2[:, None]
    if bb.ndim == 1:
        bb = bb[:, None]
    n = aa1.size
    if not (aa2.size == n == ll1.shape[0] == ll2.shape[0] == bb.shape[0]):
        raise ValueError("shape mismatch")
    b1 = np.unique(aa1).size <= 2
    b2 = np.unique(aa2).size <= 2
    # marginal numerator f(A1) f(A2|A1): weighting then makes
    # A1 and A2 independent of the whole covariate history
    # (L1, L2, baseline) in the pseudo-population, so an
    # outcome model on cumulative dose alone is unbiased.
    empty = np.empty((n, 0), dtype=np.float64)
    num = _dens(empty, aa1, b1) * _dens(aa1[:, None], aa2, b2)
    den = _dens(np.column_stack([bb, ll1]), aa1, b1) * _dens(
        np.column_stack([bb, aa1, ll2]), aa2, b2
    )
    sw = num / den
    cap = float(np.quantile(sw, trunc))
    return np.asarray(np.clip(sw, 0.0, cap), dtype=np.float64)


def msm_cumulative_effect(
    y: FloatArray,
    a1: FloatArray,
    l1: FloatArray,
    a2: FloatArray,
    l2: FloatArray,
    baseline: FloatArray,
) -> dict[str, float | FloatArray]:
    """Weighted MSM for cumulative exposure effect.

    Fits  E[Y] = psi0 + psi1*(A1+A2)  weighted by stabilized
    weights; returns psi1, a naive (unweighted) comparison
    slope, and a robust SE for psi1.
    """
    yy = np.asarray(y, dtype=np.float64).ravel()
    aa1 = np.asarray(a1).ravel().astype(np.float64)
    aa2 = np.asarray(a2).ravel().astype(np.float64)
    if yy.size != aa1.size:
        raise ValueError("shape mismatch")
    sw = stabilized_weights(aa1, l1, aa2, l2, baseline)
    dose = aa1 + aa2
    x = np.column_stack([np.ones(yy.size), dose])
    # weighted least squares
    w_sqrt = np.sqrt(sw)
    xw = x * w_sqrt[:, None]
    yw = yy * w_sqrt
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    resid = yy - x @ beta
    # Huber-White sandwich on the weighted fit
    meat = x.T @ ((sw * resid**2)[:, None] * x)
    bread = x.T @ (sw[:, None] * x)
    cov = np.linalg.solve(bread, meat @ np.linalg.inv(bread))
    se1 = float(math.sqrt(max(cov[1, 1], 0.0)))
    beta_naive = np.linalg.lstsq(x, yy, rcond=None)[0]
    t_stat = float(beta[1] / max(se1, 1e-12))
    return {
        "psi1": float(beta[1]),
        "psi1_se": se1,
        "psi1_p": float(2 * stats.norm.sf(abs(t_stat))),
        "naive_beta1": float(beta_naive[1]),
        "sw_mean": float(sw.mean()),
        "sw_max": float(sw.max()),
        "weights": np.asarray(sw, dtype=np.float64),
    }


def bench_msm_causal(seed: int = 20261231 + 465) -> dict[str, float]:
    """SYNTHETIC check — MSM recovers causal effect under feedback."""
    rng = np.random.default_rng(seed)
    n = 4000
    base = rng.normal(size=n)
    # continuous Gaussian treatments; measured confounders
    # with treatment-confounder feedback:
    # L1 <- base ; A1 <- L1 ; L2 <- L1 + A1 ; A2 <- L2 + A1 ;
    # Y  <- psi*(A1+A2) + L1 + noise.  L2 confounds A2 but is
    # not in the outcome equation, so the marginal dose
    # effect stays psi; naive regression is biased through
    # (L1, L2) -> treatment.  MSM weights on (L1, A1, L2)
    # recover psi.
    l1 = 0.8 * base + rng.normal(scale=0.5, size=n)
    a1 = 0.5 * l1 + rng.normal(scale=0.9, size=n)
    l2 = 0.6 * l1 + 0.7 * a1 + rng.normal(scale=0.6, size=n)
    a2 = 0.5 * l2 + 0.2 * a1 + rng.normal(scale=0.9, size=n)
    true_psi = 0.5
    y = true_psi * (a1 + a2) + 1.3 * l1 + rng.normal(scale=0.9, size=n)
    out = msm_cumulative_effect(y, a1, l1, a2, l2, base)
    psi = float(out["psi1"])
    naive = float(out["naive_beta1"])
    if abs(psi - true_psi) > 0.12:
        raise ValueError(f"msm off: psi={psi} true={true_psi}")
    if abs(naive - true_psi) < abs(psi - true_psi):
        raise ValueError(f"msm not better: psi={psi} naive={naive}")
    return {
        "synthetic_psi1": psi,
        "synthetic_naive_gap": float(naive - true_psi),
        "synthetic_msm_gap": float(psi - true_psi),
        "synthetic_score": 1.0,
    }
