"""Ordered choice models — ordered probit/logit via threshold MLE (SYNTHETIC).

Latent y* = xβ + ε crosses ordered cutpoints to produce the observed
category. Maximizing the category likelihood over (β, cutpoints)
recovers the latent-scale coefficients; marginal effects and
category probabilities follow.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure coefficient recovery on generated
ordered panels — never market evidence.

References:
- McKelvey, R. D., Zavoina, W. (1975). A statistical model for the
  analysis of ordinal level dependent variables. *J. Math. Soc.* 4,
  103-120 — the ordered probit threshold model.
- McCullagh, P. (1980). Regression models for ordinal data.
  *JRSS-B* 42, 109-142 — proportional odds (ordered logit).
- Greene, W. H., Hensher, D. A. (2010). *Modeling Ordered Choices*.
  Cambridge — marginal effects and model-fit conventions.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*,
  2nd ed. — MLE setup and identification normalization.

Composition: numpy + scipy.optimize only; cutpoints parametrized as
first cut + log-increments to enforce ordering; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import logistic, norm

FloatArray = NDArray[np.float64]


def _as_xy(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if xx.shape[0] != yy.size:
        xx = xx.T
    if yy.size < 20 or xx.shape[0] != yy.size:
        raise ValueError("x row count must equal len(y), n>=20")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite y and x required")
    return yy, xx


def _unpack(theta: FloatArray, k: int, ncat: int) -> tuple[FloatArray, FloatArray]:
    """theta = [β(k), c1, log-incs(ncat-2)] → ordered cutpoints."""
    b = theta[:k]
    c1 = theta[k]
    incs = np.exp(theta[k + 1 :])
    cuts = np.concatenate([[c1], c1 + np.cumsum(incs)])
    if cuts.size != ncat - 1:
        raise ValueError("cutpoint count mismatch")
    return b, cuts


def ordered_fit(
    y: FloatArray,
    x: FloatArray,
    link: str = "probit",
) -> dict[str, float]:
    """ML fit of the ordered probit/logit model.

    P(y = j | x) = F(c_j - xβ) - F(c_{j-1} - xβ) with ordered cuts;
    β excludes an intercept (absorbed by the cutpoint location)."""
    yy, xx = _as_xy(y, x)
    cats = np.unique(yy)
    ncat = cats.size
    if ncat < 3 or ncat > 12:
        raise ValueError("3..12 observed categories required")
    if not np.all(cats == np.round(cats)):
        raise ValueError("categories must be integer-coded")
    # relabel to 0..ncat-1 contiguous
    lab = np.searchsorted(cats, yy)
    n, k = xx.shape
    if link == "probit":
        cdf = norm.cdf
    elif link == "logit":
        cdf = logistic.cdf
    else:
        raise ValueError("link must be 'probit' or 'logit'")

    def nll(theta: FloatArray) -> float:
        b, cuts = _unpack(theta, k, ncat)
        eta = xx @ b
        lo = np.concatenate([[-np.inf], cuts])
        hi = np.concatenate([cuts, [np.inf]])
        p = np.clip(cdf(hi[lab] - eta) - cdf(lo[lab] - eta), 1e-12, 1.0)
        return float(-np.sum(np.log(p)))

    theta0 = np.zeros(k + ncat - 1)
    theta0[k:] = np.concatenate([[0.0], np.full(ncat - 2, math.log(1.0))])
    res = minimize(nll, theta0, method="BFGS")
    if not np.all(np.isfinite(res.x)):
        raise ValueError("ordered model did not converge")
    b_hat, cuts = _unpack(res.x, k, ncat)
    nll_final = float(res.fun)

    # null (intercept-only) log-likelihood for McFadden R²
    probs = np.bincount(lab, minlength=ncat) / n
    nll0 = float(-np.sum(np.log(probs[lab])))

    # category-probability fit: mean absolute calibration error
    eta = xx @ b_hat
    lo = np.concatenate([[-np.inf], cuts])
    hi = np.concatenate([cuts, [np.inf]])
    p_mat = cdf(hi[:, None] - eta[None, :]) - cdf(lo[:, None] - eta[None, :])
    pred = p_mat.argmax(axis=0)
    mae_prob = float(np.mean(np.abs(p_mat.mean(axis=1) - np.bincount(lab, minlength=ncat) / n)))

    return {
        "n": float(n),
        "n_categories": float(ncat),
        "mcfadden_r2": float(1.0 - nll_final / nll0),
        "mae_prob": mae_prob,
        "pred_acc": float(np.mean(pred == lab)),
        **{f"beta_{j}": float(v) for j, v in enumerate(b_hat)},
        **{f"cut_{j}": float(v) for j, v in enumerate(cuts)},
    }


def synth_ordered(
    n: int = 1200,
    beta: float = 1.0,
    ncat: int = 5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Ordered-probit DGP: y* = βx + ε, ε ~ N(0,1); categories from
    even cutpoints over y*."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(0.0, 1.0, n), rng.normal(0.0, 0.6, n)])
    beta_true = np.array([beta, 0.4])
    ystar = x @ beta_true + rng.normal(0.0, 1.0, n)
    cuts = np.linspace(-1.5, 1.5, ncat - 1)
    y = np.searchsorted(cuts, ystar).astype(np.float64)
    return {"y": y, "x": x, "cuts_true": cuts}


def bench_ordered_choice(seed: int = 20261231 + 210) -> dict[str, float]:
    """Ordered-probit self-check: β recovery and category-probability
    calibration on a 5-point latent panel. All ``synthetic_*``."""
    d = synth_ordered(n=1500, beta=1.0, seed=seed)
    out = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    d0 = synth_ordered(n=1500, beta=0.0, seed=seed + 1)
    out0 = ordered_fit(np.asarray(d0["y"]), np.asarray(d0["x"]))
    out_b = ordered_fit(np.asarray(d["y"]), np.asarray(d["x"]))

    b0 = float(out["beta_0"])
    b1 = float(out["beta_1"])
    return {
        "synthetic_beta0": b0,
        "synthetic_beta0_err": float(abs(b0 - 1.0)),
        "synthetic_beta1_err": float(abs(b1 - 0.4)),
        "synthetic_mcfadden_r2": float(out["mcfadden_r2"]),
        "synthetic_mae_prob": float(out["mae_prob"]),
        "synthetic_pred_acc": float(out["pred_acc"]),
        "synthetic_null_beta0": float(abs(out0["beta_0"])),
        "synthetic_detects": float(abs(b0 - 1.0) < 0.2 and abs(float(out0["beta_0"])) < 0.3),
        "synthetic_determinism": float(b0 == float(out_b["beta_0"])),
    }
