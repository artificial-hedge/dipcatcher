"""Heteroscedastic GP regression — two-stage: homoscedastic RBF GP for (SYNTHETIC)
the mean, then RBF regression on log squared residuals for σ²(x), refit
with input-dependent noise. Test log-density vs Gaussian baseline.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from quant_fund.models._cd_synth import cd_data, gauss_logpdf


def _rbf(a: np.ndarray, b: np.ndarray, ell: float) -> np.ndarray:
    out: np.ndarray = np.exp(-0.5 * ((a[:, None] - b[None]) ** 2).sum(-1) / ell**2)
    return out


def bench_het_gp(seed: int = 833) -> dict[str, float]:
    x, y, x_te, y_te = cd_data(seed, n=200)
    # stage 1: homoscedastic GP mean + residual variance
    K = _rbf(x, x, 1.0) + np.eye(len(x)) * 0.05
    c = cho_factor(K)
    alpha = cho_solve(c, y)
    resid = y - K.dot(alpha) + np.diag(K) * alpha * 0  # in-sample residual approx
    resid = y - _rbf(x, x, 1.0).dot(alpha)
    # stage 2: variance GP on log resid² (softened)
    lv = np.log(resid**2 + 1e-2)
    Kv = _rbf(x, x, 1.5) + np.eye(len(x)) * 1e-4
    wv = np.linalg.solve(Kv, lv)
    sd_te = np.exp(0.5 * (_rbf(x_te, x, 1.5) @ wv))
    # stage 3: refit with input-dependent noise
    sd_tr = np.exp(0.5 * (Kv @ wv))
    K2 = _rbf(x, x, 1.0) + np.diag(sd_tr**2 + 1e-6)
    c2 = cho_factor(K2)
    alpha2 = cho_solve(c2, y)
    k_te = _rbf(x_te, x, 1.0)
    mu = k_te @ alpha2
    var = np.maximum(1.0 + sd_te**2 - (k_te * cho_solve(c2, k_te.T).T).sum(-1), 1e-4)
    ll = -0.5 * np.log(2 * np.pi * var) - (y_te - mu) ** 2 / (2 * var)
    ll_gauss = gauss_logpdf(y_te, float(y.mean()), float(y.std()))
    return {
        "synthetic_hetgp_test_ll": float(ll.mean()),
        "synthetic_hetgp_gauss_ll": float(ll_gauss.mean()),
        "synthetic_hetgp_ll_gain": float(ll.mean() - ll_gauss.mean()),
        "synthetic_hetgp_sd_ratio": float(sd_te.max() / max(sd_te.min(), 1e-9)),
    }
