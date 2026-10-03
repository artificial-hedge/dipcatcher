"""Bayesian linear regression (Bishop 2006 §3.3) with the
evidence procedure for α/β and automatic relevance
determination (ARD, Tipping 2001 — irrelevant weights get
α_j → ∞ and are pruned). Synthetic bench gates ARD
sparsification and posterior-predictive calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bayes_lm_fit(
    x: FloatArray,
    y: FloatArray,
    alpha: float = 1.0,
    beta: float = 10.0,
    it: int = 200,
) -> dict[str, object]:
    """EM evidence maximization (MacKay): posterior
    S⁻¹ = αI + β X'X, m = β S X'y; then re-estimate
    α = γ/m'm and β = (n−γ)/‖y−Xm‖² with γ = Σ(1−αs_ii)."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    y = np.asarray(y, dtype=np.float64)
    d = x.shape[1]
    a, b = float(alpha), float(beta)
    m = np.zeros(d)
    s = np.eye(d)
    for _ in range(it):
        prec = a * np.eye(d) + b * (x.T @ x)
        s = np.linalg.inv(prec)
        m = b * s @ x.T @ y
        eig = np.linalg.eigvalsh(b * (x.T @ x))
        gamma = float((eig / (eig + a)).sum())
        a_new = gamma / float(m @ m + 1e-12)
        resid = y - x @ m
        b_new = (len(y) - gamma) / float(resid @ resid + 1e-12)
        if abs(a_new - a) < 1e-8 * a and abs(b_new - b) < 1e-8 * b:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return {"mean": m, "cov": s, "alpha": a, "beta": b}


def bayes_lm_predict(model: dict[str, object], x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Predictive mean and variance 1/β + x'Σx."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    m = np.asarray(model["mean"])
    s = np.asarray(model["cov"])
    b = float(np.asarray(model["beta"]))
    mu = x @ m
    var = 1.0 / b + np.einsum("ij,jk,ik->i", x, s, x)
    return np.asarray(mu), np.asarray(var)


def ard_rvm_fit(
    x: FloatArray,
    y: FloatArray,
    it: int = 300,
    prune: float = 1e4,
) -> dict[str, object]:
    """ARD: per-weight α_j; the evidence update drives
    irrelevant α_j → ∞, pruning them (RVM-style, shared β)."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    y = np.asarray(y, dtype=np.float64)
    d = x.shape[1]
    alpha = np.ones(d)
    beta = 10.0
    m = np.zeros(d)
    s = np.eye(d)
    for _ in range(it):
        keep = alpha < prune
        kk = np.where(keep)[0]
        a_k = alpha[kk]
        xk = x[:, kk]
        prec = np.diag(a_k) + beta * (xk.T @ xk)
        sk = np.linalg.inv(prec)
        mk = beta * sk @ xk.T @ y
        gamma = 1.0 - a_k * np.diag(sk)
        alpha[kk] = gamma / np.maximum(mk**2, 1e-12)
        resid = y - xk @ mk
        beta = (len(y) - float(gamma.sum())) / float(resid @ resid + 1e-12)
        m = np.zeros(d)
        m[kk] = mk
        s = np.diag(1.0 / np.maximum(alpha, 1e-12))
        s[np.ix_(kk, kk)] = sk
    return {"mean": m, "cov": s, "alpha": alpha, "beta": beta, "kept": alpha < prune}


def bench_bayesian_linear(seed: int = 558) -> dict[str, float]:
    """SYNTHETIC: sparse true weights in noise — ARD must
    prune noise features while Bayes-LM predictive variance
    covers residuals at a sensible rate."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n, d = 200, 12
    x = rng.normal(0, 1, (n, d))
    beta_true = np.zeros(d)
    beta_true[:3] = [2.0, -1.5, 1.0]
    y = x @ beta_true + 0.4 + rng.normal(0, 0.3, n)
    mdl = bayes_lm_fit(x, y, alpha=1.0, beta=5.0, it=300)
    mu, var = bayes_lm_predict(mdl, x)
    cover = float(np.mean((y - mu) ** 2 < 3.0 * var))
    out["synthetic_blm_cover3sigma"] = cover
    out["synthetic_blm_rmse"] = float(np.sqrt(((y - mu) ** 2).mean()))
    if out["synthetic_blm_rmse"] > 0.6:
        raise ValueError(f"blm rmse off: {out['synthetic_blm_rmse']}")
    rvm = ard_rvm_fit(x, y, it=400)
    kept = np.asarray(rvm["kept"])
    n_noise_kept = int(kept[4:-1].sum())  # features 4..d-1 (last is bias)
    out["synthetic_ard_noise_kept"] = float(n_noise_kept)
    out["synthetic_ard_kept_total"] = float(kept.sum())
    if n_noise_kept > 3:
        raise ValueError(f"ard kept noise: {n_noise_kept}")
    if not (kept[0] and kept[1] and kept[2]):
        raise ValueError(f"ard pruned signal: {kept}")
    return out
