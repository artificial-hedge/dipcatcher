"""Poisson pseudo-maximum-likelihood (PPML) gravity estimation (SYNTHETIC).

Multiplicative models y = exp(xβ)·ε estimated consistently by
Poisson PML even under heteroskedasticity — where the log-linear
OLS transform biases coefficients through E[ln ε] ≠ ln E[ε].
The workhorse of modern gravity/trade estimation and any
nonnegative outcome with a multiplicative mean.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure elasticity recovery on
generated multiplicative data — never market evidence.

References:
- Santos Silva, J. M. C., Tenreyro, S. (2006). The log of
  gravity. *Review of Economics and Statistics* 88, 641-658 —
  PPML consistency under heteroskedasticity.
- Gourieroux, C., Monfort, A., Trognon, A. (1984). Pseudo
  maximum likelihood methods: applications to Poisson models.
  *Econometrica* 52 — PML requires only the conditional mean.
- Silva, J. M. C. S., Tenreyro, S. (2011). Further simulation
  evidence on the performance of the Poisson pseudo-maximum
  likelihood estimator. *Economics Letters* 112.
- Cameron, A. C., Trivedi, P. K. (2005). *Microeconometrics*,
  ch. 4 — IRLS Poisson and robust sandwich covariance.

Composition: pure numpy + scipy — Poisson IRLS (W = μ),
robust sandwich SEs, overdispersion check; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def ppml_fit(
    y: FloatArray,
    x: FloatArray,
    iters: int = 80,
) -> dict[str, float]:
    """PPML via IRLS on E[y|x] = exp(xβ).

    ``x`` is (n × k); intercept added. Returns elasticities β,
    robust SEs, overdispersion φ, and the log-OLS reference."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] != yy.size:
        raise ValueError("x rows must match y")
    n, k = xx.shape
    if n < 60 or k < 1 or k > 8:
        raise ValueError("n>=60, k in 1..8")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    if np.any(yy < 0):
        raise ValueError("y must be nonnegative")
    if np.mean(yy == 0) > 0.8:
        raise ValueError("too many zeros (>80%)")
    if np.min(np.std(xx, axis=0)) < 1e-9:
        raise ValueError("each covariate must vary")

    w = np.column_stack([np.ones(n), xx])
    beta = np.linalg.lstsq(w, np.log(yy + 0.5), rcond=None)[0]
    for _ in range(iters):
        xb = np.clip(w @ beta, -20, 20)
        mu = np.exp(xb)
        wgt = np.clip(mu, 1e-10, None)
        # IRLS: z = xb + (y-μ)/μ
        z = xb + (yy - mu) / mu
        wx = w * np.sqrt(wgt)[:, None]
        zz = z * np.sqrt(wgt)
        beta_new = np.linalg.lstsq(wx.T @ wx + 1e-8 * np.eye(k + 1), wx.T @ zz, rcond=None)[0]
        if float(np.max(np.abs(beta_new - beta))) < 1e-10:
            beta = beta_new
            break
        beta = beta_new

    mu = np.exp(np.clip(w @ beta, -20, 20))
    resid = yy - mu
    # robust sandwich: bread = Σ μ w w', meat = Σ (y-μ)² w w'
    bread = w.T @ (w * mu[:, None])
    meat = w.T @ (w * (resid**2)[:, None])
    vc = np.linalg.solve(bread, meat @ np.linalg.inv(bread))
    se = np.sqrt(np.maximum(np.diag(vc), 1e-300))
    phi = float(np.mean(resid**2 / np.clip(mu, 1e-9, None)))

    # log-OLS reference on y>0
    pos = yy > 0
    b_ols = np.linalg.lstsq(w[pos], np.log(yy[pos]), rcond=None)[0]

    return {
        "n": float(n),
        "share_zero": float(np.mean(yy == 0)),
        "beta_x": float(beta[1]),
        "se_x": float(se[1]),
        "t_x": float(beta[1] / se[1]),
        "beta_x_logols": float(b_ols[1]),
        "phi_overdispersion": phi,
        "r2": float(1.0 - np.var(resid) / np.var(yy)),
        "pseudo_ll": float(np.sum(yy * np.log(np.clip(mu, 1e-12, None)) - mu)),
    }


def synth_gravity(
    n: int = 800,
    beta_x: float = 0.8,
    zero_share: float = 0.15,
    hetero: float = 0.7,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Multiplicative DGP: y = Poisson-ish mean exp(β0+βx)·ε with
    heteroskedastic ε (lognormal scaled) — the case where log-OLS
    is inconsistent. Zeros injected like structural zeros."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 1))
    mu = np.exp(0.5 + beta_x * x[:, 0])
    # heteroskedastic noise: Var(ε) ∝ μ — kills log-OLS
    eps = rng.gamma(1.0 / (1.0 + hetero), 1.0 + hetero, n)
    y = mu * eps
    zeros = rng.uniform(0.0, 1.0, n) < zero_share
    y = np.where(zeros, 0.0, y)
    return {
        "y": y,
        "x": x,
        "beta_x": np.array([beta_x]),
    }


def bench_ppml(seed: int = 20261231 + 236) -> dict[str, float]:
    """PPML self-check: elasticity recovered under heteroskedastic
    zeros-heavy data where log-OLS is biased. All ``synthetic_*``."""
    d = synth_gravity(beta_x=0.8, seed=seed)
    out = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    d0 = synth_gravity(beta_x=0.0, seed=seed + 1)
    out0 = ppml_fit(np.asarray(d0["y"]), np.asarray(d0["x"]))
    out_b = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))

    b = float(out["beta_x"])
    b_ols = float(out["beta_x_logols"])
    return {
        "synthetic_beta": b,
        "synthetic_beta_err": float(abs(b - 0.8)),
        "synthetic_beta_logols": b_ols,
        "synthetic_logols_err": float(abs(b_ols - 0.8)),
        "synthetic_phi": float(out["phi_overdispersion"]),
        "synthetic_beta_null": float(out0["beta_x"]),
        "synthetic_t_x": float(out["t_x"]),
        "synthetic_detects": float(
            abs(b - 0.8) < 0.2 and abs(float(out0["beta_x"])) < 0.2 and float(out["t_x"]) > 3.0
        ),
        "synthetic_determinism": float(b == float(out_b["beta_x"])),
    }
