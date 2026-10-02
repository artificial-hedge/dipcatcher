"""Conformal risk control (Angelopoulos et al. 2022) — calibrate lambda
so E[risk(lambda)] <= alpha for a non-binary risk (relative absolute
error truncated); risk level vs naive quantile calibration.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cp2_synth import cp2_data, fit_ridge, ridge_pred


def bench_risk_cp(seed: int = 1321, alpha: float = 0.2) -> dict[str, float]:
    X, y, Xt, yt = cp2_data(seed)
    n_tr = 2 * len(X) // 3
    w = fit_ridge(X[:n_tr], y[:n_tr])
    mu_c = ridge_pred(w, X[n_tr:])
    r = np.abs(y[n_tr:] - mu_c)
    mu_t = ridge_pred(w, Xt)
    # risk(lambda) = E[ clip(|resid| - lambda, 0, 1) ]; choose smallest lam with
    # (n+1)/n * mean risk + B/(n+1) <= alpha
    grid = np.linspace(0, 2.0, 80)
    lam_star = grid[-1]
    n_c = len(r)
    for lam in grid:
        risk = np.clip(r - lam, 0, 1).mean() * (n_c + 1) / n_c
        if risk <= alpha:
            lam_star = lam
            break
    risk_t = float(np.clip(np.abs(yt - mu_t) - lam_star, 0, 1).mean())
    lam_naive = float(np.quantile(r, 1 - alpha))
    risk_t2 = float(np.clip(np.abs(yt - mu_t) - lam_naive, 0, 1).mean())
    return {
        "synthetic_rcp_risk": risk_t,
        "synthetic_rcp_target": alpha,
        "synthetic_rcp_naive_risk": risk_t2,
        "synthetic_rcp_risk_gain": risk_t2 - risk_t,
        "synthetic_rcp_radius": lam_star,
        "torch_available": 0.0,
    }
