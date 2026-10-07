"""Learn-Then-Test (Angelopoulos et al. 2021) — conformal risk control:
choose lambda over a grid so the FWER/FDR-style bound on set-size risk
holds; interval/set size vs naive fixed threshold.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cp2_synth import cp2_data, fit_ridge, ridge_pred


def bench_ltt_cp(seed: int = 1313, alpha: float = 0.1) -> dict[str, float]:
    X, y, Xt, yt = cp2_data(seed)
    n_tr = 2 * len(X) // 3
    w = fit_ridge(X[:n_tr], y[:n_tr])
    mu_cal = ridge_pred(w, X[n_tr:])
    r = np.abs(y[n_tr:] - mu_cal)
    mu_t = ridge_pred(w, Xt)
    # LTT: pick smallest radius whose UCB on miscoverage <= alpha
    grid = np.linspace(0, 2.5, 60)
    n_c = len(r)
    lam_star = grid[-1]
    for lam in grid:
        loss = (r > lam).astype(float)
        ucb = loss.mean() + np.sqrt(loss.var() / n_c * np.log(1 / 0.05) / 2)
        if ucb <= alpha:
            lam_star = lam
            break
    lo, hi = mu_t - lam_star, mu_t + lam_star
    cov = float(np.mean((yt >= lo) & (yt <= hi)))
    # naive: alpha quantile radius (no conservativeness)
    lam_naive = float(np.quantile(r, 1 - alpha))
    cov2 = float(np.mean((yt >= mu_t - lam_naive) & (yt <= mu_t + lam_naive)))
    return {
        "synthetic_ltt_coverage": cov,
        "synthetic_ltt_target": 1 - alpha,
        "synthetic_ltt_width": 2 * lam_star,
        "synthetic_ltt_naive_coverage": cov2,
        "synthetic_ltt_cov_gain": cov - cov2,
        "synthetic_ltt_radius": lam_star,
        "synthetic_torch_available": 0.0,
    }
