"""Full (transductive) conformal — refit ridge per candidate y on a
coarse grid and invert the p-value; tighter intervals than split
conformal at the cost of refits. Width vs split conformal.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cp2_synth import cp2_data, fit_ridge, ridge_pred


def bench_full_cp(seed: int = 1319, alpha: float = 0.1, n_eval: int = 120) -> dict[str, float]:
    X, y, Xt, yt = cp2_data(seed, n=400)
    Xt, yt = Xt[:n_eval], yt[:n_eval]
    Xc = np.concatenate([X, Xt[:1] * 0])  # calibration = train pool
    Xc = X
    yc = y
    grid_half = 4.0
    lo_f = np.zeros(n_eval)
    hi_f = np.zeros(n_eval)
    for i in range(n_eval):
        mu0 = ridge_pred(fit_ridge(Xc, yc), Xt[i : i + 1])[0]
        cands = np.linspace(mu0 - grid_half, mu0 + grid_half, 33)
        inside = np.zeros(len(cands), bool)
        for j, cand in enumerate(cands):
            Xa = np.vstack([Xc, Xt[i : i + 1]])
            ya = np.concatenate([yc, [cand]])
            w = fit_ridge(Xa, ya, lam=1.0)
            r = np.abs(ya - ridge_pred(w, Xa))
            inside[j] = r[-1] <= np.quantile(r[:-1], 1 - alpha)
        kept = cands[inside]
        lo_f[i] = kept.min() if kept.size else mu0 - grid_half
        hi_f[i] = kept.max() if kept.size else mu0 + grid_half
    cov = float(np.mean((yt >= lo_f) & (yt <= hi_f)))
    # split conformal width on same data
    n_tr = len(X) // 2
    w = fit_ridge(X[:n_tr], y[:n_tr])
    r = np.abs(y[n_tr:] - ridge_pred(w, X[n_tr:]))
    Q = np.quantile(r, np.ceil((len(r) + 1) * (1 - alpha)) / len(r))
    mu_t = ridge_pred(w, Xt)
    cov2 = float(np.mean((yt >= mu_t - Q) & (yt <= mu_t + Q)))
    return {
        "synthetic_fcp_coverage": cov,
        "synthetic_fcp_target": 1 - alpha,
        "synthetic_fcp_width": float(np.mean(hi_f - lo_f)),
        "synthetic_fcp_split_coverage": cov2,
        "synthetic_fcp_split_width": 2 * Q,
        "synthetic_fcp_width_gain": 2 * Q - float(np.mean(hi_f - lo_f)),
        "torch_available": 0.0,
    }
