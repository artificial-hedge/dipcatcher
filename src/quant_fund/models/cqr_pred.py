"""Conformalized Quantile Regression (Romano et al. 2019) — split
conformal on quantile scores E = max(q_lo − y, y − q_hi); interval
[q_lo − Q, q_hi + Q]. Coverage + width vs split conformal.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cp2_synth import cov_width, cp2_data


def _fit_qr(X: np.ndarray, y: np.ndarray, tau: float, lam: float = 1.0) -> np.ndarray:
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = np.zeros(A.shape[1])
    for _ in range(300):
        r = y - A @ w
        grad = -A.T @ (tau * (r > 0) - (1 - tau) * (r <= 0)) / len(y) + lam * w
        w -= 0.05 * grad
    return w


def bench_cqr_pred(seed: int = 1301, alpha: float = 0.1) -> dict[str, float]:
    X, y, Xt, yt = cp2_data(seed)
    n_tr = 2 * len(X) // 3
    Xtr, Xcal, ytr, ycal = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    wlo = _fit_qr(Xtr, ytr, alpha / 2)
    whi = _fit_qr(Xtr, ytr, 1 - alpha / 2)
    Ap = lambda W, Z: np.concatenate([Z, np.ones((len(Z), 1))], 1) @ W  # noqa: E731
    e = np.maximum(Ap(wlo, Xcal) - ycal, ycal - Ap(whi, Xcal))
    Q = np.quantile(e, np.ceil((len(e) + 1) * (1 - alpha)) / len(e))
    lo, hi = Ap(wlo, Xt) - Q, Ap(whi, Xt) + Q
    c, w = cov_width(lo, hi, yt)
    # split conformal on |resid| with median fit
    wm = _fit_qr(Xtr, ytr, 0.5)
    r = np.abs(ycal - Ap(wm, Xcal))
    Q2 = np.quantile(r, np.ceil((len(r) + 1) * (1 - alpha)) / len(r))
    lo2, hi2 = Ap(wm, Xt) - Q2, Ap(wm, Xt) + Q2
    c2, w2 = cov_width(lo2, hi2, yt)
    return {
        "synthetic_cqr_coverage": c,
        "synthetic_cqr_target": 1 - alpha,
        "synthetic_cqr_width": w,
        "synthetic_cqr_split_width": w2,
        "synthetic_cqr_width_gain": w2 - w,
        "synthetic_cqr_split_coverage": c2,
        "synthetic_torch_available": 0.0,
    }
