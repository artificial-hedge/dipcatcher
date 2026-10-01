"""SPCI: sequential predictive conformal inference via residual forecasting.

Xu & Xie's SPCI differs from adaptive-alpha conformal (ACI — already in
``models/conformal.py``) in *what* adapts: rather than moving the quantile
level alpha_t, SPCI keeps alpha fixed and *forecasts the residual quantile
itself* — a small quantile regressor on the conformity-score series' own
lag features predicts the next score quantile, which becomes the interval
half-width. Under distribution shift the forecaster tracks the residual
level directly instead of waiting for miscoverage feedback.

Also includes the ACI-dual comparator (``aci_comparison``) which runs the
same stream through ``models.conformal.AdaptiveConformal`` so the bench can
report adaptive-alpha vs residual-forecasting side by side.

References
----------
- Xu & Xie (2021). Conformal prediction interval for dynamic time-series.
  *ICML 2021*. arXiv:2010.09107 (verified — SPCI).
- Gibbs & Candès (2021). Adaptive conformal inference under distribution
  shift. *NeurIPS 2021*. arXiv:2106.00170 (verified — ACI baseline only;
  implementation already lives in ``models/conformal.py``).
- Gibbs & Candès (2022). Conformal inference for online prediction with
  arbitrary distribution shifts. arXiv:2208.08401 (verified).
- Zaffran, Féron, Goude, Josse & Dieuleveut (2022). Adaptive conformal
  predictions for time series. *ICML 2022*. arXiv:2202.07282 (verified).
- Koenker & Bassett (1978). Regression quantiles. *Econometrica* 46 —
  pinball-loss quantile fitting used by the residual forecaster.

Honesty
-------
All benches run on seeded SYNTHETIC drift/heteroscedastic series generated
in-module. Coverage numbers validate the tracking machinery only — never
market evidence.

Composition notes
-----------------
- ``models/conformal.py``: split-CQR + ACI (adaptive alpha). Composed here
  as the comparator; this module's contribution is the *score-forecasting*
  mechanism, not alpha adaptation.
- ``models/online_crc.py``: CRC threshold with ACI dual — risk-control
  framing, different object (bounded loss threshold, not interval).
- ``models/enbpi.py`` / ``conformal_pid.py``: ensemble bootstrap PIs and
  PID-weighted conformal — sibling online-conformal mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.conformal import AdaptiveConformal

FloatArray = NDArray[np.float64]


def _check_series(x: FloatArray, min_len: int = 32) -> FloatArray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size < min_len:
        raise ValueError(f"series needs >= {min_len} points")
    if not np.isfinite(a).all():
        raise ValueError("series contains non-finite values")
    return a


def pinball_ridge_fit(
    x: FloatArray, y: FloatArray, alpha: float, ridge: float = 1e-3, iters: int = 60
) -> FloatArray:
    """Linear quantile fit via iteratively-reweighted least squares + L2 ridge.

    IRLS on the pinball loss: each iterate solves the weighted ridge
    problem with weights rho'_alpha(r_i)/r_i (smoothed near r=0). Features
    are raw lag blocks; intercept appended internally. Returns the
    coefficient vector (last entry = intercept).
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 8:
        raise ValueError("x must be (n>=8, d)")
    if not np.isfinite(a).all():
        raise ValueError("x contains non-finite values")
    y = np.asarray(y, dtype=float).ravel()
    if y.size != a.shape[0] or not np.isfinite(y).all():
        raise ValueError("y length must match x rows")
    if not 0 < alpha < 1:
        raise ValueError("alpha in (0,1)")
    n, d = a.shape
    xb = np.hstack([a, np.ones((n, 1))])
    # start from the ridge-LS solution
    gram = xb.T @ xb + ridge * np.eye(d + 1)
    w = np.linalg.solve(gram, xb.T @ y)
    eps = max(1e-3 * float(np.std(y)), 1e-6)
    for _ in range(iters):
        resid = y - xb @ w
        # smoothed pinball derivative: rho'(r) = alpha - (1-alpha) for r<0,
        # smoothed to a smooth ramp in |r| < eps
        psi = np.where(resid >= 0, alpha, alpha - 1.0)
        wi = np.abs(psi) / np.maximum(np.abs(resid), eps)
        wi = np.clip(wi, 1e-4, None)
        wx = xb * wi[:, None]
        gram = xb.T @ wx + ridge * np.eye(d + 1)
        # weighted normal equations, damped update
        w_new = np.linalg.solve(gram, xb.T @ (wi * y))
        w = 0.5 * w + 0.5 * w_new
    return w


def pinball_ridge_predict(w: FloatArray, x: FloatArray) -> FloatArray:
    """Predict quantiles with a fitted pinball-ridge coefficient vector."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    xb = np.hstack([a, np.ones((a.shape[0], 1))])
    if xb.shape[1] != w.size:
        raise ValueError("feature width mismatch")
    return xb @ w


def lag_features(scores: FloatArray, n_lags: int = 6) -> FloatArray:
    """Lagged-score design matrix: row t has [s_{t-1}, ..., s_{t-L}]."""
    s = np.asarray(scores, dtype=float).ravel()
    if s.size < n_lags + 2 or n_lags < 1:
        raise ValueError("need series longer than n_lags+2")
    return np.column_stack([s[n_lags - 1 - k : s.size - 1 - k] for k in range(n_lags)])


@dataclass(frozen=True)
class SPCIPath:
    """Walk-forward SPCI interval run."""

    lower: FloatArray
    upper: FloatArray
    qhat: FloatArray  # forecast residual quantile per step
    covered: FloatArray  # 0/1 realized coverage per step
    n_calib: int

    @property
    def coverage(self) -> float:
        return float(np.nanmean(self.covered))

    @property
    def mean_width(self) -> float:
        return float(np.nanmean(self.upper - self.lower))


def spci_walk_forward(
    y: FloatArray,
    yhat: FloatArray,
    alpha: float = 0.1,
    n_lags: int = 6,
    min_train: int = 60,
    retrain_every: int = 10,
    ridge: float = 1e-3,
    seed: int = 0,
) -> SPCIPath:
    """SPCI walk-forward intervals on a (y, point-forecast) stream.

    At each t >= min_train, refit the residual-quantile forecaster on the
    last ``min_train`` residuals' lag features (every ``retrain_every``
    steps; otherwise reuse), predict the 1-alpha quantile of the next
    conformity score, emit [yhat - qhat, yhat + qhat].
    """
    v = _check_series(y)
    p = _check_series(yhat)
    if v.size != p.size:
        raise ValueError("y and yhat must have equal length")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha in (0, 0.5)")
    if v.size < min_train + n_lags + 4:
        raise ValueError("series too short for walk-forward")
    if retrain_every < 1:
        raise ValueError("retrain_every >= 1")

    scores = np.abs(v - p)
    n = v.size
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    qhat = np.full(n, np.nan)
    cov = np.full(n, np.nan)
    _ = seed  # deterministic; kept for signature parity
    w = None
    for t in range(min_train, n):
        if w is None or (t - min_train) % retrain_every == 0:
            win = scores[max(0, t - min_train * 2) : t]
            xtr = lag_features(win, n_lags)
            ytr = win[n_lags:]
            if xtr.shape[0] >= 8:
                w = pinball_ridge_fit(xtr, ytr, alpha=1 - alpha, ridge=ridge)
        if w is None:
            continue
        # predict from the last n_lags observed scores
        feats = scores[t - n_lags : t][::-1]
        if feats.size < n_lags:
            continue
        q = float(pinball_ridge_predict(w, feats.reshape(1, -1))[0])
        q = max(q, 1e-6)
        lo[t], hi[t], qhat[t] = p[t] - q, p[t] + q, q
        cov[t] = float(lo[t] <= v[t] <= hi[t])
    valid = ~np.isnan(cov)
    return SPCIPath(
        lower=np.where(valid, lo, np.nan),
        upper=np.where(valid, hi, np.nan),
        qhat=qhat,
        covered=np.where(valid, cov, np.nan),
        n_calib=min_train,
    )


def synth_drift_series(
    n: int, shift_at: float = 0.5, shift_size: float = 2.0, seed: int = 0
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC series with a variance/mean shift — (y, naive yhat=lag-1)."""
    if n < 64:
        raise ValueError("n >= 64 required")
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    vol = np.where(t < n * shift_at, 1.0, 1.0 + shift_size)
    mean = np.where(t < n * shift_at, 0.0, shift_size * 0.3)
    y = mean + vol * rng.standard_normal(n)
    yhat = np.concatenate([[y[0]], y[:-1]])  # naive persistence forecast
    return y, yhat


def aci_comparison(
    y: FloatArray, yhat: FloatArray, alpha: float = 0.1, gamma: float = 0.05
) -> float:
    """Post-shift coverage error of the sibling ACI mechanism (comparator).

    Runs the same (y, point-forecast) stream through
    ``models.conformal.AdaptiveConformal`` with symmetric intervals
    [yhat - qhat, yhat + qhat], then reports |empirical - nominal| coverage
    on the post-shift tail.
    """
    v, p = _check_series(y), _check_series(yhat)
    n = v.size
    n_init = n // 4
    # ACI scores are CQR scores on symmetric intervals; start with a wide
    # interval so the initial score distribution is the raw residual scale
    base_lo, base_hi = p - 1.0, p + 1.0
    aci = AdaptiveConformal(alpha=alpha, gamma=gamma)
    aci.initialize(v[:n_init], base_lo[:n_init], base_hi[:n_init])
    path = aci.run(
        v[n_init:],
        base_lo[n_init:],
        base_hi[n_init:],
        dates=np.arange(n - n_init),
    )
    cov = np.asarray(path.covered, dtype=float)
    cov = cov[np.isfinite(cov)]
    post = cov[int(cov.size * 0.5) :]
    return float(abs(np.mean(post) - (1 - alpha)))


def bench_spci_conformal(seed: int = 20260131) -> dict[str, float]:
    """SYNTHETIC bench for SPCI residual-forecast conformal. Correctness only."""
    out: dict[str, float] = {}
    y, yhat = synth_drift_series(1200, shift_size=2.5, seed=seed)
    alpha = 0.1
    path = spci_walk_forward(y, yhat, alpha=alpha, min_train=80, seed=seed)
    n = y.size
    pre = path.covered[~np.isnan(path.covered)][: int(0.3 * n)]
    post = path.covered[~np.isnan(path.covered)][-int(0.3 * n) :]
    out["synthetic_spci_coverage_pre"] = float(pre.mean())
    out["synthetic_spci_coverage_post"] = float(post.mean())
    out["synthetic_spci_coverage_err_post"] = float(abs(post.mean() - (1 - alpha)))
    out["synthetic_spci_mean_width"] = path.mean_width
    # ACI comparator on the same stream
    out["synthetic_aci_coverage_err_post"] = aci_comparison(y, yhat, alpha=alpha)
    # re-cover time: steps after the shift until rolling-50 coverage >= 0.8
    shift_idx = int(n * 0.5)
    post_cov = path.covered[~np.isnan(path.covered)]
    post_cov = post_cov[shift_idx - path.n_calib :]
    rc = np.convolve(post_cov, np.ones(50) / 50, mode="valid")
    recover = int(np.argmax(rc >= 0.8)) if (rc >= 0.8).any() else -1
    out["synthetic_spci_recover_steps"] = float(recover)
    # determinism
    p2 = spci_walk_forward(y, yhat, alpha=alpha, min_train=80, seed=seed)
    out["synthetic_determinism"] = float(
        np.allclose(np.nan_to_num(path.qhat), np.nan_to_num(p2.qhat))
    )
    return out
