"""Dynamic-factor nowcasting in the Banbura-Modugno tradition.

A panel of ``N`` noisy indicators is driven by ``q`` latent factors:

``y_t = Λ f_t + ε_t,  ε_t ~ N(0, diag(r))``
``f_t = A f_{t-1} + η_t,  η_t ~ N(0, Q)``

Estimation is EM (Doz-Giannone-Reichlin): the E-step runs the Kalman
filter + RTS smoother on a *ragged* panel — each ``NaN`` observation is
masked out of that period's update, which is exactly how real
nowcasting handles publication lags — and the M-step is closed-form
least squares on smoothed moments.

The nowcast for series ``i`` at the ragged edge is
``Λ_i · E[f_T | observed]``, and :func:`news_decomp` decomposes the
revision between two vintages into Kalman-gain × surprise terms per
release — the Banbura-Modugno "news" accounting.

Functions
---------
- :func:`kalman_smooth` — filter + RTS smoother with missing-obs masks.
- :func:`em_dfm` — EM estimation of ``(Λ, A, Q, r)`` on a ragged panel.
- :func:`factor_nowcast` — fit + smoothed factor path + final nowcast.
- :func:`news_decomp` — per-release contribution to a nowcast revision.
- :func:`synth_panel` — synthetic factor panel with ground truth.
- :func:`bench_factor_nowcast` — SYNTHETIC telemetry blob.

References
----------
- Banbura & Modugno (2014). Maximum likelihood estimation of factor
  models on datasets with arbitrary pattern of missing data. *Journal of
  Applied Econometrics* 29(1), ECB WP 1189 (journal — not on arXiv;
  the arXiv id in circulation for it, 1208.4190, is a laser-physics
  paper).
- Giannone, Reichlin & Small (2008). Nowcasting: the real-time
  informational content of macroeconomic data. *JME* 55(4) (journal).
- Stock & Watson (2002). Macroeconomic forecasting using diffusion
  indexes. *JBES* 20(2) (journal).
- Doz, Giannone & Reichlin (2012). A quasi-maximum likelihood approach
  for large, approximate dynamic factor models. *ReStat* 94(4)
  (journal).

Honesty
-------
All reported numbers are SYNTHETIC recovery checks on seeded factor
panels — they validate the filter/smoother/EM machinery, never macro
data, and are not forecasting claims. EM is quasi-ML on a misspecified
short panel and reports convergence status rather than hiding it.

Composition notes
-----------------
- ``models/nowcasting.py``: MIDAS/bridge-equation nowcasting — this
  module is the state-space/dynamic-factor complement; composition is by
  shared target-series convention, not shared code.
- ``metrics/kalman*``-adjacent helpers stay local: the missing-obs mask
  here is release-calendar aware (per-``t`` row masks), not the
  whole-series dropout used elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_panel(y: FloatArray) -> FloatArray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 8 or arr.shape[1] < 1:
        raise ValueError("y must be a (T>=8, N) panel")
    if np.isnan(arr).all():
        raise ValueError("y is entirely NaN")
    return arr


def _state_mats(
    load: FloatArray, ar_coef: FloatArray, q: FloatArray, r: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    lam = np.asarray(load, dtype=np.float64)
    a = np.asarray(ar_coef, dtype=np.float64)
    qq = np.asarray(q, dtype=np.float64)
    rr = np.asarray(r, dtype=np.float64).ravel()
    nf, kf = lam.shape
    if a.shape != (kf, kf) or qq.shape != (kf, kf) or rr.shape != (nf,):
        raise ValueError("inconsistent (load, ar_coef, q, r) shapes")
    if (rr <= 0).any() or not np.isfinite(lam).all() or not np.isfinite(a).all():
        raise ValueError("bad load/ar/r values")
    return lam, a, qq, rr


def kalman_smooth(
    y: FloatArray,
    load: FloatArray,
    ar_coef: FloatArray,
    q: FloatArray,
    r: FloatArray,
    x0: FloatArray,
    p0: FloatArray,
) -> dict[str, FloatArray | float]:
    """Kalman filter + RTS smoother with per-period NaN masking.

    Returns a dict with smoothed means ``f_s`` (T, q), smoothed
    covariances ``p_s`` (T, q, q), lag-one smoothed cross-covariances
    ``p_lag`` (T, q, q) (``Cov[f_t, f_{t-1} | all data]``), filtered
    means/covs, and the ``loglik``.
    """
    arr = _check_panel(y)
    lam, a, qq, rr = _state_mats(load, ar_coef, q, r)
    t_n, n_n = arr.shape
    kf = lam.shape[1]
    x = np.asarray(x0, dtype=np.float64).ravel()
    p = np.asarray(p0, dtype=np.float64)
    if x.shape != (kf,) or p.shape != (kf, kf):
        raise ValueError("bad x0/p0 shapes")

    f_filt = np.zeros((t_n, kf))
    p_filt = np.zeros((t_n, kf, kf))
    f_pred = np.zeros((t_n, kf))
    p_pred = np.zeros((t_n, kf, kf))
    loglik = 0.0
    eye = np.eye(kf)

    for t in range(t_n):
        if t > 0:
            x = a @ x
            p = a @ p @ a.T + qq
        f_pred[t] = x
        p_pred[t] = p
        obs = np.isfinite(arr[t])
        if obs.any():
            h = lam[obs]  # (m, kf)
            yv = arr[t, obs]
            s = h @ p @ h.T + np.diag(rr[obs])
            s_inv = np.linalg.solve(s, np.eye(s.shape[0]))
            innov = yv - h @ x
            gain = p @ h.T @ s_inv  # (kf, m)
            x = x + gain @ innov
            p = (eye - gain @ h) @ p
            sign, logdet = np.linalg.slogdet(s)
            if sign <= 0:
                raise ValueError("non-PD innovation covariance")
            loglik += float(-0.5 * (innov @ s_inv @ innov + logdet + yv.size * np.log(2 * np.pi)))
        f_filt[t] = x
        p_filt[t] = p

    # RTS smoother
    f_s = f_filt.copy()
    p_s = p_filt.copy()
    p_lag = np.zeros((t_n, kf, kf))
    for t in range(t_n - 2, -1, -1):
        j_gain = p_filt[t] @ a.T @ np.linalg.inv(p_pred[t + 1])
        f_s[t] = f_filt[t] + j_gain @ (f_s[t + 1] - f_pred[t + 1])
        p_s[t] = p_filt[t] + j_gain @ (p_s[t + 1] - p_pred[t + 1]) @ j_gain.T
        p_lag[t + 1] = p_s[t + 1] @ j_gain.T
    return {
        "f_s": f_s,
        "p_s": p_s,
        "p_lag": p_lag,
        "f_filt": f_filt,
        "p_filt": p_filt,
        "loglik": loglik,
    }


def _expected_moments(
    f_s: FloatArray, p_s: FloatArray, p_lag: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """E[f_t], E[f_t f_t'], E[f_t f_{t-1}'] under the smoothed posterior."""
    t_n, kf = f_s.shape
    e_ff = p_s + np.einsum("ti,tj->tij", f_s, f_s)
    e_lag = np.zeros((t_n, kf, kf))
    e_lag[1:] = p_lag[1:] + np.einsum("ti,tj->tij", f_s[1:], f_s[:-1])
    return f_s, e_ff, e_lag


def em_dfm(
    y: FloatArray,
    n_factors: int = 1,
    n_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict[str, FloatArray | int | bool]:
    """EM fit of the DFM. Raises ``ValueError`` if not converged."""
    arr = _check_panel(y)
    t_n, n_n = arr.shape
    if not 1 <= n_factors <= min(4, n_n):
        raise ValueError("n_factors out of range")
    rng = np.random.default_rng(seed)
    lam = rng.standard_normal((n_n, n_factors)) * 0.5
    a = np.eye(n_factors) * 0.8
    qq = np.eye(n_factors) * 0.2
    rr = np.nanvar(arr, axis=0) * 0.5 + 1e-3
    x0 = np.zeros(n_factors)
    p0 = np.eye(n_factors)

    prev_ll = -np.inf
    converged = False
    it_done = 0
    for it in range(1, n_iter + 1):
        ks = kalman_smooth(arr, lam, a, qq, rr, x0, p0)
        f_s = np.asarray(ks["f_s"], dtype=np.float64)
        p_s = np.asarray(ks["p_s"], dtype=np.float64)
        p_lag = np.asarray(ks["p_lag"], dtype=np.float64)
        e_ff = p_s + np.einsum("ti,tj->tij", f_s, f_s)
        e_lag = np.zeros_like(e_ff)
        e_lag[1:] = p_lag[1:] + np.einsum("ti,tj->tij", f_s[1:], f_s[:-1])
        # M-step: loadings per series on observed cells
        for i in range(n_n):
            mask = np.isfinite(arr[:, i])
            if not mask.any():
                continue
            denom = e_ff[mask].sum(axis=0)
            numer = (arr[mask, i, None] * f_s[mask]).sum(axis=0)
            lam[i] = np.linalg.solve(denom + 1e-10 * np.eye(n_factors), numer)
        denom_a = e_lag[1:].sum(axis=0)
        denom_prev = e_ff[:-1].sum(axis=0)
        a = np.linalg.solve(denom_prev + 1e-10 * np.eye(n_factors), denom_a.T).T
        qq = (e_ff[1:].sum(axis=0) - a @ e_lag[1:].sum(axis=0).T) / (t_n - 1)
        qq = (qq + qq.T) / 2 + 1e-10 * np.eye(n_factors)
        for i in range(n_n):
            mask = np.isfinite(arr[:, i])
            if not mask.any():
                continue
            li = lam[i]
            # E[(y - Λf)^2] = y² - 2 y ΛE[f] + Λ E[ff'] Λ'
            quad = np.einsum("tij,j,i->t", e_ff[mask], li, li)
            resid2 = arr[mask, i] ** 2 - 2.0 * arr[mask, i] * (f_s[mask] @ li) + quad
            rr[i] = max(float(resid2.mean()), 1e-8)
        ll = float(ks["loglik"])
        it_done = it
        if ll - prev_ll < tol * max(1.0, abs(prev_ll)):
            converged = True
            break
        prev_ll = ll
    if not converged:
        raise ValueError(f"EM did not converge in {n_iter} iterations")
    # identification: unit-variance factors, positive mean loading
    scale = np.sqrt(np.diag(qq)).clip(1e-12)
    a = a * (scale[None, :] / scale[:, None])
    lam = lam * scale[None, :]
    qq = np.eye(n_factors)
    flip = np.sign(lam.mean(axis=0))
    flip[flip == 0] = 1.0
    lam = lam * flip[None, :]
    a = a * (flip[None, :] * flip[:, None])
    return {
        "load": lam,
        "ar_coef": a,
        "q": qq,
        "r": rr,
        "n_iter": it_done,
        "converged": True,
    }


@dataclass(frozen=True)
class FactorNowcast:
    """Fitted DFM nowcast result."""

    factor: FloatArray  # smoothed factor path (T, q)
    load: FloatArray  # (N, q)
    nowcast: FloatArray  # (N,) model-implied value of y at T-1
    r2: FloatArray  # per-series in-sample fit share
    n_iter: int


def factor_nowcast(
    y: FloatArray,
    n_factors: int = 1,
    n_iter: int = 200,
    seed: int = 0,
) -> FactorNowcast:
    """Fit the DFM on a ragged panel and nowcast the final period."""
    arr = _check_panel(y)
    fit = em_dfm(arr, n_factors=n_factors, n_iter=n_iter, seed=seed)
    lam = np.asarray(fit["load"], dtype=np.float64)
    a = np.asarray(fit["ar_coef"], dtype=np.float64)
    qq = np.asarray(fit["q"], dtype=np.float64)
    rr = np.asarray(fit["r"], dtype=np.float64)
    ks = kalman_smooth(arr, lam, a, qq, rr, np.zeros(lam.shape[1]), np.eye(lam.shape[1]))
    f_s = np.asarray(ks["f_s"], dtype=np.float64)
    nowcast = lam @ f_s[-1]
    r2 = np.zeros(arr.shape[1])
    for i in range(arr.shape[1]):
        mask = np.isfinite(arr[:, i])
        if mask.sum() < 4:
            r2[i] = 0.0
            continue
        resid = arr[mask, i] - f_s[mask] @ lam[i]
        var_y = float(arr[mask, i].var())
        r2[i] = 1.0 - float((resid**2).mean()) / var_y if var_y > 0 else 0.0
    return FactorNowcast(
        factor=f_s,
        load=lam,
        nowcast=nowcast,
        r2=r2,
        n_iter=int(fit["n_iter"]),
    )


def news_decomp(
    y_old: FloatArray,
    y_new: FloatArray,
    fit: dict[str, FloatArray | int | bool],
    target: int = 0,
) -> FloatArray:
    """Per-release contribution to the target-series nowcast revision.

    For each cell that is NaN in ``y_old`` and observed in ``y_new`` at
    time ``t``, the contribution is the regression coefficient of the
    target nowcast on that release's innovation:

    ``w_j = Λ_target · A^{T-1-t} · P_t^- · h_i' · S_j^{-1}``,
    ``contrib_j = w_j · (y_j - ŷ_j)``.

    Returns an array shaped like ``y`` with the contribution of each
    release in its cell (0 elsewhere).
    """
    old = _check_panel(y_old)
    new = _check_panel(y_new)
    if old.shape != new.shape:
        raise ValueError("y_old and y_new shapes differ")
    lam = np.asarray(fit["load"], dtype=np.float64)
    a = np.asarray(fit["ar_coef"], dtype=np.float64)
    qq = np.asarray(fit["q"], dtype=np.float64)
    rr = np.asarray(fit["r"], dtype=np.float64)
    t_n, _ = old.shape
    kf = lam.shape[1]
    lam_t = lam[target]

    # refilter the old panel, recording per-t predictive moments
    x = np.zeros(kf)
    p = np.eye(kf)
    contrib = np.zeros_like(old)
    p_pred_store = np.zeros((t_n, kf, kf))
    f_pred_store = np.zeros((t_n, kf))
    for t in range(t_n):
        if t > 0:
            x = a @ x
            p = a @ p @ a.T + qq
        p_pred_store[t] = p
        f_pred_store[t] = x
        obs = np.isfinite(old[t])
        if obs.any():
            h = lam[obs]
            s = h @ p @ h.T + np.diag(rr[obs])
            x = x + p @ h.T @ np.linalg.solve(s, old[t, obs] - h @ x)
            p = (np.eye(kf) - p @ h.T @ np.linalg.solve(s, h)) @ p

    releases = np.argwhere(~np.isfinite(old) & np.isfinite(new))
    for t, i in releases:
        h_i = lam[i]
        p_t = p_pred_store[t]
        pred_mean = float(h_i @ f_pred_store[t])
        s_j = float(h_i @ p_t @ h_i + rr[i])
        if s_j <= 0:
            continue
        news = float(new[t, i] - pred_mean)
        horizon = t_n - 1 - t
        w = lam_t @ np.linalg.matrix_power(a, horizon) @ p_t @ h_i / s_j
        contrib[t, i] = w * news
    return contrib


def synth_panel(
    n_obs: int,
    n_series: int,
    n_factors: int = 1,
    rho_factor: float = 0.85,
    missing_share: float = 0.15,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Synthetic DFM panel; returns ``(y_with_nans, true_factors, load)``."""
    if n_obs < 30 or n_series < 2 or not 1 <= n_factors <= 3:
        raise ValueError("bad panel shape")
    if not 0 <= rho_factor < 0.99 or not 0 <= missing_share < 0.9:
        raise ValueError("bad rho/missing_share")
    rng = np.random.default_rng(seed)
    f = np.zeros((n_obs, n_factors))
    eta = rng.standard_normal((n_obs, n_factors)) * np.sqrt(1 - rho_factor**2)
    for t in range(1, n_obs):
        f[t] = rho_factor * f[t - 1] + eta[t]
    lam = rng.uniform(0.4, 1.2, (n_series, n_factors))
    lam[:: n_factors + 1] = np.abs(lam[:: n_factors + 1])  # anchor signs
    y = f @ lam.T + rng.standard_normal((n_obs, n_series)) * 0.5
    mask = rng.random((n_obs, n_series)) < missing_share
    # never mask the interior of the target series entirely; keep ragged edge
    mask[:, 0] = False
    tail = np.arange(n_obs)[:, None] >= n_obs - 8
    mask = mask | (tail & (rng.random((n_obs, n_series)) < 0.6))
    mask[:, 0] = False
    y = np.where(mask, np.nan, y)
    return y, f, lam


def bench_factor_nowcast(seed: int = 0) -> dict[str, float]:
    """SYNTHETIC DFM blob — factor-recovery and nowcast diagnostics."""
    y, f_true, _lam = synth_panel(160, 6, 1, 0.85, 0.15, seed)
    fit = factor_nowcast(y, n_factors=1, n_iter=150, seed=seed)
    corr = float(abs(np.corrcoef(fit.factor[:, 0], f_true[:, 0])[0, 1]))

    # masked-cell nowcast error vs naive previous-value carry
    obs_idx = np.flatnonzero(np.isfinite(y[:, 0]))
    hold = obs_idx[-10:]
    y_cv = y.copy()
    y_cv[hold, 0] = np.nan
    fit_cv = em_dfm(y_cv, n_factors=1, n_iter=150, seed=seed + 1)
    ks = kalman_smooth(
        y_cv,
        np.asarray(fit_cv["load"]),
        np.asarray(fit_cv["ar_coef"]),
        np.asarray(fit_cv["q"]),
        np.asarray(fit_cv["r"]),
        np.zeros(1),
        np.eye(1),
    )
    f_s = np.asarray(ks["f_s"], dtype=np.float64)
    lam0 = float(np.asarray(fit_cv["load"])[0, 0])
    dfm_err = float(np.sqrt(np.mean((lam0 * f_s[hold, 0] - y[hold, 0]) ** 2)))
    prev = y[hold - 1, 0]
    naive_err = float(np.sqrt(np.mean((prev - y[hold, 0]) ** 2)))
    rmse_ratio = dfm_err / naive_err if naive_err > 0 else 1.0

    # news decomposition: reveal a seeded subset of masked cells
    rng = np.random.default_rng(seed + 2)
    reveal = np.argwhere(np.isnan(y_cv) & np.isfinite(y))
    pick = reveal[rng.choice(len(reveal), size=min(8, len(reveal)), replace=False)]
    y_new = y_cv.copy()
    for t, i in pick:
        y_new[t, i] = y[t, i]
    contrib = news_decomp(y_cv, y_new, fit_cv, target=0)
    revision = float(contrib.sum())

    blob_det = factor_nowcast(y, n_factors=1, n_iter=150, seed=seed)
    determinism = float(np.allclose(fit.factor, blob_det.factor))

    return {
        "synthetic_factor_corr": corr,
        "synthetic_nowcast_err_ratio": rmse_ratio,
        "synthetic_em_converged": float(fit_cv["converged"]),
        "synthetic_em_iters": float(fit_cv["n_iter"]),
        "synthetic_mean_r2": float(fit.r2.mean()),
        "synthetic_masked_rmse_ratio": rmse_ratio,
        "synthetic_news_revision_abs": abs(revision),
        "synthetic_news_max_contrib": float(np.abs(contrib).max()),
        "synthetic_load_sign_consistency": float(
            (np.sign(fit.load[:, 0]) == np.sign(fit.load[:, 0].mean())).mean()
        ),
        "synthetic_determinism": determinism,
    }
