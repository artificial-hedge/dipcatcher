"""Instrumental-variable quantile regression (Chernozhukov–Hansen).

IVQR estimates structural quantile treatment effects when a regressor is
endogenous: ordinary quantile regression is inconsistent because the
regressor correlates with the rank variable. The Chernozhukov & Hansen
(2005, 2008) estimator inverts a quantile regression of residuals on the
instruments: for a candidate effect ``β``, project ``y − x·β`` on
``[Z, X_ex]`` at quantile ``τ`` and take the instrument block ``γ(β)``;
``β̂`` minimizes the weighted norm of ``γ(β)``. Inference uses an
Anderson–Rubin-type statistic that stays valid under weak instruments.

Honesty
-------
All bench outputs are ``synthetic_*`` correctness diagnostics on a seeded
endogenous DGP — IVQR bias vs naive-QR bias, AR size/power, first-stage F.
They verify estimator mechanics, never market evidence. No
Sharpe/Sortino/Calmar/P&L/NAV ever.

References
----------
- Chernozhukov, V. & Hansen, C. (2005). An IV model of quantile treatment
  effects. *Econometrica* 73(1):245–261.
- Chernozhukov, V. & Hansen, C. (2008). Instrumental variable quantile
  regression: a robust inference approach. *Journal of Econometrics*
  142(1):379–398. (AR-type robust inference variant used here.)
- Anderson, T.W. & Rubin, H. (1949). Estimation of the parameters of a
  single equation in a complete system of stochastic equations. *Annals of
  Mathematical Statistics* 20(1):46–63.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
  (QR linear-program formulation and sandwich covariance.)

Composition notes
-----------------
- ``metrics.regression``: ordinary QR/expectile machinery — this module is
  the endogeneity-corrected sibling.
- ``metrics.inference``: generic CI helpers; ``ivqr_ci`` here inverts the
  AR statistic over a grid instead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]
FORBIDDEN_KEYS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _check_vector(v: object, n: int, name: str) -> FloatArray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 1:
        raise ValueError(f"{name} must be (n, k) with n={n} rows")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _design(z_inst: FloatArray, x_exog: FloatArray | None, n: int) -> FloatArray:
    parts = [np.ones((n, 1)), z_inst]
    if x_exog is not None and x_exog.shape[1] > 0:
        parts.append(x_exog)
    return np.hstack(parts)


def _qr_coef(x_mat: FloatArray, y: FloatArray, tau: float) -> FloatArray:
    """Exact τ-quantile regression via the standard LP formulation."""
    n, k = x_mat.shape
    # vars: [beta (k), u+ (n), u- (n)]
    c = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1.0 - tau)])
    a_eq = np.hstack([x_mat, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise ValueError(f"quantile LP failed: {res.message}")
    return np.asarray(res.x[:k], dtype=np.float64)


def _resid_density(resid: FloatArray) -> float:
    """Kernel density of residuals at 0 (Silverman bandwidth)."""
    n = resid.size
    s = float(np.std(resid))
    iqr = float(np.subtract(*np.percentile(resid, [75, 25])))
    h = 0.9 * min(s, iqr / 1.349 if iqr > 0 else s) * n ** (-0.2)
    if not np.isfinite(h) or h <= 0:
        h = max(s * n ** (-0.2), 1e-8)
    z = resid / h
    return float(np.mean(norm.pdf(z)) / h)


def ivqr_objective(
    y: object,
    x_endog: object,
    z_inst: object,
    x_exog: object | None,
    tau: float,
    beta: float,
) -> float:
    """Chernozhukov–Hansen criterion at a candidate endogenous effect ``β``.

    Runs QR of ``y − x·β`` on ``[1, Z, X_ex]`` at ``τ`` and returns
    ``γ̂(β)' (Z'Z/n) γ̂(β)`` — the weighted norm of the instrument block;
    the IVQR estimate is its minimizer. Multiple endogenous regressors are
    out of scope (one ``x_endog`` column enforced).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1 or y_arr.size < 20 or not np.isfinite(y_arr).all():
        raise ValueError("y must be a finite vector with n >= 20")
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")
    n = y_arr.size
    x_e = _check_vector(x_endog, n, "x_endog")
    if x_e.shape[1] != 1:
        raise ValueError("x_endog must have exactly one column")
    z = _check_vector(z_inst, n, "z_inst")
    x_ex = None if x_exog is None else _check_vector(x_exog, n, "x_exog")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")

    resid = y_arr - x_e[:, 0] * beta
    w = _design(z, x_ex, n)
    coef = _qr_coef(w, resid, tau)
    n_z = z.shape[1]
    gamma = coef[1 : 1 + n_z]
    weight = (z.T @ z) / n
    return float(gamma @ weight @ gamma)


def ivqr_fit(
    y: object,
    x_endog: object,
    z_inst: object,
    x_exog: object | None,
    tau: float,
    grid: object | None = None,
) -> dict[str, float | FloatArray]:
    """Grid-search IVQR estimate of the single endogenous effect.

    The documented small-grid estimator: evaluate ``ivqr_objective`` over a
    ``grid`` (default: a data-driven symmetric grid around the naive QR
    coefficient), take the argmin, then run the final QR of ``y − x·β̂`` on
    ``[1, X_ex]`` for the remaining coefficients.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1 or y_arr.size < 20:
        raise ValueError("y must be a vector with n >= 20")
    n = y_arr.size
    x_e = _check_vector(x_endog, n, "x_endog")
    z = _check_vector(z_inst, n, "z_inst")
    x_ex = None if x_exog is None else _check_vector(x_exog, n, "x_exog")
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")

    naive_w = _design(np.zeros((n, 1))[:, :0], x_e if x_ex is None else np.hstack([x_e, x_ex]), n)
    naive_coef = _qr_coef(naive_w, y_arr, tau)
    naive_beta = float(naive_coef[1])
    if grid is None:
        span = 2.0 + abs(naive_beta)
        g = np.linspace(naive_beta - span, naive_beta + span, 41)
    else:
        g = np.asarray(grid, dtype=np.float64)
        if g.ndim != 1 or g.size < 3 or not np.isfinite(g).all():
            raise ValueError("grid must be a finite vector with >= 3 points")

    objs = np.array([ivqr_objective(y_arr, x_e, z, x_ex, tau, float(b)) for b in g])
    beta_hat = float(g[int(np.argmin(objs))])

    final_w = _design(np.zeros((n, 1))[:, :0], x_ex, n)
    final_coef = _qr_coef(final_w, y_arr - x_e[:, 0] * beta_hat, tau)
    return {
        "beta": beta_hat,
        "coef": np.concatenate([[beta_hat], final_coef]),
        "naive_beta": naive_beta,
        "objective_min": float(objs.min()),
        "grid": g,
        "objective": objs,
    }


def anderson_rubin_ivq(
    y: object,
    x_endog: object,
    z_inst: object,
    x_exog: object | None,
    tau: float,
    beta0: float,
) -> dict[str, float | FloatArray]:
    """Weak-instrument-robust AR statistic for ``H0: β = β0`` at quantile τ.

    QR of ``y − x·β0`` on ``[1, Z, X_ex]``; the instrument block ``γ̂`` is
    tested with the standard QR sandwich covariance
    ``τ(1−τ)/f̂² · (W'W)⁻¹`` restricted to the Z block — the Chernozhukov &
    Hansen (2008) projection variant. ``stat ~ χ²(n_z)`` under H0
    regardless of instrument strength.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1 or y_arr.size < 20 or not np.isfinite(y_arr).all():
        raise ValueError("y must be a finite vector with n >= 20")
    n = y_arr.size
    x_e = _check_vector(x_endog, n, "x_endog")
    z = _check_vector(z_inst, n, "z_inst")
    x_ex = None if x_exog is None else _check_vector(x_exog, n, "x_exog")
    if not 0 < tau < 1 or not np.isfinite(beta0):
        raise ValueError("tau in (0,1) and finite beta0 required")

    resid = y_arr - x_e[:, 0] * beta0
    w = _design(z, x_ex, n)
    coef = _qr_coef(w, resid, tau)
    n_z = z.shape[1]
    gamma = coef[1 : 1 + n_z]

    qr_resid = resid - w @ coef
    f_hat = max(_resid_density(qr_resid), 1e-8)
    xtx_inv = np.linalg.pinv(w.T @ w)
    vcov = (tau * (1.0 - tau) / f_hat**2) * xtx_inv
    s_zz = vcov[1 : 1 + n_z, 1 : 1 + n_z]
    stat = float(gamma @ np.linalg.pinv(s_zz) @ gamma)
    p_val = float(chi2.sf(stat, n_z))
    return {
        "stat": stat,
        "p_value": p_val,
        "df": float(n_z),
        "gamma": gamma,
    }


def ivqr_ci(
    y: object,
    x_endog: object,
    z_inst: object,
    x_exog: object | None,
    tau: float,
    level: float = 0.95,
    grid: object | None = None,
) -> dict[str, float | FloatArray]:
    """Confidence interval for the endogenous effect by AR inversion."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    fit = ivqr_fit(y, x_endog, z_inst, x_exog, tau, grid)
    g = np.asarray(fit["grid"])
    pvals = np.array(
        [anderson_rubin_ivq(y, x_endog, z_inst, x_exog, tau, float(b))["p_value"] for b in g]
    )
    inside = g[pvals > (1.0 - level)]
    lo = float(inside.min()) if inside.size else float("nan")
    hi = float(inside.max()) if inside.size else float("nan")
    return {
        "beta": float(fit["beta"]),
        "lo": lo,
        "hi": hi,
        "grid": g,
        "p_value": pvals,
    }


def first_stage_f(
    x_endog: object,
    z_inst: object,
    x_exog: object | None = None,
) -> float:
    """First-stage F statistic for instrument relevance (OLS on Z | X_ex)."""
    x_arr = np.asarray(x_endog, dtype=np.float64).ravel()
    if x_arr.size < 20 or not np.isfinite(x_arr).all():
        raise ValueError("x_endog must be a finite vector with n >= 20")
    n = x_arr.size
    z = _check_vector(z_inst, n, "z_inst")
    x_ex = None if x_exog is None else _check_vector(x_exog, n, "x_exog")

    w_full = _design(z, x_ex, n)
    beta_full = np.linalg.lstsq(w_full, x_arr, rcond=None)[0]
    rss_full = float(((x_arr - w_full @ beta_full) ** 2).sum())
    w_r = _design(np.zeros((n, 1))[:, :0], x_ex, n)
    beta_r = np.linalg.lstsq(w_r, x_arr, rcond=None)[0]
    rss_r = float(((x_arr - w_r @ beta_r) ** 2).sum())
    q = z.shape[1]
    denom = rss_full / max(n - w_full.shape[1], 1)
    if denom <= 0:
        return 0.0
    return float(((rss_r - rss_full) / q) / denom)


def synth_iv_data(
    n: int,
    beta_true: float,
    inst_strength: float,
    endog_corr: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Endogenous synthetic DGP: ``y = β·x + u``, ``x = π·z + v``, corr(u,v)=ρ.

    ``inst_strength`` is π (first-stage loading). ``endog_corr`` ρ drives
    the naive-QR bias. Returns y, x_endog (n,1), z_inst (n,1), x_exog (n,1).
    """
    if n < 50 or not np.isfinite(beta_true):
        raise ValueError("n >= 50 and finite beta_true required")
    if not np.isfinite(inst_strength) or inst_strength <= 0:
        raise ValueError("inst_strength must be positive")
    if not np.isfinite(endog_corr) or not -1 < endog_corr < 1:
        raise ValueError("endog_corr must be in (-1, 1)")

    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    ex = rng.normal(0.0, 1.0, n)
    cov = np.array([[1.0, endog_corr], [endog_corr, 1.0]])
    uv = rng.multivariate_normal(np.zeros(2), cov, size=n)
    u, v = uv[:, 0], uv[:, 1]
    x = inst_strength * z + 0.3 * ex + v
    y = beta_true * x + 0.5 * ex + u
    return {
        "y": y,
        "x_endog": x[:, None],
        "z_inst": z[:, None],
        "x_exog": ex[:, None],
    }


def bench_instrumental_quantile(seed: int = 0) -> dict[str, float]:
    """SYNTHETIC bench for IVQR: bias, AR size/power, first-stage F."""
    n = 300
    beta_true = 1.0
    dgp = synth_iv_data(n, beta_true, inst_strength=1.0, endog_corr=0.6, seed=seed)
    y, x_e, z_i, x_ex = dgp["y"], dgp["x_endog"], dgp["z_inst"], dgp["x_exog"]

    grid = np.linspace(-1.5, 3.5, 41)
    fit_med = ivqr_fit(y, x_e, z_i, x_ex, 0.5, grid)
    fit_tail = ivqr_fit(y, x_e, z_i, x_ex, 0.9, grid)
    naive_med = float(fit_med["naive_beta"])
    naive_tail = float(fit_tail["naive_beta"])

    ci = ivqr_ci(y, x_e, z_i, x_ex, 0.5, level=0.95, grid=grid)

    # AR size: rejection rate under the true null over a rep panel
    reps = 30
    rng = np.random.default_rng(seed + 77)
    rej_true = 0
    rej_false = 0
    for _ in range(reps):
        d = synth_iv_data(200, 1.0, 0.8, 0.6, seed=int(rng.integers(10**6)))
        ar_t = anderson_rubin_ivq(d["y"], d["x_endog"], d["z_inst"], d["x_exog"], 0.5, 1.0)
        ar_f = anderson_rubin_ivq(d["y"], d["x_endog"], d["z_inst"], d["x_exog"], 0.5, 3.0)
        rej_true += int(ar_t["p_value"] < 0.05)
        rej_false += int(ar_f["p_value"] < 0.05)
    ar_size = rej_true / reps
    ar_power = rej_false / reps

    f_stat = first_stage_f(x_e, z_i, x_ex)
    weak = synth_iv_data(200, 1.0, inst_strength=0.08, seed=seed)
    f_weak = first_stage_f(weak["x_endog"], weak["z_inst"], weak["x_exog"])

    det = ivqr_fit(y, x_e, z_i, x_ex, 0.5, grid)
    determinism = float(np.asarray(det["coef"]).tolist() == np.asarray(fit_med["coef"]).tolist())

    iv_bias_m = float(fit_med["beta"]) - beta_true
    iv_bias_t = float(fit_tail["beta"]) - beta_true
    naive_bias_m = naive_med - beta_true
    naive_bias_t = naive_tail - beta_true

    covers = float(
        np.isfinite(ci["lo"]) and np.isfinite(ci["hi"]) and ci["lo"] <= beta_true <= ci["hi"]
    )

    blob: dict[str, float] = {
        "synthetic_ivqr_bias_median": iv_bias_m,
        "synthetic_ivqr_bias_tail": iv_bias_t,
        "synthetic_naive_qr_bias_median": naive_bias_m,
        "synthetic_naive_qr_bias_tail": naive_bias_t,
        "synthetic_bias_improvement_median": abs(naive_bias_m) - abs(iv_bias_m),
        "synthetic_ar_size": ar_size,
        "synthetic_ar_power": ar_power,
        "synthetic_first_stage_f": f_stat,
        "synthetic_weak_first_stage_f": f_weak,
        "synthetic_weak_iv_flag": float(f_weak < 10.0),
        "synthetic_ci_covers_truth": covers,
        "synthetic_ci_width": float(ci["hi"] - ci["lo"]),
        "synthetic_determinism": determinism,
    }
    for k in blob:
        if FORBIDDEN_KEYS.intersection(k.split("_")):
            raise ValueError(f"forbidden bench key {k!r}")
    return blob
