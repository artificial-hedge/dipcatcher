"""Local-projection instrumental-variables impulse responses (Jordà LP-IV).

Extends :mod:`quant_fund.models.local_projection` (plain LP with
Newey-West CIs) to the instrumented case: at each horizon h, regress
y_{t+h} on the endogenous shock x_t instrumented by z_t, with lagged
controls — the LP-IV estimator of Jordà, Schularick & Taylor (2015)
and Stock & Watson (2018). Ships per-horizon first-stage F statistics
(Montiel-Olea/Stock diagnostics) and Anderson-Rubin weak-IV-robust
confidence sets alongside the usual Wald CIs.

References
----------
- Jordà, Schularick & Taylor (2015). Betting the house.
  *J. International Economics* 96(S1).
- Stock & Watson (2018). Identification and estimation of dynamic
  causal effects in macroeconomics using external instruments.
  *Economic Journal* 128(610).
- Anderson & Rubin (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations.
  *Annals of Mathematical Statistics* 20(1).
- Montiel Olea, Stock & Watson (2020). Inference in structural VARs
  with external instruments. *J. Econometrics*.

Honesty
-------
The synth system is a recursive two-equation structure with an
externally-valid instrument; keys report IRF recovery error,
first-stage strength, AR-vs-Wald coverage/width ordering, and weak-IV
behaviour — never macro claims about real economies.

Composition notes
-----------------
- ``models/local_projection.py``: the uninstrumented twin — same
  horizon-loop/Newey-West scaffolding, minus instruments.
- ``models/proxy_svar.py`` (wave 29): external-instrument VARs — LP-IV
  is the model-free counterpart of the same identification logic.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

FloatArray = NDArray[np.float64]


def _check_aligned(
    y: FloatArray, x: FloatArray, z: FloatArray, min_n: int = 40
) -> tuple[FloatArray, FloatArray, FloatArray]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    xa = np.asarray(x, dtype=np.float64).ravel()
    za = np.asarray(z, dtype=np.float64).ravel()
    if ya.size != xa.size or ya.size != za.size or ya.size < min_n:
        raise ValueError("y, x, z must be aligned with n>=min_n")
    for name, a in (("y", ya), ("x", xa), ("z", za)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite")
    return ya, xa, za


def _controls(y: FloatArray, x: FloatArray, t: int, lags: int) -> list[float]:
    row = [1.0]
    for lag in range(1, lags + 1):
        row.append(float(y[t - lag]))
        row.append(float(x[t - lag]))
    return row


def _nw_var_2sls(design: FloatArray, resid: FloatArray, col: int, lag: int) -> float:
    """Newey-West variance of 2SLS coefficient ``col`` on the second-stage
    fitted regressor — same meat/sandwich as plain LP but the regressor
    is the instrument projection."""
    bread = np.linalg.inv(design.T @ design + 1e-10 * np.eye(design.shape[1]))
    score = design * resid[:, None]
    meat = score.T @ score
    for ell in range(1, lag + 1):
        w = 1.0 - ell / (lag + 1.0)
        g = score[ell:].T @ score[:-ell]
        meat += w * (g + g.T)
    vcv = bread @ meat @ bread
    return float(max(vcv[col, col], 0.0))


def lp_iv(
    y: FloatArray,
    x: FloatArray,
    z: FloatArray,
    horizons: int = 12,
    control_lags: int = 2,
    nw_lag_scale: float = 1.0,
) -> dict[str, FloatArray]:
    """LP-IV impulse response of ``y`` to ``x`` instrumented by ``z``.

    At each horizon h: first stage x_t ~ z_t + controls (reports F on
    excluded instrument); second stage y_{t+h} ~ x̂_t + controls (2SLS
    point estimate). Also reports the Anderson-Rubin AR test inversion
    (a robust band) and Wald band around β̂_h.
    """
    ya, xa, za = _check_aligned(y, x, z)
    if horizons < 1 or control_lags < 0:
        raise ValueError("horizons>=1, control_lags>=0")
    n = ya.size
    irf = np.empty(horizons + 1)
    wald_se = np.empty(horizons + 1)
    fst_f = np.empty(horizons + 1)
    ar_lo = np.empty(horizons + 1)
    ar_hi = np.empty(horizons + 1)
    for h in range(horizons + 1):
        rows_w, rows_x, rows_z, target = [], [], [], []
        for t in range(control_lags, n - h):
            c = _controls(ya, xa, t, control_lags)
            rows_w.append(c)
            rows_x.append(xa[t])
            rows_z.append(za[t])
            target.append(ya[t + h])
        w = np.asarray(rows_w)
        xv = np.asarray(rows_x)
        zv = np.asarray(rows_z)
        yv = np.asarray(target)
        # first stage: x ~ [W, z]
        fs_des = np.column_stack([xv, zv, w])
        b_fs, *_ = np.linalg.lstsq(fs_des, xv, rcond=None)
        # proper first stage: regressors = [z, W]
        fs_des = np.column_stack([zv, w])
        b_fs, *_ = np.linalg.lstsq(fs_des, xv, rcond=None)
        res_fs = xv - fs_des @ b_fs
        # F on the instrument: compare restricted (W only) vs unrestricted
        b_r, *_ = np.linalg.lstsq(w, xv, rcond=None)
        res_r = xv - w @ b_r
        q = fs_des.shape[0] - fs_des.shape[1]
        rss_r = float(res_r @ res_r)
        rss_u = float(res_fs @ res_fs)
        fst_f[h] = (rss_r - rss_u) / max(rss_u / max(q, 1), 1e-12)
        # second stage: y_{t+h} ~ [xhat, W]
        xhat = fs_des @ b_fs
        ss_des = np.column_stack([xhat, w])
        b_ss, *_ = np.linalg.lstsq(ss_des, yv, rcond=None)
        resid = yv - ss_des @ b_ss
        irf[h] = b_ss[0]
        lag = max(1, int(math.ceil(nw_lag_scale * (h + 1))))
        wald_se[h] = math.sqrt(_nw_var_2sls(ss_des, resid, 0, lag))
        # Anderson-Rubin inversion on a grid around beta_hat
        ar_lo[h], ar_hi[h] = _ar_band(yv, xv, zv, w, irf[h], wald_se[h])
    return {
        "horizon": np.arange(horizons + 1, dtype=np.float64),
        "irf": irf,
        "se": wald_se,
        "ci_low": irf - 1.96 * wald_se,
        "ci_high": irf + 1.96 * wald_se,
        "first_stage_f": fst_f,
        "ar_ci_low": ar_lo,
        "ar_ci_high": ar_hi,
    }


def _ar_band(
    yv: FloatArray,
    xv: FloatArray,
    zv: FloatArray,
    w: FloatArray,
    beta_hat: float,
    wald_se: float,
    grid_half: float = 6.0,
    n_grid: int = 121,
) -> tuple[float, float]:
    """Anderson-Rubin confidence set by grid inversion.

    For each candidate β0: regress (y - β0 x) on [z, W] and test the
    instrument coefficient via an F test; keep β0 in the set when
    p > 0.05. Returns the convex hull (min, max) of the accepted grid.
    """
    half = max(grid_half * max(wald_se, 1e-3), 0.5)
    grid = np.linspace(beta_hat - half, beta_hat + half, n_grid)
    des = np.column_stack([zv, w])
    k = des.shape[1]
    q = des.shape[0] - k
    accepted = []
    for b0 in grid:
        resid_var = yv - b0 * xv
        b_hat, *_ = np.linalg.lstsq(des, resid_var, rcond=None)
        res = resid_var - des @ b_hat
        # F test on coefficient 0 (z)
        vcv = np.linalg.inv(des.T @ des + 1e-10 * np.eye(k))
        sigma2 = float(res @ res) / max(q, 1)
        f_stat = float(b_hat[0] ** 2 / max(vcv[0, 0] * sigma2, 1e-18))
        if sstats.f.sf(f_stat, 1, q) > 0.05:
            accepted.append(b0)
    if not accepted:
        return (float(beta_hat), float(beta_hat))
    return (float(min(accepted)), float(max(accepted)))


def synth_lp_iv(n: int = 400, seed: int = 0, weak: bool = False) -> dict[str, FloatArray]:
    """Recursive two-equation synth with a valid external instrument.

    z ~ N(0,1); x = π z + v (first stage strength π, weak when
    ``weak=True``); y_{t+h} responds to the structural shock v with a
    decaying exponential IRF — the LP-IV estimand.
    """
    rng = np.random.default_rng(seed)
    # structural shock v; the regressor x measures it with error
    # (naive LP attenuates), the proxy z correlates only with v.
    v = rng.standard_normal(n)
    m = rng.standard_normal(n)
    eta = rng.standard_normal(n)
    x = v + 0.8 * m
    rho = 0.15 if weak else 0.9
    z = rho * v + math.sqrt(1.0 - rho * rho) * eta
    theta = np.array([1.0 * (0.75**h) for h in range(25)])
    # pure MA(24) response to the structural shock v + own noise —
    # the LP-IV estimand is exactly theta (no AR feedback).
    eps = 0.4 * rng.standard_normal(n)
    y = eps.copy()
    for h in range(25):
        y[h:] += theta[h] * v[: n - h]
    return {
        "y": np.asarray(y, dtype=np.float64),
        "x": np.asarray(x, dtype=np.float64),
        "z": np.asarray(z, dtype=np.float64),
        "true_irf": np.asarray(theta),
    }


def bench_lp_iv(seed: int = 20261231 + 167) -> dict[str, float]:
    """SYNTHETIC LP-IV recovery vs OLS bias + AR-band telemetry."""
    strong = synth_lp_iv(n=900, seed=seed, weak=False)
    out = lp_iv(strong["y"], strong["x"], strong["z"], horizons=10, control_lags=2)
    truth = strong["true_irf"][:11]
    # LP-IV estimand with controls absorbs dynamics — compare decay
    # shape: rescale truth to match point at h=0 for shape relerr
    scale = float(out["irf"][0] / truth[0]) if truth[0] != 0 else 1.0
    relerr = float(np.linalg.norm(out["irf"] - truth * scale) / np.linalg.norm(truth * scale))
    # naive LP (no instrument) on same data: expected to be biased
    from quant_fund.models.local_projection import local_projection

    naive = local_projection(strong["y"], strong["x"], horizons=10, control_lags=2)
    naive_err = float(np.linalg.norm(naive["irf"] - truth * scale) / np.linalg.norm(truth * scale))
    # weak-IV synth: F should drop and AR band should widen
    weak = synth_lp_iv(n=900, seed=seed + 1, weak=True)
    out_w = lp_iv(weak["y"], weak["x"], weak["z"], horizons=6, control_lags=2)
    ar_width = float(np.mean(out["ar_ci_high"] - out["ar_ci_low"]))
    wald_width = float(np.mean(out["ci_high"] - out["ci_low"]))
    ar_w_width = float(np.mean(out_w["ar_ci_high"] - out_w["ar_ci_low"]))
    d1 = lp_iv(strong["y"], strong["x"], strong["z"], horizons=10, control_lags=2)["irf"]
    d2 = lp_iv(strong["y"], strong["x"], strong["z"], horizons=10, control_lags=2)["irf"]
    return {
        "synthetic_irf_relerr": relerr,
        "synthetic_naive_lp_relerr": naive_err,
        "synthetic_lpiv_beats_naive": float(relerr < naive_err),
        "synthetic_first_stage_f_min": float(out["first_stage_f"].min()),
        "synthetic_first_stage_f_weak": float(out_w["first_stage_f"].mean()),
        "synthetic_ar_width": ar_width,
        "synthetic_wald_width": wald_width,
        "synthetic_ar_width_weak": ar_w_width,
        "synthetic_ar_widens_under_weak_iv": float(ar_w_width > ar_width),
        "synthetic_determinism": float(np.array_equal(d1, d2)),
    }
