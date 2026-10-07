"""IVX robust inference on persistent predictors (KMS 2015).

References
----------
- Kostakis, A., Magdalinos, T. & Stamatogiannis, M.P.
  (2015). "Robust Econometric Inference for Stock Return
  Predictability." *Review of Financial Studies* 28(5),
  1506-1553.
- Magdalinos, T. & Phillips, P.C.B. (2009). "Econometric
  Inference in the Vicinity of Unity." CoFie WP 7-2009.
- Phillips, P.C.B. & Lee, J.H. (2013). "Predictive
  Regression Under Various Degrees of Persistence and
  Robust Long-Horizon Regression." *Journal of
  Econometrics* 177(2), 250-264.
- Demetrescu, M., Georgiev, I., Rodrigues, P. & Taylor,
  A.M.R. (2023). "Extensions to IVX Methods of Inference
  for Return Predictability." *Journal of Econometrics*
  237(2), 105271.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Predictive regression ``y_t = a + b x_{t-1} + u_t`` with a
near-integrated regressor ``x_t = rho x_{t-1} + v_t`` makes
the plain t-test on ``b`` nonstandard (Stambaugh bias). IVX
constructs a mildly-integrated instrument

    z_t = sum_{i=1..t} rho_z^{t-i} * dx_i,
    rho_z = 1 - c / T^delta,  delta in (0,1) (delta=0.95, c=1),

then runs the *joint* regression of ``y`` on
``[1, x_{t-1}, z_{t-1}]``: the IVX-Wald statistic on
``b = 0`` uses a standard chi-squared limit because the
dominant instrument is mildly, not locally, integrated. We
implement the Demetrescu et al. augmented form — the joint
regression IS the IVX test — with Newey-West HAC standard
errors. The bench couples a rho=0.98 predictor to the
target (reject) against a decoupled one (accept).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

FloatArray = NDArray[np.float64]


def _nw_cov(xm: FloatArray, resid: FloatArray, lags: int) -> FloatArray:
    """Newey-West HAC covariance of OLS coefficients."""
    n = resid.size
    g0 = resid[:, None] * xm
    meat = g0.T @ g0 / n
    for lag_i in range(1, lags + 1):
        w = 1.0 - lag_i / (lags + 1.0)
        g = (xm[lag_i:].T * resid[lag_i:]) @ (resid[:-lag_i, None] * xm[:-lag_i]) / n
        meat += w * (g + g.T)
    # indefinite lag-weighted sums can flip small-sample diagonal
    # terms negative — project the meat onto the PSD cone first.
    eigv, eigw = np.linalg.eigh((meat + meat.T) / 2.0)
    meat = (eigw * np.maximum(eigv, 0.0)) @ eigw.T
    bread = np.linalg.pinv(xm.T @ xm / n)
    return bread @ meat @ bread / n


def ivx_wald(
    y: FloatArray,
    x: FloatArray,
    delta: float = 0.95,
    c: float = 1.0,
) -> dict[str, float]:
    """IVX-Wald test of ``b = 0`` in ``y_t = a + b x_{t-1} + u``."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.ndim != 1 or xx.ndim != 1 or yy.size != xx.size or yy.size < 120:
        raise ValueError("bad inputs")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("nonfinite")
    if float(np.std(xx)) < 1e-12:
        raise ValueError("degenerate")
    t = xx.size
    dx = np.diff(xx)
    rho_z = 1.0 - c / (t**delta)
    z = np.zeros(t)
    acc = 0.0
    for i in range(dx.size):
        acc = rho_z * acc + dx[i]
        z[i + 1] = acc
    # joint IVX regression: y_t on [1, x_{t-1}, z_{t-1}];
    # the mildly-integrated z is nearly collinear with x, so
    # scale columns before the HAC covariance.
    xm = np.column_stack([np.ones(t - 1), xx[:-1], z[:-1]])
    scale = np.maximum(np.abs(xm).std(axis=0), 1e-8)
    scale[0] = 1.0
    xs = xm / scale
    coef, *_ = np.linalg.lstsq(xs, yy[1:], rcond=None)
    resid = yy[1:] - xs @ coef
    lags = int(4 * ((t - 1) / 100) ** 0.25) + 1
    cov = _nw_cov(xs, resid, lags)
    se_b = float(np.sqrt(max(cov[1, 1], 1e-30)) / scale[1])
    # scaled-space Wald is scale-free: coef_s/se_s = b/se_b
    wald = float(coef[1] ** 2 / max(cov[1, 1], 1e-30))
    # AR(1) persistence of the predictor for reporting
    ar_xm = np.column_stack([np.ones(t - 1), xx[:-1]])
    ar_coef, *_ = np.linalg.lstsq(ar_xm, xx[1:], rcond=None)
    return {
        "b": float(coef[1] / scale[1]),
        "se_b": se_b,
        "wald": wald,
        "pval": float(chi2.sf(wald, 1)),
        "rho_hat": float(ar_coef[1]),
        "rho_z": float(rho_z),
        "reject5": float(chi2.sf(wald, 1) < 0.05),
    }


def synth_ivx(
    seed: int = 20261231 + 332,
    n: int = 400,
    rho: float = 0.98,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC near-integrated predictor, coupled + null pairs."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.standard_normal()
    y_true = 0.8 * x[:-1] + 1.5 * rng.standard_normal(n - 1)
    y_true = np.concatenate([[0.0], y_true])
    # null target: stationary noise — a near-integrated regressor
    # that does NOT drive it must not reject.
    y_null = 1.5 * rng.standard_normal(n)
    return x, y_true, y_null


def bench_ivx(seed: int = 20261231 + 332) -> dict[str, float]:
    """Wave-57 self-check: coupled predictor rejects, null accepts."""
    x, y, y_null = synth_ivx(seed=seed)
    r_t = ivx_wald(y[1:], x[1:])
    r_n = ivx_wald(y_null[1:], x[1:])
    ok = r_t["reject5"] == 1.0 and r_n["reject5"] == 0.0
    return {
        "synthetic_wald_true": r_t["wald"],
        "synthetic_pval_true": r_t["pval"],
        "synthetic_wald_null": r_n["wald"],
        "synthetic_pval_null": r_n["pval"],
        "synthetic_rho_hat": r_t["rho_hat"],
        "synthetic_score": float(ok),
    }
