"""Lewbel heteroskedasticity-generated instruments.

References
----------
- Lewbel, A. (2012). "Using Heteroscedasticity to Identify and Estimate
  Mismeasured and Endogenous Regressor Models." *Journal of Business &
  Economic Statistics* 30(1), 67-80.
- Lewbel, A. (2018). "Identification and Estimation Using Heteroscedastic
  Transformations Without Instruments." *Journal of Econometrics* 205(1),
  18-32.
- Baum, C.F. & Lewbel, A. (2019). "Advice on Using Heteroscedasticity
  Based Identification." *Stata Journal* 19(4), 757-767.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
When no external instrument exists, Lewbel's construction builds
generated instruments

    z_j = (x_j - mean(x_j)) * v_hat

where ``v_hat`` is the residual from the reduced-form regression
``X_endo ~ Z``. Provided ``E[Z eps] = 0`` and ``cov(Z, v^2) != 0``
(heteroskedastic reduced-form errors — the identifying content), the z_j
are valid: relevant by construction and excludable whenever the
covariates are. The structural equation is then fit by 2SLS using the
generated instrument set. This module reports the 2SLS coefficient, an
instrument-strength diagnostic (first-stage partial R^2 of the generated
set), and the heteroskedasticity-correlation statistic the
identification rests on.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ols(y: FloatArray, x: FloatArray) -> FloatArray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _resid(y: FloatArray, x: FloatArray) -> FloatArray:
    return y - x @ _ols(y, x)


def lewbel_iv(
    y: FloatArray,
    x_endo: FloatArray,
    x_exo: FloatArray | None = None,
) -> dict[str, float]:
    """2SLS with Lewbel-generated instruments.

    ``x_endo`` is the single endogenous regressor; ``x_exo`` are included
    exogenous covariates (intercept appended). The generated instrument
    block is ``(exo - mean(exo)) * v_hat`` where ``v_hat`` is the
    first-stage residual ``x_endo ~ [1, exo]``.
    """
    yy = np.asarray(y, dtype=np.float64)
    xe = np.asarray(x_endo, dtype=np.float64)
    if xe.ndim == 1:
        xe = xe[:, None]
    if xe.shape[1] != 1:
        raise ValueError("x_endo must be a single regressor")
    if x_exo is None:
        xx = np.zeros((xe.shape[0], 1), dtype=np.float64)
    else:
        xx = np.asarray(x_exo, dtype=np.float64)
        if xx.ndim == 1:
            xx = xx[:, None]
    n = yy.shape[0]
    if yy.ndim != 1 or xe.shape[0] != n or xx.shape[0] != n:
        raise ValueError("y, x_endo, x_exo must share n rows")
    if n < 60:
        raise ValueError("need n >= 60")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xe)) or not np.all(np.isfinite(xx)):
        raise ValueError("non-finite inputs")

    one = np.ones((n, 1))
    z_mat = np.column_stack([one, xx])
    v_hat = _resid(xe[:, 0], z_mat)

    exo_c = xx - xx.mean(axis=0)
    gen = exo_c * v_hat[:, None]
    if gen.shape[1] == 0 or float(np.max(np.abs(exo_c))) == 0.0:
        gen = v_hat[:, None]

    full = np.column_stack([z_mat, gen])
    fit2 = _ols(xe[:, 0], full)
    fs_excl = _resid(xe[:, 0], z_mat)
    fs_incl = xe[:, 0] - full @ fit2
    df1 = gen.shape[1]
    df2 = n - full.shape[1]
    rss0 = float(fs_excl @ fs_excl)
    rss1 = float(fs_incl @ fs_incl)
    f_gen = float(((rss0 - rss1) / df1) / (rss1 / df2)) if rss1 > 0 else 0.0

    x_hat = full @ _ols(xe[:, 0], full)
    reg2 = np.column_stack([one, x_hat, xx])
    beta2 = _ols(yy, reg2)
    e2 = yy - np.column_stack([one, xe, xx]) @ beta2
    s2 = float(e2 @ e2) / (n - reg2.shape[1])
    xtx_inv = np.linalg.pinv(reg2.T @ reg2)
    se = float(np.sqrt(s2 * xtx_inv[1, 1]))
    beta_ols = float(_ols(yy, np.column_stack([one, xe, xx]))[1])

    het_stat = float(np.abs(np.corrcoef(xx.mean(axis=1), v_hat**2)[0, 1])) if xx.shape[1] else 0.0

    return {
        "beta_lewbel": float(beta2[1]),
        "se": se,
        "beta_ols": beta_ols,
        "f_first_stage": f_gen,
        "het_corr": het_stat,
        "n_gen_iv": float(gen.shape[1]),
    }


def synth_lewbel(
    n: int = 1500,
    seed: int = 20261231 + 276,
    beta: float = 1.0,
    rho: float = 0.6,
) -> dict[str, FloatArray]:
    """Endogenous regressor with no valid external IV.

    ``exo`` enters only the reduced-form error variance: v = s(exo) * e1
    with s = exp(0.5*exo), while u = rho*e1/s + independent noise so
    cov(v, u | exo) is constant (the shape Lewbel identification needs).
    OLS is biased; the generated instruments recover beta.
    """
    rng = np.random.default_rng(seed)
    if n < 60:
        raise ValueError("n too small")
    exo = rng.normal(0.0, 1.0, n)
    e1 = rng.normal(0.0, 1.0, n)
    e2 = rng.normal(0.0, 1.0, n)
    s = np.exp(0.5 * exo)
    v = s * e1
    # constant cov(v, u | exo) while var(v | exo) varies — Lewbel's
    # identifying moment requires exactly this shape.
    u = rho * e1 / s + np.sqrt(1.0 - rho**2) * e2
    x = 0.4 * exo + v
    y = 0.5 + beta * x + u
    return {"y": y, "x_endo": x, "x_exo": exo, "true_beta": np.full(n, beta)}


def bench_lewbel_iv(seed: int = 20261231 + 276) -> dict[str, float]:
    """Wave-48 self-check: generated instruments correct most of the OLS
    endogeneity bias."""
    d = synth_lewbel(seed=seed)
    a = lewbel_iv(np.asarray(d["y"]), np.asarray(d["x_endo"]), np.asarray(d["x_exo"]))
    b = lewbel_iv(np.asarray(d["y"]), np.asarray(d["x_endo"]), np.asarray(d["x_exo"]))
    truth = 1.0
    detects = float(abs(a["beta_lewbel"] - truth) < abs(a["beta_ols"] - truth) * 0.4)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == b),
        "synthetic_beta_lewbel": a["beta_lewbel"],
        "synthetic_beta_ols": a["beta_ols"],
        "synthetic_f_first": a["f_first_stage"],
        "synthetic_het_corr": a["het_corr"],
        "synthetic_se": a["se"],
    }
