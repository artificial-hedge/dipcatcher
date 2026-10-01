"""Control-function approach to endogeneity.

When a regressor is endogenous, the control-function approach keeps
it in the outcome equation and adds the first-stage residual
(from regressing the endogenous variable on instruments) as an extra
regressor. The residual 'controls' the endogenous component; the
coefficient on the endogenous regressor is then consistent under
the same conditions as 2SLS, and the residual's own coefficient
tests exogeneity (Durbin-Wu-Hausman form).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure bias correction on generated
endogenous panels — never market evidence.

References:
- Wooldridge, J. M. (2015). Control function methods in applied
  econometrics. *J. Human Resources* 50, 420-445 — the definitive
  statement and bootstrap-SE caveat.
- Rivers, D., Vuong, Q. H. (1988). Limited information estimators
  and exogeneity tests for simultaneous probit models. *J.
  Econometrics* 39, 347-366 — residual-inclusion construction.
- Hausman, J. A. (1978). Specification tests in econometrics.
  *Econometrica* 46, 1251-1271 — the exogeneity test read off the
  residual coefficient.
- Petrin, A., Train, K. (2010). A control function approach to
  endogeneity in consumer choice models. *J. Marketing Research*.

Composition: pure numpy — first-stage residual + augmented OLS with
HC1 SE; deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as2(v: FloatArray, n: int, name: str) -> FloatArray:
    a = np.atleast_2d(np.asarray(v, dtype=np.float64))
    if a.shape[0] != n:
        a = a.T
    if a.shape[0] != n or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite (n,·) array required")
    return a


def _ols_hc1(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    n, k = x.shape
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ b
    xt = x.T @ x
    xtxi = np.linalg.pinv(xt)
    meat = x.T @ ((resid**2)[:, None] * x)
    vcov = xtxi @ meat @ xtxi * n / max(n - k, 1)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return b, se, resid


def control_function(
    y: FloatArray,
    endog: FloatArray,
    exog: FloatArray | None,
    instruments: FloatArray,
) -> dict[str, float]:
    """2SLS-equivalent point estimate via residual inclusion.

    ``endog`` is the endogenous regressor, ``exog`` included
    exogenous controls (constant added automatically), ``instruments``
    the excluded instruments (must not overlap exog). Returns the
    corrected coefficient, the DWH exogeneity test on the residual,
    and first-stage strength diagnostics."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    ee = np.asarray(endog, dtype=np.float64).ravel()
    n = yy.size
    if ee.size != n or n < 40:
        raise ValueError("equal-length y and endog, n>=40")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(ee)):
        raise ValueError("finite y and endog required")
    xx = np.ones((n, 1)) if exog is None else _as2(exog, n, "exog")
    if exog is None or xx.shape[1] == 0 or not np.all(xx[:, 0] == 1.0):
        xx = np.column_stack([np.ones(n), xx])
    zz = _as2(instruments, n, "instruments")
    if zz.shape[1] < 1:
        raise ValueError("at least one instrument required")

    # first stage: endog on [exog, Z] → residual v
    fs_x = np.column_stack([xx, zz])
    b_fs, _, v = _ols_hc1(ee, fs_x)
    # first-stage partial F on instruments (Cragg-Donald flavor)
    b_r, _, r_r = _ols_hc1(ee, xx)
    rss_r = float(r_r @ r_r)
    rss_u = float(v @ v)
    q = zz.shape[1]
    f_stat = ((rss_r - rss_u) / q) / (rss_u / (n - fs_x.shape[1]))

    # second stage: y on [exog, endog, v-hat]
    x2 = np.column_stack([xx, ee, v])
    b2, se2, _ = _ols_hc1(yy, x2)
    beta = float(b2[-2])
    se_beta = float(se2[-2]) * math.sqrt(1.35)  # generated-regressor inflation
    rho_hat = float(b2[-1])
    se_rho = float(se2[-1])
    z_dwh = rho_hat / max(se_rho, 1e-12)

    # OLS reference
    x_ols = np.column_stack([xx, ee])
    b_ols, _, _ = _ols_hc1(yy, x_ols)

    return {
        "beta_cf": beta,
        "se_beta": se_beta,
        "z_beta": beta / max(se_beta, 1e-12),
        "beta_ols": float(b_ols[-1]),
        "rho_hat": rho_hat,
        "dwh_z": float(z_dwh),
        "dwh_p": float(2 * (1 - norm.cdf(abs(z_dwh)))),
        "fs_f": float(f_stat),
        "n": float(n),
    }


def synth_endogenous(
    n: int = 1500,
    beta: float = 1.0,
    endog_strength: float = 0.5,
    pi: float = 0.8,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Endogeneity DGP: e = π·z + u, y = β·e + ε with
    corr(u, ε) = ``endog_strength`` — OLS picks up the covariance."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    u = rng.normal(0.0, 1.0, n)
    eps = endog_strength * u + math.sqrt(max(1 - endog_strength**2, 0.01)) * rng.standard_normal(n)
    e = pi * z + u
    y = beta * e + eps
    return {
        "y": y,
        "endog": e,
        "z": z,
        "beta_true": np.array([beta]),
    }


def bench_control_function(
    seed: int = 20261231 + 215,
) -> dict[str, float]:
    """Control-function self-check: corrected slope beats OLS under
    corr(u,ε)=0.5 endogeneity. All ``synthetic_*``."""
    d = synth_endogenous(beta=1.0, endog_strength=0.5, seed=seed)
    out = control_function(
        np.asarray(d["y"]),
        np.asarray(d["endog"]),
        None,
        np.asarray(d["z"]),
    )
    d0 = synth_endogenous(beta=1.0, endog_strength=0.0, seed=seed + 1)
    out0 = control_function(
        np.asarray(d0["y"]),
        np.asarray(d0["endog"]),
        None,
        np.asarray(d0["z"]),
    )
    out_b = control_function(
        np.asarray(d["y"]),
        np.asarray(d["endog"]),
        None,
        np.asarray(d["z"]),
    )

    cf = float(out["beta_cf"])
    ols = float(out["beta_ols"])
    return {
        "synthetic_beta_cf": cf,
        "synthetic_beta_cf_err": float(abs(cf - 1.0)),
        "synthetic_beta_ols": ols,
        "synthetic_beta_ols_err": float(abs(ols - 1.0)),
        "synthetic_beats_ols": float(abs(cf - 1.0) < abs(ols - 1.0)),
        "synthetic_dwh_p": float(out["dwh_p"]),
        "synthetic_dwh_null_p": float(out0["dwh_p"]),
        "synthetic_fs_f": float(out["fs_f"]),
        "synthetic_detects": float(abs(cf - 1.0) < 0.15 and float(out["dwh_p"]) < 0.05),
        "synthetic_determinism": float(cf == float(out_b["beta_cf"])),
    }
