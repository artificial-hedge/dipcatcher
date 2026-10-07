"""Hausman specification tests: FE vs RE and Durbin–Wu–Hausman (SYNTHETIC).

Two classical specification diagnostics:

* ``hausman_fe_re`` — the FE/RE test: under random effects
  (unit effect uncorrelated with X), both estimators are
  consistent and RE is efficient; under correlation only FE
  is. The statistic

    H = (β_FE − β_RE)' [V_FE − V_RE]^{-1} (β_FE − β_RE)

  is χ²_K asymptotically; the variance difference is formed
  from clustered OLS covariance estimates of the two fits.

* ``dwh_test`` — Durbin–Wu–Hausman endogeneity: regress the
  suspect regressor on the instruments, then test whether the
  fitted residual enters the outcome equation. Rejecting ⇒
  OLS is inconsistent and IV is required.

Honesty: synthetic panels with a controllable correlation
between the unit effect and the regressor (FE/RE lane) and a
controllable endogeneity strength with a valid instrument
(DWH lane); the bench checks rejection behavior in each
direction — a proper diagnostic, never market evidence.

References:
- Hausman, J. A. (1978). Specification tests in econometrics.
  *Econometrica* 46 — the general m-test.
- Mundlak, Y. (1978). On the pooling of time series and
  cross section data. *Econometrica* 46 — the auxiliary
  regression form the test nests.
- Durbin, J. (1954). Errors in variables. *Review of the
  International Statistical Institute* 22; Wu, D.-M. (1973).
  Alternative tests of independence. *Econometrica* 41;
  Hausman (1978) — the DWH endogeneity triad.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross
  Section and Panel Data* ch. 10-11 — implementation notes.

Composition: numpy + scipy only — OLS/within/demeaned fits
plus chi-square and F p-values; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2
from scipy.stats import f as f_dist

FloatArray = NDArray[np.float64]


def _ols(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    s2 = float(e @ e) / max(1, y.size - x.shape[1])
    cov = s2 * np.linalg.pinv(x.T @ x)
    return np.asarray(b), np.asarray(cov)


def hausman_fe_re(
    y: FloatArray,
    x: FloatArray,
    firm: FloatArray,
) -> dict[str, float]:
    """Hausman FE-vs-RE on a single-regressor panel. ``x`` is the
    (N,) regressor of interest (the test coefficient); ``firm``
    the unit index. Returns H, df, p and both estimates."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    ff = np.asarray(firm, dtype=np.float64)
    n = yy.size
    if yy.ndim != 1 or xx.shape != (n,) or ff.shape != (n,) or n < 50:
        raise ValueError("matched (N,) arrays, N>=50 required")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("finite inputs required")
    firms = np.unique(ff)
    if firms.size < 10:
        raise ValueError(">=10 units required")
    # FE: within transform (demean per unit) — no intercept
    y_w = yy.copy()
    x_w = xx.copy()
    for fm in firms:
        sel = ff == fm
        y_w[sel] = yy[sel] - yy[sel].mean()
        x_w[sel] = xx[sel] - xx[sel].mean()
    keep = np.abs(x_w) > 1e-12
    xw = x_w[keep, None]
    yw = y_w[keep]
    b_fe, c_fe = _ols(xw, yw)
    # RE: quasi-demeaned with θ from variance components
    sig2_e = float(np.sum((yw - xw @ b_fe) ** 2) / max(1, yw.size - 1))
    tbar = np.array([np.sum(ff == fm) for fm in firms], dtype=np.float64)
    # unit effects via between regression on means
    ym = np.array([yy[ff == fm].mean() for fm in firms])
    xm = np.array([xx[ff == fm].mean() for fm in firms])
    xb = np.column_stack([np.ones(firms.size), xm])
    b_be, _ = _ols(xb, ym)
    resid_be = ym - xb @ b_be
    sig2_a = max(
        0.0,
        float(np.sum(resid_be**2) / max(1, firms.size - 2)) - sig2_e / float(tbar.mean()),
    )
    theta = 1.0 - np.sqrt(sig2_e / (sig2_e + float(tbar.mean()) * sig2_a))
    # quasi-demean each observation by its unit mean
    means_y = {fm: float(yy[ff == fm].mean()) for fm in firms}
    means_x = {fm: float(xx[ff == fm].mean()) for fm in firms}
    y_q = yy - theta * np.array([means_y[fm] for fm in ff])
    x_q = xx - theta * np.array([means_x[fm] for fm in ff])
    one = 1.0 - theta
    xq = np.column_stack([np.full(n, one), x_q])
    b_re, c_re = _ols(xq, y_q)
    diff = float(b_fe[0] - b_re[1])
    var = max(1e-12, float(c_fe[0, 0] - c_re[1, 1]))
    h = diff**2 / var
    return {
        "h": float(h),
        "df": 1.0,
        "p": float(chi2.sf(h, 1)),
        "beta_fe": float(b_fe[0]),
        "beta_re": float(b_re[1]),
        "theta": float(theta),
        "sig2_alpha": float(sig2_a),
    }


def dwh_test(
    y: FloatArray,
    x_endo: FloatArray,
    z: FloatArray,
    x_exo: FloatArray | None = None,
) -> dict[str, float]:
    """Durbin–Wu–Hausman: first stage x_endo ~ Z (+exo), take the
    residual ν̂; outcome y ~ x_endo + ν̂ (+exo); reject when
    ν̂'s coefficient ≠ 0. Returns the t/F stat and p."""
    yy = np.asarray(y, dtype=np.float64)
    xe = np.asarray(x_endo, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    n = yy.size
    if yy.ndim != 1 or xe.shape != (n,) or n < 50:
        raise ValueError("matched (N,) arrays, N>=50 required")
    zz2 = zz[:, None] if zz.ndim == 1 else zz
    if zz2.shape[0] != n:
        raise ValueError("z rows must match n")
    parts = [np.ones(n), zz2]
    if x_exo is not None:
        xo = np.asarray(x_exo, dtype=np.float64)
        xo2 = xo[:, None] if xo.ndim == 1 else xo
        if xo2.shape[0] != n:
            raise ValueError("x_exo rows must match n")
        parts.append(xo2)
    else:
        xo2 = np.zeros((n, 0))
    first = np.column_stack(parts)
    b1 = np.linalg.lstsq(first, xe, rcond=None)[0]
    nu = xe - first @ b1
    aug = np.column_stack([np.ones(n), xe, xo2, nu])
    b2, _ = _ols(aug, yy)
    e2 = yy - aug @ b2
    s2 = float(e2 @ e2) / max(1, n - aug.shape[1])
    cov = s2 * np.linalg.pinv(aug.T @ aug)
    t_nu = float(b2[-1] / np.sqrt(max(1e-30, cov[-1, -1])))
    fstat = t_nu**2
    return {
        "t_nu": t_nu,
        "f": float(fstat),
        "p": float(f_dist.sf(fstat, 1, max(1, n - aug.shape[1]))),
        "first_stage_f": float(
            np.sum((first @ b1 - xe.mean()) ** 2) / np.sum(nu**2) * (n - aug.shape[1])
        ),
        "beta_iv_aug": float(b2[1]),
    }


def synth_fe_panel(
    n_firms: int = 60,
    t: int = 6,
    corr_alpha_x: float = 0.6,
    beta: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y_it = α_i + β·x_it + e_it; α_i correlated with x̄_i by
    construction when corr_alpha_x>0 → RE biased, FE clean."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float]] = []
    for fi in range(n_firms):
        xbar = float(rng.normal(0.0, 1.0))
        alpha = corr_alpha_x * xbar + rng.normal(0.0, 0.5)
        for _ in range(t):
            x_it = xbar + rng.normal(0.0, 0.7)
            y_it = alpha + beta * x_it + rng.normal(0.0, 1.0)
            rows.append((y_it, x_it, float(fi)))
    arr = np.array(rows)
    return {"y": arr[:, 0], "x": arr[:, 1], "firm": arr[:, 2]}


def synth_endo(
    n: int = 800,
    rho: float = 0.6,
    beta: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y = β·x + u, x = π·z + v, cov(u,v)=rho → OLS endogeneity."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    v = rng.normal(0.0, 1.0, n)
    u = rho * v + rng.normal(0.0, 0.8, n)
    x = 0.8 * z + v
    y = beta * x + u
    return {"y": y, "x": x, "z": z}


def bench_hausman_tests(seed: int = 20261231 + 273) -> dict[str, float]:
    """Spec-test self-check: FE/RE rejects under α-x correlation
    and not under independence; DWH rejects under endogeneity
    and not under exogeneity. All ``synthetic_*``."""
    fe_bad = hausman_fe_re(
        *[np.asarray(synth_fe_panel(corr_alpha_x=0.8, seed=seed)[k]) for k in ("y", "x", "firm")]
    )
    fe_ok = hausman_fe_re(
        *[np.asarray(synth_fe_panel(corr_alpha_x=0.0, seed=seed)[k]) for k in ("y", "x", "firm")]
    )
    dw_bad = dwh_test(*[np.asarray(synth_endo(rho=0.8, seed=seed)[k]) for k in ("y", "x", "z")])
    dw_ok = dwh_test(*[np.asarray(synth_endo(rho=0.0, seed=seed)[k]) for k in ("y", "x", "z")])
    fe_bad2 = hausman_fe_re(
        *[np.asarray(synth_fe_panel(corr_alpha_x=0.8, seed=seed)[k]) for k in ("y", "x", "firm")]
    )
    return {
        "synthetic_h_reject": fe_bad["h"],
        "synthetic_p_reject": fe_bad["p"],
        "synthetic_h_null": fe_ok["h"],
        "synthetic_p_null": fe_ok["p"],
        "synthetic_dwh_f_reject": dw_bad["f"],
        "synthetic_dwh_p_reject": dw_bad["p"],
        "synthetic_dwh_p_null": dw_ok["p"],
        "synthetic_beta_fe": fe_bad["beta_fe"],
        "synthetic_beta_re": fe_bad["beta_re"],
        "synthetic_detects": float(
            fe_bad["p"] < 0.05 and fe_ok["p"] > 0.05 and dw_bad["p"] < 0.05 and dw_ok["p"] > 0.05
        ),
        "synthetic_determinism": float(fe_bad2["h"] == fe_bad["h"]),
    }
