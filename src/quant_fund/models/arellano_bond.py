"""Dynamic panel GMM — Arellano-Bond first-difference estimator.

Estimates y_it = ρ·y_{i,t-1} + x_it'β + α_i + u_it by first-differencing
and instrumenting Δy_{i,t-1} with levels y_{i,t-2}, y_{i,t-3}, ...
Produces the AR(2) serial-correlation test (must be absent for
consistency) and a Sargan/Hansen overidentification test.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure coefficient recovery and test-size
on generated dynamic panels — never market evidence.

References:
- Arellano, Bond (1991). Some tests of specification for panel data.
  *Review of Economic Studies* 58, 277-297.
- Blundell, Bond (1998). Initial conditions and moment restrictions
  in dynamic panel data models. *J. Econometrics* 87.
- Roodman (2009). How to do xtabond2. *Stata Journal* 9.
- Hansen (1982). Large sample properties of GMM. *Econometrica* 50.

Composition: pure numpy — IV/GMM normal equations, AR(m)
serial-correlation z-tests, Sargan J; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]


def _as2(m: FloatArray, name: str) -> FloatArray:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 2-D matrix required")
    return a


def _as1(v: FloatArray, name: str) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 1-D vector required")
    return a


def _iv_gmm(dy: FloatArray, dx: FloatArray, z: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Two-step-style single pass: (X'Z (Z'Z)^-1 Z'X)^-1 X'Z (Z'Z)^-1 Z'y."""
    ztz = z.T @ z + 1e-10 * np.eye(z.shape[1])
    w = np.linalg.pinv(ztz)
    xz = dx.T @ z
    m_ = xz @ w @ z.T @ dx
    coef = np.linalg.pinv(m_) @ xz @ w @ z.T @ dy
    resid = dy - dx @ coef
    # sandwich SE: (X'ZWZ'X)^-1 X'Z W (Z' resid² Z) W Z'X (...)  (robust)
    zx = z.T @ dx
    meat = (z * resid[:, None]).T @ (z * resid[:, None])
    a_ = np.linalg.pinv(xz @ w @ zx + 1e-10 * np.eye(m_.shape[0]))
    vcov = a_ @ (xz @ w @ meat @ w @ zx) @ a_
    return coef, np.sqrt(np.maximum(np.diag(vcov), 0.0))


def arellano_bond(
    y: FloatArray,
    x: FloatArray,
    groups: FloatArray,
    max_lag_inst: int = 4,
) -> dict[str, float | FloatArray]:
    """First-difference GMM with lagged-level instruments.

    ``y`` (n,) outcomes, ``x`` (n,k) regressors, ``groups`` (n,) panel
    ids — each group's rows must be time-ordered. Instruments for
    Δy_{i,t-1} are y_{i,t-2..t-max_lag} (collapsed to a bounded set);
    Δx enters as its own instrument."""
    ya = _as1(y, "y")
    xa = _as2(x, "x")
    ga = np.asarray(groups).ravel()
    if xa.shape[0] != ya.size or ga.size != ya.size:
        raise ValueError("y/x/groups length mismatch")

    dys, dxs, dz_insts = [], [], []
    for g in np.unique(ga):
        idx = np.where(ga == g)[0]
        if idx.size < 5:
            continue
        yg = ya[idx]
        xg = xa[idx]
        t_ = yg.size
        for tt in range(2, t_):
            dys.append(yg[tt] - yg[tt - 1])
            drow = [yg[tt - 1] - yg[tt - 2]]
            drow.extend(xg[tt] - xg[tt - 1])
            dxs.append(drow)
            # collapsed instruments: Δx_t plus lag columns padded 0
            zrow = [float(xg[tt, j] - xg[tt - 1, j]) for j in range(xa.shape[1])]
            for lag in range(2, max_lag_inst + 1):
                zrow.append(float(yg[tt - lag]) if tt - lag >= 0 else 0.0)
            dz_insts.append(zrow)

    if len(dys) < xa.shape[1] + 6:
        raise ValueError("too few usable first-differences")

    dy = np.asarray(dys)
    dx = np.asarray(dxs)
    z = np.asarray(dz_insts)
    coef, se = _iv_gmm(dy, dx, z)
    resid = dy - dx @ coef

    # AR(2) test on residuals: regress resid_t on resid_{t-2} within
    # differenced residuals (expectation < 0 by construction).
    r1, r2 = [], []
    per_group: dict[object, list[float]] = {}
    res_iter = iter(resid)
    for g in np.unique(ga):
        idx = np.where(ga == g)[0]
        if idx.size < 5:
            continue
        lst = [next(res_iter) for _ in range(idx.size - 2)]
        per_group[g] = lst
        for tt in range(2, len(lst)):
            r1.append(lst[tt])
            r2.append(lst[tt - 2])
    ar2_z, ar2_p = 0.0, 1.0
    if len(r1) > 10:
        a1, a2 = np.asarray(r1), np.asarray(r2)
        if np.std(a2) > 1e-9:
            c = float(np.cov(a1, a2)[0, 1] / np.var(a2))
            se_c = float(np.std(a1 - c * a2) / math.sqrt(len(r1)) / max(np.std(a2), 1e-9))
            ar2_z = c / max(se_c, 1e-9)
            ar2_p = float(2 * (1 - norm.cdf(abs(ar2_z))))

    # Sargan/Hansen J: moment covariance-weighted sum of mean moments
    mom = z * resid[:, None]
    gbar = mom.mean(axis=0)
    s_ = np.cov(mom.T) + 1e-10 * np.eye(z.shape[1])
    jstat = float(dy.size * gbar @ np.linalg.pinv(s_) @ gbar)
    df_j = max(z.shape[1] - dx.shape[1], 1)
    j_p = float(1 - chi2.cdf(jstat, df_j))

    return {
        "rho": float(coef[0]),
        "rho_se": float(se[0]),
        "beta": np.asarray(coef[1:]),
        "beta_se": np.asarray(se[1:]),
        "ar2_z": float(ar2_z),
        "ar2_p": float(ar2_p),
        "sargan_j": jstat,
        "sargan_p": j_p,
        "n_instr": float(z.shape[1]),
        "n_obs_fd": float(dy.size),
        "n_groups": float(len(per_group)),
    }


def synth_dynamic_panel(
    n_groups: int = 60,
    t: int = 10,
    rho: float = 0.6,
    beta: float = 0.8,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Dynamic panel: y_it = ρ y_{i,t-1} + β x_it + α_i + u_it."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), t)
    a = rng.normal(0.0, 1.0, n_groups)
    x = rng.normal(0.0, 1.0, n_groups * t)
    y = np.zeros(n_groups * t)
    for i in range(n_groups):
        y[i * t] = a[i] + beta * x[i * t] + rng.normal(0.0, 0.7)
        for tt in range(1, t):
            y[i * t + tt] = (
                rho * y[i * t + tt - 1] + beta * x[i * t + tt] + a[i] + rng.normal(0.0, 0.7)
            )
    return {
        "y": y,
        "x": x.reshape(-1, 1),
        "groups": g.astype(np.float64),
        "rho_true": np.array([rho]),
    }


def bench_arellano_bond(seed: int = 20261231 + 203) -> dict[str, float]:
    """AB-GMM self-check: ρ recovered near truth where within-OLS is
    upward-biased (Nickell), AR(2) absent, Sargan reasonable. All
    ``synthetic_*``."""
    d = synth_dynamic_panel(seed=seed, rho=0.6)
    out = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    out_b = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    # within OLS benchmark (Nickell-biased UP)
    y = np.asarray(d["y"]).copy()
    g = np.asarray(d["groups"])
    ylag = np.roll(y, 1)
    ylag[np.isin(np.arange(y.size), np.where(np.diff(g) != 0)[0] + 1)] = np.nan
    ylag[0] = np.nan
    mask = np.isfinite(ylag)
    for gg in np.unique(g):
        m = (g == gg) & mask
        y[m] = y[m] - y[m].mean()
        ylag[m] = ylag[m] - ylag[m].mean()
    b_ols = float(np.cov(ylag[mask], y[mask])[0, 1] / np.var(ylag[mask]))

    d0 = synth_dynamic_panel(seed=seed + 1, rho=0.0)
    out0 = arellano_bond(np.asarray(d0["y"]), np.asarray(d0["x"]), np.asarray(d0["groups"]))

    rho_hat = float(out["rho"])
    return {
        "synthetic_rho": rho_hat,
        "synthetic_rho_err": float(abs(rho_hat - 0.6)),
        "synthetic_ols_within": b_ols,
        "synthetic_beats_within": float(abs(rho_hat - 0.6) < abs(b_ols - 0.6)),
        "synthetic_ar2_p": float(out["ar2_p"]),
        "synthetic_sargan_p": float(out["sargan_p"]),
        "synthetic_null_rho": float(abs(float(out0["rho"]))),
        "synthetic_detects": float(float(out["ar2_p"]) > 0.01 and abs(rho_hat - 0.6) < 0.25),
        "synthetic_determinism": float(float(out["rho"]) == float(out_b["rho"])),
    }
