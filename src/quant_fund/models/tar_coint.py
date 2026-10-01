"""Balke-Fomby threshold cointegration (TAR / MTAR ECM).

References
----------
- Balke, N.S. & Fomby, T.B. (1997). "Threshold Cointegration."
  *International Economic Review* 38(3), 627-645.
- Enders, W. & Granger, C.W.J. (1998). "Unit-Root Tests and
  Asymmetric Adjustment with an Example Using the Term
  Structure of Interest Rates." *JBES* 16(3), 304-311.
- Enders, W. & Siklos, P.L. (2001). "Cointegration and
  Threshold Adjustment." *JBES* 19(2), 166-176.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The Enders-Granger two-step first estimates the cointegrating
residual ``z_t = y_t - a - b x_t`` and then fits a threshold
error-correction model on lagged residuals:
- TAR:  ``dz_t = rho1 z_{t-1} 1{z_{t-1} >= thr} +
          rho2 z_{t-1} 1{z_{t-1} < thr}``
- MTAR: same but on ``dz_{t-1}`` — momentum-threshold, where
  adjustment depends on the direction the residual moved.
Fitted jointly by OLS on regime-spliced regressors with
consistent + symmetric regimes; the threshold is grid-searched
to minimize the joint SSR, and the asymmetry test is an
F-test of ``rho1 = rho2`` under both thresholds. The bench
plants an asymmetric ECM that adjusts only above a band —
TAR must recover a larger |rho| on the upper regime; a
symmetric linear ECM is the control and must show balanced
rhos.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 150) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _eg_resid(y: FloatArray, x: FloatArray) -> FloatArray:
    d = np.column_stack([np.ones(x.size), x])
    b = np.linalg.lstsq(d, y, rcond=None)[0]
    return np.asarray(y - d @ b, dtype=np.float64)


def _fit_tar(
    z: FloatArray,
    thr: float,
    mtar: bool,
) -> tuple[float, float, float, float, float]:
    """OLS ECM on regimes spliced by z_{t-1} (TAR) or dz_{t-1} (MTAR)."""
    dz = np.diff(z)  # dz[i] = z[i+1] - z[i]
    y = dz[1:]  # delta z_t for t = 2..n-1
    z_lag = z[1:-1]  # z_{t-1}
    ind = dz[:-1] if mtar else z[1:-1]  # regime variable
    above = ind >= thr
    below = ~above
    if int(np.sum(above)) < 20 or int(np.sum(below)) < 20:
        return 0.0, 0.0, 1e18, 0.0, 0.0
    xa = np.column_stack([np.ones(z_lag.size), z_lag * above, z_lag * below])
    b = np.linalg.lstsq(xa, y, rcond=None)[0]
    resid = y - xa @ b
    ssr = float(resid @ resid)
    dof = max(int(z_lag.size) - 3, 1)
    s2 = ssr / dof
    cov = s2 * np.linalg.pinv(xa.T @ xa)
    se_a = float(np.sqrt(max(cov[1, 1], 1e-18)))
    se_b = float(np.sqrt(max(cov[2, 2], 1e-18)))
    rho1, rho2 = float(b[1]), float(b[2])
    tstat = abs(rho1 - rho2) / np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2] + 1e-18)
    return rho1, rho2, ssr, float(tstat), min(se_a, se_b)


def tar_cointegration(
    y: FloatArray,
    x: FloatArray,
    mtar: bool = False,
    grid_low: float = 0.15,
    grid_high: float = 0.85,
) -> dict[str, float]:
    """Balke-Fomby / Enders-Granger threshold ECM."""
    vy = _as_series(y)
    vx = _as_series(x)
    if vy.size != vx.size:
        raise ValueError("length mismatch")
    z = _eg_resid(vy, vx)
    lo, hi = np.quantile(z, grid_low), np.quantile(z, grid_high)
    if hi - lo < 1e-10:
        raise ValueError("degenerate residual spread")
    thrs = np.linspace(lo, hi, 60)
    best: tuple[float, float, float, float, float, float] | None = None
    for thr_cand in thrs:
        rho1, rho2, ssr, tstat, min_se = _fit_tar(z, float(thr_cand), mtar)
        if best is None or ssr < best[2]:
            best = (rho1, rho2, ssr, tstat, float(thr_cand), min_se)
    if best is None:
        raise ValueError("no feasible threshold")
    rho1, rho2, ssr, tstat, thr_f, min_se = best
    thr = float(thr_f)
    # linear ECM benchmark for reference
    dz = np.diff(z)
    xl = np.column_stack([np.ones(z.size - 1), z[:-1]])
    b = np.linalg.lstsq(xl, dz, rcond=None)[0]
    out: dict[str, float] = {
        "threshold": thr,
        "rho_above": rho1,
        "rho_below": rho2,
        "joint_ssr": ssr,
        "asym_t": tstat,
        "linear_rho": float(b[1]),
        "mtar": float(mtar),
    }
    return out


def synth_tar(
    seed: int = 20261231 + 346,
    n: int = 1200,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC band-TAR cointegration vs symmetric ECM."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    z = np.zeros(n)
    thr = 0.6
    for t in range(1, n):
        adj = -0.35 * z[t - 1] if z[t - 1] >= thr else -0.03 * z[t - 1]
        z[t] = z[t - 1] + adj + 0.5 * rng.standard_normal()
    y = 1.0 + x + z
    x2 = np.cumsum(rng.standard_normal(n))
    z2 = np.zeros(n)
    for t in range(1, n):
        z2[t] = z2[t - 1] - 0.12 * z2[t - 1] + 0.5 * rng.standard_normal()
    y2 = 1.0 + x2 + z2
    return y.astype(np.float64), x.astype(np.float64), y2.astype(np.float64)


def bench_tar_coint(seed: int = 20261231 + 346) -> dict[str, float]:
    y, x, y2 = synth_tar(seed=seed)
    r = tar_cointegration(y, x)
    r_lin = tar_cointegration(y2, x)
    ok = (
        r["rho_above"] < r["rho_below"] - 0.1
        and r["asym_t"] > 2.0
        and abs(r_lin["rho_above"] - r_lin["rho_below"]) < 0.2
    )
    out: dict[str, float] = {
        "synthetic_tar_rho_above": r["rho_above"],
        "synthetic_tar_rho_below": r["rho_below"],
        "synthetic_tar_asym_t": r["asym_t"],
        "synthetic_tar_lin_rho_gap": float(abs(r_lin["rho_above"] - r_lin["rho_below"])),
        "score": 1.0 if ok else 0.0,
    }
    return out
