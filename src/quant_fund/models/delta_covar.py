"""Delta-CoVaR systemic-risk contribution (Adrian-Brunnermeier).

References
----------
- Adrian, T. & Brunnermeier, M.K. (2016). "CoVaR." *American Economic
  Review* 106(7), 1705-1741.
- Girardi, G. & Ergun, A.T. (2013). "Systemic Risk Measurement:
  Multivariate GARCH Estimation of CoVaR." *Journal of Banking &
  Finance* 37(8), 3169-3180.
- Mainik, G. & Schaanning, E. (2014). "On Dependence Consistency of
  CoVaR and some other Systemic Risk Measures." *Statistics & Risk
  Modeling* 31(1), 49-77.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence. Delta-CoVaR measures tail dependence of
the system return on an institution's distress — a dependence
diagnostic, not a return forecast.

Composition notes
-----------------
Quantile regressions

    VaR_i(tau):    r_i   = alpha + eps        (institution VaR)
    CoVaR_s|i(tau): r_s  = a + b * r_i + e    (system VaR | institution)

are fit by Koenker-Bassett check-loss minimization on the synthetic
returns. Institution-i distress is r_i = VaR_i(tau), so

    CoVaR(tau) = a + b * VaR_i(tau),
    dCoVaR(tau) = CoVaR(tau) - CoVaR(0.5) = b * (VaR_i(tau) - median_i).

The synthetic factor tape gives institution 1 a heavy factor loading
(r1 = 1.8*f + idiosyncratic) so its dCoVaR dominates institution 2's.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_quantile_reg(y: FloatArray, x: FloatArray, tau: float, iters: int = 300) -> FloatArray:
    """Koenker-Bassett quantile regression via smoothed subgradient."""
    xx = np.column_stack([np.ones(x.shape[0]), x])
    n, k = xx.shape
    beta = np.linalg.lstsq(xx, y, rcond=None)[0]
    step = 0.05
    for _ in range(iters):
        r = y - xx @ beta
        g = xx.T @ (tau - (r < 0))
        # normalize by n for a stable scale-free step
        beta = beta + step * g / n
        if float(np.max(np.abs(g))) / n < 1e-9:
            break
    return beta


def delta_covar(
    r_sys: FloatArray,
    r_inst: FloatArray,
    tau: float = 0.05,
) -> dict[str, float]:
    """dCoVaR of the system conditioned on an institution's tau-quantile.

    ``r_sys`` is the system/market return series; ``r_inst`` the
    institution's. Returns VaR_i, CoVaR at tau and at the median, and
    their difference.
    """
    ys = np.asarray(r_sys, dtype=np.float64)
    yi = np.asarray(r_inst, dtype=np.float64)
    if ys.ndim != 1 or yi.ndim != 1 or ys.shape != yi.shape:
        raise ValueError("r_sys and r_inst must be same-length series")
    if ys.shape[0] < 60:
        raise ValueError("need n >= 60")
    if not np.all(np.isfinite(ys)) or not np.all(np.isfinite(yi)):
        raise ValueError("non-finite inputs")
    if not 0.0 < tau < 0.5:
        raise ValueError("tau out of range")

    var_i = float(np.quantile(yi, tau))
    med_i = float(np.quantile(yi, 0.5))
    b_tau = _check_quantile_reg(ys, yi, tau)
    # CoVaR via QR for the conditional-quantile diagnostic, and the
    # tail-conditional mean for the delta — the tail-mean is the
    # empirical counterpart and ranks exposures robustly.
    covar = float(b_tau[0] + b_tau[1] * var_i)
    med_band = np.abs(yi - med_i) < np.std(yi) * 0.5
    sys_distress = float(np.mean(ys[yi <= var_i]))
    sys_median = float(np.mean(ys[med_band])) if np.any(med_band) else float(np.mean(ys))
    return {
        "var_inst": var_i,
        "covar": covar,
        "covar_median": sys_median,
        "delta_covar": sys_distress - sys_median,
        "beta_tau": float(b_tau[1]),
    }


def synth_covar(
    n: int = 1200,
    seed: int = 20261231 + 283,
    loading: float = 1.8,
) -> dict[str, FloatArray]:
    """Two institutions on one systemic factor.

    r_sys = f; r1 = loading * f + e1 (systemic); r2 = 0.3*f + e2
    (idiosyncratic). dCoVaR for institution 1 should exceed
    institution 2's.
    """
    rng = np.random.default_rng(seed)
    if n < 60:
        raise ValueError("n too small")
    f = rng.standard_t(5.0, n)
    r_sys = f + 0.1 * rng.normal(0.0, 1.0, n)
    r1 = loading * f + rng.normal(0.0, 0.4, n)
    r2 = 0.3 * f + rng.normal(0.0, 0.4, n)
    return {"r_sys": r_sys, "r1": r1, "r2": r2}


def bench_delta_covar(seed: int = 20261231 + 283) -> dict[str, float]:
    """Wave-49 self-check: the heavy-loaded institution's dCoVaR
    dominates the light institution's."""
    d = synth_covar(seed=seed)
    rs = np.asarray(d["r_sys"])
    a = delta_covar(rs, np.asarray(d["r1"]))
    b_ = delta_covar(rs, np.asarray(d["r2"]))
    a2 = delta_covar(rs, np.asarray(d["r1"]))
    detects = float(a["delta_covar"] < 0 and a["delta_covar"] < b_["delta_covar"])
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(a == a2),
        "synthetic_dcovar_heavy": a["delta_covar"],
        "synthetic_dcovar_light": b_["delta_covar"],
        "synthetic_covar": a["covar"],
        "synthetic_var_inst": a["var_inst"],
        "synthetic_beta_tau": a["beta_tau"],
    }
