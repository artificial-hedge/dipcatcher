"""Adrian-Boyarchenko-Giannone Growth-at-Risk quantile machinery.

References
----------
- Adrian, T., Boyarchenko, N. & Giannone, D. (2019). "Vulnerable
  Growth." *American Economic Review* 109(4), 1263-1289.
- Koenker, R. & Bassett, G. (1978). "Regression Quantiles."
  *Econometrica* 46(1), 33-50.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The ABG pipeline: (1) quantile regressions of the target on the
conditioning variable over a fixed tau grid — fit by Koenker-Bassett
pinball minimization cast as an LP and solved deterministically;
(2) the fitted quantile function Q(tau|x) is interpolated
monotonically (PCHIP-free, linear on a fine grid with
isotonic enforcement); (3) Growth-at-Risk is the expected
shortfall under the interpolated conditional quantile function,
``GaR(x) = E[y | y <= Q(alpha|x)] = (1/alpha) int_0^alpha
Q(tau|x) dtau``, evaluated at alpha = 0.05, with the upside analog
at 0.95 for asymmetry measurement. The synth plants a conditional
location that falls in x plus conditional scale that widens in x,
so tighter conditions must depress GaR and widen the interquantile
spread.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt

FloatArray = NDArray[np.float64]

_TAU_GRID = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
_ALPHA = 0.05


def _rq_line(y: FloatArray, x: FloatArray, tau: float) -> tuple[float, float]:
    """Fit y ~ a + b x at quantile tau via the Koenker-Bassett LP."""
    n = y.size
    # variables: a, b, u_i>=0 (pred minus resid part), v_i>=0
    # constraint a + bx - u_i + v_i = y_i  =>  u = max(pred-y,0)
    # (overestimate, cost 1-tau), v = max(y-pred,0) (cost tau)
    c = np.concatenate([[0.0, 0.0], np.full(n, 1.0 - tau), np.full(n, tau)])
    a_eq = np.zeros((n, 2 + 2 * n))
    a_eq[:, 0] = 1.0
    a_eq[:, 1] = x
    for i in range(n):
        a_eq[i, 2 + i] = -1.0
        a_eq[i, 2 + n + i] = 1.0
    bounds = [(None, None), (None, None)] + [(0.0, None)] * (2 * n)
    res = _opt.linprog(
        c,
        A_eq=a_eq,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError("quantile LP failed")
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    return a_hat, b_hat


def quantile_fit(
    y: FloatArray,
    x: FloatArray,
    taus: FloatArray | None = None,
) -> dict[str, float | FloatArray]:
    """Fit the ABG quantile panel over the tau grid."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.ndim != 1 or yy.size != xx.size or yy.size < 30:
        raise ValueError("bad series")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("non-finite data")
    tt = _TAU_GRID if taus is None else np.asarray(taus, dtype=np.float64)
    if tt.ndim != 1 or np.any(tt <= 0.0) or np.any(tt >= 1.0):
        raise ValueError("bad tau grid")
    coefs = np.empty((tt.size, 2))
    for i, tau in enumerate(tt):
        coefs[i] = _rq_line(yy, xx, float(tau))
    return {"taus": tt, "coefs": coefs, "n": float(yy.size)}


def conditional_quantile_fn(
    fit: dict[str, float | FloatArray],
    x_new: float,
) -> FloatArray:
    """Fitted quantile values Q(tau|x_new) at the grid."""
    coefs = np.asarray(fit["coefs"])
    taus = np.asarray(fit["taus"])
    q = coefs[:, 0] + coefs[:, 1] * x_new
    # isotonic enforcement: quantiles must be nondecreasing in tau
    q = np.maximum.accumulate(q)
    _ = taus
    return q


def _interp_q(taus: FloatArray, q: FloatArray, tau: float) -> float:
    return float(np.interp(tau, taus, q))


def growth_at_risk(
    fit: dict[str, float | FloatArray],
    x_new: float,
    alpha: float = _ALPHA,
) -> dict[str, float]:
    """GaR = expected shortfall below the alpha quantile of y|x.

    Integrates the interpolated quantile function below ``alpha``
    and above ``1-alpha`` for the upside analog; the interquantile
    range and skew measures follow ABG's vulnerability summary.
    """
    taus = np.asarray(fit["taus"])
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5)")
    q = conditional_quantile_fn(fit, x_new)
    lo_grid = np.linspace(1e-4, alpha, 100)
    hi_grid = np.linspace(1.0 - alpha, 1.0 - 1e-4, 100)
    gar = float(np.mean([_interp_q(taus, q, t) for t in lo_grid]))
    upside = float(np.mean([_interp_q(taus, q, t) for t in hi_grid]))
    med = _interp_q(taus, q, 0.5)
    q_lo = _interp_q(taus, q, alpha)
    q_hi = _interp_q(taus, q, 1.0 - alpha)
    return {
        "gar": gar,
        "median": med,
        "q_low": q_lo,
        "q_high": q_hi,
        "upside": upside,
        "iqr_width": q_hi - q_lo,
        "asymmetry": (upside - med) - (med - gar),
    }


def synth_gar(
    seed: int = 20261231 + 300,
    t: int = 600,
    loc_slope: float = -0.8,
    scale_slope: float = 0.9,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC location+scale-shifting DGP for GaR validation."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(t)
    eps = rng.standard_normal(t)
    # asymmetric vulnerability: the downside scale grows with x
    # while the upside scale stays constant (the ABG stylized fact)
    down_scale = 0.3 * (1.0 + scale_slope * np.maximum(x, 0.0))
    sd = np.where(eps < 0.0, np.maximum(down_scale, 0.05), 0.3)
    y = 0.5 + loc_slope * x + sd * eps
    return {
        "y": y,
        "x": x,
        "loc_slope": loc_slope,
        "scale_slope": scale_slope,
    }


def bench_growth_at_risk(seed: int = 20261231 + 300) -> dict[str, float]:
    """Wave-52 self-check: GaR falls and spread widens in x."""
    d = synth_gar(seed=seed)
    fit = quantile_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    g_lo = growth_at_risk(fit, x_new=-1.5)
    g_hi = growth_at_risk(fit, x_new=1.5)
    gar_drop = g_lo["gar"] - g_hi["gar"]
    spread_widen = g_hi["iqr_width"] - g_lo["iqr_width"]
    ok = gar_drop > 1.0 and spread_widen > 0.5
    return {
        "gar_lo": g_lo["gar"],
        "gar_hi": g_hi["gar"],
        "gar_drop": gar_drop,
        "spread_widen": spread_widen,
        "med_lo": g_lo["median"],
        "med_hi": g_hi["median"],
        "score": float(ok),
    }
