"""Distribution regression — conditional CDF via link regressions.

Instead of modeling quantiles one at a time, distribution regression
fits ``F(y|x) = Λ(x'β(y))`` for each threshold y in a grid, where Λ
is a link CDF (logit or probit). The whole conditional distribution —
and any functional of it (quantiles, CDF slices, Lorenz pieces) —
follows, at each x, from the β(y) path.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure distributional recovery on
generated panels — never market evidence.

References:
- Chernozhukov, V., Fernández-Val, I., Melly, B. (2013). Inference
  on counterfactual distributions. *Econometrica* 81, 2205-2268 —
  distribution-regression framework and uniform inference.
- Chernozhukov, V., Fernández-Val, I., Melly, B. (2020). Fast
  algorithms for the quantile regression process. *Empirical
  Economics* — computational shortcuts.
- Foresi, S., Peracchi, F. (1995). The conditional distribution of
  excess returns: an empirical analysis. *JASA* 90, 451-466 — the
  original per-threshold logit construction.
- Peracchi, F. (2002). On estimating conditional quantiles and
  distribution functions. *Computational Statistics & Data
  Analysis* 38.

Composition: per-threshold logit MLE via scipy BFGS on the binary
cross-entropy; monotone rearrangement of the estimated CDF;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

FloatArray = NDArray[np.float64]


def _as_xy(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if xx.shape[0] != yy.size:
        xx = xx.T
    if yy.size < 30 or xx.shape[0] != yy.size:
        raise ValueError("x row count must equal len(y), n>=30")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite y and x required")
    return yy, xx


def _logit_nll_factory(
    ind: FloatArray, xx: FloatArray
) -> tuple[Callable[[FloatArray], float], Callable[[FloatArray], FloatArray]]:
    """Binary cross-entropy nll over coef b (including intercept)."""

    def nll(b: FloatArray) -> float:
        eta = xx @ b
        p = np.clip(expit(eta), 1e-10, 1 - 1e-10)
        return float(-np.sum(ind * np.log(p) + (1 - ind) * np.log(1 - p)))

    def grad(b: FloatArray) -> FloatArray:
        p = np.asarray(expit(xx @ b), dtype=np.float64)
        return np.asarray(xx.T @ (p - ind), dtype=np.float64)

    return nll, grad


def distribution_regression(
    y: FloatArray,
    x: FloatArray,
    thresholds: FloatArray | None = None,
    link: str = "logit",
) -> dict[str, float | FloatArray]:
    """Fit per-threshold regressions; return the β(t) path plus
    conditional CDF/quantile evaluation machinery as floats.

    ``thresholds`` default = inner 10th-90th percentile grid of y.
    Only logit link implemented (Chernozhukov et al. default)."""
    if link != "logit":
        raise ValueError("only the logit link is implemented")
    yy, xx = _as_xy(y, x)
    n = yy.size
    xx1 = np.column_stack([np.ones(n), xx])
    k = xx1.shape[1]
    if thresholds is None:
        taus = np.quantile(yy, np.linspace(0.1, 0.9, 15))
    else:
        taus = np.asarray(thresholds, dtype=np.float64).ravel()
    if taus.size < 5 or not np.all(np.isfinite(taus)):
        raise ValueError("need >=5 finite thresholds")
    if np.unique(taus).size != taus.size:
        raise ValueError("thresholds must be distinct")

    betas = np.zeros((taus.size, k))
    for i, t in enumerate(taus):
        ind = (yy <= t).astype(np.float64)
        p_edge = float(ind.mean())
        if not 0.02 < p_edge < 0.98:
            raise ValueError("threshold too extreme — empty cell")
        nll, grad = _logit_nll_factory(ind, xx1)
        res = minimize(nll, np.zeros(k), jac=grad, method="BFGS")
        if not np.all(np.isfinite(res.x)):
            raise ValueError("threshold regression failed to converge")
        betas[i] = res.x

    # monotone-rearranged CDF on a reference x* (x = 0 in raw units)
    xref = np.zeros(k)
    xref[0] = 1.0
    cdf_ref = expit(betas @ xref)
    cdf_mono = np.maximum.accumulate(cdf_ref)

    # conditional median from the estimated CDF
    med = float(np.interp(0.5, cdf_mono, taus))
    # slope signal: mean |dβ1/dt| measures how the x effect shifts
    # across the distribution (0 for pure location data).
    # β₁(t) path slope: under pure-location (homoskedastic) DGPs the
    # logit-CDF coefficient steepens along t; heteroskedastic scale
    # flattens it. Abs value is the detection signal.
    b1_slope = float(np.polyfit(taus, betas[:, 1], 1)[0])
    grad_var = float(np.var(np.gradient(betas[:, 1], taus)))
    grad_mean = float(np.mean(np.abs(np.gradient(betas[:, 1], taus))))

    return {
        "n": float(n),
        "n_thresholds": float(taus.size),
        "median_at_ref": med,
        "beta_path_var": float(np.var(betas[:, 1])),
        "beta1_t_slope": b1_slope,
        "mean_abs_beta1": float(np.mean(np.abs(betas[:, 1]))),
        "grad_var": grad_var,
        "grad_mean_abs": grad_mean,
        "cdf_monotone": float(np.all(np.diff(cdf_mono) >= -1e-9)),
        "mean_cdf_slope": float(np.mean(np.diff(cdf_mono))),
        "taus": taus,
        "betas": betas,
    }


def synth_dist_reg(n: int = 2000, hetero: bool = True, seed: int = 0) -> dict[str, FloatArray]:
    """Distribution-regression DGP: y = x·β + (1 + c·|x|)·ε with
    heteroskedastic scale when ``hetero`` — larger |x| widens the
    spread, so the logit-CDF slope in x falls along the threshold
    grid (β₁(t) decreases in t)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    eps = rng.normal(0.0, 0.5, n)
    scale = (0.3 + 1.6 * np.abs(x)) if hetero else np.ones(n)
    y = 0.8 * x + scale * eps
    return {"y": y, "x": x, "hetero": np.array([float(hetero)])}


def bench_distribution_regression(
    seed: int = 20261231 + 212,
) -> dict[str, float]:
    """Distribution-regression self-check: the β₁(t) path is flat
    under |x|-driven heteroskedasticity (scale absorbs the x signal)
    and steep under pure-location homoskedasticity. All
    ``synthetic_*``."""
    d = synth_dist_reg(hetero=True, seed=seed)
    out = distribution_regression(np.asarray(d["y"]), np.asarray(d["x"]))
    d0 = synth_dist_reg(hetero=False, seed=seed + 1)
    out0 = distribution_regression(np.asarray(d0["y"]), np.asarray(d0["x"]))
    out_b = distribution_regression(np.asarray(d["y"]), np.asarray(d["x"]))

    sl = float(abs(cast(float, out["beta1_t_slope"])))
    sl0 = float(abs(cast(float, out0["beta1_t_slope"])))
    mb = float(cast(float, out["mean_abs_beta1"]))
    mb0 = float(cast(float, out0["mean_abs_beta1"]))
    return {
        "synthetic_b1_slope_hetero": sl,
        "synthetic_b1_slope_homo": sl0,
        "synthetic_mean_b1_hetero": mb,
        "synthetic_mean_b1_homo": mb0,
        "synthetic_mean_b1_ratio": float(mb0 / max(mb, 1e-9)),
        "synthetic_homo_ratio": float(sl0 / max(sl, 1e-9)),
        "synthetic_median": float(cast(float, out["median_at_ref"])),
        "synthetic_cdf_monotone": float(cast(float, out["cdf_monotone"])),
        "synthetic_beta_path_var": float(cast(float, out["beta_path_var"])),
        "synthetic_detects": float(mb0 > 1.5 * mb),
        "synthetic_determinism": float(sl == float(abs(cast(float, out_b["beta1_t_slope"])))),
    }
