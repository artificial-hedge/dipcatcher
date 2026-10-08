"""Binscatter — Cattaneo, Crump, Farrell, Feng (2024) (SYNTHETIC).

The canonical conditional-expectation plot: partition x into
J equal-mass (quantile) bins and replace the cloud by per-bin
means (x̄_j, ȳ_j). Nonparametric binning exposes functional-
form misspecification that a global linear/quadratic fit
smooths over; a formal check compares bin means to the
parametric fit at bin centers.

Honesty: synthetic data generate a piecewise-linear CEF; the
bench verifies the linear-fit rejection and the spline-fit
acceptance — proper diagnostics, never market evidence.

References:
- Cattaneo, M. D., Crump, R. K., Farrell, M. H., Feng, Y.
  (2024). On binscatter. *American Economic Review* 114 —
  the estimator, binning choices and specification tests.
- Cattaneo, M. D., Farrell, M. H. (2013). Optimal
  convergence rates, Bahadur representation, and asymptotic
  normality of partitioning estimators. *Journal of
  Econometrics* 174 — partitioning theory.
- Starr, E., Goldfarb, B. (2020). Binned scatterplots: a
  simple tool to make research easier and better. *Strategic
  Management Journal* 41 — practitioner guidance.
- Card, D., Lee, D. S., Pei, Z., Weber, A. (2015). Inference
  on causal effects in a generalized regression kink design.
  *Econometrica* 83 — binning for identification.

Composition: pure numpy — quantile binning + OLS basis;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2 as chi2_dist

FloatArray = NDArray[np.float64]


def _quantile_bins(x: FloatArray, j: int) -> FloatArray:
    qs = np.quantile(x, np.linspace(0, 1, j + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    return np.asarray(qs, dtype=np.float64)


def binscatter(
    x: FloatArray,
    y: FloatArray,
    j: int = 20,
) -> dict[str, FloatArray]:
    """Per-bin means: returns bin centers x̄_j, means ȳ_j and
    counts."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    n = xx.shape[0]
    if yy.shape != (n,) or xx.ndim != 1 or n < 50:
        raise ValueError("matched 1-D arrays, n>=50 required")
    if not (5 <= j <= n // 4):
        raise ValueError("j in [5, n/4] required")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("finite inputs required")
    edges = _quantile_bins(xx, j)
    idx = np.clip(np.digitize(xx, edges) - 1, 0, j - 1)
    bx = np.zeros(j)
    by = np.zeros(j)
    cnt = np.zeros(j)
    for b in range(j):
        sel = idx == b
        if sel.any():
            bx[b] = xx[sel].mean()
            by[b] = yy[sel].mean()
            cnt[b] = sel.sum()
    keep = cnt > 0
    out: dict[str, FloatArray] = {
        "x_bar": bx[keep],
        "y_bar": by[keep],
        "count": cnt[keep],
        "j": np.asarray(float(keep.sum())),
    }
    return out


def binspec_test(
    x: FloatArray,
    y: FloatArray,
    degree: int = 1,
    j: int = 20,
) -> dict[str, float]:
    """Specification check: regress y on poly-degree x, then
    test whether bin means deviate from the fit — Wald-type
    chi-square on the J residuals (rough, documented
    approximation)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if degree < 1 or degree > 6:
        raise ValueError("degree 1..6 required")
    bs = binscatter(xx, yy, j)
    jj = int(bs["j"])
    # parametric fit on full data
    coef = np.polyfit(xx, yy, degree)
    resid = yy - np.polyval(coef, xx)
    s2 = float(np.var(resid))
    # bin-mean deviations from fit, scaled by bin counts
    pred_b = np.polyval(coef, np.asarray(bs["x_bar"]))
    d = np.asarray(bs["y_bar"]) - pred_b
    chi2 = float(np.sum(d**2 * np.asarray(bs["count"]) / s2))
    # crude df = j - degree - 1
    df = max(1, jj - degree - 1)
    p = float(chi2_dist.sf(chi2, df))
    return {"chi2": chi2, "df": float(df), "p": p, "j": float(jj)}


def synth_cef(
    n: int = 3000,
    kind: str = "kink",
    seed: int = 0,
) -> dict[str, FloatArray]:
    """CEF variants: 'kink' piecewise-linear (fails linear
    spec), 'linear' clean, 'quad' smooth quadratic."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    if kind == "kink":
        mu = np.where(x > 0, 1.0 + 0.2 * x, 1.0 + 2.0 * x)
    elif kind == "linear":
        mu = 1.0 + 0.8 * x
    elif kind == "quad":
        mu = 1.0 + 0.5 * x + 0.6 * x**2
    else:
        raise ValueError("unknown kind")
    y = mu + rng.normal(0, 1.0, n)
    return {"x": x, "y": y}


def bench_binscatter(seed: int = 20261231 + 267) -> dict[str, float]:
    """Binscatter self-check: linear spec rejected on a kinked
    CEF (p≈0), accepted on a linear CEF (p large); quad spec
    accepted on a quadratic CEF. All ``synthetic_*``."""
    dk = synth_cef(kind="kink", seed=seed)
    p_kink = binspec_test(np.asarray(dk["x"]), np.asarray(dk["y"]), degree=1)["p"]
    dl = synth_cef(kind="linear", seed=seed + 1)
    p_lin = binspec_test(np.asarray(dl["x"]), np.asarray(dl["y"]), degree=1)["p"]
    dq = synth_cef(kind="quad", seed=seed + 2)
    p_quad = binspec_test(np.asarray(dq["x"]), np.asarray(dq["y"]), degree=2)["p"]
    bs = binscatter(np.asarray(dk["x"]), np.asarray(dk["y"]), j=10)
    p2 = binspec_test(np.asarray(dk["x"]), np.asarray(dk["y"]), degree=1)["p"]
    return {
        "synthetic_p_linear_on_kink": p_kink,
        "synthetic_p_linear_on_linear": p_lin,
        "synthetic_p_quad_on_quad": p_quad,
        "synthetic_bins": float(np.asarray(bs["y_bar"]).size),
        "synthetic_detects": float(p_kink < 0.01 and p_lin > 0.05 and p_quad > 0.05),
        "synthetic_determinism": float(p2 == p_kink),
    }
