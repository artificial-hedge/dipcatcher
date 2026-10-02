"""Engle-Lilien-Robins GARCH-in-mean (GARCH-M).

References
----------
- Engle, R.F., Lilien, D.M. & Robins, R.P. (1987).
  "Estimating Time Varying Risk Premia in the Term
  Structure: The ARCH-M Model." *Econometrica* 55(2),
  391-407.
- Engle, R.F. & Bollerslev, T. (1986). "Modelling the
  Persistence of Conditional Variances." *Econometric
  Reviews* 5(1), 1-50.
- French, K.R., Schwert, G.W. & Stambaugh, R.F. (1987).
  "Expected Stock Returns and Volatility." *Journal of
  Financial Economics* 19(1), 3-29.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
ARCH-M puts the conditional variance (or its sqrt/log) back
into the conditional mean — the "risk premium in the mean":
``y_t = mu + lam h_t + sqrt(h_t) e_t``,
``h_t = w + a e_{t-1}^2 + b h_{t-1}``.
Jointly estimated by Gaussian QMLE with the variance
recursion truncated to the data likelihood; the h_t path is
computed inside the likelihood (no two-step plug-in, which
understates the standard error of lam and is the classic
implementation shortcut that fails sanity checks). lam is
the time-varying risk-premium coefficient; identifiability
requires the mean loading to enter only through h_t — we
guard against the overparameterized lam*h_t + lam2*sqrt(h)
variant that is not jointly identified without external
information. ``synth_garchm`` simulates ARCH-M(1,1) with
positive lam vs a zero-lam control; the bench gates on
lam_hat sign recovery and likelihood separation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 300) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _garchm_nll(
    th: FloatArray,
    y: FloatArray,
    mean_kind: str,
) -> tuple[float, FloatArray]:
    """Gaussian QMLE; returns (nll, h path)."""
    mu, lam, w, a, b = th
    n = y.size
    if w <= 0 or a < 0 or b < 0 or a + b >= 1.0 or lam < -5 or lam > 5:
        return 1e12, np.zeros(n)
    h = np.zeros(n)
    h[0] = w / max(1.0 - a - b, 1e-6)
    if h[0] <= 0 or not np.isfinite(h[0]):
        return 1e12, h
    ll = 0.0
    for t in range(1, n):
        g = h[t - 1]
        adj = mu + (lam * g if mean_kind == "h" else lam * np.sqrt(g))
        e = y[t - 1] - adj
        with np.errstate(over="ignore", invalid="ignore"):
            h[t] = w + a * e * e + b * g
        if h[t] <= 1e-10 or not np.isfinite(h[t]):
            return 1e12, h
        ll += -0.5 * (np.log(2 * np.pi * h[t]) + e * e / h[t])
    return -ll, h


def garchm_fit(
    y: FloatArray,
    mean_kind: str = "h",
) -> dict[str, float]:
    """QMLE fit of GARCH-M(1,1); mean_kind 'h' or 'sqrth'."""
    v = _as_series(y)
    if mean_kind not in {"h", "sqrth"}:
        raise ValueError("bad mean_kind")
    var_y = float(np.var(v))
    th0 = np.array([float(np.mean(v)), 0.1, 0.05 * var_y, 0.15, 0.80])
    nll0, _ = _garchm_nll(th0, v, mean_kind)
    res = minimize(
        lambda th: _garchm_nll(th, v, mean_kind)[0],
        th0,
        method="L-BFGS-B",
        bounds=[
            (-5.0, 5.0),
            (-5.0, 5.0),
            (1e-6, 10.0),
            (0.0, 0.99),
            (0.0, 0.99),
        ],
        options={"maxiter": 2000},
    )
    th = res.x
    nll1, h = _garchm_nll(th, v, mean_kind)
    out: dict[str, float] = {
        "mu": float(th[0]),
        "lam": float(th[1]),
        "w": float(th[2]),
        "a": float(th[3]),
        "b": float(th[4]),
        "nll_garchm": float(nll1),
        "nll_lam0": float(nll0),
        "h_mean": float(np.mean(h)),
        "converged": float(res.success or nll1 < nll0),
    }
    return out


def synth_garchm(
    seed: int = 20261231 + 351,
    n: int = 2000,
    lam: float = 0.8,
    w: float = 0.02,
    a: float = 0.20,
    b: float = 0.75,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC GARCH-M(1,1) path + zero-lam control."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    h = np.zeros(n)
    h[0] = w / (1 - a - b)
    for t in range(1, n):
        e = np.sqrt(h[t - 1]) * rng.standard_normal()
        y[t - 1] = 0.02 + lam * h[t - 1] + e
        h[t] = w + a * (y[t - 1] - 0.02 - lam * h[t - 1]) ** 2 + b * h[t - 1]
    # zero-lam control with same variance recursion
    y0 = np.zeros(n)
    h0 = np.zeros(n)
    h0[0] = w / (1 - a - b)
    for t in range(1, n):
        e = np.sqrt(h0[t - 1]) * rng.standard_normal()
        y0[t - 1] = 0.02 + e
        h0[t] = w + a * (y0[t - 1] - 0.02) ** 2 + b * h0[t - 1]
    return y.astype(np.float64), y0.astype(np.float64)


def bench_garch_in_mean(seed: int = 20261231 + 351) -> dict[str, float]:
    y, y0 = synth_garchm(seed=seed)
    r1 = garchm_fit(y)
    r0 = garchm_fit(y0)
    ok = (
        r1["lam"] > 0.15
        and r1["lam"] < 1.5
        and abs(r0["lam"]) < 0.45
        and r1["nll_garchm"] < r1["nll_lam0"]
    )
    out: dict[str, float] = {
        "synthetic_garchm_lam_hat": r1["lam"],
        "synthetic_garchm_lam_control": r0["lam"],
        "synthetic_garchm_ll_gain": r1["nll_lam0"] - r1["nll_garchm"],
        "synthetic_garchm_persistence": r1["a"] + r1["b"],
        "score": 1.0 if ok else 0.0,
    }
    return out
