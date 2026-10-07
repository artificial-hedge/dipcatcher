"""Davis-Mikosch extremogram + Ferro-Segers extremal index.

References
----------
- Davis, R.A. & Mikosch, T. (2009). "The Extremogram: A
  Correlogram for Extreme Events." *Bernoulli* 15(4), 977-1009.
- Ferro, C.A.T. & Segers, J. (2003). "Inference for Clusters of
  Extreme Values." *JRSS B* 65(2), 545-556.
- Leadbetter, M.R. (1983). "Extremes and Local Dependence in
  Stationary Sequences." *ZWVG* 65, 291-306.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The extremogram is the extremal analogue of the ACF:
``rho_h = P(X_{t+h} > a | X_t > a)`` with ``a`` an empirical
high quantile. The natural estimator
``sum_t 1{x_t>a, x_{t+h}>a} / sum_t 1{x_t>a}`` inherits
boundary bias in the ``h`` tail (the last ``h`` observations
cannot contribute to the numerator) — we restrict the
denominator count to ``n - h`` admissible exceedances so the
ratio stays a proper conditional-frequency estimate. The
extremal index ``theta`` (Leadbetter) measures cluster size:
``theta = 1 / mean(cluster size)``, and the Ferro-Segers
intervals estimator is
``theta = min(1, 2 * sum(S_i)**2 / ((N-1) * sum(S_i**2)))``
over inter-exceedance times ``S_i`` — the ``min`` guards the
super-1 boundary case on short samples. ``tail_dependence``
returns the pairwise lower/upper tail dependence profile
``chi(u) = P(Y > F_y^{-1}(u) | X > F_x^{-1}(u))`` on empirical-
quantile ranks. ``synth_extremogram`` compares a max-AR(1)
Frechet process — whose extremogram decays geometrically and
whose extremal index is ``1 - phi`` — against iid noise. The
bench gates on the AR process showing a geometrically decaying
extremogram and a depressed extremal index, with the iid
series flat near ``1 - q``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 200) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def extremogram(
    x: FloatArray,
    q: float = 0.95,
    max_lag: int = 20,
) -> dict[str, FloatArray]:
    """Davis-Mikosch extremogram up to ``max_lag``."""
    v = _as_series(x)
    if not 0.5 < q < 0.999 or max_lag < 1 or max_lag >= v.size // 4:
        raise ValueError("bad quantile/lag")
    a = float(np.quantile(v, q))
    exc = (v > a).astype(np.float64)
    n = v.size
    lags = np.arange(1, max_lag + 1)
    rho = np.zeros(max_lag)
    for i, h in enumerate(lags):
        num = float(np.sum(exc[: n - h] * exc[h:]))
        den = float(np.sum(exc[: n - h]))
        rho[i] = num / den if den > 0 else np.nan
    out: dict[str, FloatArray] = {
        "lags": lags.astype(np.float64),
        "extremogram": rho,
        "threshold": np.array([a]),
    }
    return out


def extremal_index(x: FloatArray, q: float = 0.95) -> dict[str, float]:
    """Ferro-Segers intervals estimator of the extremal index."""
    v = _as_series(x)
    a = float(np.quantile(v, q))
    exceed_at = np.where(v > a)[0]
    if exceed_at.size < 4:
        raise ValueError("too few exceedances")
    gaps = np.diff(exceed_at).astype(np.float64)
    m = gaps.size
    theta = min(1.0, 2.0 * float(np.sum(gaps)) ** 2 / (m * float(np.sum(gaps**2))))
    cluster_gap = float(np.median(gaps))
    out: dict[str, float] = {
        "theta": theta,
        "n_exceedances": float(exceed_at.size),
        "median_gap": cluster_gap,
        "threshold": a,
    }
    return out


def tail_dependence(
    x: FloatArray,
    y: FloatArray,
    grid: tuple[float, ...] = (0.90, 0.95, 0.975, 0.99),
) -> dict[str, FloatArray]:
    """Empirical upper-tail dependence profile chi(u)."""
    vx = _as_series(x)
    vy = _as_series(y)
    if vx.size != vy.size:
        raise ValueError("length mismatch")
    rx = np.argsort(np.argsort(vx)) / (vx.size + 1.0)
    ry = np.argsort(np.argsort(vy)) / (vy.size + 1.0)
    chi = np.zeros(len(grid))
    for i, u in enumerate(grid):
        den = float(np.sum(rx > u))
        chi[i] = float(np.sum((rx > u) & (ry > u))) / den if den > 0 else np.nan
    out: dict[str, FloatArray] = {
        "u": np.asarray(grid, dtype=np.float64),
        "chi": chi,
    }
    return out


def synth_extremogram(
    seed: int = 20261231 + 336,
    n: int = 8000,
    phi: float = 0.8,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC clustered (max-AR) vs iid Frechet extremes."""
    rng = np.random.default_rng(seed)
    eps = rng.pareto(2.0, size=n) + 1.0
    iid = np.asarray(eps, dtype=np.float64)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = max(phi * x[t - 1], eps[t])
    return x, iid


def bench_extremogram(seed: int = 20261231 + 336) -> dict[str, float]:
    x_ar, x_iid = synth_extremogram(seed=seed)
    g_ar = extremogram(x_ar, q=0.95, max_lag=10)
    g_iid = extremogram(x_iid, q=0.95, max_lag=10)
    idx_ar = extremal_index(x_ar, q=0.95)
    idx_iid = extremal_index(x_iid, q=0.95)
    rho_ar = float(g_ar["extremogram"][0])
    rho_iid = float(g_iid["extremogram"][0])
    spread_ar = float(g_ar["extremogram"][0] - g_ar["extremogram"][5])
    ok = (
        rho_ar > 0.4
        and rho_iid < 0.12
        and idx_ar["theta"] < 0.45
        and idx_iid["theta"] > 0.7
        and spread_ar > 0.25
    )
    out: dict[str, float] = {
        "synthetic_extremogram_lag1_ar": rho_ar,
        "synthetic_extremogram_lag1_iid": rho_iid,
        "synthetic_extremogram_decay_ar": spread_ar,
        "synthetic_extremal_index_ar": idx_ar["theta"],
        "synthetic_extremal_index_iid": idx_iid["theta"],
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
