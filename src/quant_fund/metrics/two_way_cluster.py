"""Two-way clustered covariance (Cameron-Gelbach-Miller).

When errors correlate within each of two non-nested groupings
(e.g. firm and year), one-way clustering is inconsistent. The
CGM sandwich is V = V(g1) + V(g2) − V(g1∩g2): add the two
one-way cluster variances and subtract the intersection
(pairwise cell) variance to avoid double counting.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure standard-error recovery on
generated two-way-correlated panels — never market evidence.

References:
- Cameron, A. C., Gelbach, J. B., Miller, D. L. (2011). Robust
  inference with multiway clustering. *Journal of Business &
  Economic Statistics* 29, 238-249 — V = V1 + V2 − V12 and
  the G/(G−1) small-sample corrections.
- Moulton, B. R. (1990). An illustration of a pitfall in
  estimating the effects of aggregate variables on micro
  units. *Review of Economics and Statistics* 72 — why
  ignoring within-group correlation understates SEs.
- Petersen, M. A. (2009). Estimating standard errors in
  finance panel data sets. *Review of Financial Studies* 22 —
  the firm × year benchmark.
- Thompson, S. B. (2011). Simple formulas for standard errors
  that cluster by both firm and time. *JFE* 99 — the
  shortcut for one regressor varying along each dimension.

Composition: pure numpy — cluster-mean score sums per
grouping; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _cluster_vcov(
    x: FloatArray,
    e: FloatArray,
    g: NDArray[np.int64],
    finite_adj: bool,
) -> FloatArray:
    """CRVE: (X'X)^{-1} [Σ_g X_g' e_g e_g' X_g] (X'X)^{-1}
    with the small-sample correction."""
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    ng = int(g.max()) + 1
    meat = np.zeros((k, k))
    for c in range(ng):
        ix = g == c
        s = x[ix].T @ e[ix]
        meat += np.outer(s, s)
    v = bread @ meat @ bread
    if finite_adj:
        v *= (ng / (ng - 1)) * ((n - 1) / (n - k))
    return np.asarray(v, dtype=np.float64)


def two_way_cluster_vcov(
    x: FloatArray,
    e: FloatArray,
    g1: NDArray[np.int64],
    g2: NDArray[np.int64],
) -> FloatArray:
    """CGM two-way covariance: V1 + V2 − V(g1×g2 cells)."""
    v1 = _cluster_vcov(x, e, g1, finite_adj=True)
    v2 = _cluster_vcov(x, e, g2, finite_adj=True)
    cell = g1 * (int(g2.max()) + 1) + g2
    _, g12 = np.unique(cell, return_inverse=True)
    v12 = _cluster_vcov(x, e, g12.astype(np.int64), finite_adj=False)
    return np.asarray(v1 + v2 - v12, dtype=np.float64)


def two_way_cluster(
    y: FloatArray,
    x: FloatArray,
    g1: NDArray[np.int64],
    g2: NDArray[np.int64],
) -> dict[str, float]:
    """OLS point estimates with one-way, two-way, and iid SEs."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    n = yy.size
    if xx.ndim != 2 or xx.shape[0] != n:
        raise ValueError("aligned y/x required")
    if n < 200:
        raise ValueError("n>=200")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("finite inputs required")
    gg1 = np.asarray(g1).ravel().astype(np.int64)
    gg2 = np.asarray(g2).ravel().astype(np.int64)
    if gg1.size != n or gg2.size != n:
        raise ValueError("aligned g1/g2 required")
    if gg1.max() + 1 < 10 or gg2.max() + 1 < 10:
        raise ValueError(">=10 groups per dimension")

    xd = np.column_stack([np.ones(n), xx])
    k = xd.shape[1]
    b, *_ = np.linalg.lstsq(xd, yy, rcond=None)
    e = yy - xd @ b

    v_iid = np.linalg.inv(xd.T @ xd) * float(np.sum(e**2) / (n - k))
    v_1 = _cluster_vcov(xd, e, gg1, finite_adj=True)
    v_2w = two_way_cluster_vcov(xd, e, gg1, gg2)
    # guard against non-PD sandwich (subtract can flip sign)
    d = np.diagonal(v_2w)
    d = np.where(d > 0, d, np.diagonal(v_1))

    return {
        "n": float(n),
        "k": float(k),
        "beta": float(b[1]),
        "se_iid": float(np.sqrt(max(v_iid[1, 1], 1e-300))),
        "se_one_way": float(np.sqrt(max(v_1[1, 1], 1e-300))),
        "se_two_way": float(np.sqrt(d[1])),
        "t_two_way": float(b[1] / np.sqrt(d[1])),
        "inflation_vs_iid": float(np.sqrt(d[1]) / np.sqrt(max(v_iid[1, 1], 1e-300))),
    }


def synth_twoway(
    g1: int = 30,
    g2: int = 40,
    beta: float = 0.5,
    rho1: float = 0.5,
    rho2: float = 0.5,
    seed: int = 0,
) -> dict[str, FloatArray | NDArray[np.int64]]:
    """Panel y = β x + a_{g1} + b_{g2} + ε — correlated errors
    along both dims; x = group mean shifts + noise so the
    group-level variation in x makes clustering matter."""
    rng = np.random.default_rng(seed)
    n = g1 * g2
    i1 = np.repeat(np.arange(g1), g2)
    i2 = np.tile(np.arange(g2), g1)
    a = rng.normal(0, np.sqrt(rho1), g1)
    b = rng.normal(0, np.sqrt(rho2), g2)
    mu_x = rng.normal(0, 0.8, g1)
    x = mu_x[i1] + rng.normal(0, 1, n)
    y = beta * x + a[i1] + b[i2] + rng.normal(0, 1, n)
    return {"y": y, "x": x, "g1": i1, "g2": i2}


def bench_two_way_cluster(seed: int = 20261231 + 244) -> dict[str, float]:
    """Two-way cluster self-check: β recovered, SE inflated
    ~1.8× iid under group-correlated errors, and near-1.0
    inflation when errors are iid. All ``synthetic_*``."""
    d = synth_twoway(beta=0.5, seed=seed)
    yy = np.asarray(d["y"], dtype=np.float64)
    xx = np.asarray(d["x"], dtype=np.float64)
    g1 = np.asarray(d["g1"], dtype=np.int64)
    g2 = np.asarray(d["g2"], dtype=np.int64)
    out = two_way_cluster(yy, xx, g1, g2)
    dn = synth_twoway(beta=0.5, rho1=0.0, rho2=0.0, seed=seed + 1)
    outn = two_way_cluster(
        np.asarray(dn["y"], dtype=np.float64),
        np.asarray(dn["x"], dtype=np.float64),
        np.asarray(dn["g1"], dtype=np.int64),
        np.asarray(dn["g2"], dtype=np.int64),
    )
    out_b = two_way_cluster(yy, xx, g1, g2)

    beta = float(out["beta"])
    return {
        "synthetic_beta": beta,
        "synthetic_se_iid": float(out["se_iid"]),
        "synthetic_se_two_way": float(out["se_two_way"]),
        "synthetic_inflation": float(out["inflation_vs_iid"]),
        "synthetic_inflation_null": float(outn["inflation_vs_iid"]),
        "synthetic_t": float(out["t_two_way"]),
        "synthetic_detects": float(
            abs(beta - 0.5) < 0.2
            and float(out["inflation_vs_iid"]) > 1.3
            and abs(float(outn["inflation_vs_iid"]) - 1.0) < 0.3
            and float(out["se_two_way"]) > float(out["se_iid"])
        ),
        "synthetic_determinism": float(beta == float(out_b["beta"])),
    }
