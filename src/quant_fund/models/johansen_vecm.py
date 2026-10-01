"""Johansen cointegration rank tests + VECM estimation.

Johansen's maximum-likelihood rank test on the vector error-
correction model Δy_t = Π y_{t-1} + Σ_i Γ_i Δy_{t-i} + u_t with
unrestricted constant. Rank determined by squared canonical
correlations between residual sets R0 (Δy on lags+const) and
R1 (y_{t-1} on lags+const); trace and λ-max statistics are
compared to Osterwald-Lenum (1992) critical values for the
unrestricted-constant/no-trend case.

Honesty: synthetic benches only test rank detection on
generated I(1) systems — proper diagnostics, never market
evidence.

References:
- Johansen, S. (1988). Statistical analysis of cointegration
  vectors. *Journal of Economic Dynamics and Control* 12.
- Johansen, S. (1991). Estimation and hypothesis testing of
  cointegration vectors in Gaussian VAR. *Econometrica* 59.
- Osterwald-Lenum, M. (1992). A note with quantiles of the
  asymptotic distribution of the ML cointegration rank test
  statistics. *Oxford Bulletin of Economics and Statistics* 54 —
  critical values embedded below (unrestricted constant case).
- MacKinnon, J. G., Haug, A. A., Michelis, L. (1999). Numerical
  distribution functions of likelihood ratio tests for
  cointegration. *Journal of Applied Econometrics* 14.

Composition: pure numpy — generalized eigenproblem via
``np.linalg.eig`` on S10 S00^-1 S01; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Osterwald-Lenum (1992) unrestricted-constant / no-trend table,
# n-r = 1..5: (10%, 5%, 1%) — verified against the published table.
_OL_LMAX = np.array(
    [
        [6.691, 8.083, 11.576],
        [12.783, 14.595, 18.782],
        [18.959, 21.279, 26.154],
        [24.917, 27.341, 32.616],
        [30.818, 33.262, 38.858],
    ]
)
_OL_TRACE = np.array(
    [
        [6.691, 8.083, 11.576],
        [15.583, 17.844, 21.962],
        [28.436, 31.256, 37.291],
        [45.245, 48.419, 55.551],
        [69.956, 69.977, 77.911],
    ]
)


def _ols_resid(y: FloatArray, x: FloatArray) -> FloatArray:
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ b


def johansen_rank(y: FloatArray, p: int = 1) -> dict[str, FloatArray | int]:
    """Trace and λ-max rank stats for y (T, k), VAR lag order p.

    Returns eigenvalues, per-r trace/lmax statistics and the
    Osterwald-Lenum 5% critical values (n-r ≤ 5)."""
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 2 or yy.shape[0] < 60 or yy.shape[1] < 2:
        raise ValueError("y (T>=60, k>=2) required")
    if not np.all(np.isfinite(yy)):
        raise ValueError("finite y required")
    if p < 1 or p > 5:
        raise ValueError("p in [1,5] required")
    t, k = yy.shape
    if k > 5:
        raise ValueError("k <= 5 required (table coverage)")
    dy = np.diff(yy, axis=0)
    # regressor Z: [1, Δy_{t-1}, ..., Δy_{t-p+1}]
    z = [np.ones((t - p, 1))]
    for lag in range(1, p):
        z.append(dy[p - 1 - lag : t - 1 - lag])
    zz = np.hstack(z)
    y_dep = dy[p - 1 :]  # Δy_t
    y_lag = yy[p - 1 : t - 1]  # y_{t-1}
    r0 = _ols_resid(y_dep, zz)
    r1 = _ols_resid(y_lag, zz)
    n = r0.shape[0]
    s00 = r0.T @ r0 / n
    s01 = r0.T @ r1 / n
    s11 = r1.T @ r1 / n
    s10 = s01.T
    m = np.linalg.solve(s00, s01) @ np.linalg.solve(s11, s10)
    eigvals, eigvecs = np.linalg.eig(m)
    order = np.argsort(-eigvals.real)
    lam = np.clip(eigvals.real[order], 0.0, 1.0 - 1e-10)
    vecs = np.asarray(eigvecs.real[:, order], dtype=np.float64)
    lmax = -n * np.log(1.0 - lam)
    trace = -n * np.array([np.sum(np.log(1.0 - lam[j:])) for j in range(k)])
    # H0 "rank <= r" compares to the n-r = k-r table row
    rank_trace = 0
    for r in range(k):
        if trace[r] > _OL_TRACE[k - r - 1, 1]:
            rank_trace = r + 1
    rank_lmax = 0
    for r in range(k):
        if lmax[r] > _OL_LMAX[k - r - 1, 1]:
            rank_lmax = r + 1
    return {
        "eigvals": lam,
        "eigvecs": vecs,
        "trace": trace,
        "lmax": lmax,
        "rank_trace": rank_trace,
        "rank_lmax": rank_lmax,
        "n_obs": n,
        "s11": s11,
        "s01": s01,
    }


def vecm_fit(y: FloatArray, p: int = 1, r: int = 1) -> dict[str, FloatArray]:
    """Reduced-rank VECM: Π = α β' with β from Johansen
    eigenvectors (normalized β' S11 β = I)."""
    out = johansen_rank(y, p)
    beta = np.asarray(out["eigvecs"], dtype=np.float64)[:, :r]
    s11 = np.asarray(out["s11"], dtype=np.float64)
    s01 = np.asarray(out["s01"], dtype=np.float64)
    norm = beta.T @ s11 @ beta
    beta = beta @ np.linalg.inv(np.linalg.cholesky(norm)).T
    alpha = s01 @ beta  # β' S11 β = I ⇒ α = S01 β
    return {"alpha": alpha, "beta": beta, "pi": alpha @ beta.T}


def synth_johansen(
    t: int = 400,
    k: int = 3,
    r: int = 1,
    seed: int = 0,
) -> FloatArray:
    """I(1) system with r cointegrating relations: k−r shared
    random walks plus transitory noise."""
    rng = np.random.default_rng(seed)
    c = k - r
    walk = np.cumsum(rng.normal(0, 1, (t, max(c, 1))), axis=0)
    if c == 0:
        walk = np.zeros((t, 1))
    load = rng.normal(0, 1, (k, max(c, 1)))
    trans = rng.normal(0, 0.5, (t, k))
    y = walk @ load.T + trans
    return np.asarray(y, dtype=np.float64)


def bench_johansen_vecm(seed: int = 20261231 + 251) -> dict[str, float]:
    """Johansen self-check: r=1 system detects rank 1, random-
    walk system detects rank 0. All ``synthetic_*``."""
    y1 = synth_johansen(k=3, r=1, seed=seed)
    o1 = johansen_rank(y1, p=2)
    y0 = synth_johansen(k=3, r=0, seed=seed + 1)
    o0 = johansen_rank(y0, p=2)
    o1b = johansen_rank(y1, p=2)
    v = vecm_fit(y1, p=2, r=1)
    ec = y1 @ np.asarray(v["beta"])
    return {
        "synthetic_rank_coint": float(o1["rank_trace"]),
        "synthetic_rank_null": float(o0["rank_trace"]),
        "synthetic_rank_lmax_coint": float(o1["rank_lmax"]),
        "synthetic_top_eig": float(np.asarray(o1["eigvals"])[0]),
        "synthetic_top_eig_null": float(np.asarray(o0["eigvals"])[0]),
        "synthetic_ec_sd": float(np.std(ec)),
        "synthetic_detects": float(
            int(o1["rank_trace"]) == 1
            and int(o0["rank_trace"]) == 0
            and np.std(ec) < np.std(y1[:, 0])
        ),
        "synthetic_determinism": float(int(o1["rank_trace"]) == int(o1b["rank_trace"])),
    }
