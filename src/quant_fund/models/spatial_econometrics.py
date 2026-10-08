"""Spatial econometrics — spatial autoregressive (SAR) regression (SYNTHETIC).

The SAR model ``y = ρ·W·y + X·β + ε`` treats each unit's outcome as
linear in its neighbours' outcomes (spatial spillovers). OLS on y~X is
biased under nonzero ρ; maximum likelihood on the concentrated
log-likelihood in ρ recovers it. Moran's I on residuals tests for
leftover spatial dependence.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure ρ/β recovery on generated lattice
panels — never market evidence.

References:
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
  Kluwer — SAR specification, Lagrange-multiplier diagnostics.
- LeSage, J., Pace, R. K. (2009). *Introduction to Spatial
  Econometrics*. CRC Press — concentrated log-likelihood and
  log-determinant evaluation.
- Ord, J. K. (1975). Estimation methods for models of spatial
  interaction. *JASA* 70, 120-126 — eigendecomposition of W for the
  Jacobian term.
- Cliff, A., Ord, J. K. (1973). *Spatial Autocorrelation*. Pion —
  Moran's I statistic and moments.
- Kelejian, H. H., Prucha, I. R. (1998). A generalized spatial
  two-stage least squares procedure. *J. Real Estate Finance Econ.*

Composition: pure numpy — row-standardized weights, concentrated
likelihood over a ρ-grid refined by golden section; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def row_standardize(weights: FloatArray) -> FloatArray:
    """Row-standardize a symmetric weight matrix (rows sum to 1)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 3:
        raise ValueError("weights: square matrix of size >=3 required")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights: finite nonnegative entries required")
    np.fill_diagonal(w, 0.0)
    rs = w.sum(axis=1)
    if np.any(rs <= 0):
        raise ValueError("weights: every unit needs at least one neighbour")
    return w / rs[:, None]


def lattice_weights(side: int) -> FloatArray:
    """Rook contiguity weights on a ``side x side`` lattice."""
    if side < 2:
        raise ValueError("side >= 2 required")
    n = side * side
    w = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, side)
        if r > 0:
            w[i, i - side] = 1.0
        if r < side - 1:
            w[i, i + side] = 1.0
        if c > 0:
            w[i, i - 1] = 1.0
        if c < side - 1:
            w[i, i + 1] = 1.0
    return row_standardize(w)


def _concentrated_nll(
    rho: float,
    wy_eigs: FloatArray,
    y: FloatArray,
    wy: FloatArray,
    x: FloatArray,
) -> float:
    """Negative concentrated log-likelihood of the SAR model.

    log|I - ρW| = Σ log(1 - ρ·λ_i); profile β̂(ρ) and σ²(ρ) out."""
    resid_t = y - rho * wy
    b = np.linalg.lstsq(x, resid_t, rcond=None)[0]
    e = resid_t - x @ b
    n = y.size
    s2 = float(e @ e / n)
    if s2 <= 0:
        return 1e12
    logdet = float(np.sum(np.log(1.0 - rho * wy_eigs)))
    return float(0.5 * n * math.log(s2) - logdet)


def sar_fit(y: FloatArray, x: FloatArray, weights: FloatArray) -> dict[str, float]:
    """ML fit of the spatial autoregressive model y = ρWy + Xβ + ε.

    Returns ρ, β vector, residual variance, and Moran's I on the
    fitted residuals."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if xx.shape[0] != yy.size:
        xx = xx.T
    n = yy.size
    if n < 8 or xx.shape[0] != n:
        raise ValueError("x row count must equal len(y), n>=8")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite y and x required")
    w = row_standardize(weights)
    if w.shape[0] != n:
        raise ValueError("weights dimension must equal len(y)")

    eigs = np.linalg.eigvals(w).real
    wy = w @ yy
    # feasible interval: 1/lambda_min < rho < 1/lambda_max
    lam_min, lam_max = float(eigs.min()), float(eigs.max())
    lo = -1.0 if lam_min >= -1e-10 else min(1.0 / lam_min, 0.999)
    hi = 1.0 if lam_max <= 1e-10 else min(1.0 / lam_max, 0.999)
    res = minimize_scalar(
        _concentrated_nll,
        bounds=(lo + 1e-6, hi - 1e-6),
        args=(eigs, yy, wy, xx),
        method="bounded",
        options={"xatol": 1e-8},
    )
    rho = float(res.x)
    b = np.linalg.lstsq(xx, yy - rho * wy, rcond=None)[0]
    e = yy - rho * wy - xx @ b
    s2 = float(e @ e / n)

    # OLS benchmark (biased under SAR)
    b_ols = np.linalg.lstsq(xx, yy, rcond=None)[0]

    # Moran's I on residuals
    ec = e - e.mean()
    s0 = float(np.sum(w))
    i_stat = float(n / s0 * (ec @ w @ ec) / (ec @ ec))
    ei = -1.0 / (n - 1)
    var_i = n**2 * 2.0 / (6.0 * (n - 1) * s0**2)  # normality approx
    z_i = (i_stat - ei) / math.sqrt(max(var_i, 1e-12))

    return {
        "rho": rho,
        "sigma2": s2,
        "n": float(n),
        "moran_i": i_stat,
        "moran_z": z_i,
        "moran_p": float(2 * (1 - norm.cdf(abs(z_i)))),
        **{f"beta_{j}": float(v) for j, v in enumerate(b)},
        **{f"beta_ols_{j}": float(v) for j, v in enumerate(b_ols)},
    }


def synth_sar(
    side: int = 8, rho: float = 0.6, beta: float = 1.0, seed: int = 0
) -> dict[str, FloatArray]:
    """SAR DGP on a lattice: y = (I - ρW)^{-1}(x·β + ε)."""
    rng = np.random.default_rng(seed)
    n = side * side
    w = lattice_weights(side)
    x = np.column_stack([np.ones(n), rng.normal(0.0, 1.0, n)])
    eps = rng.normal(0.0, 0.5, n)
    a_inv = np.linalg.inv(np.eye(n) - rho * w)
    y = a_inv @ (x[:, 1] * beta + eps)
    return {"y": y, "x": x, "w": w, "rho_true": np.array([rho])}


def bench_spatial_econometrics(
    seed: int = 20261231 + 209,
) -> dict[str, float]:
    """SAR self-check: ρ recovery, β improvement over OLS, residual
    Moran's I cleared. All ``synthetic_*``."""
    d = synth_sar(side=9, rho=0.6, beta=1.0, seed=seed)
    out = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))
    d0 = synth_sar(side=9, rho=0.0, beta=1.0, seed=seed + 1)
    out0 = sar_fit(np.asarray(d0["y"]), np.asarray(d0["x"]), np.asarray(d0["w"]))
    out_b = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))

    rho = float(out["rho"])
    b1 = float(out["beta_1"])
    b1_ols = float(out["beta_ols_1"])
    return {
        "synthetic_rho_hat": rho,
        "synthetic_rho_err": float(abs(rho - 0.6)),
        "synthetic_beta_err": float(abs(b1 - 1.0)),
        "synthetic_beta_ols_err": float(abs(b1_ols - 1.0)),
        "synthetic_beta_beats_ols": float(abs(b1 - 1.0) < abs(b1_ols - 1.0)),
        "synthetic_moran_z": float(out["moran_z"]),
        "synthetic_null_rho": float(abs(out0["rho"])),
        "synthetic_detects": float(abs(rho - 0.6) < 0.2 and abs(out0["rho"]) < 0.2),
        "synthetic_determinism": float(rho == float(out_b["rho"])),
    }
