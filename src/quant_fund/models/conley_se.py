"""Conley (1999) spatial HAC standard errors.

OLS with spatially dependent errors needs kernel-weighted
covariance across observations: the meat is a distance-decayed
double sum S = Σ_i Σ_j w(d_ij/L) x_i x_j' e_i e_j with a
Bartlett spatial kernel — collapsing to Newey-West when the
coordinate is time. Implemented for a 1-D coordinate (time or
ordering) and a 2-D (lat, lon) coordinate.

Honesty: synthetic benches generate fields with known spatial
correlation length and check the Conley SE exceeds the OLS SE
in proportion to that length — proper diagnostics, never
market evidence.

References:
- Conley, T. G. (1999). GMM estimation with cross sectional
  dependence. *Journal of Econometrics* 92 — the spatial HAC.
- Conley, T. G. (2008). Spatial econometrics. *The New
  Palgrave Dictionary of Economics* — kernel choices.
- Bester, C. A., Conley, T. G., Hansen, C. B. (2011).
  Inference with dependent data using cluster covariance
  estimators. *Journal of Econometrics* 165.
- Kelejian, H. H., Prucha, I. R. (2007). HAC estimation in a
  spatial framework. *Journal of Econometrics* 140.

Composition: pure numpy — distance kernel quadratic form;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _bartlett(u: FloatArray) -> FloatArray:
    return np.maximum(0.0, 1.0 - np.abs(u))


def conley_vcov(
    x: FloatArray,
    resid: FloatArray,
    coord: FloatArray,
    cutoff: float,
) -> FloatArray:
    """Conley variance of OLS coefs for design x (n,k),
    residuals e (n,), coordinate coord (n,) 1-D or (n,2)."""
    xx = np.asarray(x, dtype=np.float64)
    e = np.asarray(resid, dtype=np.float64)
    c = np.asarray(coord, dtype=np.float64)
    n = xx.shape[0]
    if xx.ndim != 2 or e.shape != (n,) or n < 40:
        raise ValueError("x (n,k), e (n,), n>=40 required")
    if c.ndim == 1:
        c = c[:, None]
    if c.ndim != 2 or c.shape[0] != n or c.shape[1] not in (1, 2):
        raise ValueError("coord (n,) or (n,2) required")
    if cutoff <= 0:
        raise ValueError("positive cutoff required")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(e)):
        raise ValueError("finite inputs required")
    d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
    w = _bartlett(d / cutoff)
    xe = xx * e[:, None]
    meat = xe.T @ w @ xe
    bread = np.linalg.inv(xx.T @ xx)
    return np.asarray(bread @ meat @ bread, dtype=np.float64)


def synth_conley(
    n: int = 400,
    beta: float = 1.0,
    rho_d: float = 30.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y = β x + e on a 1-D line with errors smoothed over
    window rho_d (spatial correlation length)."""
    rng = np.random.default_rng(seed)
    coord = np.sort(rng.uniform(0, n, n))
    k = int(rho_d)
    kern = np.ones(2 * k + 1) / (2 * k + 1) if k else np.ones(1)
    x = np.convolve(rng.normal(0, 1, n), kern, mode="same")
    e = np.convolve(rng.normal(0, 1, n), kern, mode="same")
    y = beta * x + e * 2.0
    return {"y": y, "x": x[:, None], "coord": coord}


def bench_conley_se(seed: int = 20261231 + 257) -> dict[str, float]:
    """Conley self-check: SE inflates ~sqrt(2·rho_d)-fold vs OLS
    under spatial correlation; iid errors leave it near OLS.
    All ``synthetic_*``."""
    d = synth_conley(rho_d=30.0, seed=seed)
    x = np.asarray(d["x"])
    xd = np.column_stack([np.ones(x.shape[0]), x])
    b, *_ = np.linalg.lstsq(xd, np.asarray(d["y"]), rcond=None)
    e = np.asarray(d["y"]) - xd @ b
    vc = conley_vcov(xd, e, np.asarray(d["coord"]), cutoff=60.0)
    se_c = float(np.sqrt(vc[1, 1]))
    se_ols = float(np.sqrt((e @ e) / (e.size - 2) * np.linalg.inv(xd.T @ xd)[1, 1]))
    d0 = synth_conley(rho_d=0.0, seed=seed + 1)
    x0 = np.column_stack([np.ones(d0["x"].shape[0]), np.asarray(d0["x"])])
    b0, *_ = np.linalg.lstsq(x0, np.asarray(d0["y"]), rcond=None)
    e0 = np.asarray(d0["y"]) - x0 @ b0
    v0 = conley_vcov(x0, e0, np.asarray(d0["coord"]), cutoff=60.0)
    se0 = float(np.sqrt(v0[1, 1]))
    se0_ols = float(np.sqrt((e0 @ e0) / (e0.size - 2) * np.linalg.inv(x0.T @ x0)[1, 1]))
    vc_b = conley_vcov(xd, e, np.asarray(d["coord"]), cutoff=60.0)
    ratio = se_c / se_ols
    return {
        "synthetic_se_conley": se_c,
        "synthetic_se_ols": se_ols,
        "synthetic_inflation": ratio,
        "synthetic_inflation_null": se0 / se0_ols,
        "synthetic_detects": float(ratio > 2.0 and se0 / se0_ols < ratio),
        "synthetic_determinism": float(float(np.sqrt(vc_b[1, 1])) == se_c),
    }
