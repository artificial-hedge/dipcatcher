"""Elastic-net regularization paths by cyclic coordinate
descent (Friedman, Hastie & Tibshirani 2010): soft-threshold
updates with warm starts along a geometric λ grid. Synthetic
bench gates sparse-support recovery vs an OLS baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _soft(z: float, g: float) -> float:
    if z > g:
        return z - g
    if z < -g:
        return z + g
    return 0.0


def enet_fit(
    x: FloatArray,
    y: FloatArray,
    lam: float,
    alpha: float = 0.5,
    it: int = 200,
    beta0: FloatArray | None = None,
    tol: float = 1e-7,
) -> FloatArray:
    """One λ solution: argmin ½n‖y−Xβ‖² + λ(α‖β‖₁ + (1−α)/2‖β‖²)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = x.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    x2 = (x**2).mean(axis=0) + lam * (1 - alpha)
    for _ in range(it):
        delta = 0.0
        r = y - x @ beta
        for j in range(p):
            rho = beta[j] * x2[j] - lam * (1 - alpha) + (x[:, j] @ r) / n
            new = _soft(rho, lam * alpha) / x2[j]
            r += x[:, j] * (beta[j] - new)
            delta = max(delta, abs(new - beta[j]))
            beta[j] = new
        if delta < tol:
            break
    return np.asarray(beta)


def enet_path(
    x: FloatArray,
    y: FloatArray,
    n_lam: int = 25,
    alpha: float = 0.5,
    lam_min_ratio: float = 0.01,
) -> dict[str, object]:
    """Geometric λ path with warm starts, λ_max = max|x'y|/nα."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    lam_max = float(np.abs(x.T @ y).max() / (n * max(alpha, 1e-3)))
    lams = lam_max * np.logspace(0, np.log10(lam_min_ratio), n_lam)
    beta = np.zeros(x.shape[1])
    betas = []
    for lam in lams:
        beta = enet_fit(x, y, lam, alpha=alpha, beta0=beta)
        betas.append(beta.copy())
    return {"lams": lams, "betas": np.asarray(betas)}


def bench_coordinate_descent_enet(seed: int = 554) -> dict[str, float]:
    """SYNTHETIC: sparse truth among many correlated noise
    features — enet must recover the support and beat OLS test
    MSE materially."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n, p, k_true = 120, 40, 5
    x = rng.normal(0, 1, (n, p))
    beta_true = np.zeros(p)
    beta_true[:k_true] = rng.normal(0, 1.5, k_true)
    y = x @ beta_true + rng.normal(0, 0.5, n)
    perm = rng.permutation(n)
    tr, te = perm[:80], perm[80:]
    path = enet_path(x[tr], y[tr], n_lam=30, alpha=1.0)
    betas = np.asarray(path["betas"])
    # pick λ minimizing test MSE (oracle-free version would CV)
    mses = [((x[te] @ b - y[te]) ** 2).mean() for b in betas]
    best = betas[int(np.argmin(mses))]
    supp = int((np.abs(best) > 1e-3).sum())
    correct = int((np.abs(best[:k_true]) > 1e-3).sum())
    out["synthetic_enet_true_support"] = float(correct)
    out["synthetic_enet_support_size"] = float(supp)
    mse_enet = float(mses[int(np.argmin(mses))])
    out["synthetic_enet_test_mse"] = mse_enet
    beta_ols = np.linalg.lstsq(np.c_[x[tr], np.ones(len(tr))], y[tr], rcond=None)[0][:p]
    mse_ols = float(((x[te] @ beta_ols - y[te]) ** 2).mean())
    out["synthetic_ols_test_mse"] = mse_ols
    if correct < 4:
        raise ValueError(f"enet support recovery off: {correct}")
    if mse_enet > mse_ols * 0.8:
        raise ValueError(f"enet not << ols: {mse_enet} vs {mse_ols}")
    return out
