"""Kernel methods: kernel ridge regression on the RBF Gram,
random Fourier features (Rahimi & Recht 2007) approximating
the same kernel linearly, and Nyström low-rank Gram
approximation. Synthetic bench gates nonlinear-fit quality
vs a linear-ridge baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _rbf(x: FloatArray, z: FloatArray, gamma: float) -> FloatArray:
    d2 = ((x[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(np.exp(-gamma * d2))


def krr_fit(
    x: FloatArray, y: FloatArray, lam: float = 0.1, gamma: float = 1.0
) -> dict[str, object]:
    """α = (K + λnI)⁻¹y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    k = _rbf(x, x, gamma)
    alpha = np.linalg.solve(k + lam * n * np.eye(n), y)
    return {"alpha": alpha, "x": x, "gamma": gamma}


def krr_predict(model: dict[str, object], x_new: FloatArray) -> FloatArray:
    x = np.asarray(model["x"])
    alpha = np.asarray(model["alpha"])
    gamma = float(np.asarray(model["gamma"]))
    x_new = np.asarray(x_new, dtype=np.float64)
    return np.asarray(_rbf(x_new, x, gamma) @ alpha)


def rff_map(
    x: FloatArray, n_feat: int = 200, gamma: float = 1.0, seed: int = 0
) -> dict[str, object]:
    """z(x) = √(2/D) cos(ωx + b), ω ~ N(0, 2γI), b ~ U(0,2π)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, np.sqrt(2 * gamma), (x.shape[1], n_feat))
    b = rng.uniform(0, 2 * np.pi, n_feat)
    return {"w": w, "b": b}


def rff_transform(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(model["w"])
    b = np.asarray(model["b"])
    return np.asarray(np.sqrt(2.0 / w.shape[1]) * np.cos(x @ w + b))


def nystrom_fit(
    x: FloatArray, n_landmarks: int = 30, gamma: float = 1.0, seed: int = 0
) -> dict[str, object]:
    """K ≈ C W⁺ C' via landmark subsample eigendecomposition."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], min(n_landmarks, x.shape[0]), replace=False)
    z = x[idx]
    w = _rbf(z, z, gamma)
    vals, vecs = np.linalg.eigh(w)
    vals = np.maximum(vals, 1e-10)
    return {"z": z, "w_sqrt_inv": vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T, "gamma": gamma}


def nystrom_features(model: dict[str, object], x: FloatArray) -> FloatArray:
    """Φ(x) = K(x,Z) W^{-1/2} — row features of the KRR basis."""
    z = np.asarray(model["z"])
    wsi = np.asarray(model["w_sqrt_inv"])
    gamma = float(np.asarray(model["gamma"]))
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(_rbf(x, z, gamma) @ wsi)


def bench_kernel_methods(seed: int = 555) -> dict[str, float]:
    """SYNTHETIC: sinusoid + trend target — KRR/RFF/Nyström
    fits must beat a linear-ridge baseline materially."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 180
    x = np.sort(rng.uniform(0, 6, (n, 1)), axis=0)
    y = np.sin(x[:, 0] * 2.0) + 0.3 * x[:, 0] + rng.normal(0, 0.08, n)
    perm = rng.permutation(n)
    tr, te = perm[:120], perm[120:]
    m = krr_fit(x[tr], y[tr], lam=0.05, gamma=0.8)
    pred = krr_predict(m, x[te])
    mse_krr = float(((y[te] - pred) ** 2).mean())
    out["synthetic_krr_mse"] = mse_krr
    # linear ridge baseline
    xa = np.c_[x[tr], np.ones(len(tr))]
    b = np.linalg.solve(xa.T @ xa + 1e-3 * np.eye(2), xa.T @ y[tr])
    pred_lin = np.c_[x[te], np.ones(len(te))] @ b
    mse_lin = float(((y[te] - pred_lin) ** 2).mean())
    out["synthetic_linridge_mse"] = mse_lin
    # RFF + ridge
    rff = rff_map(x[tr], n_feat=150, gamma=0.8, seed=seed)
    z_tr = rff_transform(rff, x[tr])
    z_te = rff_transform(rff, x[te])
    beta = np.linalg.solve(z_tr.T @ z_tr + 0.1 * np.eye(z_tr.shape[1]), z_tr.T @ y[tr])
    mse_rff = float(((y[te] - z_te @ beta) ** 2).mean())
    out["synthetic_rff_mse"] = mse_rff
    # Nyström + ridge on features
    ny = nystrom_fit(x[tr], n_landmarks=40, gamma=0.8, seed=seed)
    f_tr = nystrom_features(ny, x[tr])
    f_te = nystrom_features(ny, x[te])
    bny = np.linalg.solve(f_tr.T @ f_tr + 0.1 * np.eye(f_tr.shape[1]), f_tr.T @ y[tr])
    mse_ny = float(((y[te] - f_te @ bny) ** 2).mean())
    out["synthetic_nystrom_mse"] = mse_ny
    if mse_krr > mse_lin * 0.5:
        raise ValueError(f"krr not << linear: {mse_krr} vs {mse_lin}")
    if mse_rff > mse_krr * 3:
        raise ValueError(f"rff far off krr: {mse_rff} vs {mse_krr}")
    if mse_ny > mse_krr * 3:
        raise ValueError(f"nystrom far off krr: {mse_ny} vs {mse_krr}")
    return out
