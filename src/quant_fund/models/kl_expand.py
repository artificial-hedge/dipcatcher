"""Karhunen–Loève expansion of a Gaussian process.

Eigen-decomposes the covariance matrix sampled on a grid — the KL
basis are the eigenfunctions weighted by √eigenvalues; projecting a
path onto the basis yields uncorrelated coefficients. Supports
sampling, truncation, and energy-based order selection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def kl_decompose(cov: FloatArray) -> tuple[FloatArray, FloatArray]:
    """KL decomposition of a covariance sampled on an n-point grid.

    Returns (eigvals_desc, eigenvectors columns).
    """
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def kl_energy_order(eigvals: FloatArray, frac: float = 0.99) -> int:
    """Smallest order capturing `frac` of total variance."""
    tot = float(np.sum(eigvals))
    cum = np.cumsum(eigvals) / tot
    return int(np.searchsorted(cum, frac) + 1)


def kl_sample(
    eigvals: FloatArray,
    eigvecs: FloatArray,
    order: int,
    rng: np.random.Generator,
    mean: FloatArray | None = None,
) -> FloatArray:
    """Sample a path from the truncated KL expansion."""
    z = rng.normal(size=order)
    path = eigvecs[:, :order] @ (np.sqrt(np.clip(eigvals[:order], 0, None)) * z)
    if mean is not None:
        path = path + mean
    return np.asarray(path, dtype=np.float64)


def kl_reconstruct(
    path: FloatArray,
    eigvecs: FloatArray,
    order: int,
    mean: FloatArray | None = None,
) -> FloatArray:
    """Project a path onto the top-`order` KL basis and reconstruct."""
    x = np.asarray(path) - (mean if mean is not None else 0.0)
    coefs = eigvecs[:, :order].T @ x
    rec = eigvecs[:, :order] @ coefs
    if mean is not None:
        rec = rec + mean
    return np.asarray(rec, dtype=np.float64)


def bench_kl_expand(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: exponential-kernel GP — eigenvalue decay, 99% energy
    order, and reconstruction fidelity of sampled paths."""
    rng = np.random.default_rng(seed)
    n = 400
    t = np.linspace(0, 1, n)
    cov = np.exp(-np.abs(t[:, None] - t[None, :]) / 0.2)
    w, v = kl_decompose(cov)
    out: dict[str, float] = {}
    order99 = kl_energy_order(w, 0.99)
    out["synthetic_kl_order99"] = float(order99)
    out["synthetic_kl_eig1_frac"] = float(w[0] / w.sum())
    path = kl_sample(w, v, order99, rng)
    rec = kl_reconstruct(path, v, order99)
    out["synthetic_kl_rec_err"] = float(np.linalg.norm(path - rec) / np.linalg.norm(path))
    # sample covariance of KL coefficients ≈ I
    nsm = 800
    coefs = np.empty((nsm, order99))
    for i in range(nsm):
        z = rng.normal(size=order99)
        coefs[i] = z * np.sqrt(np.clip(w[:order99], 0, None))
    empirical = np.cov(coefs.T)
    target = np.diag(np.clip(w[:order99], 0, None))
    out["synthetic_kl_coef_cov_err"] = float(
        np.linalg.norm(empirical - target) / np.linalg.norm(target)
    )
    out["synthetic_kl_valid"] = float(order99 < n and out["synthetic_kl_coef_cov_err"] < 0.15)
    return out


if __name__ == "__main__":
    print(bench_kl_expand())
