"""Shared fixture for wave-196 eigen canon — planted-spectrum matrices (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sym_planted(
    seed: int, n: int = 12, spectrum: list[float] | None = None
) -> tuple[FloatArray, FloatArray]:
    """Random orthogonal similarity of a known diagonal spectrum."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    lam = np.asarray(spectrum if spectrum is not None else np.linspace(0.1, 5.0, n))
    if lam.shape[0] != n:
        raise ValueError(f"spectrum has {lam.shape[0]} entries but n={n}")
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(lam) @ Q.T
    return np.asarray((A + A.T) / 2), lam


def nonsym_planted(seed: int, n: int = 10) -> tuple[FloatArray, FloatArray]:
    """Upper-triangular-with-known-eigenvalues nonsymmetric matrix."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    lam = np.sort(rng.uniform(0.5, 4.0, n))[::-1]
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = U @ (np.diag(lam) + np.triu(rng.standard_normal((n, n)) * 0.3, 1)) @ U.T
    return np.asarray(A), np.asarray(lam)


def top_eig_err(lam_hat: np.ndarray, lam_true: np.ndarray) -> float:
    return float(np.abs(np.sort(lam_hat)[-len(lam_true) :][::-1] - np.sort(lam_true)[::-1]).mean())
