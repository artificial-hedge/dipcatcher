"""Active subspace discovery — gradient-based dimension reduction.

Estimates C = E[∇f ∇fᵀ] from finite-difference gradients, finds the
dominant eigenspace (the "active subspace"), and projects samples
onto it for a reduced-dimensional sufficient summary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def grad_fd(f, x: FloatArray, h: float = 1e-5) -> FloatArray:
    d = x.size
    g = np.empty(d)
    for k in range(d):
        e = np.zeros(d)
        e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def active_subspace(
    f,
    d: int,
    n_samples: int,
    rng: np.random.Generator,
    h: float = 1e-5,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Estimate the active subspace on the unit cube.

    Returns (eigvals_desc, W (d×d eigenvectors), spectral_gaps
    eigval_i/eigval_{i+1}).
    """
    C = np.zeros((d, d))
    for _ in range(n_samples):
        x = rng.random(d) * 2.0 - 1.0  # domain [-1,1]^d
        g = grad_fd(f, x, h)
        C += np.outer(g, g)
    C /= n_samples
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    gaps = w[:-1] / np.maximum(w[1:], 1e-300)
    return w, v, np.asarray(gaps)


def project_samples(W: FloatArray, x: FloatArray, k: int) -> FloatArray:
    """Map samples onto the first-k active directions."""
    return np.asarray(x) @ W[:, :k]


def bench_active_subspace(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: a 5-D ridge function f = g(w·x) has a 1-D active
    subspace; the recovered direction aligns with w."""
    rng = np.random.default_rng(seed)
    d = 5
    w_true = np.array([1.0, 2, 0.5, -1, 0.3])
    w_true = w_true / np.linalg.norm(w_true)
    out: dict[str, float] = {}

    def f(x: FloatArray) -> float:
        return float(np.sin(3.0 * (w_true @ x)) * np.exp(w_true @ x))

    w, v, gaps = active_subspace(f, d, 300, rng)
    align = abs(float(v[:, 0] @ w_true))
    out["synthetic_as_align"] = align
    out["synthetic_as_gap1"] = float(gaps[0])
    out["synthetic_as_eig1_frac"] = float(w[0] / w.sum())
    out["synthetic_as_1d_ok"] = float(align > 0.99 and gaps[0] > 10)
    # non-ridge: isotropic quadratic → flat spectrum
    w2, v2, gaps2 = active_subspace(lambda x: float(x @ x), d, 200, rng)
    out["synthetic_as_flat_gap"] = float(gaps2[0])
    out["synthetic_as_flat_ok"] = float(gaps2[0] < 5.0)
    return out


if __name__ == "__main__":
    print(bench_active_subspace())
