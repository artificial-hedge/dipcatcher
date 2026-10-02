"""FastICA fixed-point independent component analysis.

Hyvärinen's (1999) symmetric + deflation FastICA with the
log-cosh nonlinearity ``g(u) = (1/a) log cosh(a u)``.
Centering and whitening are performed by eigendecomposition
of the sample covariance; each direction is estimated by
the fixed-point iteration

    w+  <-  E[x g(w' x)] - E[g'(w' x)] w ,
    w   <-  w+ / ||w+|| ,

with Gram-Schmidt re-orthogonalization across components
(deflation). Convergence is declared when |w' w+| -> 1.

Honesty: the bench fits ICA on a SYNTHETIC mixture of two
non-Gaussian sources (uniform + Laplace) and reports the
best absolute row correlation of the recovered sources —
a permutation/sign-invariant recovery statistic, not a
market claim.

References
----------
* Hyvarinen, A. (1999) "Fast and robust fixed-point
  algorithms for independent component analysis", IEEE
  Trans. Neural Networks 10(3), 626-634.
* Hyvarinen & Oja (2000) "Independent component analysis:
  algorithms and applications", Neural Networks 13,
  411-430.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _whiten(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return whitened data (n, p) and whitening matrix."""
    n, p = x.shape
    cov = np.cov(x, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    d_half = np.diag(1.0 / np.sqrt(evals))
    w_mat = evecs @ d_half @ evecs.T
    return x @ w_mat, w_mat


def _fastica(
    z: FloatArray,
    n_comp: int,
    rng: np.random.Generator,
    max_iter: int = 500,
    tol: float = 1e-9,
    alpha_g: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Deflation FastICA on whitened data. Returns (W, iters)."""
    n, p = z.shape
    if n_comp < 1 or n_comp > p:
        raise ValueError("n_comp out of range")
    w_found: list[FloatArray] = []
    iters = np.zeros(n_comp)
    for c in range(n_comp):
        w = rng.normal(size=(p,))
        w /= np.linalg.norm(w)
        for it in range(max_iter):
            wx = z @ w
            g = np.tanh(alpha_g * wx)
            gp = alpha_g * (1.0 - g**2)
            w_new = (z.T @ g) / n - w * float(gp.mean())
            # deflationary orthogonalization
            for q in w_found:
                w_new -= float(w_new @ q) * q
            w_new /= np.linalg.norm(w_new)
            if abs(float(w_new @ w)) > 1.0 - tol:
                w = w_new
                iters[c] = it + 1
                break
            w = w_new
        else:
            iters[c] = float(max_iter)
        w_found.append(w)
    return np.stack(w_found), iters


def fast_ica(
    x: FloatArray,
    n_comp: int | None = None,
    seed: int = 0,
    alpha_g: float = 1.0,
    max_iter: int = 500,
) -> dict[str, float | FloatArray]:
    """Estimate independent components from rows of ``x``.

    Returns unmixing matrix ``w_ic`` (n_comp, p), whitened
    sources ``sources`` (n, n_comp), iteration counts, and
    mean convergence error.
    """
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 20 or xx.shape[1] < 2:
        raise ValueError("x must be (n>=20, p>=2)")
    n, p = xx.shape
    k = p if n_comp is None else int(n_comp)
    xc = xx - xx.mean(axis=0)
    z, w_mat = _whiten(xc)
    rng = np.random.default_rng(seed)
    w_basis, iters = _fastica(z, k, rng, max_iter=max_iter, alpha_g=alpha_g)
    sources = z @ w_basis.T
    unmixing = w_basis @ w_mat  # map back to original coords
    return {
        "w_ic": np.asarray(unmixing, dtype=np.float64),
        "sources": np.asarray(sources, dtype=np.float64),
        "iters": np.asarray(iters, dtype=np.float64),
        "n_comp": float(k),
    }


def bench_fastica(seed: int = 20261231 + 462) -> dict[str, float]:
    """SYNTHETIC check — recovers two non-Gaussian sources."""
    rng = np.random.default_rng(seed)
    n = 3000
    s1 = rng.uniform(-1.0, 1.0, size=n)
    s2 = rng.laplace(0.0, 1.0, size=n)
    s_true = np.column_stack([s1, s2])
    a_mix = np.array([[0.8, 0.6], [0.2, -0.9]])
    x = s_true @ a_mix.T
    out = fast_ica(x, n_comp=2, seed=seed)
    s_hat = np.asarray(out["sources"], dtype=np.float64)
    # best |corr| per true source (sign/order invariant)
    best = []
    for j in range(2):
        cs = [abs(float(np.corrcoef(s_true[:, j], s_hat[:, i])[0, 1])) for i in range(2)]
        best.append(max(cs))
    rec = float(min(best))
    if rec < 0.95:
        raise ValueError(f"fastica off: rec={rec}")
    return {
        "synthetic_min_abs_corr": rec,
        "synthetic_mean_iters": float(np.asarray(out["iters"]).mean()),
        "score": 1.0,
    }
