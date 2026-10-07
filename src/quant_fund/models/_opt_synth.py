"""Shared synthetic optimization task for the optimizer canon: (SYNTHETIC)
ill-conditioned quadratic regression via an MLP — optimizer quality =
loss after K steps (and steps to threshold) vs Adam at matched lr grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def opt_task(
    seed: int = 7, n: int = 256, d: int = 16
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Quadratic bowl with condition number ~100 → exposes optimizer
    geometry: y = x'Ax/2 + noise with eigenvalues log-spaced."""
    if n < 1 or d < 2:
        raise ValueError(f"need n>=1 and d>=2 (eigenspectrum needs >=2 modes), got {n},{d}")
    rng = np.random.default_rng(seed)
    q, r_ = np.linalg.qr(rng.standard_normal((d, d)))
    # canonicalize the QR sign convention (LAPACK-arbitrary across BLAS
    # builds): force positive R diagonal so A's spectrum/basis are stable.
    s = np.sign(np.diag(r_))
    s[s == 0] = 1.0
    q = q * s[None, :]
    eigs = np.logspace(-2, 0, d)
    a = q @ np.diag(eigs) @ q.T
    x = rng.standard_normal((n, d))
    y = np.einsum("ni,ij,nj->n", x, a, x) * 0.5 + rng.standard_normal(n) * 0.05
    return x.astype(np.float64), y.astype(np.float64)
