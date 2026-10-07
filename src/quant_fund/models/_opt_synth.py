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
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = np.logspace(-2, 0, d)
    a = q @ np.diag(eigs) @ q.T
    x = rng.standard_normal((n, d))
    y = np.einsum("ni,ij,nj->n", x, a, x) * 0.5 + rng.standard_normal(n) * 0.05
    return x.astype(np.float64), y.astype(np.float64)
