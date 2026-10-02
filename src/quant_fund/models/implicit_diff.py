"""Implicit function theorem differentiation — through the ridge
fixed-point z*(θ) = (θI + A^T A)^{-1} A^T b: dz/dθ via IFT vs finite
difference on θ.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._da_synth import grad_corr


def bench_implicit_diff(seed: int = 2405) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((8, 6))
    b = rng.standard_normal(8)
    th = 1.0
    AtA = A.T @ A

    def z_star(t: float) -> np.ndarray:
        return np.linalg.solve(t * np.eye(6) + AtA, A.T @ b)

    z = z_star(th)
    # IFT: (θI + AtA) dz/dθ = -z → dz = -M^{-1} z
    dz_ift = -np.linalg.solve(th * np.eye(6) + AtA, z)
    eps = 1e-5
    dz_fd = (z_star(th + eps) - z_star(th - eps)) / (2 * eps)
    return {
        "synthetic_ift_corr": grad_corr(dz_ift, dz_fd),
        "synthetic_ift_max_err": float(np.abs(dz_ift - dz_fd).max()),
        "torch_available": 0.0,
    }
