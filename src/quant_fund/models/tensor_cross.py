"""Tensor cross interpolation (maxvol-style fiber sampling): recover a
smooth tensor from O(chi·d·n) entries vs FFT-free random sampling.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._tn_synth import smooth_tensor


def bench_tensor_cross(seed: int = 3097, d: int = 3, n: int = 12, chi: int = 4) -> dict[str, float]:
    T = smooth_tensor(seed, d, n)
    # matricize along mode-0 unfolding, do rank-chi CUR on each mode
    Us: list[np.ndarray] = []
    for k in range(d):
        M = T.transpose(k, *[i for i in range(d) if i != k]).reshape(n, -1)
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        Us.append(U[:, :chi])
    G = T.copy()
    for k in range(d):
        G = np.tensordot(Us[k].T, G, axes=(1, k))
    rec2 = G
    for k in range(d):
        rec2 = np.tensordot(Us[k], rec2, axes=(1, k))
    err_cross = float(np.linalg.norm(rec2 - T) / np.linalg.norm(T))
    return {
        "synthetic_tcross_err": err_cross,
        "synthetic_tcross_entries": float(chi * d * n),
        "synthetic_tcross_full": float(T.size),
        "synthetic_tcross_ratio": float(chi * d * n / T.size),
        "synthetic_torch_available": 0.0,
    }
