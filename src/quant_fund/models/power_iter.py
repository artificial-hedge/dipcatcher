"""Power iteration + Wielandt deflation for the top-k eigenpairs of a
symmetric matrix — eigenvalue recovery vs NumPy on planted spectra.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._eig_synth import sym_planted


def _power_deflate(A: np.ndarray, k: int, iters: int, rng) -> np.ndarray:
    A = A.copy()
    vals = []
    n = A.shape[0]
    for _ in range(k):
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(iters):
            w = A @ v
            v = w / np.linalg.norm(w)
            lam = float(v @ A @ v)
        vals.append(lam)
        A = A - lam * np.outer(v, v)
    return np.asarray(vals)


def bench_power_iter(seed: int = 2993, k: int = 4) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A, lam = sym_planted(seed)
    top = np.sort(lam)[-k:][::-1]
    est = _power_deflate(A, k, 300, rng)
    err = float(np.abs(np.sort(est)[::-1] - top).mean())
    eig_true = np.linalg.eigvalsh(A)[-k:][::-1]
    err_np = float(np.abs(est - eig_true).mean())
    return {
        "synthetic_power_topk_err": err,
        "synthetic_power_vs_numpy": err_np,
        "synthetic_torch_available": 0.0,
    }
