"""VAR-LiNGAM (Hyvärinen et al. 2010) — VAR(1) dynamics + instantaneous
non-Gaussian SEM: fit lag matrix by OLS, then LiNGAM-order the
residuals to recover the instantaneous B0 graph.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.direct_lingam import _mi

FloatArray = NDArray[np.float64]


def _var_sem(seed: int, n: int = 800, d: int = 5) -> tuple[FloatArray, FloatArray, FloatArray]:
    """VAR(1) with instantaneous B0: x_t = B0 x_t + M x_{t-1} + e."""
    rng = np.random.default_rng(seed)
    B0 = np.zeros((d, d))
    B0[0, 1] = 0.8
    B0[0, 2] = -0.6
    B0[2, 3] = 0.9
    B0[1, 4] = 0.5
    M = rng.uniform(-0.3, 0.3, (d, d))
    X = np.zeros((n, d))
    for t in range(1, n):
        e = rng.uniform(-0.8, 0.8, d)
        X[t] = np.linalg.solve(np.eye(d) - B0, M @ X[t - 1] + e)
    return X, B0, M


def bench_var_lingam(seed: int = 2813, trials: int = 4) -> dict[str, float]:
    errs, f1s = [], []
    for t in range(trials):
        X, B0, _M = _var_sem(seed + 7 * t)
        d = X.shape[1]
        # VAR(1) OLS on innovations: x_t = A x_{t-1} + u_t
        A = np.linalg.lstsq(X[:-1], X[1:], rcond=None)[0]
        U = X[1:] - X[:-1] @ A
        # instantaneous effects: u_t = B0 u_t + e (B0 acyclic → order by
        # DirectLiNGAM-style root extraction on U)
        rem = list(range(d))
        ord_hat: list[int] = []
        Uw = U.copy()
        while rem:
            scores = {}
            for i in rem:
                dep = 0.0
                for j in rem:
                    if i == j:
                        continue
                    b = np.polyfit(Uw[:, i], Uw[:, j], 1)[0]
                    r = Uw[:, j] - b * Uw[:, i]
                    dep += abs(np.corrcoef(Uw[:, i], r)[0, 1]) + 0.3 * _mi(Uw[:, i], r)
                scores[i] = dep
            root = min(rem, key=lambda i: scores[i])
            ord_hat.append(root)
            rem.remove(root)
            for j in rem:
                Uw[:, j] = Uw[:, j] - np.polyfit(Uw[:, root], Uw[:, j], 1)[0] * Uw[:, root]
        B_hat = np.zeros((d, d))
        for k in range(1, d):
            j = ord_hat[k]
            ps = ord_hat[:k]
            b = np.linalg.lstsq(U[:, ps], U[:, j], rcond=None)[0]
            B_hat[ps, j] = b
        B_hat[np.abs(B_hat) < 0.15] = 0.0
        sk_t = ((B0 != 0) | (B0.T != 0)).astype(float)
        sk_h = ((B_hat != 0) | (B_hat.T != 0)).astype(float)
        tp = float((sk_t * sk_h).sum())
        f1s.append(
            2 * tp / max(2 * tp + ((1 - sk_t) * sk_h).sum() + (sk_t * (1 - sk_h)).sum(), 1e-9)
        )
        errs.append(float(np.abs(B_hat - B0).max()))
    return {
        "synthetic_varling_skel_f1": float(np.mean(f1s)),
        "synthetic_varling_max_err": float(np.mean(errs)),
        "synthetic_torch_available": 0.0,
    }
