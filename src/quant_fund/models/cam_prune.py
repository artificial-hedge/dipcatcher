"""CAM pruning (Bühlmann et al. 2014) — order by residual-variance
greedy search then prune edges by significance of GAM-style nonlinear
fit; here a two-stage regression-prune implementation. SHD vs corr.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cs_synth import corr_baseline, sem_data, shd

FloatArray = NDArray[np.float64]


def bench_cam_prune(seed: int = 2347, edges: int = 7) -> dict[str, float]:
    X, B = sem_data(seed, edges=edges)
    n, d = X.shape
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # stage 1: greedy ordering by residual variance reduction
    order: list[int] = []
    remaining = list(range(d))
    resid = Xn.copy()
    while remaining:
        best_j, best_r = -1, np.inf
        for j in remaining:
            r = (
                resid[:, j].var()
                if not order
                else (
                    resid[:, j]
                    - resid[:, order] @ np.linalg.lstsq(resid[:, order], resid[:, j], rcond=None)[0]
                ).var()
            )
            if r < best_r:
                best_r, best_j = r, j
        order.append(best_j)
        remaining.remove(best_j)
    # stage 2: regression prune — each var on its predecessors, keep sig coefs
    Bhat = np.zeros((d, d))
    for pos in range(1, d):
        j = order[pos]
        preds = order[:pos]
        Z = np.column_stack([Xn[:, preds], np.ones(n)])
        coef, res, *_ = np.linalg.lstsq(Z, Xn[:, j], rcond=None)
        e = Xn[:, j] - Z @ coef
        se = np.sqrt((e @ e) / max(n - len(preds) - 1, 1) * np.diag(np.linalg.pinv(Z.T @ Z)))
        for k, p in enumerate(preds):
            if abs(coef[k] / max(se[k], 1e-9)) > 2.0:
                Bhat[p, j] = coef[k]
    shd_cam = shd(B, Bhat)
    Bb = corr_baseline(X, edges)
    shd_cb = shd(B, Bb)
    return {
        "synthetic_cam_shd": float(shd_cam),
        "synthetic_corr_shd": float(shd_cb),
        "synthetic_cam_gain": float(shd_cb - shd_cam),
        "synthetic_torch_available": 0.0,
    }
