"""DirectLiNGAM (Shimizu et al. 2011) — repeatedly extract the root
variable as the one least dependent on residuals of regressions on all
others (pairwise independence measure via residual correlation +
mutual-information-lite); order error + skeleton F1 vs baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._csl_synth import corr_baseline_order, lingam_sem, order_err, skeleton_f1


def _mi(x: np.ndarray, y: np.ndarray) -> float:
    """Simple histogram MI estimate."""
    hx = np.histogram(x, 20)[0] + 1e-9
    hy = np.histogram(y, 20)[0] + 1e-9
    hxy = np.histogram2d(x, y, 20)[0] + 1e-9
    px = hx / hx.sum()
    py = hy / hy.sum()
    pxy = hxy / hxy.sum()
    return float((pxy * np.log(pxy / (px[:, None] * py[None]))).sum())


def bench_direct_lingam(seed: int = 2807, trials: int = 4, d: int = 6) -> dict[str, float]:
    f1s, oes, oes_b = [], [], []
    for t in range(trials):
        X, B, order = lingam_sem(seed + 5 * t, d=d)
        rem = list(range(d))
        order_hat: list[int] = []
        Xw = X.copy()
        while rem:
            # root = var minimizing Σ_j dependence(x_i, resid of x_j on x_i)
            scores = {}
            for i in rem:
                dep = 0.0
                for j in rem:
                    if i == j:
                        continue
                    b = np.polyfit(Xw[:, i], Xw[:, j], 1)[0]
                    r = Xw[:, j] - b * Xw[:, i]
                    dep += abs(np.corrcoef(Xw[:, i], r)[0, 1]) + 0.3 * _mi(Xw[:, i], r)
                scores[i] = dep
            root = min(rem, key=lambda i: scores[i])
            order_hat.append(root)
            rem.remove(root)
            # regress root out of remaining
            for j in rem:
                Xw[:, j] = Xw[:, j] - np.polyfit(Xw[:, root], Xw[:, j], 1)[0] * Xw[:, root]
        # B_hat: OLS of each var on its predecessors in order_hat
        B_hat = np.zeros((d, d))
        for k in range(1, d):
            j = order_hat[k]
            ps = order_hat[:k]
            b = np.linalg.lstsq(X[:, ps], X[:, j], rcond=None)[0]
            B_hat[ps, j] = b
        B_hat[np.abs(B_hat) < 0.15] = 0.0
        f1s.append(skeleton_f1(B, B_hat))
        oes.append(order_err(order, order_hat))
        oes_b.append(order_err(order, corr_baseline_order(X)))
    return {
        "synthetic_dling_order_err": float(np.mean(oes)),
        "synthetic_dling_skel_f1": float(np.mean(f1s)),
        "synthetic_corr_order_err": float(np.mean(oes_b)),
        "synthetic_torch_available": 0.0,
    }
