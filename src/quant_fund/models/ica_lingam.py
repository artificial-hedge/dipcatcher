"""ICA-LiNGAM (Shimizu et al. 2006) — linear SEM with non-Gaussian
noise: whiten, FastICA unmixing → mixing matrix → prune to DAG and
recover a topological order. Skeleton F1 + order error vs baselines.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._csl_synth import corr_baseline_order, lingam_sem, order_err, skeleton_f1


def _fastica(X: np.ndarray, iters: int = 200) -> np.ndarray:
    """Symmetric FastICA, tanh contrast; returns unmixing W (d×d)."""
    n, d = X.shape
    Xc = X - X.mean(0)
    C = np.cov(Xc.T)
    ev, E = np.linalg.eigh(C)
    Wh = E @ np.diag(1 / np.sqrt(ev)) @ E.T
    Xw = Xc @ Wh.T
    W = np.eye(d)
    for _ in range(iters):
        WX = Xw @ W.T
        g = np.tanh(WX)
        gp = 1 - g**2
        Wn = (g.T @ Xw) / n - np.diag(gp.mean(0)) @ W
        # symmetric decorrelation
        sq = np.linalg.eigh(Wn @ Wn.T)[1]
        Wn = sq @ np.diag(1 / np.sqrt(np.linalg.eigh(Wn @ Wn.T)[0])) @ sq.T @ Wn
        if np.abs(np.abs(np.diag(Wn @ W.T)) - 1).max() < 1e-6:
            W = Wn
            break
        W = Wn
    return W @ Wh  # X ≈ mixing @ sources; W @ X → sources


def bench_ica_lingam(seed: int = 2801, trials: int = 4, d: int = 6) -> dict[str, float]:
    f1s, oes, oes_b = [], [], []
    for t in range(trials):
        X, B, order = lingam_sem(seed + 3 * t, d=d)
        W = _fastica(X)
        # rows of W ≈ B^{-1}-like: scale so diag=1, off-diag→ -B
        D = np.diag(W).copy()
        Wn = W / np.abs(D)[:, None]
        B_hat = -Wn + np.diag(np.diag(Wn))
        B_hat[np.abs(B_hat) < 0.3] = 0.0
        f1s.append(skeleton_f1(B, B_hat))
        # order by row weight of |B_hat| (roots have small parent weights)
        mass = np.abs(B_hat).sum(0)
        ord_hat = np.argsort(mass).tolist()
        oes.append(order_err(order, ord_hat))
        oes_b.append(order_err(order, corr_baseline_order(X)))
    return {
        "synthetic_ical_skel_f1": float(np.mean(f1s)),
        "synthetic_ical_order_err": float(np.mean(oes)),
        "synthetic_corr_order_err": float(np.mean(oes_b)),
        "torch_available": 0.0,
    }
