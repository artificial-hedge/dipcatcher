"""APS — adaptive prediction sets (Romano et al. 2020) — split-conformal
sets accumulating class probabilities until the quantile threshold;
set size vs top-k/threshold-set baselines at fixed coverage.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cp2_synth import cls_data


def _fit_softmax(X: np.ndarray, y: np.ndarray, K: int, iters: int = 400) -> np.ndarray:
    rng = np.random.default_rng(0)
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    W = 0.01 * rng.standard_normal((A.shape[1], K))
    for _ in range(iters):
        z = A @ W
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[range(len(y)), y] -= 1
        W -= 0.1 * (A.T @ p / len(y) + 0.01 * W)
    return W


def _probs(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.concatenate([X, np.ones((len(X), 1))], 1) @ W
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return np.asarray(p / p.sum(1, keepdims=True))


def bench_aps_cp(seed: int = 1309, alpha: float = 0.1) -> dict[str, float]:
    X, y, Xt, yt = cls_data(seed)
    K = 2
    n_tr = 2 * len(X) // 3
    W = _fit_softmax(X[:n_tr], y[:n_tr], K)
    Pcal = _probs(W, X[n_tr:])
    # APS score: cumulative mass needed to reach true class
    srt = np.argsort(-Pcal, 1)
    csum = np.take_along_axis(np.cumsum(np.take_along_axis(Pcal, srt, 1), 1), np.argsort(srt, 1), 1)
    scores = csum[range(len(y[n_tr:])), y[n_tr:]]
    Q = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    Pt = _probs(W, Xt)
    srt_t = np.argsort(-Pt, 1)
    cum_t = np.cumsum(np.take_along_axis(Pt, srt_t, 1), 1)
    sizes = (cum_t <= Q).sum(1) + 1
    sizes = np.minimum(sizes, K)
    cov = float(np.mean([yt[i] in srt_t[i, : sizes[i]] for i in range(len(yt))]))
    # threshold-set baseline: include classes with p >= 1 - (1-alpha)=alpha fixed
    sizes2 = (Pt >= alpha).sum(1).clip(1)
    cov2 = float(np.mean([yt[i] in srt_t[i, : int(sizes2[i])] for i in range(len(yt))]))
    return {
        "synthetic_aps_coverage": cov,
        "synthetic_aps_target": 1 - alpha,
        "synthetic_aps_set_size": float(np.mean(sizes)),
        "synthetic_aps_thresh_coverage": cov2,
        "synthetic_aps_thresh_size": float(np.mean(sizes2)),
        "torch_available": 0.0,
    }
