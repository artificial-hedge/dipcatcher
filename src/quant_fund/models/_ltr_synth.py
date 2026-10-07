"""Synthetic learning-to-rank fixture: queries → candidate docs with (SYNTHETIC)
graded relevance driven by a true linear model + feature noise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ltr_data(
    seed: int = 7,
    n_q: int = 200,
    n_doc: int = 20,
    d: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """Returns x_tr (q,n,d), y_tr (q,n), x_te, y_te with grades 0..2."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    x = rng.standard_normal((n_q * 2, n_doc, d))
    scores = x @ w + rng.standard_normal((n_q * 2, n_doc)) * 0.7
    y = np.digitize(scores, np.quantile(scores, [1 / 3, 2 / 3])).astype(int)
    return (
        x[:n_q].astype(np.float64),
        y[:n_q],
        x[n_q:].astype(np.float64),
        y[n_q:],
    )


def ndcg_at(y_true: NDArray[np.int64], scores: NDArray[np.float64], k: int = 10) -> float:
    """Mean NDCG@k across queries."""
    out = []
    gains = 2.0**y_true - 1
    for i in range(len(y_true)):
        order = np.argsort(-scores[i])[:k]
        ideal = np.argsort(-y_true[i])[:k]
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (gains[i][order] * disc).sum()
        idcg = (gains[i][ideal] * disc).sum()
        out.append(dcg / max(idcg, 1e-9))
    return float(np.mean(out))
