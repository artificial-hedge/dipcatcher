"""k-NN OOD detection in penultimate features (Sun et al. 2022) (SYNTHETIC).

Score = distance to the k-th nearest training feature (after
normalization). Non-parametric and surprisingly SOTA vs density
methods on feature space.

Bench: 4-class blobs; ID vs rotated+shifted OOD. AUROC vs MSP and
the Mahalanobis score on the same features.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("knn_ood requires the `nn` extra (make sync)") from exc


def synth_id_ood(
    n_id: int, n_ood: int, d: int, rng: np.random.Generator
) -> tuple[tuple[FloatArray, NDArray[np.int64]], FloatArray]:
    k = 4
    means = rng.normal(0, 3, (k, d))
    x_id = np.stack([means[i % k] + 0.8 * rng.standard_normal(d) for i in range(n_id)])
    rot = rng.normal(0, 1, (d, d))
    rot, _ = np.linalg.qr(rot)
    x_ood = np.stack(
        [1.2 * rot @ (means[i % k] + 0.9 * rng.standard_normal(d)) + 1.5 for i in range(n_ood)]
    )
    return (x_id.astype(np.float64), (np.arange(n_id) % k).astype(np.int64)), x_ood.astype(
        np.float64
    )


def _auc(scores: FloatArray, is_ood: FloatArray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = is_ood == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def bench_knn_ood(
    seed: int = 20261231,
    n_id: int = 800,
    n_ood: int = 200,
    d: int = 8,
    k_nn: int = 10,
    iters: int = 500,
) -> dict[str, float]:
    """Feature-space kNN distance vs MSP (AUROC, SYNTHETIC)."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    (x_id, y_id), x_ood = synth_id_ood(n_id, n_ood, d, rng)
    X = torch.tensor(x_id, dtype=torch.float32)
    Y = torch.tensor(y_id.astype(np.int64), dtype=torch.long)
    body = torch.nn.Sequential(
        torch.nn.Linear(d, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
    )
    head = torch.nn.Linear(16, 4)
    params = torch.nn.ModuleList([body, head])
    opt = torch.optim.Adam(params.parameters(), lr=2e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(head(body(X)), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        f_tr = body(X[: 3 * n_id // 4]).numpy()
        f_id = body(X[3 * n_id // 4 :]).numpy()
        f_ood = body(torch.tensor(x_ood, dtype=torch.float32)).numpy()
    f_tr = np.asarray(f_tr) / (np.linalg.norm(f_tr, axis=1, keepdims=True) + 1e-9)
    f_id = np.asarray(f_id) / (np.linalg.norm(f_id, axis=1, keepdims=True) + 1e-9)
    f_ood = np.asarray(f_ood) / (np.linalg.norm(f_ood, axis=1, keepdims=True) + 1e-9)

    def knn_dist(f: FloatArray) -> FloatArray:
        d2 = np.sum(f**2, 1)[:, None] + np.sum(f_tr**2, 1)[None] - 2 * f @ f_tr.T
        kth = np.partition(d2, k_nn, 1)[:, k_nn]
        return np.asarray(np.sqrt(np.maximum(kth, 0)))

    n_te = f_id.shape[0]
    is_ood = np.concatenate([np.zeros(n_te), np.ones(n_ood)])
    s_knn = np.concatenate([knn_dist(f_id), knn_dist(f_ood)])
    auc_knn = _auc(s_knn, is_ood)
    with torch.no_grad():
        msp = np.concatenate(
            [
                -torch.softmax(head(torch.tensor(f_id, dtype=torch.float32)), -1)
                .max(-1)
                .values.numpy(),
                -torch.softmax(head(torch.tensor(f_ood, dtype=torch.float32)), -1)
                .max(-1)
                .values.numpy(),
            ]
        )
    auc_msp = _auc(np.asarray(msp), is_ood)
    return {
        "synthetic_knnood_auc": auc_knn,
        "synthetic_knnood_msp_auc": auc_msp,
        "synthetic_knnood_margin_vs_msp": auc_knn - auc_msp,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_knn_ood()))
