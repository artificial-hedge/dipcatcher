"""Mahalanobis OOD detection (Lee et al. 2018) (SYNTHETIC).

Per-class Gaussian on penultimate features with a shared covariance;
score = min over classes of the Mahalanobis distance. Near-class
samples score low; shifted/OOD samples score high.

Bench: 4-class Gaussian blobs in 8-D; a trained MLP's penultimate
features; OOD = rotated+scaled distribution. Metric: AUROC vs the
max-softmax-probability baseline.
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
        raise ImportError("mahalanobis_ood requires the `nn` extra (make sync)") from exc


def synth_id_ood(
    n_id: int, n_ood: int, d: int, rng: np.random.Generator
) -> tuple[tuple[FloatArray, NDArray[np.int64]], FloatArray]:
    """4 Gaussian blobs (ID); OOD = blobs rotated 45° + scaled 2.5×."""
    k = 4
    means = rng.normal(0, 3, (k, d))
    x_id = np.stack([means[i % k] + 0.8 * rng.standard_normal(d) for i in range(n_id)])
    rot = rng.normal(0, 1, (d, d))
    rot, _ = np.linalg.qr(rot)
    x_ood = np.stack(
        [1.2 * rot @ (means[i % k] + 0.9 * rng.standard_normal(d)) + 1.5 for i in range(n_ood)]
    )
    return (
        (x_id.astype(np.float64), (np.arange(n_id) % k).astype(np.int64)),
        x_ood.astype(np.float64),
    )


def _fit_features(x: FloatArray, y: NDArray[np.int64], iters: int = 500) -> Any:
    torch = _torch()
    X = torch.tensor(x, dtype=torch.float32)
    Y = torch.tensor(y.astype(np.int64), dtype=torch.long)
    body = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU()
    )
    head = torch.nn.Linear(16, 4)
    params = torch.nn.ModuleList([body, head])
    opt = torch.optim.Adam(params.parameters(), lr=2e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(head(body(X)), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return body, head


def _auc(scores: FloatArray, is_ood: FloatArray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = is_ood == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def bench_mahalanobis_ood(
    seed: int = 20261231,
    n_id: int = 800,
    n_ood: int = 200,
    d: int = 8,
) -> dict[str, float]:
    """Feature-space Mahalanobis vs max-softmax baseline (AUROC)."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    (x_id, y_id), x_ood = synth_id_ood(n_id, n_ood, d, rng)
    body, head = _fit_features(x_id, y_id)
    with torch.no_grad():
        f_id = body(torch.tensor(x_id, dtype=torch.float32)).numpy()
        f_ood = body(torch.tensor(x_ood, dtype=torch.float32)).numpy()
    f_id = np.asarray(f_id, dtype=np.float64)
    f_ood = np.asarray(f_ood, dtype=np.float64)
    k = 4
    mus = np.stack([f_id[y_id == c].mean(0) for c in range(k)])
    cov = np.cov(np.concatenate([f_id[y_id == c] - mus[c] for c in range(k)]).T)
    prec = np.linalg.pinv(cov + 1e-4 * np.eye(cov.shape[0]))

    def maha(f: FloatArray) -> FloatArray:
        d_ = np.stack([np.einsum("bi,ij,bj->b", f - mus[c], prec, f - mus[c]) for c in range(k)])
        return d_.min(0)

    scores = np.concatenate([maha(f_id), maha(f_ood)])
    is_ood = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    auc_m = _auc(scores, is_ood)

    with torch.no_grad():
        p_id = torch.softmax(head(torch.tensor(f_id, dtype=torch.float32)), -1).numpy()
        p_ood = torch.softmax(head(torch.tensor(f_ood, dtype=torch.float32)), -1).numpy()
    msp = np.concatenate([-np.asarray(p_id).max(1), -np.asarray(p_ood).max(1)])
    auc_msp = _auc(np.asarray(msp), is_ood)
    return {
        "synthetic_maha_auc": auc_m,
        "synthetic_maha_msp_auc": auc_msp,
        "synthetic_maha_margin_vs_msp": auc_m - auc_msp,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_mahalanobis_ood()))
