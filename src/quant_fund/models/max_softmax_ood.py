"""Max-softmax-probability OOD baseline + ODIN input perturbation.

Hendrycks & Gimpel 2017 (MSP) / Liang et al. 2018 (ODIN): the max
softmax probability is the simplest OOD signal; ODIN adds a small
input perturbation in the score-gradient direction plus temperature
scaling to sharpen the separation.

Bench: 4-class Gaussian blobs; ID vs rotated+shifted OOD. Metric:
AUROC of MSP, ODIN-boosted MSP.
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
        raise ImportError("max_softmax_ood requires the `nn` extra (make sync)") from exc


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


def bench_max_softmax_ood(
    seed: int = 20261231,
    n_id: int = 800,
    n_ood: int = 200,
    d: int = 8,
    iters: int = 500,
    temp: float = 100.0,
    eps: float = 0.05,
) -> dict[str, float]:
    """MSP AUROC, ODIN AUROC (SYNTHETIC)."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    (x_id, y_id), x_ood = synth_id_ood(n_id, n_ood, d, rng)
    X = torch.tensor(x_id, dtype=torch.float32)
    Y = torch.tensor(y_id.astype(np.int64), dtype=torch.long)
    net = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    params = torch.nn.ModuleList([net])
    opt = torch.optim.Adam(params.parameters(), lr=2e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(net(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        msp = np.concatenate(
            [
                -torch.softmax(net(X), -1).max(-1).values.numpy(),
                -torch.softmax(net(torch.tensor(x_ood, dtype=torch.float32)), -1)
                .max(-1)
                .values.numpy(),
            ]
        )
    auc_msp = _auc(np.asarray(msp), np.concatenate([np.zeros(n_id), np.ones(n_ood)]))

    def odin_score(x_in: Any) -> FloatArray:
        xt = x_in.clone().requires_grad_(True)
        logits = net(xt) / temp
        loss = torch.logsumexp(logits, -1).sum()
        grad = torch.autograd.grad(loss, xt)[0]
        xp = xt - eps * torch.sign(grad)
        with torch.no_grad():
            return np.asarray(-torch.softmax(net(xp) / temp, -1).max(-1).values.numpy())

    s_id = odin_score(X)
    s_ood = odin_score(torch.tensor(x_ood, dtype=torch.float32))
    auc_odin = _auc(
        np.concatenate([s_id, np.asarray(s_ood)]),
        np.concatenate([np.zeros(n_id), np.ones(n_ood)]),
    )
    return {
        "synthetic_msp_auc": auc_msp,
        "synthetic_odin_auc": float(auc_odin),
        "synthetic_odin_margin_vs_msp": float(auc_odin - auc_msp),
        "torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_max_softmax_ood()))
