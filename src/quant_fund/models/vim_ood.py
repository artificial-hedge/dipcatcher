"""ViM: virtual-logit matching for OOD (Wang et al. 2022) (SYNTHETIC).

Combines (i) the residual energy of a feature projected off the
principal subspace of training features, with (ii) the max-logit — a
virtual logit computed as alpha * ||residual|| stands in for "none of
the above".

Bench: 4-class blobs; OOD has a subspace component orthogonal to
train features (rotation). AUROC vs MSP and energy score.
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
        raise ImportError("vim_ood requires the `nn` extra (make sync)") from exc


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


def bench_vim_ood(
    seed: int = 20261231,
    n_id: int = 800,
    n_ood: int = 200,
    d: int = 8,
    iters: int = 500,
) -> dict[str, float]:
    """Virtual-logit matching vs MSP / energy (AUROC, SYNTHETIC)."""
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
        l_id = head(torch.tensor(f_id, dtype=torch.float32)).numpy()
        l_ood = head(torch.tensor(f_ood, dtype=torch.float32)).numpy()
    f_tr = np.asarray(f_tr, dtype=np.float64)
    mu = f_tr.mean(0)
    cov = np.cov((f_tr - mu).T)
    _, V = np.linalg.eigh(cov)
    P = V[:, :8] @ V[:, :8].T  # principal subspace (top-8 of 16)

    def resid(f: FloatArray) -> FloatArray:
        r = (f - mu) - (f - mu) @ P
        return np.sqrt((r**2).sum(1))

    r_id, r_ood = resid(f_id), resid(f_ood)
    lse_id = np.log(np.exp(np.asarray(l_id)).sum(1))
    lse_ood = np.log(np.exp(np.asarray(l_ood)).sum(1))
    alpha = float(lse_id.mean() / max(r_id.mean(), 1e-9))
    # virtual logit: alpha * ||residual|| acts as the (K+1)-th logit;
    # score = v - logsumexp(real logits) — high when residual dominates
    vim_id = alpha * r_id - lse_id
    vim_ood = alpha * r_ood - lse_ood
    is_ood = np.concatenate([np.zeros(f_id.shape[0]), np.ones(n_ood)])
    auc_vim = _auc(np.concatenate([vim_id, vim_ood]), is_ood)
    with torch.no_grad():
        sm_id = torch.softmax(torch.tensor(l_id), -1).numpy()
        sm_ood = torch.softmax(torch.tensor(l_ood), -1).numpy()
    msp_s = np.concatenate([-np.asarray(sm_id).max(1), -np.asarray(sm_ood).max(1)])
    auc_msp = _auc(np.asarray(msp_s), is_ood)
    e_s = np.concatenate(
        [
            -np.log(np.exp(np.asarray(l_id)).sum(1)),
            -np.log(np.exp(np.asarray(l_ood)).sum(1)),
        ]
    )
    auc_e = _auc(np.asarray(e_s), is_ood)
    return {
        "synthetic_vim_auc": auc_vim,
        "synthetic_vim_msp_auc": auc_msp,
        "synthetic_vim_energy_auc": auc_e,
        "synthetic_vim_margin_vs_msp": auc_vim - auc_msp,
        "synthetic_vim_margin_vs_energy": auc_vim - auc_e,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_vim_ood()))
