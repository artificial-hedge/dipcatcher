"""Differentiable soft-rank / NeuralSort-style layer (Grover et al. 2019).

Rank via Σ_j σ((s_i−s_j)/τ) — fully differentiable, so an upstream score
model can be trained end-to-end on a ranking loss. On the fixture the
target order weights even dims differently, so the identity ordering
is wrong and the learned score must reorder — hard argsort passes no
gradient.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._geo_synth import synth_sort

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("sort_net needs the torch `nn` extra") from exc


def bench_sort_net(
    seed: int = 87,
    n_train: int = 300,
    n_test: int = 120,
    m: int = 8,
    iters: int = 600,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, _r = synth_sort(n_train, m, rng)
    xte, _rte = synth_sort(n_test, m, np.random.default_rng(seed + 1))
    mask = (np.arange(m) % 2).astype(np.float64)
    u_tr = xtr * (1 + 0.8 * mask[None, :])
    u_te = xte * (1 + 0.8 * mask[None, :])
    rank_tr = u_tr.argsort(-1).argsort(-1)
    xtr_t = torch.tensor(xtr).float()
    rtr_t = torch.tensor(rank_tr).float()
    xte_t = torch.tensor(xte).float()
    w = torch.nn.Parameter(torch.ones(m))
    opt = torch.optim.Adam([w], lr=5e-3)
    tau = 0.5

    def soft_rank(s):
        return torch.sigmoid((s[:, :, None] - s[:, None, :]) / tau).sum(-1)

    for _i in range(iters):
        s = xtr_t * w[None, :]
        loss = (soft_rank(s) - rtr_t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_rank = soft_rank(xte_t * w[None, :]).numpy()
        base_rank = soft_rank(xte_t).numpy()
    spear = float(
        np.mean([np.corrcoef(np.argsort(u_te[i]), pred_rank[i]) for i in range(xte.shape[0])])
    )
    spear_id = float(
        np.mean([np.corrcoef(np.argsort(u_te[i]), base_rank[i]) for i in range(xte.shape[0])])
    )
    w_grad = float(w.grad.abs().max().item()) if w.grad is not None else 0.0
    return {
        "synthetic_sortnet_spearman": spear,
        "synthetic_sortnet_identity_spearman": spear_id,
        "synthetic_sortnet_gain": float(spear - spear_id),
        "synthetic_sortnet_grad_flow": w_grad,
        "synthetic_torch_available": 1.0,
    }
