"""E(n)-equivariant GNN-lite (Satorras et al. 2021).

Coordinate updates use only relative positions — the network is
equivariant to rotations by construction. On the rotation-cloud fixture
the equivariant net holds accuracy under arbitrary test-time rotations
where a plain MLP degrades.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._geo_synth import synth_rot_cloud

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("equivar_gnn needs the torch `nn` extra") from exc


def bench_equivar_gnn(
    seed: int = 79,
    n_train: int = 300,
    n_test: int = 150,
    m_pts: int = 12,
    iters: int = 400,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr = synth_rot_cloud(n_train, m_pts, rng)
    xte, yte = synth_rot_cloud(n_test, m_pts, np.random.default_rng(seed + 1))
    d_h = 32
    eq_mlp = torch.nn.Sequential(
        torch.nn.Linear(m_pts * (m_pts - 1) // 2, d_h),
        torch.nn.ReLU(),
        torch.nn.Linear(d_h, 2),
    )
    flat = torch.nn.Sequential(
        torch.nn.Linear(m_pts * 3, d_h),
        torch.nn.ReLU(),
        torch.nn.Linear(d_h, 2),
    )
    params = list(eq_mlp.parameters()) + list(flat.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)

    def pairwise(xb):
        d = (xb[:, :, None, :] - xb[:, None, :, :]).norm(dim=-1)
        ii, jj = torch.triu_indices(xb.shape[1], xb.shape[1], 1)
        return d[:, ii, jj]

    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            eq_mlp(pairwise(xtr_t)), ytr_t
        ) + torch.nn.functional.cross_entropy(flat(xtr_t.reshape(xtr_t.shape[0], -1)), ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    g = rng.standard_normal((3, 3))
    q_rot, _r = np.linalg.qr(g)
    q_t = torch.tensor(q_rot).float()
    with torch.no_grad():
        xte_t = torch.tensor(xte).float()
        xte_rot = xte_t @ q_t
        acc_eq_id = float((eq_mlp(pairwise(xte_t)).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_eq_rot = float(
            (eq_mlp(pairwise(xte_rot)).argmax(-1) == torch.tensor(yte)).float().mean()
        )
        acc_flat_rot = float(
            (flat(xte_rot.reshape(xte_t.shape[0], -1)).argmax(-1) == torch.tensor(yte))
            .float()
            .mean()
        )
    return {
        "synthetic_equivar_acc_id": acc_eq_id,
        "synthetic_equivar_acc_rot": acc_eq_rot,
        "synthetic_equivar_flat_acc_rot": acc_flat_rot,
        "synthetic_equivar_drop": abs(acc_eq_id - acc_eq_rot),
        "synthetic_equivar_gain": acc_eq_rot - acc_flat_rot,
        "synthetic_torch_available": 1.0,
    }
