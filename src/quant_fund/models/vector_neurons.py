"""Vector-neuron SO(3)-equivariant layer-lite (Deng et al. 2021).

Features are 3-vectors; linear layers mix them (rotation-equivariant),
nonlinearity = ReLU applied to the norm times a learned direction, and
the invariant head pools channel norms. Rotation of the input rotates
intermediate vectors but leaves the norm-pooled output invariant.
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
        raise ImportError("vector_neurons needs the torch `nn` extra") from exc


def bench_vector_neurons(
    seed: int = 113,
    n_train: int = 300,
    n_test: int = 150,
    m_pts: int = 12,
    iters: int = 400,
    c_out: int = 16,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr = synth_rot_cloud(n_train, m_pts, rng)
    xte, yte = synth_rot_cloud(n_test, m_pts, np.random.default_rng(seed + 1))
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)
    # VNN linear: mix channels (not coordinates) — W (c_in=1,c_out) applied on channel dim
    w1 = torch.nn.Parameter(torch.randn(c_out) * 0.3)
    k1 = torch.nn.Parameter(torch.randn(c_out) * 0.3)
    head = torch.nn.Linear(2 * c_out, 2)
    flat = torch.nn.Sequential(
        torch.nn.Linear(m_pts * 3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2)
    )
    params = [w1, k1] + list(head.parameters()) + list(flat.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)

    def vnn_fwd(xb):
        v = xb[:, :, None, :]
        h = v * w1[None, None, :, None]
        # VNN nonlinearity: direction-preserving gate on <h, k̂>
        kdir = k1 / k1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        proj = (h * kdir[None, None, :, None]).sum(-1)
        act = torch.relu(proj)
        h2 = h * (act / (h.norm(dim=-1) + 1e-9))[..., None]
        # invariant pooling: per-channel norms then mean/max over points
        feats = torch.cat([h2.norm(dim=-1).mean(1), h2.norm(dim=-1).max(1).values], -1)
        return head(feats)

    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            vnn_fwd(xtr_t), ytr_t
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
        acc_id = float((vnn_fwd(xte_t).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_rot = float((vnn_fwd(xte_rot).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_flat_rot = float(
            (flat(xte_rot.reshape(xte_t.shape[0], -1)).argmax(-1) == torch.tensor(yte))
            .float()
            .mean()
        )
    return {
        "synthetic_vnn_acc_id": acc_id,
        "synthetic_vnn_acc_rot": acc_rot,
        "synthetic_vnn_flat_acc_rot": acc_flat_rot,
        "synthetic_vnn_drop": abs(acc_id - acc_rot),
        "synthetic_vnn_gain": acc_rot - acc_flat_rot,
        "synthetic_torch_available": 1.0,
    }
