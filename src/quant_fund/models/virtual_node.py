"""Virtual-node GNN (Gilmer et al. 2017) — a supernode aggregates all (SYNTHETIC)
node states each round and broadcasts back; helps on planted-clique
detection where global context matters. AUC vs plain GCN.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._anom_synth import auc
from quant_fund.models._gex_synth import planted_clique


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("virtual_node requires torch (pip install -e .[nn])") from exc
    return torch


def bench_virtual_node(seed: int = 883, iters: int = 200) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    enc = torch.nn.Linear(4, 8)
    msg = torch.nn.Linear(8, 8)
    vn_up = torch.nn.Linear(8, 8)
    read = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(
        list(enc.parameters())
        + list(msg.parameters())
        + list(vn_up.parameters())
        + list(read.parameters()),
        lr=0.01,
    )
    for _i in range(iters):
        A_np, x, y = planted_clique(int(rng.integers(1 << 30)))
        A = torch.tensor(A_np).float()
        deg = A.sum(1, keepdim=True).clamp_min(1)
        z = enc(torch.tensor(x).float())
        vn = z.mean(0, keepdim=True)
        for _k in range(3):
            a = msg((A @ z) / deg + vn)
            z = torch.relu(z + a)
            vn = vn_up(z.mean(0, keepdim=True) + vn)
        logits = read(z).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(y).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    A_np, x, y = planted_clique(seed + 999)
    A = torch.tensor(A_np).float()
    deg = A.sum(1, keepdim=True).clamp_min(1)
    with torch.no_grad():
        z = enc(torch.tensor(x).float())
        vn = z.mean(0, keepdim=True)
        for _k in range(3):
            a = msg((A @ z) / deg + vn)
            z = torch.relu(z + a)
            vn = vn_up(z.mean(0, keepdim=True) + vn)
        sc = read(z).squeeze(-1).numpy()
    auc_vn = auc(sc, y.astype(np.int64))
    # plain GCN baseline
    enc2 = torch.nn.Linear(4, 8)
    msg2 = torch.nn.Linear(8, 8)
    read2 = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(
        list(enc2.parameters()) + list(msg2.parameters()) + list(read2.parameters()), lr=0.01
    )
    for _i in range(iters):
        A_np2, x2, y2 = planted_clique(int(rng.integers(1 << 30)))
        A2 = torch.tensor(A_np2).float()
        deg2 = A2.sum(1, keepdim=True).clamp_min(1)
        z = enc2(torch.tensor(x2).float())
        for _k in range(3):
            z = torch.relu(z + msg2((A2 @ z) / deg2))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            read2(z).squeeze(-1), torch.tensor(y2).float()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = enc2(torch.tensor(x).float())
        for _k in range(3):
            z = torch.relu(z + msg2((A @ z) / deg))
        sc2 = read2(z).squeeze(-1).numpy()
    auc_gcn = auc(sc2, y.astype(np.int64))
    return {
        "synthetic_vn_auc": auc_vn,
        "synthetic_vn_gcn_auc": auc_gcn,
        "synthetic_vn_gain": auc_vn - auc_gcn,
        "synthetic_torch_available": 1.0,
    }
