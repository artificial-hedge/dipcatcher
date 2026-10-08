"""DGN-lite (Beaini et al. 2021) — directional aggregation: message along (SYNTHETIC)
edge (i,j) weighted by directionality (unit vector between node positions
in a learned embedding). Planted-clique AUC vs isotropic GCN.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._anom_synth import auc
from quant_fund.models._gex_synth import planted_clique


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("dgn_directional requires torch (pip install -e .[nn])") from exc
    return torch


def bench_dgn_directional(seed: int = 893, iters: int = 200) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pos_enc = torch.nn.Linear(4, 8)
    dir_net = torch.nn.Linear(8, 8)  # direction-conditioned weight
    read = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(
        list(pos_enc.parameters()) + list(dir_net.parameters()) + list(read.parameters()), lr=0.005
    )
    for _i in range(iters):
        A_np, x, y = planted_clique(int(rng.integers(1 << 30)))
        A = torch.tensor(A_np).float()
        p = pos_enc(torch.tensor(x).float())  # pseudo-positions
        # directional kernel: edge weight = |cos(p_i - p_j)| via normalized diffs
        d = p[:, None, :] - p[None, :, :]
        dn = d / (d.norm(dim=2, keepdim=True) + 1e-6)
        w = torch.abs((dn * dn.mean(2, keepdim=True).expand_as(dn)).sum(2)) * A
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        z = torch.relu(dir_net(w @ p))
        logits = read(z).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(y).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    A_np, x, y = planted_clique(seed + 999)
    A = torch.tensor(A_np).float()
    with torch.no_grad():
        p = pos_enc(torch.tensor(x).float())
        d = p[:, None, :] - p[None, :, :]
        dn = d / (d.norm(dim=2, keepdim=True) + 1e-6)
        w = torch.abs((dn * dn.mean(2, keepdim=True).expand_as(dn)).sum(2)) * A
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        z = torch.relu(dir_net(w @ p))
        sc = read(z).squeeze(-1).numpy()
    auc_dgn = auc(sc, y.astype(np.int64))
    return {
        "synthetic_dgn_auc": auc_dgn,
        "synthetic_dgn_random_auc": 0.5,
        "synthetic_dgn_lift": auc_dgn - 0.5,
        "synthetic_torch_available": 1.0,
    }
