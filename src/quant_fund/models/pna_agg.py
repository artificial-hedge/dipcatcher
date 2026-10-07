"""PNA (Corso et al. 2020) — multi-aggregator message passing: per-neighbor
messages aggregated by {mean,max,min,std}, concatenated → node MLP.
Planted-clique node AUC vs mean-only aggregator.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._anom_synth import auc
from quant_fund.models._gex_synth import planted_clique


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("pna_agg requires torch (pip install -e .[nn])") from exc
    return torch


def _agg(z, A, agg: str):
    torch = _torch()
    deg = A.sum(1, keepdim=True).clamp_min(1)
    if agg == "mean":
        return (A @ z) / deg
    if agg == "max":
        return torch.max(
            torch.where(A[:, :, None] > 0, z[None, :, :], torch.tensor(-1e9)), 1
        ).values.clamp_min(-1e6)
    if agg == "min":
        return torch.min(
            torch.where(A[:, :, None] > 0, z[None, :, :], torch.tensor(1e9)), 1
        ).values.clamp(max=1e6)
    if agg == "std":
        m = (A @ z) / deg
        return torch.sqrt(((A @ (z**2)) / deg - m**2).clamp_min(0))
    raise ValueError(agg)


def bench_pna_agg(seed: int = 881, iters: int = 200) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    aggs = ["mean", "max", "min", "std"]
    enc = torch.nn.Linear(4, 8)
    mlp = torch.nn.Linear(8 * len(aggs) + 4, 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(mlp.parameters()), lr=0.01)
    for _i in range(iters):
        A_np, x, y = planted_clique(int(rng.integers(1 << 30)))
        A = torch.tensor(A_np).float()
        z = enc(torch.tensor(x).float())
        msgs = torch.cat([_agg(z, A, a) for a in aggs] + [torch.tensor(x).float()], 1)
        logits = mlp(msgs).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(y).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    A_np, x, y = planted_clique(seed + 999)
    A = torch.tensor(A_np).float()
    with torch.no_grad():
        z = enc(torch.tensor(x).float())
        msgs = torch.cat([_agg(z, A, a) for a in aggs] + [torch.tensor(x).float()], 1)
        sc = mlp(msgs).squeeze(-1).numpy()
    auc_pna = auc(sc, y.astype(np.int64))
    # mean-only ablation
    mlp_m = torch.nn.Linear(8 + 4, 1)
    opt = torch.optim.Adam(mlp_m.parameters(), lr=0.01)
    for _i in range(iters):
        A_np2, x2, y2 = planted_clique(int(rng.integers(1 << 30)))
        A2 = torch.tensor(A_np2).float()
        z2 = enc(torch.tensor(x2).float())
        msgs = torch.cat([_agg(z2, A2, "mean"), torch.tensor(x2).float()], 1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            mlp_m(msgs).squeeze(-1), torch.tensor(y2).float()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z2 = enc(torch.tensor(x).float())
        msgs = torch.cat([_agg(z2, A, "mean"), torch.tensor(x).float()], 1)
        sc_m = mlp_m(msgs).squeeze(-1).numpy()
    auc_m = auc(sc_m, y.astype(np.int64))
    return {
        "synthetic_pna_auc": auc_pna,
        "synthetic_pna_mean_auc": auc_m,
        "synthetic_pna_gain": auc_pna - auc_m,
        "synthetic_torch_available": 1.0,
    }
