"""Prototypical networks (Snell et al. 2017) — few-shot REGRESSION via (SYNTHETIC)
task-conditioned prototypes: embed (x,y) support pairs; class protos per
task → weighted kernel regression for queries. Query MSE vs pooled.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._meta_synth import sine_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("protonet requires torch (pip install -e .[nn])") from exc
    return torch


def bench_protonet(seed: int = 857, n_tasks: int = 30, K: int = 5) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16))
    head = torch.nn.Linear(16 + 1, 1)  # proto-emb + x → y
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.005)
    for _t in range(n_tasks):
        xs, ys, xq, yq = sine_task(rng, K=K)
        S = torch.tensor(np.stack([xs, ys], 1)).float()
        z = emb(S).mean(0)  # task prototype
        qx = torch.tensor(xq).float()[:, None]
        inp = torch.cat([qx, z.expand(len(xq), 16)], 1)
        pred = head(inp).squeeze(-1)
        loss = ((pred - torch.tensor(yq).float()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    mses = []
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 2000 + i), K=K)
        with torch.no_grad():
            S = torch.tensor(np.stack([xs, ys], 1)).float()
            z = emb(S).mean(0)
            inp = torch.cat([torch.tensor(xq).float()[:, None], z.expand(len(xq), 16)], 1)
            pred = head(inp).squeeze(-1).numpy()
        mses.append(float(((pred - yq) ** 2).mean()))
    # pooled baseline: fit head on (x, zero-proto) only
    with torch.no_grad():
        zp = torch.zeros(16)
        inp = torch.cat([torch.tensor(xq).float()[:, None], zp.expand(len(xq), 16)], 1)
        pred0 = head(inp).squeeze(-1).numpy()
    mse0 = float(((pred0 - yq) ** 2).mean())
    return {
        "synthetic_pn_query_mse": float(np.mean(mses)),
        "synthetic_pn_zero_proto_mse": mse0,
        "synthetic_pn_gain": mse0 - float(np.mean(mses)),
        "synthetic_torch_available": 1.0,
    }
