"""Matching networks (Vinyals et al. 2016) — attention kernel regression:
query embeds against support embeddings; y = softmax-attention-weighted
support y's. Query MSE vs flat 1-NN cosine baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._meta_synth import sine_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("matching_net requires torch (pip install -e .[nn])") from exc
    return torch


def bench_matching_net(seed: int = 859, n_tasks: int = 30, K: int = 5) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Sequential(torch.nn.Linear(1, 24), torch.nn.ReLU(), torch.nn.Linear(24, 12))
    opt = torch.optim.Adam(emb.parameters(), lr=0.005)
    for _t in range(n_tasks):
        xs, ys, xq, yq = sine_task(rng, K=K)
        zs = emb(torch.tensor(xs).float()[:, None])
        zq = emb(torch.tensor(xq).float()[:, None])
        sim = zq @ zs.T / np.sqrt(12)
        att = torch.softmax(sim, 1)
        pred = att @ torch.tensor(ys).float()
        loss = ((pred - torch.tensor(yq).float()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    mses, mses_1nn = [], []
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 3000 + i), K=K)
        with torch.no_grad():
            zs = emb(torch.tensor(xs).float()[:, None])
            zq = emb(torch.tensor(xq).float()[:, None])
            att = torch.softmax(zq @ zs.T / np.sqrt(12), 1)
            pred = (att @ torch.tensor(ys).float()).numpy()
        mses.append(float(((pred - yq) ** 2).mean()))
        # 1-NN on raw x
        idx = np.argmin(np.abs(xq[:, None] - xs[None]), 1)
        mses_1nn.append(float(((ys[idx] - yq) ** 2).mean()))
    return {
        "synthetic_mn_query_mse": float(np.mean(mses)),
        "synthetic_mn_1nn_mse": float(np.mean(mses_1nn)),
        "synthetic_mn_gain": float(np.mean(mses_1nn) - np.mean(mses)),
        "torch_available": 1.0,
    }
