"""ANIL (Raghu et al. 2020) — meta-learn the body only; the linear head is (SYNTHETIC)
re-fit on each task's support. Query MSE vs pooled baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._meta_synth import sine_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("anil_meta requires torch (pip install -e .[nn])") from exc
    return torch


def bench_anil_meta(seed: int = 863, n_tasks: int = 30, K: int = 5) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    body = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16))
    head = torch.nn.Linear(16, 1)
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=0.005)
    for _t in range(n_tasks):
        xs, ys, xq, yq = sine_task(rng, K=K)
        Xs = torch.tensor(xs).float()[:, None]
        ys_t = torch.tensor(ys).float()
        # inner: update head only
        h = torch.nn.Linear(16, 1)
        with torch.no_grad():
            h.weight.copy_(head.weight)
            h.bias.copy_(head.bias)
        io = torch.optim.SGD(h.parameters(), lr=0.01)
        for _ in range(10):
            zs = body(Xs)
            loss_i = ((h(zs).squeeze(-1) - ys_t) ** 2).mean()
            io.zero_grad()
            loss_i.backward()
            io.step()
        # outer: meta-update body on query loss via adapted head (first-order)
        Xq = torch.tensor(xq).float()[:, None]
        pred = h(body(Xq)).squeeze(-1)
        loss = ((pred - torch.tensor(yq).float()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    mses = []
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 4000 + i), K=K)
        h = torch.nn.Linear(16, 1)
        io = torch.optim.SGD(h.parameters(), lr=0.01)
        Xs = torch.tensor(xs).float()[:, None]
        for _ in range(20):
            loss_i = ((h(body(Xs)).squeeze(-1) - torch.tensor(ys).float()) ** 2).mean()
            io.zero_grad()
            loss_i.backward()
            io.step()
        with torch.no_grad():
            pred = h(body(torch.tensor(xq).float()[:, None])).squeeze(-1).numpy()
        mses.append(float(((pred - yq) ** 2).mean()))
    # pooled: linear on raw x
    w, _, _, _ = np.linalg.lstsq(np.stack([xs, np.ones(len(xs))], 1), ys, rcond=None)
    mse_pool = float(((xq * w[0] + w[1] - yq) ** 2).mean())
    return {
        "synthetic_anil_query_mse": float(np.mean(mses)),
        "synthetic_anil_pooled_mse": mse_pool,
        "synthetic_anil_gain": mse_pool - float(np.mean(mses)),
        "synthetic_torch_available": 1.0,
    }
