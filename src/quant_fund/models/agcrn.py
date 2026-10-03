"""AGCRN (Bai et al. 2020) — adaptive graph conv recurrent network:
node-embedding-generated adjacency inside a GRU (no fixed A needed).
Next-step MSE vs DCRNN-style fixed-A ablation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gt_synth import ar2_baseline, gt_data, ring_adj


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("agcrn requires torch (pip install -e .[nn])") from exc
    return torch


def bench_agcrn(seed: int = 1633, iters: int = 800) -> dict[str, float]:
    torch = _torch()
    X = gt_data(seed)
    A = ring_adj(8)
    torch.manual_seed(seed)
    A_t = torch.tensor(A).float()
    E = torch.nn.Parameter(torch.randn(8, 8))
    H = 16
    Wr = torch.nn.Linear(1 + H, 2 * H, bias=False)
    Wu = torch.nn.Linear(1 + H, H, bias=False)
    head = torch.nn.Linear(H, 1)
    opt = torch.optim.Adam(
        [E] + list(Wr.parameters()) + list(Wu.parameters()) + list(head.parameters()), lr=0.01
    )
    Xt = torch.tensor(X).float()
    T = len(X)
    rng = np.random.default_rng(seed)

    def step(x, h, adapt=True):
        A_ad = torch.softmax(torch.relu(E @ E.T), -1) if adapt else A_t
        xin = A_ad @ torch.cat([x[:, None], h], 1)
        zr = torch.sigmoid(Wr(xin))
        z, r = zr.chunk(2, -1)
        u = torch.tanh(Wu(A_ad @ torch.cat([x[:, None], r * h], 1)))
        return z * h + (1 - z) * u

    for _ in range(iters):
        t = int(rng.integers(10, T - 1))
        h = torch.zeros(8, H)
        for tt in range(t - 10, t):
            h = step(Xt[tt], h)
        loss = ((head(h).squeeze(-1) - Xt[t]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds = np.zeros((T - 1, 8))
    h = torch.zeros(8, H)
    with torch.no_grad():
        for t in range(T - 1):
            h = step(Xt[t], h)
            preds[t] = head(h).squeeze(-1).numpy()
    # fixed-A ablation
    preds2 = np.zeros((T - 1, 8))
    h = torch.zeros(8, H)
    with torch.no_grad():
        for t in range(T - 1):
            h = step(Xt[t], h, adapt=False)
            preds2[t] = head(h).squeeze(-1).numpy()
    mse = float(np.mean((preds[2:] - X[3:]) ** 2))
    mse2 = float(np.mean((preds2[2:] - X[3:]) ** 2))
    base = ar2_baseline(X)
    return {
        "synthetic_agcrn_mse": mse,
        "synthetic_agcrn_fixed_mse": mse2,
        "synthetic_agcrn_ar2_mse": base,
        "synthetic_agcrn_mse_gain": base - mse,
        "synthetic_agcrn_adapt_gain": mse2 - mse,
        "torch_available": 1.0,
    }
