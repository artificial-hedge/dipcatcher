"""MTGNN-lite (Wu et al. 2020) — mixhop propagation (sum of A^k powers)
+ dilated inception temporal conv + learned uni-directional adjacency.
Next-step MSE vs plain GCN-TCN.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gt_synth import ar2_baseline, gt_data, ring_adj


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mtgnn_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_mtgnn_lite(seed: int = 1627, iters: int = 800, K: int = 3) -> dict[str, float]:
    torch = _torch()
    X = gt_data(seed)
    A = ring_adj(8)
    torch.manual_seed(seed)
    A_t = torch.tensor(A).float()
    E1 = torch.nn.Parameter(torch.randn(8, 6))
    E2 = torch.nn.Parameter(torch.randn(8, 6))
    C = 8
    tcn = torch.nn.Conv1d(1, 2 * C, 3, padding=1, dilation=2)
    gc = torch.nn.Linear(K * C, C)
    head = torch.nn.Linear(C, 1)
    opt = torch.optim.Adam(
        [E1, E2] + list(tcn.parameters()) + list(gc.parameters()) + list(head.parameters()), lr=0.01
    )
    Xt = torch.tensor(X).float()
    T = len(X)
    rng = np.random.default_rng(seed)

    def encode(x_win):
        x = x_win.T[:, None, :]
        h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]
        A_ad = torch.relu(E1 @ E2.T)
        A_ad = A_ad / A_ad.sum(-1, keepdim=True).clamp_min(1e-6)
        hops = [h]
        cur = h
        for _ in range(K - 1):
            cur = A_ad @ cur
            hops.append(cur)
        return torch.tanh(gc(torch.cat(hops, -1)))

    for _ in range(iters):
        t = int(rng.integers(10, T - 1))
        loss = ((head(encode(Xt[t - 10 : t])).squeeze(-1) - Xt[t]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds = np.zeros((T - 10, 8))
    with torch.no_grad():
        for t in range(10, T):
            preds[t - 10] = head(encode(Xt[max(0, t - 10) : t])).squeeze(-1).numpy()
    # plain GCN ablation
    gc2 = torch.nn.Linear(C, C)
    head2 = torch.nn.Linear(C, 1)
    opt2 = torch.optim.Adam(list(gc2.parameters()) + list(head2.parameters()), lr=0.01)
    for _ in range(iters // 2):
        t = int(rng.integers(10, T - 1))
        x = Xt[t - 10 : t].T[:, None, :]
        h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]
        loss = ((head2(torch.tanh(A_t @ gc2(h))).squeeze(-1) - Xt[t]) ** 2).mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    preds2 = np.zeros((T - 10, 8))
    with torch.no_grad():
        for t in range(10, T):
            x = Xt[max(0, t - 10) : t].T[:, None, :]
            h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]
            preds2[t - 10] = head2(torch.tanh(A_t @ gc2(h))).squeeze(-1).numpy()
    mse = float(np.mean((preds - X[10:]) ** 2))
    mse2 = float(np.mean((preds2 - X[10:]) ** 2))
    base = ar2_baseline(X)
    return {
        "synthetic_mtg_mse": mse,
        "synthetic_mtg_gcn_mse": mse2,
        "synthetic_mtg_ar2_mse": base,
        "synthetic_mtg_mse_gain": base - mse,
        "synthetic_mtg_mixhop_gain": mse2 - mse,
        "synthetic_torch_available": 1.0,
    }
