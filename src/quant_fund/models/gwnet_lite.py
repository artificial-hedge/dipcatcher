"""Graph WaveNet-lite (Wu et al. 2019) — adaptive adjacency learned as
softmax(E1 E2^T) node embeddings + dilated temporal convs + graph conv.
Next-step MSE vs fixed-adjacency GCN ablation.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gt_synth import ar2_baseline, gt_data, ring_adj


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("gwnet_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_gwnet_lite(seed: int = 1613, iters: int = 800) -> dict[str, float]:
    torch = _torch()
    X = gt_data(seed)
    A = ring_adj(8)
    torch.manual_seed(seed)
    A_t = torch.tensor(A).float()
    E1 = torch.nn.Parameter(torch.randn(8, 8))
    E2 = torch.nn.Parameter(torch.randn(8, 8))
    C = 8
    tcn = torch.nn.Conv1d(1, 2 * C, 3, padding=1, dilation=2)
    gc = torch.nn.Linear(2 * C, C)  # input [fixed-A, adapt-A] concat
    head = torch.nn.Linear(C, 1)
    opt = torch.optim.Adam(
        [E1, E2] + list(tcn.parameters()) + list(gc.parameters()) + list(head.parameters()), lr=0.01
    )
    Xt = torch.tensor(X).float()
    T = len(X)
    rng = np.random.default_rng(seed)

    def encode(x_win, adapt=True):
        x = x_win.T[:, None, :]
        h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]
        A_ad = torch.softmax(E1 @ E2.T, -1) if adapt else A_t
        hf = torch.tanh(A_t @ h)
        ha = torch.tanh(A_ad @ h)
        return torch.tanh(gc(torch.cat([hf, ha], -1)))

    for _ in range(iters):
        t = int(rng.integers(10, T - 1))
        loss = ((head(encode(Xt[t - 10 : t])).squeeze(-1) - Xt[t]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # ablation: fixed A only
    gc2 = torch.nn.Linear(C, C)
    head2 = torch.nn.Linear(C, 1)
    opt2 = torch.optim.Adam(
        list(tcn.parameters()) + list(gc2.parameters()) + list(head2.parameters()), lr=0.01
    )
    for _ in range(iters // 2):
        t = int(rng.integers(10, T - 1))
        x = Xt[t - 10 : t].T[:, None, :]
        h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]
        loss = ((head2(torch.tanh(A_t @ gc2(h))).squeeze(-1) - Xt[t]) ** 2).mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    preds = np.zeros((T - 10, 8))
    preds2 = np.zeros((T - 10, 8))
    with torch.no_grad():
        for t in range(10, T):
            w = Xt[max(0, t - 10) : t]
            preds[t - 10] = head(encode(w)).squeeze(-1).numpy()
            x = w.T[:, None, :]
            h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]
            preds2[t - 10] = head2(torch.tanh(A_t @ gc2(h))).squeeze(-1).numpy()
    mse = float(np.mean((preds - X[10:]) ** 2))
    mse2 = float(np.mean((preds2 - X[10:]) ** 2))
    base = ar2_baseline(X)
    return {
        "synthetic_gwn_mse": mse,
        "synthetic_gwn_fixed_mse": mse2,
        "synthetic_gwn_ar2_mse": base,
        "synthetic_gwn_mse_gain": base - mse,
        "synthetic_gwn_adapt_gain": mse2 - mse,
        "torch_available": 1.0,
    }
