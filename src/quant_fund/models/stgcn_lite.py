"""STGCN-lite (Yu et al. 2018) — temporal gated conv (GLU over 3 taps)
+ spatial graph conv block; linear readout to next-step values.
Next-step MSE vs per-node AR(2).
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gt_synth import ar2_baseline, gt_data, ring_adj


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("stgcn_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_stgcn_lite(seed: int = 1607, iters: int = 800) -> dict[str, float]:
    torch = _torch()
    X = gt_data(seed)
    A = ring_adj(8)
    torch.manual_seed(seed)
    A_t = torch.tensor(A).float()
    C = 8
    tcn = torch.nn.Conv1d(1, 2 * C, 3, padding=1)
    gc = torch.nn.Linear(C, C)
    head = torch.nn.Linear(C, 1)
    opt = torch.optim.Adam(
        list(tcn.parameters()) + list(gc.parameters()) + list(head.parameters()), lr=0.01
    )
    Xt = torch.tensor(X).float()
    T = len(X)
    rng = np.random.default_rng(seed)

    def encode(x_win):  # x_win: (W, 8) -> (8, C)
        x = x_win.T[:, None, :]  # 8 x 1 x W
        h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]  # 8 x C last tap
        return torch.tanh(A_t @ gc(h))

    for _ in range(iters):
        t = int(rng.integers(10, T - 1))
        h = encode(Xt[t - 10 : t])
        loss = ((head(h).squeeze(-1) - Xt[t]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds = np.zeros((T - 10, 8))
    with torch.no_grad():
        for t in range(10, T):
            preds[t - 10] = head(encode(Xt[max(0, t - 10) : t])).squeeze(-1).numpy()
    mse = float(np.mean((preds - X[10:]) ** 2))
    base = ar2_baseline(X)
    return {
        "synthetic_stgcn_mse": mse,
        "synthetic_stgcn_ar2_mse": base,
        "synthetic_stgcn_mse_gain": base - mse,
        "torch_available": 1.0,
    }
