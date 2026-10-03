"""DCRNN-lite (Li et al. 2018) — diffusion-convolutional GRU: gates act
on A-propagated [x, h] instead of plain concatenation; linear readout.
Next-step MSE vs per-node AR(2).
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gt_synth import ar2_baseline, gt_data, ring_adj


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("dcrnn_lite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_dcrnn_lite(seed: int = 1601, iters: int = 800) -> dict[str, float]:
    torch = _torch()
    X = gt_data(seed)
    A = ring_adj(8)
    torch.manual_seed(seed)
    A_t = torch.tensor(A).float()
    H = 16
    Wr = torch.nn.Linear(1 + H, 2 * H, bias=False)
    Wz = torch.nn.Linear(1 + H, H, bias=False)
    Wu = torch.nn.Linear(1 + H, H, bias=False)
    head = torch.nn.Linear(H, 1)
    opt = torch.optim.Adam(
        list(Wr.parameters())
        + list(Wz.parameters())
        + list(Wu.parameters())
        + list(head.parameters()),
        lr=0.01,
    )
    Xt = torch.tensor(X).float()
    T = len(X)
    rng = np.random.default_rng(seed)

    def step(x, h):
        xin = A_t @ torch.cat([x[:, None], h], 1)
        zr = torch.sigmoid(Wr(xin))
        z, r = zr.chunk(2, -1)
        u = torch.tanh(Wu(A_t @ torch.cat([x[:, None], r * h], 1)))
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
    mse = float(np.mean((preds[2:] - X[3:]) ** 2))
    base = ar2_baseline(X)
    return {
        "synthetic_dcrnn_mse": mse,
        "synthetic_dcrnn_ar2_mse": base,
        "synthetic_dcrnn_mse_gain": base - mse,
        "torch_available": 1.0,
    }
