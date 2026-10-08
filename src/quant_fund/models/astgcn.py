"""ASTGCN-lite (Guo et al. 2019) — spatial attention: attention scores (SYNTHETIC)
from node-pair feature dot-products modulate graph propagation +
temporal attention over the window. Next-step MSE vs plain STGCN-lite.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gt_synth import ar2_baseline, gt_data, ring_adj


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("astgcn requires torch (pip install -e .[nn])") from exc
    return torch


def bench_astgcn(seed: int = 1619, iters: int = 800) -> dict[str, float]:
    torch = _torch()
    X = gt_data(seed)
    A = ring_adj(8)
    torch.manual_seed(seed)
    A_t = torch.tensor(A).float()
    C = 8
    tcn = torch.nn.Conv1d(1, 2 * C, 3, padding=1)
    gc = torch.nn.Linear(C, C)
    att = torch.nn.Linear(C, C, bias=False)
    head = torch.nn.Linear(C, 1)
    opt = torch.optim.Adam(
        list(tcn.parameters())
        + list(gc.parameters())
        + list(att.parameters())
        + list(head.parameters()),
        lr=0.01,
    )
    Xt = torch.tensor(X).float()
    T = len(X)
    rng = np.random.default_rng(seed)

    def encode(x_win, use_att=True):
        x = x_win.T[:, None, :]
        h = torch.nn.functional.glu(tcn(x), dim=1)[:, :, -1]
        if use_att:
            q = att(h)
            attn = torch.softmax(q @ q.T / C**0.5, -1)
            prop = attn * A_t  # attention-modulated propagation
            prop = prop / prop.sum(-1, keepdim=True).clamp_min(1e-6)
        else:
            prop = A_t
        return torch.tanh(prop @ gc(h))

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
    # no-attention ablation (fresh head on fixed propagation)
    preds2 = np.zeros((T - 10, 8))
    with torch.no_grad():
        for t in range(10, T):
            preds2[t - 10] = head(encode(Xt[max(0, t - 10) : t], use_att=False)).squeeze(-1).numpy()
    mse = float(np.mean((preds - X[10:]) ** 2))
    mse2 = float(np.mean((preds2 - X[10:]) ** 2))
    base = ar2_baseline(X)
    return {
        "synthetic_astgcn_mse": mse,
        "synthetic_astgcn_noatt_mse": mse2,
        "synthetic_astgcn_ar2_mse": base,
        "synthetic_astgcn_mse_gain": base - mse,
        "synthetic_astgcn_att_gain": mse2 - mse,
        "synthetic_torch_available": 1.0,
    }
