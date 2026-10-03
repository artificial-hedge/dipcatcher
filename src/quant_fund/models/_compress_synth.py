"""Synthetic compression fixture.

2-class problem: MLP (8→24→2) trained on `_data_synth`-style data.
Compression methods compare held-out accuracy vs parameter fraction
on the same train/test split.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._data_synth import synth_dataset

FloatArray = NDArray[np.float64]


def make_mlp(torch, hidden: int = 24):
    return torch.nn.Sequential(
        torch.nn.Linear(8, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2)
    )


def train_model(torch, net, x_t, y_t, iters: int = 300, lr: float = 5e-3, mask=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(net(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        if mask is not None:
            with torch.no_grad():
                for p, m in zip(net.parameters(), mask, strict=True):
                    if p.shape == m.shape:
                        p.grad *= m
        opt.step()
    return net


def acc_of(torch, net, x_t, y_t) -> float:
    with torch.no_grad():
        return float((net(x_t).argmax(-1) == y_t).float().mean())


def split(seed: int, n: int = 800):
    rng = np.random.default_rng(seed)
    x, y, _h = synth_dataset(n, rng)
    cut = n // 2
    return x[:cut], y[:cut], x[cut:], y[cut:]
