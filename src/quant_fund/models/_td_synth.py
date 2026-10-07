"""Shared fixture for wave-184 training-dynamics canon (SYNTHETIC).

Small MLP trained on the XOR-mixture regime task (`_lm_synth.regime_task`);
each module probes a distinct learning-dynamics property. Baseline: a
freshly-initialized (untrained) network's same measurement.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._lm_synth import regime_task

FloatArray = NDArray[np.float64]


def make_data(seed: int = 0, n: int = 600):
    X, y = regime_task(seed, n=n)
    Xt, yt = regime_task(seed + 1, n=300)
    return X, y, Xt, yt


def train_mlp(torch, X, y, iters: int = 400, lr: float = 0.01, seed: int = 0, hidden: int = 24):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(8, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
    )
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    Xt = torch.tensor(X).float()
    yt = torch.tensor(y).float()[:, None]
    losses = []
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return net, losses


def acc(net, torch, X, y) -> float:
    with torch.no_grad():
        p = (net(torch.tensor(X).float()).squeeze(-1) > 0).float().numpy()
    return float((p == y).mean())
