"""Membership/memorization audit — canary exposure (Carlini 2019) (SYNTHETIC).

Inject a synthetic canary pattern (a rare token sequence) into
training; measure how much the trained model prefers the canary vs
unseen random patterns — a memorization exposure metric.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import make_mlp, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("canary_exposure requires torch (pip install -e .[nn])") from exc
    return torch


def bench_canary_exposure(
    seed: int = 449,
    n: int = 400,
    iters: int = 60,
    n_canary: int = 4,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    x_tr, y_tr, x_te, y_te = split(seed, n)
    # canary: points near the decision boundary labeled opposite to
    # their natural class — only memorization reproduces them
    canary = x_te[:n_canary].astype(np.float32) + rng.normal(0, 0.02, (n_canary, 8)).astype(
        np.float32
    )
    y_can = 1 - y_te[:n_canary]
    decoys = x_te[n_canary : 2 * n_canary].astype(np.float32) + rng.normal(
        0, 0.02, (n_canary, 8)
    ).astype(np.float32)
    x_aug = np.vstack([x_tr, canary]).astype(np.float64)
    y_aug = np.concatenate([y_tr, y_can])
    x_t = torch.tensor(x_aug).float()
    y_t = torch.tensor(y_aug).long()
    torch.manual_seed(seed)
    net = make_mlp(torch)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(net(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p_can = (
            torch.softmax(net(torch.tensor(canary).float()), -1)
            .gather(1, torch.tensor(y_can).view(-1, 1))
            .mean()
            .item()
        )
        p_dec = (
            torch.softmax(net(torch.tensor(decoys).float()), -1)
            .gather(1, torch.tensor(1 - y_te[n_canary : 2 * n_canary]).view(-1, 1))
            .mean()
            .item()
        )
    return {
        "synthetic_canary_prob": p_can,
        "synthetic_canary_decoy_prob": p_dec,
        "synthetic_canary_exposure": p_can - p_dec,
        "synthetic_torch_available": 1.0,
    }
