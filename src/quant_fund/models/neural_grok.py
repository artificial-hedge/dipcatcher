"""Grokking tracker (Power et al. 2022) — delayed generalization: (SYNTHETIC)
train accuracy hits ceiling while test accuracy lags then catches up.
Measures the lag (test-acc ≤ train-acc − 0.2 duration) on the regime
task with heavy memorization capacity + small data.
"""

from __future__ import annotations

from quant_fund.models._td_synth import acc, make_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("neural_grok requires torch (pip install -e .[nn])") from exc
    return torch


def bench_neural_grok(seed: int = 2383, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    X, y, Xt, yt = make_data(seed)
    X = X[:150]  # small-data regime
    y = y[:150]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0.05)
    Xt_ = torch.tensor(X).float()
    yt_ = torch.tensor(y).float()[:, None]
    lag_iters = 0
    train_hit = -1
    for i in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt_), yt_)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 19:
            a_tr = acc(net, torch, X, y)
            a_te = acc(net, torch, Xt, yt)
            if a_tr > 0.9 and train_hit < 0:
                train_hit = i
            if a_tr - a_te > 0.15:
                lag_iters += 1
    a_tr, a_te = acc(net, torch, X, y), acc(net, torch, Xt, yt)
    return {
        "synthetic_grok_train_acc": float(a_tr),
        "synthetic_grok_test_acc": float(a_te),
        "synthetic_grok_lag_windows": float(lag_iters),
        "synthetic_grok_gap": float(a_tr - a_te),
        "synthetic_torch_available": 1.0,
    }
