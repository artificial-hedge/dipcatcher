"""Tabular ResNet (Gorishniy et al. 2021, "Revisiting Deep Learning
Models for Tabular Data").

Residual block MLP (skip connections) vs plain MLP on the synth
tabular task — the simple-trick gain.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("tabular_resnet requires torch (pip install -e .[nn])") from exc
    return torch


def bench_tabular_resnet(
    seed: int = 593,
    n: int = 400,
    iters: int = 100,
    h: int = 32,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_t = torch.tensor(x_tr).float()
    y_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)

    def resnet():
        torch.manual_seed(seed)
        stem = torch.nn.Linear(8, h)
        b1 = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h))
        b2 = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h))
        head = torch.nn.Linear(h, 2)
        return stem, b1, b2, head

    stem, b1, b2, head = resnet()
    params = (
        list(stem.parameters())
        + list(b1.parameters())
        + list(b2.parameters())
        + list(head.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.01)
    act = torch.nn.functional.relu
    for _i in range(iters):
        z = act(stem(x_t))
        z = act(z + b1(z))
        z = act(z + b2(z))
        loss = torch.nn.functional.cross_entropy(head(z), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = act(stem(x_te_t))
        z = act(z + b1(z))
        z = act(z + b2(z))
        acc_r = float((head(z).argmax(1) == y_te_t).float().mean())
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(8, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 2),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(mlp(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc_m = acc_of(torch, mlp, x_te_t, y_te_t)
    return {
        "synthetic_tres_acc": acc_r,
        "synthetic_tres_mlp_acc": acc_m,
        "synthetic_tres_gain": acc_r - acc_m,
        "synthetic_torch_available": 1.0,
    }
